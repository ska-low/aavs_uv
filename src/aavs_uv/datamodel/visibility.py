import os
import h5py
import numpy as np
import xarray as xp
import pandas as pd
from dataclasses import dataclass

from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Angle
from astropy.time import Time
import pyuvdata.utils as uvutils

from aavs_uv.aavs_uv import load_observation_metadata
from aavs_uv.io.mccs_yaml import station_location_from_platform_yaml


# Define the data class for UV data
@dataclass 
class UV:
    name: str               # Antenna array name, e.g. AAVS3
    antennas: xp.Dataset    # An xarray dataset (generated with create_antenna_data_array)
    data: xp.DataArray      # An xarray DataArray (generated with create_visibility_array)
    timestamps: Time        # Astropy timestamps Time() array
    origin: EarthLocation   # Astropy EarthLocation for array origin
    phase_center: SkyCoord  # Astropy SkyCoord corresponding to phase center
    provenance: dict        # Provenance/history information and other metadata
        

def create_antenna_data_array(antpos: pd.DataFrame, eloc: EarthLocation) -> xp.Dataset:
    """ Create an xarray Dataset for antenna locations 
    
    Args:
        antpos (pd.Dataframe): Pandas dataframe with antenna positions. Should have 
                               columns: id | name | E | N | U | flagged
        eloc (EarthLocation): Astropy EarthLocation corresponding to array center
    
    Returns:
        dant (xp.Dataset): xarray Dataset with antenna locations
    
    Notes:
        <xarray.Dataset>
            Dimensions:  (antenna: N_ant, spatial: 3)
            Coordinates:
            * antenna  (antenna) int64 0 1 2 3 4 5 6 7 ... N_ant
            * spatial  (spatial) <U1 'x' 'y' 'z'
            Data variables:
                enu      (antenna, spatial) float64 East-North-Up coordinates relative to eloc
                ecef     (antenna, spatial) float64 ECEF XYZ coordinates (XYZ - eloc.XYZ0)
            Attributes:
                identifier:               Antenna names / identifiers
                flags:                    Flags if antenna is bad
                array_origin_geocentric:  Array origin (ECEF)
                array_origin_geodetic:    Array origin (lat/lon/height) 
    """
    lat_rad = eloc.lat.to('rad').value
    lon_rad = eloc.lon.to('rad').value
    x0, y0, z0 = [_.to('m').value for _ in eloc.to_geocentric()]

    antpos_enu   = np.column_stack((antpos['E'], antpos['N'], antpos['U']))  
    antpos_ecef  = uvutils.ECEF_from_ENU(antpos_enu, lat_rad, lon_rad, eloc.height)  - (x0, y0, z0)
    
    data_vars = {
        'enu': xp.DataArray(antpos_enu, 
               dims=('antenna', 'spatial'),
               attrs={'units': 'm',
                     'description': 'Antenna locations in local East-North-Up coordinates'}),
        'ecef': xp.DataArray(antpos_ecef,
                dims=('antenna', 'spatial'),
                attrs={'units': 'm',
                      'description': 'Antenna WGS84 locations in Earth-centered, Earth-fixed (ECEF) coordinate system. \
                      Note array center (origin) position (X0, Y0, Z0) has been subtracted.'}),
    }
    
    attrs = {
        'identifier': xp.DataArray(antpos['name'], dims=('antenna'), attrs={'description': 'Antenna name/identifier'}),
        'flags': xp.DataArray(antpos['flagged'], dims=('antenna'), attrs={'description': 'Data quality issue flag'}),
    }
    
    coords = {
        'antenna': np.arange(256),
        'spatial': np.array(('x', 'y', 'z'))
        }

    # Add array origin
    array_origin_m = (eloc['x'].value, eloc['y'].value, eloc['z'].value)
    array_origin_ecef = xp.DataArray(np.array(array_origin_m), 
                                attrs={'unit': 'm',
                                      'description': 'Array center in WGS84 ECEF coordinates'},
                                coords={'spatial': np.array(('x', 'y', 'z'))},
                                dims=('spatial'))
    
    array_origin_geodetic = xp.DataArray(np.array((eloc.lon.value, eloc.lat.value, eloc.height.value)),
                                attrs={'unit': np.array(('deg', 'deg', 'm')),
                                      'description': 'Geodetic array center in Longitude, Latitude, Height'},
                                coords={'spatial': np.array(('longitude', 'latitude', 'height'))},
                                dims=('spatial'))
    
    attrs['array_origin_geocentric'] = array_origin_ecef
    attrs['array_origin_geodetic']   = array_origin_geodetic
    
    dant = xp.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    
    return dant

def create_visibility_array(data: np.ndarray, md: dict, eloc: EarthLocation) -> (Time, xp.DataArray):
    """ Create visibility array out of data array + metadata 
    
    Args:
        data (np.array): Numpy array or duck-type similar data (e.g. h5py.dataset)
        md (dict): Dictionary of metadata, as found in raw HDF5 file.
        eloc (EarthLocation): Astropy EarthLocation for array center
    
    Returns:
        t (Time): Astropy time array corresponding to timestamps
        vis (xp.DataArray): xarray DataArray object, see notes below
    
    Notes:
        <xarray.DataArray (time: N_time, frequency: N_freq, baseline: N_bl, polarization: N_pol)>
            Coordinates:
            * time          (time) object MultiIndex
              * mjd           (time) time in MJD
              * lst           (time) time in LST
            * polarization  (polarization) <U2 'XX' 'XY' 'YX' 'YY'
            * baseline      (baseline) object MultiIndex
              * ant1          (baseline) int64 0 0 0 0 0 0 0 ... N_ant
              * ant2          (baseline) int64 0 1 2 3 4 5 6 ... N_ant
            * frequency     (frequency) float64 channel frequency values, in Hz   

    Speed notes:
        this code generates MJD and LST timestamps attached as coordinates, as well as an
        astropy Time() array (which provides useful conversion between time formats that
        DataArray does not). Conversion to/from datetime64 takes significantly longer than
        generation from an array of MJD values.    
    """
    # Coordinate - time
    t  = Time(np.arange(md['n_integrations'], dtype='float64') * md['tsamp'] + md['ts_start'], 
              format='unix', location=eloc)
    lst = t.sidereal_time('apparent').to('hourangle')
    t_coord = pd.MultiIndex.from_arrays((t.mjd, lst.value), names=('mjd', 'lst'))
    
    # Coordinate - baseline
    ix, iy = np.triu_indices(256)
    bl_coord = pd.MultiIndex.from_arrays((ix, iy), names=('ant1', 'ant2'))
    
    # Coordinate - polarization
    pol_coord = np.array(('XX', 'XY', 'YX', 'YY'))
    
    # Coordinate - frequency
    f_center  = (np.arange(md['n_chans'], dtype='float64') + 1) * md['channel_spacing'] * md['channel_id']
    f_coord = xp.DataArray(f_center, dims=('frequency',), attrs={'unit': 'Hz', 'description': 'Frequency at channel center'})
    # channel_bandwidth (ndarray) - 1D numpy array containing channel bandwidths in Hz
    f_coord.attrs['channel_bandwidth'] = md['channel_width']
    f_coord.attrs['channel_id'] = md['channel_id']
    
    coords={
        'time': t_coord,
        'polarization': pol_coord,
        'baseline': bl_coord,
        'frequency': f_coord
    }
    
    vis = xp.DataArray(data, 
                      coords=coords, 
                      dims=('time', 'frequency', 'baseline', 'polarization')
                     )
    return t, vis


def create_uv(fn_data: str, fn_config: str, from_platform_yaml: bool=False) -> UV:
    """ Create UV from HDF5 data and config file
    
    Args:
        fn_data (str): Path to HDF5 data
        fn_config (str): Path to uv_config.yaml configuration file
        from_platform_yaml (bool=False): If true, uv_config.yaml setting 'antenna_locations'
                                         points to a mccs_platform.yaml file. Otherwise, a 
                                         simple CSV text file is used.
        
    Returns:
        uv (UV): A UV dataclass object with xarray datasets
    
    Notes:
        class UV:
            name: str
            antennas: xp.Dataset
            data: xp.DataArray
            timestamps: Time
            origin: EarthLocation
            phase_center: SkyCoord
            provenance: dict
    """
    md = load_observation_metadata(fn_data, fn_config)

    h5 = h5py.File(fn_data, mode='r') 
    data = h5['correlation_matrix']['data']

    if from_platform_yaml:
        eloc, antpos = station_location_from_platform_yaml(md['antenna_locations_file'])

    else:
        # Telescope location
        # Also instantiate an EarthLocation observer for LST / Zenith calcs
        xyz = np.array(list(md[f'telescope_ECEF_{q}'] for q in ('X', 'Y', 'Z')))
        eloc = EarthLocation.from_geocentric(*xyz, unit='m')

        # Load baselines and antenna locations (ENU)
        antpos = pd.read_csv(md['antenna_locations_file'], delimiter=' ')


    antennas = create_antenna_data_array(antpos, eloc)
    t, data  = create_visibility_array(data, md, eloc)
    provenance = {'data_filename': os.path.abspath(fn_data),
                  'config_filename': os.path.abspath(fn_config),
                  'input_metadata': md}

    # Compute zenith RA/DEC for phase center
    zen_aa = AltAz(alt=Angle(90, unit='degree'), az=Angle(0, unit='degree'), obstime=t[0], location=t.location)
    zen_sc = SkyCoord(zen_aa).icrs

    # Create UV object
    uv = UV(name=md['telescope_name'], 
            antennas=antennas, 
            data=data, 
            timestamps=t, 
            origin=eloc, 
            phase_center=zen_sc,
            provenance=provenance)

    return uv