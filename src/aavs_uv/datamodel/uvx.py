import os
import h5py
import numpy as np
import xarray as xp
import pandas as pd
from dataclasses import dataclass

from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Angle
from astropy.time import Time
from astropy.units import Quantity
import pyuvdata.utils as uvutils


# Define the data class for UV data
@dataclass 
class UVX:
    name: str               # Antenna array name, e.g. AAVS3
    context: dict           # Contextual information (observation intent, notes, observer name)
    antennas: xp.Dataset    # An xarray dataset (generated with create_antenna_data_array)
    data: xp.DataArray      # An xarray DataArray (generated with create_visibility_array)
    timestamps: Time        # Astropy timestamps Time() array
    origin: EarthLocation   # Astropy EarthLocation for array origin
    phase_center: SkyCoord  # Astropy SkyCoord corresponding to phase center
    provenance: dict        # Provenance/history information and other metadata


def create_empty_context_dict():
    context = {
        'intent': '',
        'date': '',
        'notes': '',
        'observer': '',
        'execution_block': '',
    }
    return context


def create_empty_provenance_dict():
    provenance = {
        'aavs_uv_config': {},
        'input_files': {},
        'input_metadata': {}
    }
    return provenance


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


def create_visibility_array(data: np.ndarray, f: Quantity, t: Time, eloc: EarthLocation, conj: bool=True) -> xp.DataArray:
    """ Create visibility array out of data array + metadata 

    Takes a data array, frequency and time axes, and an EarthLocation. 
    Currently assumes XX/XY/YX/YY polarization and upper triangle baseline coordinates. 
    
    Args:
        data (np.array): Numpy array or duck-type similar data (e.g. h5py.dataset)
        md (dict): Dictionary of metadata, as found in raw HDF5 file.
        eloc (EarthLocation): Astropy EarthLocation for array center
        conj (bool): Conjugate visibility data (default True). 

    
    Returns:
        t (Time): Astropy time array corresponding to timestamps
        f (Quantity): Astropy quantity array of frequency for channel centers
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
    t.location = eloc
    lst = t.sidereal_time('apparent').to('hourangle')
    t_coord = pd.MultiIndex.from_arrays((t.mjd, lst.value, t.unix), names=('mjd', 'lst', 'unix'))
    
    # Coordinate - baseline
    ix, iy = np.triu_indices(256)
    bl_coord = pd.MultiIndex.from_arrays((ix, iy), names=('ant1', 'ant2'))
    
    # Coordinate - polarization
    pol_coord = np.array(('XX', 'XY', 'YX', 'YY'))
    
    # Coordinate - frequency
    f_center  = f.to('Hz').value
    f_coord = xp.DataArray(f_center, dims=('frequency',), attrs={'unit': 'Hz', 'description': 'Frequency at channel center'})
    
    coords={
        'time': t_coord,
        'polarization': pol_coord,
        'baseline': bl_coord,
        'frequency': f_coord
    }
    
    if conj:
        data = np.conj(data)

    vis = xp.DataArray(data, 
                      coords=coords, 
                      dims=('time', 'frequency', 'baseline', 'polarization')
                     )
    return vis

