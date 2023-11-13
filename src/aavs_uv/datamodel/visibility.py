import os

import numpy as np
import xarray as xp
import pandas as pd

from astropy.coordinates import EarthLocation
from astropy.time import Time
import pyuvdata.utils as uvutils

from aavs_uv.aavs_uv import load_observation_metadata
from aavs_uv.io.mccs_yaml import station_location_from_platform_yaml
import h5py

        
def create_antenna_data_array(platform_yaml_file: str) -> (EarthLocation, xp.Dataset):
    eloc, antpos = station_location_from_platform_yaml(platform_yaml_file)
    antpos_enu   = np.column_stack((antpos['E'], antpos['N'], antpos['U']))    
    antpos_names = antpos['name']
    antpos_flags = antpos['flagged']

    lat_rad = eloc.lat.to('rad').value
    lon_rad = eloc.lon.to('rad').value
    x0, y0, z0 = [_.to('m').value for _ in eloc.to_geocentric()]
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
        'flags': xp.DataArray(antpos_flags, dims=('antenna'), attrs={'description': 'Data quality issue flag'}),
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
    
    return eloc, dant

def create_visibility_array(fn_data: str, fn_config: str, eloc: EarthLocation) -> (Time, xp.DataArray):
    md = load_observation_metadata(fn_data, fn_config)
    
    h5 = h5py.File(fn_data, mode='r') 
    d = h5['correlation_matrix']['data']
        
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
    
    dx = xp.DataArray(d, 
                      coords=coords, 
                      dims=('time', 'frequency', 'baseline', 'polarization')
                     )
    return t, dx

       
class UV(dict):
    def __init__(self, fn_data, fn_config):
        md = load_observation_metadata(fn_data, fn_config)
        
        eloc, antennas = create_antenna_data_array(md['antenna_locations_file'])
        t, data = create_visibility_array(fn_data, fn_config, eloc)
        
        self.t            = t        

        self['name']     = md['telescope_name']
        self['antennas'] = antennas
        self['data']     = data
        self['origin']   = eloc
        self['provenance'] = {'data_filename': os.path.abspath(fn_data),
                           'config_filename': os.path.abspath(fn_config),
                           'input_metadata': md
                          }

        self.antennas     = self['antennas']
        self.data         = self['data']
        self.provenance   = self['provenance']
        self.name         = self['name']
        self.origin       = self['origin']

