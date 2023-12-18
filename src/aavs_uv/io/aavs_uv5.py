import h5py
from loguru import logger
import xarray as xp
import pandas as pd
import numpy as np

from astropy.time import Time
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Angle

from aavs_uv.io.mccs_yaml import station_location_from_platform_yaml
from aavs_uv.io.yaml import load_yaml
from aavs_uv.datamodel.visibility import UV, create_antenna_data_array, create_visibility_array
from aavs_uv.utils import get_config_path
import aavs_uv


def uv_to_uv5(uv: aavs_uv.datamodel.UV, filename: str):
    """ Write a aavs UV object to a HDF5 file 
    
    Args:
        uv (UV): aavs_uv.datamodel.UV object
        filename (str): name of output file
    """
    def _str2bytes(darr):
        if 'U' in str(darr.dtype) or 'obj' in str(darr.dtype):
            return darr.astype('bytes')
        else:
            return darr
    
    def _create_dset(group, name, dobj):
        data = _str2bytes(dobj.values)
        dset = group.create_dataset(name, data=data)
        for k, v in dobj.attrs.items():
            
            if isinstance(v, dict):
                for kk, vv in v.items():
                    _k = f'_{k}_kk'
                    try:
                        dset.attrs[_k] = vv
                    except TypeError:
                        dset.attrs[_k] = vv.astype('bytes')
            else:
                try:
                    dset.attrs[k] = v
                except TypeError:
                    dset.attrs[k] = v.astype('bytes')

    with h5py.File(filename, mode='w') as h:
        # Basic metadata 
        h.attrs['CLASS'] = 'AAVS_UV'
        h.attrs['VERSION'] = aavs_uv.__version__
        h.attrs['name'] = uv.name
        
        ####################
        # VISIBILITY GROUP #
        ####################
        g_vis = h.create_group('visibilities')
        g_vis_c = g_vis.create_group('coords')
        # g_vis_a = g_vis.create_group('attrs')
        g_vis_d = g_vis.create_group('dims')

        _create_dset(g_vis, 'data', uv.data)
        dims = ('time', 'frequency', 'baseline', 'polarization')
        g_vis['data'].attrs['dims'] = dims

        for coord in ('mjd', 'lst', 'polarization', 'ant1', 'ant2', 'frequency'):
            _create_dset(g_vis_c, coord, uv.data.coords[coord])

        g_vis_c['mjd'].attrs['description'] = 'Modified Julian Date'
        g_vis_c['lst'].attrs['description'] = 'Local apparent sidereal time'
        g_vis_c['polarization'].attrs['description'] = 'Polarization products'
        g_vis_c['ant1'].attrs['description'] = 'Baseline antenna 1 index'
        g_vis_c['ant2'].attrs['description'] = 'Baseline antenna 2 index'


        for ii, dim in enumerate(dims):
            g_vis_d.attrs[dim] = uv.data.shape[ii]        

        #################
        # ANTENNA GROUP #
        ################# 
        g_ant = h.create_group('antennas')
        g_ant_c = g_ant.create_group('coords')
        g_ant_a = g_ant.create_group('attrs')
        g_ant_d = g_ant.create_group('dims')
        
        for dset_name in ('enu', 'ecef'):
            _create_dset(g_ant, dset_name, uv.antennas[dset_name])
            g_ant[dset_name].attrs['dims'] = ('antenna', 'spatial')
            
        for coord in ('antenna', 'spatial'):
            _create_dset(g_ant_c, coord, uv.antennas.coords[coord])                

        for attr in ('identifier', 'flags', 'array_origin_geocentric', 'array_origin_geodetic'):
            _create_dset(g_ant_a, attr, uv.antennas.attrs[attr])
        
        # Add dimension sizes 
        g_ant_d.attrs['antenna'] = uv.antennas.dims['antenna']
        g_ant_d.attrs['spatial'] = uv.antennas.dims['spatial']

        ################
        # PHASE CENTER #
        ################
        g_pc = h.create_group('phase_center')
        pc = uv.phase_center.icrs
        ra, dec = pc.ra.to('hourangle'), pc.dec.to('deg')
        if pc.isscalar:
            ra, dec = np.expand_dims(ra, 0), np.expand_dims(dec, 0)
        d_pc_ra = g_pc.create_dataset('ra',   data=ra)
        d_pc_dec = g_pc.create_dataset('dec', data=dec)
        d_pc_ra.attrs['unit']  = 'hourangle'
        d_pc_ra.attrs['description']  = 'Right Ascension (J2000)'
        d_pc_dec.attrs['unit'] = 'deg'
        d_pc_dec.attrs['description'] = 'Declination (J2000)'


        ##############
        # PROVENANCE #
        ##############
        g_prov = h.create_group('provenance')
        for k, v in uv.provenance.items():
            if isinstance(v, dict):
                g_prov_a = g_prov.create_group(k)
                for sk, sv in v.items():
                    g_prov_a.attrs[sk] = sv
            else:
                g_prov.attrs[k] = v


def uv5_to_uv(filename: str) -> aavs_uv.datamodel.UV:
    """ Load aavs_uv object from uv5 (HDF5) file 
    
    Args:
        filename (str): path to uv5 file
    
    Returns:
        uv (aavs_uv.datamodel.UV): UV object
    """
    with h5py.File(filename, mode='r') as h:
        
        ################
        # ANTENNA DSET #
        ################
        coords = {
            'antenna': h['antennas']['coords']['antenna'][:],
            'spatial': h['antennas']['coords']['spatial'][:].astype('str')
            }

        data_vars = {
        'enu': xp.DataArray(h['antennas']['enu'], 
               dims=list(h['antennas']['dims'].attrs.keys()),
               attrs=dict(h['antennas']['enu'].attrs.items()),
               coords=coords
               ),
        'ecef': xp.DataArray(h['antennas']['ecef'],
               dims=list(h['antennas']['dims'].attrs.keys()),
               attrs=dict(h['antennas']['ecef'].attrs.items()),
               coords=coords
               )
        }
    
        attrs = {
            'identifier': xp.DataArray(h['antennas']['attrs']['identifier'][:].astype('str'), 
                                    dims=('antenna'), 
                                    attrs=dict(h['antennas']['attrs']['identifier'].attrs.items())
                                    ),
            'flags': xp.DataArray(h['antennas']['attrs']['flags'][:], 
                                dims=('antenna'), 
                                attrs=dict(h['antennas']['attrs']['flags'].attrs.items())
                                ),
            'array_origin_geocentric': xp.DataArray(h['antennas']['attrs']['array_origin_geocentric'][:], 
                                dims=('spatial'), 
                                attrs=dict(h['antennas']['attrs']['array_origin_geocentric'].attrs.items())
                                ),
            'array_origin_geodetic': xp.DataArray(h['antennas']['attrs']['array_origin_geodetic'][:], 
                                dims=('spatial'), 
                                attrs=dict(h['antennas']['attrs']['array_origin_geodetic'].attrs.items())
                                )
        }
    
        antennas = xp.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        
        # Small bytes -> string fixup
        antennas.attrs['array_origin_geodetic'].attrs['unit'] = antennas.attrs['array_origin_geodetic'].attrs['unit'].astype('str')

        ################
        # VISIBILITIES #
        ################

        # Coordinate - time
        mjd = h['visibilities']['coords']['mjd'][:]
        lst = h['visibilities']['coords']['lst'][:] 
        t_coord = pd.MultiIndex.from_arrays((mjd, lst), names=('mjd', 'lst'))
        
        # Coordinate - baseline
        bl_coord = pd.MultiIndex.from_arrays(
            (h['visibilities']['coords']['ant1'][:], h['visibilities']['coords']['ant2'][:]), 
            names=('ant1', 'ant2'))
        
        # Coordinate - polarization
        pol_coord = h['visibilities']['coords']['polarization'][:].astype('str')
        
        # Coordinate - frequency
        f_center  = h['visibilities']['coords']['frequency'][:]
        f_coord   = xp.DataArray(f_center, dims=('frequency',), 
                                attrs=dict(h['visibilities']['coords']['frequency'].attrs.items()))
        
        coords={
            'time': t_coord,
            'polarization': pol_coord,
            'baseline': bl_coord,
            'frequency': f_coord
        }
        

        vis = xp.DataArray(h['visibilities']['data'], 
                        coords=coords, 
                        dims=('time', 'frequency', 'baseline', 'polarization')
                        )
        

        ################
        # PHASE CENTER #
        ################
        phase_center = SkyCoord(h['phase_center']['ra'][:], 
                                h['phase_center']['dec'][:], 
                                unit=(h['phase_center']['ra'].attrs['unit'], 
                                      h['phase_center']['dec'].attrs['unit']))

        ##############
        # PROVENANCE #
        ##############
        provenance = dict(h['provenance'].attrs.items())
        for k, v in h['provenance'].items():
                provenance[k] = dict(v.attrs.items())

        # Add time and earth location
        eloc = EarthLocation.from_geocentric(*h['antennas']['attrs']['array_origin_geocentric'][:], 
                                        unit=h['antennas']['attrs']['array_origin_geocentric'].attrs['unit'])    
        t = Time(mjd, format='mjd', location=eloc)


        uv = UV(name=h.attrs['name'], 
            antennas=antennas, 
            data=vis, 
            timestamps=t, 
            origin=eloc, 
            phase_center=phase_center,
            provenance=provenance)

    return uv