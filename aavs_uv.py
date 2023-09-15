import pyuvdata
import numpy as np
from astropy.io import fits as pf
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, Angle, SkyCoord
from pprint import pprint
import yaml
import pylab as plt
import h5py
import glob
import os
import pandas as pd
import warnings

def load_yaml(filename: str) -> dict:
    """ Read YAML file into a Python dict """ 
    d = yaml.load(open(filename, 'r'), yaml.Loader)
    return d

def get_hdf5_metadata(filename: str) -> dict:
    """ Extract metadata from HDF5 and perform checks """
    with h5py.File(filename, mode='r') as datafile:
        expected_keys = ['n_antennas', 'ts_end', 'n_pols', 'n_beams', 'tile_id', 'n_chans', 'n_samples', 'type',
                         'data_type', 'data_mode', 'ts_start', 'n_baselines', 'n_stokes', 'channel_id', 'timestamp',
                         'date_time', 'n_blocks']
    
        # Check that keys are present
        if set(expected_keys) - set(datafile.get('root').attrs.keys()) != set():
            raise Exception("Missing metadata in file")
    
        # All good, get metadata
        metadata = {k: v for (k, v) in datafile.get('root').attrs.items()}
        metadata['nof_integrations'] = metadata['n_blocks'] * metadata['n_samples']
    return metadata

def hdf5_to_pyuvdata(filename: str, yaml_config: str) -> pyuvdata.UVData:
    """ Convert AAVS2/3 HDF5 correlator output to UVData object

    Args:
        filename (str): Name of file to open
        yaml_config (str): YAML configuration file with basic telescope info.
                           See README for more information
    Returns:
        uv (pyuvdata.UVData): A UVData object that can be used to create 
                              UVFITS/MIRIAD/UVH5/etc files
    """
    
    # Create empty UVData object
    uv = pyuvdata.UVData()
    
    # Load metadata from config and HDF5 file
    md      = get_hdf5_metadata(filename)
    md_yaml = load_yaml(yaml_config)
    md.update(md_yaml)
    
    #with h5py.File(filename, mode='r') as h:
    uv.Nants_data      = md['n_antennas']
    uv.Nants_telescope = md['n_antennas']
    uv.Nbls            = md['n_baselines']
    uv.Nblts           = md['n_baselines'] * md['n_samples']
    uv.Nfreqs          = md['n_chans']
    uv.Npols           = md['n_stokes']
    uv.Nspws           = md['Nspws']
    uv.Nphase          = md['Nphase']
    uv.Ntimes          = md['n_samples']
    uv.channel_width   = md['channel_width']
    uv.flex_spw        = md['flex_spw']
    uv.future_array_shapes = md['future_array_shapes']
    
    uv.history         = md['history']
    uv.instrument      = md['instrument']

    uv.telescope_name = md['telescope_name']
    uv.vis_units      = md['vis_units']
    
    df_ant = pd.read_csv(md['antenna_locations_file'], delimiter=' ', skiprows=4, names=['name', 'X', 'Y', 'Z'])
    df_bl  = pd.read_csv(md['baseline_order_file'], delimiter=' ')

    uv.antenna_names     = df_ant['name'].values
    uv.antenna_numbers   = np.array(list(df_ant.index), dtype='int32') + 1
    uv.antenna_positions = np.column_stack((df_ant['X'], df_ant['Y'], df_ant['Z']))

    uv.ant_1_array = df_bl['ant1'].values
    uv.ant_2_array = df_bl['ant2'].values
    uv.baseline_array = df_bl['baseline'].values

    # Frequency axis
    f0 = md['channel_spacing'] * md['channel_id']
    uv.freq_array = np.array([[f0]])

    # Polarization axis
    _pol_types = {
        'stokes':   [1,   2,  3,  4],
        'linear':   [-5, -6, -7, -8],
        'circular': [-1, -2, -3, -4]
    }
    uv.polarization_array = _pol_types[md['polarization_type'].lower()]

    # Spectral window axis
    uv.spw_array = np.array([0])

    # Telescope location
    # Also instantiate an EarthLocation observer for LST / Zenith calcs
    xyz = np.array(list(md[f'telescope_ECEF_{q}'] for q in ('X', 'Y', 'Z')))
    uv.telescope_location = xyz
    telescope_earthloc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')

    # Time axis
    # Compute JD from unix time and LST - we can do this as we set an EarthLocation on t0 Time
    t0   = Time(md['ts_start'], format='unix', location=telescope_earthloc)
    lst0 = t0.sidereal_time('apparent').to('rad').value
    uv.time_array = np.zeros(uv.Nblts, dtype='float64') + t0.jd
    uv.lst_array = np.zeros_like(uv.time_array) + lst0
    uv.integration_time = np.zeros_like(uv.time_array) + md['tsamp']

    # Compute zenith phase center
    zen_aa = AltAz(alt=Angle(90, unit='degree'), az=Angle(0, unit='degree'), obstime=t0, location=t0.location)
    zen_sc = SkyCoord(zen_aa)

    phs_id = uv._add_phase_center(
        cat_name=f"zenith_at_jd{t0.jd}",
        cat_type='driftscan',
        cat_lon=zen_aa.alt.rad,
        cat_lat=zen_aa.az.rad,
        cat_frame='altaz',
        cat_epoch='J2000',
        info_source='user',
        force_update=False,
        cat_id=None,
    )

    uv.phase_center_id_array  = np.zeros(uv.Nblts, dtype='int32') + phs_id
    uv.phase_center_app_ra    = np.zeros(uv.Nblts, dtype='float64') + zen_sc.icrs.ra.rad
    uv.phase_center_app_dec   = np.zeros(uv.Nblts, dtype='float64') + zen_sc.icrs.dec.rad
    uv.phase_center_frame_pa  = np.zeros(uv.Nblts, dtype='float64')
    
    # Compute zenith UVW coordinates
    # Miriad convention is xyz(ant2) - xyz(ant1)
    # We assume numbers start at 1, not 0
    uv.uvw_array = np.zeros((uv.Nblts, 3), dtype='float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uv.set_uvws_from_antenna_positions(update_vis=False)

    return uv