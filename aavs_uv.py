# Basic imports
import glob
import os
import warnings
import yaml
from pprint import pprint

# Basic science stuff
import h5py
import pylab as plt
import pandas as pd
import numpy as np

# Astropy + pyuvdata
from astropy.io import fits as pf
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, AltAz, Angle, SkyCoord, get_sun
import pyuvdata
from pyuvdata import UVData
import pyuvdata.utils as uvutils

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
    

def phase_to_sun(uv: UVData, t0: Time) -> UVData:
    """ Phase UVData to sun, based on timestamp 

    Computes the sun's RA/DEC in GCRS for the given time, then applies phasing.
    This will then recompute UVW and apply phase corrections to data.

    Note: 
        Phase center is set to 'sidereal', i.e. fixed RA and DEC, not 'ephem', so that
        we can apply calibration solutions taken at time t0 (where the Sun was when calibration
        was run, not where it is now!)

    Args:
        uv (UVData): UVData object to apply phasing to (needs to have a phase center defined)
        t0 (Time): Astropy Time() to use to compute Sun's RA/DEC

    Returns:
        uv (UVData): Same UVData as input, but with new phase center applied
    """
    sun = get_sun(t0)
    
    # sun will be returned in GCRS (Geocentric)
    # Need to use GCRS, not ICRS! 
    uv.phase(ra=sun.ra.rad, 
             dec=sun.dec.rad, 
             cat_type='sidereal',
             cat_name=f'sun_{t0.isot}'
            )
    return uv


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
    
    # Telescope location
    # Also instantiate an EarthLocation observer for LST / Zenith calcs
    xyz = np.array(list(md[f'telescope_ECEF_{q}'] for q in ('X', 'Y', 'Z')))
    uv.telescope_location = xyz
    telescope_earthloc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')

    # Load baselines and antenna locations (ENU)
    df_ant = pd.read_csv(md['antenna_locations_file'], delimiter=' ', skiprows=4, names=['name', 'X', 'Y', 'Z'])
    df_bl  = pd.read_csv(md['baseline_order_file'], delimiter=' ')

    # Convert ENU locations to 'local' ECEF
    # Following https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/mwa_corr_fits.py#L1456
    antpos_ENU  = np.column_stack((df_ant['X'], df_ant['Y'], df_ant['Z']))
    antpos_ECEF = uvutils.ECEF_from_ENU(antpos_ENU, *uv.telescope_location_lat_lon_alt) - uv.telescope_location

    # Now fill in antenna info fields
    uv.antenna_positions = antpos_ECEF
    uv.antenna_names     = df_ant['name'].values.astype('str')
    uv.antenna_numbers   = np.array(list(df_ant.index), dtype='int32')
    uv.ant_1_array = df_bl['ant1'].values
    uv.ant_2_array = df_bl['ant2'].values
    uv.baseline_array = df_bl['baseline'].values

    # Frequency axis
    f0 = md['channel_spacing'] * md['channel_id']
    uv.freq_array = np.array([[f0]])

    # Polarization axis
    _pol_types = {
        'stokes':   np.array([1,   2,  3,  4]),
        'linear':   np.array([-5, -6, -7, -8]),
        'linear_crossed': np.array([-5, -6, -8, -7]),
        'circular': np.array([-1, -2, -3, -4])
    }
    uv.polarization_array = _pol_types[md['polarization_type'].lower()]

    # Spectral window axis
    uv.spw_array = np.array([0])
    uv.flex_spw_id_array = np.array([0])

    # Time axis
    # Compute JD from unix time and LST - we can do this as we set an EarthLocation on t0 Time
    t0    = Time(md['ts_start'], format='unix', location=telescope_earthloc)

    # Time array is based on center of integration, so we add tdelt / 2 to t0
    tdelt = TimeDelta(md['tsamp'], format='sec')
    t0   += tdelt / 2
    lst0  = t0.sidereal_time('apparent').to('rad').value
    uv.time_array = np.zeros(uv.Nblts, dtype='float64') + t0.jd
    uv.lst_array = np.zeros_like(uv.time_array) + lst0
    uv.integration_time = np.zeros_like(uv.time_array) + md['tsamp']

    # Reference date RDATE and corresponding Greenwich sidereal time at midnight GST0
    # See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/uvfits.py#L1305C13-L1305C85 
    # See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/uvfits.py#L1318C20-L1318C21
    # See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/uvfits.py#L1323C40-L1323C45
    # See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/uvfits.py#L1338C13-L1338C47
    rdate_obj = Time(np.floor(uv.time_array[0]), format="jd", scale="utc")
    uv.rdate = rdate_obj.strftime("%Y-%m-%d")
    uv.gst0  = float(rdate_obj.sidereal_time("apparent", "tio").deg)
    uv.dut1  = float(rdate_obj.delta_ut1_utc)
    uv.earth_omega = 360.9856438593
    uv.timesys = "UTC"

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
        cat_id=0,
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

    # Finally, load up data
    with h5py.File(filename, mode='r') as datafile:
        # Data have shape (nchan, nspw, nbaseline, npol)
        # Need to transpose to (nbaseline, nspw, nchan, npol)
        data = datafile['correlation_matrix']['data'][:]
        uv.data_array = np.transpose(data, (2, 0, 1, 3))

        # Add optional arrays
        uv.flag_array = np.zeros_like(uv.data_array, dtype='bool')
        uv.nsample_array = np.ones_like(uv.data_array, dtype='float32')

    return uv