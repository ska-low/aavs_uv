# Basic imports
import glob
import os
import warnings

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

from aavs_uv.io.yaml import load_yaml
from aavs_uv.io.aavs_hdf5 import get_hdf5_metadata

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


def load_observation_metadata(filename: str, yaml_config: str):
    """ Load observation metadata """
    # Load metadata from config and HDF5 file
    md      = get_hdf5_metadata(filename)
    md_yaml = load_yaml(yaml_config)
    md.update(md_yaml)

    # Update path to antenna location files to use absolute path
    config_abspath = os.path.dirname(os.path.abspath(yaml_config))
    md['antenna_locations_file'] = os.path.join(config_abspath, md['antenna_locations_file'])
    md['baseline_order_file']  = os.path.join(config_abspath, md['baseline_order_file'])

    return md

def hdf5_to_pyuvdata(filename: str, yaml_config: str, phase_to_t0: bool=True) -> pyuvdata.UVData:
    """ Convert AAVS2/3 HDF5 correlator output to UVData object

    Args:
        filename (str): Name of file to open
        yaml_config (str): YAML configuration file with basic telescope info.
                           See README for more information
        phase_to_t0 (bool): Instead of phasing to Zenith, phase all timestamps to 
                            the RA/DEC position of zenith at the first timestamp (t0).
                            This is needed if writing UVFITS files, but not if you
                            are doing snapshot imaging of each timestep. Default True.
    Returns:
        uv (pyuvdata.UVData): A UVData object that can be used to create 
                              UVFITS/MIRIAD/UVH5/etc files
    """

    # Load metadata
    md = load_observation_metadata(filename, yaml_config)
    
    # Create empty UVData object
    uv = pyuvdata.UVData()
    
    #with h5py.File(filename, mode='r') as h:
    uv.Nants_data      = md['n_antennas']
    uv.Nants_telescope = md['n_antennas']
    uv.Nbls            = md['n_baselines']
    uv.Nblts           = md['n_baselines'] * md['n_integrations']
    uv.Nfreqs          = md['n_chans']
    uv.Npols           = md['n_stokes']
    uv.Nspws           = md['Nspws']
    uv.Nphase          = md['Nphase']
    uv.Ntimes          = md['n_integrations']
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

    df_ant = pd.read_csv(fn_antloc, delimiter=' ', skiprows=4, names=['name', 'X', 'Y', 'Z'])
    df_bl  = pd.read_csv(fn_bls, delimiter=' ')

    # Convert ENU locations to 'local' ECEF
    # Following https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/mwa_corr_fits.py#L1456
    antpos_ENU  = np.column_stack((df_ant['X'], df_ant['Y'], df_ant['Z']))
    antpos_ECEF = uvutils.ECEF_from_ENU(antpos_ENU, *uv.telescope_location_lat_lon_alt) - uv.telescope_location

    # Now fill in antenna info fields
    uv.antenna_positions = antpos_ECEF
    uv.antenna_names     = df_ant['name'].values.astype('str')
    uv.antenna_numbers   = np.array(list(df_ant.index), dtype='int32')
    uv.ant_1_array       = np.tile(df_bl['ant1'].values, md['n_integrations'])
    uv.ant_2_array       = np.tile(df_bl['ant2'].values, md['n_integrations'])
    # Create baseline array - note: overwrites baseline ordering file with pyuvdata standard.
    uv.baseline_array    = uvutils.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array, md['n_antennas'])
    #uv.baseline_array    = np.repeat(df_bl['baseline'].values, md['n_integrations'])

    # Frequency axis
    f0 = md['channel_spacing'] * md['channel_id']
    uv.freq_array = np.array([[f0]])

    # Polarization axis
    _pol_types = {
        'stokes':   np.array([1,   2,  3,  4]),
        'linear':   np.array([-5, -6, -7, -8]),
        'linear_crossed': np.array([-5, -7, -8, -6]),  # Note: we convert this to linear when loading data
        'circular': np.array([-1, -2, -3, -4])
    }
    uv.polarization_array = _pol_types[md['polarization_type'].lower()]

    # Spectral window axis
    uv.spw_array = np.array([0])
    uv.flex_spw_id_array = np.array([0])

    # Time axis
    # Compute JD from unix time and LST - we can do this as we set an EarthLocation on t0 Time
    t    = Time(np.zeros(uv.Nblts, dtype='float64') + md['ts_start'], 
                 format='unix', location=telescope_earthloc)

    # Time array is based on center of integration, so we add tdelt / 2 to t0
    tdelt  = TimeDelta(md['tsamp'], format='sec')
    t    += tdelt / 2
    
    # And for each time step we add an integration time step 
    # repeat each item in array (0, 1, N_int) for Nbl -> (0, 0, ..., 1, 1, ..., N_int, N_int, ...)
    tsteps = TimeDelta(np.repeat(np.arange(md['n_integrations']), uv.Nbls) * md['tsamp'], format='sec')
    t    += tsteps
    
    lst  = t.sidereal_time('apparent').to('rad').value
    uv.time_array = t.jd
    uv.lst_array = lst
    uv.integration_time = np.zeros_like(uv.time_array) + md['tsamp']

    t0 = t[0]

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
        # Data have shape (nint, nbaseline, nspw, npol)
        # Need to flatten to (nbaseline * nint (Nblts), nspw, nchan, npol)
        data = datafile['correlation_matrix']['data'][:]
        #uv.data_array = np.transpose(data, (0, 2, 1, 3))
        uv.data_array = data.reshape((uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols))

        # HDF5 data are written as XX, XY, YX, YY (AIPS codes -5, -7, -8, -6)
        if md['polarization_type'].lower() == 'linear_crossed':
            # A little irritating, but we need to rearrange to get into AIPS standard 
            # xx = np.copy(uv.data_array[..., 0])  # (Already in right spot)
            xy = np.copy(uv.data_array[..., 1])
            yx = np.copy(uv.data_array[..., 2])
            yy = np.copy(uv.data_array[..., 3])
            # uv.data_array[..., 0] = xx           # (Already in right spot)
            uv.data_array[..., 1] = yy
            uv.data_array[..., 2] = xy
            uv.data_array[..., 2] = yx
            uv.polarization_array = _pol_types['linear']

        # Add optional arrays
        uv.flag_array = np.zeros_like(uv.data_array, dtype='bool')
        uv.nsample_array = np.ones_like(uv.data_array, dtype='float32')

        # We have phased to Zenith, but pyuvdata's UVFITS writer needs data to be phased
        # to the zenith of the first timestamp. So, we do this by default
        if phase_to_t0:
            phase_time = Time(uv.time_array[0], format="jd")
            uv.phase_to_time(phase_time)

    return uv