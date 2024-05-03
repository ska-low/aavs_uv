# Basic imports
import glob
import os
import warnings
from loguru import logger
from pprint import pprint
from datetime import datetime

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

from aavs_uv.io.aavs_hdf5 import load_observation_metadata
from aavs_uv.datamodel import UVX
from aavs_uv import __version__

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


def uvx_to_pyuvdata(uvx: UVX, phase_to_t0: bool=True,
                     start_int: int=0, max_int: int=None) -> pyuvdata.UVData:
    """ Convert AAVS2/3 HDF5 correlator output to UVData object

    Args:
        filename (str):     Name of file to open
        phase_to_t0 (bool): Instead of phasing to Zenith, phase all timestamps to
                            the RA/DEC position of zenith at the first timestamp (t0).
                            This is needed if writing UVFITS files, but not if you
                            are doing snapshot imaging of each timestep. Default True.
        start_int (int):    First integration index to read (allows skipping ahead through file)
        max_int (int):      Maximum number of intergrations to read. Default None (read all)

    Returns:
        uv (pyuvdata.UVData): A UVData object that can be used to create
                              UVFITS/MIRIAD/UVH5/etc files
    """

    # Create empty UVData object
    uv = pyuvdata.UVData()

    md = {
        'telescope_name': uvx.name,
        'instrument': uvx.name,
        'n_integrations': uvx.data.time.shape[0],
        'n_antennas':     uvx.antennas.antenna.shape[0],
        'n_baselines':    uvx.data.baseline.shape[0],
        'n_chans':        uvx.data.frequency.shape[0],
        'n_stokes':       uvx.data.polarization.shape[0],
        'tsamp':          uvx.data.time.attrs['resolution'],
        'channel_width': uvx.data.frequency.attrs['channel_bandwidth'],
        'channel_spacing': uvx.data.frequency.attrs['channel_spacing'],
        'flex_spw': False,
        'Nspws': 1,
        'Nphase': 1,
        'future_array_shapes': False,
        'polarization_type': 'linear_crossed', # [XX, XY, YX, YY]
        'history': f"Generated with {__version__} at {Time(datetime.now()).iso}",
    }

    if max_int is None:
        max_int = md['n_integrations'] - start_int

    if max_int < uvx.data.time.shape[0]:
        md['n_integrations'] = max_int

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
    uv.vis_units      = uvx.data.attrs.get('unit', 'uncalib')

    # Telescope location
    # Also instantiate an EarthLocation observer for LST / Zenith calcs
    xyz = np.array((uvx.origin.x.value, uvx.origin.y.value, uvx.origin.z.value))
    uv.telescope_location = xyz
    telescope_earthloc = uvx.origin

    # Load baselines and antenna locations (ENU)
    antpos_ENU  = uvx.antennas.enu.values
    antpos_ECEF = uvx.antennas.ecef.values

    # Now fill in antenna info fields
    uv.antenna_positions = antpos_ECEF
    uv.antenna_names     = uvx.antennas.identifier.values
    uv.antenna_numbers   = uvx.antennas.antenna.values
    uv.ant_1_array       = np.tile(uvx.data.baseline.ant1, md['n_integrations'])
    uv.ant_2_array       = np.tile(uvx.data.baseline.ant2, md['n_integrations'])

    # Create baseline array - note: overwrites baseline ordering file with pyuvdata standard.
    uv.baseline_array    = uvutils.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array, md['n_antennas'])

    # Frequency axis
    f0 = uvx.data.frequency.values[0]
    uv.freq_array = uvx.data.frequency.values

    # Polarization axis
    _pol_types = {
        'stokes':   np.array([1,   2,  3,  4]),
        'linear':   np.array([-5, -6, -7, -8]),
        'linear_crossed': np.array([-5, -7, -8, -6]),  # Note: we convert this to linear when loading data
        'circular': np.array([-1, -2, -3, -4])
    }
    uv.polarization_array = _pol_types['linear_crossed']

    # Spectral window axis
    uv.spw_array = np.array([0])
    uv.flex_spw_id_array = np.array([0])

    # Time axis
    t = uvx.timestamps
    t0 = t[0]

    # Surprisingly, this line calculating sidereal time can be a bottleneck
    # So we need to compute it before we apply np.repeat() or its much slower
    lst_rad = t.sidereal_time('apparent').to('rad').value

    # repeat each item in array (0, 1, N_int) for Nbl -> (0, 0, ..., 1, 1, ..., N_int, N_int, ...)
    t_jd    = t.jd
    t       = np.repeat(t_jd, uv.Nbls)
    lst_rad = np.repeat(lst_rad,  uv.Nbls)

    uv.time_array = t
    uv.lst_array = lst_rad
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
    zen_aa = AltAz(alt=Angle(90, unit='degree'), az=Angle(0, unit='degree'), obstime=t0, location=uvx.origin)
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

    # Next, load up data
    # Data have shape (time, freq, baseline, pol)
    # Need to flatten to (nbaseline * nint (Nblts), nspw, nchan, npol)
    n_int = md['n_integrations']
    uv.data_array = uvx.data[start_int:(start_int + n_int)].values
    uv.data_array = np.transpose(uv.data_array, (0, 2, 1, 3)) # tfbp -> tbfp
    uv.data_array = uv.data_array.reshape((uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols))

    # HDF5 data are written as XX, XY, YX, YY (AIPS codes -5, -7, -8, -6)
    if md['polarization_type'].lower() == 'linear_crossed':
        # AIPS expects -5 -6 -7 -8, so we need to remap pols
        pol_remap = [0, 3, 1, 2]
        uv.data_array = uv.data_array[..., pol_remap]
        uv.polarization_array = _pol_types['linear']

    # Add optional arrays
    uv.flag_array = np.zeros_like(uv.data_array, dtype='bool')
    uv.nsample_array = np.ones_like(uv.data_array, dtype='float32')

    # Finally, compute zenith UVW coordinates
    # Miriad convention is xyz(ant2) - xyz(ant1)
    # We assume numbers start at 1, not 0
    uv.uvw_array = np.zeros((uv.Nblts, 3), dtype='float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uv.set_uvws_from_antenna_positions(update_vis=True)

    # We have phased to Zenith, but pyuvdata's UVFITS writer needs data to be phased
    # to the zenith of the first timestamp. So, we do this by default
    if phase_to_t0:
        phase_time = Time(uv.time_array[0], format="jd")
        uv.phase_to_time(phase_time)

    return uv


def hdf5_to_pyuvdata(filename: str, yaml_config: str=None, telescope_name: str=None, phase_to_t0: bool=True,
                     start_int: int=0, max_int: int=None, conj: bool=True) -> pyuvdata.UVData:
    """ Convert AAVS2/3 HDF5 correlator output to UVData object

    Args:
        filename (str):     Name of file to open
        yaml_config (str):  YAML configuration file with basic telescope info.
                            See README for more information
        phase_to_t0 (bool): Instead of phasing to Zenith, phase all timestamps to
                            the RA/DEC position of zenith at the first timestamp (t0).
                            This is needed if writing UVFITS files, but not if you
                            are doing snapshot imaging of each timestep. Default True.
        start_int (int):    First integration index to read (allows skipping ahead through file)
        max_int (int):      Maximum number of intergrations to read. Default None (read all)
        conj (bool):        Conjugate visibility data (default True).

    Returns:
        uv (pyuvdata.UVData): A UVData object that can be used to create
                              UVFITS/MIRIAD/UVH5/etc files
    """

    # Load metadata
    md = load_observation_metadata(filename, yaml_config, load_config=telescope_name)

    # Create empty UVData object
    uv = pyuvdata.UVData()

    if max_int is None:
        max_int = md['n_integrations'] - start_int

    if max_int < md['n_integrations']:
        md['n_integrations'] = max_int

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
    uv.channel_width   = md['channel_spacing']  # NOTE: channel_spacing is used for frequency delta
                                                # 'channel_width' exists in metadata, but not used here
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
    df_ant = pd.read_csv(md['antenna_locations_file'], delimiter=' ')
    df_bl  = pd.read_csv(md['baseline_order_file'], delimiter=' ')

    # Convert ENU locations to 'local' ECEF
    # Following https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/mwa_corr_fits.py#L1456
    antpos_ENU  = np.column_stack((df_ant['E'], df_ant['N'], df_ant['U']))
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
    _t = np.arange(md['n_integrations'], dtype='float64') * md['tsamp'] + md['ts_start']
    _t += md['tsamp'] / 2 # Time array is based on center of integration, so add tdelt / 2
    _t += start_int *  md['tsamp']  # Add time offset if not reading from t0

    t = Time(_t, format='unix', location=telescope_earthloc)
    t0 = t[0]

    # Surprisingly, this line calculating sidereal time can be a bottleneck
    # So we need to compute it before we apply np.repeat() or its much slower
    lst_rad = t.sidereal_time('apparent').to('rad').value

    # repeat each item in array (0, 1, N_int) for Nbl -> (0, 0, ..., 1, 1, ..., N_int, N_int, ...)
    t_jd    = Time(_t, format='unix').jd
    t       = np.repeat(t_jd, uv.Nbls)
    lst_rad = np.repeat(lst_rad,  uv.Nbls)

    uv.time_array = t
    uv.lst_array = lst_rad
    uv.integration_time = np.zeros_like(uv.time_array) + md['tsamp']

    # Reference date RDATE and corresponding Greenwich sidereal time at midnight GST0
    # See https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/f703a985869b974892fc4732910c83790f9c72b4/pyuvdata/uvdata/uvfits.py#L1305C13-L1305C85
    rdate_obj = Time(uv.time_array[0], format="jd", scale="utc", location=t0.location)
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

    # Next, load up data
    with h5py.File(filename, mode='r') as datafile:
        # Data have shape (nint, nbaseline, nspw, npol)
        # Need to flatten to (nbaseline * nint (Nblts), nspw, nchan, npol)
        n_int = md['n_integrations']
        data = datafile['correlation_matrix']['data'][start_int:(start_int + n_int)]
        #uv.data_array = np.transpose(data, (0, 2, 1, 3))
        uv.data_array = data.reshape((uv.Nblts, uv.Nspws, uv.Nfreqs, uv.Npols))

        # HDF5 data are written as XX, XY, YX, YY (AIPS codes -5, -7, -8, -6)
        if md['polarization_type'].lower() == 'linear_crossed':
            # AIPS expects -5 -6 -7 -8, so we need to remap pols
            pol_remap = [0, 3, 1, 2]
            uv.data_array = uv.data_array[..., pol_remap]
            uv.polarization_array = _pol_types['linear']

        if conj:
            logger.info('Conjugating data')
            uv.data_array = np.conj(uv.data_array)

        # Add optional arrays
        uv.flag_array = np.zeros_like(uv.data_array, dtype='bool')
        uv.nsample_array = np.ones_like(uv.data_array, dtype='float32')

    # Finally, compute zenith UVW coordinates
    # Miriad convention is xyz(ant2) - xyz(ant1)
    # We assume numbers start at 1, not 0
    uv.uvw_array = np.zeros((uv.Nblts, 3), dtype='float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uv.set_uvws_from_antenna_positions(update_vis=True)

        # We have phased to Zenith, but pyuvdata's UVFITS writer needs data to be phased
        # to the zenith of the first timestamp. So, we do this by default
        if phase_to_t0:
            phase_time = Time(uv.time_array[0], format="jd")
            uv.phase_to_time(phase_time)

    return uv