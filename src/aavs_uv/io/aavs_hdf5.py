import os
import h5py
import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Angle

from aavs_uv.io.mccs_yaml import station_location_from_platform_yaml
from aavs_uv.io.yaml import load_yaml
from aavs_uv.datamodel.visibility import UV, create_antenna_data_array, create_visibility_array


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


def get_hdf5_metadata(filename: str) -> dict:
    """ Extract metadata from HDF5 and perform checks """
    with h5py.File(filename, mode='r') as datafile:
        expected_keys = ['n_antennas', 'ts_end', 'n_pols', 'n_beams', 'tile_id', 'n_chans', 'n_samples', 'type',
                         'data_type', 'data_mode', 'ts_start', 'n_baselines', 'n_stokes', 'channel_id', 'timestamp',
                         'date_time', 'n_blocks']
    
        # Check that keys are present
        if set(expected_keys) - set(datafile.get('root').attrs.keys()) != set(): # pragma: no cover
            raise Exception("Missing metadata in file")
    
        # All good, get metadata
        metadata = {k: v for (k, v) in datafile.get('root').attrs.items()}
        metadata['n_integrations'] = metadata['n_blocks'] * metadata['n_samples']
        metadata['data_shape'] = datafile['correlation_matrix']['data'].shape
    return metadata


def hdf5_to_uv(fn_data: str, fn_config: str, from_platform_yaml: bool=False) -> UV:
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
        The dataclass is defined as:
        class UV:
            name: str
            antennas: xp.Dataset - dimensions ('antenna', 'spatial')
            data: xp.DataArray   - dimensions ('time', 'frequency', 'baseline', 'polarization')
            timestamps: Time     
            origin: EarthLocation
            phase_center: SkyCoord
            provenance: dict

    Metadata notes:
        The following dict items are required to generate coordinate arrays:
            n_integrations
            tsamp
            ts_start
            n_chans
            channel_spacing
            channel_id
            channel_width
    """
    md = load_observation_metadata(fn_data, fn_config)
    h5   = h5py.File(fn_data, mode='r') 
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

    t     = Time(np.arange(md['n_integrations'], dtype='float64') * md['tsamp'] + md['ts_start'], 
                 format='unix', location=eloc)
    f_arr = (np.arange(md['n_chans'], dtype='float64') + 1) * md['channel_spacing'] * md['channel_id']
    f     = Quantity(f_arr, unit='Hz')

    antennas = create_antenna_data_array(antpos, eloc)
    data     = create_visibility_array(data, f, t, eloc)

    # channel_bandwidth channel bandwidth in Hz
    data.frequency.attrs['channel_bandwidth'] = md['channel_width']
    data.frequency.attrs['channel_id']        = md['channel_id']

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