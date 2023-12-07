import h5py
import numpy as np
import pandas as pd

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.units import Quantity

from aavs_uv.datamodel.visibility import create_antenna_data_array, create_visibility_array
from aavs_uv.io import hdf5_to_uv, load_observation_metadata

FN_DATA = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
FN_CONFIG = '../example-config/aavs3/uv_config.yaml'

def test_create_arrays():
    md             = load_observation_metadata(FN_DATA, FN_CONFIG)

    xyz = np.array(list(md[f'telescope_ECEF_{q}'] for q in ('X', 'Y', 'Z')))
    eloc = EarthLocation.from_geocentric(*xyz, unit='m')
    antpos = pd.read_csv(md['antenna_locations_file'], delimiter=' ')
    antennas = create_antenna_data_array(antpos, eloc)
    print(antennas)

    h5 = h5py.File(FN_DATA, mode='r') 
    data = h5['correlation_matrix']['data']

    t  = Time(np.arange(md['n_integrations'], dtype='float64') * md['tsamp'] + md['ts_start'], 
              format='unix', location=eloc)
    f_arr = (np.arange(md['n_chans'], dtype='float64') + 1) * md['channel_spacing'] * md['channel_id']
    f = Quantity(f_arr, unit='Hz')
    data        = create_visibility_array(data, f, t, eloc)
    print(data)

def test_create_uv():
    aavs = hdf5_to_uv(FN_DATA, FN_CONFIG)
    print(aavs)

if __name__ == "__main__":
    test_create_arrays()
    test_create_uv()