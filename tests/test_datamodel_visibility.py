from aavs_uv.datamodel.visibility import create_antenna_data_array, create_visibility_array, create_uv
from aavs_uv.aavs_uv import load_observation_metadata
import numpy as np
from astropy.coordinates import EarthLocation
import pandas as pd
import h5py

FN_DATA = '../data/2023_09_27_2x500/correlation_burst_100_20230927_35116_0.hdf5'
FN_CONFIG = '../config/aavs3/uv_config.yaml'

def test_create_arrays():
    md             = load_observation_metadata(FN_DATA, FN_CONFIG)

    xyz = np.array(list(md[f'telescope_ECEF_{q}'] for q in ('X', 'Y', 'Z')))
    eloc = EarthLocation.from_geocentric(*xyz, unit='m')
    antpos = pd.read_csv(md['antenna_locations_file'], delimiter=' ')
    antennas = create_antenna_data_array(antpos, eloc)
    print(antennas)

    h5 = h5py.File(FN_DATA, mode='r') 
    data = h5['correlation_matrix']['data']

    t, data        = create_visibility_array(data, md, eloc)
    print(data)

def test_create_uv():
    aavs = create_uv(FN_DATA, FN_CONFIG)
    print(aavs)

if __name__ == "__main__":
    test_create_arrays()
    test_create_uv()