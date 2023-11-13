from aavs_uv.datamodel.visibility import create_antenna_data_array, create_visibility_array, UV
from aavs_uv.aavs_uv import load_observation_metadata

FN_DATA = '../data/2023_09_27_2x500/correlation_burst_100_20230927_35116_0.hdf5'
FN_CONFIG = '../config/aavs3v2/uv_config.yaml'

def test_create_arrays():
    md             = load_observation_metadata(FN_DATA, FN_CONFIG)
    eloc, antennas = create_antenna_data_array(md['antenna_locations_file'])
    print(antennas)

    t, data        = create_visibility_array(FN_DATA, FN_CONFIG, eloc)
    print(data)

def test_UV():
    aavs = UV(FN_DATA, FN_CONFIG)
    print(aavs)

if __name__ == "__main__":
    test_create_arrays()
    test_UV()