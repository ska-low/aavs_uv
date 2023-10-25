from aavs_uv.sdp_uv import hdf5_to_sdp_vis, uvdata_to_sdp_vis
from aavs_uv.aavs_uv import hdf5_to_pyuvdata

def test_sdp_vis():
    """ Load data and test visibility generation  """
    yaml_raw = '../config/aavs2/uv_config.yaml'
    fn_raw   = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    v = hdf5_to_sdp_vis(fn_raw, yaml_raw)
    print(v)

def test_uvdata_to_sdp_vis():
    fn_raw = '/Users/daniel.price/Data/aavs_uv/example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    yaml_raw = '/Users/daniel.price/Data/aavs_uv/config/aavs2/uv_config.yaml'

    uv = hdf5_to_pyuvdata(fn_raw, yaml_raw)
    v = uvdata_to_sdp_vis(uv)

if __name__ == "__main__":
    test_sdp_vis()
    test_uvdata_to_sdp_vis()