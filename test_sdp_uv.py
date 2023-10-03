from sdp_uv import hdf5_to_sdp_vis


def test_sdp_vis():
    """ Load data and test visibility generation  """
    yaml_raw = 'config/aavs2_uv_config.yaml'
    fn_raw   = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    v = hdf5_to_sdp_vis(fn_raw, yaml_raw)
    print(v)

if __name__ == "__main__":
    test_sdp_vis()