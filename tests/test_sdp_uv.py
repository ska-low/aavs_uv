from aavs_uv.io import hdf5_to_sdp_vis, uvdata_to_sdp_vis, hdf5_to_pyuvdata
import numpy as np

def test_sdp_vis():
    """ Load data and test visibility generation  """
    yaml_raw = '../example-config/aavs2/uv_config.yaml'
    fn_raw   = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    v = hdf5_to_sdp_vis(fn_raw, yaml_raw)
    print(v)

def test_sdp_vis_conj():
    """ Load data and test visibility generation  """
    yaml_raw = '../example-config/aavs2/uv_config.yaml'
    fn_raw   = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    v1 = hdf5_to_sdp_vis(fn_raw, yaml_raw, conj=True, flip_uvw=True, apply_phasing=False)
    v2 = hdf5_to_sdp_vis(fn_raw, yaml_raw, conj=False, flip_uvw=False, apply_phasing=False)
    assert np.allclose(v1.uvw, -1 * v2.uvw)
    assert np.allclose(np.conj(v1.vis.values), v2.vis.values)

def test_uvdata_to_sdp_vis():
    fn_raw = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    yaml_raw = '../example-config/aavs2/uv_config.yaml'

    uv = hdf5_to_pyuvdata(fn_raw, yaml_raw)
    v = uvdata_to_sdp_vis(uv)

    v2 = hdf5_to_sdp_vis(fn_raw, yaml_raw, apply_phasing=True)

    # Check that data are complex conjugate of each other
    print("---")
    print(v.vis.values[0,1])
    print("---")
    print(v2.vis.values[0, 1])
    print("---")

    assert np.allclose(v.uvw, v2.uvw, atol=0.5e-4)
    assert np.allclose(np.abs(v.vis.values), np.abs(v2.vis.values))
    assert np.allclose(v.vis.values, v2.vis.values)
    print("Hooray!")

if __name__ == "__main__":
    test_sdp_vis()
    test_sdp_vis_conj()
    test_uvdata_to_sdp_vis()
