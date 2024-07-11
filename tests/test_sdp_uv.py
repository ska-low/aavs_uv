"""test_sdp: test aa_uv.io SDP read/write."""
import numpy as np
from aa_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, uvdata_to_sdp_vis
from aa_uv.utils import get_aa_config, get_test_data

YAML_RAW = get_aa_config('aavs2')
FN_RAW   = get_test_data('aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5')

def test_sdp_vis():
    """Load data and test visibility generation."""
    v = hdf5_to_sdp_vis(FN_RAW, yaml_config=YAML_RAW)
    print(v)

def test_sdp_vis_conj():
    """Load data and test visibility generation."""
    v1 = hdf5_to_sdp_vis(FN_RAW, yaml_config=YAML_RAW, conj=True, flip_uvw=True, apply_phasing=False)
    v2 = hdf5_to_sdp_vis(FN_RAW, yaml_config=YAML_RAW, conj=False, flip_uvw=False, apply_phasing=False)
    assert np.allclose(v1.uvw, -1 * v2.uvw)
    assert np.allclose(np.conj(v1.vis.values), v2.vis.values)

def test_uvdata_to_sdp_vis():
    """Test conversion of UVData to SDP visibility."""
    uv = hdf5_to_pyuvdata(FN_RAW, yaml_config=YAML_RAW)
    v = uvdata_to_sdp_vis(uv)

    v2 = hdf5_to_sdp_vis(FN_RAW, yaml_config=YAML_RAW, apply_phasing=True)

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
