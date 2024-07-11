import numpy as np
from aa_uv.io import hdf5_to_uvx
from aa_uv.io.to_pyuvdata import hdf5_to_pyuvdata, uvx_to_pyuvdata


def test_compare_uv_paths():
    """Compare HDF5 -> UVX -> UVData against HDF5 -> UVData path"""
    fn = './test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'

    uvx = hdf5_to_uvx(fn, telescope_name='aavs2')
    uv  = hdf5_to_pyuvdata(fn, telescope_name='aavs2')
    uv2 = uvx_to_pyuvdata(uvx)

    assert np.allclose(np.abs(uv.data_array), np.abs(uv2.data_array))
    assert np.allclose(uv.data_array, uv2.data_array)

    assert np.allclose(uv.uvw_array, uv2.uvw_array)
    assert np.allclose(uv.ant_1_array, uv2.ant_1_array)
    assert np.allclose(uv.ant_2_array, uv2.ant_2_array)
    assert np.allclose(uv.baseline_array, uv2.baseline_array)

    for k in ('cat_name', 'cat_type', 'cat_lon', 'cat_lat'):
        assert uv.phase_center_catalog[1][k] == uv2.phase_center_catalog[1][k]

if __name__ == "__main__":
    test_compare_uv_paths()