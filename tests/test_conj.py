"""Test conjugation."""

import os

import numpy as np
from pyuvdata import UVData
from ska_ost_low_uv.converter import run
from ska_ost_low_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, hdf5_to_uvx
from ska_ost_low_uv.utils import get_aa_config, get_test_data

FN_RAW = get_test_data('aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5')
YAML_RAW = get_aa_config('aavs2')


def test_conj():
    """Test data conjugation is working."""
    print('Testing conjgation: aavs UV')
    vis = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    visc = hdf5_to_uvx(FN_RAW, yaml_config=YAML_RAW, conj=True)
    visn = hdf5_to_uvx(FN_RAW, yaml_config=YAML_RAW, conj=False)

    assert np.allclose(vis.data, visc.data)
    assert np.allclose(visc.data, np.conj(visn.data))

    print('Testing conjgation: SDP vis')
    sdp = hdf5_to_sdp_vis(FN_RAW, telescope_name='aavs2', apply_phasing=False)
    sdpc = hdf5_to_sdp_vis(FN_RAW, yaml_config=YAML_RAW, conj=True, apply_phasing=False)
    sdpn = hdf5_to_sdp_vis(
        FN_RAW, yaml_config=YAML_RAW, conj=False, apply_phasing=False
    )

    assert np.allclose(sdp.vis, sdpc.vis)
    assert np.allclose(sdp.vis, np.conj(sdpn.vis))

    print('Testing conjgation: pyuvdata')
    vis = hdf5_to_pyuvdata(FN_RAW, telescope_name='aavs2', phase_to_t0=False)
    visc = hdf5_to_pyuvdata(FN_RAW, yaml_config=YAML_RAW, phase_to_t0=False, conj=True)
    visn = hdf5_to_pyuvdata(FN_RAW, yaml_config=YAML_RAW, phase_to_t0=False, conj=False)

    assert np.allclose(vis.data_array, visc.data_array)
    assert np.allclose(visc.data_array, np.conj(visn.data_array))


def test_converter():
    """Test converter command-line tool."""
    try:
        cmd = [
            '-o',
            'uvfits',
            '-n',
            'aavs3',
            '-j',
            'tests/test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5',
            'tests/test_noconj.uvfits',
        ]
        run(cmd)
        cmd = [
            '-o',
            'uvfits',
            '-n',
            'aavs3',
            'tests/test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5',
            'tests/test.uvfits',
        ]
        run(cmd)

        vis = UVData()
        vis.read_uvfits('tests/test.uvfits')

        visn = UVData()
        visn.read_uvfits('tests/test_noconj.uvfits')
        assert not np.allclose(vis.data_array, visn.data_array)

    finally:
        if os.path.exists('tests/test.uvfits'):
            os.remove('tests/test.uvfits')
        if os.path.exists('tests/test_noconj.uvfits'):
            os.remove('tests/test_noconj.uvfits')


if __name__ == '__main__':
    test_converter()
    test_conj()
