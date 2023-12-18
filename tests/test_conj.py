import os
import numpy as np
from pyuvdata import UVData

from aavs_uv.io import hdf5_to_uv, hdf5_to_pyuvdata, hdf5_to_sdp_vis
from aavs_uv.converter import parse_args, run

FN_RAW   = 'test-data/aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5'
YAML_RAW = '../example-config/aavs2/uv_config.yaml'

def test_conj():
    """ Test data conjugation is working """
    print("Testing conjgation: aavs UV")
    vis  = hdf5_to_uv(FN_RAW, telescope_name='aavs2')
    visc = hdf5_to_uv(FN_RAW, YAML_RAW, conj=True)
    visn = hdf5_to_uv(FN_RAW, YAML_RAW, conj=False)

    assert np.allclose(vis.data, visc.data)
    assert np.allclose(visc.data, np.conj(visn.data))

    print("Testing conjgation: SDP vis")
    sdp = hdf5_to_sdp_vis(FN_RAW, telescope_name='aavs2')
    sdpc = hdf5_to_sdp_vis(FN_RAW, YAML_RAW, conj=True)
    sdpn = hdf5_to_sdp_vis(FN_RAW, YAML_RAW, conj=False)

    assert np.allclose(sdp.vis, sdpc.vis)
    assert np.allclose(sdp.vis, np.conj(sdpn.vis))

    print("Testing conjgation: pyuvdata")
    vis = hdf5_to_pyuvdata(FN_RAW, telescope_name='aavs2',  phase_to_t0=False)
    visc = hdf5_to_pyuvdata(FN_RAW, YAML_RAW, phase_to_t0=False, conj=True)
    visn = hdf5_to_pyuvdata(FN_RAW, YAML_RAW, phase_to_t0=False, conj=False)

    assert np.allclose(vis.data_array, visc.data_array)
    assert np.allclose(visc.data_array, np.conj(visn.data_array))

def test_converter():
    try:
        cmd = [
        "-o", "uvfits", "-n", "aavs3", "-j",
        "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
        "test_noconj.uvfits"]
        run(cmd)
        cmd = [
        "-o", "uvfits", "-n", "aavs3",
        "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
        "test.uvfits"]
        run(cmd)

        vis = UVData()
        vis.read_uvfits("test.uvfits")

        visn = UVData()
        visn.read_uvfits("test_noconj.uvfits")       
        assert not np.allclose(vis.data_array, visn.data_array)

    finally:
        if os.path.exists("test.uvfits"):
            os.remove("test.uvfits")  
        if os.path.exists("test_noconj.uvfits"):
            os.remove("test_noconj.uvfits")   

if __name__ == "__main__":
    test_converter()
    test_conj()
