import pytest
import pylab as plt
from aavs_uv.io import hdf5_to_uvx
from aavs_uv.postx import ApertureArray

FN_RAW   = './test-data/aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5'

def test_holography_errs():
    uvx = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    aa = ApertureArray(uvx)

    with pytest.raises(RuntimeError):
        aa.holography.plot_aperture()

    with pytest.raises(RuntimeError):
        aa.holography.plot_aperture_xy()

    with pytest.raises(RuntimeError):
        aa.holography.plot_farfield_beam_pattern()

    with pytest.raises(RuntimeError):
        aa.holography.plot_phasecal_iterations()

def test_holography_selfholo():
    uvx = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    aa = ApertureArray(uvx)

    aa.holography.set_cal_src(aa.get_sun())
    holo_dict = aa.holography.run_selfholo()
    plt.figure()
    aa.holography.plot_aperture()
    #plt.figure()
    aa.holography.plot_aperture_xy()
    plt.figure()
    aa.holography.plot_farfield_beam_pattern()
    plt.show()

if __name__ == "__main__":
    test_holography_errs()
    test_holography_selfholo()
