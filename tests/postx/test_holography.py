import pytest
import pylab as plt
from aa_uv.io import hdf5_to_uvx
from aa_uv.postx import ApertureArray

FN_RAW   = './test-data/aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5'

def test_holography_errs():
    uvx = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    aa = ApertureArray(uvx)

    with pytest.raises(RuntimeError):
        aa.calibration.holography.plot_aperture()

    with pytest.raises(RuntimeError):
        aa.calibration.holography.plot_aperture_xy()

    with pytest.raises(RuntimeError):
        aa.calibration.holography.plot_farfield_beam_pattern()

    with pytest.raises(RuntimeError):
        aa.calibration.holography.plot_phasecal_iterations()

def test_holography_selfholo():
    uvx = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    aa = ApertureArray(uvx)

    aa.calibration.holography.set_cal_src(aa.coords.get_sun())
    holo_dict = aa.calibration.holography.run_selfholo()
    plt.figure()
    aa.calibration.holography.plot_aperture()
    #plt.figure()
    aa.calibration.holography.plot_aperture_xy()
    plt.figure()
    aa.calibration.holography.plot_farfield_beam_pattern()
    plt.show()

if __name__ == "__main__":
    test_holography_errs()
    test_holography_selfholo()
