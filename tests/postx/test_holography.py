"""test_holography: Run holographic calibration tests."""
import pylab as plt
import pytest
from aa_uv.io import hdf5_to_uvx
from aa_uv.postx import ApertureArray
from aa_uv.utils import get_test_data

FN_RAW   = get_test_data('aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5')


def setup_test():
    """Setup test data."""
    uvx = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    aa = ApertureArray(uvx)

    aa.calibration.holography.set_cal_src(aa.coords.get_sun())
    holo_dict = aa.calibration.holography.run_selfholo()
    print(holo_dict.keys())
    return aa


def test_holography():
    """Run things and make sure they don't crash."""
    aa = setup_test()
    aa.calibration.holography.run_phasecal()
    aa.calibration.holography.run_jishnucal()

def test_holography_errs():
    """Test that errors are raised."""
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


@pytest.mark.mpl_image_compare
def test_holo_plot_aperture():
    """Test plotting."""
    aa = setup_test()
    fig = plt.figure()
    aa.calibration.holography.plot_aperture()
    return fig


@pytest.mark.mpl_image_compare
def test_holo_plot_aperture_xy():
    """Test plotting."""
    aa = setup_test()
    fig = plt.figure()
    aa.calibration.holography.plot_aperture_xy()
    return fig


@pytest.mark.mpl_image_compare
def test_holo_plot_farfield():
    """Test plotting."""
    aa = setup_test()
    fig = plt.figure()
    aa.calibration.holography.plot_farfield_beam_pattern()
    return fig


@pytest.mark.mpl_image_compare
def test_holo_plot_phasecal_iterations():
    """Test plotting."""
    aa = setup_test()
    aa.calibration.holography.run_phasecal()
    fig = plt.figure()
    aa.calibration.holography.plot_phasecal_iterations()
    return fig


if __name__ == "__main__":
    test_holo_plot_aperture()
