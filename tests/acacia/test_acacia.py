"""Test reading data from acacia."""
import pylab as plt
import pytest
from ska_ost_low_uv.acacia import AcaciaStorage
from ska_ost_low_uv.postx import ApertureArray


def test_acacia():
    """Test read from acacia using ROS3 VFD."""
    acacia = AcaciaStorage()
    bucket = 'devel'
    fpath = 'test/correlation_burst_204_20210612_16699_0.uvx'
    h5 = acacia.get_h5(bucket, fpath, debug=True)
    print(h5.keys())

    uvx = acacia.read_uvx(bucket, fpath)
    print(uvx)


@pytest.mark.mpl_image_compare
def test_acacia_plots():
    """Test read and then plot from acacia using ROS3 VFD."""
    acacia = AcaciaStorage()
    bucket = 'devel'
    fpath  = 'test/correlation_burst_204_20210612_16699_0.uvx'
    h5 = acacia.get_h5(bucket, fpath, debug=True)
    print(h5.keys())

    uvx = acacia.read_uvx(bucket, fpath)
    aa = ApertureArray(uvx)

    fig = plt.figure()
    aa.calibration.holography.set_cal_src(aa.coords.get_sun())
    aa.calibration.holography.run_selfholo()
    aa.calibration.holography.plot_aperture(plot_type='phs')
    return fig


if __name__ == '__main__':
    test_acacia()
    test_acacia_plots()
