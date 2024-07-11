import pylab as plt
from aa_uv.acacia import AcaciaStorage
from aa_uv.postx import ApertureArray


def test_acacia():
    acacia = AcaciaStorage()
    bucket = 'devel'
    fpath  = 'test/correlation_burst_204_20210612_16699_0.uvx'
    h5 = acacia.get_h5(bucket, fpath, debug=True)

    uvx = acacia.read_uvx(bucket, fpath)
    aa = ApertureArray(uvx)

    aa.calibration.holography.set_cal_src(aa.coords.get_sun())
    holo_dict = aa.calibration.holography.run_selfholo()
    aa.calibration.holography.plot_aperture(plot_type='phs')
    plt.show()

if __name__ == "__main__":
    test_acacia()