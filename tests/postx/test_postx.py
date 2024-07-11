import numpy as np
import pylab as plt
from aa_uv.io import hdf5_to_uvx
from aa_uv.postx import ApertureArray
from aa_uv.postx.aa_viewer import AllSkyViewer
from aa_uv.postx.sky_model import generate_skycat, sun_model
from astropy.coordinates import SkyCoord


def setup_test():

    fn_data = 'test-data/aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5'
    v = hdf5_to_uvx(fn_data, telescope_name='aavs2')

    # RadioArray basics
    aa = ApertureArray(v)
    return aa

def test_postx():
    aa = setup_test()
    print(aa.coords.get_zenith())

    # RadioArray - images
    img  = aa.imaging.make_image()
    hmap = aa.imaging.make_healpix(n_side=64)
    hmap = aa.imaging.make_healpix(n_side=64, apply_mask=False)
    gsm  = aa.simulation.mollview_gsm()

    # Sky catalog
    skycat = generate_skycat(aa)
    sun = sun_model(aa, 0)

    # AllSkyViewer
    asv = AllSkyViewer(aa, skycat=skycat)
    asv.load_labels(skycat)
    asv.update()

    print(asv.get_pixel(aa.coords.get_zenith()))
    sc_north = SkyCoord('12:00', '80:00:00', unit=('hourangle', 'deg'))
    assert asv.get_pixel(sc_north) == (0, 0)

def test_postx_plotting():
    aa = setup_test()

    # Sky catalog
    skycat = generate_skycat(aa)

    # AllSkyViewer
    asv = AllSkyViewer(aa, skycat=skycat)

    asv.new_fig()
    asv.orthview()
    plt.show()

    plt.subplot(2,1,1)
    asv.orthview(overlay_srcs=True, subplot_id=(2,1,1), colorbar=True)
    plt.subplot(2,1,2)
    pdata = asv.orthview(subplot_id=(2,1,2), colorbar=True, return_data=True)
    plt.show()

    asv.mollview(apply_mask=True, fov=np.pi/2)
    plt.show()

if __name__ == "__main__":
    test_postx()
    test_postx_plotting()