"""test_postx: tests for postx utilities and tools."""
import os

import pytest
from ska_ost_low_uv.io import hdf5_to_uvx
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.aa_viewer import AllSkyViewer
from ska_ost_low_uv.postx.sky_model import generate_skycat, sun_model
from ska_ost_low_uv.utils import get_test_data
from astropy.coordinates import SkyCoord


def setup_test() -> ApertureArray:
    """Setup tests.

    Returns:
        aa (ApertureArray): An ApertureArray object to use in testing.
    """
    fn_data = get_test_data('aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5')
    v = hdf5_to_uvx(fn_data, telescope_name='aavs2')

    # RadioArray basics
    aa = ApertureArray(v)
    return aa


def test_postx():
    """Test postx - basic."""
    aa = setup_test()
    print(aa.coords.get_zenith())

    # RadioArray - images
    img  = aa.imaging.make_image()
    hmap = aa.imaging.make_healpix(n_side=64)
    hmap = aa.imaging.make_healpix(n_side=64, apply_mask=False)

    assert img.shape == (128, 128, 4)
    assert hmap.ndim == 2

    # Sky catalog
    skycat = generate_skycat(aa)
    assert isinstance(skycat, dict)
    sun_model(aa, 0)

    # AllSkyViewer
    asv = AllSkyViewer(aa, skycat=skycat)
    asv.load_labels(skycat)
    asv.update()

    print(asv.get_pixel(aa.coords.get_zenith()))
    sc_north = SkyCoord('12:00', '80:00:00', unit=('hourangle', 'deg'))
    assert asv.get_pixel(sc_north) == (0, 0)

    # Simulation
    aa.simulation.sim_vis_gsm()



def test_viewer():
    """Test viewer tools."""
    aa = setup_test()

    # All-sky-viewer via aa
    aa.viewer.orthview()
    aa.viewer.mollview()

    img  = aa.imaging.make_image(n_pix=150)
    hmap = aa.imaging.make_healpix(n_side=64)
    aa.viewer.orthview(img)
    aa.viewer.mollview(hmap)

    try:
        aa.viewer.write_fits(img, 'tests/test.fits')
    finally:
        if os.path.exists('tests/test.fits'):
            os.system('rm tests/test.fits')


@pytest.mark.mpl_image_compare
def test_viewer_orthview():
    """Test orthview."""
    aa = setup_test()
    fig = aa.viewer.new_fig()
    aa.viewer.orthview()
    return fig


@pytest.mark.mpl_image_compare
def test_viewer_mollview():
    """Test mollview."""
    aa = setup_test()
    fig = aa.viewer.new_fig()
    aa.viewer.mollview(fig=fig)
    return fig

if __name__ == "__main__":
    test_postx()
