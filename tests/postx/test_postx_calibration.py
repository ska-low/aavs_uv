"""test_postx_calibration: test routines in postx.calibration."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import get_sun
from ska_ost_low_uv.io import hdf5_to_uvx
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.postx.aa_viewer import AllSkyViewer
from ska_ost_low_uv.postx.simulation.simple_sim import simulate_visibilities_pointsrc
from ska_ost_low_uv.postx.sky_model import generate_skycat
from ska_ost_low_uv.utils import get_aa_config, get_test_data

FN_RAW = get_test_data('aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5')
YAML_RAW = get_aa_config('aavs2')


@pytest.mark.mpl_image_compare
def test_calibration():
    """Test calibration.

    TODO: Get stefcal working.
    """
    vis = hdf5_to_uvx(FN_RAW, yaml_config=YAML_RAW)

    aa = ApertureArray(vis, conjugate_data=False)
    asv = AllSkyViewer(aa)
    sc = generate_skycat(aa)
    asv.load_labels(sc)

    # Uncalibrated
    img_raw = aa.imaging.make_image(128)

    # Generate model visibilities (Sun)
    sky_model = {'sun': get_sun(aa.t[0])}
    v_model = simulate_visibilities_pointsrc(aa, sky_model=sky_model)
    aa.simulation.model.visibilities = v_model

    # Image model visibilities
    img_model = aa.imaging.make_image(128, vis='model')

    # Run stefcal and make calibrated image
    # aa, g = simple_stefcal(aa, sky_model, t_idx=0, f_idx=0, pol_idx=0)
    # img_c = aa.make_image(128)

    # plt.plot(g)
    # plt.show()

    fig = plt.figure(figsize=(10, 4))
    asv.orthview(
        np.log(img_raw),
        overlay_srcs=True,
        subplot_id=(1, 3, 1),
        title='data',
        colorbar=True,
    )
    # asv.orthview(np.log(img_c), overlay_srcs=True, subplot_id=(1,3,2),  title='cal', colorbar=True)
    asv.orthview(
        np.log(img_model),
        overlay_srcs=True,
        subplot_id=(1, 3, 3),
        title='model',
        colorbar=True,
    )
    return fig


if __name__ == '__main__':
    test_calibration()
