import pylab as plt
import numpy as np
from astropy.coordinates import get_sun

from aavs_uv.io import hdf5_to_uv
from aavs_uv.postx import RadioArray, AllSkyViewer, generate_skycat
from aavs_uv.calibration import simulate_visibilities, simple_stefcal

FN_RAW   = '/Users/daniel.price/Data/aavs3/correlation_burst_204_20231027_18926_0.hdf5'
YAML_RAW = '/Users/daniel.price/Data/aavs_uv/config/aavs3/uv_config.yaml'

def test_calibration():
    vis = hdf5_to_uv(FN_RAW, YAML_RAW)

    aa = RadioArray(vis, conjugate_data=True)
    asv = AllSkyViewer(aa)
    sc = generate_skycat(aa)
    asv.load_skycat(sc)

    # Uncalibrated
    img_raw = aa.make_image(128)

    # Generate model visibilities (Sun)
    sky_model = {'sun': get_sun(aa.t[0])}
    v_model = simulate_visibilities(aa, sky_model=sky_model)

    # Image model visibilities
    aa.workspace['data'] = v_model
    img_model = aa.make_image(128)

    # Run stefcal and make calibrated image
    aa, g = simple_stefcal(aa, sky_model, t_idx=0, f_idx=0, pol_idx=0)
    img_c = aa.make_image(128)

    plt.figure(figsize=(10, 4))
    asv.plot(np.log(img_raw), overlay_srcs=True, subplot_id=(1,3,1), title='data',  colorbar=True)
    asv.plot(np.log(img_c), overlay_srcs=True, subplot_id=(1,3,2),  title='cal', colorbar=True)
    asv.plot(np.log(img_model), overlay_srcs=True, subplot_id=(1,3,3), title='model', colorbar=True)
    plt.show()

if __name__ == "__main__":
    test_calibration()