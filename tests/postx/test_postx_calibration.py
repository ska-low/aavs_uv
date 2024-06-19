import pylab as plt
import numpy as np
from astropy.coordinates import get_sun

from aa_uv.io import hdf5_to_uvx
from aa_uv.postx import ApertureArray
from aa_uv.postx.aa_viewer import AllSkyViewer
from aa_uv.postx.sky_model import generate_skycat
from aa_uv.postx.simulation.simple_sim import simulate_visibilities_pointsrc
from aa_uv.postx.simulation.gsm_sim import simulate_visibilities_gsm


FN_RAW   = 'test-data/aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5'
YAML_RAW = '../src/aa_uv/config/aavs2/uv_config.yaml'

def test_calibration():
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
    #aa, g = simple_stefcal(aa, sky_model, t_idx=0, f_idx=0, pol_idx=0)
    #img_c = aa.make_image(128)

    #plt.plot(g)
    #plt.show()

    plt.figure(figsize=(10, 4))
    asv.orthview(np.log(img_raw), overlay_srcs=True, subplot_id=(1,3,1), title='data',  colorbar=True)
    #asv.orthview(np.log(img_c), overlay_srcs=True, subplot_id=(1,3,2),  title='cal', colorbar=True)
    asv.orthview(np.log(img_model), overlay_srcs=True, subplot_id=(1,3,3), title='model', colorbar=True)
    plt.show()

if __name__ == "__main__":
    test_calibration()