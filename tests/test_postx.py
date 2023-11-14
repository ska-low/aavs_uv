import numpy as np
import pylab as plt
from astropy.coordinates import SkyCoord

from aavs_uv.postx import RadioArray, AllSkyViewer, generate_skycat
from aavs_uv.io import hdf5_to_uv

def setup_test():
    fn_data = '../../aavs3/2023.10.12-Ravi/correlation_burst_100_20231012_13426_0.hdf5'
    fn_yaml = '../config/aavs3/uv_config.yaml'
    v = hdf5_to_uv(fn_data, fn_yaml)

    # RadioArray basics
    aa = RadioArray(v)
    return aa

def test_postx():
    aa = setup_test()
    print(aa.get_zenith())
    print(aa.zenith)
    
    # RadioArray - various aa.update() calls
    aa.verbose = True
    aa.update()
    aa.verbose = False
    aa.update(t_idx=0)
    aa.conjugate_data = True
    aa.update(f_idx=0)
    aa.update(update_gsm=True)
    
    # RadioArray - images
    img  = aa.make_image()
    hmap = aa.make_healpix(n_side=64)
    hmap = aa.make_healpix(n_side=64, apply_mask=False)
    gsm  = aa.generate_gsm()

    # Sky catalog
    skycat = generate_skycat(aa)

    # AllSkyViewer
    asv = AllSkyViewer(aa, skycat=skycat)
    asv.load_skycat(skycat)
    asv.update()

    print(asv.get_pixel(aa.zenith))
    sc_north = SkyCoord('12:00', '80:00:00', unit=('hourangle', 'deg'))
    assert asv.get_pixel(sc_north) == (0, 0)

def test_postx_plotting():
    aa = setup_test()

    # RadioArray - plotting
    aa.plot_corr_matrix()
    plt.show()
    aa.plot_antennas(overlay_names=True)
    plt.show()
    aa.plot_antennas('E', 'U')
    plt.show()

    # Sky catalog
    skycat = generate_skycat(aa)

    # AllSkyViewer
    asv = AllSkyViewer(aa, skycat=skycat)

    asv.new_fig()
    asv.plot()
    plt.show()
    
    plt.subplot(2,1,1)
    asv.plot(overlay_srcs=True, subplot_id=(2,1,1), colorbar=True)
    plt.subplot(2,1,2)
    pdata = asv.plot(subplot_id=(2,1,2), colorbar=True, return_data=True)
    plt.show()

    asv.mollview(apply_mask=True, fov=np.pi/2)
    plt.show()

if __name__ == "__main__":
    test_postx()
    test_postx_plotting()