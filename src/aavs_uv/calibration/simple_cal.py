import numpy as np
from .simple_sim import simulate_visibilities
from .stefcal import stefcal
from aavs_uv.postx import RadioArray
from aavs_uv.utils import vis_arr_to_matrix

def create_baseline_matrix(xyz: np.array) -> np.ndarray:
    """ Create NxN array of baseline lengths 
    
    Args:
        xyz (np.array): (N_ant, 3) array of antenna locations
    
    Returns:
        bls (np.array): NxN array of distances between antenna pairs
    """
    N = xyz.shape[0]
    bls = np.zeros((N, N), dtype='float32')
    for ii in range(N):
        bls[:, ii] = np.sqrt(np.sum((xyz - xyz[ii])**2, axis=1))
    return bls

def sun_model(f: np.array) -> np.array:
    """ Generate sun flux model at given frequencies.

    Flux model values taken from Table 2 of Macario et al (2022).
    A 5th order polynomial is used to interpolate between frequencies.

    Args:
        f (np.array): Frequency, in MHz

    Returns:
        S (np.array): Model flux, in Jy

    Citation:
        Characterization of the SKA1-Low prototype station Aperture Array Verification System 2 
        Macario et al (2022) 
        JATIS, 8, 011014. doi:10.1117/1.JATIS.8.1.011014 
        https://ui.adsabs.harvard.edu/abs/2022JATIS...8a1014M/abstract
    """
    f_i = (50, 100, 150, 200, 300)        # Frequency in MHz
    α_i = (2.15, 1.86, 1.61, 1.50, 1.31)  # Spectral index 
    S_i = (5400, 24000, 81000, 149000)    # Flux in Jy
    
    p_S = np.poly1d(np.polyfit(f_i, S_i, 5))

    return p_S(f)

def simple_stefcal(aa: RadioArray, model: dict, t_idx: int=0, f_idx: int=0, pol_idx: int=0) -> (RadioArray, np.array):
    """ Apply stefcal to calibrate UV data 
    
    Args:
        aa (RadioArray): A RadioArray with UV data to calibrate
        model (dict): sky model to use (dictionary of SkyCoords)
        t_idx (int): time index of UV data array
        f_idx (int): frequency index of UV data array
        pol_idx (int): polarization index of UV data array
    
    Returns:
        aa (RadioArray): RadioArray with calibration applied
        g (np.array): 1D gains vector, complex data
    """
    aa.update(t_idx, f_idx, pol_idx, update_gsm=False)

    d = aa.vis.data[t_idx, f_idx, :,  pol_idx]
    v = vis_arr_to_matrix(d, aa.n_ant, 'upper', conj=True)
    
    v_model = simulate_visibilities(aa, sky_model=model)

    flags = aa.vis.antennas.flags
    g, nit, z = stefcal(v, v_model)

    cal = np.outer(np.conj(g), g)
    v_cal = v  / cal
    v_cal[flags] = 0
    v_cal[:, flags] = 0
    
    aa.workspace['data'] = v_cal

    return aa, g