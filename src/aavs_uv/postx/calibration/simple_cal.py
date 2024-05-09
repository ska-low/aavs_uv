from __future__ import annotations

import numpy as np
from .simple_sim import simulate_visibilities
from .stefcal import stefcal
from aavs_uv.vis_utils import vis_arr_to_matrix

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


def simple_stefcal(aa: ApertureArray, model: dict, t_idx: int=0, f_idx: int=0, pol_idx: int=0) -> tuple[ApertureArray, np.ndarray]:
    """ Apply stefcal to calibrate UV data

    Args:
        aa (ApertureArray): A RadioArray with UV data to calibrate
        model (dict): sky model to use (dictionary of SkyCoords)
        t_idx (int): time index of UV data array
        f_idx (int): frequency index of UV data array
        pol_idx (int): polarization index of UV data array

    Returns:
        aa (ApertureArray): RadioArray with calibration applied
        g (np.ndarray): 1D gains vector, complex data
    """


    v = aa.generate_vis_matrix(f_idx=f_idx, t_idx=t_idx)[..., pol_idx]
    v_model = simulate_visibilities(aa, sky_model=model)

    flags = aa.uvx.antennas.flags
    g, nit, z = stefcal(v, v_model)

    cal = np.outer(np.conj(g), g)
    v_cal = v  / cal
    v_cal[flags] = 0
    v_cal[:, flags] = 0

    aa.workspace['data'] = v_cal

    return aa, g