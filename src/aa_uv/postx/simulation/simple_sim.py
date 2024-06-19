from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray
import numpy as np
from astropy.constants import c
import xarray as xr

LIGHT_SPEED = c.to('m/s').value
cos, sin = np.cos, np.sin

def simulate_visibilities_pointsrc(ant_arr: ApertureArray, sky_model: dict):
    """ Simulate model visibilities for an antenna array

    Args:
        ant_arr (AntArray): Antenna array to use
        sky_model (dict): Sky model to use

    Returns:
        model_vis_matrix (np.array): Model visibilities that should be expected given the known applied delays, (Nchan, Nant, Nant)
    """
    phsmat = None
    for srcname, src in sky_model.items():
        phs = ant_arr.coords.generate_phase_vector(src, conj=True).squeeze()
        if hasattr(src, "mag"):
            phs *= src.mag / np.sqrt(2)

        if phsmat is None:
            phsmat = np.outer(phs, np.conj(phs))
        else:
            phsmat += np.outer(phs, np.conj(phs))

    # Convert to 4-pol
    v0 = np.zeros_like(phsmat)
    V_shape = list(v0.shape)
    V_shape.append(4)
    V = np.zeros_like(phsmat, shape=V_shape)
    V[..., 0] = phsmat
    V[..., 1] = v0
    V[..., 2] = v0
    V[..., 3] = phsmat
    V = np.expand_dims(V, axis=(0, 1))
    V = xr.DataArray(V, dims=('time', 'frequency', 'ant1', 'ant2', 'polarization'))

    return V
