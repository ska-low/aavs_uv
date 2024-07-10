from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray
import numpy as np
import healpy as hp
import xarray as xr

from ..coords.coord_utils import hpix2sky
from pyuvsim.analyticbeam import AnalyticBeam
from matvis import simulate_vis

def simulate_visibilities_gsm(aa: ApertureArray, beam_func: function=None) -> xr.DataArray:  # noqa: F821
    """ Use pygdsm + matvis to simulate visibilites, add in Sun """

    f_mhz    = aa.uvx.data.frequency[0] / 1e6
    lsts_rad = aa.uvx.data.lst.values / 24 * np.pi * 2
    flags    = aa.uvx.antennas.flags
    array_lat_rad = float(aa.gsm.lat)

    n_side   = 32

    aa.gsm.generate(f_mhz)

    # Generate sky model flux with GSM
    sm = aa.gsm.observed_gsm
    sm64 = hp.ud_grade(sm, n_side)
    sky_model = np.expand_dims(sm64[~sm64.mask], axis=1)

    # Compute corresponding RA/DEC
    sky_coord = hpix2sky(n_side, np.arange(len(sm64)))
    sky_coord = sky_coord[~sm64.mask].icrs
    ra  = sky_coord.ra.to('rad')
    dec = sky_coord.dec.to('rad')

    if beam_func is not None:
        beams = [AnalyticBeam("func", func=beam_func), ]
    else:
        beams = [AnalyticBeam("uniform"), ]
    ants = dict(zip(np.arange(len(aa.xyz_enu)), aa.xyz_enu))

    vis_vc = simulate_vis(
            ants=ants,
            fluxes=sky_model,
            ra=ra,
            dec=dec,
            freqs=np.array([f_mhz * 1e6, ]),
            lsts=lsts_rad,
            beams=beams,
            polarized=False,
            precision=1,
            latitude=array_lat_rad
        )

    # Convert to 4-pol
    v0 = np.zeros_like(vis_vc)
    V_shape = list(v0.shape)
    V_shape.append(4)
    V = np.zeros_like(vis_vc, shape=V_shape)
    V[..., 0] = vis_vc
    V[..., 1] = v0
    V[..., 2] = v0
    V[..., 3] = vis_vc

    V = xr.DataArray(V, dims=('time', 'frequency', 'ant1', 'ant2', 'polarization'))

    # Add in Sun - TODO
    # Need to get units to match (Jy vs K)
    #sun = sun_model(aa)
    #sky_model = {'sun': sun}
    #v_sun = simulate_visibilities(aa, sky_model=sky_model)

    return V