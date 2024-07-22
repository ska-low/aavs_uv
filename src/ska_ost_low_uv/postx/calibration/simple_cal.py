"""simple_cal: tools to apply simple stefcal."""

from __future__ import annotations

import typing

from loguru import logger

if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray

import numpy as np
from ska_ost_low_uv.datamodel.cal import create_uvx_antenna_cal

from ..aa_module import AaBaseModule
from .stefcal import stefcal


def create_baseline_matrix(xyz: np.array) -> np.ndarray:
    """Create NxN array of baseline lengths.

    Args:
        xyz (np.array): (N_ant, 3) array of antenna locations

    Returns:
        bls (np.array): NxN array of distances between antenna pairs
    """
    N = xyz.shape[0]
    bls = np.zeros((N, N), dtype='float32')
    for ii in range(N):
        bls[:, ii] = np.sqrt(np.sum((xyz - xyz[ii]) ** 2, axis=1))
    return bls


def simple_stefcal(
    aa: ApertureArray,
    sky_model: dict,
    antenna_flags: np.ndarray = None,
    min_baseline: float = None,
    sigma_thr: float = 10,
) -> tuple[ApertureArray, np.ndarray]:
    """Apply stefcal to calibrate UV data.

    Args:
        aa (ApertureArray): A RadioArray with UV data to calibrate
        sky_model (dict): sky model to use (dictionary of SkyCoords)
        antenna_flags (np.ndarray): Antenna flags, shape (N_ant)
        min_baseline (float): Minimum baseline cut in meters. Useful for mitigating
                              diffuse emission, which can dominate short baselines.
        sigma_thr (float): Flag an antenna if its cal magnitude is more than
                           `sigma_thr` standard deviations away from median.

    Returns:
        aa (ApertureArray): RadioArray with calibration applied
        g (np.ndarray): 1D gains vector, complex data
    """
    # Setup indexes and create calibration array
    f_idx, t_idx = aa.idx['f'], aa.idx['t']
    # Arrays have shape (freq, antenna, pol), freq=1, pol=2
    cal_arr = np.zeros((1, aa.n_ant, 2), dtype='complex128')
    flag_arr = np.zeros((1, aa.n_ant, 2), dtype='bool')

    sc_log = {}

    # loop over polarization (and matching stokes index)
    for p_idx, s_idx in ((0, 0), (1, 3)):
        # Generate model visibilities, and convert raw data to visibility matrix
        v_meas = aa.generate_vis_matrix(vis='data')[..., s_idx]
        v_model = aa.simulation.sim_vis_pointsrc(sky_model)[
            t_idx, f_idx, :, :, s_idx
        ].values

        # If minimum baseline is set, flag short baselines
        if min_baseline:
            v_meas[aa.bl_matrix < min_baseline] = 0
            v_model[aa.bl_matrix < min_baseline] = 0

        # load any flags that may be in workspace cal (e.g. from holography)
        # TODO: Add logger info
        if antenna_flags is None and 'cal' in aa.workspace:
            if aa.workspace['cal'] is not None:
                logger.info(f'Using antenna flags from cal in workspace (pol {p_idx})')
                cal = aa.workspace['cal']
                flags = cal.flags[t_idx, ..., p_idx].values
            else:
                flags = np.zeros(aa.n_ant, dtype='bool')
        elif antenna_flags is None and 'cal' not in aa.workspace:
            flags = np.zeros(aa.n_ant, dtype='bool')
        else:
            flags = antenna_flags

        # Strip out flagged antennas
        v_meas_sc = v_meas[~flags][:, ~flags]
        v_model_sc = v_model[~flags][:, ~flags]

        # Run stefcal
        _g, nit, z = stefcal(v_meas_sc, v_model_sc)
        sc_log[f'pol_{p_idx}'] = {'nit': nit, 'z': z}
        logger.info(f'Stefcal pol{p_idx}: nit: {nit} z: {z:.4f}')

        cal_arr[0, ~flags, p_idx] = 1 / np.conj(_g)
        flag_arr[0, :, p_idx] = flags

    # Now, flag if median is > sigma_thr standard deviations away
    cal_abs = np.abs(cal_arr)
    cal_arr_abs_norm = (cal_abs - np.median(cal_abs)) / np.std(cal_abs)

    flag_arr = np.logical_or(flag_arr, cal_arr_abs_norm > sigma_thr)
    cal_arr[flag_arr] = 0

    cal = create_uvx_antenna_cal(
        telescope=aa.name,
        method='stefcal',
        antenna_cal_arr=cal_arr,
        antenna_flags_arr=flag_arr,
        f=aa.f,
        a=aa.ant_names,
        p=np.array(('X', 'Y')),
        provenance={'stefcal': sc_log},
    )

    return cal


####################
## HOLOGRAPHER CLASS
####################


class AaStefcal(AaBaseModule):
    """A class version of the above simple stefcal routine.

    Provides the following functions:
    set_sky_model() - set point source sky model
    run_stefcal() - run stefcal

    """

    def __init__(self, aa: ApertureArray, sky_model: dict = None):
        """Setup AaStefcal.

        Args:
            aa (ApertureArray): Aperture array 'parent' object to use
            sky_model (dict): Point source sky model to use (dictionary of SkyCoords).
        """
        self.aa = aa
        self.sky_model = sky_model
        self.cal = None
        self.__setup_docstrings('stefcal')

    def set_sky_model(self, sky_model: dict):
        """Set/change calibration source.

        Args:
            sky_model (dict): Point source sky model to use (dictionary of SkyCoords).
        """
        self.sky_model = sky_model

    def __setup_docstrings(self, name):
        self.__name__ = name
        self.name = name
        # Inherit docstrings
        self.run_stefcal.__func__.__doc__ = simple_stefcal.__doc__

    def __check_sky_model(self):
        if self.sky_model is None:
            e = 'Point source sky model not set! Run set_sky_model() first.'
            logger.error(e)
            raise RuntimeError(e)

    def run_stefcal(self, *args, **kwargs):
        # Docstring inherited from simple_stefcal function
        self.__check_sky_model()
        if 'sky_model' not in kwargs:
            kwargs['sky_model'] = self.sky_model
        else:
            self.sky_model = kwargs['sky_model']
        self.cal = simple_stefcal(self.aa, *args, **kwargs)
        return self.cal
