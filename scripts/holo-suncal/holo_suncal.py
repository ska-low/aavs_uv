#!/usr/bin/env python
"""Run suncal using Jishnu self-holography on HDF5 data."""
import glob
import os
import warnings

import numpy as np
import pylab as plt
from astropy.units import Quantity
from lmfit import Parameters, minimize
from loguru import logger
from ska_ost_low_uv.datamodel.cal import UVXAntennaCal, create_uvx_antenna_cal
from ska_ost_low_uv.io import hdf5_to_uvx, read_cal, write_cal
from ska_ost_low_uv.parallelize import run_in_parallel, task
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.utils import reset_logger
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_cal_filelist(fpath: str, cmin: int=64, cmax: int=448, ext: str='hdf5', gstr: str='correlation_burst') -> list:
    """Load list of HDF5 files, sorted by channel index.

    Notes:
        Filenames must follow `correlation_burst_{cidx}_*.hdf5` pattern

    Args:
        fpath (str): Path to parent directory of HDF5 files.
        cmin (int): Lowest channel to load. Default 64.
        cmax (int): Highest channel to load. Default 448.
        ext (str): File extension. Default hdf5
        gstr (str): Start of filename for filtering, default 'correlation_burst'

    Returns:
        filelist (list): Sorted list of HDF5 files
    """
    fl = sorted(glob.glob(f'{fpath}/{gstr}*.{ext}'))
    bl = []

    for fn in fl:
        idx = int(os.path.basename(fn).split('_')[2])
        if cmin <= idx <= cmax:
            bl.append([idx, fn])

    return [y[1] for y in sorted(bl, key=lambda x: x[0])]


def generate_chisq_flags(chisqr: np.ndarray, sigma: float=5) -> np.ndarray:
    """Flag antennas where chi-squared fit is bad.

    Args:
        chisqr (np.ndarray): Output of least-squares fit chi-squared.
                             Shape is (N_ant,)
        sigma (float): Flag threshold in terms of STDEV(log(chisq)).
                             Shape is (N_ant,)

    Returns:
        flags (np.ndarray): Flags, True means bad. Shape is (N_ant,)
    """
    chisq_log = np.log(np.array(chisqr) + 1)
    chisq_log_ok = chisq_log[chisq_log > 0.7]

    cs_avg = np.median(chisq_log_ok)

    # Redo STD calc with 95% of data
    cs_std = np.std(chisq_log_ok)
    potentially_bad = np.abs(chisq_log_ok - cs_avg) > 2 * cs_std
    cs_std = np.std(chisq_log_ok[~potentially_bad])

    flags = np.abs(chisq_log - cs_avg) > sigma * cs_std
    return flags


def generate_basic_frequency_flags(freqs: np.ndarray) -> np.ndarray:
    """Create basic frequency flags based on known RFI.

    Flags 85-110, 130-145, and > 240 MHz.

    Args:
        freqs (np.ndarray): Frequency in Hz, shape (N_freq,)

    Returns:
        flags (np.ndarray): Boolean flags (True is bad). Shape (N_freq,)
    """
    rfi0 = np.logical_and(freqs > 85e6, freqs < 110e6)
    rfi1 = np.logical_and(freqs > 130e6, freqs < 145e6)
    rfi2 = freqs > 240e6

    flags = np.logical_or(rfi0, rfi1)
    flags = np.logical_or(flags, rfi2)

    return flags


def run_suncal(filelist: list, n_workers: int=8, outpath: str='cal'):
    """Run jishnucal."""

    @task
    def calibrate(fn, telescope_name, outpath):
        """Helper task for running a calibration."""
        reset_logger(use_tqdm=True, level='ERROR')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        uvx = hdf5_to_uvx(fn, telescope_name='aavs3')
        aa  = ApertureArray(uvx)

        aa.calibration.holography.set_cal_src(aa.coords.get_sun())
        cal = aa.calibration.holography.run_jishnucal()

        calpath = f'cal/{os.path.basename(fn).replace(".hdf5", ".cal")}'
        write_cal(cal, calpath)
        return 'Done'

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Run jishnucal on each file
    tasklist = [calibrate(fn, 'aavs3', outpath) for fn in filelist]
    run_in_parallel(tasklist, n_workers=n_workers, backend='dask')
    reset_logger(use_tqdm=True, level='INFO')

    # Generate calibration
    calfiles = get_cal_filelist('cal', ext='cal')
    cals = [read_cal(c) for c in calfiles]

    # Combine calibrations
    c0 = cals[0]
    cal_sweep = np.zeros(shape=(len(cals), *c0.gains.shape[1:]), dtype='complex64')
    cal_sweep_flags = np.zeros_like(cal_sweep, dtype='bool')

    freqs = np.zeros(shape=(len(cals),))
    for ii in range(len(cals)):
        cal_sweep[ii] = cals[ii].gains
        cal_sweep_flags[ii] = cals[ii].flags
        freqs[ii] = cals[ii].gains.frequency[0]

    # Convert into suncal
    suncal = create_uvx_antenna_cal(
        telescope='aavs3',
        method='suncal',
        antenna_gains_arr=cal_sweep,
        antenna_flags_arr=cal_sweep_flags,
        f=Quantity(freqs, unit='Hz'),
        a=c0.gains.antenna,
        p=c0.gains.polarization,
        provenance=c0.provenance)

    write_cal(suncal, f'{outpath}/suncal_combined.cal')

    return suncal


def fit_phase(cal: UVXAntennaCal, n_workers: int=8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit linear phase to calibration solutions.

    Args:
        cal (UVXAntennaCal): Calibration solutions. Shape (N_freq, N_ant, N_pol)
        n_workers (int): Number of worker threads

    Returns:
        taus (np.ndarray): time delay solution, shape (N_ant, N_pol)
        p0s (np.ndarray): phase angle zero point solution, shape (N_ant, N_pol)
        flags_ant (np.ndarray): Antenna flags, boolean, shape (N_ant, N_pol)
    """

    def residual(params, f, data):
        tau = params['tau']
        phs0 = params['phs0']
        model = phs(f, tau, phs0)
        return np.unwrap(data) - np.unwrap(model)

    @task
    def minimize_task(residual, f, freq_flags, data):
        reset_logger(use_tqdm=True, level='ERROR')
        params = Parameters()
        params.add('tau', value=2, min=0, max=10)
        params.add('phs0', value=0, min=-np.pi, max=np.pi)

        out = minimize(residual, params,
                           args=(f[~freq_flags], data),
                           method='ampgo')
        return out

    f = cal.gains.frequency.values
    cal_sweep = cal.gains

    freq_flags = generate_basic_frequency_flags(f)

    taus = np.zeros((cal.gains.shape[1], cal.gains.shape[2]))
    p0s  = np.zeros_like(taus)
    chisq = np.zeros_like(taus)
    flags_ant = np.zeros_like(taus, dtype='bool')

    tasks = []
    for ii in range(taus.shape[0]):
        for pp in range(taus.shape[1]):
            data = np.angle(cal_sweep[~freq_flags, ii, pp])
            out = minimize_task(residual, f, freq_flags, data)
            tasks.append(out)

    out_list = run_in_parallel(tasks, n_workers=n_workers, backend='dask')

    for idx, out in enumerate(out_list):
        ii = idx // 2
        pp = idx % 2
        taus[ii, pp] = out.params['tau'].value
        p0s[ii, pp] = out.params['phs0'].value
        chisq[ii, pp] = out.chisqr

        # Flag any antenna that is flagged more than 50% of the time
        flags_cal = cal.flags[..., pp].sum(axis=0) > cal.flags.shape[0] / 2
        flags_chisq = generate_chisq_flags(chisq[..., pp], sigma=10)

        flags_ant[..., pp] = np.logical_or(flags_chisq, flags_cal)

    return taus, p0s, flags_ant


def phs(f, tau, phs0, return_as_angle=True):
    """Phase slope model exp(2i pi (tau f / c + phs0))."""
    m = np.exp(2j*np.pi*(tau*f/2.99e8 + phs0))
    if return_as_angle:
        return np.angle(m)
    else:
        return m

def _plot_phs(f_mhz, freq_flags, ant_flag, phs_m, phs_d):
    """Plot phase helper."""
    plt.scatter(f_mhz[~freq_flags], phs_d[~freq_flags], marker='.')
    if not ant_flag:
        plt.scatter(f_mhz, phs_m, marker='.')
    plt.xlim(50, 250)
    plt.xticks([50, 100, 150, 200, 250], fontsize=8)
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["-$\\pi$", "-$\\pi/2$", 0, "$\\pi/2$", "$\\pi$"], fontsize=8)

if __name__ == "__main__":
    import argparse

    import holoviews as hv
    hv.extension('bokeh')

    p = argparse.ArgumentParser(prog='HoloSuncal',
                                description='Run calibration using Sun and self-holography')
    p.add_argument('dirpath', help='Path to directory with correlator HDF5 file frequency sweep.')
    p.add_argument('-l', '--low_channel', type=int, default=64, help='Start channel ID (default 64).')
    p.add_argument('-m', '--high_channel', type=int, default=448, help='Stop channel ID (default 448).')
    p.add_argument('-n', '--n_workers', type=int, default=8, help='Number of worker threads in parallel loop.')
    p.add_argument('-o', '--outpath', type=str, default='cal', help='Calibration output directory.')
    p.add_argument('-p', '--plot_fmt', type=str, default='png', help='Plot output format, one of png (default), svg, html, gif.')

    args = p.parse_args()
    reset_logger()

    # Get list of files
    fl = get_cal_filelist(args.dirpath, cmin=args.low_channel, cmax=args.high_channel)

    # Run Suncal
    logger.info("Running suncal...")
    suncal = run_suncal(fl, n_workers=args.n_workers, outpath=args.outpath)
    cal_sweep = suncal.gains.values
    f = suncal.gains.frequency.values
    freq_flags = generate_basic_frequency_flags(f)

    # Fit phase slope
    logger.info("Fitting phases...")
    taus, p0s, flags_chisq = fit_phase(suncal)

    # Save fitted gains to file
    logger.info(f"Saving to {args.outpath}/suncal_combined_fitted.cal...")
    fitted_cal = read_cal(f'{args.outpath}/suncal_combined.cal')
    fitted_gains = np.zeros_like(fitted_cal.gains.values, dtype='complex64')
    for ii in range(fitted_gains.shape[1]):
        fitted_gains[:, ii, 0] = phs(f, taus[ii, 0], p0s[ii, 0], return_as_angle=False)
        fitted_gains[:, ii, 1] = phs(f, taus[ii, 1], p0s[ii, 1], return_as_angle=False)
    fitted_cal.gains[:] = fitted_gains
    write_cal(fitted_cal, f'{args.outpath}/suncal_combined_fitted.cal')

    f = suncal.gains.frequency.values
    f_mhz = f / 1e6

    # fmt: off
    freq_flags   = generate_basic_frequency_flags(f)
    cal_sweep    = suncal.gains.values
    fitted_gains = fitted_cal.gains.values
    ant_ids      = suncal.gains.antenna.values
    pol_ids      = suncal.gains.polarization.values
    # fmt: on


    # Generate plots
    logger.info("Plotting...")

    bad_antennas_x_idx = np.arange(256)[flags_chisq[:, 0]]
    bad_antennas_y_idx = np.arange(256)[flags_chisq[:, 1]]

    print("Bad X:")
    print(bad_antennas_x_idx)
    print(ant_ids[bad_antennas_x_idx])
    print("Bad Y:")
    print(bad_antennas_y_idx)
    print(ant_ids[bad_antennas_y_idx])

    for tpm_idx in tqdm(range(16)):
        for pp in (0, 1):
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 6))
            plt.subplots_adjust(wspace=0.3, hspace=0.5, left=0.1, right=0.9)
            for ii in range(16):
                data_idx = 16*tpm_idx + ii
                ant_name = ant_ids[data_idx]
                pol_name = pol_ids[pp]

                plt.subplot(4, 4, ii+1)
                ant_flag = flags_chisq[data_idx, pp]
                _plot_phs(f_mhz, freq_flags, ant_flag, np.angle(fitted_gains[:, data_idx, pp]), np.angle(cal_sweep[:, data_idx, pp]))
                plt.title(f"{ant_name} {pol_name}", fontsize=8)
            plt.savefig(f"cal/cal_solutions_tpm_{tpm_idx}_{pp}.png")
            plt.clf()
            plt.close()
