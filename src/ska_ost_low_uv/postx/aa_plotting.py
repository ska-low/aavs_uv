"""aa_plotting: ApertureArray plotting tools submodule."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ..postx.aperture_array import ApertureArray

import numpy as np
import pylab as plt

from .aa_module import AaBaseModule


def plot_corr_matrix(
    aa: ApertureArray,
    vis: str = 'data',
    t_idx: int = 0,
    f_idx: int = 0,
    p_idx: int = 0,
    sfunc: np.ufunc = np.log,
    **kwargs,
):
    """Plot correlation matrix.

    Args:
        aa (ApertureArray): Aperture array 'parent' object to use
        vis (str): One of 'data', 'corrected', or 'model'
        t_idx (int): Time index of visibility data
        f_idx (int): Frequency index of visibility data
        p_idx (int): Polarization index of visibility data
        sfunc (np.ufunc): scaling function to apply to data, e.g. np.log
        **kwargs (dict): Keyword arguments to pass on to plt.imshow
    """
    data = np.abs(aa.generate_vis_matrix(vis, t_idx, f_idx))
    data = data[..., p_idx]
    if sfunc is not None:
        data = sfunc(data)

    pol_label = aa.uvx.data.polarization.values[p_idx]
    plt.imshow(data, aspect='equal', **kwargs)
    plt.title(pol_label)
    plt.xlabel('Antenna P')
    plt.ylabel('Antenna Q')
    ufunc_str = str(sfunc).replace("<ufunc '", '').replace("'>", '')
    plt.colorbar(label=f'counts ({ufunc_str})')


def plot_corr_matrix_4pol(aa: ApertureArray, **kwargs):
    """Plot correlation matrix, for all pols.

    Args:
        aa (ApertureArray): Aperture array 'parent' object to use
        vis (str): One of 'data', 'corrected', or 'model'
        t_idx (int): Time index of visibility data
        f_idx (int): Frequency index of visibility data
        sfunc (np.ufunc): scaling function to apply to data, e.g. np.log
        **kwargs (dict): Keyword arguments to pass on to plt.imshow
    """
    fig = plt.figure(figsize=(10, 8))
    for ii in range(4):
        plt.subplot(2, 2, ii + 1)
        plot_corr_matrix(aa, p_idx=ii, **kwargs)
    plt.tight_layout()
    return fig


def plot_antennas(
    aa: ApertureArray,
    x: str = 'E',
    y: str = 'N',
    overlay_names: bool = False,
    overlay_fontsize: str = 'x-small',
    **kwargs,
):
    """Plot antenna locations in ENU.

    Args:
        aa (ApertureArray): Aperture array 'parent' object to use
        x (str): One of 'E', 'N', or 'U'
        y (str): One of 'E', 'N', or 'U'
        overlay_names (bool): Overlay the antenna names on the plot. Default False
        overlay_fontsize (str): Font size for antenna names 'xx-small', 'x-small', 'small', 'medium',
                                                            'large', 'x-large', 'xx-large'
        **kwargs (dict): Keyword arguments to pass on to plt.scatter
    """
    ax = plt.subplot(1, 1, 1)
    ax.axis('equal')
    enu_map = {'E': 0, 'N': 1, 'U': 2}
    title = f'{aa.name} | Lon: {aa.earthloc.to_geodetic().lon:.2f} | Lat: {aa.earthloc.to_geodetic().lat:.2f}'
    plt.scatter(aa.xyz_enu[:, enu_map[x.upper()]], aa.xyz_enu[:, enu_map[y.upper()]], **kwargs)
    plt.xlabel(f'{x} [m]')
    plt.ylabel(f'{y} [m]')

    if overlay_names:
        names = aa.vis.antennas.attrs['identifier'].data
        for ii in range(aa.n_ant):
            plt.text(
                aa.xyz_enu[:, enu_map[x]][ii],
                aa.xyz_enu[:, enu_map[y]][ii],
                names[ii],
                fontsize=overlay_fontsize,
            )
    plt.title(title)


def plot_uvdist_amp(
    aa: ApertureArray,
    vis: str = 'data',
    pol_idx: int = 0,
    sfunc: np.ufunc = None,
    **kwargs,
):
    """Plot amplitude as function of UV distance.

    Args:
        aa (ApertureArray): Aperture array 'parent' object to use
        vis (str): One of 'data', 'corrected', or 'model'
        pol_idx (int): Polarization index
        sfunc (np.unfunc): Scaling function to apply, e.g. np.log
        **kwargs (dict): Keyword arguments to pass on to plt.scatter
    """
    bls = aa.bl_matrix
    amp = np.abs(aa.generate_vis_matrix(vis)[..., pol_idx])

    title = f'{aa.name} | UV Amplitude'

    if 'marker' not in kwargs.keys():
        kwargs['marker'] = '.'

    if 's' not in kwargs.keys():
        kwargs['s'] = 8

    if 'alpha' not in kwargs.keys():
        kwargs['alpha'] = 0.2

    # Apply scaling function
    pdata = amp.ravel() if sfunc is None else sfunc(amp.ravel())

    plt.scatter(bls.ravel(), pdata, **kwargs)
    plt.xlabel('UV Distance [m]')
    plt.ylabel('Amplitude')
    plt.title(title)


####################
## AA_PLOTTER CLASS
####################
class AaPlotter(AaBaseModule):
    """A class for plotting utilties.

    Provides the following functions:
    plot_corr_matrix()
    plot_corr_matrix_4pol()
    plot_antennas()
    plot_uv_dist_amp()

    """

    def __init__(self, aa: ApertureArray):
        """Setup AaPlotter.

        Args:
            aa (ApertureArray): Aperture array 'parent' object to use
        """
        self.aa = aa
        self.__setup_docstrings('plotting')

    def __setup_docstrings(self, name):
        self.__name__ = name
        self.name = name
        # Inherit docstrings
        self.plot_corr_matrix.__func__.__doc__ = plot_corr_matrix.__doc__
        self.plot_corr_matrix_4pol.__func__.__doc__ = plot_corr_matrix_4pol.__doc__
        self.plot_antennas.__func__.__doc__ = plot_antennas.__doc__
        self.plot_uvdist_amp.__func__.__doc__ = plot_uvdist_amp.__doc__

    def plot_corr_matrix(self, *args, **kwargs):
        # Docstring inherited
        plot_corr_matrix(self.aa, *args, **kwargs)

    def plot_corr_matrix_4pol(self, *args, **kwargs):
        # Docstring inherited
        return plot_corr_matrix_4pol(self.aa, *args, **kwargs)

    def plot_antennas(self, *args, **kwargs):
        # Docstring inherited
        plot_antennas(self.aa, *args, **kwargs)

    def plot_uvdist_amp(self, *args, **kwargs):
        # Docstring inherited
        plot_uvdist_amp(self.aa, *args, **kwargs)
