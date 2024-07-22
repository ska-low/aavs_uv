"""cal: Data models for calibration data."""

from dataclasses import dataclass

import numpy as np
import xarray as xp
from astropy.units import Quantity
from ska_ost_low_uv.utils import get_resource_path, get_software_versions, load_yaml


@dataclass
class UVXAntennaCal:
    """Dataclass for antenna calibration to accompany UVX data."""

    # fmt: off
    telescope: str          # Antenna array name, e.g. AAVS3
    method: str             # Calibration method name (e.g. JishnuCal)
    gains: xp.DataArray     # An xarray dataset  (frequency, antenna, pol)
    flags: xp.DataArray     # Flag xarray dataset (frequency, antenna, pol)
    provenance: dict        # Provenance/history information and other metadata
    # fmt: on

    def to_matrix(self, f_idx: int = 0) -> np.ndarray:
        """Convert to visibility matrix.

        Args:
            f_idx (int): Frequency index to load

        Returns:
            cal_mat (np.ndarray): Calibration matrix, (N_ant, N_ant, N_stokes)
        """
        gc = self.gains
        cal_mat = np.zeros((gc.shape[1], gc.shape[1], 4), dtype='complex64')
        cal_mat[..., 0] = np.outer(gc[f_idx, ..., 0], gc[f_idx, ..., 0])
        cal_mat[..., 1] = np.outer(gc[f_idx, ..., 0], gc[f_idx, ..., 1])
        cal_mat[..., 2] = np.outer(gc[f_idx, ..., 1], gc[f_idx, ..., 0])
        cal_mat[..., 3] = np.outer(gc[f_idx, ..., 1], gc[f_idx, ..., 1])

        return cal_mat

    def report_flagged_antennas(self):
        """Report bad (flagged) antennas.

        Returns:
            bad_ants (dict):
        """
        flags_x, flags_y = np.max(self.flags, axis=0).T
        flags_idx = np.arange(len(flags_x))

        flag_dict = {
            'x': {'idx': flags_idx[flags_x], 'name': flags_x.antenna[flags_x].values},
            'y': {'idx': flags_idx[flags_y], 'name': flags_y.antenna[flags_y].values},
        }

        return flag_dict


def create_provenance_dict():
    """Create a provenance dict, fill in software versions."""
    provenance = {'ska_ost_low_uv_config': get_software_versions()}
    return provenance


def _create_antenna_cal_coords(f: Quantity, a: np.ndarray, p: np.ndarray) -> dict:
    """Create dictionary of coords, for xp.DataArray kwarg.

    Args:
        f (Quantity): Astropy Quantity array corresponding to frequency axis (specified at channel center)
        a (np.ndarray): List of antenna IDs
        p (np.ndarray): Polarization labels, e.g ('X', 'Y')
        cal_type (str): Calibration type,

    Returns:
        coords (dict): coords for xarray DataArray kwarg
    """
    cal_schema = load_yaml(get_resource_path('datamodel/cal.yaml'))

    # Coordinate - antenna
    a_coord = xp.DataArray(
        a,
        dims=('antenna',),
        attrs={'description': cal_schema['cal/coords/antenna']['description']},
    )

    # Coordinate - polarization
    p_coord = xp.DataArray(
        p,
        dims=('polarization',),
        attrs={'description': cal_schema['cal/coords/polarization']['description']},
    )

    # Coordinate - frequency
    f_center = f.to('Hz').value
    f_coord = xp.DataArray(
        f_center,
        dims=('frequency',),
        attrs={
            'units': cal_schema['cal/coords/frequency']['units'],
            'description': cal_schema['cal/coords/frequency']['description'],
        },
    )

    coords = {'polarization': p_coord, 'antenna': a_coord, 'frequency': f_coord}
    return coords


def create_antenna_flags(
    antenna_flag_arr: np.ndarray, f: Quantity, a: np.ndarray, p: np.ndarray
) -> xp.DataArray:
    """Create an xarray dataarray for antenna calibration cofficients.

    Args:
        antenna_flag_arr (np.array): Boolean antenna flags.
                                    Shape: (freq, antenna, pol) dtype=bool
        a (np.ndarray): List of antenna IDs
        f (Quantity): Astropy Quantity array corresponding to frequency axis (specified at channel center)
        p (np.ndarray): Polarization labels, e.g ('XX','XY','YX','YY')

    Returns:
        antenna_cal (xp.DataArray): xarray Dataset with antenna locations
    """
    cal_schema = load_yaml(get_resource_path('datamodel/cal.yaml'))

    coords = _create_antenna_cal_coords(f, a, p)

    antenna_flags = xp.DataArray(
        antenna_flag_arr,
        coords=coords,
        dims=cal_schema['cal/gains']['dims'],
        attrs={'description': cal_schema['cal/gains']['description']},
    )

    return antenna_flags


def create_antenna_gains(
    antenna_cal_arr: np.ndarray, f: Quantity, a: np.ndarray, p: np.ndarray
) -> xp.DataArray:
    """Create an xarray dataarray for antenna calibration cofficients.

    Args:
        antenna_cal_arr (np.array): Complex antenna calibration coefficients.
                                    Shape: (freq, antenna, pol) complex-valued
        a (np.ndarray): List of antenna IDs
        f (Quantity): Astropy Quantity array corresponding to frequency axis (specified at channel center)
        p (np.ndarray): Polarization labels, e.g ('XX','XY','YX','YY')

    Returns:
        antenna_cal (xp.DataArray): xarray Dataset with antenna locations
    """
    cal_schema = load_yaml(get_resource_path('datamodel/cal.yaml'))

    coords = _create_antenna_cal_coords(f, a, p)

    antenna_cal = xp.DataArray(
        antenna_cal_arr,
        coords=coords,
        dims=cal_schema['cal/flags']['dims'],
        attrs={'description': cal_schema['cal/flags']['description']},
    )

    return antenna_cal


def create_uvx_antenna_cal(
    telescope: str,
    method: str,
    antenna_gains_arr: np.ndarray,
    antenna_flags_arr: np.ndarray,
    f: Quantity,
    a: np.ndarray,
    p: np.ndarray,
    provenance: dict = None,
) -> UVXAntennaCal:
    """Create an UVXAntennaCal for antenna locations.

    Args:
        telescope (str): Name of telescope
        method (str): Calibration method used to generate
        antenna_gains_arr (np.array): Antenna calibration coefficients.
                                    Shape: (freq, antenna, pol) complex-valued
        antenna_flags_arr (np.ndarray): Boolean antenna flag array, True = flagged (bad)
                                        Shape: (freq, antenna, pol), boolean
        a (np.ndarray): List of antenna IDs
        f (Quantity): Astropy Quantity array corresponding to frequency axis (specified at channel center)
        p (np.ndarray): Polarization labels, e.g ('XX','XY','YX','YY')
        provenance (dict): Dictionary of provenance metadata to add

    Returns:
        antenna_cal (xp.Dataset): xarray Dataset with antenna locations
    """
    antenna_cal = create_antenna_gains(antenna_gains_arr, f, a, p)
    antenna_flags = create_antenna_flags(antenna_flags_arr, f, a, p)

    # Create empty provenance dictionary if not passed, then fill with creation info
    provenance = create_provenance_dict() if provenance is None else provenance
    provenance.update({'ska_ost_low_uv_config': get_software_versions()})

    uvx_cal = UVXAntennaCal(
        telescope=telescope,
        method=method,
        gains=antenna_cal,
        flags=antenna_flags,
        provenance=provenance,
    )

    return uvx_cal
