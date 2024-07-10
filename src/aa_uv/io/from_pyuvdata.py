from loguru import logger
import pandas as pd
import numpy as np

from astropy.time import Time
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, EarthLocation

from aa_uv.datamodel.uvx import (
    UVX, create_antenna_data_array, create_visibility_array,
    create_empty_context_dict, create_empty_provenance_dict
)

from pyuvdata import UVData
import pyuvdata.utils as uvutils

def convert_data_to_uvx_convention(uv: UVData, check: bool=False) -> np.array:
    """Convert the uv.data_array to UVX data convention

    Notes:
        * Assumes the data are in triangular format with baselines sorted and contiguous.
        * Assumes the data polarization runs (XX, YY, XY, YX) i.e. (-5, -6, -7, -8)
        * pyuvdata data_array convention has shape (baseline * time, 1, freq, pol)
        * UVX convention has shape (time, freq, baseline, pol)
        * UVX polarization convention is (XX, XY, YX, YY) i.e. (-5, -8, -6, -7)

    Args:
        uv (UVData): UV data object with uv.data_array to convert
        check (bool): Basic sanity checks on incoming data arrangement

    Returns:
        data (np.array): Numpy array of data after remapping to UVX convention
    """
    if check:
        # Pol in correct input order
        assert np.allclose((-5, -6, -7, -8), uv.polarization_array)
        # No missing baselines
        assert uv.Nbls == uv.Nants_data * (uv.Nants_data + 1) // 2

    # Create empty numpy array
    data = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Nbls, uv.Npols), dtype=uv.data_array.dtype)

    # Generate indexes
    idx_t  = np.repeat(np.arange(uv.Ntimes), uv.Nbls * uv.Nfreqs * uv.Npols)
    idx_f  = np.repeat(np.arange(uv.Nfreqs), uv.Nbls * uv.Ntimes * uv.Npols)
    idx_bl = np.repeat(np.tile(np.arange(uv.Nbls),     uv.Nfreqs * uv.Ntimes), uv.Npols)
    idx_p  = np.tile([0, 3, 1, 2],     uv.Nbls * uv.Nfreqs * uv.Ntimes)

    data[idx_t, idx_f, idx_bl, idx_p] = uv.data_array.flatten()
    return data

def pyuvdata_to_uvx(uv: UVData, check: bool=False) -> UVX:
    """Convert pyuvdata UVData object to UVX

    Notes:
        This is experimental, and will only work under certain
        circumstances:

            * Nspws = 1
            * Data are in lower-triangular order, no missing data.
            * Data not phase-tracking (if multiple timesteps)

    Args:
        uv (UVData): Pyuvdata input object

    Returns:
        uvx (UVX): Converted UVX object

    """
    eloc = EarthLocation(*uv.telescope_location, unit='m')

    # Create Antenna dataset
    antpos_ENU = uvutils.ENU_from_ECEF(uv.antenna_positions + uv.telescope_location,
                                       *uv.telescope_location_lat_lon_alt)
    df = pd.DataFrame(antpos_ENU, columns=('E', 'N', 'U'))
    df['flagged'] = False
    df['name']    = uv.antenna_names
    df = df[['name', 'E', 'N', 'U', 'flagged']]
    antennas = create_antenna_data_array(df, eloc)

    # Create frequency, time, and data
    if uv.Nspws != 1:
        logger.warning("Multiple SPWs not supported!")
    f = Quantity(uv.freq_array[0], unit='Hz')
    t = Time(uv.time_array[::uv.Nbls], format='jd', scale='utc', location=eloc)
    data_arr = convert_data_to_uvx_convention(uv, check=check)
    data  = create_visibility_array(data_arr, f, t, eloc, conj=False)

    # Create phase center
    pc_cat  = list(uv.phase_center_catalog.values())
    if len(pc_cat) > 1:
        logger.warning("Multiple phase centers detected. Failure likely.")
    pc_dict = list(uv.phase_center_catalog.values())[0]
    pc_sc = SkyCoord(pc_dict['cat_lon'], pc_dict['cat_lat'], frame=pc_dict['cat_frame'], unit=('rad', 'rad'))

    # Create UV object
    uvx = UVX(name=uv.telescope_name,
            antennas=antennas,
            context=create_empty_context_dict(),
            data=data,
            timestamps=t,
            origin=eloc,
            phase_center=pc_sc,
            provenance=create_empty_provenance_dict()
            )

    return uvx