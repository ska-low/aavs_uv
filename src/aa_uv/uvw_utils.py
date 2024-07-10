import numpy as np
from pyuvdata import utils as uvutils

from aa_uv.datamodel import UVX
from astropy.constants import c
LIGHT_SPEED = c.value

def calc_zenith_apparent_coords(uv: UVX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the apparent RA/DEC coordinates of the array zenith

    Args:
        uv (UVX): UVX data object

    Returns:
        app_ras (np.ndarray): Apparent RAs, in radians
        app_decs (np.ndarray): Apparent DECs, in radians
        frame_pos_angle (np.ndarray): Frame position angle, in radians

    Notes:
        Uses ERFA as astrometry library (pyuvdata default). The function
        erfa.atco13() (ICRS-observed, 2013) is called under pyuvdata's hood.
        From ERFA guide: 'Observed' RA,Dec is the position that would be
        seen by a perfect equatorial with its polar axis aligned to
        the Earth's axis of rotation.
    """
    app_ras, app_decs = uvutils.transform_icrs_to_app(
        uv.timestamps,
        uv.phase_center.ra.to('rad').value,
        uv.phase_center.dec.to('rad').value,
        telescope_loc=uv.origin,
        epoch=2000.0,
        pm_ra=None,
        pm_dec=None,
        vrad=None,
        dist=None,
        astrometry_library=None,
    )

    frame_pos_angle = uvutils.calc_frame_pos_angle(uv.timestamps.jd,
                                 app_ra=app_ras,
                                 app_dec=app_decs,
                                 ref_frame='icrs',
                                 telescope_loc=uv.origin)

    return app_ras, app_decs, frame_pos_angle

def calc_uvw(uv: UVX) -> np.ndarray:
    """Calculate the UVW coordinates for a UV array

    Args:
        uv (UVX): aa_uv UVX object

    Returns:
        uvw (np.ndarray): Numpy array of UVW coordinates, in meters.
                        Shape is (n_timestep, n_baseline, 3)

    Notes:
        Computes UVW coordinates relative to the phase center for the array.
        Currently, the phase center is always a celestial (RA/DEC) source, so
        if set to zenith at t0 (e.g. zenith at JD xxx.xx), it will track that
        location.

        The code uses methods from pyuvdata.utils, and does the following:
            * Computes apparent RA and DEC values for phase center (uvutils.transform_icrs_to_app)
            * Computes corresponding position angle for frame (uvutils.transform_icrs_to_app)
            * Computes UVW coordinates (uvutils.calc_uvw)

    """
    n_bl = len(uv.data.baseline)
    n_ts = len(uv.timestamps)

    # Get LST in radians
    lst_rad = (uv.data.time.lst / 12 * np.pi).values

    # Compute apparent RA and DECs, and frame position angle
    app_ras, app_decs, frame_pos_angle = calc_zenith_apparent_coords(uv)

    # And now we can compute UVWs
    uvw = uvutils.calc_uvw(
        app_ra=np.repeat(app_ras, n_bl),
        app_dec=np.repeat(app_decs, n_bl),
        lst_array=np.repeat(lst_rad, n_bl),
        use_ant_pos=True,
        frame_pa=np.repeat(frame_pos_angle, n_bl),
        antenna_positions=uv.antennas.ecef.values,
        antenna_numbers=uv.antennas.coords['antenna'].values,
        ant_1_array=np.tile(uv.data.ant1.values, n_ts),
        ant_2_array=np.tile(uv.data.ant2.values, n_ts),
        telescope_lat=uv.origin.to_geodetic().lat.to('rad').value,
        telescope_lon=uv.origin.to_geodetic().lon.to('rad').value,
    )

    return uvw.reshape((n_ts, n_bl, 3))

def calc_zenith_tracking_phase_corr(uv: UVX) -> np.ndarray:
    """Calculate phase corrections to phase to zenith at t0 and then track it.

    Args:
        uv (UV): aa_uv UV object

    Returns:
        phs_corr (np.ndarray): Phase corrections to track zenith_at_t0.
                             Has shape (n_timestep, n_baseline, n_freq, n_pol)

    Notes:
        Following pyuvdata's phasing memo, phased data sets have baseline-dependent
        phase corrections applied to the visibilities to make them add coherently
        toward a given RA/DEC at a given epoch.
        Primarily used for phasing SDP data to zenith. Uses pyuvdata to calculate w-term
        for time delays.
    """
    n_bl = len(uv.data.baseline)
    n_ts = len(uv.timestamps)
    n_pol = len(uv.data.polarization)

    # Get LST in radians
    lst_rad = (uv.data.time.lst / 12 * np.pi).values

    # Calculate tracked UVW (i.e. standard)
    tracking_uvw = calc_uvw(uv)

    # Calculate non-tracked UVW
    # This differs from tracked as it uses non-apparent position (i.e. J2000)
    # for the phase center and a frame position angle = 0
    non_tracking_uvw = uvutils.calc_uvw(
        app_ra=np.repeat(uv.phase_center.ra.to('rad').value, n_bl * n_ts),
        app_dec=np.repeat(uv.phase_center.dec.to('rad').value, n_bl * n_ts),
        lst_array=np.repeat(lst_rad, n_bl),
        use_ant_pos=True,
        frame_pa=np.repeat(0, n_bl * n_ts),
        antenna_positions=uv.antennas.ecef.values,
        antenna_numbers=uv.antennas.coords['antenna'].values,
        ant_1_array=np.tile(uv.data.ant1.values, n_ts),
        ant_2_array=np.tile(uv.data.ant2.values, n_ts),
        telescope_lat=uv.origin.to_geodetic().lat.to('rad').value,
        telescope_lon=uv.origin.to_geodetic().lon.to('rad').value,
    )
    non_tracking_uvw = non_tracking_uvw.reshape((n_ts, n_bl, 3))

    # Compute how much w component differs (i.e. time delay)
    Δw = (non_tracking_uvw[:, :, 2] - tracking_uvw[:, :, 2])
    Δt = Δw * (1 / LIGHT_SPEED)

    # now compute phase correction
    f  = uv.data.frequency.values
    pol_dummy_axis = np.ones(n_pol)
    phs_corr = np.exp(1j * 2 * np.pi * np.einsum('tb,f,p->tbfp', Δt, f, pol_dummy_axis))

    return phs_corr