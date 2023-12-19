import numpy as np
from pyuvdata import utils as uvutils

from aavs_uv.datamodel import UV

def calc_uvw(uv: UV) -> np.array:
    """ Calculate the UVW coordinates for a UV array
    
    Args:
        uv (UV): aavs_uv UV object

    Returns:
        uvw (np.array): Numpy array of UVW coordinates, in meters.

    Notes:
        Computes UVW coordinates relative to the phase center for the array. 
        Currently, the phase center is always a celestial (RA/DEC) source, so
        if set to zenith at t0 (e.g. zenith at JD xxx.xx), it will track that
        location.

        The code uses methods from pyuvdata.utils, and does the following:
            * Computes apparent RA and DEC values for phase center (uvutils.transform_icrs_to_app)
            * Computes corresponding position angle for frame (uvutils.transform_icrs_to_app)
            * Comptues UVW coordinates (uvutils.calc_uvw)
    """
    n_bl = len(uv.data.baseline)
    n_ts = len(uv.timestamps)
    
    # Get LST in radians
    lst_rad = (uv.data.time.lst / 12 * np.pi).values
    
    # Compute apparent RA and DECs
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
    
    # Now compute position angle 
    frame_pos_angle = uvutils.calc_frame_pos_angle(uv.timestamps.jd, 
                                 app_ra=app_ras, 
                                 app_dec=app_decs,
                                 ref_frame='icrs',
                                 telescope_loc=uv.origin)
    
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