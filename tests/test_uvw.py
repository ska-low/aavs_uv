from aavs_uv.io import hdf5_to_uvx, hdf5_to_pyuvdata
from aavs_uv.uvw import calc_uvw
import numpy as np
from pyuvdata import utils as uvutils

def test_uvw():
    fn = '../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    uv = hdf5_to_uvx(fn, telescope_name='aavs2')

    uvd = hdf5_to_pyuvdata(fn, telescope_name='aavs2')
    uv = hdf5_to_uvx(fn, telescope_name='aavs2')

    # Checks comparing internal calls within calc_uvw
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

    # Check apparent RA and DEC match
    assert np.allclose(uvd.phase_center_app_ra, np.repeat(app_ras, n_bl))
    assert np.allclose(uvd.phase_center_app_dec, np.repeat(app_decs, n_bl))

    # Check LST arrays match
    assert np.allclose(uvd.lst_array, np.repeat(lst_rad, n_bl))

    # Check frame position angles match
    assert np.allclose(uvd.phase_center_frame_pa, np.repeat(frame_pos_angle, n_bl))

    # Check antenna positions match
    assert np.allclose(uvd.antenna_positions, uv.antennas.ecef.values)
    assert np.allclose(uvd.ant_1_array, np.tile(uv.data.ant1.values, n_ts))
    assert np.allclose(uvd.ant_2_array, np.tile(uv.data.ant2.values, n_ts))
    assert np.allclose(uvd.antenna_numbers, uv.antennas.coords['antenna'].values)

    # Check telescope positions match
    assert np.isclose(uvd.telescope_location_lat_lon_alt[0], uv.origin.to_geodetic().lat.to('rad').value)
    assert np.isclose(uvd.telescope_location_lat_lon_alt[1], uv.origin.to_geodetic().lon.to('rad').value)

    # Finally, check UVW positions match
    assert np.allclose(uvd.uvw_array, uvw, atol=1e-7)

    # And now call and check the actual in-built calc_uvw method
    uvw_ = calc_uvw(uv)
    assert np.allclose(uvw.reshape((n_ts, n_bl, 3)), uvw_)

if __name__ == "__main__":
    test_uvw()