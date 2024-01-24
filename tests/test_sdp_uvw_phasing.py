from aavs_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, hdf5_to_uvx
import numpy as np
import h5py
from ska_sdp_datamodels.visibility import Visibility, create_visibility, export_visibility_to_hdf5, create_visibility_from_ms, convert_hdf_to_visibility
from pyuvdata import utils as uvutils
from astropy.constants import c
LIGHT_SPEED = c.value

import os

def test_phasing():

    try:
        os.system('rm -rf test.ms')
    except:
        pass

    try:   
        # Read in test data, converting into a pyuvdata.UVData and sdp Visibility 
        test_data_fn = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
        uvd  = hdf5_to_pyuvdata(test_data_fn, telescope_name='aavs2', phase_to_t0=False)
        uvdp = hdf5_to_pyuvdata(test_data_fn, telescope_name='aavs2', phase_to_t0=True)
        sdp = hdf5_to_sdp_vis(test_data_fn, telescope_name='aavs2')
        uv  = hdf5_to_uvx(test_data_fn, telescope_name='aavs2')

        # Write out the UVData as a MS, and write out the SDP visibilities as a H5 file
        uvdp.write_ms('test.ms')
        export_visibility_to_hdf5(sdp, 'test.sdp')

        # Now, load back in the SDP and MS files
        with h5py.File('test.sdp') as h:
            sdp_sdp = convert_hdf_to_visibility(h['Visibility0'])
        sdp_ms = create_visibility_from_ms('test.ms')[0]

        # WAR - Fix YY and XY swap 
        vis_sdp_corr = np.zeros_like(sdp_ms.vis.values)
        vis_sdp_corr[..., 0] = sdp_ms.vis[..., 0] # XX
        vis_sdp_corr[..., 1] = sdp_ms.vis[..., 2] # XY
        vis_sdp_corr[..., 2] = sdp_ms.vis[..., 3] # YX
        vis_sdp_corr[..., 3] = sdp_ms.vis[..., 1] # YY

        sdp_ms.vis.values = vis_sdp_corr

        # Check that data are complex conjugate of each other
        print(sdp_ms.vis.values[0])
        print(sdp_sdp.vis.values[0])
        assert np.allclose(sdp_ms.vis.values, sdp_sdp.vis.values)
        assert np.allclose(sdp_ms.uvw, sdp_sdp.uvw, atol=0.5e-7)
    finally:
        if os.path.exists('test.ms'):
            os.system('rm -rf test.ms')
        if os.path.exists('test.sdp'):
            os.remove('test.sdp')

if __name__ == "__main__":
    test_phasing()