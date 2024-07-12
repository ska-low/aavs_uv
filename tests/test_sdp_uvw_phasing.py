"""test_sdp_uvw_phasing: Test phasing routines for SDP Visibility."""
import os

import h5py
import numpy as np
from aa_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis
from aa_uv.utils import get_test_data
from astropy.constants import c
from ska_sdp_datamodels.visibility import (
    convert_hdf_to_visibility,
    create_visibility_from_ms,
    export_visibility_to_hdf5,
)

LIGHT_SPEED = c.value


def test_phasing():
    """Run phasing test, comparing to UVData phasing solutions."""
    try:
        os.system('rm -rf tests/test.ms')
    except: # noqa: E722
        pass

    try:
        # Read in test data, converting into a pyuvdata.UVData and sdp Visibility
        test_data_fn = get_test_data('aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5')
        uvdp = hdf5_to_pyuvdata(test_data_fn, telescope_name='aavs2', phase_to_t0=True)
        sdp = hdf5_to_sdp_vis(test_data_fn, telescope_name='aavs2')

        # Write out the UVData as a MS, and write out the SDP visibilities as a H5 file
        uvdp.write_ms('tests/test.ms')
        export_visibility_to_hdf5(sdp, 'tests/test.sdp')

        # Now, load back in the SDP and MS files
        with h5py.File('tests/test.sdp') as h:
            sdp_sdp = convert_hdf_to_visibility(h['Visibility0'])
        sdp_ms = create_visibility_from_ms('tests/test.ms')[0]

        # Check that data are complex conjugate of each other
        print("--- SDP ---")
        print(sdp.vis.values[0,1])
        print("--- MS -> SDP ---")
        print(sdp_ms.vis.values[0, 1])
        #print("---")
        #print(sdp_sdp.vis.values[0,1])

        assert np.allclose(sdp_ms.uvw, sdp_sdp.uvw, atol=0.5e-7)
        assert np.allclose(sdp.vis.values, sdp_sdp.vis.values)
        assert np.allclose(sdp_ms.vis.values, sdp_sdp.vis.values)

    finally:
        if os.path.exists('tests/test.ms'):
            os.system('rm -rf tests/test.ms')
        if os.path.exists('tests/test.sdp'):
            os.remove('tests/test.sdp')

if __name__ == "__main__":
    test_phasing()