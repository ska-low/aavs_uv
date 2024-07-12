"""test_file_creation: test output data formats."""
import os

from aa_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, phase_to_sun
from aa_uv.utils import get_aa_config, get_test_data
from astropy.time import Time
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5


def test_file_creation():
    """Create a whole bunch of files from HDF5 data."""
    try:
        # Files to open
        yaml_raw = get_aa_config('aavs2')
        fn_raw   = get_test_data('aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5')

        # Load raw data and phase to sun
        uv = hdf5_to_pyuvdata(fn_raw, yaml_raw)
        t0 = Time(uv.time_array[0], format='jd')
        uv = phase_to_sun(uv, t0)

        # Write out files in different formats
        uv.write_uvfits('tests/test.uvfits')
        uv.write_ms('tests/test.ms')
        uv.write_miriad('tests/test.mir')
        uv.write_uvh5('tests/test.uvh5')

        # Write out SDP data format too
        vis = hdf5_to_sdp_vis(fn_raw, yaml_config=yaml_raw)
        export_visibility_to_hdf5(vis, 'tests/test.sdpuv5')
    finally:
        os.system('rm -rf tests/test.ms')
        os.system('rm -rf tests/test.mir')

        for fn in ('test.uvfits', 'test.uvh5', 'test.sdpuv5'):
            try:
                os.remove(f'tests/{fn}')
            except FileNotFoundError:
                pass

if __name__ == "__main__":
    test_file_creation()