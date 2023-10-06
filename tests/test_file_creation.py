from aavs_uv import get_hdf5_metadata, hdf5_to_pyuvdata, phase_to_sun
from sdp_uv import hdf5_to_sdp_vis 
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5
from astropy.time import Time
import os

# Files to open
yaml_raw = 'config/aavs2_uv_config.yaml'
fn_raw   = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'

# Load raw data and phase to sun
uv = hdf5_to_pyuvdata(fn_raw, yaml_raw)
t0 = Time(uv.time_array[0], format='jd')
uv = phase_to_sun(uv, t0)

# Write out files in different formats
uv.write_uvfits('test.uvfits')
uv.write_ms('test.ms')
uv.write_miriad('test.mir')
uv.write_uvh5('test.uvh5')

# Write out SDP data format too
vis = hdf5_to_sdp_vis(fn_raw, yaml_raw)
export_visibility_to_hdf5(vis, 'test.sdp_hdf5')
