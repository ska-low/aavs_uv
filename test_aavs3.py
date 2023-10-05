# Basic imports
import os, glob

from astropy.time import Time, TimeDelta
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5
from aavs_uv import get_hdf5_metadata, hdf5_to_pyuvdata, phase_to_sun
from sdp_uv import hdf5_to_sdp_vis

if __name__ == "__main__":
    yaml_raw = 'config/aavs3/uv_config.yaml'
    filelist = glob.glob('data/aavs3/correlation_burst_*.hdf5')
    
    for fn_raw in filelist:
        print(f'--- Processing {fn_raw} ---')
    
        # Common UV data format output
        uv = hdf5_to_pyuvdata(fn_raw, yaml_raw)
        bn = os.path.splitext(fn_raw)[0]
        uv.write_uvfits(bn + '.uvfits')
        uv.write_miriad(bn + '.miriad')
        uv.write_ms(bn + '.ms')
        uv.write_uvh5(bn + '.uvh5')
        
        # SDP HDF5 output
        sdp_vis = hdf5_to_sdp_vis(fn_raw, yaml_raw)
        export_visibility_to_hdf5(sdp_vis, bn + '.sdpvis')