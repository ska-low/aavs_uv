# Basic imports
import os, glob

from astropy.time import Time, TimeDelta
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5
from aavs_uv.io import get_hdf5_metadata, hdf5_to_pyuvdata, phase_to_sun
from aavs_uv.io import hdf5_to_sdp_vis

def tidyup(bn):
    for ext in ('miriad', 'ms'):
        fn_out = bn + '.' + ext
        if os.path.exists(fn_out):
            os.system(f'rm -rf {fn_out}')
    for ext in ('uvfits', 'uvh5', 'sdpuv5'):
        fn_out = bn + '.' + ext
        if os.path.exists(fn_out):
            os.system(f'rm {fn_out}')

def _test_aavs3_dataset(fn_raw, yaml_raw):
        uv = hdf5_to_pyuvdata(fn_raw, yaml_raw)
        bn = os.path.splitext(fn_raw)[0]
        tidyup(bn)

        uv.write_uvfits(bn + '.uvfits')
        uv.write_miriad(bn + '.miriad')
        uv.write_ms(bn + '.ms')
        uv.write_uvh5(bn + '.uvh5')
        
        # SDP HDF5 output
        sdp_vis = hdf5_to_sdp_vis(fn_raw, yaml_raw)
        export_visibility_to_hdf5(sdp_vis, bn + '.sdpvis')

if __name__ == "__main__":
    yaml_raw = '../example-config/aavs3/uv_config.yaml'
    filelist = glob.glob('../data/aavs3/correlation_burst_*.hdf5')
    
    for fn_raw in filelist:
        print(f'--- Processing {fn_raw} ---')
        _test_aavs3_dataset(fn_raw, yaml_raw)

        
        

