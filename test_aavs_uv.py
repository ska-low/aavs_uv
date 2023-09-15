from aavs_uv import hdf5_to_pyuvdata
import glob, os
import pandas as pd

def get_aavs2_correlator_filelist(filepath: str) -> list:
    """ Return sorted filelist, 
    making sure two-digit channels are before three-digit channels """
    fl = glob.glob(os.path.join(filepath, 'correlation_burst_*.hdf5'))
    idx = [int(os.path.basename(f).split('_')[2]) for f in fl]
    df = pd.DataFrame({'filename': fl, 'idx': idx}).sort_values('idx')
    return df['filename'].values

filepath    = 'data/2023_08_23-13:53/'
yaml_config = 'config/aavs2_uv_config.yaml'
filelist = get_aavs2_correlator_filelist(filepath)
uv = hdf5_to_pyuvdata(filelist[0], yaml_config)
