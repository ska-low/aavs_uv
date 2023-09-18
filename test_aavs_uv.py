from pyuvdata import UVData
from pyuvdata.parameter import UVParameter
from aavs_uv import hdf5_to_pyuvdata, phase_to_sun
import numpy as np
from astropy.time import Time
import glob, os
import pandas as pd

def compare_uv_datasets(uv_orig, uv_comp):
    for key, param in uv_orig.__dict__.items():
        if key not in ('_history', '_antenna_names', '_data_array', '_filename'):
            if isinstance(param, UVParameter):
                if isinstance(param.value, (np.int32, np.int64, str, bool, int)):
                    try:
                        assert param.value == uv_comp.__dict__[key].value
                    except AssertionError:
                        print(f" --- {key} --- \n Original: \n {param.value} \n Comparison: \n {uv_comp.__dict__[key].value}")
    
                elif isinstance(param.value, np.ndarray):
                    if type(param.value) != type(uv_comp.__dict__[key].value):
                        print(f"ERROR: Type mismatch:{type(param.value)} {type(uv_comp.__dict__[key].value)}")
                        uv_comp.__dict__[key].value = np.array(uv_comp.__dict__[key].value)
                    try:
                        assert np.allclose(param.value, uv_comp.__dict__[key].value)
                    except AssertionError:
                        print(f" --- {key} --- \n Original: \t {param.value[:10]} \n Comparison: \t {uv_comp.__dict__[key].value[:10]}")
                    except:
                        print(f"ERROR: {key}")
                        
                elif param.value is None:
                    print(f"Unset: {key}")
                    try:
                        assert uv_comp.__dict__[key].value is None
                    except AssertionError:
                        print(f" --- {key} --- \n Original: None \n Comparison: {uv_comp.__dict__[key].value}")                
                else:
                    print(f"UNTESTED: {key} {type(param.value)}")
    
        if key == '_data_array':
            d0 = np.array(param.value[:])
            d1 = np.array(uv_comp.__dict__[key].value[:])
            assert d0.shape == d1.shape
    
            #assert np.allclose(d0.real, d1.real)
            #assert np.allclose(d0.imag, d1.imag)


def get_aavs2_correlator_filelist(filepath: str) -> list:
    """ Return sorted filelist, 
    making sure two-digit channels are before three-digit channels """
    fl = glob.glob(os.path.join(filepath, 'correlation_burst_*.hdf5'))
    idx = [int(os.path.basename(f).split('_')[2]) for f in fl]
    df = pd.DataFrame({'filename': fl, 'idx': idx}).sort_values('idx')
    return df['filename'].values

def test0():
    filepath    = 'data/2023_08_23-13:53/'
    yaml_config = 'config/aavs2_uv_config.yaml'
    filelist = get_aavs2_correlator_filelist(filepath)
    uv = hdf5_to_pyuvdata(filelist[0], yaml_config)

def test():
    yaml_raw = 'config/aavs2_uv_config.yaml'
    fn_raw = 'test-data/correlation_burst_204_20230823_21356_0.hdf5'
    fn_uvf = 'test-data/chan_204_20230823T055556.uvfits'
    fn_mir = 'test-data/chan_204_20230823T055556.uv'

    uv_raw = hdf5_to_pyuvdata(fn_raw, yaml_raw)
    uv_uvf = UVData.from_file(fn_uvf, file_type='uvfits', run_check=False)
    uv_mir = UVData.from_file(fn_mir, file_type='miriad')

    t0 = Time(uv_raw.time_array[0], format='jd')
    uv_phs = phase_to_sun(uv_raw, t0)

    uv_orig = uv_phs
    uv_comp = uv_mir
    compare_uv_datasets(uv_orig, uv_comp)

if __name__ == "__main__":
    test()
