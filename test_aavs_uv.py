from pyuvdata import UVData
from pyuvdata.parameter import UVParameter
from aavs_uv import hdf5_to_pyuvdata, phase_to_sun
import numpy as np
from astropy.time import Time
import glob, os
import pandas as pd
from colored import Fore, Style

def compare_uv_datasets(uv_orig, uv_comp):

    tolerances = {
        '_phase_center_app_dec': 0.001,
        '_phase_center_frame_pa': 0.001,
        '_uvw_array': 0.1,
        '_antenna_positions': 0.1
    }

    for key, param in uv_orig.__dict__.items():
        if key not in ('_history', '_antenna_names', '_data_array', '_filename'):
            if isinstance(param, UVParameter):
                param_comp = uv_comp.__dict__[key]

                if isinstance(param.value, (np.int32, np.int64, str, bool, int)):
                    try:
                        assert param.value == param_comp.value
                    except AssertionError:
                        print(f" --- {key} --- \n Original: \n {param.value} \n Comparison: \n {param_comp.value}")
                
                elif isinstance(param.value, (float, np.float32, np.float64)):
                    try:
                        assert np.isclose(param.value, param_comp.value)
                    except AssertionError:
                        print(f" --- {key} --- \n Original: \n {param.value} \n Comparison: \n {param_comp.value}")

                elif isinstance(param.value, np.ndarray):
                    if type(param.value) != type(param_comp.value):
                        print(f"ERROR: Type mismatch:{type(param.value)} {type(param_comp.value)}")
                        param_comp.value = np.array(param_comp.value)
                    try:
                        if key in tolerances.keys():
                            tol = tolerances[key]
                        else:
                            tol = 0.0000001
                        assert np.allclose(param.value, param_comp.value, atol=tol)
                    except AssertionError:
                        print(f" --- {key} --- \n Original: \t {param.value[:4]} \n Comparison: \t {param_comp.value[:4]}")
                    except:
                        print(f"{Fore.red} ERROR: {key} {Style.reset}")

                elif isinstance(param.value, (list, tuple)):
                    if type(param.value) != type(param_comp.value):
                        print(f"{Fore.red} ERROR: Type mismatch:{type(param.value)} {type(param_comp.value)} {Style.reset}")
                        param_comp.value = np.array(param_comp.value)
                    try:
                        assert np.allclose(param.value, param_comp.value)
                    except AssertionError:
                        print(f" --- {key} --- \n Original: \t {param.value[:4]} \n Comparison: \t {param_comp.value[:4]}")
                    except:
                        print(f"{Fore.red} ERROR: {key} {Style.reset}")

                elif isinstance(param.value, dict):
                    not_exact_match = False
                    try:
                        assert isinstance(param_comp.value, dict)
                    except AssertionError:
                        print(f"{Fore.red} ERROR: type mismatch {type(param.value)} {type(param_comp.value)} {Style.reset}")
                        not_exact_match = True

                    for dk, dv in param.value.items():
                        try:
                            assert dk in param_comp.value.keys()
                        except AssertionError:
                            not_exact_match = True
                            print(f"{Fore.red} ERROR: {key} dict: subkey missing: {dk} {Style.reset} ")
                    if not_exact_match:
                        print(f" --- {key} --- \n Original: \t {param.value} \n Comparison: \t {param_comp.value}")

                elif param.value is None:
                    
                    try:
                        assert param_comp.value is None
                        print(f"{Fore.cornsilk_1} Unset in both: {key} {Style.reset}")
                    except AssertionError:
                        print(f"{Fore.orange_1} Unset in orig: {key} {Style.reset}")
                        print(f" --- {key} --- \n Original: None \n Comparison: {param_comp.value}")                
                else:
                    print(f"{Fore.orange_1} UNTESTED: {key} {type(param.value)} {Style.reset}")
    
        if key == '_data_array':
            param_comp = uv_comp.__dict__[key]
            d0 = np.array(param.value[:])
            d1 = np.array(param_comp.value[:])
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



def _setup_test(test_name=None, load_comp=False):

    yaml_raw = 'config/aavs2_uv_config.yaml'
    fn_raw = 'test-data/correlation_burst_204_20230823_21356_0.hdf5'    
    fn_uvf = 'test-data/chan_204_20230823T055556.uvfits'
    fn_mir = 'test-data/chan_204_20230823T055556.uv'

    def _load_and_phase_hdf5():
        uv_raw = hdf5_to_pyuvdata(fn_raw, yaml_raw)
        t0 = Time(uv_raw.time_array[0], format='jd')
        uv_phs = phase_to_sun(uv_raw, t0)
        uv_phs.check()
        return uv_phs
    
    if test_name is not None:
        print(f"{Fore.green} #### Test: {test_name} #### {Style.reset}")

    uv_phs = _load_and_phase_hdf5()

    if load_comp:
        uv_uvf = UVData.from_file(fn_uvf, file_type='uvfits', run_check=False)
        uv_mir = UVData.from_file(fn_mir, file_type='miriad')
        return uv_phs, uv_uvf, uv_mir

    else:
        return uv_phs

def test_compare():
    uv_phs, uv_uvf, uv_mir = _setup_test('Compare to MIRIAD', load_comp=True)
    compare_uv_datasets(uv_phs, uv_mir)

def test_write():
    uv_phs = _setup_test('Write to UVFITS')
    fn_out = 'pyuv_chan_204_20230823T055556.uvfits'
    uv_phs.write_uvfits(fn_out, fix_autos=True)

    uv_gen = UVData.from_file(fn_out)

    compare_uv_datasets(uv_phs, uv_gen)

if __name__ == "__main__":
    #test_compare()
    test_write()