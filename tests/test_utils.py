from aavs_uv.utils import vis_arr_to_matrix, zipit
import numpy as np
import pytest
import os, shutil

def test_zipit():
     def create_dummy_dir():
          os.mkdir('test-zip')
          os.system('touch test-zip/hello.txt')
          os.system('touch test-zip/hi.txt')
          
     create_dummy_dir()
     zipit('test-zip')
     assert os.path.exists('test-zip.zip')
     shutil.rmtree('test-zip')

     create_dummy_dir()
     zipit('test-zip', rm_dir=True)
     assert os.path.exists('test-zip.zip')
     assert ~os.path.exists('test-zip')

     os.remove('test-zip.zip')


def test_vis_arr_to_matrix():
    """ Test vis_array_to_matrix """
    # Upper
    d = np.arange(32896)
    V0 = vis_arr_to_matrix(d, 256)
    V  = vis_arr_to_matrix(d, 256, 'upper')
    assert np.allclose(V, V0)

    # Roundtrip
    ix, iy = np.triu_indices(256)
    assert np.allclose(V[ix, iy], d)   

    # Lower
    d = np.arange(32896)
    V = vis_arr_to_matrix(d, 256, 'lower')
    # Roundtrip: lower
    ix, iy = np.tril_indices(256)
    assert np.allclose(V[ix, iy], d)   

    # Use existing array
    V = np.zeros((256, 256))
    V = vis_arr_to_matrix(d, 256, 'lower', V=V)

    # Error - wrong N_ant
    with pytest.raises(RuntimeError):
         V = vis_arr_to_matrix(d, 99, 'lower')

    # Error - not upper/lower
    with pytest.raises(RuntimeError):
         V = vis_arr_to_matrix(d, 256, 'chicken')
        
    # Error - wrong vis shape
    with pytest.raises(RuntimeError):
         V = np.zeros((99, 99))
         V = vis_arr_to_matrix(d, 256, 'upper', V=V)   

if __name__ == "__main__":
    test_vis_arr_to_matrix()
    test_zipit()