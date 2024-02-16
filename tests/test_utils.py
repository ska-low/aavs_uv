from aavs_uv.utils import zipit
from aavs_uv.vis_utils import vis_arr_to_matrix
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

if __name__ == "__main__":
    test_vis_arr_to_matrix()
    test_zipit()