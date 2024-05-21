import numpy as np
import pylab as plt
from aa_uv.io import hdf5_to_uvx
from aa_uv.postx import ApertureArray

test_data =  {
    'aavs2': ('test-data/aavs2/correlation_burst_100_20211113_14447_0.hdf5',
              'test-data/aavs2/correlation_burst_204_20211113_14653_0.hdf5' ),
    'aavs3': ('test-data/aavs3/correlation_burst_100_20240107_19437_0.hdf5',
              'test-data/aavs3/correlation_burst_204_20240107_19437_0.hdf5' ),
    'eda2':  ('test-data/eda2/correlation_burst_100_20211211_14167_0.hdf5',
              'test-data/eda2/correlation_burst_204_20211211_14373_0.hdf5' ),
}
