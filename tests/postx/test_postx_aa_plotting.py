from aavs_uv.postx import ApertureArray
from aavs_uv.io import hdf5_to_uvx
import pylab as plt

def test_aa_plotter():
    uvx = hdf5_to_uvx('./test-data/aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5', telescope_name='aavs2')
    aa = ApertureArray(uvx)

    plt.figure("ANT")
    aa.plotting.plot_antennas()

    plt.figure("CORR")
    aa.plotting.plot_corr_matrix()

    #plt.figure("CORR 4x")
    aa.plotting.plot_corr_matrix_4pol()

if __name__ == "__main__":
    test_aa_plotter()