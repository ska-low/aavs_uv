"""Test aa_simulation plotting routines."""

import pylab as plt
import pytest
from ska_ost_low_uv.io import hdf5_to_uvx
from ska_ost_low_uv.postx import ApertureArray
from ska_ost_low_uv.utils import get_test_data

uvx = hdf5_to_uvx(
    get_test_data('aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5'),
    telescope_name='aavs2',
)
aa = ApertureArray(uvx)


@pytest.mark.mpl_image_compare
def test_orthview():
    """Test plotting routines."""
    fig = plt.figure('orth')
    aa.simulation.orthview_gsm()
    return fig


@pytest.mark.mpl_image_compare
def test_mollview():
    """Test plotting routines."""
    fig = plt.figure('orth')
    aa.simulation.mollview_gsm()
    return fig


if __name__ == '__main__':
    test_orthview()
    test_mollview()
