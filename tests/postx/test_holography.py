"""test_holography: Run holographic calibration tests."""
from aa_uv.io import hdf5_to_uvx
from aa_uv.postx import ApertureArray
from aa_uv.utils import get_test_data

FN_RAW   = get_test_data('aavs2_1x1000ms/correlation_burst_204_20230823_21356_0.hdf5')


def test_holography_selfholo():
    """Test self-holography function."""
    uvx = hdf5_to_uvx(FN_RAW, telescope_name='aavs2')
    aa = ApertureArray(uvx)

    aa.calibration.holography.set_cal_src(aa.coords.get_sun())
    holo_dict = aa.calibration.holography.run_selfholo()
    print(holo_dict)

if __name__ == "__main__":
    test_holography_selfholo()
