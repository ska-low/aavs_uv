from aavs_uv.io import hdf5_to_uv, uv5_to_uv, uv_to_uv5

def test_roundtrip():
    fn = 'test-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5'
    uv = hdf5_to_uv(fn, telescope_name='aavs2')

    uv_to_uv5(uv, 'test.h5')

    uv2 = uv5_to_uv('test.h5')

if __name__ == "__main__":
    test_roundtrip()