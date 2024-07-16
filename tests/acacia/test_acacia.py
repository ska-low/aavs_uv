"""Test reading data from acacia."""

from ska_ost_low_uv.acacia import AcaciaStorage


def test_acacia():
    """Test read from acacia using ROS3 VFD."""
    acacia = AcaciaStorage()
    bucket = 'devel'
    fpath = 'test/correlation_burst_204_20210612_16699_0.uvx'
    h5 = acacia.get_h5(bucket, fpath, debug=True)
    print(h5.keys())

    uvx = acacia.read_uvx(bucket, fpath)
    print(uvx)


if __name__ == '__main__':
    test_acacia()
