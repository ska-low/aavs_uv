from aa_uv.io.from_pyuvdata import convert_data_to_uvx_convention, pyuvdata_to_uvx
from pyuvdata import UVData


def test_pyuvdata_to_uvx():
    fn='./test-data/aavs3/aavs3_chan204_20231115.uvfits'
    uv = UVData().from_file(fn)

    data_arr = convert_data_to_uvx_convention(uv, check=True)
    data_arr = convert_data_to_uvx_convention(uv, check=False)

    uvx = pyuvdata_to_uvx(uv)

    print(uvx)

if __name__ == "__main__":
    test_pyuvdata_to_uvx()