from aavs_uv.converter import parse_args, run
from aavs_uv.utils import get_resource_path
import os

def test_converter():
    for fmt in('uvfits', 'miriad', 'mir', 'ms', 'uvh5', 'sdp'):
        try:
            # Delete temp file if it exists
            if os.path.exists(f"test.{fmt}"):
                if fmt in ('miriad', 'mir', 'ms'):
                    os.system(f"rm -rf test.{fmt}")
                else:
                    os.remove(f"test.{fmt}")

            # Create command-line args
            cmd = ["-n", "aavs2", 
                "-o", fmt, 
                "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
                f"test.{fmt}"]
            
            # Run the script
            args = parse_args(cmd)
            run(cmd)
        finally:
            # clean up temp files
            if os.path.exists(f"test.{fmt}"):
                if fmt in ('miriad', 'mir', 'ms'):
                    os.system(f"rm -rf test.{fmt}")
                else:
                    os.remove(f"test.{fmt}")

def test_custom_config():
    try:
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
        "-o", "sdp", 
        "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
        "test.sdp"]
        run(cmd)
    finally:
        if os.path.exists("test.sdp"):
            os.remove("test.sdp")

def test_errors():
        # Create command-line args
        cmd = ["-c", "carmen/santiago", 
            "-o", "svg", 
            "input.hdf5",
            "output.sdp"]
        # Run the script
        args = parse_args(cmd)
        run(cmd)

if __name__ == "__main__":
    test_custom_config()
    test_errors()
    test_converter()