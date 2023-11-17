from aavs_uv.converter import parse_args, run
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
            cmd = ["-c", "../config/aavs2/uv_config.yaml", 
                "-o", fmt, 
                "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
                f"test.{fmt}"]
            
            # Run the script
            args = parse_args(cmd)
            run(cmd)
        finally:
            # clean up temp files
            if fmt in ('miriad', 'mir', 'ms'):
                os.system(f"rm -rf test.{fmt}")
            else:
                os.remove(f"test.{fmt}")


if __name__ == "__main__":
    test_converter()