from aavs_uv.converter import parse_args, run
from aavs_uv.utils import get_resource_path
from aavs_uv.io import read_uvx
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

def test_phase_to_sun():
    try:
        cmd = [
        "-o", "uvfits", "-s", "-n", "aavs3",
        "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
        "test.uvfits"]
        run(cmd)
    finally:
        if os.path.exists("test.uvfits"):
            os.remove("test.uvfits")    

def test_errors():
        # Create command-line args
        cmd = ["-c", "carmen/santiago", 
            "-o", "svg", 
            "input.hdf5",
            "output.sdp"]
        # Run the script
        args = parse_args(cmd)
        run(cmd)

def test_batch():
    try:
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-b", 
               "-o", "uvx", 
               "../example-data/aavs2_2x500ms",
               "test-batch-data"]
        run(cmd)

        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-B", 
               "-o", "uvx", 
               "test-data/", # Note -B MEGABATCH test 
               "test-batch-data"]
        run(cmd)
    finally:
        if os.path.exists('test-batch-data'):
            os.system('rm -rf test-batch-data')

def test_batch_zip():
    try:
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-b", 
               "-z",
               "-o", "ms", 
               "../example-data/aavs2_2x500ms",
               "test-batch-data"]
        run(cmd)

        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-B", 
               "-z", 
               "-o", "miriad", 
               "test-data/", # Note -B MEGABATCH test 
               "test-batch-data2"]
        run(cmd)

    finally:
        if os.path.exists('test-batch-data.zip'):
            os.system('rm -rf test-batch-data.zip')
        if os.path.exists('test-batch-data2'):
            os.system('rm -rf test-batch-data2')
        pass

def test_batch_multi():
    try:
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-b", 
               "-z",
               "-o", "ms,miriad,uvx", 
               "../example-data/aavs2_2x500ms",
               "test-batch-data"]
        run(cmd)

    finally:
        pass
        #if os.path.exists('test-batch-data.zip'):
        #    os.system('rm -rf test-batch-data.zip')


def test_context():
    try:
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-i", "test-data/context.yml", 
               "-o", "uvx", 
               "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
               "test.uvx5"]
        run(cmd)
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-i", "test-data/context.yml", 
               "-o", "sdp", 
               "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
               "test.sdp"]
        run(cmd)
        uv = read_uvx("test.uvx5")
        print(uv.context)
        assert(uv.context['intent'] == "Test routine for AAVS_UV package")
    finally:
        if os.path.exists("test.sdp"):
            os.remove("test.sdp")
        if os.path.exists("test.uvx5"):
            os.remove("test.uvx5")

def test_parallel():
    try:
        # One worker
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-i", "test-data/context.yml", 
               "-o", "sdp", 
               "-w", "2",
               "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
               "test.sdp"]
        run(cmd)
        # Change number of workers
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-i", "test-data/context.yml", 
               "-o", "sdp", 
               "-w", "8",
               "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
               "test.sdp"]
        run(cmd)
        # Use dask instead
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-i", "test-data/context.yml", 
               "-o", "sdp", 
               "-w", "8",
               "-p", "dask",
               "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
               "test.sdp"]
        run(cmd)
        # Verbose
        cmd = ["-c", get_resource_path('config/aavs3/uv_config.yaml'), 
               "-i", "test-data/context.yml", 
               "-o", "sdp", 
               "-w", "1",
               "-v",
               "../example-data/aavs2_2x500ms/correlation_burst_204_20230927_35116_0.hdf5",
               "test.sdp"]
        run(cmd)

    finally:
        if os.path.exists("test.sdp"):
            os.remove("test.sdp")
        if os.path.exists("test.uvx5"):
            os.remove("test.uvx5")

if __name__ == "__main__":
    test_batch_multi()
    #test_batch()
    #test_context()
    #test_phase_to_sun()
    #test_custom_config()
    #test_errors()
    #test_converter()
    #test_batch_zip()