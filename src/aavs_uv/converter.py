import argparse
import sys
import os
import time
from loguru import logger

from aavs_uv import __version__
from aavs_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5

def parse_args(args):
    """ Parse command-line arguments """
    p = argparse.ArgumentParser(description="AAVS UV file conversion utility")
    p.add_argument("infile", help="Input filename")
    p.add_argument("-o", 
                   "--output_format", 
                   help="Output file format (uvfits, miriad, ms, uvh5, sdp)")
    p.add_argument("-c",
                   "--array_config",
                   help="Array configuration YAML file")
    p.add_argument("outfile", help="Output filename")
    args = p.parse_args(args)
    return args


def run(args=None):
    """ Command-line utility for file conversion """
    args = parse_args(args)

    logger.info(f"aavs_uv {__version__}")

    output_format = args.output_format.lower()
    t0 = time.time()

    if output_format in ('uvfits', 'miriad', 'mir', 'ms', 'measurementset', 'uvh5'):
        logger.info(f"Loading {args.infile}")
        uv = hdf5_to_pyuvdata(args.infile, args.array_config)
        tr = time.time()

        _writers = {
            'uvfits': uv.write_uvfits,
            'miriad': uv.write_miriad,
            'mir':    uv.write_miriad,
            'ms':     uv.write_ms,
            'measurementset': uv.write_ms,
            'uvh5':   uv.write_uvh5
        }

        writer = _writers[output_format]
        logger.info(f"Creating {args.output_format} file: {args.outfile}")
        writer(args.outfile)
        tw = time.time()
    
    elif output_format == 'sdp':
        logger.info(f"Loading {args.infile}")
        vis = hdf5_to_sdp_vis(args.infile, args.array_config)
        tr = time.time()
        logger.info(f"Creating {args.output_format} file: {args.outfile}")
        export_visibility_to_hdf5(vis, args.outfile)
        tw = time.time()

    logger.info(f"Done. Time taken: Read: {tr-t0:.2f} s Write: {tw-tr:.2f} s Total: {tw-t0:.2f} s")

if __name__ == "__main__":
    print(sys.argv[1:])
    args = parse_args(sys.argv[1:])
    run(args)
    

