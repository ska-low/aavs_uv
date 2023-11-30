import argparse
import sys
import os
import time
from loguru import logger

from astropy.time import Time, TimeDelta
from aavs_uv import __version__
from aavs_uv.utils import get_config_path
from aavs_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, hdf5_to_uv, phase_to_sun
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5

def parse_args(args):
    """ Parse command-line arguments """
    p = argparse.ArgumentParser(description="AAVS UV file conversion utility")
    p.add_argument("infile", help="Input filename")
    p.add_argument("outfile", help="Output filename")
    p.add_argument("-o", 
                   "--output_format", 
                   help="Output file format (uvfits, miriad, ms, uvh5, sdp)",
                   required=True)
    p.add_argument("-c",
                   "--array_config",
                   help="Array configuration YAML file. If supplied, will override aavs_uv internal array configs.",
                   required=False)
    p.add_argument("-n",
                   "--telescope_name",
                   help="Telescope name, e.g. 'aavs3'. If supplied, will attempt to use aavs_uv internal array config.",
                   required=False)
    p.add_argument("-s", 
                   "--phase-to-sun",
                   help="Re-phase to point toward Sun (the sun must be visible!). If flag not set, data will be phased toward zenith.",
                   required=False,
                   action="store_true",
                   default=False)
    p.add_argument("-j", 
                   "--no_conj",
                   help="Do not conjugate visibility data (note AAVS2 and AAVS3 require conjugation)",
                   required=False,
                   action="store_true",
                   default=False)
    
    args = p.parse_args(args)
    return args


def run(args=None):
    """ Command-line utility for file conversion """
    args = parse_args(args)

    logger.info(f"aavs_uv {__version__}")

    if args.telescope_name:
        logger.info(f"Telescope name: {args.telescope_name}")
        array_config = get_config_path(args.telescope_name)
    else:
        array_config = args.array_config
    

    output_format = args.output_format.lower()

    # Check input file exists
    config_error_found = False
    if not os.path.exists(args.infile):
        logger.error(f"Cannot find input file: {args.infile}")
        config_error_found = True
           
    # Check array config file exists
    if not os.path.exists(array_config):
        logger.error(f"Cannot find array config: {array_config}")
        config_error_found = True
    
    # Check output format
    if output_format not in ('uvfits', 'miriad', 'mir', 'ms', 'uvh5', 'sdp'):
        logger.error(f"Output format not valid: {output_format}")
        config_error_found = True       
    
    if config_error_found:
        logger.error("Errors found. Please check arguments.")
        return
    
    logger.info(f"Input path:    {args.infile}")
    logger.info(f"Array config:  {array_config}")
    logger.info(f"Output path:   {args.outfile}")
    logger.info(f"Output format: {output_format} \n")

    # Load file and read basic metadata
    vis = hdf5_to_uv(args.infile, array_config)
    logger.info(f"Data shape:     {vis.data.shape}")
    logger.info(f"Data dims:      {vis.data.dims}")
    logger.info(f"UTC start:      {vis.timestamps[0].iso}")
    logger.info(f"MJD start:      {vis.timestamps[0].mjd}")
    logger.info(f"LST start:      {vis.data.time.data[0][1]:.5f}")
    logger.info(f"Frequency 0:    {vis.data.frequency.data[0]} {vis.data.frequency.attrs['unit']}")
    logger.info(f"Polarization:   {vis.data.polarization.data}\n")

    # begin timing
    t0 = time.time()

    if output_format in ('uvfits', 'miriad', 'mir', 'ms', 'uvh5'):
        logger.info(f"Loading {args.infile}")
        conj = False if args.no_conj else True
        uv = hdf5_to_pyuvdata(args.infile, array_config, conj=conj)

        if args.phase_to_sun:
            logger.info(f"Phasing to sun")
            ts0 = Time(uv.time_array[0], format='jd') + TimeDelta(uv.integration_time[0]/2, format='sec')
            uv = phase_to_sun(uv, ts0)

        tr = time.time()

        _writers = {
            'uvfits': uv.write_uvfits,
            'miriad': uv.write_miriad,
            'mir':    uv.write_miriad,
            'ms':     uv.write_ms,
            'uvh5':   uv.write_uvh5
        }

        writer = _writers[output_format]
        logger.info(f"Creating {args.output_format} file: {args.outfile}")
        writer(args.outfile)
        tw = time.time()
    
    elif output_format == 'sdp':
        logger.info(f"Loading {args.infile}")
        vis = hdf5_to_sdp_vis(args.infile, array_config)
        tr = time.time()
        logger.info(f"Creating {args.output_format} file: {args.outfile}")
        export_visibility_to_hdf5(vis, args.outfile)
        tw = time.time()

    logger.info(f"Done. Time taken: Read: {tr-t0:.2f} s Write: {tw-tr:.2f} s Total: {tw-t0:.2f} s")

if __name__ == "__main__":
    print(sys.argv[1:])
    args = parse_args(sys.argv[1:])
    run(args)
    

