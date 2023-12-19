import argparse
import sys
import os
import time
import glob
from loguru import logger
from tqdm import tqdm

from astropy.time import Time, TimeDelta
from aavs_uv import __version__
from aavs_uv.utils import get_config_path
from aavs_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, hdf5_to_uv, phase_to_sun, uv_to_uv5
from aavs_uv.io.yaml import load_yaml
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

def parse_args(args):
    """ Parse command-line arguments """
    p = argparse.ArgumentParser(description="AAVS UV file conversion utility")
    p.add_argument("infile", help="Input filename")
    p.add_argument("outfile", help="Output filename")
    p.add_argument("-o", 
                   "--output_format", 
                   help="Output file format (uvx, uvfits, miriad, ms, uvh5, sdp)",
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
    p.add_argument("-b",
                   "--batch",
                   help="Batch mode. Input and output are treated as directories, and all subfiles are converted.",
                   required=False,
                   action="store_true",
                   default=False
                   )
    p.add_argument("-B",
                   "--megabatch",
                   help="MEGA batch mode. Runs on subdirectories too, e.g. eb-aavs3/2023_12_12/*.hdf5.",
                   required=False,
                   action="store_true",
                   default=False
                   )
    p.add_argument("-x",
                   "--file_ext",
                   help="File extension to search for in batch mode ",
                   required=False,
                   default="hdf5")
    p.add_argument("-i", 
                   "--context_yaml",
                   help="Path to observation context YAML (for SDP / UVX formats)",
                   required=False,
                   default=None)
    
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
    
    conj = False if args.no_conj else True
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
    if output_format not in ('uvfits', 'miriad', 'mir', 'ms', 'uvh5', 'sdp', 'uvx'):
        logger.error(f"Output format not valid: {output_format}")
        config_error_found = True       
    
    if config_error_found:
        logger.error("Errors found. Please check arguments.")
        return
    
    logger.info(f"Input path:       {args.infile}")
    logger.info(f"Array config:     {array_config}")
    logger.info(f"Output path:      {args.outfile}")
    logger.info(f"Output format:    {output_format} \n")
    logger.info(f"Conjugating data: {conj} \n")

    # Check if context yaml was included
    context = load_yaml(args.context_yaml) if args.context_yaml is not None else None

    # Setup filelist, globbing for files if in batch mode
    if args.batch or args.megabatch:
        ext_lut = {
            'ms': '.ms', 
            'uvfits': '.uvfits',
            'miriad': '.uv',
            'mir': '.uv',
            'sdp': '.sdph5',
            'uvx': '.uvx5',
            'uvh5': '.uvh5'
        }

        if not os.path.exists(args.outfile):
            logger.info(f"Creating directory {args.outfile}")
            os.mkdir(args.outfile)

        if args.batch:
            filelist = sorted(glob.glob(os.path.join(args.infile, f'*.{args.file_ext}')))
        else:
            filelist = sorted(glob.glob(os.path.join(args.infile, f'*/*.{args.file_ext}'), recursive=True))
            
        filelist_out = []
        for fn in filelist:
            bn = os.path.basename(fn)
            bn_out = os.path.splitext(bn)[0] + ext_lut[output_format]
            if args.megabatch:
                subdir = os.path.join(args.outfile, os.path.basename(os.path.dirname(fn)))
                filelist_out.append(os.path.join(args.outfile, subdir, bn_out))
            else:
                filelist_out.append(os.path.join(args.outfile, bn_out))
    else:
        filelist     = [args.infile]
        filelist_out = [args.outfile]

    ######
    # Start main conversion loop
    for fn_in, fn_out in tqdm(zip(filelist, filelist_out)):
        
        # Create subdirectories as needed
        if args.batch or args.megabatch:
            subdir = os.path.dirname(fn_in)
            if not os.path.exists(subdir):
                logger.info(f"Creating sub-directory {subdir}")
                os.mkdir(subdir)

        # Load file and read basic metadata
        vis = hdf5_to_uv(fn_in, array_config, conj=False)  # Conj=False flag so data is not read into memory
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
            logger.info(f"Loading {fn_in}")
            
            uv = hdf5_to_pyuvdata(fn_in, array_config, conj=conj)

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
                'uvh5':   uv.write_uvh5,
            }

            writer = _writers[output_format]
            logger.info(f"Creating {args.output_format} file: {fn_out}")
            writer(fn_out)
            tw = time.time()
        
        elif output_format == 'sdp':
            logger.info(f"Loading {fn_in}")
            if context is not None:
                vis = hdf5_to_sdp_vis(fn_in, array_config, scan_intent=context['intent'], execblock_id=context['execution_block'])
            else:
                vis = hdf5_to_sdp_vis(fn_in, array_config)
            tr = time.time()
            logger.info(f"Creating {args.output_format} file: {fn_out}")
            export_visibility_to_hdf5(vis, fn_out)
            tw = time.time()
        
        elif output_format == 'uvx':
            logger.info(f"Loading {fn_in}")
            vis = hdf5_to_uv(fn_in, array_config, conj=conj, context=context)
            tr = time.time()
            logger.info(f"Creating {args.output_format} file: {fn_out}")
            uv_to_uv5(vis, fn_out)
            tw = time.time()

        logger.info(f"Done. Time taken: Read: {tr-t0:.2f} s Write: {tw-tr:.2f} s Total: {tw-t0:.2f} s")

if __name__ == "__main__":
    print(sys.argv[1:])
    args = parse_args(sys.argv[1:])
    run(args)
    

