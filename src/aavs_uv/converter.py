import argparse
import sys
import os
import time
import glob

import warnings
from loguru import logger
import pprint

from tqdm import tqdm
import dask
from dask.distributed import LocalCluster
from dask.diagnostics import ProgressBar, ResourceProfiler, Profiler
from dask.bag import from_sequence

from astropy.time import Time, TimeDelta
from aavs_uv import __version__
from aavs_uv.utils import get_config_path
from aavs_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis, hdf5_to_uvx, phase_to_sun, write_uvx
from aavs_uv.io.yaml import load_yaml
from ska_sdp_datamodels.visibility import export_visibility_to_hdf5

def reset_logger():
    """ Reset loguru logger and setup output format """
    logger.remove()
    logger.add(sys.stderr, format="<g>{time:HH:mm:ss.S}</g> | <w><b>{level}</b></w> | {message}", colorize=True)

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
    p.add_argument("-N",
                   "--num-workers",
                   help="Number of parallel processors (i.e. number of files to read in parallel).",
                   required=False,
                   default=1,
                   type=int
                   )
    p.add_argument("-v",
                   "--verbose",
                   help="Run with verbose output.",
                   action="store_true",
                   default=False
                   )
    p.add_argument("-P",
                   "--profile",
                   help="Run Dask resource profiler.",
                   action="store_true",
                   default=False
                   )
    
    args = p.parse_args(args)
    return args

def convert_single_file(args, fn_in, fn_out, array_config, output_format, conj, context):
        
        # Create subdirectories as needed
        if args.batch or args.megabatch:
            subdir = os.path.dirname(fn_out)
            if not os.path.exists(subdir):
                logger.info(f"Creating sub-directory {subdir}")
                os.mkdir(subdir)

        # Load file and read basic metadata
        vis = hdf5_to_uvx(fn_in, array_config, conj=False)  # Conj=False flag so data is not read into memory

        # Print basic info to screen (skip if in batch mode)
        if not args.batch and not args.megabatch:
            logger.info(f"Loading {fn_in}")
            logger.info(f"Data shape:     {vis.data.shape}")
            logger.info(f"Data dims:      {vis.data.dims}")
            logger.info(f"UTC start:      {vis.timestamps[0].iso}")
            logger.info(f"MJD start:      {vis.timestamps[0].mjd}")
            logger.info(f"LST start:      {vis.data.time.data[0][1]:.5f}")
            logger.info(f"Frequency 0:    {vis.data.frequency.data[0]} {vis.data.frequency.attrs['unit']}")
            logger.info(f"Polarization:   {vis.data.polarization.data}\n")

        if output_format in ('uvfits', 'miriad', 'mir', 'ms', 'uvh5'):
           
            tr0 = time.time()
            uv = hdf5_to_pyuvdata(fn_in, array_config, conj=conj)

            if args.phase_to_sun:
                logger.info(f"Phasing to sun")
                ts0 = Time(uv.time_array[0], format='jd') + TimeDelta(uv.integration_time[0]/2, format='sec')
                uv = phase_to_sun(uv, ts0)
            tr = time.time() - tr0

            tw0 = time.time()
            _writers = {
                'uvfits': uv.write_uvfits,
                'miriad': uv.write_miriad,
                'mir':    uv.write_miriad,
                'ms':     uv.write_ms,
                'uvh5':   uv.write_uvh5,
            }

            writer = _writers[output_format]
            logger.info(f"Creating {args.output_format} file: {fn_out}")
            if os.path.exists(fn_out):
                logger.warning(f"File exists, skipping: {fn_out}")
            else:
                writer(fn_out)
            tw = time.time() - tw0
            del uv
        
        elif output_format == 'sdp':
            tr0 = time.time()
            if context is not None:
                vis = hdf5_to_sdp_vis(fn_in, array_config, scan_intent=context['intent'], execblock_id=context['execution_block'])
            else:
                vis = hdf5_to_sdp_vis(fn_in, array_config)
            tr = time.time() - tr0

            tw0 = time.time()
            logger.info(f"Creating {args.output_format} file: {fn_out}")
            export_visibility_to_hdf5(vis, fn_out)
            tw = time.time() - tw0
            del vis
        
        elif output_format == 'uvx':
            tr0 = time.time()
            vis = hdf5_to_uvx(fn_in, array_config, conj=conj, context=context)
            tr = time.time() - tr0
            tw0 = time.time()
            logger.info(f"Creating {args.output_format} file: {fn_out}")
            write_uvx(vis, fn_out)
            tw = time.time() - tw0    
            del vis

        return (fn_in, tr, tw)

@dask.delayed
def convert_single_file_dask(fns, args, array_config, output_format, conj, context):
    from loguru import logger  # Import needed for dask
    logger.remove()
    
    fn_in, fn_out = fns

    if args.verbose:
        worker = dask.distributed.get_worker()
        worker_id = f"Worker {worker.name}"
        logger.add(sys.stderr, format="<m><b>" + worker_id + "</b></m> | <g>{time:HH:mm:ss.S}</g> | <w><b>{level}</b></w> | {message}", 
                   colorize=True)

    return convert_single_file(args, fn_in, fn_out, array_config, output_format, conj, context)

def run(args=None):
    """ Command-line utility for file conversion """
    args = parse_args(args)
    config_error_found = False

    # Reset logger
    reset_logger()
    logger.info(f"aavs_uv {__version__}")
    
    # Load array configuration
    array_config = args.array_config

    if args.telescope_name:
        logger.info(f"Telescope name: {args.telescope_name}")
        array_config = get_config_path(args.telescope_name)
   
    if array_config is None:
        logger.error(f"No telescope name or array config file passed. Please re-run with -n or -c flag set")
        config_error_found = True
    else:
        if not os.path.exists(array_config):
            logger.error(f"Cannot find array config: {array_config}")
            config_error_found = True

    conj = False if args.no_conj else True
    output_format = args.output_format.lower()

    # Check input file exists
    if not os.path.exists(args.infile):
        logger.error(f"Cannot find input file: {args.infile}")
        config_error_found = True
              
    # Check output format
    if output_format not in ('uvfits', 'miriad', 'mir', 'ms', 'uvh5', 'sdp', 'uvx'):
        logger.error(f"Output format not valid: {output_format}")
        config_error_found = True       
    
    # Raise error and quit if config issues exist
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
                filelist_out.append(os.path.join(subdir, bn_out))
            else:
                filelist_out.append(os.path.join(args.outfile, bn_out))
    else:
        filelist     = [args.infile]
        filelist_out = [args.outfile]

    ######
    # Start main conversion loop

    if not args.verbose:
        logger.remove()

    if args.num_workers <= 1:
        with warnings.catch_warnings() as wc:
            if not args.verbose:
                warnings.simplefilter("ignore")

            logger.info(f"Starting conversion on {len(filelist)} files (single-process)")
            for fn_in, fn_out in tqdm(zip(filelist, filelist_out)):
                res = convert_single_file(args, fn_in, fn_out, array_config, output_format, conj, context)

    else:
        # Create a local cluster and setup workers
        logger.info(f"Starting dash LocalCluster with {args.num_workers} workers (multi-process)")
        cluster = LocalCluster(n_workers=args.num_workers, threads_per_worker=1)
        client = cluster.get_client()
        logger.info(f"Starting dash Client on {client.dashboard_link}")
        logger.info(f"Starting conversion on {len(filelist)} files")
        
        ## Setup dask and form delayed queue of tasks
        npartitions = args.num_workers * 64
        dask_bag = from_sequence(zip(filelist, filelist_out), npartitions=npartitions)
        logger.info(f"Using dask bag: {dask_bag}")
        dask_bag.map(convert_single_file_dask, args, array_config, output_format, conj, context)

        # Run compute and measure progress
        with ResourceProfiler() as rprof, Profiler() as prof, ProgressBar() as pbar:
            with warnings.catch_warnings() as wc:
                if not args.verbose:
                    warnings.simplefilter("ignore")
                dask_bag.compute(rerun_exceptions_locally=True)

        if args.profile:
            reset_logger()
            from bokeh.plotting import save, output_file
            logger.opt(ansi=True).info("\n <m><b>################# Profiler #################</b></m>")
            logger.info("Resources:")
            pprint.pprint(rprof.results)
            logger.info("Tasks:")
            pprint.pprint(prof.results)
            bp = prof.visualize()
            output_file("profile.html")
            save(bp)
            output_file("resource_profile.html")
            brp = rprof.visualize()
            save(brp)
            logger.info("Profiling info saved to profile.html")

if __name__ == "__main__": #pragma: no cover
    print(sys.argv[1:])
    args = parse_args(sys.argv[1:])
    run(args)
    

