## ska-ost-low-uv

Utilities for handling UV data products for low-frequency aperture array telescopes.

`ska-ost-low-uv` provides a `UVX` data format and Python class for storing and handling interferometric data.

These codes use the `UVData` class from [pyuvdata](https://pyuvdata.readthedocs.io) to convert the raw HDF5 correlator output files to science-ready data formats like UVFITS, MIRIAD, and CASA MeasurementSets.

Additionally, data can be loaded into the `Visibility` data model from [ska-sdp-datamodels](https://developer.skao.int/projects/ska-sdp-datamodels/en/latest/), which is based on [xarray](https://docs.xarray.dev/en/stable/).

![aavsuv-overview](https://github.com/ska-sci-ops/aa_uv/blob/main/docs/images/uv_flow.png?raw=true)

Some simple calibration and imaging utilities are provided in the `postx` submodule. This is an optional extra,
that requires the `matvis` package for simulations.

### Installation

Download this repository, then install via `pip install .`. To install optional extras:

```
pip install .[postx]  # Post-correlation imaging and QA tools
pip install .[sdp]    # Support for SDP Visibility
pip install .[casa]   # Installs python-casacore for MS support
```

Or run `pip install .[all]` to install all optional extras.

#### Fresh conda install

To install from scratch using `conda`, download this repository, cd into the directory, and run

```
conda env create -f environment.yml
conda activate ska_ost_low_uv
```

You can then use `conda activate ska_ost_low_uv` to start up the environment, and `conda deactivate` to leave it.

Then run `pip install .` and any extras (e.g. `pip install .[postx]`).

### File conversion: command-line script

Once installed, a command-line utility, `aa_uv`, will be available:

```
> aa_uv -h

usage: aa_uv [-h] -o OUTPUT_FORMAT [-c ARRAY_CONFIG] [-n TELESCOPE_NAME] [-s] [-j] [-b] [-B] [-x FILE_EXT] [-i CONTEXT_YAML] [-w NUM_WORKERS] [-v] [-p PARALLEL_BACKEND] [-N N_INT_PER_FILE] [-z] infile outfile

AAVS UV file conversion utility

positional arguments:
  infile                Input filename
  outfile               Output filename

options:
  -h, --help            show this help message and exit
  -o OUTPUT_FORMAT, --output_format OUTPUT_FORMAT
                        Output file format (uvx, uvfits, miriad, ms, uvh5, sdp). Can be comma separated for multiple formats.
  -c ARRAY_CONFIG, --array_config ARRAY_CONFIG
                        Array configuration YAML file. If supplied, will override ska_ost_low_uv internal array configs.
  -n TELESCOPE_NAME, --telescope_name TELESCOPE_NAME
                        Telescope name, e.g. 'aavs3'. If supplied, will attempt to use ska_ost_low_uv internal array config.
  -s, --phase-to-sun    Re-phase to point toward Sun (the sun must be visible!). If flag not set, data will be phased toward zenith.
  -b, --batch           Batch mode. Input and output are treated as directories, and all subfiles are converted.
  -B, --megabatch       MEGA batch mode. Runs on subdirectories too, e.g. eb-aavs3/2023_12_12/*.hdf5.
  -x FILE_EXT, --file_ext FILE_EXT
                        File extension to search for in batch mode
  -i CONTEXT_YAML, --context_yaml CONTEXT_YAML
                        Path to observation context YAML (for SDP / UVX formats)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of parallel processors (i.e. number of files to read in parallel).
  -v, --verbose         Run with verbose output.
  -p PARALLEL_BACKEND, --parallel_backend PARALLEL_BACKEND
                        Joblib backend to use: 'loky' (default) or 'dask'
  -N N_INT_PER_FILE, --n_int_per_file N_INT_PER_FILE
                        Set number of integrations to write per file. Only valid for MS, Miriad, UVFITS, uvh5 output.
  -z, --zipit           Zip up a MS or Miriad file after conversion (flag ignored for other files)
```

The converter needs a [yaml configuration file](https://github.com/ska-sci-ops/aa_uv/tree/main/example-config), which can be supplied with the `-c` argument, or internal defaults can be used instead via the `-n` argument (for '-n aavs2' and '-n aavs3'):

```
# Convert AAVS3 HDF5 data into a MeasurementSet
> aa_uv -n aavs3 -o ms correlator_data.hdf5 my_new_measurement_set.ms
```

### File conversion: Python API

```python

from ska_ost_low_uv.io import hdf5_to_pyuvdata, hdf5_to_sdp_vis

def hdf5_to_pyuvdata(filename: str, yaml_config: str) -> pyuvdata.UVData:
    """ Convert AAVS2/3 HDF5 correlator output to UVData object

    Args:
        filename (str): Name of file to open
        yaml_config (str): YAML configuration file with basic telescope info.
                           See README for more information
    Returns:
        uv (pyuvdata.UVData): A UVData object that can be used to create
                              UVFITS/MIRIAD/UVH5/etc files
    """

def hdf5_to_sdp_vis(fn_raw: str, yaml_raw: str) -> Visibility:
    """ Generate a SDP Visibility object from a AAVS2 HDF5 file

    Args:
        fn_raw (str): Filename of raw HDF5 data to load.
        yaml_raw (str): YAML config data with telescope information
                        See https://github.com/ska-low/aa_uv/tree/main/config#uv_configyaml

    Notes:
        The HDF5 files generated by AAVS2/3 are NOT the same format as that found in
        ska-sdp-datamodels HDF5 visibilty specification.
        The AAVS DAQ receiver code in aavs-system has some info on the HDF5 format, here:
        https://gitlab.com/ska-telescope/aavs-system/-/blob/master/python/pydaq/persisters/corr.py
    """
```

### Installation

#### Mamba / conda

To help install into a fresh conda environment, a [environment.yml](https://github.com/ska-sci-ops/aa_uv/blob/main/environment.yml) is provided. To create a new environment, download this repo then run:

```
conda env create -f environment.yml
```

This will create an environment called `aavs`, which you enter by typing `conda activate aavs`. You can then activate this and install via:

```
conda activate aavs
pip install .
```
Pip will then install `ska_ost_low_uv` and the few final packages that are not available in the conda-forge package manager (e.g. `pygdsm`, `pyuvdata`, `ska-sdp-datamodels`).

#### Pip / manual

If you have an existing Python 3 installation, you can install with pip via:

```
pip install git+https://github.com/ska-sci-ops/aa_uv/edit/main/README.md
```

Alternatively, download this repository and install using `pip install .`. A list of required packages can be found in the [pyproject.toml](https://github.com/ska-sci-ops/aa_uv/blob/main/pyproject.toml#L13).

#### Astronomy packages

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

`ska_ost_low_uv` is built upon the following astronomy packages:
* [astropy](http://www.astropy.org/) for coordinate calculations.
* [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) for interferometric data format conversion.
* [matvis](https://github.com/HERA-Team/matvis) for visibility simulation.
* [pygdsm](https://github.com/telegraphic/pygdsm) for diffuse sky model generation.# ska-ost-low-uv

SKA Ost Low UV provides utilities for handling UV data products for SKA Low.


## Documentation

[![Documentation Status](https://readthedocs.org/projects/ska-telescope-ska-ost-low-uv/badge/?version=latest)](https://developer.skao.int/projects/ska-ost-low-uv/en/latest/?badge=latest)

The documentation for this project, including how to get started with it, can be found in the `docs` folder, or browsed in the SKA development portal:

* [ska-ost-low-uv documentation](https://developer.skatelescope.org/projects/ska-ost-low-uv/en/latest/index.html "SKA Developer Portal: ska-ost-low-uv documentation")
