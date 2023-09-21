### aavs_uv

Utilities for handling UV data products for AAVS.

These codes use the `UVData` class from [pyuvdata](https://pyuvdata.readthedocs.io) to convert the raw HDF5 correlator output files to science-ready data formats like UVFITS, MIRIAD, and CASA MeasurementSets.

#### Main methods

```python 

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

def phase_to_sun(uv: UVData, t0: Time) -> UVData:
    """ Phase UVData to sun, based on timestamp 

    Computes the sun's RA/DEC in GCRS for the given time, then applies phasing.
    This will then recompute UVW and apply phase corrections to data.

    Note: 
        Phase center is set to 'sidereal', i.e. fixed RA and DEC, not 'ephem', so that
        we can apply calibration solutions taken at time t0 (where the Sun was when calibration
        was run, not where it is now!)

    Args:
        uv (UVData): UVData object to apply phasing to (needs to have a phase center defined)
        t0 (Time): Astropy Time() to use to compute Sun's RA/DEC

    Returns:
        uv (UVData): Same UVData as input, but with new phase center applied
    """

```

#### Dependencies

```
pandas
h5py
astropy
pyuvdata
numpy
```