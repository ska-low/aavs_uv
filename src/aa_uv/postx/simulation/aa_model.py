""" Very simple model class for visibilities / sky models"""
from dataclasses import dataclass
import xarray as xr
from pygdsm import GlobalSkyModel

@dataclass
class Model:
    visibilities: xr.DataArray=None       # Model visibilities
    point_source_skymodel: dict=None      # Point source sky model
    beam: xr.DataArray=None               # Primary beam model
    gains: xr.DataArray=None              # Per-antenna gain model
    gsm: GlobalSkyModel=None              # Pygdsm diffuse sky model
