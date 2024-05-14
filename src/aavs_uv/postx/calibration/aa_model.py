""" Very simple model class for visibilities / sky models"""
from dataclasses import dataclass
import xarray as xr
from pygdsm import GlobalSkyModel

@dataclass
class Model:
    visibilities: xr.DataArray=None
    sky_model: dict=None
    gsm: GlobalSkyModel=None
