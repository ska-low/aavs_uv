""" Very simple model class for visibilities / sky models"""
from dataclasses import dataclass
import xarray as xr

@dataclass
class Model:
    visibilities: xr.DataArray
    sky_model: dict
