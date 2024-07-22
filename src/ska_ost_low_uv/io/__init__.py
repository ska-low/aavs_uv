"""Default __init__ imports for io submodule."""

from ..utils import import_optional_dependency
from .to_uvx import get_hdf5_metadata as get_hdf5_metadata
from .to_uvx import hdf5_to_uvx as hdf5_to_uvx
from .to_uvx import load_observation_metadata as load_observation_metadata

try:
    import_optional_dependency('ska_sdp_datamodels')
    from .to_sdp import hdf5_to_sdp_vis as hdf5_to_sdp_vis
    from .to_sdp import uvdata_to_sdp_vis as uvdata_to_sdp_vis
except ImportError:  # pragma: no cover
    pass

from .cal import read_cal as read_cal
from .cal import write_cal as write_cal
from .to_pyuvdata import hdf5_to_pyuvdata as hdf5_to_pyuvdata
from .to_pyuvdata import phase_to_sun as phase_to_sun
from .uvx import read_uvx as read_uvx
from .uvx import write_uvx as write_uvx
