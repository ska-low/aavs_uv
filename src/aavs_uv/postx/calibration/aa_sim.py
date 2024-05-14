
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray

from .simple_sim import simulate_visibilities_pointsrc
from .gsm_sim import simulate_visibilities_gsm
from .aa_model import Model

class AaSimulator(object):
    def __init__(self, aa: ApertureArray):
        """ Setup AaPlotter

        Args:
            aa (ApertureArray): Aperture array 'parent' object to use
        """
        self.aa = aa
        self.model = Model(visibilities=None, sky_model=None)
        self.__setup_docstrings()

    def __setup_docstrings(self):
        # Inherit docstrings
        self.sim_vis_pointsrc.__func__.__doc__  = simulate_visibilities_pointsrc.__doc__
        self.sim_vis_gsm.__func__.__doc__  = simulate_visibilities_gsm.__doc__

    def sim_vis_pointsrc(self, *args, **kwargs):
        # Docstring inherited
        self.model.visibilities = simulate_visibilities_pointsrc(self.aa, *args, **kwargs)
        return self.model.visibilities

    def sim_vis_gsm(self, *args, **kwargs):
        # Docstring inherited
        self.model.visibilities = simulate_visibilities_gsm(self.aa, *args, **kwargs)
        return self.model.visibilities