"""aa_simulation: ApertureArray simulation tools submodule."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray

from ..aa_module import AaBaseModule
from .aa_model import Model
from .gsm_sim import simulate_visibilities_gsm
from .simple_sim import simulate_visibilities_pointsrc


class AaSimulator(AaBaseModule):
    """Simulate visibilities using matvis.

    Provides the following:
        sim_vis_pointsrc(): Simulate visibilities from point source dictionary
        sim_vis_gsm(): Simulate visibilites using pygdsm sky model
        orthview_gsm(): View observed diffuse sky model (Orthographic)
        mollview_gsm(): View observed diffuse sky model (Mollview)
    """

    def __init__(self, aa: ApertureArray):
        """Setup AaPlotter.

        Args:
            aa (ApertureArray): Aperture array 'parent' object to use
        """
        self.aa = aa
        self.model = Model(
            visibilities=None,
            point_source_skymodel=None,
            beam=None,
            gains=None,
            gsm=None,
        )

        self.__setup_docstrings('simulation')

    def __setup_docstrings(self, name):
        self.__name__ = name
        self.name = name
        # Inherit docstrings
        self.sim_vis_pointsrc.__func__.__doc__ = simulate_visibilities_pointsrc.__doc__
        self.sim_vis_gsm.__func__.__doc__ = simulate_visibilities_gsm.__doc__

    def sim_vis_pointsrc(self, *args, **kwargs):
        # Docstring inherited
        self.model.visibilities = simulate_visibilities_pointsrc(
            self.aa, *args, **kwargs
        )
        return self.model.visibilities

    def sim_vis_gsm(self, *args, **kwargs):
        # Docstring inherited
        self.model.visibilities = simulate_visibilities_gsm(self.aa, *args, **kwargs)
        return self.model.visibilities

    def orthview_gsm(self, *args, **kwargs):
        """View diffuse sky model (Orthographic)."""
        if self.model.gsm.observed_sky is None:
            self.model.gsm.generate()
        self.model.gsm.view()

    def mollview_gsm(self, *args, **kwargs):
        """View diffuse sky model (Mollweide)."""
        if self.model.gsm.observed_sky is None:
            self.model.gsm.generate()
        self.model.gsm.view_observed_gsm()
