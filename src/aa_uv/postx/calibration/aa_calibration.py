from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray

from ..aa_module import AaBaseModule

from . import holography
from . import stefcal


######################
## AA_CALIBRATOR CLASS
######################

class AaCalibrator(AaBaseModule):
    """ApertureArray Calibration module

    Provides the following sub-modules:
    holography - self-holography techniques
    stefcal - calibration based on stefcal approach

    """
    def __init__(self, aa: ApertureArray):
        """Setup AaCalibrator

        Args:
            aa (ApertureArray): Aperture array 'parent' object to use
        """
        self.aa = aa
        self.holography = holography.AaHolographer(aa)
        self.stefcal = stefcal

        self.__setup_docstrings('calibration')

    def __setup_docstrings(self, name):
        self.__name__ = name
        self.name = name
        # Inherit docstrings
        #self.make_image.__func__.__doc__  = make_image.__doc__
