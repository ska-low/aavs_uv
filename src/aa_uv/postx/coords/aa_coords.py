"""aa_coords: ApertureArray coordinate tools submodule."""
from __future__ import annotations

import typing

import numpy as np
from aa_uv.postx.coords.coord_utils import phase_vector, skycoord_to_lmn
from aa_uv.postx.imaging.aa_imaging import SPEED_OF_LIGHT
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.coordinates import get_sun as astropy_get_sun

if typing.TYPE_CHECKING:
    from ..aperture_array import ApertureArray

from ..aa_module import AaBaseModule


def generate_phase_vector(aa, src: SkyCoord, conj: bool=False, coplanar: bool=False):
    """Generate a phase vector for a given source.

    Args:
        aa (ApertureArray): Aperture array 'parent' object to use
        src (astropy.SkyCoord or ephem.FixedBody): Source to compute delays toward
        conj (bool): Conjugate data if True
        coplanar (bool): Treat array as coplanar if True. Sets antenna z-pos to zero

    Returns:
        c (np.array): Per-antenna phase weights
    """
    lmn = skycoord_to_lmn(src, aa.coords.get_zenith())
    ant_pos = aa.xyz_enu
    if coplanar:
        ant_pos[..., 2] = 0

    t_g = np.einsum('id,pd', lmn, aa.xyz_enu, optimize=True) / SPEED_OF_LIGHT
    c = phase_vector(t_g, aa._ws('f').to('Hz').value, conj=conj, dtype='complex64')
    return c


def get_zenith(aa) -> SkyCoord:
    """Return the sky coordinates at zenith.

    Args:
        aa (ApertureArray): Aperture array objecy to use

    Returns:
        zenith (SkyCoord): Zenith SkyCoord object
    """
    zen_aa = AltAz(alt=Angle(90, unit='degree'), az=Angle(0, unit='degree'),
                    obstime=aa._ws('t'), location=aa.earthloc)
    zen_sc = SkyCoord(zen_aa).icrs
    return zen_sc


def get_alt_az(aa, sc: SkyCoord) -> SkyCoord:
    """Convert SkyCoord into alt/az coordinates.

    Args:
        aa (ApertureArray): Aperture array object to use
        sc (SkyCoord): Input sky coordinate

    Returns:
        sc_altaz (SkyCoord): Same coordinates, in alt/az frame.
    """
    sc.obstime = aa._ws('t')
    sc.location = aa.earthloc
    return sc.altaz


def get_sun(aa) -> SkyCoord:
    """Return the sky coordinates of the Sun.

    Args:
        aa (ApertureArray): Aperture array object to use

    Returns:
        sun_sc (SkyCoord): sun SkyCoord object
    """
    sun_sc = SkyCoord(astropy_get_sun(aa._ws('t')), location=aa.earthloc)
    return sun_sc


####################
## AA_COORDS CLASS
####################

class AaCoords(AaBaseModule):
    """Coordinate utils.

    Provides the following:
        get_sun() - Get the position of the sun as a SkyCoord
        get_zenith() - Get the zenith as a SkyCoord
        get_alt_az() - Get the alt/az of a given SkyCoord
        generate_phase_vector() - Generate a phase vector toward a given SkyCoord

    """
    def __init__(self, aa: ApertureArray):
        """Setup AaCoords.

        Args:
            aa (ApertureArray): Aperture array 'parent' object to use
        """
        self.aa = aa
        self.__setup_docstrings('coords')

    def __setup_docstrings(self, name):
        self.__name__ = name
        self.name = name
        # Inherit docstrings
        self.generate_phase_vector.__func__.__doc__  = generate_phase_vector.__doc__
        self.get_sun.__func__.__doc__  = get_sun.__doc__
        self.get_zenith.__func__.__doc__  = get_zenith.__doc__
        self.get_alt_az.__func__.__doc__  = get_alt_az.__doc__

    def generate_phase_vector(self, *args, **kwargs):
        # Docstring inherited
        return generate_phase_vector(self.aa, *args, **kwargs)

    def get_sun(self, *args, **kwargs):
        # Docstring inherited
        return get_sun(self.aa, *args, **kwargs)

    def get_zenith(self, *args, **kwargs):
        # Docstring inherited
        return get_zenith(self.aa, *args, **kwargs)

    def get_alt_az(self, *args, **kwargs):
        # Docstring inherited
        return get_alt_az(self.aa, *args, **kwargs)