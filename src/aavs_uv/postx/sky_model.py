"""
Simple sky model class for ephemeris using pyephem
"""
import ephem
import numpy as np

from astropy.coordinates import SkyCoord, get_body
from .ant_array import RadioArray

class RadioSource(SkyCoord):
    """ A SkyCoordinate with a magnitude """
    def __init__(self, *args, mag=1.0, unit=None, **kwargs):
        if unit is None:
            unit=('hourangle', 'degree')
        super().__init__(*args, unit=unit, **kwargs)
        self.mag = mag

def generate_skycat(observer: RadioArray):
    """ Generate a SkyModel for a given observer with major radio sources
    
    Args:
        observer (AntArray / ephem.Observer): Observatory instance
    
    Returns:
        skycat (SkyModel): A sky catalog with the A-team sources 
    """
    skycat = {
        'Virgo_A':     RadioSource('12h 30m 49s',    '+12:23:28',    ),
        'Hydra_A':     RadioSource('09h 18m 5.6s',   '-12:5:44.0',   ),
        'Centaurus_A': RadioSource('13h 25m 27.6s',  '−43:01:09',    ),
        'Pictor_A':    RadioSource('05h 19m 49.72s', '−45:46:43.85', ),
        'Hercules_A':  RadioSource('16h 51m 08.15',  '+04:59:33.32', ),
        'Fornax_A':    RadioSource('03h 22m 41.7',   '−37:12:30',    ),
    }
    skycat.update(generate_skycat_solarsys(observer))
    return skycat

def generate_skycat_solarsys(observer: RadioArray):
    """ Generate Sun + Moon for observer """
    sun_gcrs  = get_body('sun', observer.workspace['t'])
    moon_gcrs = get_body('moon', observer.workspace['t'])
    skycat = {
        'Sun': RadioSource(sun_gcrs.ra, sun_gcrs.dec, mag=1.0),
        'Moon': RadioSource(moon_gcrs.ra, moon_gcrs.dec, mag=1.0),
    }
    return skycat