"""
Simple sky model class for ephemeris using pyephem
"""
import ephem
import numpy as np

from astropy.coordinates import SkyCoord, get_body
from .ant_array import RadioArray

def generate_skycat(observer: RadioArray):
    """ Generate a SkyModel for a given observer with major radio sources
    
    Args:
        observer (AntArray / ephem.Observer): Observatory instance
    
    Returns:
        skycat (SkyModel): A sky catalog with the A-team sources 
    """
    skycat = {
        'Virgo_A': SkyCoord('12h 30m 49s', '+12:23:28', unit=('hourangle', 'degree')),
        'Hydra_A': SkyCoord('09h 18m 5.6s', '-12:5:44.0',  unit=('hourangle', 'degree')),
        'Centaurus_A':  SkyCoord('13h 25m 27.6s', '−43:01:09', unit=('hourangle', 'degree')),
        'Pictor_A': SkyCoord('05h 19m 49.721s', '−45:46:43.85', unit=('hourangle', 'degree')),
        'Hercules_A': SkyCoord('16h 51m 08.15', '+04:59:33.32', unit=('hourangle', 'degree')),
        'Fornax_A': SkyCoord('03h 22m 41.7', '−37:12:30', unit=('hourangle', 'degree')),

    }
    skycat.update(generate_skycat_solarsys(observer))
    return skycat

def generate_skycat_solarsys(observer: RadioArray):
    """ Generate Sun + Moon for observer """
    skycat = {
        'Sun': get_body('sun', observer.workspace['t']),
        'Moon': get_body('moon', observer.workspace['t'])
    }
    return skycat