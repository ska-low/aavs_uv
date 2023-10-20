from operator import itemgetter

from aavs_uv.io.yaml import load_yaml
import pandas as pd
from astropy.coordinates import EarthLocation

def station_location_from_platform_yaml(fn_yaml: str) -> (EarthLocation, pd.DataFrame):
    """ Load station location from AAVS3 MCCS yaml config 

    Args:
        fn_yaml (str): Filename path to yaml config

    Returns:
        (earth_loc, ant_enu): astropy EarthLocation and antenna ENU locations in m
    """
    d =load_yaml(fn_yaml)
    d_station = d['platform']['array']['station_clusters']['a1']['stations']['1']

    # Generate pandas dataframe of antenna positions
    d_ant = d_station['antennas']

    location_getter = itemgetter("east", "north", "up")
    ant_enu = [
        [f"SB{a['smartbox']}-{a['smartbox_port']}", *location_getter(a["location_offset"])]
        for a in sorted(d_ant.values(), key=lambda x: (int(x["tpm"]), x["tpm_y_channel"] // 2))
    ]

    ant_enu = pd.DataFrame(ant_enu, columns=('name', 'E', 'N', 'U'))

    # Generate array central reference position
    # NOTE: Currently using WGS84 instead of GDA2020
    loc = d_station['reference']
    earth_loc = EarthLocation.from_geodetic(loc['longitude'], loc['latitude'], loc['ellipsoidal_height'])

    return  earth_loc, ant_enu
    