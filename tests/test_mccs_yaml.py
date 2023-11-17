from aavs_uv.io.mccs_yaml import station_location_from_platform_yaml 
from aavs_uv.utils import get_resource_path

def test_mccs_yaml():
    fn = get_resource_path('config/aavs3/aavs3_mccs.yaml')
    earth_loc, ant_enu = station_location_from_platform_yaml(fn)
    print(earth_loc)
    print(ant_enu)

if __name__ == "__main__":
    test_mccs_yaml()