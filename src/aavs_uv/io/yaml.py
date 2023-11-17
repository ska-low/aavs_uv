import yaml
from aavs_uv.utils import get_config_path

def load_yaml(filename: str) -> dict:
    """ Read YAML file into a Python dict """ 
    d = yaml.load(open(filename, 'r'), yaml.Loader)
    return d

def load_config(telescope_name: str) -> dict:
    """ Load internal array configuration by telescope name """
    yaml_path = get_config_path(telescope_name)
    d = load_yaml(yaml_path)
    return d