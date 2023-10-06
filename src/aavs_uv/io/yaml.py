import yaml

def load_yaml(filename: str) -> dict:
    """ Read YAML file into a Python dict """ 
    d = yaml.load(open(filename, 'r'), yaml.Loader)
    return d