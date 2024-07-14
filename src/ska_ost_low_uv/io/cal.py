"""cal: calibration data I/O utils."""
from ska_ost_low_uv.datamodel.cal import UVXAntennaCal

#from ska_ost_low_uv.utils import get_resource_path, load_yaml


def write_uvx(cal: UVXAntennaCal, filename: str):
    """Write a UVXAntennaCAl object to a HDF5 file.

    Args:
        cal (UVXAntennaCal): ska_ost_low_uv.datamodel.UVXAntennaCal object
        filename (str): name of output file
    """
    # Load UVX schema from YAML. We can use this to load descriptions
    # And other metadata from the schema (e.g. dimensions)
    #uvx_schema = load_yaml(get_resource_path('datamodel/cal.yaml'))
    pass
