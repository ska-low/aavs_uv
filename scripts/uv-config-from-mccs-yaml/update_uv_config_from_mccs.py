""" Update UV config from MCCS YAML

* Retrieves the lasest ska-low-deployment git repository (where station YAML files are located)
* Generates aa_uv's internally-used UV Configuration for a station
* Copies these over to aa_uv/src/aa_uv/config
"""
import os
from datetime import datetime
from astropy.time import Time

from aa_uv.io.mccs_yaml import station_location_from_platform_yaml


MCCS_CONFIG_PATH = 'ska-low-deployment/tmdata/instrument/mccs-configuration'


def generate_uv_config(name):
    """ Generate UV configs, create directories """
    now = Time(datetime.now())

    # Read the YAML file and return an EarthLocation and pandas Dataframe of antenna positions
    eloc, antennas = station_location_from_platform_yaml(f'{MCCS_CONFIG_PATH}/{name}.yaml', name)

    # Create Directory
    os.mkdir(name)

    # Generate uv_config.yaml
    uvc = f"""# UVX configuration file
    history: Created with scripts/uv-config-from-mccs-yaml at {now.iso}
    instrument: {name}
    telescope_name: {name}
    telescope_ECEF_X: {eloc.x.value}
    telescope_ECEF_Y: {eloc.y.value}
    telescope_ECEF_Z: {eloc.z.value}
    channel_spacing: 781250.0           # Channel spacing in Hz
    channel_width: 925926.0             # 781250 Hz * 32/27 oversampling gives channel width
    antenna_locations_file: antenna_locations.txt
    baseline_order_file: baseline_order.txt
    polarization_type: linear_crossed  # stokes, circular, linear (XX, YY, XY, YX) or linear_crossed (XX, XY, YX, YY)
    vis_units: uncalib"""

    # Write to file
    with open(os.path.join(name, 'uv_config.yaml'), 'w') as fh:
        for line in uvc.split('\n'):
            fh.write(line.strip() + '\n')

    # Write antenna csv
    antennas.to_csv(os.path.join(name, 'antenna_locations.txt'), sep=' ', header=('name', 'E', 'N', 'U', 'flagged'), index_label='idx')

    # Copy over baseline order
    os.system(f"cp config/baseline_order.txt {name}/")


if __name__ == "__main__":
    import glob

    os.system("bash update_mccs_configuration_yaml.sh")

    yaml_list = sorted(glob.glob(f"{MCCS_CONFIG_PATH}/*.yaml"))
    for fn in yaml_list:
        name = os.path.splitext(os.path.basename(fn))[0]
        if name.startswith('s'):
            print(f"Generating {name}")
            generate_uv_config(name)
            os.system(f"mv {name} ../../src/aa_uv/config")