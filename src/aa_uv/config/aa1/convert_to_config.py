from astropy.coordinates import EarthLocation
import pandas as pd
import os
import glob

def read_aa1_coordinates_csv_header(fn: str) -> dict:
    """ Read the header for a AA1 coordinates CSV file

    Returns:
        hdr (dict): For example:
                    {'ID': 'S8-1',
                     'Lat': -26.856150423,
                     'Lon': 116.729640683,
                     'HAE': 330.1044,
                     'Datum': 'WGS84',
                     'Epoch': '2024.2'}
    """
    def parse_hdr_line(fh):
        k, v = fh.readline().strip('#\n').split(':')
        v = v.strip()
        if k in ('Lat', 'Lon', 'HAE'):
            v = float(v)
        return k, v

    with open(fn, 'r') as fh:
        hdr = { 'ID': fh.readline().strip('#\n') }
        for ii in range(5):
            k, v = parse_hdr_line(fh)
            hdr[k] = v
    return hdr

def read_aa1_coordinates_csv(fn: str) -> pd.DataFrame:
    hdr = read_aa1_coordinates_csv_header(fn)

    cols = 'SB-Antenna,EEP,Easting,Northing,AHD,Lat,Lon,HAE,ECEF-X,ECEF-Y,ECEF-Z,E,N,U'.split(',')
    df = pd.read_csv(fn, comment='#', names=cols)

    return hdr, df

def generate_uv_config_from_aa1_coordinates_csv(fn):
    """ Generate UV configs, create directories """

    # Read coordinates file
    hdr, df = read_aa1_coordinates_csv(fn)

    # Create Directory
    os.mkdir(hdr['ID'])

    eloc = EarthLocation(lon=hdr['Lon'], lat=hdr['Lat'], height=hdr['HAE'])

    # Generate uv_config.yaml
    uvc = f"""# UVX configuration file
    history: Created with generate_uv_config_from_aa1_coordinates_csv()
    instrument: {hdr['ID']}
    telescope_name: {hdr['ID']}
    telescope_ECEF_X: {eloc.x.value}
    telescope_ECEF_Y: {eloc.y.value}
    telescope_ECEF_Z: {eloc.z.value}
    channel_spacing: 781250.0           # Channel spacing in Hz
    channel_width: 925926.0             # 781250 Hz * 32/27 oversampling gives channel width
    antenna_locations_file: antenna_locations.txt
    baseline_order_file: baseline_order.txt
    polarization_type: linear_crossed  # stokes, circular, linear (XX, YY, XY, YX) or linear_crossed (XX, XY, YX, YY)
    vis_units: uncalib
    # PyUVData specific keywords
    future_array_shapes: False
    flex_spw: False
    Nspws: 1
    Nphase: 1"""

    # Write to file
    with open(os.path.join(hdr['ID'], 'uv_config.yaml'), 'w') as fh:
        for line in uvc.split('\n'):
            fh.write(line.strip() + '\n')

    # Write antenna csv
    df['flagged'] = False
    df['name']  = df['SB-Antenna']

    dsel = df[['name', 'E', 'N', 'U', 'flagged']]
    dsel.to_csv(os.path.join(hdr['ID'], 'antenna_positions.txt'), sep=' ', header=('name', 'E', 'N', 'U', 'flagged'), index_label='idx')

    # Copy over baseline order
    os.system(f"cp baseline_order.txt {hdr['ID']}/")

if __name__ == "__main__":
    fl = glob.glob('*.csv')
    for fn in fl:
        generate_uv_config_from_aa1_coordinates_csv(fn)