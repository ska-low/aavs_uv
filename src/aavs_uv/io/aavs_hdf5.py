import h5py

def get_hdf5_metadata(filename: str) -> dict:
    """ Extract metadata from HDF5 and perform checks """
    with h5py.File(filename, mode='r') as datafile:
        expected_keys = ['n_antennas', 'ts_end', 'n_pols', 'n_beams', 'tile_id', 'n_chans', 'n_samples', 'type',
                         'data_type', 'data_mode', 'ts_start', 'n_baselines', 'n_stokes', 'channel_id', 'timestamp',
                         'date_time', 'n_blocks']
    
        # Check that keys are present
        if set(expected_keys) - set(datafile.get('root').attrs.keys()) != set():
            raise Exception("Missing metadata in file")
    
        # All good, get metadata
        metadata = {k: v for (k, v) in datafile.get('root').attrs.items()}
        metadata['n_integrations'] = metadata['n_blocks'] * metadata['n_samples']
        metadata['data_shape'] = datafile['correlation_matrix']['data'].shape
    return metadata