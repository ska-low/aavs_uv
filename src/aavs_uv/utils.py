import numpy as np
import os
import aavs_uv
from loguru import logger


def get_resource_path(relative_path: str) -> str:
    """ Get the path to an internal package resource (e.g. data file) 
    
    Args:
        relative_path (str): Relative path to data file, e.g. 'config/aavs3/uv_config.yaml'
    
    Returns:
        abs_path (str): Absolute path to the data file
    """

    path_root = os.path.abspath(aavs_uv.__path__[0])
    abs_path = os.path.join(path_root, relative_path)
    
    if not os.path.exists(abs_path):
        logger.warning(f"File not found: {abs_path}")

    return abs_path


def get_config_path(name: str) -> str:
    """ Get path to internal array configuration by telescope name 
    
    Args:
        name (str): Name of telescope to load array config, e.g. 'aavs2'
    
    Returns:
        abs_path (str): Absolute path to config file.
    """
    relative_path = f"config/{name}/uv_config.yaml"
    return get_resource_path(relative_path)


def vis_arr_to_matrix(d: np.ndarray, n_ant: int, tri: str='upper', V: np.array=None, conj=False):
    """ Convert 1D visibility array (lower/upper) triangular to correlation matrix 
    
    Args:
        d (np.ndarray): 1D Numpy array with len N_ant * (N_ant + 1) / 2
        n_ant (int): Number of antennas in array
        tri (str): Either 'upper' or 'lower' triangular, to match data
        V (np.ndarray): Either None (create new array) or an (N_ant x N_ant) Numpy array
    
    Returns:
        V (np.ndarray): (N_ant x N_ant) correlation matrix
    """
    n_bl = n_ant * (n_ant + 1) // 2
    if d.shape[0] != n_bl:
        raise RuntimeError(f"N_ant: {n_ant} -> N_bl: {n_bl}, but d.shape = {d.shape[0]}")
    if tri == 'upper':
        ix, iy = np.triu_indices(n_ant)
    elif tri == 'lower':
        ix, iy = np.tril_indices(n_ant)
    else:
        raise RuntimeError('Must be upper or lower triangular')
    
    if V is None:
        V = np.zeros((n_ant, n_ant), dtype='complex64')
    else:
        try:
            assert V.shape == (n_ant, n_ant)
        except AssertionError:
            raise RuntimeError("Correlation matrix shape mismatch")
        
    V[ix, iy] = d[:]
    V[iy, ix] = np.conj(d[:])

    if conj:
        V = np.conj(V)
    
    return V