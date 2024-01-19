import tqdm
import joblib
from joblib import Parallel
from joblib import delayed
from .utils import reset_logger

def task(*args, **kwargs):
    """ Rename 'delayed' to 'task', and setup logger """
    reset_logger(use_tqdm=True)
    return delayed(*args, **kwargs)
    
def run_in_parallel(task_list: list, n_workers: int=-1, show_progressbar=True, backend: str='loky', verbose: bool=False):
    """ Run a list of tasks in parallel, using joblib + tqdm 
    
    Args:
        task_list (list): A list of tasks (using @parallelize.task 'delayed' lazy loading)
        n_workers (int): Number of workers to use. Defaults to -1, i.e. use all cores.
        show_progressbar (bool): Show tqdm progress bar (default True)
        backend (str): Parallel processing backend, one of 'loky' (default) or 'dask'
    
    Example usage:
        ```python
        from .parallelize import task, run_in_parallel

        @task
        def my_task(filename_in, filename_out):
            ...
            return result
        
        for fn_in, fn_out in zip(filelist_in, filelist_out):
            task_list.append(my_task(fn_in, fn_out))

        run_in_parallel(task_list, n_workers=8)
        ```
    """
    if backend == 'dask':
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = cluster.get_client()
        level = "INFO" if verbose else "WARNING"
        logger = reset_logger(use_tqdm=True, level=level)
        logger.info(f"Using dask LocalCluster(n_workers={n_workers}) backend")
        logger.info(f"Dashboard running at {client.dashboard_link}")

    with joblib.parallel_backend(backend):
        return Parallel(n_jobs=n_workers)(tqdm.tqdm(task_list))
