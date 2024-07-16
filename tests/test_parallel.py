"""test_parallel: Test running in parallel.

NOTE: This script seems to break pytest, if it runs before ipynb tests.
Hence it being in a folder starting with a z_, so it runs last.
"""

import time

from ska_ost_low_uv.parallelize import run_in_parallel, task


@task
def my_task(x: float) -> int:
    """Simple task to sleep 0.1s.

    Args:
        x (float): Dummy argument

    Returns:
        ret (int): Returns 0
    """
    time.sleep(0.1)
    return 0


def test_run_in_parallel():
    """Test running in parallel."""
    task_list = []
    for x in range(40):
        task_list.append(my_task(x))

    run_in_parallel(task_list, n_workers=1)
    run_in_parallel(task_list, n_workers=4)

    run_in_parallel(task_list, n_workers=2, backend='loky')
    run_in_parallel(task_list, n_workers=2, backend='dask')


if __name__ == '__main__':
    run_in_parallel()
