from aa_uv.parallelize import run_in_parallel, task
import time

@task
def my_task(x):
    time.sleep(0.1)
    return 0

def test_run_in_parallel():
    task_list = []
    for x in range(40):
        task_list.append(my_task(x))

    run_in_parallel(task_list, n_workers=1)
    run_in_parallel(task_list, n_workers=4)

    run_in_parallel(task_list, n_workers=2, backend='loky')
    run_in_parallel(task_list, n_workers=2, backend='dask')

if __name__ == "__main__":
    run_in_parallel()