"""Catch CPU Ray-driver CUDA visibility errors before large retrieval staging."""
import json
import os
from pathlib import Path
import time


def main():
    from reproduction.hpc3_dynrpo_periodic.ray_threads import init_options
    thread_options=init_options()
    import ray
    from reproduction.hpc3_dynrpo_periodic.ray_driver import driver_import_smoke
    from reproduction.a800.owned_processes import publish
    b=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((b/'logs/run-manifest.json').read_text())
    allocation=json.loads((b/'logs/native-allocation-admission.json').read_text())
    assert allocation['accepted'] and allocation['run_id']==run['run_id']
    assert allocation['slurm_job_id']==os.environ['SLURM_JOB_ID']
    root=Path('/tmp')/('g1d-'+run['run_id'][:12]);root.mkdir(mode=0o700,exist_ok=False)
    ray.init(address='local',num_cpus=16,num_gpus=4,object_store_memory=8*1024**3,
             include_dashboard=False,_temp_dir=str(root),**thread_options)
    try:
        proof=ray.get(driver_import_smoke.remote(allocation['actual_cuda_runtime_uuids'][0]))
        publish(b/'logs/periodic-driver-import-smoke.json',dict(accepted=True,
            run_id=run['run_id'],slurm_job_id=os.environ['SLURM_JOB_ID'],
            source_commit=run['source_commit'],checked_unix=time.time(),**proof))
        print('PERIODIC_CPU_RAY_DRIVER_ACTUAL_CUDA_IMPORT_PASS',flush=True)
    finally:ray.shutdown()


if __name__=='__main__':main()
