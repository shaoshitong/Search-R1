"""Own isolated CPU-only Ray regression: trigger the actual LocalGC RPC path.

Run only inside a CPU Slurm allocation, never on the login node. Starts and
shuts down its private Ray instance only; no training assets or existing Ray
instance are attached. Retains small logs remotely under its private /tmp root.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import re
import time
import uuid

from .ray_threads import init_options,verify_environment
from .common import DEFAULT_BASE


def snapshot():
    return dict(pid=os.getpid(),uid=os.getuid(),threads=len(list(Path('/proc/self/task').iterdir())),
                rpc_threads=verify_environment())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    base=args.base
    if (base.resolve()!=base or base.is_symlink() or not base.is_dir() or base.stat().st_uid!=os.getuid() or
            re.fullmatch(re.escape(str(DEFAULT_BASE))+r'-r[1-9][0-9]*',str(base)) is None):
        raise ValueError('Fixture requires this task own remote recovery namespace')
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal() or not Path('/proc/self/task').is_dir():
        raise ValueError('Real CPU Slurm allocation required; never start this Ray fixture on login')
    scratch=base/'tmp'/('ray-thread-smoke-'+uuid.uuid4().hex[:8]);scratch.mkdir(mode=0o700,exist_ok=False)
    os.environ.update(TMPDIR=str(scratch),XDG_CACHE_HOME=str(scratch/'cache'),PYTHONDONTWRITEBYTECODE='1')
    options=init_options();os.environ['CUDA_VISIBLE_DEVICES']=''
    import ray
    if ray.__version__!='2.40.0' or ray.is_initialized():
        raise ValueError('Fixture requires actual Ray2.40 and an unattached fresh driver')
    root=Path('/tmp')/('g1t-'+str(os.getuid())+'-'+job+'-'+uuid.uuid4().hex[:8])
    root.mkdir(mode=0o700,exist_ok=False)
    # Only the fixture disables GC cooldown so three real requests complete
    # promptly. Production changes ONLY the three RPC thread pool sizes.
    options['_system_config'].update(global_gc_min_interval_s=0,local_gc_min_interval_s=0)
    ray.init(address='local',num_cpus=2,num_gpus=0,object_store_memory=128*1024**2,
             _memory=512*1024**2,include_dashboard=False,_temp_dir=str(root),**options)
    result=None
    try:
        @ray.remote(num_cpus=.5,max_restarts=0,max_task_retries=0)
        class GCWitness:
            def __init__(self):
                self.collections=0
                def callback(phase,info):
                    if phase=='stop' and info.get('generation')==2:self.collections+=1
                self.callback=callback;gc.callbacks.append(callback)
            def report(self):
                return dict(snapshot(),actual_generation2_collections=self.collections)
        actors=[GCWitness.remote() for _ in range(2)]
        before=ray.get([actor.report.remote() for actor in actors],timeout=30)
        previous=before;rounds=[]
        from ray._private.internal_api import global_gc
        for number in range(1,4):
            global_gc();deadline=time.monotonic()+15
            while True:
                current=ray.get([actor.report.remote() for actor in actors],timeout=10)
                if all(new['actual_generation2_collections']>old['actual_generation2_collections']
                       for old,new in zip(previous,current)):break
                if time.monotonic()>=deadline:
                    raise RuntimeError('Actual LocalGC callbacks did not reach every fixture worker')
                time.sleep(.1)
            rounds.append(dict(round=number,workers=current));previous=current
        deltas=[max(row['workers'][rank]['threads'] for row in rounds)-min(row['workers'][rank]['threads'] for row in rounds)
                for rank in range(2)]
        if any(delta>4 for delta in deltas):
            raise RuntimeError('Worker threads expanded across the repeated actual GC requests: '+repr(deltas))
        config=ray._private.worker.global_worker.node._config
        for key,value in verify_environment().items():
            if config.get(key)!=value:raise ValueError('Actual initialized Ray config differs: '+key)
        result=dict(status='ACTUAL_RAY240_RPC_BOUNDS_AND_LOCALGC_PASS',checked_unix=time.time(),
            slurm_job_id=job,hostname=socket.gethostname(),ray_version=ray.__version__,base=str(base),scratch=str(scratch),
            ray_root=str(root),driver=snapshot(),workers_before=before,workers_after=previous,
            actual_gc_rounds=rounds,actual_cluster_system_config={key:config[key] for key in verify_environment()},
            num_gpus=0,num_cpus=2,private_instance_only=True,training_or_model_updates=0,
            gc_round_thread_count_ranges=deltas,max_allowed_gc_round_thread_growth=4,
            repeated_gc_thread_counts_bounded=True)
    finally:ray.shutdown()
    result['owned_ray_shutdown_complete']=not ray.is_initialized()
    if args.output:
        if not args.output.is_absolute() or not args.output.parent.is_dir() or not args.output.resolve().is_relative_to(base):
            raise ValueError('Small fixture receipt requires an existing absolute output directory')
        with args.output.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
