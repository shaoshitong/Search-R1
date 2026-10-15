"""Run all saved-point reloads after training closure, inside one native allocation.

No submission, preparation of training, or evaluation is performed here. The
pinned b88 controller supplies already-reviewed terminal/native/UUID ownership
checks. Science/runtime imports remain this c5-derived isolated source tree.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from reproduction.restore_only.contract import ROLE,STEPS,read,sha,require,validate_binding

B88='b88c7156202abfa7968aa8f59012297ccd52a858'
B88_ROOT=Path('/mnt/data/shaoshitong/search-g1-compatible-eval-b88c715-20261015')


def publish(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def execution_controls(binding):
    """Use only verified execution guards; never its evaluation pipeline runner."""
    proof=B88_ROOT/'publication-proof.json'
    require(sha(proof)==binding['execution_control_proof_sha256'],'EXECUTION_CONTROL_PROOF_CHANGED')
    document=read(proof);require(document['published_control_commit']==B88 and document['all_blobs_full_sha_match'] is True,'UNPUBLISHED_EXECUTION_CONTROL')
    for name,digest in document['files'].items():
        # Git code only. No model/data/checkpoint hashes in this execution proof.
        require(sha(B88_ROOT/'control'/name)==digest,'EXECUTION_CONTROL_SOURCE_CHANGED')
    import reproduction
    reproduction.__path__.append(str(B88_ROOT/'control/reproduction'))
    submit=importlib.import_module('reproduction.compatible_async_evaluation.submit')
    pipeline=importlib.import_module('reproduction.compatible_async_evaluation.pipeline')
    from reproduction.a800.owned_processes import same
    from reproduction.common8.native_allocation import allocated_indices,cuda_uuids,fields
    from reproduction.common8.stage_retrieval import memory_bytes
    pipeline.same=same;pipeline.cuda_uuids=cuda_uuids;pipeline.fields=fields
    pipeline.allocated_indices=allocated_indices;pipeline.memory_bytes=memory_bytes;pipeline.publish=publish
    return submit,pipeline


def wait_owned_release(owned):
    from reproduction.a800.owned_processes import same,identity
    births=[r for job in owned.jobs.values() for r in job['ledger'].values()]
    deadline=time.monotonic()+60
    while True:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        live=[r for r in births if same(r)]
        contexts=set()
        for pid in {str(r['pid']) for r in births}&set(apps.split()):
            try:current=identity(int(pid))
            except (FileNotFoundError,ProcessLookupError):contexts.add(pid);continue
            if any(str(r['pid'])==pid and all(current[k]==r[k] for k in ('start_ticks','sid','uid')) for r in births):contexts.add(pid)
        if not live and not contexts:return dict(owned_births_released=True,owned_gpu_contexts_released=True)
        require(time.monotonic()<deadline,'OWNED_RESOURCES_NOT_RELEASED')
        time.sleep(.5)


class DeferredInterrupt:
    """Never interrupt Popen -> actual birth/ledger registration."""
    def __init__(self):self.requested=None
    def handler(self,sig,frame):self.requested=sig
    def check(self):
        if self.requested is not None:raise KeyboardInterrupt('restore-only native signal '+str(self.requested))


def start_owned(interrupt,owned,name,args,env,log,cwd):
    # OwnedProcesses.start records its leader birth/ledger before returning.
    process=owned.start(name,args,env,log,cwd)
    owned.observe(name)
    interrupt.check()
    return process


def child(binding_path,stepbase,step):
    binding=validate_binding(read(binding_path))
    require(step in STEPS,'INVALID_STEP')
    require(stepbase==Path(binding['base']).resolve()/('step-'+str(step)),'STEP_NAMESPACE_CHANGED')
    require(os.environ['SEARCH_G1_BASE']==str(stepbase),'WORKER_BASE_CHANGED')
    from reproduction.a800.owned_processes import same
    intent=read(Path(binding['base'])/'logs/restore-only-intent.json')
    native=read(Path(binding['base'])/'logs/compatible-native-allocation.json')
    require(intent['binding_sha256']==sha(binding_path) and same(intent['coordinator_birth']) and
        intent['coordinator_birth']['pid']==os.getppid(),'CHILD_REQUIRES_LIVE_BOUND_COORDINATOR')
    require(native['accepted'] is True and native['slurm_job_id']==os.environ['SLURM_JOB_ID'] and
        native['actual_cuda_runtime_uuids']==binding['physical_gpu_uuids'],'CHILD_ALLOCATION_CHANGED')
    require(not Path('/tmp/g1ro-'+binding['execution_id'][:12]+'-'+str(step)).exists(),'RAY_NAMESPACE_OCCUPIED')
    import ray
    # Ray must be private; never attach to existing training/evaluation services.
    ray.init(address='local',num_cpus=96,num_gpus=8,
        namespace='g1restore-'+binding['execution_id'][:12]+'-'+str(step),
        _temp_dir='/tmp/g1ro-'+binding['execution_id'][:12]+'-'+str(step),include_dashboard=False,
        runtime_env={'env_vars':{'SEARCH_G1_BASE':str(stepbase),'SEARCH_G1_PROFILE':'common8',
            'PYTHONPATH':str(Path(__file__).resolve().parents[2])}})
    try:
        from reproduction.restore_only.runtime import run_step
        run_step(binding,stepbase,Path(binding['receipts'][str(step)]['path']))
    finally:ray.shutdown()


def run(binding_path):
    binding=validate_binding(read(binding_path));base=Path(binding['base']).resolve()
    require(socket.gethostname().split('.')[0]=='h100-8' and os.environ.get('SLURM_JOB_ID'),'NATIVE_H1008_REQUIRED')
    require(re.fullmatch('[0-9a-f]{32}',binding['execution_id']) is not None,'PRIVATE_EXECUTION_ID_REQUIRED')
    require(not base.exists(),'ONCE_NAMESPACE_OCCUPIED_INSPECT_NO_REPLAY')
    require(all(not Path('/tmp/g1ro-'+binding['execution_id'][:12]+'-'+str(step)).exists() for step in STEPS),'RAY_NAMESPACE_OCCUPIED')
    submit,pipeline=execution_controls(binding)
    # Must authenticate true completed200, native terminal, mandatory cleanup,
    # original birth/cgroup/GPU release BEFORE any new CUDA context is created.
    terminal=submit.trainer_guard(binding['trainer_terminal_proof'],binding['trainer_terminal_proof_sha256'])
    base.mkdir();(base/'logs').mkdir()
    from reproduction.a800.owned_processes import identity
    publish(base/'logs/restore-only-intent.json',dict(role=ROLE,binding_sha256=sha(binding_path),trainer_release=terminal,coordinator_birth=identity(os.getpid())))
    manifest=dict(binding,checkpoint_step=200,split='restore_only',recipe_sha256=binding['resolved_config_sha256'])
    native=pipeline.native_admission(base,manifest)
    native['training_world_size']=8
    lease=pipeline.UUIDLocks(manifest)
    from reproduction.a800.owned_processes import OwnedProcesses
    owned=OwnedProcesses(base/'logs/process-ledger.json',binding['run_id'])
    results=[];success=False
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    interrupt=DeferredInterrupt()
    for sig in previous:signal.signal(sig,interrupt.handler)
    try:
        for step in STEPS:
            interrupt.check()
            stepbase=base/('step-'+str(step));stepbase.mkdir()
            for name in ('logs','artifacts/live','artifacts/question-probe'): (stepbase/name).mkdir(parents=True)
            publish(stepbase/'logs/native-allocation-admission.json',native)
            env=os.environ.copy();env.update(SEARCH_G1_BASE=str(stepbase),SEARCH_G1_PROFILE='common8',
                PYTHONPATH=str(Path(__file__).resolve().parents[2]),RAY_ADDRESS='')
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
            process=start_owned(interrupt,owned,'restore-'+str(step),[sys.executable,'-m','reproduction.restore_only.controller',
                '--binding',str(binding_path),'--child-step',str(step),'--step-base',str(stepbase)],
                env,str(stepbase/'logs/console.log'),str(Path(__file__).resolve().parents[2]))
            while process.poll() is None:
                owned.observe('restore-'+str(step));interrupt.check();time.sleep(.25)
            owned.observe('restore-'+str(step));interrupt.check()
            require(process.returncode==0,'RESTORE_STEP_EXIT_NONZERO_'+str(step))
            result=read(stepbase/'logs/restore-only-result.json')
            require(result['step']==step and result['actual_cuda_full_state_restore_accepted'] is True and
                result['training_updates_executed']==0 and result['generation_calls']==0 and
                result['next_outer_executed'] is False,'INVALID_ACTUAL_RESTORE_RESULT')
            owned.stop('restore-'+str(step));released=wait_owned_release(owned)
            results.append(dict(step=step,result_path=str(stepbase/'logs/restore-only-result.json'),
                result_sha256=sha(stepbase/'logs/restore-only-result.json'),**released))
        success=True
    finally:
        try:
            owned.stop_all();release=wait_owned_release(owned)
            publish(base/'logs/cleanup-status.json',dict(role=ROLE,cleaned=True,**release))
        finally:
            lease.close()
            for sig,handler in previous.items():signal.signal(sig,handler)
    if success:
        publish(base/'logs/restore-only-complete.json',dict(role=ROLE,run_id=binding['run_id'],
            results=results,steps=list(STEPS),actual_cuda_full_state_restore_accepted=True,
            actor_updates_executed=0,quality_evaluations_executed=0,
            native_actual_exit0_verified=False,native_cgroup_release_verified=False,
            post_exit_observer_required=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--child-step',type=int);p.add_argument('--step-base',type=Path);a=p.parse_args()
    if a.child_step is None:run(a.binding.resolve())
    else:
        require(a.step_base is not None,'STEP_BASE_REQUIRED')
        child(a.binding.resolve(),a.step_base.resolve(),a.child_step)

if __name__=='__main__':main()
