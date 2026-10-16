"""Remote, atomic snapshots of actual worker and driver state."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'Snapshot already exists: '+str(path)
    temp=path.with_name(path.name+'.partial')
    with temp.open('xb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.rename(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def driver_rng():
    return dict(torch_cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())


def compact_stream_state(state):
    """Discard only a fully consumed source batch; retain partial batches exactly."""
    state = dict(state)
    pending = state['pending_source']
    offset = state['pending_offset']
    if pending is not None:
        assert 0 <= offset <= len(pending), 'Invalid source batch offset'
        if offset == len(pending):
            state.update(pending_source=None,pending_offset=0)
    else:
        assert offset == 0, 'Absent source batch has a nonzero offset'
    return state


def worker_rng(worker):
    from verl.third_party.vllm import vllm_version
    assert vllm_version=='0.6.3', 'Sampling snapshot requires the verified SPMD vLLM implementation'
    manager=worker.rollout_sharding_manager
    engine=worker.rollout.inference_engine.llm_engine
    assert not engine.has_unfinished_requests(), 'Cannot snapshot a busy sampling engine as a completed boundary'
    runner=engine.model_executor.worker.model_runner
    generators=runner.get_generators()
    assert isinstance(generators,dict)
    return dict(**driver_rng(),torch_cuda=torch.cuda.get_rng_state(),cuda_device=torch.cuda.current_device(),
                actor_cuda_saved=cpu_tree(manager.torch_random_states),
                rollout_cuda_saved=cpu_tree(manager.gen_random_states),
                request_generators={k:dict(device=str(v.device),state=v.get_state().cpu()) for k,v in generators.items()},
                vllm_version=vllm_version,boundary='idle synchronous generation engine')


def capture_worker(worker,directory,checkpoint=False):
    from reproduction.a800.owned_processes import identity
    base=Path(os.environ['SEARCH_G1_BASE']);run=json.loads((base/'logs/run-manifest.json').read_text())
    directory=Path(directory).resolve();assert directory.is_relative_to(base.resolve())
    state=dict(run_id=run['run_id'],source_commit=run['source_commit'],rank=worker.rank,
               world_size=torch.distributed.get_world_size(),process_identity=identity(os.getpid()),
               rng=worker_rng(worker),actual_optimizer_writes=getattr(worker.actor,'_optimizer_writes',0),
               completed_outer=getattr(worker.actor,'_complete_outers',0),
               full_budget_outers=getattr(worker.actor,'_full_budget_outers',0),
               optimizer_calls=getattr(worker.actor,'_optimizer_calls',getattr(worker.actor,'_optimizer_writes',0)),
               optimizer_rollbacks=getattr(worker.actor,'_optimizer_rollbacks',0),
               last_attempt_index=getattr(worker.actor,'_last_attempt_index',0),
               used_attempt_ids=sorted(getattr(worker.actor,'_used_attempt_ids',set())),
               policy_rejections=getattr(worker.actor,'_policy_rejections',0))
    if run.get('data_selection_mode')=='prefix20_probe':
        state['actual_adam_steps']=[float(value['step']) for value in worker.actor_optimizer.state.values()
                                    if 'step' in value]
        state['actual_scheduler_last_epoch']=worker.actor_lr_scheduler.last_epoch
    if checkpoint:
        state.update(optimizer=cpu_tree(worker.actor_optimizer.state_dict()),
                     scheduler=cpu_tree(worker.actor_lr_scheduler.state_dict()),
                     optimizer_layout='rank-local FSDP flat-parameter Adam; exact same rank/world-size shard layout required')
    proof=atomic_save(directory/('rank-'+str(worker.rank)+'.pt'),state)
    proof.update(rank=worker.rank,actual_optimizer_writes=state['actual_optimizer_writes'],run_id=run['run_id'])
    return proof
