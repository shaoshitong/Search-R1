"""Remote, atomic snapshots of actual worker and driver state."""
import hashlib
import copy
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


def capture_easy2560_driver(trainer, run, *, live_probe=None, callbacks_complete=False):
    """Capture the actual Easy stream and driver, without a selection head.

    Checkpoints at50/100/150 precede the grounding refit at that boundary.
    Their saved grounding version is intentional; recovery must finish that
    pending refit before beginning the next outer. A rolling post-callback
    sidecar instead captures the adopted probe and the next-outer RNG boundary.
    Final200 has no refit.
    """
    from reproduction.common8.precision_contract import easy2560_native4
    assert easy2560_native4(run)
    assert getattr(trainer, 'qprobe_runtime', None) is None
    if live_probe is None:
        live_probe = json.loads((Path(os.environ['SEARCH_G1_BASE'])/
                                 'artifacts/live/manifest.json').read_text())
    step = trainer.global_steps
    assert type(step) is int and 1 <= step <= 200
    assert live_probe['run_id'] == run['run_id']
    assert live_probe['version'] == trainer.reward_fn.expected_probe_version
    assert type(live_probe['round']) is int and 0 <= live_probe['round'] <= step
    progress = trainer.inline_attempt_progress
    attempt = {name: getattr(progress, name) for name in
               ('completed_outer', 'cumulative_writes', 'attempt_index',
                'aborted_attempts', 'full_budget_outer')}
    assert attempt['completed_outer'] == step
    stream = copy.deepcopy(compact_stream_state(trainer.mixed_stream.state()))
    # Same canonical membership as Co; reusing the stream type does not imply
    # a Prefix20 model, selector, feature extraction or warmup.
    assert stream['prefix20_pool_binding']['eligible_pool_sha256']
    return dict(schema_version=1, selection_mode='mixed_first', easy2560_selection=True,
        run_id=run['run_id'], source_commit=run['source_commit'], completed_outer=step,
        candidate_stream=stream, rng=driver_rng(),
        probe_version=live_probe['version'], probe_round=live_probe['round'],
        reward_refresh_pending=step in (50, 100, 150) and not callbacks_complete,
        callbacks_complete=bool(callbacks_complete), question_probe=None,
        attempt_progress=copy.deepcopy(attempt))


def restore_easy2560_driver(trainer, saved, run, *, source_run_id, live_probe,
                           stream_factory=None):
    """Restore and read back driver/stream state from an authenticated snapshot.

    The caller authenticates the checkpoint/file SHA and restores model/Adam/
    worker state separately. This function consumes no samples, fits no probe,
    and performs no training. A pending grounding refit is returned explicitly.
    """
    from reproduction.common8.precision_contract import easy2560_native4
    from search_g1.full_state_restore import equal_tree
    assert easy2560_native4(run)
    assert saved['schema_version'] == 1 and saved['easy2560_selection'] is True
    assert saved['selection_mode'] == 'mixed_first' and saved['question_probe'] is None
    assert getattr(trainer, 'qprobe_runtime', None) is None
    assert source_run_id == saved['run_id'] == run['run_id'] == live_probe['run_id']
    step = saved['completed_outer']
    assert type(step) is int and 1 <= step <= 200
    assert saved['probe_version'] == live_probe['version']
    assert saved['probe_round'] == live_probe['round']
    assert type(saved['reward_refresh_pending']) is bool
    assert saved['reward_refresh_pending'] == (step in (50, 100, 150) and
                                              not saved.get('callbacks_complete', False))
    progress = saved['attempt_progress']
    assert progress['completed_outer'] == step
    assert all(type(progress[name]) is int and progress[name] >= 0 for name in
               ('completed_outer', 'cumulative_writes', 'attempt_index', 'aborted_attempts', 'full_budget_outer'))
    assert progress['attempt_index'] >= step and progress['full_budget_outer'] <= step
    assert step <= progress['cumulative_writes'] <= step * 20
    assert trainer.reward_fn.no_self is True, 'Uncheckpointed adaptive reward cannot be resumed'
    managers = [trainer.reward_fn]
    if getattr(trainer, 'val_reward_fn', None) is not None:
        managers.append(trainer.val_reward_fn)
    for manager in managers:
        assert getattr(manager, 'expected_probe_version', None) in (None, saved['probe_version'])
    candidate = copy.deepcopy(compact_stream_state(saved['candidate_stream']))
    if stream_factory is None:
        from search_g1.prefix20_data import Prefix20CandidateStream
        pool_path = run['selection_eligible_pool_path']
        stream = Prefix20CandidateStream(trainer.train_dataloader, pool_path,
                                         restored_state=candidate)
    else:
        stream = stream_factory(trainer.train_dataloader, restored_state=candidate)
    equal_tree(compact_stream_state(stream.state()), candidate)
    torch.set_rng_state(saved['rng']['torch_cpu'])
    np.random.set_state(saved['rng']['numpy'])
    random.setstate(saved['rng']['python'])
    equal_tree(driver_rng(), saved['rng'])
    trainer.mixed_stream = stream
    trainer._restored_candidate_state = candidate
    trainer._restored_driver_rng = copy.deepcopy(saved['rng'])
    trainer._restored_attempt_progress = copy.deepcopy(progress)
    trainer._restored_reward_refresh_pending = saved['reward_refresh_pending']
    for manager in managers:
        manager.expected_probe_version = saved['probe_version']
    return dict(completed_outer=step, actual_driver_rng_readback_equal=True,
        actual_candidate_stream_readback_equal=True,
        actual_candidate_source_cursor=candidate['consumed_source_rows'],
        candidate_order_count=len(candidate['candidate_order']),
        pending_offset=candidate['pending_offset'], actual_probe_version=saved['probe_version'],
        actual_probe_round=saved['probe_round'], reward_refresh_pending=saved['reward_refresh_pending'],
        attempt_progress=copy.deepcopy(progress), question_probe=None)


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
    from reproduction.common8.precision_contract import easy2560_native4
    if run.get('data_selection_mode')=='prefix20_probe' or easy2560_native4(run):
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
