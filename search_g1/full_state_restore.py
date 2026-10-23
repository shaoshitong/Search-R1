"""Explicit restoration of this experiment's independently validated full state.

The original fresh-experiment contract remains the scientific origin. A resumed
allocation restores actual counters and state; it never substitutes fresh Adam
or treats unavailable driver metrics as reconstructed measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

FINAL_EASY_STATUS = 'EASY2560_FINAL200_CHAINED_FULL_SHA_CPU_STATES_PASS'
FINAL_EASY_KIND = 'easy2560_final200_restore_only'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source(receipt_path):
    p=Path(receipt_path).resolve();r=json.loads(p.read_text())
    if r.get('kind') == 'dynrpo7b_native4':
        from reproduction.dynrpo7b.restore import source as dynrpo_source
        return dynrpo_source(p)
    if r.get('status')==FINAL_EASY_STATUS:
        from search_g1.easy2560_final_checkpoint import source as final_easy_source
        return final_easy_source(p)
    from search_g1.easy2560_restore import STATUS as EASY_STATUS, source as easy_source
    if r.get('status')==EASY_STATUS:return easy_source(p)
    from search_g1.qprobe_checkpoint import PAIR_STATUS,read_source
    if r.get('status')==PAIR_STATUS:return read_source(p)
    assert r['status']=='SOURCE_CHECKPOINT5_FULL_SHA_AND_CPU_STATES_PASS'
    assert r['step']==5 and r['cpu_regression_tests']==3
    assert len(r['ranks'])==8 and sorted(v['rank'] for v in r['ranks'])==list(range(8))
    assert all(v['actual_optimizer_writes']==100 and v['scheduler_last_epoch']==5 and v['optimizer_tensors_finite'] for v in r['ranks'])
    assert r['driver_state']['completed_outer']==5 and r['driver_state']['source_consumed']==10240
    assert r['hf_saved_tensor_count']==435 and len(r['files'])==23
    checkpoint=Path(next(v['path'] for v in r['files'] if v['relative_path']=='complete.json')).parent
    origin=checkpoint.parents[3];run=json.loads((origin/'logs/run-manifest.json').read_text())
    assert run['run_id']==r['run_id'] and run['source_commit']=='2de313e3e455e005e0d7db83104942bad8bd977a'
    assert run['actor_micro_batch_per_rank']==2 and run['vllm_gpu_memory_utilization']==0.1
    assert {v['relative_path'] for v in r['files']}=={str(f.relative_to(checkpoint)) for f in checkpoint.rglob('*') if f.is_file()}
    for v in r['files']:
        f=Path(v['path']);st=f.stat()
        assert f.resolve().is_relative_to(checkpoint.resolve()) and not f.is_symlink()
        assert (st.st_ino,st.st_size,st.st_mtime_ns)==(v['inode'],v['bytes'],v['mtime_ns'])
    return p,r,checkpoint,origin,run


def execution_budget_compatible(run, parent):
    from reproduction.common8.precision_contract import NATIVE_BF16_PROFILE, bf16_policy
    if run.get('precision_profile') != NATIVE_BF16_PROFILE:
        assert run['budget'] == parent['budget']
        return
    assert bf16_policy(run)
    allowed = {'compute_dtype', 'rollout_dtype', 'allow_tf32'}
    assert run['budget'].keys() == parent['budget'].keys()
    assert all(run['budget'][k] == parent['budget'][k] for k in run['budget'] if k not in allowed)
    assert run['budget']['compute_dtype'] == run['budget']['rollout_dtype'] == 'bfloat16'
    assert run['budget']['allow_tf32'] is None
    run['execution_transition'] = dict(source_precision_profile=parent.get('precision_profile'),
        target_precision_profile=NATIVE_BF16_PROFILE, master_optimizer_reduction_dtype='float32',
        bitwise_continuation=False, scientific_budget_unchanged=True,
        discarded_unpersisted_old_suffix='old3368 outers6-10 and partial11 are not stitched into resumed counters')


def bind(run,receipt_path):
    p,r,checkpoint,origin,parent=source(receipt_path)
    if r.get('kind') == 'dynrpo7b_native4':
        from reproduction.dynrpo7b.restore import bind as dynrpo_bind
        return dynrpo_bind(run, receipt_path)
    if r.get('status')==FINAL_EASY_STATUS:
        raise ValueError('EASY2560_FINAL200_RESTORE_ONLY_NOT_A_TRAINING_BIND')
    from search_g1.easy2560_restore import STATUS as EASY_STATUS, bind as easy_bind
    if r.get('status')==EASY_STATUS:return easy_bind(run,receipt_path)
    from search_g1.qprobe_checkpoint import PAIR_STATUS,bind_restore
    if r.get('status')==PAIR_STATUS:return bind_restore(run,receipt_path)
    execution_budget_compatible(run, parent)
    assert run['input_sha256']==parent['input_sha256']
    assert run['initial_model_sha256']==parent['initial_model_sha256']
    assert run['actor_micro_batch_per_rank']==parent['actor_micro_batch_per_rank']==2
    run.update(run_id=parent['run_id'],native_ray_execution_id=uuid.uuid4().hex,
        pretraining_baseline_policy='reuse_historical',
        full_state_restore=dict(source_receipt=str(p),source_receipt_sha256=digest(p),source_checkpoint=str(checkpoint),
            source_base=str(origin),source_run_manifest_sha256=digest(origin/'logs/run-manifest.json'),
            completed_outer=5,actual_optimizer_writes_per_rank=100,candidate_source_cursor=10240,
            followup_acceptance_outer=10,next_outer_after_followup=11),
        optimizer_semantics='actual full checkpoint5 HF/Adam100/scheduler5/RNG/candidate cursor10240 restoration; no fresh optimizer or replayed prefix')
    return run


def active(run=None):
    if run is None:
        base=Path(os.environ['SEARCH_G1_BASE']);run=json.loads((base/'logs/run-manifest.json').read_text())
    return run.get('full_state_restore')


def from_config(config):
    receipt=config.trainer.get('full_state_restore_receipt',None)
    if not receipt:return None
    p,r,checkpoint,origin,parent=source(receipt)
    if r.get('kind') == 'dynrpo7b_native4':
        from reproduction.dynrpo7b.restore import from_config as dynrpo_from_config
        return dynrpo_from_config(config)
    if r.get('status')==FINAL_EASY_STATUS:
        raise ValueError('EASY2560_FINAL200_RESTORE_ONLY_NOT_A_TRAINING_BIND')
    assert Path(config.actor_rollout_ref.model.actor_checkpoint_path).resolve()==checkpoint
    assert config.trainer.total_training_steps==201 and config.trainer.acceptance_step==5
    from search_g1.qprobe_checkpoint import PAIR_STATUS
    from search_g1.easy2560_restore import STATUS as EASY_STATUS
    if r.get('status')==EASY_STATUS:
        from reproduction.common8.precision_contract import easy2560_native4
        from search_g1.easy2560_restore import physical_micro_transition
        assert easy2560_native4(parent) and r['world_size']==4
        global_micro=config.actor_rollout_ref.actor.ppo_micro_batch_size
        assert type(global_micro) is int and global_micro % 4 == 0
        physical_micro_transition(parent,global_micro//4)
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size==128
        assert config.actor_rollout_ref.actor.ppo_logical_micro_batch_size==4
        assert config.data.selection_mode=='mixed_first' and not config.algorithm.all_correct_virtual_zero
    elif r.get('status')==PAIR_STATUS:
        from reproduction.common8.topology import world_size
        assert world_size(parent)==r['world_size']
        assert config.actor_rollout_ref.actor.ppo_micro_batch_size==parent['actor_micro_batch_per_rank']*r['world_size']
        prefix20=parent['question_probe'].get('probe_architecture')=='qwen2_prefix20_trajectory_em_v1'
        assert config.data.get('selection_mode',None)==('prefix20_probe' if prefix20 else 'question_probe')
    else:assert config.actor_rollout_ref.actor.ppo_micro_batch_size==16
    return r


def equal_tree(a,b):
    import torch,numpy as np
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    elif all(hasattr(a,key) for key in ('batch','non_tensor_batch','meta_info')):
        assert type(a) is type(b)
        assert set(a.batch.keys())==set(b.batch.keys())
        for key in a.batch.keys():equal_tree(a.batch[key],b.batch[key])
        equal_tree(a.non_tensor_batch,b.non_tensor_batch);equal_tree(a.meta_info,b.meta_info)
    else:assert a==b


def publish(path,value):
    path=Path(path)
    with path.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def final_collective_inputs(load):
    """A failed local final-state read must not strand peers in model gather."""
    import torch.distributed as dist
    value=None;failure=None
    try:value=load()
    except Exception as error:failure=type(error).__name__+': '+str(error)[:512]
    failures=[None]*dist.get_world_size()
    dist.all_gather_object(failures,failure)
    assert not any(failures),'EASY2560_FINAL200_COLLECTIVE_STATE_INPUTS '+repr(failures)
    return value


def restore_worker(worker,receipt_path):
    if json.loads(Path(receipt_path).read_text()).get('kind') == 'dynrpo7b_native4':
        from reproduction.dynrpo7b.restore import restore_worker as dynrpo_restore_worker
        return dynrpo_restore_worker(worker, receipt_path)
    # In-memory diagnostics are never part of checkpoint state or restorable.
    worker.actor._invalidate_monitor_forward_cache(model_changed=True)
    import torch,numpy as np
    from safetensors import safe_open
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad,offload_fsdp_param_and_grad
    from search_g1.training_state import worker_rng,cpu_tree
    p,r,checkpoint,origin,parent=source(receipt_path);rank=worker.rank
    from search_g1.qprobe_checkpoint import PAIR_STATUS,validate_rank_state
    paired=r.get('status')==PAIR_STATUS
    from search_g1.easy2560_restore import STATUS as EASY_STATUS
    final_easy=r.get('status')==FINAL_EASY_STATUS
    easy=r.get('status')==EASY_STATUS or final_easy
    stateful=paired or easy
    size=r['world_size'] if stateful else 8
    outer=r['step'] if stateful else 5
    writes=r['attempt_progress']['cumulative_writes'] if stateful else 100
    expected_tensors=r['hf_saved_tensor_count'] if stateful else 435
    assert torch.distributed.get_world_size()==size
    def load_saved():
        proof=next(v for v in r['files'] if v['relative_path']==f'training_state/rank-{rank}.pt')
        assert digest(proof['path'])==proof['full_read_sha256']
        saved=torch.load(proof['path'],map_location='cpu',weights_only=False)
        assert saved['rank']==rank and saved['world_size']==size and saved['actual_optimizer_writes']==writes
        if stateful:validate_rank_state(saved,rank,size,outer,expected_writes=writes)
        return proof,saved
    proof,saved=final_collective_inputs(load_saved) if final_easy else load_saved()
    # Compare all actual FSDP-restored model tensors with saved HF tensors.
    if worker._is_offload_param:
        load_fsdp_param_and_grad(module=worker.actor_module_fsdp,device_id=torch.cuda.current_device(),load_grad=False)
    with FSDP.state_dict_type(worker.actor_module_fsdp,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        actual_model=worker.actor_module_fsdp.state_dict()
    checked=0
    model_shas={}
    if rank==0:
        weightmap=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
        assert set(actual_model)==set(weightmap)
        for shard in sorted(set(weightmap.values())):
            if final_easy:
                model_proof=next(v for v in r['files'] if v['relative_path']==shard)
                model_shas[shard]=digest(checkpoint/shard)
                assert model_shas[shard]==model_proof['full_read_sha256']
            with safe_open(str(checkpoint/shard),framework='pt',device='cpu') as f:
                for name in f.keys():equal_tree(actual_model[name],f.get_tensor(name));checked+=1
        assert checked==expected_tensors
    del actual_model
    from reproduction.common8.precision_contract import remove_padding
    current_run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
    if remove_padding(current_run):
        from search_g1.varlen_execution import verify_execution_configuration
        verify_execution_configuration(worker.actor_module_fsdp, rank)
    if worker._is_offload_param:
        offload_fsdp_param_and_grad(module=worker.actor_module_fsdp,offload_grad=worker._is_offload_grad)
    worker.actor_optimizer.load_state_dict(saved['optimizer'])
    worker.actor_lr_scheduler.load_state_dict(saved['scheduler'])
    equal_tree(cpu_tree(worker.actor_optimizer.state_dict()),saved['optimizer'])
    equal_tree(cpu_tree(worker.actor_lr_scheduler.state_dict()),saved['scheduler'])
    worker.actor._optimizer_writes=writes;worker.actor._complete_outers=outer
    worker.actor._full_budget_outers=saved.get('full_budget_outers',outer if writes==outer*20 else 0)
    worker.actor._optimizer_calls=saved.get('optimizer_calls',writes)
    worker.actor._optimizer_rollbacks=saved.get('optimizer_rollbacks',0)
    worker.actor._last_attempt_index=saved['last_attempt_index'] if stateful else 5
    worker.actor._inline_initial_optimizer_published=True
    if stateful:
        # Counters/attempt identities come from this checkpoint only. Parent
        # logs may contain a later suffix and must never leak future attempts.
        worker.actor._used_attempt_ids=set(saved['used_attempt_ids'])
        worker.actor._policy_rejections=saved['policy_rejections']
    else:
        parent_attempts=[json.loads(line) for line in (origin/'logs/actual-attempts.jsonl').read_text().splitlines()]
        assert [v['attempt_index'] for v in parent_attempts]==list(range(1,6))
        worker.actor._used_attempt_ids={v['attempt_id'] for v in parent_attempts}
        parent_monitor=[v for v in map(json.loads,(origin/f'logs/optimizer-rank-{rank}.jsonl').read_text().splitlines()) if v['kind']=='POLICY_MONITOR']
        assert parent_monitor and parent_monitor[-1]['optimizer_writes']==100
        worker.actor._policy_rejections=parent_monitor[-1]['rejection_count']
    if paired:
        from search_g1.qprobe_checkpoint import boundary_worker_rng
        rng=boundary_worker_rng(r,rank,saved)
    elif easy and r.get('post_callback_rng_captured'):
        from search_g1.easy2560_restore import boundary_worker_rng
        rng=boundary_worker_rng(r,rank,saved)
    else:rng=saved['rng']
    manager=worker.rollout_sharding_manager
    engine=worker.rollout.inference_engine.llm_engine;assert not engine.has_unfinished_requests()
    runner=engine.model_executor.worker.model_runner;generators=runner.get_generators();generators.clear()
    for key,v in rng['request_generators'].items():
        gen=torch.Generator(device=v['device']);gen.set_state(v['state']);generators[key]=gen
    manager.torch_random_states=cpu_tree(rng['actor_cuda_saved']);manager.gen_random_states=cpu_tree(rng['rollout_cuda_saved'])
    torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state(rng['torch_cuda'],device=torch.cuda.current_device())
    np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    equal_tree(worker_rng(worker),rng)
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.native_allocation import cuda_uuids
    actual_uuids=cuda_uuids(count=torch.cuda.device_count())
    cuda_uuid=actual_uuids[torch.cuda.current_device()]
    allocation=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/native-allocation-admission.json').read_text())
    assert cuda_uuid in allocation['actual_cuda_runtime_uuids']
    row=dict(rank=rank,process_identity=identity(os.getpid()),actual_cuda_uuid=cuda_uuid,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),source_checkpoint=str(checkpoint),source_state_full_sha256=proof['full_read_sha256'],
        actual_optimizer_writes=writes,scheduler_last_epoch=outer,model_hf_exact_tensors_checked_on_rank0=checked,
        actual_adam_scheduler_readback_equal=True,actual_rng_readback_equal=True,world_size=size,
        last_attempt_index=worker.actor._last_attempt_index,
        policy_rejections=worker.actor._policy_rejections,
        source_run_id=saved['run_id'],source_step=outer,checked_unix=time.time())
    if paired:row['sampling_rng_source']='authenticated post-evaluation/refit boundary-worker-rng'
    if easy:row['sampling_rng_source']=r['sampling_rng_stage']
    if final_easy:
        row.update(kind=FINAL_EASY_KIND, restore_only=True, source_model_shards_sha256=model_shas,
                   sampling_rng_source=r['sampling_rng_stage'], post_callback_rng_captured=False)
    if easy:
        row.update(run_id=current_run['run_id'],source_receipt_sha256=digest(p),
            source_job=r['source_job'],source_commit=parent['source_commit'],
            actual_model_collective_readback_equal=True)
    publish(Path(os.environ['SEARCH_G1_BASE'])/'logs'/f'full-state-restored-rank-{rank}.json',row)
    return row


def bind_restored_probe(trainer,saved,run,live,source_run_id):
    """Initialize the reward manager's lazy binding before its first RPC.

    Real probe replies still require this exact version. Reward alpha is fixed
    in this experiment; historical EMA/call diagnostics were not checkpointed.
    """
    version=saved['probe_version']
    assert source_run_id==run['run_id']==live['run_id']
    if 'run_id' in saved:assert saved['run_id']==source_run_id
    assert version==run['initial_probe_version']==live['version']
    binding=run.get('full_state_restore') or {}
    if binding.get('kind')=='question_probe_pair':
        assert live['round']==binding['reward_probe_round'] and version==binding['reward_probe_version']
    else:assert live['round']==0
    assert trainer.reward_fn.no_self is True, 'Uncheckpointed adaptive reward cannot be resumed'
    managers=[trainer.reward_fn]
    if getattr(trainer,'val_reward_fn',None) is not None:managers.append(trainer.val_reward_fn)
    for manager in managers:
        previous=getattr(manager,'expected_probe_version',None)
        assert previous is None or previous==version, 'Reward manager bound to another probe'
    for manager in managers:manager.expected_probe_version=version
    return version


def restore_driver(trainer,receipt_path):
    if json.loads(Path(receipt_path).read_text()).get('kind') == 'dynrpo7b_native4':
        from reproduction.dynrpo7b.restore import restore_driver as dynrpo_restore_driver
        return dynrpo_restore_driver(trainer, receipt_path)
    if json.loads(Path(receipt_path).read_text()).get('status')==FINAL_EASY_STATUS:
        return restore_final_easy_driver(trainer,receipt_path)
    from search_g1.easy2560_restore import STATUS as EASY_STATUS,restore_driver as restore_easy_driver
    if json.loads(Path(receipt_path).read_text()).get('status')==EASY_STATUS:
        return restore_easy_driver(trainer,receipt_path)
    from search_g1.qprobe_checkpoint import PAIR_STATUS,restore_driver as restore_pair_driver
    if json.loads(Path(receipt_path).read_text()).get('status')==PAIR_STATUS:
        return restore_pair_driver(trainer,receipt_path)
    import torch,numpy as np
    from search_g1.training_state import driver_rng,compact_stream_state
    p,r,checkpoint,origin,parent=source(receipt_path)
    proof=next(v for v in r['files'] if v['relative_path']=='training_state/driver.pt')
    assert digest(proof['path'])==proof['full_read_sha256']
    saved=torch.load(proof['path'],map_location='cpu',weights_only=False)
    assert saved['completed_outer']==5 and saved['candidate_stream']['consumed_source_rows']==10240
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    live=json.loads((base/'artifacts/live/manifest.json').read_text())
    bind_restored_probe(trainer,saved,run,live,r['run_id'])
    torch.set_rng_state(saved['rng']['torch_cpu']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    equal_tree(driver_rng(),saved['rng'])
    trainer._restored_candidate_state=compact_stream_state(saved['candidate_stream'])
    assert trainer._restored_candidate_state['pending_source'] is None
    row=dict(source_checkpoint=str(checkpoint),actual_driver_rng_readback_equal=True,completed_outer=5,
             actual_candidate_source_cursor=10240,candidate_order_count=169028,pending_offset=0,
             actual_probe_version=saved['probe_version'],reward_alpha_fixed=True,
             historical_reward_diagnostics_restored=False,source_driver_full_sha256=proof['full_read_sha256'],checked_unix=time.time())
    publish(Path(os.environ['SEARCH_G1_BASE'])/'logs/full-state-restored-driver.json',row)
    return row


def restore_final_easy_driver(trainer,receipt_path):
    """Read back the saved outer200 Easy stream/RNG without drawing outer201."""
    import torch
    from search_g1.training_state import restore_easy2560_driver
    p,r,checkpoint,_,_=source(receipt_path)
    assert r['status']==FINAL_EASY_STATUS and r['step']==200 and r['restore_only'] is True
    base=Path(os.environ['SEARCH_G1_BASE']).resolve()
    run=json.loads((base/'logs/run-manifest.json').read_text())
    binding=run['full_state_restore']
    assert binding['kind']==FINAL_EASY_KIND and binding['source_receipt_sha256']==digest(p)
    proof=r['driver_file'];assert digest(proof['path'])==proof['full_read_sha256']
    saved=torch.load(proof['path'],map_location='cpu',weights_only=False)
    result=restore_easy2560_driver(trainer,saved,run,source_run_id=r['run_id'],
        live_probe=json.loads((base/'artifacts/live/manifest.json').read_text()))
    assert result['completed_outer']==200 and result['reward_refresh_pending'] is False
    result.update(schema_version=1,role='actual_easy2560_final200_restored_driver',kind=FINAL_EASY_KIND,
        run_id=run['run_id'],source_checkpoint=str(checkpoint),source_step=200,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),source_job=r['source_job'],source_commit=r['source_commit'],
        source_receipt_sha256=digest(p),source_driver_full_sha256=proof['full_read_sha256'],
        post_callback_rng_captured=False,sampling_rng_source=r['sampling_rng_stage'],
        actual_driver_dataequal=bool(result['actual_driver_rng_readback_equal'] and
            result['actual_candidate_stream_readback_equal']),restore_only=True,
        actual_cuda_restore_accepted=False,checked_unix=time.time())
    publish(base/'logs/full-state-restored-driver.json',result)
    return result


def prepare_guarded_restore(base,run,document,method,initial,receipts):
    """Prospective source-state admission, distinct from fresh0 smoke results."""
    from reproduction.common8.formal_inline_safety import ROLE,native_birth
    from reproduction.common8.formal_contract import FORMAL_SAFEGUARDS,FORMAL_COUNTING
    from reproduction.common8.policy_calibration_plan import document_sha256
    base=Path(base);binding=active(run);p,r,cp,origin,parent=source(binding['source_receipt'])
    assert digest(p)==binding['source_receipt_sha256']
    allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
    paired=binding.get('kind') in ('question_probe_pair','easy2560_native4','dynrpo7b_native4')
    actual_writes=binding['actual_optimizer_writes_per_rank'] if paired else 100
    step=binding['completed_outer'] if paired else 5
    probability=dict(role='verified_full_state_restore_source',run_id=run['run_id'],
        slurm_job_id=allocation['slurm_job_id'],actual_optimizer_writes=actual_writes,
        source_receipt_sha256=digest(p),source_checkpoint=str(cp),source_step=step,
        source_sha256=run['source_sha256'],resolved_config_sha256=document_sha256(document),
        actual_cuda_runtime_uuids=allocation['actual_cuda_runtime_uuids'],
        actual_model_gpu_restore_required_before_first_write=True,
        actual_behavior_probability_accepted=False,physical_execution_authenticated=False)
    for relative in ('logs/native-allocation-admission.json','preparation/formal-inline-method.json','logs/initial-probes-accepted.json'):
        receipts[relative]=digest(base/relative)
    artifact=dict(schema_version=4,role=ROLE,run_id=run['run_id'],source_commit=run['source_commit'],
        source_sha256=run['source_sha256'],initial_model_sha256=run['initial_model_sha256'],input_sha256=run['input_sha256'],budget=run['budget'],
        accepted_for_guarded_training=True,short_train_accepted=False,healthy_checkpoint_accepted=False,
        initial_probe_version=run['initial_probe_version'],initial_probe_archive=initial['archive'],
        method_sha256=digest(base/'preparation/formal-inline-method.json'),probability=probability,
        receipts_sha256=receipts,slurm_job_id=allocation['slurm_job_id'],native_birth=native_birth(allocation),
        actual_optimizer_writes_at_freeze=actual_writes,optimizer_counting=FORMAL_COUNTING,created_unix=time.time(),
        restoration_scope='verified source state; real model/Adam/scheduler/RNG load is mandatory before the next outer',**FORMAL_SAFEGUARDS)
    publish(base/'logs/policy-safety-calibration.json',artifact)
    return artifact


def verify_guarded_restore(base,run,probability):
    from reproduction.common8.policy_calibration_plan import document_sha256
    binding=active(run);p,r,cp,origin,parent=source(binding['source_receipt'])
    assert digest(p)==binding['source_receipt_sha256']==probability['source_receipt_sha256']
    assert probability['role']=='verified_full_state_restore_source'
    if binding.get('kind') in ('question_probe_pair','easy2560_native4','dynrpo7b_native4'):
        assert probability['actual_optimizer_writes']==binding['actual_optimizer_writes_per_rank']
        assert probability['source_step']==binding['completed_outer']
    else:assert probability['actual_optimizer_writes']==100 and probability['source_step']==5
    assert probability['source_checkpoint']==str(cp) and probability['run_id']==run['run_id']==parent['run_id']
    assert probability['source_sha256']==run['source_sha256']
    allocation=json.loads((Path(base)/'logs/native-allocation-admission.json').read_text())
    assert probability['slurm_job_id']==allocation['slurm_job_id']
    assert probability['actual_cuda_runtime_uuids']==allocation['actual_cuda_runtime_uuids']
    resolved=json.loads((Path(base)/'logs/resolved-training-config.json').read_text()) if (Path(base)/'logs/resolved-training-config.json').exists() else None
    if resolved:assert probability['resolved_config_sha256']==resolved['resolved_config_sha256']
    assert probability['actual_model_gpu_restore_required_before_first_write'] is True
    assert probability['actual_behavior_probability_accepted'] is False
    return True


def accept_resumed_five(records):
    import math
    from reproduction.common8.behavior_runtime import read_gate_receipt,verify_write_receipt_links
    from reproduction.common8.runtime import budget_rows,verify_scheduler_receipts
    from reproduction.common8.worker_evidence import read_console_form
    base=Path(os.environ['SEARCH_G1_BASE']);run=json.loads((base/'logs/run-manifest.json').read_text())
    binding=active(run)
    paired=(binding or {}).get('kind')=='question_probe_pair'
    first=binding['completed_outer']+1 if paired else 6
    last=min(first+4,200) if paired else 10
    assert binding and [r['step'] for r in records]==list(range(first,last+1))
    driver=json.loads((base/'logs/full-state-restored-driver.json').read_text());assert driver['actual_driver_rng_readback_equal']
    workers=json.loads((base/'logs/full-state-restored-workers.json').read_text())['ranks']
    from reproduction.common8.topology import world_size
    assert len(workers)==(world_size(run) if paired else 8) and all(v['actual_rng_readback_equal'] and v['actual_adam_scheduler_readback_equal'] for v in workers)
    actual=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    assert [v['step'] for v in actual]==list(range(first,last+1))
    assert all(all(math.isfinite(v) for v in r['metrics'].values()) for r in actual)
    gates={str(r['step']):read_gate_receipt(base,run,r['step'],update_record=r) for r in actual}
    verify_write_receipt_links(base,gates,first,last)
    budgets=budget_rows(read_console_form(base,run['run_id']),first,last,complete_records=actual,run=run)
    schedules=verify_scheduler_receipts(base,run,first,last,gates)
    checkpoint10=None
    if run.get('followup_checkpoint_required'):
        checkpoint10=json.loads((Path(run['checkpoint_dir'])/f'actor/global_step_{last}/complete.json').read_text())
        assert checkpoint10['step']==last and checkpoint10['actual_optimizer_writes_per_rank']==actual[-1]['optimizer_writes_after']
        assert 'training_state/manifest.json' in checkpoint10['files_sha256']
    result=dict(checkpoint10=checkpoint10,accepted=True,run_id=run['run_id'],source_completed_outer=first-1,
        source_actual_optimizer_writes_per_rank=binding['actual_optimizer_writes_per_rank'] if paired else 100,
        first_resumed_outer=first,last_resumed_outer=last,next_outer=last+1,actual_updates=actual,
        budgets=budgets,scheduler_receipts=schedules,actual_restore_workers=workers,actual_restore_driver=driver,
        extra_validation_performed=False,new_checkpoint10_claimed=bool(run.get("followup_checkpoint_required")),total_budget=200,checked_unix=time.time())
    publish(base/'logs/five-resumed-updates-acceptance.json',result)
    print(f'SEARCH_G1_RESUMED_FIVE_ACCEPTED={first}-{last};CONTINUING{last+1}',flush=True)
    return result


def verified_runtime_prefix(base,run):
    """Require actual restored worker/driver readbacks before auditing a suffix."""
    if (active(run) or {}).get('kind')=='dynrpo7b_native4':
        from reproduction.dynrpo7b.restore import verified_runtime_prefix as dynrpo_prefix
        return dynrpo_prefix(base,run)
    if (active(run) or {}).get('kind')=='easy2560_native4':
        from search_g1.easy2560_restore_runtime import verified_runtime_prefix as easy_prefix
        return easy_prefix(base,run)
    if (active(run) or {}).get('kind')=='question_probe_pair':
        from search_g1.qprobe_checkpoint import verified_runtime_prefix as pair_prefix
        return pair_prefix(base,run)
    base=Path(base);binding=active(run);assert binding
    p,r,cp,origin,parent=source(binding['source_receipt'])
    assert digest(p)==binding['source_receipt_sha256']
    workers=json.loads((base/'logs/full-state-restored-workers.json').read_text())['ranks']
    driver=json.loads((base/'logs/full-state-restored-driver.json').read_text())
    assert len(workers)==8 and sorted(v['rank'] for v in workers)==list(range(8))
    allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
    assert {v['actual_cuda_uuid'] for v in workers}==set(allocation['actual_cuda_runtime_uuids'])
    assert all(v['slurm_job_id']==allocation['slurm_job_id'] for v in workers)
    assert len({(v['process_identity']['pid'],v['process_identity']['start_ticks']) for v in workers})==8
    assert all(v['actual_adam_scheduler_readback_equal'] and v['actual_rng_readback_equal'] and v['actual_optimizer_writes']==100 and v['scheduler_last_epoch']==5 for v in workers)
    assert next(v for v in workers if v['rank']==0)['model_hf_exact_tensors_checked_on_rank0']==435
    assert driver['actual_driver_rng_readback_equal'] and driver['completed_outer']==5 and driver['actual_candidate_source_cursor']==10240
    assert len({v['policy_rejections'] for v in workers})==1
    attempts=[json.loads(line) for line in (origin/'logs/actual-attempts.jsonl').read_text().splitlines()]
    assert [v['attempt_index'] for v in attempts]==list(range(1,6))
    assert all(v['complete_outer'] and v['all_ranks_agreed'] for v in attempts)
    assert attempts[-1]['completed_outer']==5 and attempts[-1]['optimizer_writes_after']==100
    return dict(completed_outer=5,writes=100,attempt_index=5,rejections=workers[0]['policy_rejections'],aborted=attempts[-1]['cumulative_aborted_attempts'])


def record_original_baseline(base,run):
    """Link the actual official0 evaluation; do not repeat or relabel its rows."""
    p,r,cp,origin,parent=source(active(run)['source_receipt'])
    if (active(run) or {}).get('kind')=='question_probe_pair':
        historical=r.get('historical_artifacts',{}).get('development',{}).get('before')
        original=None
        if historical:
            assert digest(historical['path'])==historical['full_read_sha256']
            original=json.loads(Path(historical['path']).read_text())
            # Repeated resumes may wrap a genuine historical baseline. Unwrap
            # payloads, never follow a mutable later source log or invent rows.
            for _ in range(64):
                if original.get('role')!='full_state_restore_original_development_baseline':break
                if not isinstance(original.get('original'),dict):break
                original=original['original']
        result=dict(schema_version=1,role='full_state_restore_original_development_baseline',
            run_id=run['run_id'],step=0,source_pair_receipt=str(p),source_pair_receipt_sha256=digest(p),
            source_path=None if historical is None else historical['path'],
            source_sha256=None if historical is None else historical['full_read_sha256'],
            current_evaluation_performed=False,original=original,
            historical_measurement_available=original is not None,paired_same_run=False)
        publish(Path(base)/'logs/formal-inline-development-before.json',result)
        return result
    original=origin/'logs/formal-inline-development-before.json'
    result=dict(schema_version=1,role='full_state_restore_original_development_baseline',
        run_id=run['run_id'],step=0,source_path=str(original),source_sha256=digest(original),
        current_evaluation_performed=False,original=json.loads(original.read_text()))
    assert result['original']['step']==0 and result['original']['run_id']==run['run_id']
    publish(Path(base)/'logs/formal-inline-development-before.json',result)
    return result


def verify_restored_pair(base,run):
    if (active(run) or {}).get('kind')=='question_probe_pair':
        p,r,_,_,_=source(active(run)['source_receipt'])
        before=json.loads((Path(base)/'logs/formal-inline-development-before.json').read_text())
        assert before['source_pair_receipt_sha256']==digest(p) and before['current_evaluation_performed'] is False
        history=r.get('historical_artifacts',{}).get('development',{})
        measurements={}
        for key,item in history.items():
            assert digest(item['path'])==item['full_read_sha256']
            measurements[key]=dict(path=item['path'],sha256=item['full_read_sha256'],
                                    original=json.loads(Path(item['path']).read_text()))
        return dict(role='completed_pair_historical_development_evidence',
            source_completed_outer=r['step'],source_receipt_sha256=digest(p),
            current_evaluation_performed=False,paired_same_run=False,
            available_historical_measurements=measurements,
            missing_historical_measurements=[key for key in ('before','after5') if key not in measurements],
            accuracy_health_gate=False,semantics='actual linked historical artifacts only; no reconstructed early EM or new dev512 generation')
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.formal_inline_evaluation import score_counts
    from reproduction.common8.validation_identity import expected
    base=Path(base);p,r,cp,origin,parent=source(active(run)['source_receipt'])
    before_path=base/'logs/formal-inline-development-before.json'
    before_link=json.loads(before_path.read_text());original=origin/'logs/formal-inline-development-before.json'
    assert before_link['role']=='full_state_restore_original_development_baseline'
    assert before_link['source_path']==str(original) and before_link['source_sha256']==digest(original)
    assert before_link['current_evaluation_performed'] is False
    before=json.loads(original.read_text());assert before_link['original']==before
    after_path=base/'logs/formal-inline-development-after.json';after=json.loads(after_path.read_text())
    record,frame,ids=expected(base,'validation')
    for step,proof in ((0,before),(5,after)):
        assert proof['run_id']==run['run_id'] and proof['step']==step and proof['validation_sha256']==record['sha256']
        measured=aggregate(proof['rows'],ids,frame['data_source'].tolist());measured.update(score_counts(proof['rows']))
        assert all(proof[k]==v for k,v in measured.items() if k not in ('role','semantics'))
    return dict(role='original0_restored5_development_comparison',before=before,after=after,
        original_before_sha256=digest(original),after_sha256=digest(after_path),
        changes={k:after[k]-before[k] for k in ('weighted_em','weighted_f1')},accuracy_health_gate=False,
        repeated_baseline_evaluation=False)


def optimizer_construction_config(config,model_path):
    """Construct temporary empty objects, then require full actual state loading."""
    import copy
    run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
    binding=active(run)
    if not binding:return config
    p,r,checkpoint,origin,parent=source(binding['source_receipt'])
    assert digest(p)==binding['source_receipt_sha256']
    assert config.get('completed_updates',0)==r['step']==binding['completed_outer']
    if binding.get('kind')==FINAL_EASY_KIND:
        assert r['status']==FINAL_EASY_STATUS and r['step']==200 and r['restore_only'] is True
    elif binding.get('kind')=='easy2560_native4':
        assert type(r['step']) is int and 1 <= r['step'] < 200
    elif binding.get('kind')!='question_probe_pair':assert r['step']==5
    assert Path(model_path).resolve()==checkpoint
    construction=copy.deepcopy(dict(config));construction['completed_updates']=0
    # This does not change the resolved training counter or pretend restoration
    # happened. restore_worker must load and read back the saved Adam/scheduler
    # state before use, including the separate final200 restore-only path.
    return construction


def reuse_development_after5(base, run):
    """Reuse pinned historical CP5 measurements; new BF16 execution is not re-evaluated here."""
    old = Path('/mnt/data/shaoshitong/search-g1-resume5-475fbb6-20261011/logs/formal-inline-development-after.json')
    expected_sha = '87d5cdeb87e083240acead0a766b05dd42a4f5009b0ddf0e42d5080d22c52c8c'
    assert digest(old) == expected_sha
    proof = json.loads(old.read_text())
    assert proof['step'] == 5 and proof['run_id'] == run['run_id']
    publish(Path(base)/'logs/formal-inline-development-after.json', proof)
    result=verify_restored_pair(base, run)
    publish(Path(base)/'logs/reused-development-after5.json', dict(source_path=str(old),
        source_sha256=expected_sha, current_evaluation_performed=False,
        interpretation='historical FP32 CP5 validation; not a new BF16 execution measurement'))
    return result
