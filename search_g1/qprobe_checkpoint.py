"""General completed actor/question-head restoration, separate from legacy CP5.

The immutable HF checkpoint is never modified after reward-probe refit. A second
boundary driver snapshot and receipt live outside it and pin the *adopted* reward
probe/RNG. CPU SHA/counter checks are not CUDA/FSDP restart acceptance.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np
import torch


PAIR_STATUS = 'QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS'
PAIR_KIND = 'question_probe_pair'
SAVED_CHECKPOINT_STATUS = 'QPROBE_SAVED_CHECKPOINT_FULL_SHA_CPU_STATES_PASS'
SAVED_CHECKPOINT_KIND = 'question_probe_saved_checkpoint'
QPROBE_RESTORE_STATUSES = (PAIR_STATUS, SAVED_CHECKPOINT_STATUS)
_SOURCE_CACHE = {}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def _need(condition, message):
    if not condition: raise ValueError('QPROBE_PAIR_' + message)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _publish(path, value):
    from search_g1.full_state_restore import publish
    publish(path, value)


def _file_record(path, *, relative_path=None):
    original=Path(path);_need(original.is_file() and not original.is_symlink(),'UNSAFE_FILE')
    path=original.resolve()
    before=path.stat();sha=digest(path);after=path.stat()
    _need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),
          'FILE_CHANGED_DURING_FULL_SHA')
    return dict(path=str(path),relative_path=relative_path,inode=after.st_ino,bytes=after.st_size,
                mtime_ns=after.st_mtime_ns,full_read_sha256=sha)


def validate_rank_state(saved, rank, world_size, outer, *, expected_writes=None):
    _need(saved['rank']==rank and saved['world_size']==world_size,'WORKER_TOPOLOGY_MISMATCH')
    _need(saved.get('completed_outer')==outer and saved['scheduler']['last_epoch']==outer and
          saved['scheduler']['_step_count']==outer+1,'WORKER_OUTER_SCHEDULER_MISMATCH')
    writes=saved['actual_optimizer_writes'];attempt=saved.get('last_attempt_index')
    _need(type(writes) is int and writes>=outer and type(attempt) is int and attempt>=outer,
          'WORKER_ACTUAL_COUNTERS_INVALID')
    if expected_writes is not None:_need(writes==expected_writes,'WORKER_ADAM_COUNT_MISMATCH')
    _need(type(saved.get('policy_rejections')) is int and saved['policy_rejections']>=0,'REJECTION_COUNT_MISSING')
    used=saved.get('used_attempt_ids')
    _need(isinstance(used,list) and len(used)==len(set(used))==attempt,'SAVED_ATTEMPT_IDS_INCOMPLETE')
    _need(saved['optimizer']['state'],'EMPTY_ADAM_STATE')
    for values in saved['optimizer']['state'].values():
        for name in ('step','exp_avg','exp_avg_sq'):
            _need(name in values and torch.isfinite(values[name]).all(),'NONFINITE_OR_MISSING_ADAM_STATE')
        _need(int(values['step'])==writes and float(values['step'])==writes,'ADAM_STEP_READBACK_MISMATCH')
    from reproduction.common8.scheduler import expected_lr
    _need(all(group['lr']==expected_lr(outer) for group in saved['optimizer']['param_groups']),
          'ADAM_LR_SCHEDULER_MISMATCH')
    return dict(rank=rank,actual_optimizer_writes=writes,scheduler_last_epoch=outer,
        scheduler_step_count=outer+1,completed_outer=outer,last_attempt_index=attempt,
        used_attempt_ids=used,policy_rejections=saved['policy_rejections'],optimizer_tensors_finite=True)


def _validate_qstate(checkpoint, outer, actor_snapshot_id):
    directory=Path(checkpoint)/'training_state';path=directory/'question-probe-state.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    _need(state['completed_outer']==outer and state['actor_snapshot_id']==actor_snapshot_id,'HEAD_ACTOR_BINDING_MISMATCH')
    head_file=state['head_file'];_need(Path(head_file).name==head_file,'HEAD_FILE_ESCAPES_CHECKPOINT')
    head_path=directory/head_file;_need(digest(head_path)==state['head_sha256'],'HEAD_SHA_MISMATCH')
    from search_g1.qprobe_runtime import load_head_checkpoint
    head=load_head_checkpoint(head_path,state['config'],expected_actor_snapshot_id=actor_snapshot_id,
                             expected_feature_spec_sha=state['feature_spec_sha'],
                             expected_checkpoint_sha=state['head_sha256'])
    _need(head.completed_outer==outer and head.version==state['head_version'],'HEAD_OUTER_VERSION_MISMATCH')
    return state,path,head_path


def boundary_worker_rng(receipt, rank, checkpoint_state):
    """Return only the authenticated post-evaluation/refit sampling boundary."""
    row=next(item for item in receipt['boundary_worker_files'] if item['rank']==rank)
    _identity_record(row);_need(digest(row['path'])==row['full_read_sha256'],'BOUNDARY_WORKER_SHA_MISMATCH')
    state=torch.load(row['path'],map_location='cpu',weights_only=False)
    for key in ('run_id','rank','world_size','actual_optimizer_writes','completed_outer',
                'last_attempt_index','used_attempt_ids','policy_rejections'):
        _need(state[key]==checkpoint_state[key],'BOUNDARY_WORKER_COUNTER_MISMATCH_'+key)
    _need('optimizer' not in state and 'scheduler' not in state,'BOUNDARY_RNG_DUPLICATES_ADAM')
    return state['rng']


def _historical_artifacts(base,run,outer):
    """Freeze only existing bounded evidence; absent early outputs stay absent."""
    from reproduction.common8.formal_contract import SAVE_STEPS
    inherited={}
    if (run.get('full_state_restore') or {}).get('kind') in (PAIR_KIND,SAVED_CHECKPOINT_KIND):
        inherited=read_source(run['full_state_restore']['source_receipt'])[1].get('historical_artifacts',{})
    checkpoints={item['step']:item for item in inherited.get('checkpoints',[])}
    for step in SAVE_STEPS:
        if step>outer:continue
        path=Path(run.get('checkpoint_dir',base/'checkpoints'))/'actor'/f'global_step_{step}'/'complete.json'
        if path.is_file():checkpoints[step]=dict(step=step,**_file_record(path))
    development=dict(inherited.get('development',{}))
    for key,name in (('before','formal-inline-development-before.json'),('after5','formal-inline-development-after.json')):
        path=base/'logs'/name
        if path.is_file():development[key]=_file_record(path)
    archives={item['version']:item for item in inherited.get('reward_archives',[])}
    for path in (base/'artifacts').glob('round_*/version-manifest.json'):
        value=_read_json(path)
        if value.get('run_id')==run['run_id'] and type(value.get('round')) is int and value['round']<=min(outer//50,3)*50:
            archives[value['version']]=dict(version=value['version'],round=value['round'],**_file_record(path))
    return dict(checkpoints=[checkpoints[k] for k in sorted(checkpoints)],development=development,
        reward_archives=list(archives.values()),raw_training_prefix_verified=False,
        scope='immutable available historical artifact pointers only; raw prefix training history is not reconstructed')


def publish_completed_pair(trainer, checkpoint):
    """Call after checkpoint verification AND any scheduled old-probe adoption."""
    from reproduction.common8.topology import world_size
    from search_g1.training_state import atomic_save,driver_rng,compact_stream_state
    base=Path(os.environ['SEARCH_G1_BASE']).resolve();checkpoint=Path(checkpoint).resolve()
    _need(checkpoint.is_relative_to(base),'CHECKPOINT_OUTSIDE_TASK')
    run=_read_json(base/'logs/run-manifest.json');complete=_read_json(checkpoint/'complete.json')
    outer=trainer.global_steps;size=world_size(run);actor_id=run['run_id']+':'+str(outer)
    _need(type(outer) is int and 1<=outer<=200 and complete['step']==outer and
          complete['run_id']==run['run_id'] and complete['finite_all_saved_tensors'] is True,
          'ACTOR_COMPLETE_PROOF_MISSING')
    _need(getattr(trainer,'qprobe_runtime',None) is not None and trainer.qprobe_runtime.completed_outer==outer,
          'CURRENT_HEAD_NOT_PAIRED')
    manifest=_read_json(checkpoint/'training_state/manifest.json')
    _need(manifest['completed_outer']==outer and manifest.get('question_probe') is not None,'PAIRED_STATE_MISSING')
    records=[_file_record(path,relative_path=str(path.relative_to(checkpoint)))
             for path in sorted(checkpoint.rglob('*')) if path.is_file()]
    by_name={item['relative_path']:item for item in records}
    for name,sha in complete['files_sha256'].items():
        _need(name in by_name and by_name[name]['full_read_sha256']==sha,'COMPLETE_FILE_SHA_MISMATCH')
    index=_read_json(checkpoint/'model.safetensors.index.json')
    _need(len(index['weight_map'])==complete['validated_tensor_count'],'HF_TENSOR_COUNT_MISMATCH')
    _need(sorted(proof['rank'] for proof in manifest['worker_files'])==list(range(size)),'RANK_FILES_MISSING')
    ranks=[]
    for proof in manifest['worker_files']:
        path=Path(proof['path']).resolve();name=str(path.relative_to(checkpoint))
        _need(path.is_relative_to(checkpoint) and by_name[name]['full_read_sha256']==proof['sha256'],
              'WORKER_STATE_SHA_MISMATCH')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        _need(saved['run_id']==run['run_id'],'WORKER_RUN_MISMATCH')
        ranks.append(validate_rank_state(saved,proof['rank'],size,outer))
    ranks.sort(key=lambda item:item['rank'])
    signature=lambda row:(row['actual_optimizer_writes'],row['last_attempt_index'],
        tuple(row['used_attempt_ids']),row['policy_rejections'])
    _need(len({signature(row) for row in ranks})==1,'WORKER_COUNTERS_DIVERGE')
    qstate,qpath,head_path=_validate_qstate(checkpoint,outer,actor_id)
    from search_g1.full_state_restore import equal_tree
    current_stream=compact_stream_state(trainer.mixed_stream.state())
    equal_tree(qstate['candidate_state'],current_stream)
    _need(qstate['head_version']==trainer.qprobe_runtime.version,'PUBLISHED_HEAD_CHANGED')
    progress=trainer.inline_attempt_progress
    _need(progress.pending is None and progress.completed_outer==outer and
          progress.cumulative_writes==ranks[0]['actual_optimizer_writes'] and
          progress.attempt_index==ranks[0]['last_attempt_index'],'DRIVER_WORKER_COUNTERS_MISMATCH')
    _need(progress.aborted_attempts==progress.attempt_index-outer,'DRIVER_ABORTED_COUNTER_MISMATCH')
    live=_read_json(base/'artifacts/live/manifest.json')
    expected_round=min(outer//50,3)*50
    _need(live['run_id']==run['run_id'] and live['version']==trainer.reward_fn.expected_probe_version and
          live['round']==expected_round,'FINAL_REWARD_PROBE_NOT_ADOPTED')
    reward_files=[]
    for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
        row=_file_record(live[key],relative_path=key)
        _need(row['full_read_sha256']==live['head_sha256'][name],'REWARD_HEAD_SHA_MISMATCH')
        reward_files.append(row)
    for name,sha in live['encoder_sha256'].items():
        path=Path(live['encoder'])/name
        # The encoder may be this CP or an earlier retained CP/official snapshot.
        row=next((item for item in records if item['path']==str(path.resolve())),None)
        if row is None:row=_file_record(path,relative_path='reward_encoder/'+name)
        _need(row['full_read_sha256']==sha,'REWARD_ENCODER_SHA_MISMATCH')
        reward_files.append(row)
    directory=base/'artifacts/question-probe/completed-pairs'/f'global_step_{outer}'
    directory.mkdir(parents=True,exist_ok=False)
    # Evaluation and reward refit can advance vLLM/actor RNG AFTER the original
    # checkpoint worker snapshot. Capture the real final idle boundary separately.
    boundary_workers=trainer.actor_rollout_wg.capture_mixed_state(str(directory/'boundary-worker-rng'),checkpoint=False)
    _need(sorted(item['rank'] for item in boundary_workers)==list(range(size)),'BOUNDARY_RNG_RANKS_MISSING')
    boundary_records=[]
    for proof in boundary_workers:
        row=_file_record(proof['path'],relative_path=f'boundary-worker-rng/rank-{proof["rank"]}.pt')
        _need(Path(row['path']).is_relative_to(directory) and row['full_read_sha256']==proof['sha256'],
              'BOUNDARY_RNG_FILE_MISMATCH')
        row['rank']=proof['rank'];boundary_records.append(row)
    for proof in manifest['worker_files']:
        saved=torch.load(proof['path'],map_location='cpu',weights_only=False)
        boundary_worker_rng(dict(boundary_worker_files=boundary_records),proof['rank'],saved)
    boundary=atomic_save(directory/'boundary-driver.pt',dict(completed_outer=outer,run_id=run['run_id'],
        candidate_stream=current_stream,rng=driver_rng(),probe_version=live['version'],probe_round=live['round'],
        question_probe=manifest['question_probe'],question_probe_state_path=str(qpath),actor_snapshot_id=actor_id,
        attempt_progress=dict(completed_outer=outer,cumulative_writes=progress.cumulative_writes,
            attempt_index=progress.attempt_index,aborted_attempts=progress.aborted_attempts,
            full_budget_outer=progress.full_budget_outer),
        used_attempt_ids=ranks[0]['used_attempt_ids'],policy_rejections=ranks[0]['policy_rejections']))
    _publish(directory/'reward-probe-manifest.json',live)
    boundary_record=_file_record(boundary['path'],relative_path='boundary-driver.pt')
    run_record=_file_record(base/'logs/run-manifest.json',relative_path='run-manifest.json')
    reward_record=_file_record(directory/'reward-probe-manifest.json',relative_path='reward-probe-manifest.json')
    receipt=dict(schema_version=1,status=PAIR_STATUS,kind=PAIR_KIND,step=outer,run_id=run['run_id'],
        checkpoint=str(checkpoint),source_base=str(base),source_run_manifest=run_record,
        world_size=size,actor_snapshot_id=actor_id,ranks=ranks,files=records,boundary_driver=boundary_record,
        boundary_worker_files=boundary_records,historical_artifacts=_historical_artifacts(base,run,outer),
        reward_probe_manifest=reward_record,reward_files=reward_files,reward_probe_version=live['version'],
        reward_probe_round=live['round'],question_probe_state_path=str(qpath),head_path=str(head_path),
        feature_spec_sha=qstate['feature_spec_sha'],head_version=qstate['head_version'],
        hf_saved_tensor_count=complete['validated_tensor_count'],attempt_progress=
            dict(completed_outer=outer,cumulative_writes=progress.cumulative_writes,
                 attempt_index=progress.attempt_index,aborted_attempts=progress.aborted_attempts,
            full_budget_outer=progress.full_budget_outer),
        driver_state=dict(completed_outer=outer,source_consumed=current_stream['consumed_source_rows']),
        actual_cuda_restore_accepted=False,scope='full-file SHA and CPU saved-state checks; actual CUDA/FSDP reload pending',
        checked_unix=time.time())
    path=directory/'source-receipt.json';_publish(path,receipt)
    # Read back immutable receipt and all small boundary/head bindings.
    read_source(path)
    return dict(path=str(path),sha256=digest(path),status=PAIR_STATUS,step=outer,
                actual_cuda_restore_accepted=False)


def _identity_record(row):
    path=Path(row['path']);_need(path.is_file() and not path.is_symlink(),'SOURCE_FILE_MISSING_OR_SYMLINK')
    stat=path.stat()
    _need((stat.st_ino,stat.st_size,stat.st_mtime_ns)==(row['inode'],row['bytes'],row['mtime_ns']),
          'SOURCE_FILE_CHANGED')
    return (str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns)


def read_source(receipt_path):
    path=Path(receipt_path).resolve();receipt=_read_json(path)
    if receipt.get('status')==SAVED_CHECKPOINT_STATUS:
        from search_g1.qprobe_saved_checkpoint import read_source as read_saved_source
        return read_saved_source(path)
    _need(receipt.get('status')==PAIR_STATUS and receipt.get('schema_version')==1,'RECEIPT_SCHEMA_MISMATCH')
    outer=receipt['step'];size=receipt['world_size'];checkpoint=Path(receipt['checkpoint']).resolve()
    base=Path(receipt['source_base']).resolve()
    _need(type(outer) is int and 1<=outer<=200 and size in (4,8),'RECEIPT_COUNTER_OR_TOPOLOGY_INVALID')
    _need(checkpoint.is_relative_to(base) and receipt['actor_snapshot_id']==receipt['run_id']+':'+str(outer),
          'RECEIPT_ACTOR_PATH_BINDING_MISMATCH')
    history=receipt.get('historical_artifacts',{})
    historical_records=history.get('checkpoints',[])+list(history.get('development',{}).values())+history.get('reward_archives',[])
    all_records=receipt['files']+receipt['boundary_worker_files']+[receipt['boundary_driver'],receipt['source_run_manifest'],
        receipt['reward_probe_manifest']]+receipt['reward_files']
    all_records+=historical_records
    identity=tuple(_identity_record(row) for row in all_records)
    actual={str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()}
    _need(actual=={row['relative_path'] for row in receipt['files']},'SOURCE_CHECKPOINT_FILE_SET_CHANGED')
    for row in receipt['files']:
        _need(Path(row['path']).resolve()==checkpoint/row['relative_path'],'SOURCE_RELATIVE_PATH_MISMATCH')
    key=(str(path),digest(path),identity)
    if key not in _SOURCE_CACHE:
        # Large immutable HF files were fully read by publisher and retain their
        # inode/size/mtime binding. Hash and parse actual small state/head files;
        # each worker additionally fully hashes its own Adam state before load.
        small=[receipt['boundary_driver'],receipt['source_run_manifest'],receipt['reward_probe_manifest']]+historical_records
        small += [row for row in receipt['files'] if row['path'] in
                  (receipt['question_probe_state_path'],receipt['head_path'])]
        for row in small:_need(digest(row['path'])==row['full_read_sha256'],'SOURCE_SMALL_FILE_SHA_MISMATCH')
        parent=_read_json(receipt['source_run_manifest']['path'])
        _need(parent['run_id']==receipt['run_id'],'SOURCE_RUN_BINDING_MISMATCH')
        _need(parent.get('question_probe') is not None,'SOURCE_HAS_NO_QUESTION_PROBE_METHOD')
        ranks=receipt['ranks'];_need(sorted(row['rank'] for row in ranks)==list(range(size)),'SOURCE_RANKS_INVALID')
        _need(sorted(row['rank'] for row in receipt['boundary_worker_files'])==list(range(size)),
              'SOURCE_BOUNDARY_RNG_RANKS_INVALID')
        progress=receipt['attempt_progress'];_need(progress['completed_outer']==outer and
            progress['cumulative_writes']==ranks[0]['actual_optimizer_writes'] and
            progress['attempt_index']==ranks[0]['last_attempt_index'] and
            progress['aborted_attempts']==progress['attempt_index']-outer,'SOURCE_ATTEMPT_COUNTS_INVALID')
        _need(all(row['completed_outer']==outer and row['actual_optimizer_writes']==progress['cumulative_writes'] and
                  row['last_attempt_index']==progress['attempt_index'] and row['scheduler_last_epoch']==outer and
                  row['scheduler_step_count']==outer+1 and row['optimizer_tensors_finite'] is True and
                  row['used_attempt_ids']==ranks[0]['used_attempt_ids'] and
                  row['policy_rejections']==ranks[0]['policy_rejections'] for row in ranks),'SOURCE_WORKER_COUNTS_DIVERGE')
        boundary=torch.load(receipt['boundary_driver']['path'],map_location='cpu',weights_only=False)
        from search_g1.full_state_restore import equal_tree
        equal_tree(boundary['attempt_progress'],progress)
        _need(boundary['probe_version']==receipt['reward_probe_version'] and
              boundary['probe_round']==receipt['reward_probe_round'] and boundary['completed_outer']==outer and
              boundary['run_id']==receipt['run_id'],'BOUNDARY_DRIVER_BINDING_MISMATCH')
        _need(boundary['used_attempt_ids']==ranks[0]['used_attempt_ids'] and
              boundary['policy_rejections']==ranks[0]['policy_rejections'],'BOUNDARY_ATTEMPT_IDS_MISMATCH')
        qstate,_,_=_validate_qstate(checkpoint,outer,receipt['actor_snapshot_id'])
        equal_tree(qstate['candidate_state'],boundary['candidate_stream'])
        _need(qstate['head_version']==receipt['head_version'] and
              qstate['feature_spec_sha']==receipt['feature_spec_sha'],'SOURCE_HEAD_BINDING_MISMATCH')
        live=_read_json(receipt['reward_probe_manifest']['path'])
        _need(live['version']==receipt['reward_probe_version'] and live['round']==receipt['reward_probe_round'] and
              live['run_id']==receipt['run_id'],'PINNED_REWARD_PROBE_BINDING_MISMATCH')
        _SOURCE_CACHE[key]=parent
    return path,receipt,checkpoint,base,_SOURCE_CACHE[key]


def bind_restore(run, receipt_path):
    """Bind a new pristine execution to the complete pair; no file/GPU changes."""
    path,receipt,checkpoint,base,parent=read_source(receipt_path)
    _need(receipt['step']<200,'TRAINING_TARGET_ALREADY_COMPLETE')
    _need(run.get('full_state_restore') is None,'TARGET_ALREADY_BOUND_FOR_RESTORE')
    _need(run.get('completed_updates',0)==0,'TARGET_MUST_BE_PRISTINE')
    if run.get('question_probe') is None:
        run['question_probe']=copy.deepcopy(parent['question_probe'])
        # The restored pair is authoritative; never initialize from a later
        # bootstrap head or retain the old task's writable output directory.
        run['question_probe']['initial_head_checkpoint']=None
        run['question_probe']['snapshot_path']=None
        if run['question_probe'].get('head_kind') in ('token_binary_v1', 'qwen_native_lastblock_binary_v1'):
            # This completed actor/head pair supplies every Head/Adam/RNG/replay
            # field. The original warmup artifact is no longer an input.
            run['question_probe']['snapshot_path']=receipt['question_probe_state_path']
            for key in ('initial_head_checkpoint_sha256', 'initial_head_ready', 'initial_head_ready_sha256'):
                run['question_probe'][key]=None
        if run.get('checkpoint_dir'):
            target_base=Path(run['checkpoint_dir']).resolve().parents[1]
            run['question_probe']['output_dir']=str(target_base/'artifacts/question-probe')
    run['mixed_first_selection']=False
    run['data_selection_mode']='question_probe'
    for key in ('budget','input_sha256','initial_model_sha256','training_world_size','actor_micro_batch_per_rank'):
        _need(run.get(key)==parent.get(key),'TARGET_MODEL_DATA_OR_BUDGET_MISMATCH_'+key)
    phase=parent.get('infra_rollout')
    if phase is not None:
        from search_g1.rollout_phase_contract import validate
        phase=validate(phase,parent['training_world_size'])
        _need(run.get('infra_rollout',phase)==phase,'TARGET_ROLLOUT_PHASE_CHANGED')
        run['infra_rollout']=copy.deepcopy(phase)
    else:
        _need(run.get('infra_rollout') is None,'TARGET_ADDED_AN_UNBOUND_ROLLOUT_PHASE')
    from search_g1.qprobe_runtime import validate_config
    ignore={'cache_dir','output_dir','initial_head_checkpoint','snapshot_path','restore_state',
            'initial_head_checkpoint_sha256','initial_head_ready','initial_head_ready_sha256'}
    new={k:v for k,v in validate_config(run['question_probe']).items() if k not in ignore}
    old={k:v for k,v in validate_config(parent['question_probe']).items() if k not in ignore}
    _need(new==old,'TARGET_HEAD_CONFIGURATION_MISMATCH')
    progress=receipt['attempt_progress']
    for key in ('retriever_version','retriever_manifest_sha256'):
        if key in parent:
            _need(run.get(key,parent[key])==parent[key],'TARGET_RETRIEVER_VERSION_CHANGED')
            run[key]=parent[key]
    # Endpoints may move between executions. Index identity and retrieval depth
    # define the semantic label namespace and must remain unchanged.
    topk=parent.get('retriever_topk',3)
    _need(type(topk) is int and topk>0,'SOURCE_RETRIEVER_TOPK_INVALID')
    _need(run.get('retriever_topk',topk)==topk,'TARGET_RETRIEVER_TOPK_CHANGED')
    run['retriever_topk']=topk
    kind=SAVED_CHECKPOINT_KIND if receipt['status']==SAVED_CHECKPOINT_STATUS else PAIR_KIND
    run.update(run_id=parent['run_id'],native_ray_execution_id=uuid.uuid4().hex,
        initial_probe_version=receipt['reward_probe_version'],pretraining_baseline_policy='reuse_historical',
        full_state_restore=dict(kind=kind,source_receipt=str(path),source_receipt_sha256=digest(path),
            source_checkpoint=str(checkpoint),source_base=str(base),
            source_run_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'],
            completed_outer=receipt['step'],actual_optimizer_writes_per_rank=progress['cumulative_writes'],
            attempt_index=progress['attempt_index'],aborted_attempts=progress['aborted_attempts'],
            candidate_source_cursor=receipt['driver_state']['source_consumed'],
            reward_probe_version=receipt['reward_probe_version'],reward_probe_round=receipt['reward_probe_round'],
            question_probe_state_path=receipt['question_probe_state_path'],feature_spec_sha=receipt['feature_spec_sha'],
            followup_acceptance_outer=min(receipt['step']+5,200)),
        optimizer_semantics='actual paired actor/head/Adam/scheduler/RNG/source-stream restoration; CUDA readback required')
    if kind==SAVED_CHECKPOINT_KIND:
        run['full_state_restore'].update(boundary_type='saved_checkpoint',
            boundary_origin=receipt['boundary_origin'],boundary_scope=receipt['boundary_scope'],
            post_evaluation_rng_persisted=False)
        run['optimizer_semantics']='saved CP actor/head/Adam/scheduler/RNG/source-stream restoration; post-evaluation RNG unavailable; actual CUDA readback required'
    return run


def install_restored_reward_probe(base, run):
    """Copy only pinned small legacy reward heads into a pristine target task."""
    _,receipt,_,_,_=read_source(run['full_state_restore']['source_receipt'])
    base=Path(base).resolve();live_dir=base/'artifacts/live';live_dir.mkdir(parents=True,exist_ok=True)
    _need(not (live_dir/'manifest.json').exists(),'TARGET_REWARD_PROBE_ALREADY_EXISTS')
    old=_read_json(receipt['reward_probe_manifest']['path'])
    directory=base/'artifacts'/('restored-reward-round_'+str(old['round'])+'-'+uuid.uuid4().hex)
    directory.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(old)
    for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
        original=Path(old[key]);_need(digest(original)==old['head_sha256'][name],'REWARD_HEAD_CHANGED')
        destination=directory/name;shutil.copyfile(original,destination)
        _need(digest(destination)==digest(original),'REWARD_HEAD_COPY_SHA_MISMATCH')
        manifest[key]=str(destination)
    _publish(directory/'version-manifest.json',manifest);_publish(live_dir/'manifest.json',manifest)
    return dict(version=manifest['version'],round=manifest['round'],archive=str(directory/'version-manifest.json'),
                source_receipt_sha256=digest(run['full_state_restore']['source_receipt']),actual_rpc_adoption_pending=True)


def restore_driver(trainer, receipt_path):
    from search_g1.training_state import compact_stream_state,driver_rng
    from search_g1.full_state_restore import equal_tree,bind_restored_probe,publish
    _,receipt,checkpoint,_,_=read_source(receipt_path)
    saved_boundary=receipt['status']==SAVED_CHECKPOINT_STATUS
    driver_file=receipt['driver_file'] if saved_boundary else receipt['boundary_driver']
    saved=torch.load(driver_file['path'],map_location='cpu',weights_only=False)
    base=Path(os.environ['SEARCH_G1_BASE']);run=_read_json(base/'logs/run-manifest.json')
    live=_read_json(base/'artifacts/live/manifest.json')
    bind_restored_probe(trainer,saved,run,live,receipt['run_id'])
    torch.set_rng_state(saved['rng']['torch_cpu']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    equal_tree(driver_rng(),saved['rng'])
    trainer._restored_candidate_state=compact_stream_state(saved['candidate_stream'])
    trainer._restored_attempt_progress=copy.deepcopy(saved['attempt_progress'])
    trainer._restored_qprobe_state=dict(path=receipt['question_probe_state_path'],
        actor_snapshot_id=receipt['actor_snapshot_id'],head_version=receipt['head_version'],
        feature_spec_sha=receipt['feature_spec_sha'],source_receipt_sha256=digest(receipt_path))
    row=dict(source_checkpoint=str(checkpoint),actual_driver_rng_readback_equal=True,
        completed_outer=receipt['step'],actual_candidate_source_cursor=saved['candidate_stream']['consumed_source_rows'],
        candidate_order_count=len(saved['candidate_stream']['candidate_order']),pending_offset=saved['candidate_stream']['pending_offset'],
        actual_probe_version=saved['probe_version'],actual_probe_round=receipt['reward_probe_round'] if saved_boundary else saved['probe_round'],
        attempt_progress=copy.deepcopy(saved['attempt_progress']),
        used_attempt_ids=receipt['ranks'][0]['used_attempt_ids'] if saved_boundary else saved['used_attempt_ids'],
        reward_alpha_fixed=True,historical_reward_diagnostics_restored=False,
        source_driver_full_sha256=driver_file['full_read_sha256'],checked_unix=time.time())
    if saved_boundary:row.update(sampling_rng_source='authenticated saved-checkpoint driver RNG',post_evaluation_rng_persisted=False)
    publish(base/'logs/full-state-restored-driver.json',row)
    return row


def verified_runtime_prefix(base,run):
    _,receipt,_,_,_=read_source(run['full_state_restore']['source_receipt'])
    base=Path(base);workers=_read_json(base/'logs/full-state-restored-workers.json')['ranks']
    driver=_read_json(base/'logs/full-state-restored-driver.json')
    size=receipt['world_size'];outer=receipt['step'];progress=receipt['attempt_progress']
    _need(sorted(row['rank'] for row in workers)==list(range(size)),'RESTORED_RANK_COUNT_INVALID')
    allocation=_read_json(base/'logs/native-allocation-admission.json')
    _need({row['actual_cuda_uuid'] for row in workers}==set(allocation['actual_cuda_runtime_uuids']),
          'RESTORED_CUDA_ALLOCATION_MISMATCH')
    _need(len({(row['process_identity']['pid'],row['process_identity']['start_ticks']) for row in workers})==size,
          'RESTORED_PROCESS_IDENTITIES_NOT_UNIQUE')
    _need(all(row['slurm_job_id']==allocation['slurm_job_id'] and row['actual_adam_scheduler_readback_equal'] and
        row['actual_rng_readback_equal'] and row['actual_optimizer_writes']==progress['cumulative_writes'] and
        row['scheduler_last_epoch']==outer and row['last_attempt_index']==progress['attempt_index'] for row in workers),
        'ACTUAL_WORKER_READBACK_MISSING')
    _need(next(row for row in workers if row['rank']==0)['model_hf_exact_tensors_checked_on_rank0']==receipt['hf_saved_tensor_count'],
          'ACTUAL_HF_MODEL_READBACK_MISSING')
    _need(driver['actual_driver_rng_readback_equal'] and driver['completed_outer']==outer and
        driver['actual_candidate_source_cursor']==receipt['driver_state']['source_consumed'],
        'ACTUAL_DRIVER_READBACK_MISSING')
    if receipt['status']==SAVED_CHECKPOINT_STATUS:
        states={row['relative_path']:row for row in receipt['files']}
        _need(all(row['source_checkpoint']==receipt['checkpoint'] and row['source_run_id']==receipt['run_id'] and
            row['source_state_full_sha256']==states[f'training_state/rank-{row["rank"]}.pt']['full_read_sha256'] and
            row['post_evaluation_rng_persisted'] is False for row in workers),
            'SAVED_CHECKPOINT_ACTUAL_WORKER_SOURCE_MISMATCH')
        _need(driver['source_checkpoint']==receipt['checkpoint'] and
            driver['source_driver_full_sha256']==receipt['driver_file']['full_read_sha256'] and
            driver['post_evaluation_rng_persisted'] is False,'SAVED_CHECKPOINT_ACTUAL_DRIVER_SOURCE_MISMATCH')
    used=receipt['ranks'][0]['used_attempt_ids']
    return dict(completed_outer=outer,writes=progress['cumulative_writes'],attempt_index=progress['attempt_index'],
                rejections=receipt['ranks'][0]['policy_rejections'],aborted=progress['aborted_attempts'],
                used_attempt_ids=used)
