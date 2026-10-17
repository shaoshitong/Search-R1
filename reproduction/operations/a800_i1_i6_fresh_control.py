"""One fresh A800 I1-I6 actor200 execution after actual stop/output cleanup.

The retained epoch5 EMA is a model-only source.  No previous actor, selection
head optimizer, candidate cursor or training log is needed.  There is no stop
action.  Exact source staging, real CPU input hashes, idle UUID selection and
the existing OFD-guarded native submit remain distinct acceptance boundaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from . import a800_actorhidden_prefix5_p10_control as previous

ROOT, PYTHON, MAX_OUTPUT = previous.ROOT, previous.PYTHON, previous.MAX_OUTPUT
route = previous.route
SIDE_FILES = {'launch-config-template.json', 'human-intent.json', 'previous-terminal.json',
              'wrong-run-cleanup-receipt.json', 'accepted-source-qualification.json'}
SIDE_HASHES = {'human-intent.json': 'human_intent_sha256',
    'previous-terminal.json': 'terminal_proof_sha256',
    'wrong-run-cleanup-receipt.json': 'cleanup_receipt_sha256',
    'accepted-source-qualification.json': 'accepted_source_qualification_sha256'}
CP5_SHA = 'b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'

# Reuse only transport/ownership helpers.  Neither old_actor_handle nor any
# old actor/log path is imported into this new controller's binding contract.
_prefix = previous.BINDING[:previous.BINDING.index("old_h=h['old_actor_handle']")]
_helpers = previous.BINDING[previous.BINDING.index('def regular('):previous.BINDING.index('def verify_old():')]
COMMON = ("import json,pathlib,sys\nsys.dont_write_bytecode=True\np=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'\n"
          + _prefix + _helpers + r'''
assert h['run_id'] not in ('387d3f6e228a488f90c418e2ba47e28e','411c44aad7ba4e8c90f5fbec9e5ecbaa')
assert h['template_config_path']==str(b/'preparation/launch-config-template.json')
SIDE_HASHES={'human-intent.json':'human_intent_sha256','previous-terminal.json':'terminal_proof_sha256',
 'wrong-run-cleanup-receipt.json':'cleanup_receipt_sha256','accepted-source-qualification.json':'accepted_source_qualification_sha256'}
def prerequisites():
 for name,key in SIDE_HASHES.items():
  assert re.fullmatch('[0-9a-f]{64}',h[key]) and sha(b/'preparation'/name)==h[key]
 human=read(b/'preparation/human-intent.json')
 assert human.get('role')=='verified_latest_cross_thread_human_stop_repair_fresh_restart_authorization'
 assert human.get('authorized_target_thread_id')=='01a0fb1b-d3ff-74b3-aa13-49ca57871f92'
 assert human.get('user_messages'),'Actual human authorization is missing'
 terminal=read(b/'preparation/previous-terminal.json')
 if isinstance(terminal.get('stdout'),str):terminal=json.loads(terminal['stdout'])
 assert terminal.get('terminal') is True and terminal.get('manual_stop_not_failure') is True
 assert terminal['run_id']=='387d3f6e228a488f90c418e2ba47e28e'
 assert terminal['execution_id']=='1bf0183c24c84abfa8a65c587089145d'
 assert terminal['source_commit']=='0de9565c9b1c735635744f7e80a2526a2852e568'
 assert terminal.get('pipeline_alive') is False and not terminal.get('remaining_owned_births')
 assert not terminal.get('remaining_base_bound_processes')
 assert not alive(native_identity(terminal['pipeline_identity']))
 cleanup=read(b/'preparation/wrong-run-cleanup-receipt.json')
 assert cleanup.get('status')=='EXPLICIT_WRONG_RUN_OUTPUTS_DELETED_PROTECTED_ASSETS_RETAINED'
 assert cleanup.get('protected_inputs_retained') is True
 assert cleanup.get('warmup_sha256')=='b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'
 assert cleanup.get('outcomes') and all(x['absent_all_selected_scopes'] and x['code_preparation_preserved'] for x in cleanup['outcomes'])
 assert all(x['identity']['run_id'] in ('387d3f6e228a488f90c418e2ba47e28e','411c44aad7ba4e8c90f5fbec9e5ecbaa') for x in cleanup['outcomes'])
 assert any(x['identity']['execution_id']==terminal['execution_id'] and x['identity']['run_id']==terminal['run_id'] for x in cleanup['outcomes'])
 pin=canonical_path(cleanup['warmup_pin']);regular(pin)
 assert not any(str(pin)==str(pathlib.Path(x['identity']['base'])/scope) or str(pin).startswith(str(pathlib.Path(x['identity']['base'])/scope)+'/')
  for x in cleanup['outcomes'] for scope in ('actor','probe','outers','logs','validation-retries','final-evaluation','checkpoints','rollouts'))
 return terminal,cleanup
def source(require_receipt=True):
 assert read(b/'preparation/deployment.json')==h
 actual=set()
 for path in repo.rglob('*'):
  if path.is_dir():assert not path.is_symlink() and path.resolve()==path
  else:regular(path);actual.add(str(path.relative_to(repo)))
 assert actual==set(h['source_files']),'Exact deployed source inventory differs'
 for name,digest in h['source_files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
  assert sha(repo/name)==digest,'Published source file mismatch: '+name
 prerequisites()
 if require_receipt:
  receipt=read(b/'preparation/source-receipt.json')
  assert receipt['source_commit']==h['source_commit'] and receipt['source_files_verified']==len(h['source_files'])
def qualification():
 ended=read(b/'preparation/prepare-exit.json')
 assert ended['exit_code']==0 and ended['run_id']==h['run_id'] and ended['execution_id']==h['execution_id'] and ended['source_commit']==h['source_commit']
 assert not alive(native_identity(ended['identity'])),'CPU prepare writer is still active'
 q=read(b/'preparation/source-qualification.json');c=read(b/'preparation/launch-config.json')
 assert sha(b/'preparation/source-qualification.json')==ended['qualification_sha256']
 assert sha(b/'preparation/launch-config.json')==ended['config_sha256']
 assert c['run_id']==h['run_id'] and c['execution_id']==h['execution_id'] and c['source_commit']==h['source_commit'] and c['base']==str(b)
 assert not any(c.get(k) for k in ('resume_joint','resume_pending_probe','resume_import','continue_reuse'))
 assert q.get('finite') is True and q.get('model_only_import') is True and q['model_kind']=='ema' and q['source_epoch']==5
 assert q['source_sha256']=='b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'
 assert q['source_checkpoint']==c['probe_source_checkpoint'] and q['source_sha256']==c['probe_source_sha256']
 assert q['current_tensor_deserializations']==0 and q['warmup_optimizer_imported'] is False and q['warmup_rng_imported'] is False
 sys.path.insert(0,str(repo))
 from reproduction.prefix_selection200.launch import validate_config
 validate_config(c,b)
 return q,c
def emit(value):
 raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<=128*1024,'Output budget exceeded';print(raw)
''')

_stage = previous.STAGE[len(previous.COMMON):]
_stage = _stage.replace("assert {'launch-config-template.json','old-actor-deployment.json'}<=set(p['side_files'])",
    "assert {'launch-config-template.json','human-intent.json','previous-terminal.json','wrong-run-cleanup-receipt.json','accepted-source-qualification.json'}==set(p['side_files'])")
STAGE = COMMON + _stage.replace('A800_ACTORHIDDEN_PREFIX5_EXACT_SOURCE_STAGED', 'A800_I1_I6_FRESH_EXACT_SOURCE_STAGED')
PREPARE = COMMON + r'''
source();assert not (b/'preparation/prepare-intent.json').exists(),'Prepare onceclaim exists; inspect, never replay'
assert not (b/'preparation/prepare-exit.json').exists()
write(b/'preparation/prepare-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
 checked_unix=time.time(),fresh_actor=True,old_wrong_outputs_cleanup_verified=True),True)
command=['/data/search-g1/env/bin/python','-m','reproduction.operations.a800_i1_i6_fresh_control','--prepare-worker','--handle',str(b/'preparation/deployment.json')]
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (b/'logs/cpu-prepare.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
who=birth(child.pid);assert who['sid']==who['pid'] and who['uid']==os.getuid()
value=dict(status='FRESH_I1_I6_CPU_PREPARE_ONCE_LAUNCHED_NOT_ACCEPTED',run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],identity=who,started_unix=time.time())
write(b/'preparation/prepare-launch.json',value,True);emit(value)
'''
LAUNCH = COMMON + r'''
source();q,c=qualification();terminal,cleanup=prerequisites()
assert not (b/'preparation/i1-i6-launch-intent.json').exists(),'Launch onceclaim exists; observe only'
assert not (b/'preparation/submit-intent.json').exists(),'Submit already claimed'
assert not bound_processes(b,exclude=(os.getpid(),)),'A target namespace writer remains'
write(b/'preparation/i1-i6-launch-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
 config_sha256=sha(b/'preparation/launch-config.json'),requested_unix=time.time(),fresh_actor0=True,warmup_model_only=True,
 old_actor_state_imported=False,old_online_probe_optimizer_imported=False,terminal_proof_sha256=h['terminal_proof_sha256'],cleanup_receipt_sha256=h['cleanup_receipt_sha256']),True)
from reproduction.prefix_selection200.launch import submit
value=submit(b)
emit(dict(status='FRESH_I1_I6_ONCE_SUBMIT_RETURNED_NOT_TRAINING_ACCEPTANCE',submit=value,run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),old_wrong_outputs_cleanup_verified=True))
'''

_observe = previous.OBSERVE[len(previous.COMMON):]
_observe = _observe.replace("assert actor['actual_adam_steps_per_rank']==[20*n]*4 and actor['scheduler_completed_outer']==n",
    "sys.path.insert(0,str(repo))\n from reproduction.prefix_selection200.storage import actual_actor_boundary\n assert actual_actor_boundary(actor,n),'Unknown or invalid complete actor opportunity/write boundary'")
_observe = _observe.replace("'actual_adam_steps_per_rank','new_adam_steps_per_rank','scheduler_completed_outer','learning_rate_used',",
    "'actual_adam_steps_per_rank','new_adam_steps_per_rank','opportunities_processed','skipped_update_opportunities','complete_outer','status','scheduler_completed_outer','learning_rate_used',")
_observe = _observe.replace("'new_adam_steps_per_rank','scheduler_completed_outer','learning_rate_used','next_learning_rate',",
    "'new_adam_steps_per_rank','opportunities_processed','skipped_update_opportunities','complete_outer','status','scheduler_completed_outer','learning_rate_used','next_learning_rate',")
_old = "assert all(row['rank']==rank and row['outer']==current and row['opportunity']==index for index,row in enumerate(writes))"
_new = r'''assert all(row['rank']==rank and row['outer']==current and 0<=row['opportunity']<20 for row in writes)
   assert [row['opportunity'] for row in writes]==sorted(set(row['opportunity'] for row in writes))
   restored=[row for row in rows if row.get('event')=='ACTUAL_RESTORE']
   assert len(restored)<=1
   if writes:
    assert len(restored)==1 and [row['actual_adam_steps'] for row in writes]==list(range(restored[0]['actual_adam_before']+1,restored[0]['actual_adam_before']+len(writes)+1))
   opportunities=[row for row in rows if row.get('event')=='OPPORTUNITY']
   assert all(row['rank']==rank and row['outer']==current and row['opportunity']==index for index,row in enumerate(opportunities))'''
assert _observe.count(_old) == 1
_observe = _observe.replace(_old, _new)
_observe = _observe.replace("last_actual_adam=writes[-1]['actual_adam_steps'] if writes else None,incomplete_final_line=incomplete,",
    "last_actual_adam=writes[-1]['actual_adam_steps'] if writes else (restored[0]['actual_adam_before'] if restored else None),opportunities_processed=len(opportunities),opportunity_statuses=[row.get('status') for row in opportunities],incomplete_final_line=incomplete,")
_observe = _observe.replace("v['metadata_bytes_read']=metadata_bytes\nemit(v)", r'''
grounding_keys=('round','version','run_id','source_run_id','manifest_path','manifest_sha256','head_sha256',
 'encoder','source_actor_version','adoption_receipt_sha256','source_actor_counters','lifecycle')
def grounding_projection(value):return None if value is None else project(value,grounding_keys)
v['grounding_latest']=grounding_projection(document(b,'grounding/latest.json'))
v['grounding_fit_boundaries']=[grounding_projection(document(b,'grounding/rounds/round-%06d/fit-complete.json'%n)) for n in (0,50,100,150)]
c=document(b,'preparation/launch-config.json')
v['interaction_protocol']=None if c is None else c.get('interaction_protocol')
v['grounding_refresh_contract']=None if c is None else dict(project(c,('grounding_refresh_enabled','grounding_refit_rounds','grounding_calibration_sha256')),
 calibration_question_count=len(c.get('grounding_calibration_question_ids',[])),
 calibration_question_ids_sha256=hashlib.sha256(canonical(c.get('grounding_calibration_question_ids',[]))).hexdigest())
v['metadata_bytes_read']=metadata_bytes
emit(v)''')
OBSERVE = COMMON + _observe


def _sha(path):
    from reproduction.prefix_selection200.a800_prepare import full_sha
    return full_sha(path)


def _once(path, value):
    from reproduction.prefix_selection200.a800_prepare import once
    return once(path, value)


def _pinned_json(base, h, name):
    path=base/'preparation'/name
    if name in SIDE_HASHES and _sha(path)!=h[SIDE_HASHES[name]]:
        raise ValueError('Actual staged prerequisite bytes differ: '+name)
    return json.loads(path.read_text())


def _idle_mapping(base, receipt):
    from reproduction.prefix_selection200.a800_runtime import gpu_inventory, compute_processes
    from reproduction.prefix20_online.launch import publish
    while True:
        inventory=gpu_inventory();used={x['uuid'] for x in compute_processes()}
        idle=[i for i,row in sorted(inventory.items()) if row['uuid'] not in used and row['memory_MiB']<=64]
        training=[i for i in idle if i not in (5,6)]
        if len(training)>=4 and all(i in idle for i in (5,6)):
            chosen=training[:4]
            proof=dict(run_id=receipt['run_id'],execution_id=receipt['execution_id'],checked_unix=time.time(),
                physical_gpu_indices=chosen,physical_gpu_uuids=[inventory[i]['uuid'] for i in chosen],
                auxiliary_gpu_indices=[5,6],auxiliary_gpu_uuids=[inventory[i]['uuid'] for i in (5,6)],
                actual_inventory=[inventory[i] for i in chosen+[5,6]],idle_memory_threshold_MiB=64,
                actual_idle_mapping=True,foreign_processes_stopped=False,native_uuid_ofd_readmission_required=True)
            _once(base/'preparation/idle-mapping.json',proof);return proof
        publish(base/'preparation/prepare-progress.json',dict(receipt,stage='WAITING_ACTUAL_IDLE_FOUR_PLUS_AUX5_6',
            idle_training_count=len(training),auxiliary5_6_idle=all(i in idle for i in (5,6)),checked_unix=time.time()))
        time.sleep(30)


def _accepted_model_only(base, h, cleanup, q):
    from reproduction.prefix_selection200.contract import QMAX,WIDTH,DIM,PREFIX_TOKENS
    source=Path(cleanup['warmup_pin']);s=source.stat()
    extraction=q.get('native_model_extraction',{})
    if (source.is_symlink() or not source.is_file() or source.resolve(strict=True)!=source
            or _sha(source)!=CP5_SHA or cleanup['warmup_sha256']!=CP5_SHA
            or q.get('source_sha256')!=CP5_SHA or q.get('source_epoch')!=5
            or q.get('model_kind')!='ema' or q.get('finite') is not True
            or q.get('model_only_import') is not True or q.get('cpu_load_verified') is not True
            or q.get('full_sha_verified') is not True or q.get('warmup_optimizer_imported') is not False
            or q.get('warmup_rng_imported') is not False
            or q.get('source_warmup_run')!='8af926a49630428097b94027864da68b'
            or (q.get('question_max_tokens'),q.get('prefix_tokens'),q.get('head_max_tokens'),q.get('hidden_size'))!=(QMAX,PREFIX_TOKENS,WIDTH,DIM)
            or extraction.get('learned_tensor_count')!=14 or extraction.get('warmup_model_weights_only') is not True
            or 'rotary.inv_freq' not in extraction.get('nonpersistent_buffer_names',[])
            or (s.st_size,s.st_ino,s.st_dev)!=(q['source_bytes'],q['source_inode'],q['source_device'])):
        raise ValueError('Only exact retained, previously finite-qualified native epoch5 EMA model is authorized')
    target=base/'preparation/warmup-source-epoch-0005.pt';os.link(source,target,follow_symlinks=False)
    if (target.stat().st_ino,target.stat().st_dev)!=(s.st_ino,s.st_dev):
        raise ValueError('Actual immutable warmup inode differs after model-only pin')
    result=dict(q,source_checkpoint=str(target),path=str(target),source_inode=s.st_ino,
        source_device=s.st_dev,source_bytes=s.st_size,source_sha256=CP5_SHA,
        parent_source_qualification_sha256=h['accepted_source_qualification_sha256'],
        retained_protected_source=str(source),reused_accepted_cpu_qualification=True,cpu_load_not_reexecuted=True,
        current_tensor_deserializations=0,current_cpu_checkpoint_tensor_loads=0,current_full_checkpoint_byte_hash_reads=1,
        model_only_import=True,warmup_optimizer_imported=False,warmup_rng_imported=False,
        cuda_reload_verified=False,observed_unix=time.time())
    _once(base/'preparation/source-qualification.json',result)
    return result


def _verify_assets(c, q):
    import pyarrow.parquet as pq
    if c['model_revision']!='aa8e72537993ba99e69dfaafa59ed015b17504d1':
        raise ValueError('Official initial model revision differs')
    official=Path(c['official_model_dir'])
    if any(Path(c[k]).resolve()!=official.resolve() for k in ('model_dir','encoder_model_dir')):
        raise ValueError('Fresh actor and representation model must use the same official initial assets')
    verified=[]
    for name,expected in c['model_files_sha256'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or _sha(official/rel)!=expected:
            raise ValueError('Actual official model/tokenizer file SHA differs: '+name)
        verified.append(dict(role='official_model',name=name,sha256=expected))
    if c['common_manifest_sha256']!='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24':
        raise ValueError('Original common split manifest identity differs')
    assets=(('common_manifest_path','common_manifest_sha256',None),
        ('canonical_train','canonical_train_sha256',169028),
        ('validation_questions','validation_questions_sha256',512),
        ('final_questions','final_questions_sha256',51713),
        ('source_question_pool','source_question_pool_sha256',None),
        ('fixed_test_identity_path','fixed_test_identity_sha256',None),
        ('grounding_calibration_questions','grounding_calibration_sha256',300))
    for key,sha_key,count in assets:
        path=Path(c[key])
        if not str(path).startswith('/data/') or _sha(path)!=c[sha_key]:
            raise ValueError('Actual accepted input SHA differs: '+key)
        if count is not None and pq.read_metadata(path).num_rows!=count:
            raise ValueError('Actual canonical split/calibration row count differs: '+key)
        verified.append(dict(role=key,path=str(path),sha256=c[sha_key],expected_rows=count))
    from reproduction.prefix_selection200.pool import fixed_test_identities
    fixed=fixed_test_identities(c)
    if fixed is None or len(fixed)!=512 or q['source_feature_manifest_sha256']!=c['fixed_test_manifest_sha256']:
        raise ValueError('Accepted original fixed512 identities differ from retained warmup representation binding')
    feature_manifest=Path(c['fixed_test_features'])/'manifest.json'
    if _sha(feature_manifest)!=c['fixed_test_manifest_sha256']:
        raise ValueError('Accepted fixed512 feature manifest bytes differ; feature tensors are not rehashed')
    verified.append(dict(role='accepted_fixed_test_feature_manifest',path=str(feature_manifest),
        sha256=c['fixed_test_manifest_sha256'],feature_tensor_reads=0))
    from reproduction.common8.verify_data import probe_question_id
    calibration_ids=[probe_question_id(prompt) for prompt in pq.read_table(c['grounding_calibration_questions'],columns=['prompt'])['prompt'].to_pylist()]
    if calibration_ids!=c['grounding_calibration_question_ids']:
        raise ValueError('Retain exactly the original ordered300 grounding calibration identities')
    from reproduction.prefix_selection200.grounding import validate_config,_manifest_state
    validate_config(c)
    grounding=_manifest_state(c,c['grounding_manifest'],digest=c['grounding_manifest_sha256'])
    if (grounding['round']!=0 or grounding['version']!=c['grounding_probe_version']
            or any(c['model_files_sha256'].get(name)!=digest for name,digest in grounding['encoder_sha256'].items())):
        raise ValueError('Retained real round0 grounding fit must bind the same official actor0 encoder')
    verified.append(dict(role='retained_round0_grounding',manifest_sha256=grounding['manifest_sha256'],
        round=grounding['round'],version=grounding['version'],head_sha256=grounding['head_sha256']))
    return verified


def prepare_worker(handle_path):
    h=json.loads(Path(handle_path).read_text());base=Path(h['base']);receipt=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],identity=previous._identity(),checked_unix=time.time(),fresh_actor=True)
    try:
        _once(base/'preparation/source-cpu-prepare-intent.json',receipt)
        human=_pinned_json(base,h,'human-intent.json');terminal=_pinned_json(base,h,'previous-terminal.json')
        if isinstance(terminal.get('stdout'),str):terminal=json.loads(terminal['stdout'])
        cleanup=_pinned_json(base,h,'wrong-run-cleanup-receipt.json');accepted=_pinned_json(base,h,'accepted-source-qualification.json')
        if (terminal.get('terminal') is not True or cleanup.get('protected_inputs_retained') is not True
                or cleanup.get('status')!='EXPLICIT_WRONG_RUN_OUTPUTS_DELETED_PROTECTED_ASSETS_RETAINED'
                or human.get('authorized_target_thread_id')!='01a0fb1b-d3ff-74b3-aa13-49ca57871f92'):
            raise ValueError('Actual human-stop-cleanup prerequisites are not closed')
        c=json.loads(Path(h['template_config_path']).read_text())
        if (c['run_id']!=h['run_id'] or c['execution_id']!=h['execution_id'] or c['source_commit']!=h['source_commit'] or c['base']!=str(base)
                or any(c.get(k) for k in ('resume_joint','resume_pending_probe','resume_import','continue_reuse'))):
            raise ValueError('Fresh source identity cannot inherit a prior training state')
        from reproduction.prefix_selection200.interaction_protocol import CONTRACT
        if c.get('interaction_protocol')!=CONTRACT:
            raise ValueError('Bind exactly the new shared train/QA BF16 interaction contract')
        source=_accepted_model_only(base,h,cleanup,accepted)
        c.update(probe_source_checkpoint=source['source_checkpoint'],probe_source_sha256=CP5_SHA,
            probe_source_bytes=source['source_bytes'],probe_source_inode=source['source_inode'],probe_source_epoch=5,
            probe_source_model_kind='ema',probe_source_run='8af926a49630428097b94027864da68b',
            probe_source_cpu_proof=str(base/'preparation/source-qualification.json'),
            probe_source_cpu_proof_sha256=_sha(base/'preparation/source-qualification.json'),
            accepted_source_qualification_sha256=h['accepted_source_qualification_sha256'],
            actor_model_dir=c['official_model_dir'],actor_model_files_sha256=c['model_files_sha256'],
            actor_version=h['run_id']+'-official0',pool_state_path=str(base/'question-pool/state.json'),
            prefix_probe_receipt=str(base/'probe/latest.json'),fresh_actor_optimizer_scheduler_rng_sampler=True,
            retained_warmup_model_only=True,old_actor_state_imported=False,old_online_probe_optimizer_imported=False)
        verified=_verify_assets(c,source)
        mapping=_idle_mapping(base,receipt)
        for key in ('physical_gpu_indices','physical_gpu_uuids','auxiliary_gpu_indices','auxiliary_gpu_uuids'):
            c[key]=mapping[key]
        from reproduction.prefix_selection200.launch import validate_config
        validate_config(c,base);_once(base/'preparation/launch-config.json',c)
        complete=dict(receipt,status='A800_I1_I6_FRESH_CPU_PREPARE_COMPLETE',source_sha256=CP5_SHA,source_epoch=5,
            model_only_import=True,source_qualification_sha256=_sha(base/'preparation/source-qualification.json'),
            launch_config_sha256=_sha(base/'preparation/launch-config.json'),verified_assets=verified,
            interaction_protocol=CONTRACT,grounding_refit_rounds=c['grounding_refit_rounds'],
            grounding_calibration_sha256=c['grounding_calibration_sha256'],checkpoint_tensor_loads=0,
            new_generation=0,new_em_scoring=0,old_wrong_outputs_cleanup_verified=True,completed_unix=time.time())
        _once(base/'preparation/source-cpu-prepare-complete.json',complete)
        _once(base/'preparation/source-cpu-prepare-exit.json',dict(exit_code=0,identity=receipt['identity'],complete_sha256=_sha(base/'preparation/source-cpu-prepare-complete.json')))
        receipt.update(exit_code=0,status=complete['status'],qualification_sha256=complete['source_qualification_sha256'],
            config_sha256=complete['launch_config_sha256'],source_cpu_complete_sha256=_sha(base/'preparation/source-cpu-prepare-complete.json'),completed_unix=time.time())
    except BaseException as error:
        receipt.update(exit_code=1,error_class=type(error).__name__,error=str(error),completed_unix=time.time(),failed_namespace_preserved=True,same_onceclaim_must_not_be_replayed=True)
        _once(base/'preparation/prepare-exit.json',receipt);traceback.print_exc();raise
    _once(base/'preparation/prepare-exit.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('stage','prepare','launch','observe'))
    parser.add_argument('--handle',type=Path,required=True);parser.add_argument('--payload',type=Path)
    parser.add_argument('--prepare-worker',action='store_true');args=parser.parse_args()
    if not args.handle.is_absolute():parser.error('--handle must be absolute')
    if args.prepare_worker:prepare_worker(args.handle);return
    if args.action is None or (args.action=='stage' and args.payload is None):parser.error('Action required; stage also needs payload')
    h=json.loads(args.handle.read_text());payload=dict(handle=h)
    if args.action=='stage':
        staged=json.loads(args.payload.read_text())
        if 'handle' in staged:raise ValueError('Stage cannot replace its explicitly chosen handle')
        payload.update(staged)
    code={'stage':STAGE,'prepare':PREPARE,'launch':LAUNCH,'observe':OBSERVE}[args.action];started=time.time()
    try:
        result=subprocess.run(route(code),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
        record=dict(action=args.action,checked_unix=time.time(),elapsed_seconds=time.time()-started,exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode(),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],remote_mutations=args.action!='observe')
    except subprocess.TimeoutExpired as error:
        record=dict(action=args.action,checked_unix=time.time(),elapsed_seconds=time.time()-started,exit_code=None,timed_out=True,remote_action_outcome_unknown=True,mutation_must_not_be_replayed=True,stdout=(error.stdout or b'').decode(errors='replace'),stderr=(error.stderr or b'').decode(errors='replace'),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'])
    out=ROOT/'preparation'/('a800-i1-i6-fresh-'+args.action+'-'+str(time.time_ns())+'.json');out.write_text(json.dumps(record,indent=2)+'\n')
    if record.get('timed_out'):raise RuntimeError('Transport outcome unknown; do not replay. Receipt '+str(out))
    if record['exit_code']:raise RuntimeError('Actual action failed; receipt '+str(out)+': '+record['stderr'][:1000])
    if len(result.stdout)>MAX_OUTPUT:raise ValueError('Bounded output exceeded; retain unknown')
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__=='__main__':main()
