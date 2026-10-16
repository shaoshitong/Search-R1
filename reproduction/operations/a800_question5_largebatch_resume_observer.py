"""Observe one explicit full-state question5 performance resume, metadata only.

This reader accepts only the isolated same-run resume contract. The preceding
execution's first complete epoch remains a separately identified inherited
prefix; four current restore receipts and real Adam/EMA advances prove the new
execution. It never loads checkpoints, feature tensors or model payloads.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
CONTROL = Path(__file__).with_name('a800_question5_largebatch_resume_control.py')
CONTROL_REL = 'reproduction/operations/a800_question5_largebatch_resume_control.py'
EMIT = 'raw=json.dumps(v,allow_nan=False);assert len(raw.encode())<=128*1024;print(raw)'

EXTRA = r'''
MAX_METADATA=64*1024**2
metadata_bytes=0
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def project(value,keys):return {key:value[key] for key in keys if key in value}
def document_at(root,name):
 global metadata_bytes
 q=root/name
 assert q.resolve()==q and q.is_relative_to(root) and q.suffix=='.json','Unsafe metadata path'
 if not q.exists():return None,None
 stat=q.stat();assert q.is_file() and stat.st_uid==os.getuid() and stat.st_size<=MAX_METADATA
 with q.open('rb') as f:raw=f.read(MAX_METADATA+1)
 assert len(raw)<=MAX_METADATA
 metadata_bytes+=len(raw);assert metadata_bytes<=MAX_METADATA,'Metadata total exceeds 64MiB'
 value=json.loads(raw);assert isinstance(value,dict)
 return value,dict(path=str(q),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def document(name):return document_at(b,name)
def same(value,expected=h):
 assert all(value.get(key)==expected[key] for key in ('run_id','execution_id','source_commit')),'Metadata execution mismatch'
def counter(value):
 keys=('optimizer_steps','ema_updates','complete_epochs','global_groups_seen','global_trajectories_seen','rejected_nonfinite_updates')
 assert isinstance(value,dict) and all(type(value.get(key)) is int and value[key]>=0 for key in keys)
 assert value['optimizer_steps']==value['ema_updates']
 assert value['global_trajectories_seen']==5*value['global_groups_seen']
 return project(value,keys)
def identity(value,rank):
 same(value);assert value['rank']==rank and value['local_rank']==rank and value['world_size']==4
 result=project(value,('pid','parent_pid','birth','sid','uid','rank','local_rank','world_size','run_id','execution_id',
  'source_commit','config_sha256','physical_gpu_uuid','observed_unix'))
 result['alive']=alive(dict(pid=value['pid'],start_ticks=value['birth'],sid=value['sid'],uid=value['uid'])) if type(value.get('birth')) is int else None
 return result

parent=h['resume_parent_handle'];parent_base=pathlib.Path(parent['base'])
assert parent['run_id']==h['run_id'] and parent['execution_id']!=h['execution_id'] and parent_base!=b
assert parent_base.resolve()==parent_base and parent_base.is_absolute()
parent_deployment,parent_deployment_proof=document('preparation/resume-parent-deployment.json')
assert parent_deployment==parent and parent_deployment_proof['sha256']==h['resume_parent_handle_sha256']
assert hashlib.sha256(canonical(h['parent_binding'])).hexdigest()==h['parent_binding_sha256']
qualification,qualification_proof=document('preparation/resume-qualification.json')
acceptance,acceptance_proof=document('logs/full-state-resume-acceptance.json')
resume_acceptance=None
if acceptance is not None:
 assert qualification is not None,'Missing same-execution full-state qualification'
 # Controller writes the exact qualification after actual old-process closure.
 assert acceptance==qualification
 assert qualification['role']=='question5_native_stopped_full_state_resume_v1'
 assert qualification['run_id']==h['run_id']
 assert qualification['parent_execution_id']==parent['execution_id'] and qualification['parent_source_commit']==parent['source_commit']
 assert qualification['parent_binding_sha256']==h['parent_binding_sha256']
 assert qualification['target_execution_id']==h['execution_id'] and qualification['target_source_commit']==h['source_commit']
 assert qualification['target_output_dir']==str(b/'train')
 for key in ('all_owned_training_processes_gone','checkpoint_writer_gone','old_output_preserved'):
  assert qualification[key] is True
 retained=counter(qualification['retained_counters'])
 assert retained['complete_epochs']>=1 and retained['optimizer_steps']>=8700
 checkpoint_seal=qualification['checkpoint']
 assert checkpoint_seal['path']==str(parent_base/'train/latest.pt')
 assert re.fullmatch(r'[0-9a-f]{64}',checkpoint_seal['sha256'])
 assert type(checkpoint_seal['bytes']) is int and checkpoint_seal['bytes']>0
 latest,latest_proof=document_at(parent_base,'train/latest.json')
 assert latest is not None and latest==qualification['parent_latest_receipt']
 assert latest_proof['sha256']==qualification['parent_latest_receipt_sha256']
 same(latest,parent);assert counter(latest['counters'])==retained
 assert latest['path']==checkpoint_seal['path'] and latest['sha256']==checkpoint_seal['sha256']
 assert latest['bytes']==checkpoint_seal['bytes'] and latest['binding_sha256']==h['parent_binding_sha256']
 assert latest['actual_adam_steps_per_rank']==[retained['optimizer_steps']]*4 and latest['ema_updates']==retained['ema_updates']
 assert latest['epoch']==qualification['retained_epoch'] and latest['next_update_in_epoch']==qualification['retained_next_update_in_epoch']
 assert latest['completed_epoch']==qualification['completed_epoch']
 terminal_seals=qualification['rank_terminal_receipts']
 assert len(terminal_seals)==4 and {item['rank'] for item in terminal_seals}==set(range(4))
 for item in terminal_seals:
  rank=item['rank'];prefix='train/executions/'+parent['execution_id']+'/'
  ended,ended_proof=document_at(parent_base,prefix+'exit-rank-%d.json'%rank)
  assert ended is not None and ended_proof['sha256']==item['exit']['sha256']
  assert ended_proof['path']==item['exit']['path']
  same(ended['identity'],parent);assert ended['identity']['rank']==rank and counter(ended['counters'])==retained
  who=ended['identity']
  assert alive(dict(pid=who['pid'],start_ticks=who['birth'],sid=who['sid'],uid=who['uid'])) is False
  terminal='STOPPED_AT_COMPLETE_EPOCH_BOUNDARY' if latest['completed_epoch'] else 'STOPPED_AT_COMPLETE_ADAM_BOUNDARY'
  assert ended['terminal']==item['terminal']==terminal and ended['epoch']==latest['epoch']
  if not latest['completed_epoch']:
   stopped,stopped_proof=document_at(parent_base,prefix+'stopped-rank-%d.json'%rank)
   assert stopped is not None and stopped_proof['sha256']==item['stopped']['sha256'] and stopped_proof['path']==item['stopped']['path']
   same(stopped['identity'],parent);assert stopped['identity']['rank']==rank
   assert counter(stopped['counters'])==retained and stopped['checkpoint']==latest
 resume_acceptance=dict(acceptance_proof,value=project(qualification,('role','run_id','parent_execution_id','parent_source_commit',
  'parent_binding_sha256','target_execution_id','target_source_commit','target_output_dir','checkpoint',
  'retained_counters','all_owned_training_processes_gone',
  'checkpoint_writer_gone','old_output_preserved','retained_epoch','retained_next_update_in_epoch','completed_epoch','checked_unix')),
  qualification_sha256=qualification_proof['sha256'])
else:retained=None

execution='train/executions/'+h['execution_id']+'/'
deployment,deployment_proof=document('preparation/deployment.json')
staged_source,staged_source_proof=document('preparation/source-receipt.json')
assert deployment==h
native_binding,native_binding_proof=document(execution+'binding.json')
current_native_binding=None
if native_binding is not None:
 current_native_binding=native_binding['binding']
 assert current_native_binding['run_id']==h['run_id'] and current_native_binding['source_commit']==h['source_commit']
 assert set(current_native_binding)==set(h['parent_binding'])
 differences={key for key in current_native_binding if current_native_binding[key]!=h['parent_binding'][key]}
 assert differences=={'micro_groups','source_commit'} and current_native_binding['micro_groups']==13
 assert native_binding['binding_sha256']==hashlib.sha256(canonical(current_native_binding)).hexdigest()
ranks=[];typed_errors=[];typed_exits=[]
for rank in range(4):
 row=dict(rank=rank,identity=None,source_binding=None,initialization=None,restore=None,progress=None)
 actual_runtime_binding=None
 value,proof=document(execution+'identity-rank-%d.json'%rank)
 if value is not None:row['identity']=dict(proof,value=identity(value,rank))
 value,proof=document(execution+'source-binding-rank-%d.json'%rank)
 if value is not None:
  assert value['stage']=='EXACT_RUNTIME_SOURCE_VERIFIED_BEFORE_NCCL'
  who=identity(value['identity'],rank);binding=value['binding']
  assert binding['source_commit']==h['source_commit'] and binding['source_kind']=='authenticated_staged_archive'
  assert binding['source_tree']==h['source_tree'] and binding['source_archive_sha256']==h['source_archive_sha256']
  assert binding['deployment_handle_path']==str(b/'preparation/deployment.json') and binding['deployment_handle_sha256']==deployment_proof['sha256']
  assert binding['source_receipt_sha256']==staged_source_proof['sha256'] and binding['source_plan_sha256']==h['source_plan_sha256']
  assert binding['exact_inventory_verified'] is True and binding['source_files_verified']==len(h['source_files'])
  assert type(binding['source_bytes_verified']) is int and binding['source_bytes_verified']>0
  assert binding['actual_source_file_digests_sha256']==hashlib.sha256(canonical(h['source_files'])).hexdigest()
  assert binding['train_source_sha256']==h['source_files']['reproduction/question5_ema_warmup/train.py']
  actual_runtime_binding=binding
  row['source_binding']=dict(proof,identity=who,stage=value['stage'],observed_unix=value['observed_unix'],
   binding=project(binding,('source_kind','source_commit','source_tree','source_archive_sha256','deployment_handle_path',
    'deployment_handle_sha256','source_receipt_sha256','source_plan_sha256','source_files_verified','source_bytes_verified',
    'actual_source_file_digests_sha256','exact_inventory_verified','train_source_sha256')))
 value,proof=document(execution+'initialization-rank-%d.json'%rank)
 if value is not None:
  who=identity(value,rank);counts=counter(value['counters'])
  assert value['stage']=='READY_FOR_COMPLETE_DATASET_EPOCHS' and value['fresh'] is False and value['total_updates']==1740000
  assert retained is not None and counts==retained
  row['initialization']=dict(proof,identity=who,value=project(value,('stage','first_epoch','next_update_in_epoch','fresh','total_updates')),
   counters=counts)
 value,proof=document(execution+'restore-rank-%d.json'%rank)
 if value is not None:
  who=identity(value['identity'],rank)
  # The native producer emits this only after raw/EMA/Adam and rank RNG restore.
  assert value['stage']=='FULL_STATE_RESTORED_BEFORE_NEXT_UPDATE'
  assert retained is not None and counter(value['counters'])==retained
  assert value['parent_checkpoint']==checkpoint_seal['path'] and value['parent_checkpoint_sha256']==checkpoint_seal['sha256']
  assert value['parent_execution_id']==parent['execution_id'] and value['parent_source_commit']==parent['source_commit']
  assert value['parent_binding_sha256']==h['parent_binding_sha256']
  assert value['full_raw_ema_adam_restored'] is True and value['fresh_initialization'] is False
  assert value['partial_updates_discarded'] is False
  assert value['rng_restored']==dict(python=True,numpy=True,torch_cpu=True,torch_cuda=True,sampler=True)
  assert value['actual_adam_step']==value['ema_updates']==retained['optimizer_steps']
  assert re.fullmatch(r'[0-9a-f]{64}',value['saved_epoch_plan_sha256'])
  assert type(value['saved_epoch']) is int and value['saved_epoch']>=1
  assert type(value['saved_completed_epoch']) is bool
  assert type(value['saved_next_update_in_epoch']) is int and 0<=value['saved_next_update_in_epoch']<=8700
  assert value['restored_sampler_state']==dict(epoch=value['saved_epoch'],next_update_in_epoch=value['saved_next_update_in_epoch'],
   epoch_plan_sha256=value['saved_epoch_plan_sha256'])
  if value['saved_completed_epoch']:
   assert value['saved_epoch']==retained['complete_epochs'] and value['saved_next_update_in_epoch']==0
  else:
   assert value['saved_epoch']==retained['complete_epochs']+1
  assert retained['optimizer_steps']==retained['complete_epochs']*8700+value['saved_next_update_in_epoch']
  assert value['execution_micro_groups']==13 and value['evaluation_micro_groups']==64
  rates=value['restored_learning_rates'];assert isinstance(rates,list) and rates and all(type(rate) is float and 0<rate<=3e-5 for rate in rates)
  continuation=value['state_continuation'];assert continuation['full_state_resume'] is True
  assert continuation['kind']=='authenticated_infrastructure_transition' and continuation['parent_binding']==h['parent_binding']
  assert continuation['parent_binding_sha256']==h['parent_binding_sha256'] and continuation['parent_execution_id']==parent['execution_id']
  assert continuation['parent_source_commit']==parent['source_commit']
  assert continuation['checkpoint']==checkpoint_seal['path'] and continuation['checkpoint_sha256']==checkpoint_seal['sha256']
  assert continuation['qualification_receipt']==str(b/'preparation/resume-qualification.json')
  assert continuation['qualification_receipt_sha256']==qualification_proof['sha256']
  assert continuation['changed_binding_fields']==['micro_groups','source_commit']
  assert continuation['execution_micro_groups']==13 and continuation['evaluation_micro_groups']==64
  assert continuation['fresh_initialization'] is False and continuation['partial_updates_discarded'] is False
  assert actual_runtime_binding is not None and value['current_source_binding']==actual_runtime_binding
  assert current_native_binding is not None and value['current_binding_sha256']==native_binding['binding_sha256']
  assert native_binding['state_continuation']==continuation
  if row['initialization'] is not None:
   init=row['initialization']['value']
   expected_first=value['saved_epoch']+1 if value['saved_completed_epoch'] else value['saved_epoch']
   assert init['first_epoch']==expected_first and init['next_update_in_epoch']==value['saved_next_update_in_epoch']
  row['restore']=dict(proof,identity=who,value=project(value,('stage','parent_checkpoint','parent_checkpoint_sha256','parent_execution_id',
   'parent_source_commit','parent_binding_sha256','counters','actual_adam_step','ema_updates','saved_epoch','saved_completed_epoch',
   'saved_next_update_in_epoch','saved_epoch_plan_sha256','full_raw_ema_adam_restored','fresh_initialization','partial_updates_discarded',
   'rng_restored','restored_sampler_state','restored_learning_rates','current_binding_sha256','execution_micro_groups',
   'evaluation_micro_groups','observed_unix')))
 value,proof=document(execution+'progress-rank-%d.json'%rank)
 if value is not None:
  who=identity(value['identity'],rank);counts=counter(value['counters'])
  assert value['stage'] in ('RETURNED_ADAM_AND_EMA','SAME_EMA_TRAIN_TEST_DIAGNOSTICS')
  assert retained is not None
  for key in counts:assert counts[key]>=retained[key]
  progress_value=project(value,('stage','epoch','epoch_update','observed_unix'));progress_value['counters']=counts
  if 'latest_online_raw' in value:
   raw=value['latest_online_raw'];same(raw);assert raw['rank']==rank
   assert raw['actual_adam_step']==raw['ema_updates']==counts['optimizer_steps']
   progress_value['latest_online_raw']=project(raw,('actual_adam_step','ema_updates','global_groups','global_trajectories','local_groups',
    'local_trajectories','online_raw_trajectory_bce','online_raw_group_bce','online_raw_total_loss','metric_semantics',
    'actual_optimizer_learning_rates','skipped_adam','epoch','epoch_update','updates_per_epoch','observed_unix'))
  row['progress']=dict(proof,identity=who,value=progress_value)
 ranks.append(row)
 value,proof=document(execution+'error-rank-%d.json'%rank)
 if value is not None:
  who=identity(value['identity'],rank);kind=value['error_type']
  assert isinstance(kind,str) and re.fullmatch(r'[A-Za-z_]\w*(?:Error|Exception)?',kind)
  typed_errors.append(dict(proof,rank=rank,identity=who,error_type=kind,
   error_message_sha256=hashlib.sha256(str(value.get('error','')).encode()).hexdigest(),
   value=project(value,('epoch','next_update_in_epoch','partial_writes_not_rolled_back','observed_unix')),
   counters=counter(value['counters']) if value.get('counters') is not None else None))
 value,proof=document(execution+'exit-rank-%d.json'%rank)
 if value is not None:
  who=identity(value['identity'],rank)
  assert value['terminal'] in ('FAILED','STOPPED_AT_COMPLETE_ADAM_BOUNDARY','STOPPED_AT_COMPLETE_EPOCH_BOUNDARY','WARMUP_200_COMPLETE')
  typed_exits.append(dict(proof,rank=rank,identity=who,value=project(value,('terminal','epoch','elapsed_seconds','observed_unix')),
   counters=counter(value['counters']) if value.get('counters') is not None else None))

def metrics_projection(value,expected):
 same(value,expected);counter(value['counters']);assert value['same_train_test_weight_kind']=='ema0.99'
 result=project(value,('epoch','run_id','execution_id','source_commit','observed_unix','ema_updates','counters',
  'train_mixed_group_balanced_accuracy','test_mixed_group_balanced_accuracy','same_train_test_weight_kind','raw_online_window',
  'epoch_seconds','execution_seconds','actor_updates','execution_cuda_peaks_per_rank','execution_micro_groups','evaluation_micro_groups'))
 for split in ('train','test'):
  metric=value[split+'_ema'];assert metric['weight_kind']=='ema0.99'
  result[split+'_ema']=project(metric,('weight_kind','metric_semantics','groups','trajectories','trajectory_bce','group_bce','total_loss',
   'mixed_group','trajectory_correctness','true_mixed','threshold'))
 return result
def epoch_record(root,epoch,expected,inherited=False):
 prefix='train/epochs/epoch-%04d/'%epoch
 metrics,metrics_proof=document_at(root,prefix+'metrics.json')
 checkpoint,checkpoint_proof=document_at(root,prefix+'checkpoint.json')
 complete,complete_proof=document_at(root,prefix+'complete.json')
 result=dict(epoch=epoch,inherited_prefix=inherited,metrics=None,checkpoint=None,complete=complete_proof,fully_committed=None)
 if metrics is not None:result['metrics']=dict(metrics_proof,value=metrics_projection(metrics,expected))
 if checkpoint is not None:
  same(checkpoint,expected);counts=counter(checkpoint['counters'])
  assert checkpoint['epoch']==epoch and checkpoint['completed_epoch'] is True
  assert checkpoint['path']==str(root/'train/latest.pt')
  result['checkpoint']=dict(checkpoint_proof,value=project(checkpoint,('path','bytes','sha256','actual_adam_steps_per_rank','ema_updates',
   'epoch','completed_epoch','run_id','execution_id','source_commit','binding_sha256','counters','epoch_metrics_sha256','observed_unix')))
 if complete is not None:
  assert complete['status']=='FULL_MERGED_EPOCH_COMPLETE'
  assert metrics is not None and checkpoint is not None and complete['metrics']==metrics and complete['checkpoint']==checkpoint
  assert hashlib.sha256(canonical(metrics)).hexdigest()==checkpoint['epoch_metrics_sha256']
  assert metrics['counters']==checkpoint['counters']
  steps=epoch*8700
  assert checkpoint['actual_adam_steps_per_rank']==[steps]*4 and checkpoint['ema_updates']==steps
  assert metrics['ema_updates']==steps and metrics['counters']['optimizer_steps']==metrics['counters']['ema_updates']==steps
  assert metrics['counters']['complete_epochs']==epoch and metrics['counters']['global_groups_seen']==epoch*445440
  assert metrics['train_ema']['groups']==445440 and metrics['test_ema']['groups']==512
  assert metrics['train_ema']['trajectories']==2227200 and metrics['test_ema']['trajectories']==2560
  result['fully_committed']=True
 return result

# Reading the parent's retained scalar/hash metadata proves its completed
# prefix; no checkpoint file is opened or rehashed by the observer.
inherited_first=epoch_record(parent_base,1,parent,True)
if retained is not None:assert inherited_first['fully_committed'] is True
inherited_last=None
if retained is not None and retained['complete_epochs']>1:
 inherited_last=epoch_record(parent_base,retained['complete_epochs'],parent,True)
 assert inherited_last['fully_committed'] is True
restored=[row['restore'] for row in ranks]
initializations=[row['initialization'] for row in ranks]
progress=[row['progress'] for row in ranks]
all_restored=(all(row is not None for row in restored) and all(row is not None for row in initializations)
 and all(row['source_binding'] is not None for row in ranks))
new_advances=None;strict_rank_agreement=None
if all(row is not None for row in progress):
 counts=[row['value']['counters'] for row in progress]
 strict_rank_agreement=all(value==counts[0] for value in counts)
 if strict_rank_agreement and all_restored:
  new_advances=counts[0]['optimizer_steps']-retained['optimizer_steps']
current_epoch=None
if strict_rank_agreement:
 epoch=progress[0]['value']['counters']['complete_epochs']
 if epoch>retained['complete_epochs']:current_epoch=epoch_record(b,epoch,h)
v.update(stage='QUESTION5_LARGEBATCH_FULL_STATE_RESUME_METADATA_OBSERVATION',base=str(b),full_state_resume_acceptance=resume_acceptance,
 inherited_first_epoch=inherited_first,inherited_last_complete_epoch=inherited_last,
 inherited_complete_epochs=retained['complete_epochs'] if retained is not None else None,
 inherited_optimizer_steps=retained['optimizer_steps'] if retained is not None else None,
 inherited_ema_updates=retained['ema_updates'] if retained is not None else None,
 rank_progress=ranks,rank_errors=typed_errors,rank_exits=typed_exits,all_four_native_restores_verified=all_restored,
 current_four_rank_counters_agree=strict_rank_agreement,new_returned_adam_and_ema_updates=new_advances,
 new_execution_update_progress_proven=(new_advances>0) if new_advances is not None else None,
 latest_current_execution_complete_epoch=current_epoch,metadata_bytes_read=metadata_bytes,
 observer_source_sha256=p['observer_source_sha256'],controller_source_sha256=p['controller_source_sha256'],
 missing_metadata_semantics='unknown_not_zero',checkpoint_tensors_read=False,feature_tensors_read=False,
 probability_arrays_read=False,new_evaluation=False,checked_unix=time.time())
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle', type=Path, required=True)
    args = parser.parse_args()
    if args.handle.stat().st_size > 64 * 1024**2:
        raise ValueError('Oversized deployment handle')
    handle = json.loads(args.handle.read_bytes())
    for key, size in (('run_id', 32), ('execution_id', 32), ('source_commit', 40)):
        if re.fullmatch('[0-9a-f]{%d}' % size, handle[key]) is None:
            raise ValueError('Invalid handle ' + key)
    expected_base = '/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-' + handle['run_id'] + '-resume-' + handle['execution_id']
    if (handle['base'] != expected_base or handle['fit_config']['world_size'] != 4
            or handle['fit_config']['micro_groups'] != 13 or handle['fit_config']['eval_micro_groups'] != 64):
        raise ValueError('Require the explicit bound four-rank performance-resume deployment')
    controller_sha = hashlib.sha256(CONTROL.read_bytes()).hexdigest()
    if controller_sha != handle['source_files'][CONTROL_REL]:
        raise ValueError('Local resume controller differs from this deployed handle')
    spec = importlib.util.spec_from_file_location('published_question5_largebatch_resume_control', CONTROL)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    if control.OBSERVE.count(EMIT) != 1:
        raise ValueError('Published observation emission contract changed')
    code = control.OBSERVE.replace(EMIT, EXTRA + '\n' + EMIT)
    observer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = dict(handle=handle, observer_source_sha256=observer_sha, controller_source_sha256=controller_sha)
    started = time.time()
    result = subprocess.run(control.route(code), input=json.dumps(payload).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
    if len(result.stdout) > 128 * 1024:
        raise ValueError('Observer output exceeds 128KiB')
    record = dict(role='question5_largebatch_full_state_resume_observer', checked_unix=time.time(),
                  elapsed_seconds=time.time() - started, exit_code=result.returncode,
                  run_id=handle['run_id'], execution_id=handle['execution_id'], source_commit=handle['source_commit'],
                  base=handle['base'], observer_source_sha256=observer_sha, controller_source_sha256=controller_sha,
                  stdout=result.stdout.decode(), stderr=result.stderr[-16384:].decode(errors='replace'))
    output = ROOT / 'preparation' / ('question5-largebatch-resume-observation-%d.json' % time.time_ns())
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if result.returncode:
        raise RuntimeError('Read-only metadata observation failed; retained receipt ' + str(output))
    print(json.dumps(dict(receipt=str(output), result=json.loads(result.stdout)), allow_nan=False))


if __name__ == '__main__':
    main()
