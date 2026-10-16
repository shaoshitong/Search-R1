"""Read existing question5 cache/Adam/EMA/epoch metadata over one bounded SSH.

The deployed slim controller and training source remain unchanged. Missing
producer records are unknown; this reader never loads features or checkpoints.
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
CONTROL = Path(__file__).with_name('a800_question5_slim_warmup_control.py')
CONTROL_REL = 'reproduction/operations/a800_question5_slim_warmup_control.py'
EMIT = 'raw=json.dumps(v,allow_nan=False);assert len(raw.encode())<=128*1024;print(raw)'

EXTRA = r'''
MAX_METADATA=64*1024**2
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def project(value,keys):return {key:value[key] for key in keys if key in value}
def document(name):
 q=b/name
 assert q.resolve()==q and q.is_relative_to(b) and q.suffix=='.json','Unsafe metadata path'
 if not q.exists():return None,None
 stat=q.stat();assert q.is_file() and stat.st_uid==os.getuid() and stat.st_size<=MAX_METADATA
 with q.open('rb') as f:raw=f.read(MAX_METADATA+1)
 assert len(raw)<=MAX_METADATA
 value=json.loads(raw);assert isinstance(value,dict)
 return value,dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def same(value):
 assert all(value.get(key)==h[key] for key in ('run_id','execution_id','source_commit')),'Metadata execution mismatch'
def counter(value):
 keys=('optimizer_steps','ema_updates','complete_epochs','global_groups_seen','global_trajectories_seen','rejected_nonfinite_updates')
 assert isinstance(value,dict) and all(type(value.get(key)) is int and value[key]>=0 for key in keys)
 assert value['global_trajectories_seen']==5*value['global_groups_seen']
 return project(value,keys)
def identity(value,rank):
 same(value);assert value['rank']==rank and value['local_rank']==rank and value['world_size']==4
 result=project(value,('pid','parent_pid','birth','sid','uid','rank','local_rank','world_size','run_id','execution_id',
                      'source_commit','config_sha256','physical_gpu_uuid','observed_unix'))
 result['alive']=alive(dict(pid=value['pid'],start_ticks=value['birth'],sid=value['sid'],uid=value['uid'])) if type(value.get('birth')) is int else None
 return result

# CPU preparation seals contain no feature tensors. Aggregate source scans;
# the final verification's large per-source array inventory stays on the host.
plan,_plan_proof=document('preparation/source-plan.json')
archive_plan=plan.get('reference_archive',plan.get('prepared_reference_archive',{}))
extractions=[]
for directory in ('reference','reference5'):
 value,proof=document(directory+'/EXTRACTION-COMPLETE.json')
 if value is not None:
  if archive_plan.get('concatenated_sha256') is not None:
   assert value['archive_sha256']==archive_plan['concatenated_sha256']
  safe={key:value.get(key) for key in ('archive_sha256','source_archive_sha256','internal_manifest_sha256',
   'all_members_verified','files_verified','train_trajectories','test_trajectories','full_responses_extracted',
   'weights_extracted','no_original_large_archive_transfer')}
  extractions.append(dict(proof,value=safe))
sources=b/'data/sources'
scan_summary=None
if sources.exists():
 assert sources.resolve()==sources and sources.is_dir() and sources.stat().st_uid==os.getuid()
 directories=sorted(q for q in sources.iterdir() if q.is_dir())
 assert len(directories)<=512,'Unexpected source directory inventory'
 seals=[];unsealed=[];split_counts={'train':0,'test':0};by_kind={};last=[]
 for directory in directories:
  name=directory.name
  assert re.fullmatch(r'(?:h100-epoch-\d{4}|a800-chunk-\d{4}|fixed-test)',name),'Unknown source directory'
  assert directory.resolve()==directory and directory.stat().st_uid==os.getuid()
  scan,proof=document(str((directory/'scan-complete.json').relative_to(b)))
  if scan is None:unsealed.append(name);continue
  count=scan['count'];counts=scan['split_counts']
  assert type(count) is int and count>=0 and count%5==0
  assert set(counts)=={'train','test'} and all(type(n) is int and n>=0 and n%5==0 for n in counts.values())
  assert sum(counts.values())==count
  for split in split_counts:split_counts[split]+=counts[split]
  kind='reference_train' if name.startswith('h100-') else ('a800_train' if name.startswith('a800-') else 'fixed_test')
  summary=by_kind.setdefault(kind,dict(sealed_sources=0,trajectories=0))
  summary['sealed_sources']+=1;summary['trajectories']+=count
  seals.append(dict(source=name,sha256=proof['sha256']))
  last.append((directory.joinpath('scan-complete.json').stat().st_mtime_ns,name))
 scan_summary=dict(source_directories=len(directories),sealed_source_directories=len(seals),
  unsealed_source_directories=len(unsealed),latest_sealed_source_names=[name for _mtime,name in sorted(last)[-6:]],
  unsealed_source_names=unsealed[-6:],sealed_trajectory_counts_by_split=split_counts if seals else None,
  sealed_group_counts_by_split={key:n//5 for key,n in split_counts.items()} if seals else None,
  sealed_trajectories=sum(split_counts.values()) if seals else None,
  sealed_groups=sum(split_counts.values())//5 if seals else None,sealed_counts_by_source_kind=by_kind if seals else None,
  source_scan_seals_sha256=hashlib.sha256(canonical(seals)).hexdigest() if seals else None,
  latest_seal_mtime_ns=max(mtime for mtime,_name in last) if last else None)
composite,composite_proof=document('data/composite-manifest.json')
composite_summary=None
if composite is not None:
 assert composite['source_plan_path']==str(b/'preparation/source-plan.json') and composite['source_plan_sha256']==h['source_plan_sha256']
 assert composite['role']=='unified_question_policy5_complete_event_dataset' and composite['prefix_tokens']==5
 composite_summary=dict(composite_proof,value=project(composite,('role','prefix_tokens','encoder_revision','question_max_tokens',
  'expected_train_groups','expected_test_groups','total_trajectory_events','source_plan_sha256',
  'identical_original_teacher_in_both_archives_counted_once','original_g5_sampling_events_preserved','new_rollouts','new_em_scoring')),
  source_count=len(composite['sources']))
prepared,prepared_proof=document('data/PREPARE-COMPLETE.json')
prepared_summary=None
if prepared is not None:
 assert prepared['stage']=='PINNED_PREFIX5_COMPOSITE_PREPARED'
 assert composite_proof is not None and prepared['composite_manifest_sha256']==composite_proof['sha256']
 verification=prepared['verification'];assert verification['manifest_sha256']==composite_proof['sha256']
 prepared_summary=dict(prepared_proof,value=project(prepared,('stage','composite_manifest_sha256','train_groups','train_trajectories',
  'test_groups','test_trajectories','new_rollouts','new_em_scoring','gpu_used','finished_unix')),
  verification=project(verification,('manifest_sha256','split','accepted_event_groups','exact_duplicate_trajectories_removed',
   'fixed_test_qid_and_normalized_text_disjoint','prefix_tokens','prefix_window_unified_to_five','full_file_sha_verified',
   'retained_row_payloads_verified')),verified_source_count=len(verification['source_summaries']))
 v['metadata']['data/PREPARE-COMPLETE.json']=prepared_summary
cpu_prepare=dict(extractions=extractions if extractions else None,source_scans=scan_summary,
 composite_manifest=composite_summary,complete=prepared_summary)

# Cache records bind to this deployment's generated data manifest and exact
# published cache/data/encoder source, rather than carrying fabricated run IDs.
data_path=b/'data/composite-manifest.json'
data_sha=None
if data_path.exists():
 assert data_path.resolve()==data_path and data_path.stat().st_size<=MAX_METADATA
 data_sha=sha(data_path)
expected_cache_source={name:h['source_files']['reproduction/question5_ema_warmup/'+name] for name in ('cache.py','data.py','encoder.py')}
def cache_binding(value):
 assert value['source_manifest_path']==str(data_path) and value['source_manifest_sha256']==data_sha
 assert value['source_sha256']==expected_cache_source and value['role']=='question5_packed_bf16_features_v1'

cache_workers=[]
for rank in range(4):
 prefix='features/worker-%03d/'%rank
 claim,claim_proof=document(prefix+'claim.json')
 row=dict(worker_index=rank,claim=claim_proof,latest_progress=None,complete=None,producer_alive=None)
 if claim is not None:
  cache_binding(claim);assert claim['worker_index']==rank and claim['world_size']==4
  row.update(assigned_groups=claim['assigned_groups'],total_groups=claim['total_groups'],producer_alive=alive(claim['producer_identity']))
  paths=sorted(q for q in (b/prefix).glob('progress-*.json') if re.fullmatch(r'progress-\d{6}\.json',q.name))
  if paths:
   progress_value,proof=document(str(paths[-1].relative_to(b)))
   for key in ('completed_groups','reused_trajectories','newly_encoded_trajectories'):
    assert type(progress_value[key]) is int and progress_value[key]>=0
   assert progress_value['completed_groups']<=claim['assigned_groups']
   assert progress_value['completed_groups']*5==progress_value['reused_trajectories']+progress_value['newly_encoded_trajectories']
   row['latest_progress']=dict(proof,value=project(progress_value,('completed_groups','reused_trajectories','newly_encoded_trajectories','checked_unix')))
  complete,proof=document(prefix+'complete.json')
  if complete is not None:
   cache_binding(complete);assert complete['worker_index']==rank and complete['world_size']==4
   assert complete['stage']=='QUESTION5_PACKED_CACHE_WORKER_COMPLETE'
   row['complete']=dict(proof,value=project(complete,('stage','assigned_groups','reused_trajectories','newly_encoded_trajectories',
                                                     'encoder_optimizer_updates','encoded_padding_tokens','completed_unix')))
 cache_workers.append(row)
cache,cache_proof=document('features/manifest.json')
cache_summary=None
if cache is not None:
 cache_binding(cache);assert cache['stage']=='QUESTION5_PACKED_FEATURE_CACHE_COMPLETE'
 assert cache['groups_by_split']==dict(train=445440,test=512)
 assert cache['trajectories_by_split']==dict(train=2227200,test=2560)
 cache_summary=dict(cache_proof,value=project(cache,('stage','source_manifest_sha256','source_sha256','question_max_tokens','max_tokens',
  'feature_dim','prefix_tokens','feature_dtype','groups_by_split','trajectories_by_split','reused_trajectories',
  'newly_encoded_trajectories','padding_stored','encoder_optimizer_updates','checked_unix')))

parent_acceptance,parent_acceptance_proof=document('logs/cached-parent-acceptance.json')
parent_cache_summary=None
if parent_acceptance is not None:
 same(parent_acceptance)
 assert parent_acceptance['stage']=='QUESTION5_PARENT_CACHE_ACCEPTED_FOR_FRESH_STARTUP_REPAIR'
 parent=h['parent_handle'];parent_base=pathlib.Path(h['parent_cache_base'])
 assert str(parent_base)==parent['base'] and parent_base!=b and parent_base.is_absolute()
 expected_parent=dict(parent_cache_base=str(parent_base),parent_run_id=parent['run_id'],
  parent_execution_id=parent['execution_id'],parent_source_commit=parent['source_commit'],
  parent_handle_sha256=h['parent_handle_sha256'],parent_source_plan_sha256=parent['source_plan_sha256'],
  parent_data_manifest_path=str(parent_base/'data/composite-manifest.json'),
  parent_data_manifest_sha256=h['parent_data_manifest_sha256'],
  parent_feature_manifest_path=str(parent_base/'features/manifest.json'),parent_feature_manifest_sha256=h['parent_manifest_sha256'],
  parent_prepare_complete_sha256=h['parent_prepare_complete_sha256'],
  parent_terminal_receipt_sha256=h['parent_terminal_receipt_sha256'],
  parent_acceptance_receipt_sha256=h['parent_acceptance_receipt_sha256'])
 assert all(parent_acceptance.get(key)==value for key,value in expected_parent.items())
 assert parent_acceptance['parent_exit_code']==1 and parent_acceptance['parent_epoch']==0
 assert parent_acceptance['parent_counters'] is None and parent_acceptance['parent_owned_processes']==[]
 assert parent_acceptance['parent_adam_updates']==parent_acceptance['parent_ema_updates']==0
 assert parent_acceptance['groups_by_split']==dict(train=445440,test=512)
 assert parent_acceptance['trajectories_by_split']==dict(train=2227200,test=2560)
 assert parent_acceptance['packed_features_reused']==2229760
 assert parent_acceptance['newly_encoded_in_parent']+parent_acceptance['reused_by_parent']==2229760
 assert parent_acceptance['cache_finalizer_exit_code']==0 and parent_acceptance['encoder_optimizer_updates']==0
 for key in ('data_or_features_copied','parent_modified','feature_tensors_read','model_tensors_read','state_resume'):
  assert parent_acceptance[key] is False
 for key in ('startup_repair','fresh_head','fresh_adam','fresh_rng'):assert parent_acceptance[key] is True
 workers=parent_acceptance['cache_workers'];assert len(workers)==4 and {worker['worker_index'] for worker in workers}==set(range(4))
 for worker in workers:
  assert worker['exit_code']==0 and re.fullmatch(r'[0-9a-f]{64}',worker['complete_sha256'])
 parent_cache_summary=dict(parent_acceptance_proof,value=project(parent_acceptance,tuple(expected_parent)+(
  'stage','checked_unix','run_id','execution_id','source_commit','parent_exit_code','parent_epoch','parent_counters',
  'parent_adam_updates','parent_ema_updates','parent_owned_processes','cache_workers','cache_finalizer_exit_code',
  'groups_by_split','trajectories_by_split','packed_features_reused','newly_encoded_in_parent','reused_by_parent',
  'data_or_features_copied','parent_modified','feature_tensors_read','model_tensors_read','startup_repair','state_resume',
  'fresh_head','fresh_adam','fresh_rng','encoder_optimizer_updates')))

execution='train/executions/'+h['execution_id']+'/'
ranks=[];typed_errors=[];typed_exits=[]
deployment,_deployment_proof=document('preparation/deployment.json')
staged_source,_staged_source_proof=document('preparation/source-receipt.json')
for rank in range(4):
 row=dict(rank=rank,identity=None,source_binding=None,initialization=None,progress=None)
 value,proof=document(execution+'identity-rank-%d.json'%rank)
 if value is not None:row['identity']=dict(proof,value=identity(value,rank))
 value,proof=document(execution+'source-binding-rank-%d.json'%rank)
 if value is not None:
  assert value['stage']=='EXACT_RUNTIME_SOURCE_VERIFIED_BEFORE_NCCL'
  who=identity(value['identity'],rank);binding=value['binding']
  assert binding['source_commit']==h['source_commit']
  if binding['source_kind']=='authenticated_staged_archive':
   assert binding['source_tree']==h['source_tree'] and binding['source_archive_sha256']==h['source_archive_sha256']
   assert binding['deployment_handle_path']==str(b/'preparation/deployment.json') and deployment==h
   assert binding['deployment_handle_sha256']==_deployment_proof['sha256']
   assert binding['source_receipt_sha256']==_staged_source_proof['sha256']
   assert binding['source_plan_sha256']==h['source_plan_sha256'] and binding['exact_inventory_verified'] is True
   assert binding['source_files_verified']==len(h['source_files'])
   assert type(binding['source_bytes_verified']) is int and binding['source_bytes_verified']>0
   assert binding['actual_source_file_digests_sha256']==hashlib.sha256(canonical(h['source_files'])).hexdigest()
   assert binding['train_source_sha256']==h['source_files']['reproduction/question5_ema_warmup/train.py']
  else:
   assert binding['source_kind']=='genuine_git_checkout' and binding['tracked_clean'] is True
  row['source_binding']=dict(proof,identity=who,stage=value['stage'],observed_unix=value['observed_unix'],
   binding=project(binding,('source_kind','source_commit','source_tree','source_archive_sha256','tracked_clean',
    'deployment_handle_path','deployment_handle_sha256','source_receipt_sha256','source_plan_sha256',
    'source_files_verified','source_bytes_verified','actual_source_file_digests_sha256','exact_inventory_verified','train_source_sha256')))
 value,proof=document(execution+'initialization-rank-%d.json'%rank)
 if value is not None:
  row['initialization']=dict(proof,identity=identity(value,rank),value=project(value,('stage','first_epoch','next_update_in_epoch','fresh','total_updates')),
                            counters=counter(value['counters']))
 value,proof=document(execution+'progress-rank-%d.json'%rank)
 if value is not None:
  who=identity(value['identity'],rank);counts=counter(value['counters'])
  assert value['stage'] in ('RETURNED_ADAM_AND_EMA','SAME_EMA_TRAIN_TEST_DIAGNOSTICS')
  assert counts['optimizer_steps']==counts['ema_updates']
  progress_value=project(value,('stage','epoch','epoch_update','observed_unix'));progress_value['counters']=counts
  if 'latest_online_raw' in value:
   raw=value['latest_online_raw'];same(raw);assert raw['rank']==rank
   assert raw['actual_adam_step']==raw['ema_updates']==counts['optimizer_steps']
   progress_value['latest_online_raw']=project(raw,('actual_adam_step','ema_updates','global_groups','global_trajectories',
    'local_groups','local_trajectories','online_raw_trajectory_bce','online_raw_group_bce','online_raw_total_loss',
    'metric_semantics','actual_optimizer_learning_rates','skipped_adam','epoch','epoch_update','updates_per_epoch','observed_unix'))
  row['progress']=dict(proof,identity=who,value=progress_value)
 ranks.append(row)
 value,proof=document(execution+'error-rank-%d.json'%rank)
 if value is not None:
  who=identity(value['identity'],rank)
  kind=value['error_type'];assert isinstance(kind,str) and re.fullmatch(r'[A-Za-z_]\w*(?:Error|Exception)?',kind)
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

def metrics_projection(value):
 same(value);counter(value['counters']);assert value['epoch']==1 and value['same_train_test_weight_kind']=='ema0.99'
 result=project(value,('epoch','run_id','execution_id','source_commit','observed_unix','ema_updates','counters',
  'train_mixed_group_balanced_accuracy','test_mixed_group_balanced_accuracy','same_train_test_weight_kind',
  'raw_online_window','epoch_seconds','execution_seconds','actor_updates','execution_cuda_peaks_per_rank'))
 for split in ('train','test'):
  metric=value[split+'_ema'];assert metric['weight_kind']=='ema0.99'
  result[split+'_ema']=project(metric,('weight_kind','metric_semantics','groups','trajectories','trajectory_bce','group_bce',
   'total_loss','mixed_group','trajectory_correctness','true_mixed','threshold'))
 return result

metrics,metrics_proof=document('train/epochs/epoch-0001/metrics.json')
checkpoint,checkpoint_proof=document('train/epochs/epoch-0001/checkpoint.json')
complete,complete_proof=document('train/epochs/epoch-0001/complete.json')
first_epoch=dict(metrics=None,checkpoint=None,complete=complete_proof,fully_committed=None)
if metrics is not None:first_epoch['metrics']=dict(metrics_proof,value=metrics_projection(metrics))
if checkpoint is not None:
 same(checkpoint);counter(checkpoint['counters'])
 assert checkpoint['epoch']==1 and checkpoint['completed_epoch'] is True
 assert checkpoint['path']==str(b/'train/latest.pt')
 first_epoch['checkpoint']=dict(checkpoint_proof,value=project(checkpoint,('path','bytes','sha256','actual_adam_steps_per_rank','ema_updates',
  'epoch','completed_epoch','run_id','execution_id','source_commit','binding_sha256','counters','epoch_metrics_sha256','observed_unix')))
if complete is not None:
 assert complete['status']=='FULL_MERGED_EPOCH_COMPLETE'
 assert metrics is not None and checkpoint is not None and complete['metrics']==metrics and complete['checkpoint']==checkpoint
 assert hashlib.sha256(canonical(metrics)).hexdigest()==checkpoint['epoch_metrics_sha256']
 assert metrics['counters']==checkpoint['counters']
 assert checkpoint['actual_adam_steps_per_rank']==[8700]*4 and checkpoint['ema_updates']==8700
 assert metrics['ema_updates']==8700 and metrics['counters']['optimizer_steps']==metrics['counters']['ema_updates']==8700
 assert metrics['counters']['complete_epochs']==1 and metrics['counters']['global_groups_seen']==445440
 assert metrics['train_ema']['groups']==445440 and metrics['test_ema']['groups']==512
 first_epoch['fully_committed']=True

# Retain the original guarded controller observation, but replace its cross-
# execution glob with the exact current execution's four typed rank records.
v.update(stage='QUESTION5_SLIM_ACTUAL_PROGRESS_METADATA_OBSERVATION',base=str(b),cpu_prepare=cpu_prepare,cache_workers=cache_workers,
 cache_manifest=cache_summary,cached_parent_acceptance=parent_cache_summary,
 rank_progress=ranks,rank_errors=typed_errors,rank_exits=typed_exits,first_epoch=first_epoch,
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
    if (not re.fullmatch(r'/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-[0-9a-f]{32}', handle['base'])
            or handle['fit_config']['world_size'] != 4):
        raise ValueError('Require the bound slim question5 four-rank deployment')
    controller_sha = hashlib.sha256(CONTROL.read_bytes()).hexdigest()
    if controller_sha != handle['source_files'][CONTROL_REL]:
        raise ValueError('Local inherited controller differs from this deployed handle')
    spec = importlib.util.spec_from_file_location('published_question5_slim_control', CONTROL)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    if control.OBSERVE.count(EMIT) != 1:
        raise ValueError('Published observation emission contract changed')
    # Its complete proof grows with259 sources; project it under the explicit
    # 64MiB metadata cap instead of inheriting the old256KiB/full-output read.
    inherited = control.OBSERVE.replace("'data/PREPARE-COMPLETE.json',", '', 1)
    code = inherited.replace(EMIT, EXTRA + '\n' + EMIT)
    observer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = dict(handle=handle, observer_source_sha256=observer_sha, controller_source_sha256=controller_sha)
    started = time.time()
    result = subprocess.run(control.route(code), input=json.dumps(payload).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
    if len(result.stdout) > 128 * 1024:
        raise ValueError('Observer output exceeds 128KiB')
    record = dict(role='question5_slim_actual_progress_observer', checked_unix=time.time(),
                  elapsed_seconds=time.time() - started, exit_code=result.returncode,
                  run_id=handle['run_id'], execution_id=handle['execution_id'], source_commit=handle['source_commit'],
                  base=handle['base'], observer_source_sha256=observer_sha, controller_source_sha256=controller_sha,
                  stdout=result.stdout.decode(), stderr=result.stderr[-16384:].decode(errors='replace'))
    output = ROOT / 'preparation' / ('question5-slim-progress-observation-%d.json' % time.time_ns())
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if result.returncode:
        raise RuntimeError('Read-only metadata observation failed; retained receipt ' + str(output))
    print(json.dumps(dict(receipt=str(output), result=json.loads(result.stdout)), allow_nan=False))


if __name__ == '__main__':
    main()
