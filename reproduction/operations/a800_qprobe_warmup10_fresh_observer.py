"""One bounded metadata read of an official fresh-zero sigma3 Top512/E0 run.

Reuse published fresh from-one counters and long-run bounded snapshots. No CP5
inheritance, tensor/QA reads, evaluation, verifier, launch, retry or anchor writes.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
FLEX_PATH = 'reproduction/operations/a800_qprobe_flex_observer.py'
FLEX_COMMIT = 'd78c9d3d111adef8ee048b7d3cda4d50dc6066e2'
FLEX_SHA = 'e1dda0a55243ac0d25ba223b3cd95acdf37725ab09da051558ce93cfb70317cf'
PREDECESSOR_PATH = 'reproduction/operations/a800_qprobe_async_milestone_observer.py'
PREDECESSOR_COMMIT = 'ade1289a25de192d1bf0e6c6bfeb8223d2b9a449'
PREDECESSOR_SHA = '322975cd5d12b1ac5bb288588fcdb6e723c0e2fa5e838be93680e5cb2f1abe68'
SUPPORT_PATH = 'reproduction/operations/a800_qprobe_saved_state_longrun_observer.py'
SUPPORT_COMMIT = '927fed246f617dc81fa96c3f0c3130c0d17ff008'
SUPPORT_SHA = 'e1c0486e1b383363419ccbca351dab4eb132b1b19803ba6fe845356216b56f8b'

REMOTE_SUPPORT = r'''
import hashlib,io,json,math,os,re,signal,stat,sys,time
from pathlib import Path
TOTAL_LIMIT=576*1024*1024
SMALL_LIMIT=2*1024*1024
LEDGER_LIMIT=8*1024*1024
RANK_LEDGER_LIMIT=32*1024*1024
PROCESS_LEDGER_LIMIT=128*1024*1024
LINE_LIMIT=128*1024
deadline=time.monotonic()+65
spent=0
metadata_files=[]
bounded_failures=[];read_pass='pinned_run'
documents={};ledgers={}
class Unknown(Exception):pass
def need(value,reason):
 if not value:raise Unknown(reason)
def check():need(time.monotonic()<deadline,'deadline')
def expired(signum,frame):raise Unknown('deadline')
signal.signal(signal.SIGALRM,expired);signal.alarm(65)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def number(value):return type(value) in (int,float) and math.isfinite(value)
def integer(value,minimum=0):return type(value) is int and value>=minimum
def take(fd,start,end):
 global spent
 check();need(0<=start<=end and spent+end-start<=TOTAL_LIMIT,'aggregate_read_limit')
 chunks=[]
 while start<end:
  check();part=os.pread(fd,min(65536,end-start),start);need(part,'truncated_file')
  chunks.append(part);spent+=len(part);start+=len(part)
 return b''.join(chunks)
def snapshot(path,limit,append=False,tail=False,record=True):
 path=Path(path);check()
 need(path.resolve()==path and not path.is_symlink(),'path_not_canonical')
 try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 except FileNotFoundError:return None,None
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid(),'owner_or_type')
  if not tail and before.st_size>limit:
   location=dict(path_scope='unknown',relative_path=None)
   for scope,name in (('runtime_source','repo'),('run_metadata','base')):
    parent=globals().get(name)
    if parent is None:continue
    try:relative=path.relative_to(parent)
    except ValueError:continue
    text=str(relative)
    if re.fullmatch(r'[A-Za-z0-9._/-]{1,512}',text) and '..' not in relative.parts:
     location=dict(path_scope=scope,relative_path=text)
    break
   if len(bounded_failures)<4:
    bounded_failures.append(dict(reason='file_limit',read_pass=read_pass,file_bytes=before.st_size,
     limit_bytes=limit,data_read=False,**location))
   raise Unknown('file_limit')
  start=max(0,before.st_size-limit) if tail else 0
  raw=take(fd,start,before.st_size);again=take(fd,start,before.st_size)
  after=os.fstat(fd);current=path.stat(follow_symlinks=False)
  for value in (after,current):
   need(stat.S_ISREG(value.st_mode) and value.st_uid==os.getuid() and
    (value.st_dev,value.st_ino)==(before.st_dev,before.st_ino) and
    (value.st_size>=before.st_size if append else value.st_size==before.st_size),'snapshot_changed')
  need(raw==again,'snapshot_prefix_rewritten')
  info=dict(path=str(path),file_bytes=before.st_size,scan_start_byte=start,scan_end_byte=before.st_size,
   scan_from_start=start==0,scanned_sha256=digest(raw),verified_initial_range_unchanged=True)
  if record:metadata_files.append(info)
  return raw,info
 finally:os.close(fd)
def budget_bytes(path,limit):
 raw,_=snapshot(path,limit,record=False);need(raw is not None,'required_source_missing');return raw
def document(relative,limit=SMALL_LIMIT):
 if (relative,limit) in documents:return documents[(relative,limit)]
 raw,info=snapshot(base/relative,limit)
 if raw is None:return None,None
 value=json.loads(raw);need(type(value) is dict,'metadata_object')
 documents[(relative,limit)]=(value,info);return value,info
def rows(relative,tail=False,limit=LEDGER_LIMIT):
 if (relative,tail,limit) in ledgers:return ledgers[(relative,tail,limit)]
 raw,info=snapshot(base/relative,limit,append=True,tail=tail)
 if raw is None:return None,None
 leading=raw.find(b'\n')+1 if info['scan_start_byte'] else 0
 if info['scan_start_byte'] and not leading:leading=len(raw)
 end=max(leading,raw.rfind(b'\n')+1);prefix=raw[leading:end]
 info.update(discarded_leading_bytes=leading,partial_final_bytes=len(raw)-end,
  partial_final_status='unknown_not_counted' if len(raw)>end else None,
  complete_prefix_sha256=digest(prefix),complete_line_count=0)
 values=[]
 for line in prefix.split(b'\n')[:-1]:
  check();need(0<len(line)<=LINE_LIMIT and len(values)<20000,'line_or_row_limit')
  value=json.loads(line);need(type(value) is dict,'ledger_object');values.append(value)
 info['complete_line_count']=len(values);ledgers[(relative,tail,limit)]=(values,info);return values,info
def scalar(value):
 return value if value is None or type(value) is bool or number(value) else None
def select(value,names):return {name:scalar(value.get(name)) for name in names}
def hash_value(value):
 need(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'hash_schema');return value
def version(value):
 need(type(value) is str and re.fullmatch('[A-Za-z0-9_.:-]{1,160}',value),'version_schema');return value
def hash_inventory(value):
 need(type(value) is dict and len(value)<=4096,'hash_inventory_schema')
 for key,sha in value.items():
  need(type(key) is str and len(key)<=512,'hash_inventory_key');hash_value(sha)
 return dict(count=len(value),declared_inventory_sha256=digest(canonical(value)),
  scope='producer-declared hashes; no referenced tensor/file readback by this observer')
def process(record):
 need(type(record) is dict and all(integer(record.get(k),0 if k=='uid' else 1)
  for k in ('pid','start_ticks','sid','uid')) and record['uid']==os.getuid(),'process_identity')
 out={k:record[k] for k in ('pid','start_ticks','sid','uid')};out.update(birth_matches=False,alive=False,state=None)
 try:
  proc=Path('/proc')/str(record['pid']);text=(proc/'stat').read_text()
  fields=text[text.rfind(')')+2:].split()
  match=int(fields[19])==record['start_ticks'] and int(fields[3])==record['sid'] and proc.stat().st_uid==record['uid']
  out.update(birth_matches=match,alive=match and fields[0] not in ('Z','X'),state=fields[0] if match else None)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return out
def same_execution(value):
 need(value.get('run_id')==run['run_id'] and value.get('execution_id')==run['execution_id'] and
  value.get('execution_backend')=='ssh_local' and value.get('slurm_job_id') is None,'execution_identity')
 if 'source_commit' in value:need(value['source_commit']==run['source_commit'],'source_identity')
def section(function):
 global read_pass
 read_pass=function.__name__
 try:return function()
 except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
  return dict(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
'''

REMOTE_READ = r'''
OUTPUT_LIMIT=128*1024
JSON_LIMIT=OUTPUT_LIMIT-len('MILESTONE_JSON:')-1
EXPECTED_LOSS=dict(loss_mode='sigma3_std_aux6_v1',std_loss_weight=3.0,count_aux_loss_weight=0.1,mean_loss_weight=0.0)
EXPECTED_LOSS_IDENTITY='sigma3_std_aux6_v1:CE3(K->min(K,5-K))+3*((E[sample_std]-sample_std(K))/sqrt(0.3))^2+0.1*CE6'
LOSS_SCALARS=('sigma3_ce','count_ce','std_mse','normalized_sample_std_mse',
 'sigma3_ce_contribution','std_mse_contribution','count_ce_contribution',
 'replay_weighted_loss','bootstrap_groups','completed_outer','head_optimizer_updates')
def fresh_origin():
 need(run['completed_updates']==0 and run['target_updates']==200,'fresh_zero_origin')
 need(all(run.get(key) is None for key in ('actor_checkpoint','reference_checkpoint','resume_path',
  'inherited_policy_run_id','inherited_optimizer_run_id','inherited_scheduler_run_id','full_state_restore')),'inherited_state_not_fresh')
 need(payload['intent'].get('saved_checkpoint_restore') is None,'intent_inherited_state_not_fresh')
 q=run['question_probe']
 need(all(q.get(key) is None for key in ('initial_head_checkpoint','snapshot_path','restore_state')),'inherited_question_head_not_fresh')
 need(all(type(q.get(key)) is int and q[key]==value for key,value in
  dict(candidate_questions=5120,selected_questions=512,responses_per_question=5,exploration_questions=0).items()),'fresh_top512_e0_contract')
 initial=run.get('initial_probe_version')
 if initial is None:
  return dict(status='observed',initial_training_outer=0,initial_optimizer_writes=0,
   initial_reward_probe=dict(status='not_yet_recorded'),inherited_training_state=False)
 need(type(initial) is str and re.fullmatch(re.escape(run['run_id'])+r':0:[0-9a-f]{32}',initial),
  'initial_reward_probe_not_this_run_zero')
 origin_base='/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'
 need(run.get('initial_probe_adoption_source_base')==payload['intent'].get('initial_probes_source_base')==origin_base,
  'initial_reward_probe_origin_base')
 adoption,adoption_info=document('logs/initial-probe-adoption.json')
 live,live_info=document('artifacts/live/manifest.json')
 accepted,accepted_info=document('logs/initial-probes-accepted.json')
 need(adoption is not None and live is not None and accepted is not None,'initial_reward_probe_adoption_metadata_missing')
 need(adoption['role']=='a800_initial_probe_adoption' and adoption['schema_version']==1 and
  adoption['run_id']==run['run_id'] and adoption['execution_id']==run['execution_id'] and
  adoption['source_commit']==run['source_commit'] and adoption['adopted'] is True and adoption['version']==initial and
  adoption['inherited_training_state'] is False and adoption['new_calibration_generated'] is False and
  adoption['new_discovery_performed'] is False and adoption['actual_original_calibration_exit_code']==0,
  'initial_reward_probe_adoption_identity')
 origin=adoption['origin']
 need(type(origin) is dict and origin['base']==origin_base and origin['run_id']!=run['run_id'] and
  origin['execution_id']!=run['execution_id'] and live['adoption_origin']==origin and
  type(origin['version']) is str and origin['version'].startswith(origin['run_id']+':0:'),
  'initial_reward_probe_adoption_origin')
 need(live['run_id']==run['run_id'] and type(live['round']) is int and live['round']==0 and live['version']==initial and
  live['encoder']==run['initial_model'] and live['target_source_commit']==run['source_commit'] and
  live['target_source_sha256']==run['source_sha256'] and live['new_calibration_generated'] is False and
  live['head_sha256']==origin['head_sha256'] and live['encoder_sha256']==origin['encoder_sha256'],
  'initial_reward_probe_live_identity')
 need(type(live['encoder_sha256']) is dict and bool(live['encoder_sha256']) and
  all(run['initial_model_sha256'].get(key)==hash_value(sha) for key,sha in live['encoder_sha256'].items()),
  'initial_reward_probe_official_encoder_fingerprint')
 live_sha=live_info['scanned_sha256']
 need(hash_value(adoption['live_manifest_sha256'])==live_sha and accepted['run_id']==run['run_id'] and
  accepted['accepted'] is True and accepted['version']==initial and hash_value(accepted['live_manifest_sha256'])==live_sha,
  'initial_reward_probe_actual_admission_identity')
 token=initial.rsplit(':',1)[1]
 relative='artifacts/round_0-adopted-'+token+'/version-manifest.json'
 need(adoption['target_archive']==str(base/relative),'initial_reward_probe_archive_namespace')
 archive,archive_info=document(relative)
 need(archive is not None and archive==live and
  hash_value(adoption['target_archive_sha256'])==archive_info['scanned_sha256'],'initial_reward_probe_archive_identity')
 return dict(status='observed',initial_training_outer=0,initial_optimizer_writes=0,inherited_training_state=False,
  initial_reward_probe=dict(status='observed',version=initial,round=0,source_base=origin_base,
   adoption_metadata_sha256=adoption_info['scanned_sha256'],live_manifest_sha256=live_sha,
   accepted_metadata_sha256=accepted_info['scanned_sha256'],archive_metadata_sha256=archive_info['scanned_sha256'],
   scope='existing current-run initial G1 reward-probe adoption/admission JSON; not restored actor/optimizer/RNG/sampler/question-head state; no probe/model tensor or HTTP readback'))
def configured_loss(value):
 need(type(value) is dict,'question_probe_configuration_missing')
 for key,expected in EXPECTED_LOSS.items():
  actual=value.get(key)
  need(actual==expected and (type(actual) is str if type(expected) is str else number(actual)),'sigma3_loss_configuration_'+key)
 return dict(status='observed',configuration={key:value[key] for key in EXPECTED_LOSS})
def recorded_loss(value):
 need(type(value) is dict,'head_loss_metadata_missing')
 need(value.get('loss_mode')==EXPECTED_LOSS['loss_mode'] and value.get('loss_identity')==EXPECTED_LOSS_IDENTITY,
  'head_sigma3_loss_identity_missing_or_mismatched')
 return dict(status='observed',loss_mode=value['loss_mode'],loss_identity=value['loss_identity'],
  loss_identity_sha256=digest(value['loss_identity'].encode()),recorded_scalars=select(value,LOSS_SCALARS))


def latest_prefit_diagnostics():
 window=result['reward_window']
 if window.get('status')!='observed' or window.get('latest') is None:return dict(status='unknown',reason='no_complete_formal_row')
 latest=window['latest'];event,event_info=document(base/'artifacts/question-probe/completed'/(latest['attempt_id']+'.json'))
 need(event is not None,'prefit_complete_event_missing')
 need(event['outer']==latest['step'] and event['attempt_id']==latest['attempt_id'],'prefit_complete_row_binding')
 values=event['prequential']['all'];need(values['questions']==512 and values['binary_threshold']==.5,'prefit512_binary_schema')
 confusion=values['binary_confusion'];need(sum(confusion[k] for k in ('tp','tn','fp','fn'))==512,'prefit512_binary_counts')
 return dict(status='observed',step=latest['step'],scope='selected512 actual pre-rollout predictions and labels, before that outer fit',prediction_version=event['prediction_version'],
  recorded_scalars=select(values,('actual_mixed_fraction','actual_sample_std_mean','count_nll','std_mae','mixed_brier','binary_threshold','binary_accuracy','binary_balanced_accuracy','binary_precision','binary_recall','binary_specificity','binary_f1','predicted_mixed_fraction','binary_majority_baseline','count_accuracy','sigma3_accuracy')),binary_confusion=confusion)

def warmup_training_records():
 path=base/'artifacts/question-probe/teacher-warmup-ready.json'
 if not path.is_file():return dict(status='unknown',reason='warmup_ready_not_recorded')
 ready,ready_info=document(path);need(ready is not None,'warmup_ready_missing');q=run['question_probe'];metrics=ready['head_metrics']
 need(q.get('bootstrap_head_epochs')==10 and q.get('head_epochs_per_outer')==2 and q.get('head_batch_size')==256,'warmup10_formal2_manifest')
 need(ready['fitted_questions']==5120 and ready['completed_training_outers']==0 and metrics['fit_epochs']==10 and metrics['head_batch_size']==256 and metrics['head_optimizer_updates']==200,'warmup10_actual_schedule')
 epochs=metrics['epoch_history'];mini=metrics['minibatch_history']
 need(type(epochs) is list and len(epochs)==11 and [x['epoch'] for x in epochs]==list(range(11)),'warmup_epoch0_to10_history')
 need(type(mini) is list and len(mini)==200,'warmup200_actual_minibatch_history')
 summaries=[];curve=[]
 for epoch,row in enumerate(epochs):
  need(row['replay_groups']==5120 and row['fit_epochs']==10 and row['head_batch_size']==256 and row['head_optimizer_updates']==20*epoch and row['head_adam_steps']==20*epoch,'warmup_epoch_actual_update_counts')
  parts=recorded_loss(dict(row,loss_mode=metrics['loss_mode'],loss_identity=metrics['loss_identity']));binary=row['binary_mixed']
  need(binary['threshold']==.5 and sum(binary[k] for k in ('true_positive','true_negative','false_positive','false_negative'))==5120,'warmup_binary_counts')
  summaries.append(dict(epoch=epoch,head_adam_steps=row['head_adam_steps'],loss=parts,
   binary_mixed={k:binary[k] for k in ('threshold','true_positive','true_negative','false_positive','false_negative','accuracy','balanced_accuracy','precision','recall','specificity','f1')},
   recorded_metrics=select(row,('true_mixed_fraction','predicted_mixed_fraction','sigma3_accuracy','count6_accuracy','sigma3_majority_baseline_accuracy','count6_majority_baseline_accuracy','sigma_mae','head_learning_rate'))))
 for index,row in enumerate(mini):
  need(row['epoch']==index//20+1 and row['minibatch_index']==index%20 and row['batch_size']==256 and row['head_update_ordinal']==index+1 and row['head_adam_steps_before']==index and row['head_adam_steps_after']==index+1,'warmup_minibatch_exact_sequence')
  value=row['pre_step_weighted_loss'];need(type(value) in (int,float) and math.isfinite(value),'warmup_minibatch_loss_finite')
  curve.append(dict(head_step=index+1,epoch=row['epoch'],minibatch_index=row['minibatch_index'],pre_step_weighted_loss=value))
 return dict(status='observed',scope='same5120traininggroups; no independent validation/model inference',bootstrap_head_epochs=10,formal_head_epochs_per_outer=2,head_batch_size=256,head_optimizer_updates=200,epochs=summaries,minibatch_history_count=200,minibatch_history_sha256=hashlib.sha256(json.dumps(mini,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),minibatch_loss_curve=curve,all_loss_parts_persisted_in_ready=True,counts_toward_formal200=False)

def loss_identity():
 def manifest_loss():
  value=configured_loss(run['question_probe']);value['metadata_sha256']=run_manifest_sha;return value
 def resolved_loss():
  value,info=document('logs/resolved-training-config.json')
  need(value is not None,'resolved_training_configuration_missing');same_execution(value)
  resolved=value['resolved_config'];need(type(resolved) is dict,'resolved_training_configuration_schema')
  sha=digest(json.dumps(resolved,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
  need(sha==hash_value(value['resolved_config_sha256']),'resolved_training_configuration_sha')
  output=configured_loss(resolved.get('question_probe'))
  output.update(metadata_sha256=info['scanned_sha256'],resolved_config_sha256=sha);return output
 def warmup_loss():
  value,info=document('artifacts/question-probe/teacher-warmup-ready.json')
  need(value is not None,'teacher_warmup_loss_metadata_missing')
  head=value['head_checkpoint'];metrics=value['head_metrics']
  need(value['status']=='FROZEN_TEACHER_WARMUP_HEAD_READY' and
   value['completed_training_outers']==0 and value['target_training_outers']==200 and
   value['fitted_questions']==5120 and metrics['bootstrap_groups']==5120 and
   head['actor_snapshot_id']==run['run_id']+':0' and head['completed_outer']==metrics['completed_outer']==0 and
   head['version']==metrics['version'],'teacher_warmup_head_identity')
  return dict(status='observed',metadata_sha256=info['scanned_sha256'],actor_snapshot_id=head['actor_snapshot_id'],
   head_version=version(head['version']),feature_spec_sha=hash_value(head['feature_spec_sha']),
   declared_head_sha256=hash_value(head['checkpoint_sha']),fit=recorded_loss(metrics),save=recorded_loss(head))
 def committed_loss():
  need(complete_updates is not None,'complete_prefix_unknown')
  if not complete_updates:return dict(status='not_yet_recorded',reason='no_recorded_complete_outer')
  latest=complete_updates[-1];value,info=document('artifacts/question-probe/completed/'+latest['attempt_id']+'.json')
  need(value is not None,'committed_head_loss_metadata_missing')
  need(value['stage']=='question_probe_outer_committed' and value['outer']==latest['step'] and
   value['attempt_id']==latest['attempt_id'] and value['successful_actor_snapshot_id']==run['run_id']+':'+str(latest['step']) and
   value['head_update']['completed_outer']==latest['step'],'committed_head_loss_identity')
  return dict(status='observed',metadata_sha256=info['scanned_sha256'],outer=value['outer'],
   actor_snapshot_id=value['successful_actor_snapshot_id'],head_version=version(value['head_version']),
   prediction_version=version(value['prediction_version']),feature_spec_sha=hash_value(value['feature_spec_sha']),
   fit=recorded_loss(value['head_update']))
 evidence=dict(manifest=section(manifest_loss),resolved=section(resolved_loss),
  warmup=section(warmup_loss),latest_committed=section(committed_loss))
 status='observed' if all(evidence[key]['status']=='observed' for key in ('manifest','resolved','warmup')) and (
  evidence['latest_committed']['status']=='observed' or complete_updates==[]) else 'unknown'
 return dict(status=status,expected_configuration=EXPECTED_LOSS,expected_loss_identity=EXPECTED_LOSS_IDENTITY,evidence=evidence,
  scope='current run manifest, SHA-checked resolved configuration, and existing producer fit/save JSON declarations only; no inferred defaults, tensor readback, verifier, new fit or evaluation')


def births():
 producer=run.get('pipeline_identity');need(type(producer) is dict,'pipeline_birth_metadata_missing')
 pipeline_claim,_=document('logs/pipeline-launch-claim.json');need(pipeline_claim is not None,'pipeline_claim_missing')
 same_execution(pipeline_claim);need(pipeline_claim['pipeline_identity']==producer,'pipeline_claim_birth')
 pipeline=process(producer);claim,_=document('logs/training-launch-claim.json')
 ledger,_=document('logs/process-ledger.json',PROCESS_LEDGER_LIMIT)
 if claim is None or ledger is None:
  return dict(status='unknown',reason='training_birth_metadata_missing',pipeline=pipeline,
   trainer=dict(status='unknown',reason='training_birth_metadata_missing'))
 same_execution(claim);need(ledger['run_id']==run['run_id'],'ledger_run')
 jobs=ledger['jobs'];need(type(jobs) is dict,'process_ledger_jobs_schema')
 job=jobs.get('trainer')
 if job is None:
  return dict(status='unknown',reason='trainer_job_not_yet_recorded',pipeline=pipeline,
   trainer=dict(status='unknown',reason='trainer_job_not_yet_recorded'))
 trainer=job['leader']
 need((trainer['pid'],trainer['start_ticks'])==(claim['shell_pid'],claim['shell_start_ticks']),'trainer_claim_birth')
 return dict(status='observed',pipeline=pipeline,trainer=process(trainer))
def reward_window():
 updates,info=rows('logs/actual-updates.jsonl');attempts,attempt_info=rows('logs/actual-attempts.jsonl')
 need(attempts is not None and attempt_info['scan_from_start'],'closed_attempt_ledger_missing_or_not_from_start')
 by_attempt={};prior_index=0;completed=0;writes=0
 aborted=0;full_budget=0;completed_attempts=[]
 for attempt in attempts:
  same_execution(attempt);aid=attempt['attempt_id'];index=attempt['attempt_index']
  need(type(aid) is str and re.fullmatch('[0-9a-f]{32}',aid) and aid not in by_attempt and
   integer(index,1) and index==prior_index+1,'attempt_order')
  processed=attempt['opportunities_processed'];count=attempt['optimizer_writes_this_attempt'];statuses=attempt['opportunity_statuses']
  need(integer(processed,1) and integer(count) and count<=processed<=20 and
   attempt['logical_update']==completed+1<=200 and integer(attempt['logical_update'],1) and
   integer(attempt['optimizer_writes_before']) and attempt['optimizer_writes_before']==writes and
   integer(attempt['optimizer_writes_after']) and attempt['optimizer_writes_after']==writes+count,'closed_attempt_counters')
  complete=processed==20 and count>0;full=count==20 and processed==20
  need(type(statuses) is list and len(statuses)==processed and
   all(status in ('written','rolled_back_guard','skipped_guard') for status in statuses) and
   statuses.count('written')==count and type(attempt['recovery_protocol_version']) is int and
   attempt['recovery_protocol_version']==2 and attempt['complete_outer'] is complete and
   attempt['status']==('complete' if complete else 'partial' if count else 'recoverable_guard') and
   attempt['all_ranks_agreed'] is True and type(attempt['real_rank_count']) is int and attempt['real_rank_count']==4 and
   type(attempt['scheduler_advances']) is int and attempt['scheduler_advances']==int(complete) and
   integer(attempt['completed_outer']) and attempt['completed_outer']==completed+int(complete),'closed_attempt_full_outer_contract')
  need(attempt['full_budget_complete'] is full and integer(attempt['opportunities_unattempted']) and
   attempt['opportunities_unattempted']==20-processed and attempt['retained_partial_adam_writes'] is (count>0 and not complete) and
   integer(attempt['optimizer_calls_this_attempt']) and integer(attempt['optimizer_rollbacks_this_attempt']) and
   attempt['optimizer_rollbacks_this_attempt']==statuses.count('rolled_back_guard') and
   attempt['optimizer_calls_this_attempt']==count+attempt['optimizer_rollbacks_this_attempt']<=processed and
   attempt['current_step_rollback'] is True and attempt['no_automatic_rollback'] is False,'closed_attempt_transaction_counters')
  aborted+=int(not complete);full_budget+=int(full)
  need(integer(attempt['cumulative_aborted_attempts']) and attempt['cumulative_aborted_attempts']==aborted and
   integer(attempt['cumulative_full_budget_outers']) and attempt['cumulative_full_budget_outers']==full_budget,'closed_attempt_cumulative_counts')
  if complete:completed_attempts.append(attempt)
  by_attempt[aid]=attempt;prior_index=index;writes+=count;completed+=int(complete)
 need(payload['after_step']<=completed,'requested_anchor_ahead_of_complete_prefix')
 # A retained partial can precede the first actual-updates file. Keep its
 # authenticated driver prefix available to opportunity_counts, while the
 # absent scientific reward ledger remains explicitly unknown.
 if updates is None and not completed_attempts:
  globals()['closed_attempts']=by_attempt;globals()['complete_updates']=[]
  globals()['driver_attempt_partial']=attempt_info['partial_final_bytes']>0
  raise Unknown('complete_update_ledger_missing')
 need(updates is not None and info['scan_from_start'],'updates_missing_or_not_from_start')
 need(len(completed_attempts)==len(updates),'driver_prefixes_not_same_snapshot')
 prior_after=0;prior_index=0;observed=[]
 for step,row in enumerate(updates,0+1):
  need(row['run_id']==run['run_id'] and type(row['step']) is int and row['step']==step<=200,'complete_step_order')
  for key in ('execution_id','source_commit','execution_backend','slurm_job_id'):
   if key in row:need(row[key]==run.get(key),'complete_optional_identity')
  aid=row['attempt_id'];attempt=by_attempt.get(aid);need(attempt is not None,'complete_attempt_missing')
  index=row['attempt_index'];before=row['optimizer_writes_before'];after=row['optimizer_writes_after'];writes=row['actual_optimizer_writes']
  need(integer(index,1) and index>prior_index and integer(before) and before>=prior_after and
   integer(after,1) and integer(writes,1) and writes<=20 and after-before==writes,'returned_write_counters')
  need(attempt['logical_update']==step and attempt['completed_outer']==step and attempt['status']=='complete' and
   attempt['complete_outer'] is True and attempt['all_ranks_agreed'] is True and attempt['real_rank_count']==4 and
   attempt['scheduler_advances']==1 and attempt['opportunities_processed']==row['opportunities_processed']==20 and
   attempt['attempt_index']==index and attempt['optimizer_writes_before']==before and attempt['optimizer_writes_after']==after and
   attempt['optimizer_writes_this_attempt']==writes,'matching_completed_attempt')
  metrics=row['metrics'];reward=metrics['critic/rewards/mean']
  need(number(reward) and metrics['actor/update_opportunities_this_attempt']==20 and
   metrics['actor/optimizer_steps_this_update']==writes and metrics['actor/skipped_update_opportunities']==20-writes,'complete_metrics')
  need(type(row['recovery_protocol_version']) is int and row['recovery_protocol_version']==2 and
   row['full_budget_complete'] is (writes==20) and integer(row['opportunities_unattempted']) and
   row['opportunities_unattempted']==0,'complete_transaction_fields')
  hash_value(row['behavior_probability_gate_sha256']);hash_value(row['behavior_trace_sha256'])
  observed.append(dict(step=step,reward_mean=reward,successful_adam=writes,skip=20-writes,processed=20,unattempted=0,
   full_budget_complete=writes==20,optimizer_call_attempts=attempt['optimizer_calls_this_attempt'],
   optimizer_write_rollbacks=attempt['optimizer_rollbacks_this_attempt'],
   optimizer_writes_before=before,optimizer_writes_after=after,attempt_id=aid,attempt_index=index))
  prior_after=after;prior_index=index
 need(0<=payload['after_step']<=0+len(observed),'requested_anchor_ahead_of_complete_prefix')
 globals()['complete_updates']=updates
 globals()['closed_attempts']=by_attempt
 globals()['driver_attempt_partial']=attempt_info['partial_final_bytes']>0
 fresh=[row for row in observed if row['step']>payload['after_step']];values=[row['reward_mean'] for row in fresh]
 return dict(status='observed',recorded_complete_outers=0+len(observed),
  fresh_initial_completed_outers=0,current_execution_complete_outers=len(observed),after_complete_step=payload['after_step'],
  current_execution_closed_attempts=len(attempts),current_execution_aborted_attempts=aborted-0,
  current_execution_full_budget_complete_outers=full_budget-0,cumulative_full_budget_complete_outers=full_budget,
  newly_complete_steps=[row['step'] for row in fresh],new_complete_rows=fresh,
  latest=observed[-1] if observed else None,window_count=len(values),window_mean=math.fsum(values)/len(values) if values else None,
  window_min=min(values) if values else None,window_max=max(values) if values else None,
  complete_prefix_sha256=info['complete_prefix_sha256'],partial_final_status=info['partial_final_status'],
  closed_attempt_prefix_sha256=attempt_info['complete_prefix_sha256'],closed_attempt_partial_final_status=attempt_info['partial_final_status'],
  scope='official fresh-zero execution logical1..200 returned whole outer rows; 20 processed opportunities and positive retained Adam required; partial attempts excluded from scientific reward window')

def infra_window():
 need(complete_updates is not None,'validated_complete_prefix_unknown')
 q=run.get('question_probe',{});phase=run.get('infra_rollout')
 expected=dict(mode='independent_trajectory_phase_v1',agent_workers=256,max_inflight_trajectories=256,
  generation_max_num_seqs=64,generation_max_batched_tokens=8192,retrieval_concurrency=16,
  tool_timeout=30.0,tool_retries=2,engine_step_budget=8)
 eligible=phase==expected and all(type(q.get(key)) is int and q[key]==value for key,value in
  dict(candidate_questions=5120,selected_questions=512,responses_per_question=5,exploration_questions=0).items())
 eligibility=dict(status='eligible' if eligible else 'unknown',manifest_bound=True,
  expected_mode=expected['mode'],expected_exploration_questions=0,
  expected_complete_trajectories=512*5,scope='expected manifest configuration, not an observed performance count')
 if not eligible:return dict(status='unknown',reason='async_top512_manifest_contract_not_matched',eligibility=eligibility)
 counts=('actor/actual_monitor_forward_count','actor/reused_monitor_forward_count',
  'infra/reference_forward_skipped','infra/rollout/completed_trajectories',
  'infra/rollout/weight_syncs_per_rank','infra/rollout/generation_requests',
  'infra/rollout/environment_actions','infra/rollout/retrieval_concurrency',
  'infra/rollout/peak_inflight_trajectories','infra/rollout/engine_steps_sum',
  'infra/rollout/completed_requests_sum')
 timings=('infra/rollout/phase_elapsed_seconds','timing_s/candidate_fill','timing_s/step',
  'timing_s/gen','timing_s/ref','timing_s/values','timing_s/adv','timing_s/update_critic',
  'timing_s/update_actor','timing_s/question_probe_update','timing_s/testing','timing_s/save_checkpoint')
 def recorded(row):
  values={};unknown={};metrics=row['metrics']
  for key in counts+timings:
   value=metrics.get(key)
   reason='not_persisted_in_actual_update_metrics' if key not in metrics else None
   if reason is None and (not number(value) or value<0):reason='not_a_finite_nonnegative_scalar'
   if reason is None and key in counts and not float(value).is_integer():reason='not_an_integral_count'
   values[key]=value if reason is None else None
   if reason is not None:unknown[key]=reason
  trajectories=values['infra/rollout/completed_trajectories']
  return dict(step=row['step'],attempt_id=row['attempt_id'],attempt_index=row['attempt_index'],
   status='observed' if not unknown else 'unknown',recorded_metrics=values,unknown_fields=unknown,
   actual_trajectories_match_manifest=trajectories==512*5 if trajectories is not None else None)
 fresh=complete_updates[payload['after_step']:];latest=recorded(complete_updates[-1]) if complete_updates else None
 # A supplemental section cannot push a long scientific reward window past
 # the existing output cap. Do not truncate it and imply complete coverage.
 within_limit=len(fresh)<=8;observed=[recorded(row) for row in fresh] if within_limit else []
 return dict(status='observed' if latest is not None and latest['status']=='observed' and within_limit and
  all(row['status']=='observed' for row in observed) else 'unknown',eligibility=eligibility,
  reason='no_recorded_complete_outer' if latest is None else 'supplemental_eight_row_output_limit' if not within_limit else
   'some_whitelisted_metrics_not_persisted_or_invalid' if latest['status']=='unknown' or any(row['status']=='unknown' for row in observed) else None,
  after_complete_step=payload['after_step'],new_complete_row_count=len(fresh),new_complete_rows=observed,
  latest=latest,window_status='observed' if within_limit else 'unknown',
  window_reason=None if within_limit else 'supplemental_eight_row_output_limit',
  maximum_supplemental_window_rows=8,
  scope='only whitelisted scalars persisted in the already validated complete actual-updates metrics; no console read, inferred counts, new sampling, or missing-field training-failure claim')
def opportunity_counts():
 need(closed_attempts is not None and complete_updates is not None,'closed_attempt_prefix_unknown')
 closed=sorted(closed_attempts.values(),key=lambda item:item['attempt_index'])
 need([item['attempt_index'] for item in closed]==list(range(0+1,0+len(closed)+1)),'closed_attempt_sequence')
 need(sum(item['complete_outer'] is True for item in closed)==len(complete_updates),'driver_prefixes_not_same_snapshot')
 results=[];signatures=[];partial_files=driver_attempt_partial
 for rank in range(4):
  values,info=rows('logs/optimizer-rank-'+str(rank)+'.jsonl',limit=RANK_LEDGER_LIMIT)
  need(values is not None and info['scan_from_start'],'opportunity_prefix_missing_or_not_from_start')
  groups=[];signature=[];writes=0
  for row in values:
   need(row.get('run_id')==run['run_id'] and type(row.get('rank')) is int and row['rank']==rank,'opportunity_row_identity')
   if row.get('kind') in ('OUTER_BUDGET','ABORTED_ATTEMPT'):
    need(groups,'terminal_without_opportunities');group=groups[-1]
    complete=group['processed']==20 and group['successful_adam']>0
    need(group['terminal_kind'] is None and row['attempt_id']==group['attempt_id'] and
     row['attempt_index']==group['attempt_index'] and row['logical_update']==group['logical_update'] and
     row['kind']==('OUTER_BUDGET' if complete else 'ABORTED_ATTEMPT') and
     row['recovery_protocol_version']==2 and row['opportunities_processed']==group['processed'] and
     row['optimizer_writes']==group['successful_adam'] and row['cumulative_writes']==group['optimizer_writes_after'],
     'opportunity_terminal_disagrees')
    if complete:
     need(row['optimizer_writes_before']==group['optimizer_writes_before'] and row['optimizer_writes_after']==writes and
      row['full_budget_complete'] is (group['successful_adam']==20),'complete_terminal_counters')
    else:
     need(row['writes_before']==group['optimizer_writes_before'] and row['complete_outer'] is False and
      row['scheduler_advances']==0 and row['retained_partial_adam_writes'] is (group['successful_adam']>0) and
      row['optimizer_calls_this_attempt']==group['optimizer_call_attempts'] and
      row['optimizer_rollbacks_this_attempt']==group['optimizer_write_rollbacks'] and
      row['behavior_probability_gate_sha256']==group['behavior_probability_gate_sha256'],'aborted_terminal_counters')
    group['terminal_kind']=row['kind'];signature.append(['terminal',group['attempt_id'],row['kind']]);continue
   if row.get('kind')!='OPPORTUNITY':continue
   context=row['context'];aid=context['attempt_id'];ordinal=context['attempt_index'];outer=context['logical_update']
   gate_sha=hash_value(context['behavior_probability_gate_sha256'])
   need(type(aid) is str and re.fullmatch('[0-9a-f]{32}',aid) and integer(ordinal,1) and integer(outer,1) and outer<=200,'opportunity_context')
   if not groups or groups[-1]['attempt_id']!=aid:
    need(ordinal==0+len(groups)+1 and aid not in {item['attempt_id'] for item in groups},'opportunity_attempt_sequence')
    groups.append(dict(attempt_id=aid,attempt_index=ordinal,logical_update=outer,processed=0,successful_adam=0,
     optimizer_call_attempts=0,optimizer_write_rollbacks=0,
     opportunity_statuses=[],terminal_kind=None,
     optimizer_writes_before=writes,optimizer_writes_after=writes,behavior_probability_gate_sha256=gate_sha))
   group=groups[-1];index=row['opportunity_index'];before=row['optimizer_writes_before'];after=row['optimizer_writes_after'];wrote=row['optimizer_write_succeeded']
   need(group['terminal_kind'] is None and group['attempt_index']==ordinal and group['logical_update']==outer and type(index) is int and
    index==group['processed']<20 and context['minibatch_index']==index and context['prior_optimizer_writes']==before and
    row['processed'] is True and type(wrote) is bool and integer(before) and integer(after) and before==writes and after-before==int(wrote),
    'opportunity_sequence_or_cumulative_write')
   status=row['status'];rollback=status=='rolled_back_guard';called=status in ('written','rolled_back_guard')
   need(type(row.get('recovery_protocol_version')) is int and row['recovery_protocol_version']==2 and
    status in ('written','rolled_back_guard','skipped_guard') and wrote is (status=='written') and
    row['optimizer_write_rolled_back'] is rollback and row['optimizer_call_attempted'] is called,
    'opportunity_transaction_disposition')
   need(group['behavior_probability_gate_sha256']==gate_sha,'attempt_behavior_gate_changed')
   group['processed']+=1;group['successful_adam']+=int(wrote);group['optimizer_writes_after']=after;writes=after
   group['opportunity_statuses'].append(status)
   group['optimizer_call_attempts']+=int(called);group['optimizer_write_rollbacks']+=int(rollback)
   signature.append([aid,ordinal,outer,index,before,after,wrote,status,called,rollback,gate_sha])
  for attempt in closed:
   ordinal=attempt['attempt_index']-0;need(1<=ordinal<=len(groups),'closed_attempt_opportunities_not_yet_visible')
   group=groups[ordinal-1]
   need(group['attempt_id']==attempt['attempt_id'] and group['logical_update']==attempt['logical_update'] and
    group['processed']==attempt['opportunities_processed'] and group['successful_adam']==attempt['optimizer_writes_this_attempt'] and
    group['opportunity_statuses']==attempt['opportunity_statuses'] and
    group['terminal_kind']==('OUTER_BUDGET' if attempt['complete_outer'] else 'ABORTED_ATTEMPT') and
    group['optimizer_writes_before']==attempt['optimizer_writes_before'] and group['optimizer_writes_after']==attempt['optimizer_writes_after'] and
    attempt['recovery_protocol_version']==2 and group['optimizer_call_attempts']==attempt['optimizer_calls_this_attempt'] and
    group['optimizer_write_rollbacks']==attempt['optimizer_rollbacks_this_attempt'],
    'opportunities_disagree_with_closed_attempt')
   if attempt['complete_outer'] is True:
    update=complete_updates[attempt['logical_update']-0-1]
    need(update['attempt_id']==group['attempt_id'] and update['behavior_probability_gate_sha256']==group['behavior_probability_gate_sha256'],
     'opportunity_gate_differs_from_returned_complete_update')
  need(len(groups) in (len(closed),len(closed)+1),'more_than_one_unclosed_attempt')
  partial=groups[-1] if len(groups)>len(closed) else None
  if partial:need(partial['logical_update']==0+len(complete_updates)+1,'partial_logical_outer')
  if partial:partial.update(processed_skip=partial['processed']-partial['successful_adam'],
   opportunities_remaining_or_unattempted=20-partial['processed'])
  aborted_groups=[groups[index] for index,attempt in enumerate(closed) if not attempt['complete_outer']]
  completed_groups=[groups[index] for index,attempt in enumerate(closed) if attempt['complete_outer']]
  processed=sum(group['processed'] for group in groups);successful=sum(group['successful_adam'] for group in groups)
  need(successful==writes,'cumulative_adam_not_from_initial_zero')
  partial_files=partial_files or info['partial_final_bytes']>0
  signatures.append(digest(canonical(signature)))
  results.append(dict(rank=rank,complete_prefix_processed=processed,complete_prefix_successful_adam=successful,
   complete_prefix_skip=processed-successful,current_partial=partial,prefix_sha256=info['complete_prefix_sha256'],
   optimizer_call_attempts=sum(group['optimizer_call_attempts'] for group in groups),
   optimizer_write_rollbacks=sum(group['optimizer_write_rollbacks'] for group in groups),
   closed_aborted_attempt_count=len(aborted_groups),
   closed_aborted_attempt_successful_adam=sum(group['successful_adam'] for group in aborted_groups),
   closed_aborted_attempt_processed=sum(group['processed'] for group in aborted_groups),
   closed_aborted_attempt_skip=sum(group['processed']-group['successful_adam'] for group in aborted_groups),
   closed_attempt_unattempted=sum(20-group['processed'] for group in aborted_groups),
   complete_outer_successful_adam=sum(group['successful_adam'] for group in completed_groups),
   full_budget_complete_outers=sum(group['successful_adam']==20 for group in completed_groups),
   cumulative_optimizer_writes=writes,partial_final_status=info['partial_final_status']))
 agreement=len(set(signatures))==1
 return dict(status='observed' if agreement and not partial_files else 'unknown',
  reason=None if agreement and not partial_files else 'four_rank_prefixes_disagree' if not agreement else 'incomplete_final_row',
  four_rank_complete_prefix_agreement=agreement,per_rank=results,
  fresh_initial_optimizer_writes=0,
  current_execution_processed=results[0]['complete_prefix_processed'] if agreement and not partial_files else None,
  current_execution_successful_adam=results[0]['complete_prefix_successful_adam'] if agreement and not partial_files else None,
  current_execution_skip=results[0]['complete_prefix_skip'] if agreement and not partial_files else None,
  current_execution_aborted_attempts=results[0]['closed_aborted_attempt_count'] if agreement and not partial_files else None,
  current_execution_aborted_attempt_retained_adam=results[0]['closed_aborted_attempt_successful_adam'] if agreement and not partial_files else None,
  current_execution_aborted_attempt_processed=results[0]['closed_aborted_attempt_processed'] if agreement and not partial_files else None,
  current_execution_aborted_attempt_skip=results[0]['closed_aborted_attempt_skip'] if agreement and not partial_files else None,
  current_execution_closed_attempt_unattempted=results[0]['closed_attempt_unattempted'] if agreement and not partial_files else None,
  current_execution_complete_outer_successful_adam=results[0]['complete_outer_successful_adam'] if agreement and not partial_files else None,
  current_execution_full_budget_complete_outers=results[0]['full_budget_complete_outers'] if agreement and not partial_files else None,
  cumulative_full_budget_complete_outers=0+results[0]['full_budget_complete_outers'] if agreement and not partial_files else None,
  current_execution_optimizer_call_attempts=results[0]['optimizer_call_attempts'] if agreement and not partial_files else None,
  current_execution_optimizer_write_rollbacks=results[0]['optimizer_write_rollbacks'] if agreement and not partial_files else None,
  cumulative_processed=results[0]['complete_prefix_processed'] if agreement and not partial_files else None,
  cumulative_successful_adam=results[0]['complete_prefix_successful_adam'] if agreement and not partial_files else None,
  cumulative_skip=results[0]['complete_prefix_skip'] if agreement and not partial_files else None,
  current_partial=results[0]['current_partial'] if agreement and not partial_files else None,
  closed_attempt_partial_final_row=driver_attempt_partial,
  scope='official fresh-zero execution from-start newline-complete OPPORTUNITY receipts, including separately counted retained partial-attempt Adam; closed unattempted opportunities are not processed skips; full-budget requires20 retained writes; four replicas counted once; no inherited or tail-derived cumulative count')

def runtime_status():
 stage,_=document('logs/pipeline-status.json');ledger,_=document('logs/process-ledger.json',PROCESS_LEDGER_LIMIT)
 need(stage is not None and ledger is not None and stage['run_id']==ledger['run_id']==run['run_id'],'runtime_status_identity')
 need(type(stage['stage']) is str and re.fullmatch('[A-Z][A-Z0-9_]{0,95}',stage['stage']),'pipeline_stage_schema')
 jobs=ledger['jobs'];need(type(jobs) is dict and len(jobs)<=32,'job_count_limit')
 output=[];error_paths=[]
 for name,job in jobs.items():
  need(type(name) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',name) and
   type(job['state']) is str and re.fullmatch('[A-Z][A-Z0-9_]{0,63}',job['state']),'job_name_or_state')
  code=job.get('exit_code');need(code is None or type(code) is int,'job_exit_code')
  output.append(dict(name=name,state=job['state'],exit_code=code,leader=process(job['leader'])))
  if (code is not None and code!=0) or job['state']=='CLEANUP_FAILED':error_paths.append(Path(job['log']))
 if stage['stage'] in ('FAILED','OWNED_CLEANUP_FAILED'):error_paths.append(base/'logs/deployment-pipeline.log')
 error=stage.get('error');error_type=re.match(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))\(',error) if type(error) is str else None
 diagnostics=[]
 for path in list(dict.fromkeys(error_paths))[:4]:
  path.relative_to(base/'logs');raw,info=snapshot(path,8192,append=True,tail=True)
  if raw is None:diagnostics.append(dict(status='missing'));continue
  lines=[]
  for line in raw.decode(errors='replace').splitlines():
   line=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',line)
   line=re.sub(r'^\((?:TaskRunner|ActorRolloutRefWorker|WorkerDict|RayWorker) pid=[0-9]+\)\s*','',line)
   lines.append(line)
  text='\n'.join(lines)
  frames=[dict(file=Path(match.group(1)).name[:160],line=int(match.group(2)),function=match.group(3)[:96]) for match in
   re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',text,re.MULTILINE)][-8:]
  classes=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',text,re.MULTILINE)
  diagnostics.append(dict(status='observed',file=path.name[:160],bounded_tail_sha256=info['scanned_sha256'],
   frames=frames,exception_type=classes[-1][:96] if classes else None))
 return dict(status='observed',pipeline_stage=stage['stage'],pipeline_status_unix=scalar(stage.get('unix')),jobs=output,
  recorded_error_present=error is not None,recorded_error_type=error_type.group(1)[:96] if error_type else None,
  failure_diagnostics=diagnostics,scope='existing pipeline/job status and exact leader births; bounded exception locations/classes only on recorded failure, never error messages or source lines')
def head_commit():
 need(complete_updates is not None,'complete_prefix_unknown')
 if not complete_updates:return dict(status='missing',reason='no_recorded_complete_outer')
 latest=complete_updates[-1];value,info=document('artifacts/question-probe/completed/'+latest['attempt_id']+'.json')
 if value is None:return dict(status='missing')
 need(value['stage']=='question_probe_outer_committed' and value['outer']==latest['step'] and
  value['attempt_id']==latest['attempt_id'] and value['successful_actor_snapshot_id']==run['run_id']+':'+str(latest['step']),'committed_head_identity')
 return dict(status='observed',outer=value['outer'],actor_snapshot_id=value['successful_actor_snapshot_id'],
  head_version=version(value['head_version']),prediction_version=version(value['prediction_version']),
  feature_spec_sha=hash_value(value['feature_spec_sha']),metadata_sha256=info['scanned_sha256'],
  recorded_head_update=select(value['head_update'],('completed_outer','update_count','head_optimizer_updates','replay_groups')),
  scope='existing committed head event after returned whole actor outer; no head tensor read/reload')
def checkpoint_metadata():
 available=[]
 checkpoint_root=Path(run['checkpoint_dir']);need(checkpoint_root==base/'checkpoints/search-g1-common8','checkpoint_namespace')
 for step in (5,50,100,150,200):
  prefix='checkpoints/search-g1-common8/actor/global_step_'+str(step)
  complete,info=document(prefix+'/complete.json')
  if complete is None:continue
  need(complete['run_id']==run['run_id'] and complete['step']==step and complete['checkpoint']==str(base/prefix),'checkpoint_identity')
  manifest,manifest_info=document(prefix+'/training_state/manifest.json');need(manifest is not None and manifest['completed_outer']==step,'state_manifest_outer')
  workers=manifest['worker_files'];need(type(workers) is list and sorted(row['rank'] for row in workers)==[0,1,2,3] and
   all(row['run_id']==run['run_id'] for row in workers),'state_manifest_ranks')
  q=manifest['question_probe'];need(q['completed_outer']==step and q['actor_snapshot_id']==run['run_id']+':'+str(step),'head_manifest_actor')
  pair,pair_info=document('artifacts/question-probe/completed-pairs/global_step_'+str(step)+'/source-receipt.json',4*1024*1024)
  pair_out=dict(status='missing',actual_cuda_restore_accepted=None)
  if pair is not None:
   need(pair['run_id']==run['run_id'] and pair['step']==step and pair['source_base']==str(base) and pair['checkpoint']==str(base/prefix) and
    pair['world_size']==4 and pair['actor_snapshot_id']==q['actor_snapshot_id'],'pair_identity')
   need(pair['source_run_manifest']['full_read_sha256']==run_manifest_sha,'pair_run_manifest_hash')
   need(pair['status']=='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS' and pair['kind']=='question_probe_pair' and
    pair['actual_cuda_restore_accepted'] is False,'pair_status')
   pair_out=dict(status='observed',producer_status=pair['status'],metadata_sha256=pair_info['scanned_sha256'],
    actual_cuda_restore_accepted=False,head_version=version(pair['head_version']),feature_spec_sha=hash_value(pair['feature_spec_sha']),
    attempt_progress=select(pair['attempt_progress'],('completed_outer','cumulative_writes','attempt_index','aborted_attempts')))
  available.append(dict(step=step,metadata_sha256=info['scanned_sha256'],manifest_sha256=manifest_info['scanned_sha256'],
   recorded_complete=select(complete,('finite_all_saved_tensors','validated_tensor_count','actual_optimizer_writes_per_rank')),
   declared_files=hash_inventory(complete['files_sha256']),actor_snapshot_id=q['actor_snapshot_id'],
   head_version=version(q['head_version']),head_declared_sha256=hash_value(q['head']['checkpoint_sha']),
   saved_head_loss_identity=section(lambda:recorded_loss(q['head'])),
   question_state_declared_sha256=hash_value(q['sha256']),pair=pair_out))
 return dict(status='observed',checkpoints=available,scope='JSON producer proofs and declared versions/hashes only; no model/state/head tensor read, hash, finite recheck or CUDA/FSDP reload')
def gpu_samples():
 samples,info=rows('logs/gpu-samples-'+run['run_id']+'.jsonl',tail=True)
 if samples is None:return dict(status='missing')
 physical=run['training_physical_indices']+run['auxiliary_physical_indices'];previous=None;peaks={};latest=None;gaps=[]
 producer={key:run['pipeline_identity'][key] for key in ('pid','start_ticks','sid','uid')}
 for sample in samples:
  same_execution(sample);need(sample['role']=='actual_common8_gpu_occupancy' and sample['source_map_sha256']==source_map_sha and
   sample['sampler_identity']==producer,'gpu_sampler_identity')
  begin=sample['query_started_unix'];end=sample['unix'];need(number(begin) and number(end) and 0<begin<=end and
   (previous is None or previous<=begin),'gpu_time_order')
  devices=sample['gpus'];need(type(devices) is list and len(devices)==len(physical) and
   all(type(item) is dict and type(item.get('index')) is int for item in devices) and
   sorted(item['index'] for item in devices)==sorted(physical),'gpu_devices')
  for item in devices:
   index=item['index'];need(item['uuid']==run['assigned_gpu_uuids'][str(index)] and all(integer(item[k]) for k in ('used_mib','total_mib','util_percent')) and
    0<=item['used_mib']<=item['total_mib'] and item['total_mib']>0 and item['util_percent']<=100,'gpu_uuid_or_scalar')
   peaks[str(index)]=max(peaks.get(str(index),0),item['used_mib'])
  if previous is not None:gaps.append(end-previous)
  previous=end;latest=[select(item,('index','used_mib','total_mib','util_percent')) for item in devices]
 return dict(status='observed',sample_count=len(samples),latest_sample_unix=previous,latest_devices=latest,
  sampled_pipeline_lifetime_peak_mib=peaks if info['scan_from_start'] else None,sampled_scanned_tail_peak_mib=peaks,
  maximum_scanned_sampling_gap_seconds=max(gaps) if gaps else None,scan_from_start=info['scan_from_start'],
  complete_step_window_peaks=dict(status='unknown',reason='not_computed_by_this_metadata_entry'),
  scope='existing physical-node occupancy samples over pipeline lifetime/scanned tail, including startup; not exclusive model peaks or complete-step GPU peaks; nativefive separately reports its recorded window boundaries')
def existing_teacher_cache_progress():
 name=run['question_probe'].get('teacher_warmup_cache')
 need(type(name) is str and name==payload['intent'].get('teacher_warmup_cache'),'teacher_cache_intent_binding')
 cache=Path(name)
 need(cache.is_absolute() and re.fullmatch(r'/(data|mnt|backup)/[A-Za-z0-9._/-]+',name) and
  '..' not in cache.parts and cache.resolve()==cache and not cache.is_symlink(),'teacher_cache_canonical_remote_path')
 scope='existing stable teacher-cache JSON metadata only; no raw rollout, tensor, labels, group vectors or question IDs returned; no raw-file SHA verification, complete-cache qualification or inferred current-run generation'
 if not cache.exists():
  return dict(status='not_yet_recorded',reason='teacher_cache_directory_missing',cache_root=name,
   path_scope='intent_bound_external_teacher_cache_metadata',current_run_generation='unknown',scope=scope)
 current=cache.stat(follow_symlinks=False)
 need(stat.S_ISDIR(current.st_mode) and current.st_uid==os.getuid(),'teacher_cache_root_owner_or_type')
 files=[]
 def cache_document(relative):
  info=dict(path_scope='intent_bound_external_teacher_cache_metadata',relative_path=relative)
  try:
   raw,saved=snapshot(cache/relative,4*1024*1024,record=False)
   if raw is None:
    info.update(status='not_yet_recorded');files.append(info);return None,info
   value=json.loads(raw);need(type(value) is dict,'teacher_cache_metadata_object')
   info.update(status='observed',file_bytes=saved['file_bytes'],metadata_sha256=saved['scanned_sha256'],
    verified_initial_range_unchanged=saved['verified_initial_range_unchanged'])
   files.append(info);return value,info
  except (Unknown,KeyError,TypeError,ValueError,OSError) as failure:
   info.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
   files.append(info);return None,info
 header,header_info=cache_document('identity.json')
 header_out=dict(status=header_info['status'],metadata_sha256=header_info.get('metadata_sha256'),questions_count=None)
 if header is not None:
  questions=header.get('questions')
  need(type(questions) is list,'teacher_cache_header_question_list')
  header_out.update(questions_count=len(questions),recorded_schema=scalar(header.get('schema')),
   recorded_world_size=scalar(header.get('training_world_size')),
   recorded_frozen_evidence_required=scalar(header.get('teacher_frozen_evidence_required')),
   declared_teacher_model_matches_current=header.get('model_revision')==run['model_revision'] and
    header.get('model_files_sha256')==run['initial_model_sha256'],
   declared_train_data_matches_current=header.get('train_parquet_sha256')==run['input_sha256']['train.parquet'] and
    header.get('data_manifest_sha256')==run['budget']['common_manifest_sha256'])
  workers=(header.get('rollout_sources') or {}).get('verl/workers/fsdp_workers.py')
  header_out['declared_fsdp_worker_source_matches_current']=workers==run['source_sha256'].get('verl/workers/fsdp_workers.py') if workers is not None else None
 def frozen_metadata(value):
  need(type(value) is dict and value.get('actual_frozen_rank_readback') is True and
   type(value.get('actual_world_size')) is int and value['actual_world_size']==4,'teacher_chunk_frozen_metadata_missing')
  before=value['before'];after=value['after']
  need(type(before) is list and type(after) is list and len(before)==len(after)==4,'teacher_chunk_frozen_rank_count')
  before=sorted(before,key=lambda item:item['rank']);after=sorted(after,key=lambda item:item['rank'])
  for rank,(first,last) in enumerate(zip(before,after)):
   need(type(first['rank']) is int and type(last['rank']) is int and first['rank']==last['rank']==rank,
    'teacher_chunk_frozen_rank_identity')
   for field in ('actual_optimizer_writes','completed_outer'):
    need(type(first[field]) is int and type(last[field]) is int and first[field]==last[field]==0,
     'teacher_chunk_frozen_nonzero_counters')
   for field in ('parameter_sha256','optimizer_sha256','scheduler_sha256'):
    need(hash_value(first[field])==hash_value(last[field]),'teacher_chunk_frozen_declared_hash_mismatch')
  return dict(status='observed',evidence='producer_declared_frozen_workers_JSON',ranks=4,
   recorded_optimizer_writes_per_rank=0,recorded_completed_outer=0,recorded_before_after_hashes_equal=True)
 chunks=[];groups_total=0;trajectories_total=0
 for index in range(10):
  value,info=cache_document('chunk-%02d.json'%index)
  item=dict(index=index,status=info['status'],metadata_sha256=info.get('metadata_sha256'))
  if value is not None:
   try:
    groups=value.get('groups');need(type(value.get('index')) is int and value['index']==index and
     type(groups) is list and len(groups)==512,'teacher_chunk_metadata_counts')
    need(all(type(group) is dict and type(group.get('trajectory_ids')) is list and
     len(group['trajectory_ids'])==5 for group in groups),'teacher_chunk_trajectory_metadata_counts')
    proof=value['raw_rollout'];need(type(proof) is dict and proof['relative_path']=='rollout-%02d.pt'%index and
     integer(proof['bytes'],1),'teacher_chunk_raw_rollout_declaration')
    item.update(groups_count=512,trajectory_count=2560,evidence='validated_metadata_counts_only',
     raw_rollout_declared_bytes=proof['bytes'],raw_rollout_declared_sha256=hash_value(proof['sha256']),raw_rollout_read=False)
    item['frozen_workers']=section(lambda: frozen_metadata(value.get('frozen_workers')))
    groups_total+=512;trajectories_total+=2560
   except (Unknown,KeyError,TypeError,ValueError) as failure:
    item.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
  chunks.append(item)
 complete,complete_info=cache_document('complete.json')
 complete_out=dict(status=complete_info['status'],metadata_sha256=complete_info.get('metadata_sha256'))
 if complete is not None:
  producer_status=complete.get('status')
  need(type(producer_status) is str and re.fullmatch('[A-Z0-9_]{1,96}',producer_status),'teacher_complete_producer_status')
  complete_out.update(producer_declared_status=producer_status,evidence='producer_declaration_only',
   declared_questions=scalar(complete.get('questions')),declared_trajectories=scalar(complete.get('trajectories')),
   declared_identity_sha256=hash_value(complete['identity_sha256']),
   declared_identity_matches_read_header=complete['identity_sha256']==header_info.get('metadata_sha256') if header is not None else None,
   full_sha_cache_qualification_by_observer=False)
 known=sum(item['status']=='observed' and item.get('groups_count')==512 for item in chunks)
 unknown=any(item['status']=='unknown' for item in chunks) or header_info['status']=='unknown' or complete_info['status']=='unknown'
 return dict(status='unknown' if unknown else 'observed' if header is not None else 'not_yet_recorded',
  cache_root=name,path_scope='intent_bound_external_teacher_cache_metadata',header=header_out,complete=complete_out,
  completed_chunk_metadata_count=known,validated_metadata_groups_count=groups_total,
  validated_metadata_trajectory_count=trajectories_total,expected_chunk_count=10,expected_questions=5120,
  expected_trajectories=25600,counts_scope='completed JSON chunks only; uniqueness, labels, raw-file bytes and in-progress generation are unverified',
  current_run_generation='unknown_reusable_cache_origin_not_proven',chunks=chunks,metadata_files=files,
  file_limit_bytes=4*1024*1024,scope=scope)
raw_payload=sys.stdin.read(1024*1024+1);need(len(raw_payload.encode())<=1024*1024,'payload_limit')
request=json.loads(raw_payload);sys.stdin=io.StringIO(raw_payload)
result=dict(role='qprobe_existing_sigma3_fresh_window_readonly',run_id=request['expected_run'],execution_id=request['expected_execution'],
 source_commit=request['intent']['source_commit'],base=request['intent']['base'],status='unknown',metadata_only=True,
 read_once=True,remote_mutation=False,no_model_or_tensor_reads=True,no_qa_or_token_files=True,no_native_verifier=True)
try:
 exec(common_code,globals());run=pinned_run();result['fresh_origin']=fresh_origin()
 need(run['actor_micro_batch_per_rank']==4 and run['budget']['per_rank_actor_micro_batch']==4 and
  run['budget']['ppo_micro_batch_size']==16 and run.get('question_probe') is not None,'micro4_qprobe_contract')
 need(type(payload['after_step']) is int and 0<=payload['after_step']<=200,'after_step')
 source_map_sha=digest(canonical(run['source_sha256']));read_pass='run_manifest_hash'
 run_manifest_sha=digest(budget_bytes(base/'logs/run-manifest.json',1024*1024))
 complete_updates=None;closed_attempts=None;driver_attempt_partial=False
 read_pass='births'
 result['births']=section(births);result['reward_window']=section(reward_window)
 result['infra_window']=section(infra_window)
 result['runtime_status']=section(runtime_status);result['opportunity_counts']=section(opportunity_counts)
 result['loss_identity']=section(loss_identity)
 result['warmup_training_records']=section(warmup_training_records)
 result['latest_prefit_diagnostics']=section(latest_prefit_diagnostics)
 result['latest_head_commit']=section(head_commit)
 result['checkpoint_head_metadata']=section(checkpoint_metadata);result['gpu_samples']=section(gpu_samples)
 result['existing_teacher_cache_progress']=section(existing_teacher_cache_progress)
 need('torch' not in sys.modules,'unexpected_torch_import')
 result['status']='observed';result['metadata_files']=metadata_files
except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
 result.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
finally:
 signal.alarm(0);result.update(observed_unix=time.time(),aggregate_bytes_read=spent,bounded_failures=bounded_failures,
  limits=dict(aggregate_read_bytes=TOTAL_LIMIT,ledger_bytes=LEDGER_LIMIT,rank_ledger_bytes=RANK_LEDGER_LIMIT,process_ledger_bytes=PROCESS_LEDGER_LIMIT,
   output_bytes=128*1024,deadline_seconds=65,
   readback_recheck_included=True,source_byte_checks_included=True))
if len(json.dumps(result,allow_nan=False).encode())>JSON_LIMIT and 'warmup_training_records' in result:
 result['warmup_training_records'].pop('minibatch_loss_curve',None)
 result['warmup_training_records']['minibatch_curve_omitted_for_output_budget']=True
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>JSON_LIMIT and 'existing_teacher_cache_progress' in result:
 result['existing_teacher_cache_progress']=dict(status='unknown',reason='optional_output_budget',
  scope='optional teacher cache metadata omitted to preserve the primary scientific result')
 encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>JSON_LIMIT and 'infra_window' in result:
 supplemental=result['infra_window']
 result['infra_window']=dict(status='unknown',reason='supplemental_output_budget',
  eligibility=supplemental.get('eligibility'),latest=supplemental.get('latest'),
  scope='optional saved infra scalars omitted to preserve the primary scientific result')
 encoded=json.dumps(result,allow_nan=False)
 if len(encoded.encode())>JSON_LIMIT:
  result['infra_window'].pop('latest',None);encoded=json.dumps(result,allow_nan=False)
 if len(encoded.encode())>JSON_LIMIT:
  # Even a small unknown marker is optional when the primary fills the cap.
  result.pop('infra_window');encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>JSON_LIMIT:
 encoded=json.dumps(dict(role=result['role'],run_id=result['run_id'],execution_id=result['execution_id'],source_commit=result['source_commit'],
  metadata_only=True,read_once=True,status='unknown',reason='output_limit',bounded_failures=bounded_failures),allow_nan=False)
print('MILESTONE_JSON:'+encoded,flush=True)
'''


def load_flex():
    raw = (OPS/FLEX_PATH).read_bytes()
    env = dict(os.environ, LC_ALL='C', LANG='C')
    def git(*args):
        return subprocess.check_output(['git','-C',str(OPS),*args],env=env,timeout=30)
    commit = git('rev-parse','HEAD').decode().strip()
    own = Path(__file__).resolve()
    assert git('show',commit+':'+str(own.relative_to(OPS))) == own.read_bytes(), 'Publish this exact observer first'
    for relative,publication,sha in ((PREDECESSOR_PATH,PREDECESSOR_COMMIT,PREDECESSOR_SHA),
                                      (SUPPORT_PATH,SUPPORT_COMMIT,SUPPORT_SHA)):
        data=(OPS/relative).read_bytes()
        assert hashlib.sha256(data).hexdigest()==sha
        assert git('show',publication+':'+relative)==git('show',commit+':'+relative)==data
    support_nodes=ast.parse((OPS/SUPPORT_PATH).read_bytes()).body
    support=next(ast.literal_eval(node.value) for node in support_nodes if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='REMOTE_SUPPORT' for t in node.targets))
    assert REMOTE_SUPPORT==support
    assert hashlib.sha256(raw).hexdigest() == FLEX_SHA
    assert git('show',FLEX_COMMIT+':'+FLEX_PATH) == git('show',commit+':'+FLEX_PATH) == raw
    spec = importlib.util.spec_from_file_location('reviewed_qprobe_flex_metadata',OPS/FLEX_PATH)
    flex = importlib.util.module_from_spec(spec);spec.loader.exec_module(flex)
    common = flex.REMOTE_COMMON
    # Only replace byte reads with bounded, owned, twice-read snapshots. Every
    # reviewed identity/source/UUID/binding assertion remains the same.
    replacements = {
        'return json.loads(path.read_text())': 'return json.loads(budget_bytes(path,1024**2))',
        'owned(path).read_bytes()': 'budget_bytes(owned(path),4*1024*1024)',
        'binding_path.read_bytes()': 'budget_bytes(binding_path,1024**2)',
    }
    for before,after in replacements.items():
        assert common.count(before) == 1
        common = common.replace(before,after)
    remote = REMOTE_SUPPORT+'\ncommon_code='+repr(common)+'\n'+REMOTE_READ
    ast.parse(remote)
    return flex,commit,remote,hashlib.sha256(flex.REMOTE_COMMON.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--expected-run',required=True)
    parser.add_argument('--expected-execution',required=True)
    parser.add_argument('--after-step',type=int,required=True)
    parser.add_argument('--receipt',type=Path)
    args = parser.parse_args()
    assert 0 <= args.after_step <= 200
    flex,commit,remote,common_sha = load_flex()
    args.handle.relative_to(ROOT/'preparation');intent = flex.read(args.handle)
    assert all(re.fullmatch('[0-9a-f]{32}',value) for value in (args.expected_run,args.expected_execution))
    flex.authority(args.handle,intent,args.expected_run,args.expected_execution)
    module = flex.runtime_module(args.runtime,intent)
    route = module.ssh_command(remote);assert route[-1].startswith('python3 -c ')
    route[-1] = '/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    receipt = args.receipt or ROOT/'preparation'/('qprobe-sigma3-fresh-observation-'+str(time.time_ns())+'.json')
    receipt.relative_to(ROOT/'preparation')
    assert receipt.is_absolute() and receipt.resolve()==receipt and receipt.suffix=='.json' and receipt.parent.is_dir() and not receipt.exists()
    authority_sha = flex.authority(args.handle,intent,args.expected_run,args.expected_execution)
    result = dict(role='single_qprobe_existing_sigma3_fresh_window_receipt',operator_commit=commit,flex_commit=FLEX_COMMIT,
        predecessor_commit=PREDECESSOR_COMMIT,predecessor_sha256=PREDECESSOR_SHA,
        support_commit=SUPPORT_COMMIT,support_sha256=SUPPORT_SHA,
        flex_sha256=FLEX_SHA,reused_common_sha256=common_sha,remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(),
        authority_sha256=authority_sha,handle_sha256=hashlib.sha256(args.handle.read_bytes()).hexdigest(),
        run_id=args.expected_run,execution_id=args.expected_execution,source_commit=intent['source_commit'],
        base=intent['base'],after_complete_step=args.after_step,no_automatic_retry=True,observation=None)
    payload = json.dumps(dict(intent=intent,expected_run=args.expected_run,expected_execution=args.expected_execution,
        after_step=args.after_step),allow_nan=False).encode()
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            completed = subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=90)
            result.update(exit_code=completed.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None,status='TIMEOUT_OR_TRANSPORT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);result['bounded_stderr_sha256']=hashlib.sha256(err.read(64*1024)).hexdigest()
    if result['exit_code']==0 and len(raw)<=128*1024:
        try:
            rows = [json.loads(line.split(':',1)[1]) for line in raw.decode().splitlines() if line.startswith('MILESTONE_JSON:')]
            assert len(rows)==1
            value=rows[0];assert (value['run_id'],value['execution_id'],value['source_commit'])==(args.expected_run,args.expected_execution,intent['source_commit'])
            assert value['role']=='qprobe_existing_sigma3_fresh_window_readonly' and value['base']==intent['base']
            assert value['metadata_only'] is True and value['read_once'] is True
            result['observation']=value
        except (AssertionError,KeyError,TypeError,ValueError,UnicodeError) as failure:
            result.update(status='SANITIZED_OUTPUT_REJECTED',failure_type=type(failure).__name__)
    elif len(raw)>128*1024:result['status']='OUTPUT_LIMIT_EXCEEDED'
    result['observed_unix']=time.time();flex.once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),status=result['status'],exit_code=result['exit_code'],
        observation=result['observation'],no_automatic_retry=True),allow_nan=False))
    return 0 if result['observation'] is not None and result['observation']['status']=='observed' else 1


if __name__=='__main__':
    raise SystemExit(main())
