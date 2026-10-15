"""One bounded read of existing qprobe rewards, scores and native milestones.

JSON receipts report producer checks; this reader never reads tensor/QA/token
files, reruns a verifier, evaluates, launches, retries, signals or updates anchors.
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

REMOTE_SUPPORT = r'''
import hashlib,io,json,math,os,re,signal,stat,sys,time
from pathlib import Path
TOTAL_LIMIT=64*1024*1024
SMALL_LIMIT=2*1024*1024
LEDGER_LIMIT=8*1024*1024
LINE_LIMIT=128*1024
deadline=time.monotonic()+65
spent=0
metadata_files=[]
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
  need(tail or before.st_size<=limit,'file_limit')
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
def rows(relative,tail=False):
 if (relative,tail) in ledgers:return ledgers[(relative,tail)]
 raw,info=snapshot(base/relative,LEDGER_LIMIT,append=True,tail=tail)
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
 info['complete_line_count']=len(values);ledgers[(relative,tail)]=(values,info);return values,info
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
 try:return function()
 except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
  return dict(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
'''

REMOTE_READ = r'''
def births():
 producer=run['pipeline_identity'];claim,_=document('logs/training-launch-claim.json')
 ledger,_=document('logs/process-ledger.json',8*1024*1024)
 need(claim is not None and ledger is not None,'training_birth_metadata_missing')
 same_execution(claim);need(ledger['run_id']==run['run_id'],'ledger_run')
 trainer=ledger['jobs']['trainer']['leader']
 need((trainer['pid'],trainer['start_ticks'])==(claim['shell_pid'],claim['shell_start_ticks']),'trainer_claim_birth')
 pipeline_claim,_=document('logs/pipeline-launch-claim.json');need(pipeline_claim is not None,'pipeline_claim_missing')
 same_execution(pipeline_claim);need(pipeline_claim['pipeline_identity']==producer,'pipeline_claim_birth')
 return dict(status='observed',pipeline=process(producer),trainer=process(trainer))
def reward_window():
 updates,info=rows('logs/actual-updates.jsonl');attempts,_=rows('logs/actual-attempts.jsonl')
 need(updates is not None and attempts is not None,'complete_ledgers_missing')
 need(info['scan_from_start'],'updates_not_from_start')
 by_attempt={};prior_index=0
 for attempt in attempts:
  same_execution(attempt);aid=attempt['attempt_id'];index=attempt['attempt_index']
  need(type(aid) is str and re.fullmatch('[0-9a-f]{32}',aid) and aid not in by_attempt and
   integer(index,1) and index>prior_index,'attempt_order');by_attempt[aid]=attempt;prior_index=index
 prior_after=0;prior_index=0;observed=[]
 for step,row in enumerate(updates,1):
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
  hash_value(row['behavior_probability_gate_sha256']);hash_value(row['behavior_trace_sha256'])
  observed.append(dict(step=step,reward_mean=reward,successful_adam=writes,skip=20-writes,
   optimizer_writes_before=before,optimizer_writes_after=after,attempt_id=aid,attempt_index=index))
  prior_after=after;prior_index=index
 need(payload['after_step']<=len(observed),'requested_anchor_ahead_of_complete_prefix')
 globals()['complete_updates']=updates
 globals()['closed_attempts']=by_attempt
 fresh=observed[payload['after_step']:];values=[row['reward_mean'] for row in fresh]
 return dict(status='observed',recorded_complete_outers=len(observed),after_complete_step=payload['after_step'],
  newly_complete_steps=[row['step'] for row in fresh],new_complete_rows=fresh,
  latest=observed[-1] if observed else None,window_count=len(values),window_mean=math.fsum(values)/len(values) if values else None,
  window_min=min(values) if values else None,window_max=max(values) if values else None,
  complete_prefix_sha256=info['complete_prefix_sha256'],partial_final_status=info['partial_final_status'],
  scope='returned whole formal outer rows once per four-rank collective; no partial attempt is included')
def opportunity_counts():
 need(closed_attempts is not None and complete_updates is not None,'closed_attempt_prefix_unknown')
 closed=sorted(closed_attempts.values(),key=lambda item:item['attempt_index'])
 need([item['attempt_index'] for item in closed]==list(range(1,len(closed)+1)),'closed_attempt_sequence')
 need(sum(item['complete_outer'] is True for item in closed)==len(complete_updates),'driver_prefixes_not_same_snapshot')
 results=[];signatures=[];partial_files=False
 for rank in range(4):
  values,info=rows('logs/optimizer-rank-'+str(rank)+'.jsonl')
  need(values is not None and info['scan_from_start'],'opportunity_prefix_missing_or_not_from_start')
  groups=[];signature=[];writes=0
  for row in values:
   need(row.get('run_id')==run['run_id'] and type(row.get('rank')) is int and row['rank']==rank,'opportunity_row_identity')
   if row.get('kind')!='OPPORTUNITY':continue
   context=row['context'];aid=context['attempt_id'];ordinal=context['attempt_index'];outer=context['logical_update']
   gate_sha=hash_value(context['behavior_probability_gate_sha256'])
   need(type(aid) is str and re.fullmatch('[0-9a-f]{32}',aid) and integer(ordinal,1) and integer(outer,1) and outer<=200,'opportunity_context')
   if not groups or groups[-1]['attempt_id']!=aid:
    need(ordinal==len(groups)+1 and aid not in {item['attempt_id'] for item in groups},'opportunity_attempt_sequence')
    groups.append(dict(attempt_id=aid,attempt_index=ordinal,logical_update=outer,processed=0,successful_adam=0,
     optimizer_writes_before=writes,optimizer_writes_after=writes,behavior_probability_gate_sha256=gate_sha))
   group=groups[-1];index=row['opportunity_index'];before=row['optimizer_writes_before'];after=row['optimizer_writes_after'];wrote=row['optimizer_write_succeeded']
   need(group['attempt_index']==ordinal and group['logical_update']==outer and type(index) is int and
    index==group['processed']<20 and context['minibatch_index']==index and context['prior_optimizer_writes']==before and
    row['processed'] is True and type(wrote) is bool and integer(before) and integer(after) and before==writes and after-before==int(wrote),
    'opportunity_sequence_or_cumulative_write')
   need(row['status'] in (('written','written_with_post_guard') if wrote else ('skipped_guard',)),'opportunity_status')
   need(group['behavior_probability_gate_sha256']==gate_sha,'attempt_behavior_gate_changed')
   group['processed']+=1;group['successful_adam']+=int(wrote);group['optimizer_writes_after']=after;writes=after
   signature.append([aid,ordinal,outer,index,before,after,wrote,row['status'],gate_sha])
  for attempt in closed:
   ordinal=attempt['attempt_index'];need(ordinal<=len(groups),'closed_attempt_opportunities_not_yet_visible')
   group=groups[ordinal-1]
   need(group['attempt_id']==attempt['attempt_id'] and group['logical_update']==attempt['logical_update'] and
    group['processed']==attempt['opportunities_processed'] and group['successful_adam']==attempt['optimizer_writes_this_attempt'] and
    group['optimizer_writes_before']==attempt['optimizer_writes_before'] and group['optimizer_writes_after']==attempt['optimizer_writes_after'],
    'opportunities_disagree_with_closed_attempt')
   if attempt['complete_outer'] is True:
    update=complete_updates[attempt['logical_update']-1]
    need(update['attempt_id']==group['attempt_id'] and update['behavior_probability_gate_sha256']==group['behavior_probability_gate_sha256'],
     'opportunity_gate_differs_from_returned_complete_update')
  need(len(groups) in (len(closed),len(closed)+1),'more_than_one_unclosed_attempt')
  partial=groups[-1] if len(groups)>len(closed) else None
  if partial:need(partial['logical_update']==len(complete_updates)+1,'partial_logical_outer')
  processed=sum(group['processed'] for group in groups);successful=sum(group['successful_adam'] for group in groups)
  need(successful==writes,'cumulative_adam_not_from_initial_zero')
  partial_files=partial_files or info['partial_final_bytes']>0
  signatures.append(digest(canonical(signature)))
  results.append(dict(rank=rank,complete_prefix_processed=processed,complete_prefix_successful_adam=successful,
   complete_prefix_skip=processed-successful,current_partial=partial,prefix_sha256=info['complete_prefix_sha256'],
   partial_final_status=info['partial_final_status']))
 agreement=len(set(signatures))==1
 return dict(status='observed' if agreement and not partial_files else 'unknown',
  reason=None if agreement and not partial_files else 'four_rank_prefixes_disagree' if not agreement else 'incomplete_final_row',
  four_rank_complete_prefix_agreement=agreement,per_rank=results,
  cumulative_processed=results[0]['complete_prefix_processed'] if agreement and not partial_files else None,
  cumulative_successful_adam=results[0]['complete_prefix_successful_adam'] if agreement and not partial_files else None,
  cumulative_skip=results[0]['complete_prefix_skip'] if agreement and not partial_files else None,
  current_partial=results[0]['current_partial'] if agreement and not partial_files else None,
  scope='from-start newline-complete OPPORTUNITY receipts including closed and current unclosed attempts; four replicas counted once; no tail-derived cumulative count')
def runtime_status():
 stage,_=document('logs/pipeline-status.json');ledger,_=document('logs/process-ledger.json',8*1024*1024)
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
def scores():
 result={}
 for relative,split in (('logs/latest-stage-identity.json','stage'),('logs/latest-validation-identity.json','validation'),
  ('logs/formal-inline-development-after.json','development_after5')):
  value,info=document(relative)
  if value is None:result[split]=dict(status='missing');continue
  need(value['run_id']==run['run_id'] and integer(value['step']) and value['step']<=200,'score_identity')
  if split=='development_after5':
   need(value['role']=='formal_inline_development_evaluation' and value['step']==5 and value['source_map_sha256']==source_map_sha,'development_identity')
  else:need(value['split']==split and value['final'] is False,'score_split')
  counts=value.get('counts',value.get('source_counts'));expected={'nq':32,'hotpotqa':32} if split=='stage' else {'nq':256,'hotpotqa':256}
  need(counts==expected,'score_counts');sha=run['input_sha256']['stage.parquet' if split=='stage' else 'validation.parquet']
  need(value['validation_sha256']==sha,'score_data_identity')
  em=value['weighted_em'];f1=value['weighted_f1'];need(number(em) and number(f1) and 0<=em<=f1<=1,'score_scalars')
  result[split]=dict(status='observed',step=value['step'],weighted_em=em,weighted_f1=f1,question_count=sum(expected.values()),
   source_counts=counts,metadata_sha256=info['scanned_sha256'],validation_sha256=sha,
   scope='existing producer evaluation scalars; no answers, token rows or new scoring read')
 return dict(status='observed',scores=result)
def native_five():
 value,info=document('logs/five-updates-acceptance.json',8*1024*1024)
 if value is None:return dict(status='missing',accepted=None,scope='row5 alone never establishes native-five acceptance')
 need(value['run_id']==run['run_id'] and type(value['accepted']) is bool,'nativefive_identity')
 safety=value['formal_inline_first_five'];need(safety['run_id']==run['run_id'] and safety['source_map_sha256']==source_map_sha,'nativefive_source')
 need(complete_updates is not None and len(complete_updates)>=5 and [row['step'] for row in value['actual_updates']]==list(range(1,6)),'nativefive_steps')
 need(all(row.get('step')==step and all(row.get('metrics',{}).get(key)==metric for key,metric in complete_updates[step-1]['metrics'].items())
  for step,row in enumerate(value['actual_updates'],1)),'nativefive_recorded_metrics')
 gpu=value['actual_first_five_gpu_occupancy'];need(gpu['run_id']==run['run_id'],'nativefive_gpu_identity')
 windows=gpu.get('actual_optimizer_windows',[]);need(len(windows)<=5,'nativefive_window_limit')
 return dict(status='observed',accepted=value['accepted'],metadata_sha256=info['scanned_sha256'],
  first_five_optimizer_opportunities=scalar(value.get('first_five_optimizer_opportunities')),
  recorded_safety=select(safety,('actual_complete_outers','actual_optimizer_writes_per_rank','completed_outer_optimizer_opportunities_per_rank',
   'unique_global_monitor_count','guard_triggered_monitor_count','sampled_kl_max','negative_tail_weight_fraction_max')),
  recorded_gpu_windows=[select(window,('logical_update','begin_unix','complete_unix','sample_count_inside_optimizer_window')) for window in windows],
  gpu_receipt_scope='producer-recorded complete-window times/counts; its sampled_peak_mib spans training claim through five gate, not each window peak',
  scope='already-produced structural first1..5 acceptance; no new tensor finite/reload/accuracy gate or full200 acceptance')
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
raw_payload=sys.stdin.read(1024*1024+1);need(len(raw_payload.encode())<=1024*1024,'payload_limit')
request=json.loads(raw_payload);sys.stdin=io.StringIO(raw_payload)
result=dict(role='qprobe_existing_milestone_window_readonly',run_id=request['expected_run'],execution_id=request['expected_execution'],
 source_commit=request['intent']['source_commit'],base=request['intent']['base'],status='unknown',metadata_only=True,
 read_once=True,remote_mutation=False,no_model_or_tensor_reads=True,no_qa_or_token_files=True,no_native_verifier=True)
try:
 exec(common_code,globals());run=pinned_run()
 need(run['actor_micro_batch_per_rank']==4 and run['budget']['per_rank_actor_micro_batch']==4 and
  run['budget']['ppo_micro_batch_size']==16 and run.get('question_probe') is not None,'micro4_qprobe_contract')
 need(type(payload['after_step']) is int and 0<=payload['after_step']<=200,'after_step')
 source_map_sha=digest(canonical(run['source_sha256']));run_manifest_sha=digest(budget_bytes(base/'logs/run-manifest.json',1024*1024))
 complete_updates=None;closed_attempts=None
 result['births']=births();result['reward_window']=section(reward_window)
 result['runtime_status']=section(runtime_status);result['opportunity_counts']=section(opportunity_counts)
 result['scores']=section(scores);result['native_first_five']=section(native_five)
 result['latest_head_commit']=section(head_commit)
 result['checkpoint_head_metadata']=section(checkpoint_metadata);result['gpu_samples']=section(gpu_samples)
 need('torch' not in sys.modules,'unexpected_torch_import')
 result['status']='observed';result['metadata_files']=metadata_files
except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError) as failure:
 result.update(status='unknown',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__)
finally:
 signal.alarm(0);result.update(observed_unix=time.time(),aggregate_bytes_read=spent,
  limits=dict(aggregate_read_bytes=TOTAL_LIMIT,ledger_bytes=LEDGER_LIMIT,output_bytes=128*1024,deadline_seconds=65,
   readback_recheck_included=True,source_byte_checks_included=True))
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>128*1024:
 encoded=json.dumps(dict(role=result['role'],run_id=result['run_id'],execution_id=result['execution_id'],source_commit=result['source_commit'],
  metadata_only=True,read_once=True,status='unknown',reason='output_limit'),allow_nan=False)
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
    receipt = args.receipt or ROOT/'preparation'/('qprobe-milestone-observation-'+str(time.time_ns())+'.json')
    receipt.relative_to(ROOT/'preparation')
    assert receipt.is_absolute() and receipt.resolve()==receipt and receipt.suffix=='.json' and receipt.parent.is_dir() and not receipt.exists()
    authority_sha = flex.authority(args.handle,intent,args.expected_run,args.expected_execution)
    result = dict(role='single_qprobe_existing_milestone_window_receipt',operator_commit=commit,flex_commit=FLEX_COMMIT,
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
            assert value['role']=='qprobe_existing_milestone_window_readonly' and value['base']==intent['base']
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
