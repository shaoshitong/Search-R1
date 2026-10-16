"""Bounded read-only final TP1 score/cost/resource/cleanup audit. No model imports or writes."""
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TRAIN='778b2b43dad90fc0c1c5d95d77c2326286011b73'
RUN=ROOT/'runs'/('tree-nested-branch-fresh200-not-slurm;systemd:'+TRAIN)
UNIT='sst-tree-nested-final-eval-tp1-778b2b43-20261016.service'
INV='abd6a826997945da932cf09047806431'
EVAL='a8ab17f36f1bcf64f19193522aaed84fb2749863'
CODE_SHA='439a58113f8e12e557121ef3b7b241ebd740376c240e1cd40b27be2083c5e9b4'
RUNTIME=ROOT/'runs'/UNIT.removesuffix('.service')
OUT=RUN/'output/final-eval-fast';SCORES=OUT/'scores'
IDENTITY=dict(config_sha256='355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3',model_sha256='c0360891166e7bc7e1a10af023f9762ede122976f5c10873ccf5725035d8d416',dataset_sha256='2598a00883d70ace097e4d0dc10056066b4f6d0d99d9d6034d902dd09da732fa')
COUNTS=dict(nq=3610,triviaqa=11313,popqa=14267,hotpotqa=7405,**{'2wikimultihopqa':12576},musique=2417,bamboogle=125)
KEYS=('f1','em','answer_format','format_rate','generated_tokens','search_calls')
UINT64_MAX=18446744073709551615

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def numeric(v):return type(v) in (int,float) and math.isfinite(v)
def strict(raw):
 def unique(pairs):
  out={}
  for k,v in pairs:require(k not in out,'Duplicate JSON key');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def signature(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read(path,expected=None,limit=1000000):
 p=Path(path);require(p.is_absolute() and '..' not in p.parts and not p.is_symlink(),'Exact ordinary own metadata path')
 require(p.resolve(strict=True).is_relative_to(ROOT.resolve()),'Own retained storage only')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  before=os.fstat(f.fileno());require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=limit,'Bounded metadata size')
  raw=f.read(limit+1);after=os.fstat(f.fileno())
 require(signature(before)==signature(after)==signature(p.stat()) and len(raw)==before.st_size,'Metadata changed while read')
 require(expected is None or sha(raw)==expected,'Metadata source SHA changed: '+p.name)
 return strict(raw),dict(path=str(p),resolved_path=str(p.resolve()),bytes=len(raw),sha256=sha(raw),mtime_ns=before.st_mtime_ns,stable_read=True)
def systemd(unit):
 props='LoadState,MainPID,ActiveState,SubState,Result,ExecMainStatus,InvocationID,ControlGroup,CPUUsageNSec,IOReadBytes,IOWriteBytes,MemoryPeak,MemoryCurrent,TasksCurrent,User,UID,ActiveEnterTimestampMonotonic,InactiveEnterTimestampMonotonic,ExecMainStartTimestampMonotonic,ExecMainExitTimestampMonotonic'
 raw=subprocess.check_output(['systemctl','show',unit,'--property='+props],text=True,timeout=15)
 return dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
def available_counter(raw):
 if not isinstance(raw,str) or not raw.isdecimal():return None
 value=int(raw);return None if value==UINT64_MAX else value

def exec_wall(state):
 start=available_counter(state.get('ExecMainStartTimestampMonotonic'));end=available_counter(state.get('ExecMainExitTimestampMonotonic'))
 return (end-start)/1e6 if start is not None and end is not None and 0<start<=end else None

def auxiliary_cleanup(pins):
 value,source=read(pins['auxiliary_cleanup_remote_path'])
 require(value==pins['auxiliary_cleanup_expected_object'] and value['schema']=='tree_completed_cpu_audit_cleanup_v1' and value['status']=='COMPLETE','Pinned completed auxiliary cleanup receipt changed')
 expected={'sst-tree-nativehf200-cpu-caac340-778b2b43.service':'510e864008f548fcbefcc7143d7f5976',
           'sst-tree-trace200-durable-repair001.service':'94fd3b56080743aa884d19e9fefb40e6'}
 require(set(value['before'])==set(expected) and len(value['post_stop_readbacks'])==2,'Exact two audit units/two saved poststop reads')
 summaries=[]
 for unit,invocation in expected.items():
  before=value['before'][unit]['properties'];require(before['InvocationID']==invocation and before['MainPID']=='0' and before['Result']=='success' and before['ExecMainStatus']=='0','Actual audit pre-stop identity/counters')
  require(value['stop_results'][unit]['returncode']==0,'Actual own audit stop result')
  after=[]
  for frame in value['post_stop_readbacks']:
   item=frame[unit];state=item['properties'];require(state['MainPID']=='0' and state['ActiveState']=='inactive' and not item['own_cgroup_pids'] and state.get('InvocationID','') in ('',invocation),'Saved own audit post-stop proof')
   after.append(dict(epoch=item['epoch'],systemd=state,CPU_seconds=available_counter(state.get('CPUUsageNSec'))/1e9 if available_counter(state.get('CPUUsageNSec')) is not None else None,IO_read_bytes=available_counter(state.get('IOReadBytes')),IO_write_bytes=available_counter(state.get('IOWriteBytes')),own_pids=item['own_cgroup_pids']))
  summaries.append(dict(unit=unit,expected_invocation=invocation,pre_stop_systemd=before,pre_stop_CPU_seconds=available_counter(before['CPUUsageNSec'])/1e9,pre_stop_IO_read_bytes=available_counter(before['IOReadBytes']),pre_stop_IO_write_bytes=available_counter(before['IOWriteBytes']),post_stop_readbacks=after,root_stop_observed=True,observer_cost_is_separate_from_training_and_final_eval=True))
 return dict(source=source,local_root_receipt=pins['auxiliary_cleanup_local_proof'],units=summaries,root_stop_observed=True,live_units_requeried=False,stop_action_performed=False)

def cleanup_frame():
 state=systemd(UNIT);matched=state.get('InvocationID')==INV
 require(state.get('InvocationID','') in ('',INV),'Final evaluator invocation changed')
 require(matched or state.get('LoadState')=='not-found','Only original invocation or GC-unavailable transient unit')
 require(state.get('MainPID')=='0' and state.get('ActiveState') in ('inactive','failed'),'Wait for actual terminal service before final read')
 require(not matched or (state.get('Result')=='success' and state.get('ExecMainStatus')=='0'),'Observed original evaluator systemd exit must be zero')
 relative=state.get('ControlGroup') or '/system.slice/'+UNIT
 require(relative=='/system.slice/'+UNIT,'Exact own cgroup only')
 scope=Path('/sys/fs/cgroup')/relative.lstrip('/');lists=[];pids=set();cg={}
 if scope.exists():
  require(scope.is_dir() and not scope.is_symlink(),'Own cgroup redirected')
  for p in sorted(scope.rglob('cgroup.procs')):
   require(not p.is_symlink(),'Own cgroup process list redirected');values=p.read_text().split();require(all(v.isdecimal() for v in values),'Own PID metadata')
   pids.update(map(int,values));lists.append(dict(path=str(p),pids=list(map(int,values))))
  require(len(pids)<=4096,'Own systemd task ceiling; never machine-wide ps')
  for name in ('cpu.stat','memory.current','memory.peak','memory.events','io.stat','io.pressure','pids.current'):
   p=scope/name
   if p.is_file():cg[name]=p.read_text()[:32768]
 # Inspect only the fixed eight cooperative lease inodes; never acquire/delete a lock.
 uid=state.get('UID','');user=state.get('User','')
 if uid.isdecimal():uid=int(uid)
 elif user in ('','root'):uid=0
 else:
  import pwd
  uid=pwd.getpwnam(user).pw_uid
 lease_root=Path('/tmp')/('search-r1-gpu-leases-'+str(uid));lease_rows=[];lease_keys={}
 for uuid in GPU_UUIDS:
  p=lease_root/(uuid+'.lock');row=dict(path=str(p),exists=p.exists(),lock_holders=[])
  if p.exists():
   require(not p.is_symlink() and p.is_file(),'Own cooperative lease path redirected');s=p.stat();key=(os.major(s.st_dev),os.minor(s.st_dev),s.st_ino);lease_keys[key]=row;row.update(device_major=key[0],device_minor=key[1],inode=key[2],bytes=s.st_size)
  lease_rows.append(row)
 for line in Path('/proc/locks').read_text().splitlines():
  fields=line.replace('->','').split()
  if len(fields)<6 or not re.fullmatch(r'[0-9a-fA-F]+:[0-9a-fA-F]+:\d+',fields[5]):continue
  major,minor,inode=fields[5].split(':');key=(int(major,16),int(minor,16),int(inode))
  if key not in lease_keys:continue
  pid=int(fields[4]);own=None
  try:own=any(UNIT in q.split(':',2)[-1].split('/') for q in Path(f'/proc/{pid}/cgroup').read_text().splitlines())
  except FileNotFoundError:own=False
  lease_keys[key]['lock_holders'].append(dict(pid=pid,own_unit=own,lock_type=fields[1],mode=fields[3]))
 # Bounded port-specific socket metadata only. No request, bind or global process scan.
 listeners=[]
 for name in ('tcp','tcp6'):
  for line in Path('/proc/net/'+name).read_text().splitlines()[1:]:
   f=line.split();address,port=f[1].split(':')
   if int(port,16)==18082 and f[3]=='0A':listeners.append(dict(table=name,address_hex=address,port=18082,socket_inode=f[9]))
 own_leases_clear=all(h['own_unit'] is False for row in lease_rows for h in row['lock_holders'])
 return dict(epoch=time.time(),unit=UNIT,invocation_id=INV,systemd=state,cgroup_path=str(scope),cgroup_exists=scope.exists(),own_cgroup_process_lists=lists,own_pids=sorted(pids),cgroup_scalar_files=cg,lease_records=lease_rows,own_lease_holders_absent=own_leases_clear,lease_files_not_deleted=True,port18082_listeners=listeners,loopback18082_listen_absent=not listeners,own_loopback18082_listener_absent=not pids,port_listener_scope='When own cgroup is empty no process in this unit owns a listener; other listeners are foreign/unknown and are not operated.',
  current_invocation_metadata_available=matched,systemd_cpu_seconds=(available_counter(state.get('CPUUsageNSec'))/1e9 if matched and available_counter(state.get('CPUUsageNSec')) is not None else None),io_read_bytes=available_counter(state.get('IOReadBytes')) if matched else None,io_write_bytes=available_counter(state.get('IOWriteBytes')) if matched else None,UINT64_MAX_IO_unavailable=True)

def resources():
 p=RUNTIME/'resource-samples.jsonl';require(p.is_file() and not p.is_symlink(),'Own resource stream required');before=p.stat();require(before.st_size<=256*1024*1024,'Bounded resource stream only')
 digest=hashlib.sha256();count=bad=0;first=last=None;periods=[];peaks={};events=[]
 with p.open('rb') as f:
  for line in f:
   digest.update(line)
   try:v=strict(line)
   except (ValueError,UnicodeError):bad+=1;continue
   require(v['unit']==UNIT and v['physical_gpus']==8 and v['slurm_reserved_gpus']==0 and v['resource_protocol']=='memory_headroom_shared_v1','Own resource stream identity')
   epoch=v['epoch'];require(numeric(epoch) and (last is None or epoch>=last),'Resource epochs ordered')
   if first is None:first=epoch
   if last is not None:periods.append(epoch-last)
   last=epoch;count+=1
   if v.get('event'):events.append(dict(epoch=epoch,event=v['event'],reason=v.get('reason')))
   require({g['uuid'] for g in v['gpus']}==set(GPU_UUIDS),'Eight resource GPU identities')
   for g in v['gpus']:
    totals=[g[k] for k in ('total_bytes','free_bytes','used_bytes','total_gpu_util','own_memory_bytes')];require(all(numeric(z) and z>=0 for z in totals),'Finite resource counters')
    require(0<=g['total_gpu_util']<=100 and all(type(z) is int and z>=0 for z in g['foreign'].values()),'Whole-card utilization/foreign memory')
    peak=peaks.setdefault(g['uuid'],dict(total_bytes=g['total_bytes'],sampled_whole_card_used_peak_bytes=0,sampled_own_memory_peak_bytes=0,sampled_foreign_memory_peak_bytes=0,sampled_whole_card_gpu_util_peak=0,sampled_free_min_bytes=g['free_bytes']))
    require(peak['total_bytes']==g['total_bytes'],'GPU total memory changed')
    for key,value in [('sampled_whole_card_used_peak_bytes',g['used_bytes']),('sampled_own_memory_peak_bytes',g['own_memory_bytes']),('sampled_foreign_memory_peak_bytes',sum(g['foreign'].values())),('sampled_whole_card_gpu_util_peak',g['total_gpu_util'])]:peak[key]=max(peak[key],value)
    peak['sampled_free_min_bytes']=min(peak['sampled_free_min_bytes'],g['free_bytes'])
 require(signature(before)==signature(p.stat()) and count>=2,'Terminal resource stream changed/insufficient')
 return dict(source=dict(path=str(p),resolved_path=str(p.resolve()),bytes=before.st_size,sha256=digest.hexdigest(),mtime_ns=before.st_mtime_ns,stable_read=True),valid_sample_count=count,invalid_or_partial_jsonl_lines=bad,first_epoch=first,last_epoch=last,observed_span_seconds=last-first,configured_guard_sleep_seconds=1,actual_sample_period=dict(count=len(periods),minimum_seconds=min(periods),maximum_seconds=max(periods),mean_seconds=math.fsum(periods)/len(periods),gaps_over2_seconds=sum(z>2 for z in periods)),sampled_gpu_peaks=peaks,guard_events=events,scope='Whole-card peaks are sampled lower bounds. Own/foreign memory is recorded separately; whole-card GPU utilization is not own utilization. No exclusive billing/energy inferred.',raw_resource_samples_transferred=False)

def collect(pins):
 started=time.time();cpu=time.process_time();os.nice(19);require(socket.gethostname().split('.')[0]=='h100-10','Original retained node only')
 global GPU_UUIDS
 GPU_UUIDS=pins['gpu_uuids'];require(len(GPU_UUIDS)==8 and len(set(GPU_UUIDS))==8,'Eight pinned UUIDs')
 code,code_ref=read(ROOT/'instruct300'/('eval-nested-'+EVAL)/'code-manifest.json',CODE_SHA)
 require(code['eval_wrapper_source_commit']==EVAL and code['training_source_commit']==TRAIN,'Evaluator/trainer sources')
 for name,digest in code['files'].items():
  require(Path(name).name==name and name.endswith(('.py','.md')),'Exact bounded source basename')
  read_code=Path(code_ref['path']).parent/name;require(read_code.is_file() and not read_code.is_symlink() and read_code.stat().st_size<=1000000,'Small ordinary evaluator code only');require(sha(read_code.read_bytes())==digest,'Exact small evaluator code changed')
 assets,assets_ref=read(OUT/'asset-manifest.json',pins['asset_manifest_sha256']);require({k:assets[k] for k in IDENTITY}==IDENTITY and assets['dataset_count']==51713 and assets['expected_source_counts']==COUNTS,'Bound model/data/config scalar identity')
 handoff,handoff_ref=read(OUT/'handoff-authorized.json',pins['handoff_sha256']);require(handoff['training_source_commit']==TRAIN and handoff['actual_training_completed_outer']==200 and handoff['authority']['scope']=='only_final_evaluation_after_completed_training200','Explicit original final-only handoff')
 prefix,prefix_ref=read(handoff['prefix']['path'],handoff['prefix']['sha256']);require(prefix['rows']==[],'This execution starts from original empty bound prefix')
 run,run_ref=read(RUNTIME/'run-result.json');require(run['status']=='COMPLETE' and run['worker_exit_code']==0 and run['evaluated_count']==51713,'Actual complete evaluator required')
 require(run['unit']==UNIT and run['evaluation_wrapper_source_commit']==EVAL and run['training_source_commit']==TRAIN and run['actual_completed_training_outer']==200 and run['code_manifest_sha256']==CODE_SHA and run['handoff_sha256']==pins['handoff_sha256'],'Terminal source/provenance changed')
 require(run['training_signals_sent'] is False and run['optimizer_constructed'] is False,'Evaluation only')
 final,final_ref=read(SCORES/'final-metrics.json',run['final_metrics_sha256']);done,done_ref=read(SCORES/'final-metrics.done.json',run['final_metrics_done_sha256']);require(done['sha256']==final_ref['sha256'] and final['identity']==IDENTITY and final['evaluated_count']==51713,'Final metric/done binding')
 first=cleanup_frame();first_aux=auxiliary_cleanup(pins);allocation,allocation_ref=read(RUNTIME/'allocation-identity.json');require(allocation['unit']==UNIT and allocation['cgroup']=='/system.slice/'+UNIT and set(allocation['leased_gpu_uuids'])==set(GPU_UUIDS),'Own allocation identity')
 expected_exec=dict(local_batch_size=64,world_size=8,tensor_parallel_size=1,resident_weights=True,free_cache_engine=False,rollout_padding_divisor=1,backend='XFORMERS',dtype='bfloat16',native_emission=True,resource_protocol='memory_headroom_shared_v1',allocation_kind='shared_systemd',physical_gpus=8,slurm_reserved_gpus=0,gpu_memory_utilization=.20)
 provenance=[];require({p.name for p in SCORES.glob('rank-*-provenance.json')}=={f'rank-{r}-provenance.json' for r in range(8)},'Exactly8 provenance records')
 for rank in range(8):
  value,q=read(SCORES/f'rank-{rank}-provenance.json');require(value['identity']==IDENTITY and value['execution']==expected_exec,'Resident TP1 rank provenance mismatch');provenance.append(dict(rank=rank,source=q,value=value))
 require(final['execution_records']==[expected_exec],'Final provenance aggregate mismatch')
 batches=sorted((SCORES/'batches').glob('*.done.json'));require(len(batches)==809,'809 committed chunks required')
 require(len(list((SCORES/'batches').glob('*.json')))==1618,'No uncommitted/orphan score JSON at terminal')
 sums={s:{k:[] for k in KEYS} for s in COUNTS};all_values={k:[] for k in KEYS};counts=Counter();seen=set();proofs=[];rank_rows=Counter();rank_costs=defaultdict(lambda:defaultdict(list));cost_values=defaultdict(list)
 for batch_no,receipt_path in enumerate(batches):
  receipt,receipt_ref=read(receipt_path);p=receipt_path.with_name(receipt_path.name.replace('.done.json','.json'));value,q=read(p,receipt['sha256'],262144)
  lo=batch_no*64;hi=min(51713,lo+64)-1;require(p.name==f'{lo:05d}-{hi:05d}.json','Exact continuous committed batch names')
  require(value['identity']==IDENTITY and value['execution']==expected_exec and value['prefix_count']==0 and value['allocation_job']=='not-slurm;systemd:'+UNIT,'Batch identity/execution')
  require(type(value['rank']) is int and value['rank']==batch_no%8,'Rank scheduling provenance');rows=value['rows'];require(len(rows)==receipt['count']==hi-lo+1 and [r['index'] for r in rows]==list(range(lo,hi+1)),'Contiguous batch rows')
  for row in rows:
   index=row['index'];source=row['source'];require(type(index) is int and index not in seen and source in COUNTS,'Unique question/source identity');seen.add(index);counts[source]+=1
   require(re.fullmatch('[0-9a-f]{64}',row.get('question_sha256','')) is not None,'Stored original question hash required')
   for key in KEYS:
    z=row[key];require(numeric(z) and z>=0 and (key in ('generated_tokens','search_calls') or z<=1),'Finite bounded score')
    if key in ('generated_tokens','search_calls'):require(z==int(z),'Recorded integer event count')
    sums[source][key].append(z);all_values[key].append(z)
  rank=str(value['rank']);rank_rows[rank]+=len(rows)
  for key,z in value['cost'].items():
   require(numeric(z) and z>=0,'Finite recorded batch costs');cost_values[key].append(z);rank_costs[rank][key].append(z)
  proofs.append(dict(batch_no=batch_no,rank=value['rank'],first_index=lo,last_index=hi,count=len(rows),score_source=q,done_source=receipt_ref))
 require(seen==set(range(51713)) and dict(counts)==COUNTS,'51713 unique continuous indices and exact7 source counts')
 metrics={};score_totals={}
 for source in [*COUNTS,'all']:
  n=51713 if source=='all' else COUNTS[source];values=all_values if source=='all' else sums[source]
  metrics['val/qa_count/'+source]=n;score_totals[source]=dict(count=n)
  for key in KEYS:
   total=math.fsum(values[key]);metrics['val/'+key+'/'+source]=total/n;score_totals[source][key]=dict(sum=total,mean=total/n)
  if source!='all':metrics['val/test_score/'+source]=metrics['val/em/'+source];metrics['val/evaluated_count/'+source]=n
 require(set(metrics)==set(final['metrics']) and all(numeric(final['metrics'][key]) and math.isclose(z,final['metrics'][key],rel_tol=1e-12,abs_tol=1e-12) for key,z in metrics.items()),'Independent stored-score scalar recomputation mismatch')
 resource=resources();time.sleep(2);second=cleanup_frame();second_aux=auxiliary_cleanup(pins)
 for p,q in [(RUNTIME/'run-result.json',run_ref),(SCORES/'final-metrics.json',final_ref),(SCORES/'final-metrics.done.json',done_ref)]:read(p,q['sha256'])
 require(resource['source']['bytes']==(RUNTIME/'resource-samples.jsonl').stat().st_size and resource['source']['mtime_ns']==(RUNTIME/'resource-samples.jsonl').stat().st_mtime_ns,'Terminal resource stream changed between frames')
 original_context={}
 for key,q in pins['original_final_only_inputs'].items():
  value,meta=read(q['path'],q['sha256']);original_context[key]=dict(source=meta,value=value)
 require(original_context['run_result']['value']['code_commit']==TRAIN and original_context['run_result']['value']['status']=='FAILED','Preserve original user-stopped final terminal')
 require(original_context['partial_final_control']['value'].get('phase')=='final' and original_context['partial_final_control']['value'].get('code_commit')==TRAIN,'Exact interrupted original final control scope')
 original_state=pins['original_systemd_asof'];require(original_state.get('InvocationID')=='a313bda919834d24b4c5d491698d39ce' and original_state.get('MainPID')=='0','Exact preserved original training terminal state')
 preserved=[state for state in pins['root_preserved_unit_states'] if state.get('InvocationID')==INV]
 terminal_states=[state for state in preserved if state.get('MainPID')=='0' and state.get('Result')=='success' and state.get('ExecMainStatus')=='0']
 terminal_cpu=second['systemd_cpu_seconds']
 if terminal_cpu is None:
  numbers=[available_counter(state.get('CPUUsageNSec'))/1e9 for state in terminal_states if available_counter(state.get('CPUUsageNSec')) is not None]
  terminal_cpu=max(numbers) if numbers else None
 unit_wall=exec_wall(second['systemd']) if second['current_invocation_metadata_available'] else None
 if unit_wall is None:
  walls=[exec_wall(state) for state in terminal_states if exec_wall(state) is not None];unit_wall=max(walls) if walls else None
 service_wall=run['end_epoch']-run['start_epoch'];require(numeric(service_wall) and service_wall>=0 and math.isclose(service_wall,run['wall_seconds'],rel_tol=1e-4,abs_tol=.1),'Recorded supervisor service wall')
 wanted=set(cost_values)|{'generated_tokens','prefill_tokens','retrieval_calls','scoring_calls','scoring_wall_seconds','wall_seconds','process_cpu_seconds','HTTP_requests'}
 batch_costs={key:dict(total=math.fsum(cost_values[key]) if len(cost_values[key])==809 else None,known_sum=math.fsum(cost_values[key]),recorded_batches=len(cost_values[key]),required_batches=809) for key in sorted(wanted)}
 cleanup_ok=all(not frame['own_pids'] and frame['own_lease_holders_absent'] and frame['own_loopback18082_listener_absent'] for frame in (first,second)) and not any(state!='Z' for state in run.get('own_cleanup_residual',{}).values()) and not run.get('cleanup_error')
 return dict(schema='tree_fast_final_terminal_delta_remote_v1',status='ACTUAL_COMPLETE_FINAL51713_SCALAR_COST_CLEANUP_READBACK',started_epoch=started,observed_epoch=time.time(),unit=UNIT,original_main_pid=3409502,invocation_id=INV,evaluation_source_commit=EVAL,training_source_commit=TRAIN,identity=IDENTITY,
  sources=dict(code_manifest=code_ref,assets_manifest=assets_ref,handoff=handoff_ref,prefix=prefix_ref,run_result=run_ref,final_metrics=final_ref,final_metrics_done=done_ref,allocation=allocation_ref),run_result=run,final_metrics=final,rank_provenance=provenance,independently_recomputed_metrics=metrics,source_score_scalar_totals=score_totals,evaluated_count=51713,exact_source_counts=dict(counts),unique_continuous_index_range=[0,51712],committed_chunk_count=809,batch_proofs=proofs,rank_scored_rows=dict(rank_rows),recorded_batch_costs=batch_costs,
  per_rank_recorded_batch_cost_sums={rank:{key:math.fsum(values) for key,values in data.items()} for rank,data in rank_costs.items()},resource_summary=resource,terminal_readback_frames=[first,second],own_cleanup_observed_pass=cleanup_ok,
  auxiliary_audit_cleanup_receipt_readbacks=[first_aux,second_aux],auxiliary_audit_units_root_stop_observed=first_aux['root_stop_observed'] and second_aux['root_stop_observed'],root_preserved_unit_state_observations=pins['root_preserved_unit_states'],
  cost_scope=dict(systemd_full_exec_wall_seconds=unit_wall,supervisor_recorded_run_wall_seconds=run['wall_seconds'],supervisor_window_excludes_earlier_preflight=True,whole_unit_terminal_CPU_seconds=terminal_cpu,unit_current_metadata_GC_unavailable=not second['current_invocation_metadata_available'],root_preserved_live_CPU_asof_seconds=[available_counter(state.get('CPUUsageNSec'))/1e9 for state in preserved if state.get('MainPID')=='3409502' and available_counter(state.get('CPUUsageNSec')) is not None],batch_wall_is_overlapping_sum=True,batch_process_CPU_is_unit_subset=True,batch_costs_not_added_to_unit_CPU=True,physical_shared_gpu_capacity_upper_bound_seconds=8*unit_wall if unit_wall is not None else None,own_gpu_active_seconds=None,exclusive_billable_gpu_hours=None,gpu_energy_kwh=None,inflight_batches_at_COMPLETE=0,scoring_calls_not_inferred_from_score_rows=True,score_generated_tokens_and_search_tag_events_are_separate_from_SearchRollout_counters=True),
  original_interrupted_final_context=dict(inputs=original_context,systemd_terminal=original_state,systemd_terminal_local_asof_proof=pins['original_systemd_local_proof'],original_unit_requeried=False,whole778_segment_CPU_seconds=(available_counter(original_state.get('CPUUsageNSec'))/1e9 if available_counter(original_state.get('CPUUsageNSec')) is not None else None),original_final_only_service_CPU_seconds=None,original_final_only_service_wall_seconds=None,direct_recorded_final_validation_control=dict(source=original_context['partial_final_control']['source'],status=original_context['partial_final_control']['value'].get('status'),evaluated_rows=original_context['partial_final_control']['value'].get('evaluated_rows'),wall_seconds=original_context['partial_final_control']['value'].get('wall_seconds'),counters=original_context['partial_final_control']['value'].get('counters')),scope='User-authorized final-only stop after actual training200; FAILED is preserved, never relabelled a training200 failure. Full778 segment CPU includes training/dev/partialfinal and is not added again. Only directly recorded phase-final control counters can isolate final subset.',human_authority_local_proof=pins['human_authority_local_proof']),
  observer_process_cpu_seconds=time.process_time()-cpu,observer_wall_seconds=time.time()-started,observer_overhead_added_to_evaluator=False,raw_score_rows_transferred=False,raw_resource_stream_transferred=False,model_PT_HF_or_dataset_loaded_or_hashed=False,cleanup_actions_performed=False,other_tasks_operated=False,goal_complete=False)
