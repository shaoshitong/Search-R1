"""Stop only current f847 after the reviewed warmup10 fresh run is prepared.

The default action sends at most one PID-fd SIGTERM to the pinned superseded sigma3 pipeline.
--audit-only invokes the separately published C4 read-only cleanup verifier.
Neither action launches, retries, restores, changes runtime code, or manually
removes any model, checkpoint, teacher cache, log, or other project's process.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
OLD_RUNTIME=ROOT/'qprobe-a800-sigma3-teacher-v4'
NEW_RUNTIME=ROOT/'qprobe-a800-sigma3-warmup10'
OLD_HANDLE=ROOT/'preparation/qprobe-a800-sigma3-teacher-v4-deployment.json'
HUMAN=ROOT/'preparation/qprobe-sigma3-warmup10-human-intent.json'
PREDECESSOR_SIGNAL=ROOT/'preparation/qprobe-sigma3-old1f342-stop-once.result.json'
NEW_HANDLE=ROOT/'preparation/qprobe-a800-sigma3-warmup10-deployment.json'
READY=ROOT/'preparation/qprobe-sigma3-warmup10-reviewed-prepared-ready.json'
SOURCE_PUBLICATION=ROOT/'preparation/qprobe-sigma3-warmup10-source-publication.json'
PREDECESSOR_PATH='reproduction/operations/a800_qprobe_sigma3_switch_stop.py'
PREDECESSOR_COMMIT='a521995b09d35d70dc27a6c8d0c5835e634c7e60'
PREDECESSOR_SHA='0a5734aa1358a18601f89b46f2ed7112c137467d622f6346af16660f2423adae'
OLD_SOURCE='c200f9ebcaba6fd08039f24df8e7ffc55cf19c40'
OLD_RUN='f84749ca43be42bd91f2bc63d73347dd'
OLD_EXECUTION='ffdc8bb4690d4b8793335d9a718e3dcb'
OLD_BASE='/data/shaoshitong/Search-G1-a800-fresh200-86cf721183014e729036924e8978a6b9'
OLD_PID,OLD_TICKS=4153158,613067849
OLD_TRAINER_PID,OLD_TRAINER_TICKS=4159844,613114005
SIGMA3_LOSS=dict(loss_mode='sigma3_std_aux6_v1',mean_loss_weight=0.0,std_loss_weight=3.0,count_aux_loss_weight=0.1)
REVIEW_CHECKS=('official_model_and_canonical_data','frozen_teacher_5120_x_5',
    'bootstrap_all_5120_groups','exploration_0_top512','formal_200_not_201',
    'independent_trajectory_phase_v1','four_rank_bf16_local4_global16_accum8_ref8',
    'recoverable_guards_lr_warmup57','fresh_official_actor_reference_optimizer_scheduler_rng_sampler_head',
    'sigma3_ce_normalized_sample_std_mse3_aux6_ce_point1','teacher_namespace_full_identity_and_sha_no_old_question_head',
    'bootstrap_head_10epochs_200updates_training_fit_only',
    'formal_head_twoepochs_batch256_qualified_teacher_v4_fresh_state')
LIBRARY=ROOT/'actor-probes-ten'
LIBRARY_SOURCE='c4d156de124de7a94652d75858bcbabc8c56083c'
REMOTE_LIBRARY='/data/shaoshitong/Search-G1-a800-fresh200-6047f7f9674b49fdad718a256ab3db95/repo'
STOP_INTENT='preparation/human-warmup10-switch-stop-intent.json'
STOP_SIGNAL='logs/human-warmup10-switch-stop-signal.json'
BIRTH_KEYS=('pid','start_ticks','sid','uid')

SOURCE_CHECK=r'''
def sealed(path,limit=16*1024**2):
 path=Path(path);assert path==path.resolve() and not path.is_symlink()
 fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
 try:
  before=os.fstat(fd);assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size<=limit
  with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(before.st_size+1)
  after=os.fstat(fd)
  assert len(raw)==before.st_size and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert path.stat().st_ino==before.st_ino
 finally:os.close(fd)
 return raw
def document(path,limit=16*1024**2):return json.loads(sealed(path,limit))
def verify_source(repo,commit,mapping,count=None,allow_hydra=False):
 repo=Path(repo);assert repo==repo.resolve() and repo.is_dir() and repo.stat().st_uid==os.getuid()
 assert re.fullmatch('[0-9a-f]{40}',commit)
 def git(*args,**kw):return subprocess.check_output(['git','-C',str(repo),*args],timeout=30,**kw)
 assert git('rev-parse','HEAD').decode().strip()==commit
 assert not git('status','--porcelain').strip()
 def source(relative):
  p=Path(relative)
  assert not p.is_absolute() and '..' not in p.parts and not any(c in relative for c in '\r\n\0')
  return (p.suffix in ('.py','.sh','.yaml','.yml','.json') or relative in ('verl/version/version','verl/single_controller/version/version')) and not {'.git','__pycache__','evidence'}.intersection(p.parts)
 names=[x.decode() for x in git('ls-tree','-rz','--name-only',commit).split(b'\0') if x]
 names=sorted(x for x in names if source(x));assert names==sorted(mapping)
 if count is not None:assert len(names)==count
 data=git('cat-file','--batch',input=''.join(commit+':'+x+'\n' for x in names).encode());offset=0
 for name in names:
  end=data.index(b'\n',offset);header=data[offset:end].split();assert len(header)==3 and header[1]==b'blob'
  size=int(header[2]);start=end+1;raw=data[start:start+size]
  assert len(raw)==size and data[start+size:start+size+1]==b'\n'
  assert hashlib.sha256(raw).hexdigest()==mapping[name]==hashlib.sha256(sealed(repo/name)).hexdigest()
  offset=start+size+1
 assert offset==len(data)
 actual=[]
 for path in repo.rglob('*'):
  if not path.is_file():continue
  relative=str(path.relative_to(repo))
  if not source(relative):continue
  if relative not in mapping:
   assert allow_hydra and re.fullmatch(r'outputs/[0-9]{4}-[0-9]{2}-[0-9]{2}/[0-9]{2}-[0-9]{2}-[0-9]{2}/\.hydra/(config|hydra|overrides)\.yaml',relative)
   sealed(path);continue
  actual.append(relative)
 assert sorted(actual)==names
 return mapping
'''
exec(SOURCE_CHECK)

REMOTE_COMMON=r'''
import hashlib,json,os,re,socket,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
p=json.load(sys.stdin);intent=p['old_intent'];base=Path(intent['base']);repo=base/'repo'
'''+SOURCE_CHECK+r'''
assert sys.executable=='/data/search-g1/env/bin/python'
assert str(base)==p['old_base'] and base==base.resolve() and base.stat().st_uid==os.getuid()
assert intent['source_commit']==p['old_source'] and intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800'
assert document(base/'preparation/qprobe-intent.json')==intent
run=document(base/'logs/run-manifest.json');launch=document(base/'logs/deployment-launch-receipt.json')
claim=document(base/'logs/pipeline-launch-claim.json');allocation=document(base/'logs/native-allocation-admission.json')
assert run['run_id']==launch['run_id']==claim['run_id']==allocation['run_id']==p['old_run']
assert run['execution_id']==launch['execution_id']==claim['execution_id']==allocation['execution_id']==p['old_execution']
assert run['source_commit']==launch['source_commit']==p['old_source'] and run['source_sha256']==intent['source_sha256']
assert run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None and run['execution_host']==socket.gethostname()
verify_source(repo,p['old_source'],run['source_sha256'],481,allow_hydra=True)
training=run['training_physical_indices'];auxiliary=run['auxiliary_physical_indices'];assigned=training+auxiliary
assert run['training_world_size']==4 and len(training)==len(set(training))==4 and all(type(i) is int for i in assigned)
assert auxiliary==run['retriever_physical_indices']==[5,6] and run['probe_physical_index']==6 and len(set(assigned))==6
physical={int(row[0]):row[1].strip() for row in run['physical_gpu_inventory']}
assert len(physical)==len(run['physical_gpu_inventory']) and len(set(physical.values()))==len(physical)
assert set(assigned)<=set(physical) and run['excluded_physical_indices']==sorted(set(physical)-set(assigned))
assert run['assigned_gpu_uuids']=={str(i):physical[i] for i in assigned} and run['training_gpu_uuids']==[physical[i] for i in training]
assert run['gpu_placement_bound'] is True and run['gpu_placement_binding_receipt']=='logs/local-gpu-placement-binding.json'
binding_path=base/run['gpu_placement_binding_receipt'];binding=document(binding_path)
assert hashlib.sha256(sealed(binding_path)).hexdigest()==run['gpu_placement_binding_sha256']
for key in ('run_id','execution_id','source_commit','training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','training_gpu_uuids','assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_bound_unix'):assert binding[key]==run[key]
for key in ('training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','assigned_gpu_uuids','gpu_placement_binding_sha256'):
 assert launch[key]==claim[key]==allocation[key]==run[key]
assert allocation['accepted'] is True and allocation['actual_allocated_indices']==assigned and allocation['actual_cuda_runtime_uuids']==run['training_gpu_uuids']
keys=('pid','start_ticks','sid','uid');producer=run['pipeline_identity']
assert producer==claim['pipeline_identity']==allocation['pipeline_identity']
assert tuple(producer[k] for k in keys)==(p['old_pid'],p['old_ticks'],p['old_pid'],os.getuid())
assert tuple(launch[k] for k in ('pid','start_ticks','sid'))==(p['old_pid'],p['old_ticks'],p['old_pid'])
ledger=document(base/'logs/process-ledger.json',256*1024**2);assert ledger['run_id']==run['run_id'] and ledger['jobs']
trainer=ledger['jobs']['trainer'];leader=trainer['leader']
assert tuple(leader[k] for k in keys)==(p['old_trainer_pid'],p['old_trainer_ticks'],p['old_trainer_pid'],os.getuid())
assert trainer['arguments']==['bash','reproduction/a800_common/train.sh']
jobs={}
for name,job in ledger['jobs'].items():
 assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,64}',name)
 leader=job['leader'];records=[leader]+list(job['ledger'].values())
 assert leader['pid']==leader['sid'] and leader['uid']==os.getuid()
 assert all(type(row[k]) is int for row in records for k in keys)
 assert all(row['pid']>0 and row['start_ticks']>0 and (row['sid'],row['uid'])==(leader['sid'],leader['uid']) for row in records)
 births=sorted({tuple(row[k] for k in keys) for row in records})
 jobs[name]=dict(leader={k:leader[k] for k in keys},births=[dict(zip(keys,row)) for row in births])
inventory=dict(identity={k:run[k] for k in ('run_id','execution_id','source_commit','execution_backend','training_world_size','profile','slurm_job_id')},jobs=jobs)
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
'''

REMOTE_STOP=REMOTE_COMMON+r'''
import signal
# Refuse unsupported PID-fd APIs before creating the remote intent or signalling.
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
new=p['new_intent'];newbase=Path(new['base']);assert newbase!=base and newbase==newbase.resolve() and newbase.stat().st_uid==os.getuid()
assert document(newbase/'preparation/qprobe-intent.json')==new
newrun=document(newbase/'logs/run-manifest.json');prepared=document(newbase/'logs/qprobe-prepared.json');bound=document(newbase/'logs/qprobe-bound.json')
assert prepared==p['prepared_observation'] and bound==p['bound_observation']
assert newrun['source_commit']==new['source_commit'] and newrun['source_sha256']==new['source_sha256']
assert (newrun['run_id'],newrun['execution_id'])==(p['new_run'],p['new_execution']) and newrun['run_id']!=run['run_id'] and newrun['execution_id']!=run['execution_id']
assert prepared['source_commit']==bound['source_commit']==new['source_commit'] and prepared['run_id']==bound['run_id']==newrun['run_id']
assert prepared['execution_id']==newrun['execution_id'] and prepared['stage']=='CPU_PREPARED_NO_GPU_OR_PIPELINE_LAUNCH' and bound['stage']=='QPROBE_BOUND_PRELAUNCH'
assert prepared['trainer_started'] is False and bound['trainer_started'] is False
assert newrun['execution_backend']=='ssh_local' and newrun['slurm_job_id'] is None and newrun['execution_host']==socket.gethostname()
assert newrun['training_world_size']==4 and newrun['completed_updates']==0 and newrun['target_updates']==200 and newrun['actor_micro_batch_per_rank']==4
budget=newrun['budget'];assert budget['world_size']==4 and budget['per_rank_actor_micro_batch']==4 and budget['ppo_micro_batch_size']==16 and budget['ppo_mini_batch_size']==128
assert (budget['ppo_mini_batch_size']//newrun['training_world_size'])//newrun['actor_micro_batch_per_rank']==8
assert budget['per_rank_log_prob_micro_batch']==8 and budget['lr_warmup_steps']==57
assert newrun['initial_probe_version'] is None and newrun.get('full_state_restore') is None and newrun['gpu_placement_bound'] is False
assert new.get('saved_checkpoint_restore') is None and new.get('fresh_question_probe_loss')==p['sigma3_loss']
assert type(new.get('fresh_bootstrap_head_epochs')) is int and new['fresh_bootstrap_head_epochs']==10
for name in ('actor_checkpoint','reference_checkpoint','resume_path','inherited_policy_run_id','inherited_optimizer_run_id','inherited_scheduler_run_id','full_state_restore'):
 assert newrun.get(name) is None
assert newrun['model_revision']==run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
assert newrun['initial_model']==newrun['reference_model']==run['initial_model']==run['reference_model']
assert newrun['initial_model_sha256']==run['initial_model_sha256'] and newrun['input_sha256']==run['input_sha256']
q=newrun['question_probe'];assert (q['candidate_questions'],q['selected_questions'],q['responses_per_question'],q['exploration_questions'],q['bootstrap_max_groups_per_outer'])==(5120,512,5,0,5120)
assert {name:q.get(name) for name in p['sigma3_loss']}==p['sigma3_loss']
assert type(q.get('bootstrap_head_epochs')) is int and q['bootstrap_head_epochs']==10 and q['head_epochs_per_outer']==2 and q['head_batch_size']==256
assert (q['bootstrap_max_groups_per_outer']//q['head_batch_size'])*q['bootstrap_head_epochs']==200
assert all(type(q[name]) in (int,float) for name in p['sigma3_loss'] if name!='loss_mode')
assert all(q.get(name) is None for name in ('initial_head_checkpoint','snapshot_path','restore_state'))
assert q['teacher_warmup_cache']==new['teacher_warmup_cache']==run['question_probe']['teacher_warmup_cache']=='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
assert bound['question_probe']==q and prepared['inherited_completed_outer']==bound['completed_training_outers']==0
assert newrun['infra_rollout']==run['infra_rollout'] and newrun['mixed_first_selection'] is False
for name in ('pipeline-launch-claim.json','training-launch-claim.json','deployment-launch-receipt.json','native-allocation-admission.json','process-ledger.json','actual-updates.jsonl','actual-attempts.jsonl','resolved-training-config.json','policy-safety-calibration.json'):
 assert not (newbase/'logs'/name).exists() and not (newbase/'logs'/name).is_symlink()
verify_source(newbase/'repo',new['source_commit'],new['source_sha256'])
def counters(relative,rank=None):
 path=base/relative
 if not path.exists():return dict(exists=False,complete_rows=0,last_complete_outer=None)
 raw=sealed(path,32*1024**2);end=raw.rfind(b'\n')+1;rows=[];processed=written=0
 for line in raw[:end].splitlines():
  row=json.loads(line);assert row['run_id']==run['run_id'] and row.get('execution_id',run['execution_id'])==run['execution_id']
  if rank is None:
   assert type(row['step']) is int and 1<=row['step']<=200
   rows.append({k:row[k] for k in ('step','opportunities_processed','actual_optimizer_writes')})
  else:
   assert row['rank']==rank
   if row.get('kind')=='OPPORTUNITY' and row.get('processed') is True:processed+=1;written+=int(row.get('optimizer_write_succeeded') is True)
 result=dict(exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),partial_tail_bytes=len(raw)-end)
 if rank is None:result.update(complete_rows=len(rows),last_complete_outer=rows[-1]['step'] if rows else None,completed_update_rows=rows)
 else:result.update(rank=rank,processed=processed,returned_adam_writes=written,skips=processed-written)
 return result
updates=counters('logs/actual-updates.jsonl');rank_counts=[counters('logs/optimizer-rank-%d.jsonl'%rank,rank) for rank in range(4)]
assert updates['exists'] is True and type(updates['last_complete_outer']) is int and updates['last_complete_outer']>=3
attempt_path=base/'logs/actual-attempts.jsonl'
attempts=dict(exists=False)
if attempt_path.exists():
 raw=sealed(attempt_path,32*1024**2);end=raw.rfind(b'\n')+1;attempt_rows=[json.loads(line) for line in raw[:end].splitlines()]
 assert all(row['run_id']==run['run_id'] and row.get('execution_id',run['execution_id'])==run['execution_id'] for row in attempt_rows)
 attempts=dict(exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),partial_tail_bytes=len(raw)-end,
  complete_rows=len(attempt_rows),last_attempt={k:attempt_rows[-1].get(k) for k in ('attempt_index','logical_update','status','opportunities_processed','optimizer_writes_this_attempt','retained_partial_adam_writes')} if attempt_rows else None)
def file_metadata(path):
 assert path==path.resolve() and not path.is_symlink()
 info=path.stat();assert info.st_uid==os.getuid() and stat.S_ISREG(info.st_mode)
 return dict(path=str(path),bytes=info.st_size,inode=info.st_ino,modified_ns=info.st_mtime_ns)
logs=[file_metadata(base/'logs'/name) for name in ('search-g1-common8-training.log','deployment-pipeline.log') if (base/'logs'/name).exists()]
checkpoints=[]
for step in (5,10,50,100,150,200):
 prefix=base/'checkpoints/search-g1-common8/actor'/('global_step_'+str(step))
 if not prefix.exists():continue
 assert prefix==prefix.resolve() and prefix.is_dir()
 saved=[]
 for relative in ('complete.json','training_state/manifest.json','training_state/complete.json'):
  path=prefix/relative
  if path.exists():saved.append(dict(file_metadata(path),sha256=hashlib.sha256(sealed(path)).hexdigest()))
 checkpoints.append(dict(step=step,path=str(prefix),json_metadata=saved,tensor_bytes_read=False))
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import identity,same,signal_owned
signal_source=repo/'reproduction/a800/owned_processes.py'
assert hashlib.sha256(sealed(signal_source)).hexdigest()=='4168d7c93b4268fa8b2abb8b78efa1cddbd55a47602eca607642e10d04f0dae2'
live=same(producer);state=None
if live:
 actual=identity(producer['pid']);state=actual['state']
 assert 'reproduction.a800_common.run_pipeline' in actual['command'] and str(base) in actual['command']
def once(path,value):
 raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
assert not (base/p['stop_intent']).exists() and not (base/p['stop_signal']).exists()
snapshot=dict(role='human_warmup10_reviewed_prepared_switch_stop_intent',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 pipeline_identity={k:producer[k] for k in keys},preserved_births=inventory,old_counts=updates,old_rank_counts=rank_counts,
 old_attempts=attempts,old_full_state_restore_metadata=run.get('full_state_restore'),preserved_checkpoints=checkpoints,
 old_source_map_sha256=p['old_source_map_sha256'],old_run_manifest_sha256=hashlib.sha256(sealed(base/'logs/run-manifest.json')).hexdigest(),old_log_metadata=logs,
 new_run_id=p['new_run'],new_execution_id=p['new_execution'],new_source_commit=new['source_commit'],new_base=str(newbase),
 human_intent_sha256=p['human_intent_sha256'],human_recorded_unix=p['human_recorded_unix'],human_scope_sha256=p['human_scope_sha256'],
 reviewed_prepared_evidence_sha256=p['evidence_sha256'],new_loss=p['sigma3_loss'],actor_local_micro=4,actor_global_micro=16,gradient_accumulation=8,
 successful_predecessor_pidfd_signal_evidence_sha256=p['predecessor_signal_sha256'],
 old_source_log_checkpoint_teacher_cache_preserved=True,no_manual_deletion=True,private_corpus_cleanup_delegated_to_pipeline=True,
 bootstrap_head_epochs=10,expected_bootstrap_head_updates=200,formal_head_epochs_per_outer=2,
 scope='one verified old pipeline only; its native cleanup handles its owned components; no counters inherited by the fresh warmup10 run',checked_unix=time.time())
once(base/p['stop_intent'],snapshot)
sent=False
if live and state not in ('Z','X'):signal_owned(producer,signal.SIGTERM);sent=True
receipt=dict(role='human_warmup10_switch_pipeline_signal_result',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 pipeline_identity={k:producer[k] for k in keys},signal_sent=sent,signal='SIGTERM' if sent else None,state_before=state,
 stop_intent_sha256=hashlib.sha256(sealed(base/p['stop_intent'])).hexdigest(),old_counts=updates,old_rank_counts=rank_counts,
 old_attempts=attempts,pidfd_signal_source_sha256=hashlib.sha256(sealed(signal_source)).hexdigest(),human_scope_sha256=p['human_scope_sha256'],
 cleanup_accepted=False,cleanup_requires_actual_read_only_audit=True,no_automatic_retry=True,checked_unix=time.time())
once(base/p['stop_signal'],receipt)
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
print('WARMUP10_SWITCH_JSON:'+json.dumps(receipt,sort_keys=True,allow_nan=False),flush=True)
'''

REMOTE_AUDIT=REMOTE_COMMON+r'''
snapshot=document(base/p['stop_intent']);signal_receipt=document(base/p['stop_signal'])
for row in (snapshot,signal_receipt):
 assert (row['run_id'],row['execution_id'],row['source_commit'])==(run['run_id'],run['execution_id'],run['source_commit'])
 assert row['pipeline_identity']=={k:producer[k] for k in keys}
assert hashlib.sha256(sealed(base/p['stop_intent'])).hexdigest()==signal_receipt['stop_intent_sha256']
assert snapshot['human_scope_sha256']==signal_receipt['human_scope_sha256']==p['human_scope_sha256']
library=Path(p['library_repo']);verify_source(library,p['library_source'],p['library_source_sha256'],455,allow_hydra=True)
sys.path.insert(0,str(library))
from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup,birth_inventory
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
current=birth_inventory(base,run)
actual=verify_cleanup(base,run,preserved_births=snapshot['preserved_births'])
assert actual['accepted'] is True and actual['actual_pipeline_and_owned_sessions_exited'] is True and actual['owned_cuda_contexts_remaining']==0
assert actual['private_cache_removed'] is True and actual['original_assets_present'] is True
imported={}
for name,module in list(sys.modules.items()):
 path=getattr(module,'__file__',None)
 if not path:continue
 path=Path(path).resolve()
 if library not in path.parents:continue
 relative=str(path.relative_to(library));assert relative in p['library_source_sha256']
 digest=hashlib.sha256(sealed(path)).hexdigest();assert digest==p['library_source_sha256'][relative];imported[relative]=digest
assert 'reproduction/common8/actor_probe_owned_cleanup.py' in imported
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
receipt=dict(role='human_warmup10_switch_actual_owned_cleanup_audit',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 accepted=True,no_stop_or_signal=True,no_remote_mutation=True,old_counts=snapshot['old_counts'],old_rank_counts=snapshot['old_rank_counts'],
 stop_intent_sha256=signal_receipt['stop_intent_sha256'],auditor_source_commit=p['library_source'],imported_module_sha256=imported,
 verification={k:actual[k] for k in ('actual_pipeline_and_owned_sessions_exited','owned_cuda_contexts_remaining','private_cache_removed','original_assets_present','reused_pids_belonging_to_other_births','reused_session_ids','recorded_births')},
 files=actual['files'],checked_unix=actual['checked_unix'])
print('WARMUP10_SWITCH_JSON:'+json.dumps(receipt,sort_keys=True,allow_nan=False),flush=True)
'''


def sha(path):return hashlib.sha256(sealed(path)).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def once(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def evidence(path):
    path=Path(path);path.relative_to(ROOT/'preparation')
    return document(path),sha(path)


def authority_and_human():
    instruction=document(HUMAN)
    assert instruction['role']=='latest_human_warmup10_rerun_override'
    assert instruction['new_warmup_head_epochs']==10
    assert instruction['formal_head_epochs_per_outer_unchanged']==2 and instruction['head_batch_size']==256
    assert instruction['expected_warmup_head_optimizer_updates']==200 and instruction['warmup_training_groups']==5120
    assert instruction['teacher_trajectories_reuse_qualified_v4_only'] is True
    assert instruction['new_head_fresh_no_inherited_old_head'] is True
    assert instruction['target_formal_complete_outers']==200 and instruction['formal_recipe_loss_selection_data_model_unchanged'] is True
    assert instruction['new_worktree']==str(NEW_RUNTIME) and instruction['source_base']==OLD_SOURCE
    assert (instruction['old_run_to_stop_after_replacement_prepared'],instruction['old_execution_to_stop_after_replacement_prepared'])==(OLD_RUN,OLD_EXECUTION)
    scope=instruction['scope']
    assert scope=='Prepare/publish isolated replacement before only-owned current-run switch; no live source hotedit, teacher trajectory regeneration, duplicate trainer or foreign stop. Latest questions do not cancel rerun10.'
    assert type(instruction['raw_instructions']) is list and '重新跑warmup，跑10epoch' in instruction['raw_instructions']
    assert instruction['missing_old40step_history_not_reconstructed'] is True
    assert instruction['routine_reviews_disabled'] is True
    assert type(instruction['recorded_unix']) in (int,float) and 0<instruction['recorded_unix']<=time.time()
    return sha(HUMAN),canonical_sha(dict(recorded_unix=instruction['recorded_unix'],raw_instructions=instruction['raw_instructions'],
        old_run=OLD_RUN,old_execution=OLD_EXECUTION,starting_source_commit=OLD_SOURCE,scope=scope,new_loss=SIGMA3_LOSS,
        bootstrap_head_epochs=10,formal_head_epochs_per_outer=2,expected_bootstrap_updates=200)),instruction['recorded_unix']


def successful_pidfd_predecessor():
    previous=document(PREDECESSOR_SIGNAL)
    assert previous['role']=='sigma3_switch_one_pipeline_stop_claim' and previous['exit_code']==0
    assert previous['status']=='STOP_REQUEST_RETURNED' and previous['no_automatic_retry'] is True
    assert previous['source_commit']=='91f673c58ff415115c581b36568801b39ae3db86'
    result=previous['results'];assert len(result)==1 and result[0]['signal_sent'] is True and result[0]['signal']=='SIGTERM'
    assert result[0]['run_id']=='1f34290a70454871aea9d368dc8a0b36' and result[0]['execution_id']=='59ef9209e5724cf6adc6b126d30071c8'
    expected='4168d7c93b4268fa8b2abb8b78efa1cddbd55a47602eca607642e10d04f0dae2'
    assert result[0]['pidfd_signal_source_sha256']==expected==sha(OLD_RUNTIME/'reproduction/a800/owned_processes.py')
    prior=document(ROOT/'preparation/qprobe-a800-async-fullouter-restore5-deployment.json')
    assert prior['source_commit']==previous['source_commit'] and prior['source_sha256']['reproduction/a800/owned_processes.py']==expected
    return sha(PREDECESSOR_SIGNAL)


def approved_review(review,intent,operations_commit,observer):
    assert review['role']=='a800_qprobe_independent_code_review' and review['approved'] is True and review['independent'] is True
    assert review['reviewer'] and review['implementation_actor'] and review['reviewer']!=review['implementation_actor']
    assert review.get('blocking_findings',[])==[] and review['parent_context_inherited'] is False
    assert review['source_commit']==intent['source_commit'] and review['source_sha256']==intent['source_sha256']
    assert set(review['checks'])==set(REVIEW_CHECKS) and all(value is True for value in review['checks'].values())
    assert review['review_ops_commit']==operations_commit
    expected={str(Path(__file__).resolve().relative_to(OPS)):sha(Path(__file__)),
              str(Path(observer['path']).relative_to(OPS)):observer['sha256']}
    assert len(expected)==2 and review['reviewed_operations']==expected
    assert type(review['reviewed_unix']) in (int,float) and 0<review['reviewed_unix']<=time.time()


def ready_warmup10(args,operator_commit):
    assert args.new_handle==NEW_HANDLE and args.ready_receipt==READY
    ready,ready_sha=evidence(args.ready_receipt)
    assert ready['role']=='a800_qprobe_warmup10_reviewed_prepared_switch_ready'
    records=ready['files'];assert set(records)=={'new_handle','new_review','source_publication','prepare_receipt','bind_receipt','operations_publication'}
    opened={};hashes={'ready_receipt':ready_sha}
    for name,record in records.items():
        assert set(record)=={'path','sha256'}
        opened[name],actual=evidence(Path(record['path']));assert actual==record['sha256'];hashes[name]=actual
    assert records['new_handle']['path']==str(args.new_handle) and records['new_review']['path']==str(args.new_review)
    assert records['source_publication']['path']==str(SOURCE_PUBLICATION)
    intent=opened['new_handle'];assert intent==document(args.new_handle)
    assert intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800' and intent['source_commit']!=OLD_SOURCE
    assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id']) and intent['base']=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id'] and intent['base']!=OLD_BASE
    assert intent.get('saved_checkpoint_restore') is None and intent['fresh_question_probe_loss']==SIGMA3_LOSS
    assert type(intent.get('fresh_bootstrap_head_epochs')) is int and intent['fresh_bootstrap_head_epochs']==10
    assert intent['initial_probes_source_base']=='/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'
    assert intent['teacher_warmup_cache']=='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
    verify_source(NEW_RUNTIME,intent['source_commit'],intent['source_sha256'])
    publication=opened['source_publication']
    assert publication['source_commit']==publication['remote_commit']==intent['source_commit']
    assert publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['published_ref']==intent['published_ref'] and publication['source_map_sha256']==canonical_sha(intent['source_sha256'])
    assert publication['runtime_file_count']==len(intent['source_sha256'])
    assert ready['operations_commit']==operator_commit
    observer=ready['observer'];assert set(observer)=={'path','sha256'}
    observer_path=Path(observer['path']);relative=str(observer_path.relative_to(OPS))
    assert relative.startswith('reproduction/operations/') and observer_path.suffix=='.py'
    assert observer['sha256']==sha(observer_path)
    assert subprocess.check_output(['git','-C',str(OPS),'show',operator_commit+':'+relative],timeout=10)==sealed(observer_path)
    approved_review(opened['new_review'],intent,operator_commit,observer)
    operations=opened['operations_publication']
    assert operations['source_commit']==operations['remote_commit']==operator_commit and operations['remote_fresh_object_store_disk_git_hash_equal'] is True
    for relative,digest in opened['new_review']['reviewed_operations'].items():
        actual=[row for row in operations['files'] if row['path']==relative]
        assert len(actual)==1 and actual[0]['sha256']==digest
    prepared,bound=opened['prepare_receipt'],opened['bind_receipt']
    assert prepared['action']=='prepare' and bound['action']=='bind' and prepared['exit_code']==bound['exit_code']==0
    assert prepared['deployment_id']==bound['deployment_id']==intent['deployment_id']
    assert len(prepared['observations'])==len(bound['observations'])==1
    first,last=prepared['observations'][0],bound['observations'][0]
    assert first['stage']=='CPU_PREPARED_NO_GPU_OR_PIPELINE_LAUNCH' and last['stage']=='QPROBE_BOUND_PRELAUNCH'
    assert first['source_commit']==last['source_commit']==intent['source_commit'] and first['run_id']==last['run_id'] and first['run_id']!=OLD_RUN
    assert re.fullmatch('[0-9a-f]{32}',first['run_id']) and re.fullmatch('[0-9a-f]{32}',first['execution_id']) and first['execution_id']!=OLD_EXECUTION
    assert first['base']==intent['base'] and first['trainer_started'] is False and last['trainer_started'] is False
    assert first['inherited_completed_outer']==last['completed_training_outers']==0 and last['target_updates']==200 and last['extra_frozen_teacher_warmup_outers']==1
    q=last['question_probe'];assert (q['candidate_questions'],q['selected_questions'],q['responses_per_question'],q['exploration_questions'],q['bootstrap_max_groups_per_outer'])==(5120,512,5,0,5120)
    assert {name:q.get(name) for name in SIGMA3_LOSS}==SIGMA3_LOSS
    assert type(q.get('bootstrap_head_epochs')) is int and q['bootstrap_head_epochs']==10 and q['head_epochs_per_outer']==2 and q['head_batch_size']==256
    assert (q['bootstrap_max_groups_per_outer']//q['head_batch_size'])*q['bootstrap_head_epochs']==200
    assert all(type(q[name]) in (int,float) for name in SIGMA3_LOSS if name!='loss_mode')
    assert q['teacher_warmup_cache']==intent['teacher_warmup_cache']
    assert all(q.get(name) is None for name in ('initial_head_checkpoint','snapshot_path','restore_state'))
    assert (ready['new_run_id'],ready['new_execution_id'],ready['new_base'],ready['new_source_commit'],ready['new_source_map_sha256'])==(
        first['run_id'],first['execution_id'],intent['base'],intent['source_commit'],canonical_sha(intent['source_sha256']))
    return intent,first,last,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('new-handle','new-review','ready-receipt'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    operator_commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=10).strip()
    relative=str(Path(__file__).resolve().relative_to(OPS))
    assert subprocess.check_output(['git','-C',str(OPS),'show',operator_commit+':'+relative],timeout=10)==sealed(Path(__file__)), 'Publish this exact switch operator first'
    predecessor=OPS/PREDECESSOR_PATH
    assert sha(predecessor)==PREDECESSOR_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',PREDECESSOR_COMMIT+':'+PREDECESSOR_PATH],timeout=10)==sealed(predecessor)
    assert subprocess.check_output(['git','-C',str(OPS),'show',operator_commit+':'+PREDECESSOR_PATH],timeout=10)==sealed(predecessor)
    human_sha,scope_sha,recorded=authority_and_human()
    predecessor_sha=successful_pidfd_predecessor()
    new,prepared,bound,hashes=ready_warmup10(args,operator_commit)
    new_run,new_execution=prepared['run_id'],prepared['execution_id']
    old=document(OLD_HANDLE);assert old['source_commit']==OLD_SOURCE and old['base']==OLD_BASE
    verify_source(OLD_RUNTIME,OLD_SOURCE,old['source_sha256'],481)
    old_pub=document(ROOT/'preparation/qprobe-sigma3-teacher-v4-source-publication.json')
    assert old_pub['source_commit']==old_pub['remote_commit']==OLD_SOURCE and old_pub['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert old_pub['source_map_sha256']==canonical_sha(old['source_sha256'])
    payload=dict(old_intent=old,old_source=OLD_SOURCE,old_run=OLD_RUN,old_execution=OLD_EXECUTION,old_base=OLD_BASE,old_pid=OLD_PID,old_ticks=OLD_TICKS,
        old_trainer_pid=OLD_TRAINER_PID,old_trainer_ticks=OLD_TRAINER_TICKS,old_source_map_sha256=canonical_sha(old['source_sha256']),
        new_intent=new,new_run=new_run,new_execution=new_execution,prepared_observation=prepared,bound_observation=bound,
        human_intent_sha256=human_sha,human_scope_sha256=scope_sha,human_recorded_unix=recorded,sigma3_loss=SIGMA3_LOSS,
        predecessor_signal_sha256=predecessor_sha,
        evidence_sha256=hashes,stop_intent=STOP_INTENT,stop_signal=STOP_SIGNAL)
    remote=REMOTE_AUDIT if args.audit_only else REMOTE_STOP
    if args.audit_only:
        mapping=verify_source(LIBRARY,LIBRARY_SOURCE,document(ROOT/'preparation/actor-probes-ten-production-after-3ecc-preclaim-exit-deployment.json')['source_sha256'],455)
        payload.update(library_repo=REMOTE_LIBRARY,library_source=LIBRARY_SOURCE,library_source_sha256=mapping)
    ast.parse(remote)
    spec=importlib.util.spec_from_file_location('pinned_warmup10_stop_existing_route',OLD_RUNTIME/'reproduction/a800_common/deploy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert sha(Path(module.__file__))==old['source_sha256']['reproduction/a800_common/deploy.py']
    route=module.ssh_command(remote);assert route[-1].startswith('python3 -c ')
    route[-1]=shlex.quote('/data/search-g1/env/bin/python')+' -c '+shlex.quote(remote)
    assert authority_and_human()==(human_sha,scope_sha,recorded), 'Human authority changed before dispatch'
    assert ready_warmup10(args,operator_commit)[3]==hashes, 'Reviewed preparation changed before dispatch'
    stem='qprobe-warmup10-oldf847-cleanup-audit-'+str(time.time_ns()) if args.audit_only else 'qprobe-warmup10-oldf847-stop-once'
    claim=ROOT/'preparation'/(stem+'.json');output=claim.with_name(claim.stem+'.result.json')
    request=dict(role='warmup10_switch_readonly_audit_request' if args.audit_only else 'warmup10_switch_one_pipeline_stop_claim',run_id=OLD_RUN,execution_id=OLD_EXECUTION,source_commit=OLD_SOURCE,
        expected_pipeline_birth=dict(pid=OLD_PID,start_ticks=OLD_TICKS,sid=OLD_PID),expected_trainer_birth=dict(pid=OLD_TRAINER_PID,start_ticks=OLD_TRAINER_TICKS,sid=OLD_TRAINER_PID),
        new_run_id=new_run,new_execution_id=new_execution,new_source_commit=new['source_commit'],new_loss=SIGMA3_LOSS,bootstrap_head_epochs=10,expected_bootstrap_head_updates=200,formal_head_epochs_per_outer=2,
        operator_commit=operator_commit,operator_sha256=sha(Path(__file__)),remote_code_sha256=hashlib.sha256(remote.encode()).hexdigest(),human_intent_sha256=human_sha,human_scope_sha256=scope_sha,
        successful_predecessor_pidfd_signal_evidence_sha256=predecessor_sha,
        reviewed_prepared_evidence_sha256=hashes,read_only=args.audit_only,no_automatic_retry=True,claimed_unix=time.time())
    once(claim,request)
    with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
        try:
            result=subprocess.run(route,input=json.dumps(payload,sort_keys=True,allow_nan=False).encode(),stdout=stdout,stderr=stderr,timeout=60)
            status=dict(exit_code=result.returncode,status='READ_RETURNED' if args.audit_only else 'STOP_REQUEST_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            status=dict(exit_code=None,status='TRANSPORT_UNKNOWN_NO_RETRY',failure_type=type(failure).__name__)
        status.update(stdout_bytes=stdout.tell(),stderr_bytes=stderr.tell());stdout.seek(0);raw=stdout.read(256*1024+1)
        stderr.seek(0);error=stderr.read(64*1024)
        status['bounded_stderr_sha256']=hashlib.sha256(error).hexdigest()
        diagnostic=error.decode(errors='replace')
        status['remote_exception_frames']=[dict(file=Path(match.group(1)).name[:160],line=int(match.group(2)),function=match.group(3)[:96])
            for match in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',diagnostic,re.MULTILINE)][-8:]
        types=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',diagnostic,re.MULTILINE)
        status['remote_exception_type']=types[-1][:96] if types else None
    rows=[]
    if status['exit_code']==0 and len(raw)<=256*1024:
        try:
            rows=[json.loads(line.split(':',1)[1]) for line in raw.decode().splitlines() if line.startswith('WARMUP10_SWITCH_JSON:')]
            assert len(rows)==1 and (rows[0]['run_id'],rows[0]['execution_id'],rows[0]['source_commit'])==(OLD_RUN,OLD_EXECUTION,OLD_SOURCE)
            if args.audit_only:assert rows[0]['accepted'] is True and rows[0]['no_stop_or_signal'] is True
        except (AssertionError,ValueError,UnicodeDecodeError,KeyError,TypeError) as failure:
            rows=[];status.update(status='OUTPUT_REJECTED_NO_RETRY',failure_type=type(failure).__name__)
    elif len(raw)>256*1024:status['status']='OUTPUT_LIMIT_EXCEEDED_NO_RETRY'
    receipt=dict(request,**status,results=rows,observed_unix=time.time());once(output,receipt)
    print(json.dumps(dict(receipt=str(output),**status,results=rows,no_automatic_retry=True),allow_nan=False))
    return 0 if status['exit_code']==0 and len(rows)==1 else 1


if __name__=='__main__':raise SystemExit(main())
