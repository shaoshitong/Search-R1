"""One explicitly authorized remote CPU recovery proof; no GPU forward or payload copy."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

CWD = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPO = Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
SAVED = CWD/'work/prefix20-stop3408-terminal-query-20261015T205229983405Z.json'
SAVED_SHA = 'd662f88136d68efee51fbe90b90019a415171c4854b5def0fa1ab8dc83657ad9'
raw = SAVED.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SAVED_SHA
saved = json.loads(raw)
base = '/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f'
assert saved['base'] == base and saved['native']['JobId'] == '3408'
payload = dict(base=base, run_id='8a6b24dc19664cb3a3fb97101f21c54e',
               saved_native=saved['native_terminal_raw'], saved_summary_path=str(SAVED),
               saved_summary_sha256=SAVED_SHA, expected_checkpoint_sha256='3bab9f397de112097fd62bf07aae9f8c7139973bd5c19cf8310465561a871829')
body = r'''
import datetime,hashlib,json,os,pathlib,stat,subprocess,sys,time
p=PAYLOAD;b=pathlib.Path(p['base']);sys.path.insert(0,str(b/'repo'))
from reproduction.common8.native_allocation import fields
from reproduction.prefix_only_correctness_ab.launch import birth_states
from reproduction.a800.owned_processes import identity

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(q):return json.loads(q.read_text())
def once(q,value):
 with q.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(q.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def digest(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
terminal_path=b/'preparation/terminal3408-lr3e5-original-inode-full-sha.json'
proof_path=b/'preparation/checkpoint7-independent-cpu-lr3e5-once-v1.json'
if terminal_path.exists() or proof_path.exists():
 print(json.dumps(dict(status='EXISTING_ONCE_EVIDENCE_INSPECT_NO_REPLAY',terminal_exists=terminal_path.exists(),cpu_proof_exists=proof_path.exists(),terminal_sha256=digest(terminal_path)if terminal_path.exists()else None,cpu_proof_sha256=digest(proof_path)if proof_path.exists()else None)));raise SystemExit
cfg=read(b/'preparation/launch-config.json');assert cfg['run_id']==p['run_id']
assert read(b/'preparation/submit-receipt.json')['job_id']=='3408'
try:
 native=subprocess.check_output(['scontrol','show','job','3408','-o'],text=True,stderr=subprocess.PIPE,timeout=30);native_source='fresh_scontrol'
except subprocess.CalledProcessError:
 native=p['saved_native'];native_source='pinned_original_terminal_summary_after_Slurm_purge'
j=fields(native)
assert j['JobId']=='3408'and j['JobState']=='CANCELLED'and j['ExitCode']=='0:2'
assert j['Comment']=='g1-prefix20-online:'+p['run_id']
assert j['Command']==str(b/'preparation/native-launch.sh')and j['WorkDir']==str(b)
ledger=read(b/'logs/process-ledger.json');births=[r for job in ledger['jobs'].values()for r in job['ledger'].values()]
# De-duplicate identical birth records only for counts, never substitute new PIDs.
unique={tuple(r[k]for k in('pid','start_ticks','sid','uid')):r for r in births};births=list(unique.values())

def resources():
 present=birth_states(births)
 cgroups=list(pathlib.Path('/sys/fs/cgroup').glob('**/job_3408'))+list(pathlib.Path('/sys/fs/cgroup').glob('**/job3408'))
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
 own=[];foreign=[]
 bypid={str(r['pid']):r for r in births}
 for line in gpu.splitlines():
  if not line.strip():continue
  pid,uuid=[item.strip()for item in line.split(',')]
  if pid not in bypid:foreign.append(dict(pid=int(pid),uuid=uuid));continue
  record=bypid[pid]
  try:actual=identity(int(pid))
  except(FileNotFoundError,ProcessLookupError):actual=None
  if actual is None or all(actual[k]==record[k]for k in('start_ticks','sid','uid')):own.append(dict(pid=int(pid),uuid=uuid))
  else:foreign.append(dict(pid=int(pid),uuid=uuid))
 value=dict(observed_utc=utc(),original_owned_births_recorded=len(births),original_owned_births_present=present,native_cgroup_exists=bool(cgroups),owned_gpu_contexts=own,preserved_foreign_gpu_contexts=foreign)
 assert not present and not cgroups and not own,'Original resource release not complete'
 return value
before_resources=resources()
logs=set(q for q in(b/'logs').rglob('*')if q.is_file())
for root in(b/'epochs',b/'train'):
 logs.update(q for q in root.rglob('*')if q.is_file()and q.suffix in('.log','.jsonl','.json'))
# Full rollout gzip, model, optimizer and prediction tensors stay on the node.
log_proofs=[]
for q in sorted(logs):
 assert not q.is_symlink(),'Original log symlink requires explicit provenance'
 with q.open('rb')as f:
  before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode)
  h=hashlib.sha256()
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  after=os.fstat(f.fileno())
 assert signature(before)==signature(after)==signature(q.stat()),'Original log inode changed'
 log_proofs.append(dict(path=str(q),bytes=before.st_size,sha256=h.hexdigest(),dev=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns,stat_stable=True))
latest=read(b/'train/latest.json');cp=b/'train/latest.pt';assert latest['epoch']==7 and latest['actual_adam_steps_per_rank']==[700]*8
assert latest['sha256']==p['expected_checkpoint_sha256']
with cp.open('rb')as f:
 cp_before=os.fstat(f.fileno());h=hashlib.sha256()
 for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 cp_sha=h.hexdigest();assert cp_sha==p['expected_checkpoint_sha256']
 f.seek(0)
 import torch
 checkpoint=torch.load(f,map_location='cpu',weights_only=False)
 cp_after=os.fstat(f.fileno())
assert signature(cp_before)==signature(cp_after)==signature(cp.stat()),'Original checkpoint inode changed'
assert checkpoint['epoch']==7 and checkpoint['actual_adam_steps_per_rank']==[700]*8
assert checkpoint['binding_sha256']==latest['binding_sha256']
assert checkpoint['binding']['learning_rate']==3e-4
model_dtypes=sorted({str(v.dtype)for v in checkpoint['model'].values()if isinstance(v,torch.Tensor)})
assert model_dtypes==['torch.float32']
steps=[];moment_dtypes=set();optim=checkpoint['optimizer'];states=optim['state']
for state in states.values():
 step=state['step'];steps.append(int(step.item())if isinstance(step,torch.Tensor)else int(step))
 for key in('exp_avg','exp_avg_sq'):
  assert key in state;moment_dtypes.add(str(state[key].dtype))
assert len(states)==14 and set(steps)=={700} and moment_dtypes=={'torch.float32'}
assert all(group['lr']==3e-4 for group in optim['param_groups'])
rng=checkpoint['rank_rng'];assert len(rng)==8 and[r['rank']for r in rng]==list(range(8))
for r in rng:
 assert r['actual_adam_steps']==700
 assert all(key in r for key in('python','numpy','torch_cpu','torch_cuda'))
 assert isinstance(r['torch_cpu'],torch.Tensor)and isinstance(r['torch_cuda'],torch.Tensor)
 assert r['torch_cpu'].dtype==r['torch_cuda'].dtype==torch.uint8
 assert r['torch_cpu'].numel()>0 and r['torch_cuda'].numel()>0
boundary=read(b/'train/epochs/epoch-0007/complete.json');assert boundary['sha256']==cp_sha
pool=read(b/'question-pool/state.json');assert pool['next_epoch']==9 and pool['cursor']==40960
assert read(b/'epochs/epoch-0008/selection.json')['epoch']==8
assert not (b/'train/epochs/epoch-0008/complete.json').exists()
partial_updates=[]
for rank in range(8):
 q=b/'train/epochs/epoch-0008'/('rank-%d-updates.jsonl'%rank)
 partial_updates.append(q.stat().st_size if q.exists()else0)
assert partial_updates==[0]*8,'New partial head Adam must be separately accounted'
after_resources=resources()
cleanup=read(b/'logs/cleanup-status.json')
proof=dict(status='ACTUAL_CP7_FULL_SHA_CPU_STATE_PASS',observed_utc=utc(),checkpoint_path=str(cp),full_sha256=cp_sha,bytes=cp_before.st_size,dev=cp_before.st_dev,inode=cp_before.st_ino,mtime_ns=cp_before.st_mtime_ns,checkpoint_stat_stable=True,epoch=7,actual_adam_steps_per_rank=[700]*8,optimizer_step_values=sorted(set(steps)),optimizer_parameter_states=len(states),model_parameter_dtypes=model_dtypes,optimizer_moment_dtypes=sorted(moment_dtypes),optimizer_group_learning_rates=[g['lr']for g in optim['param_groups']],rank_rng=[r['rank']for r in rng],rank_rng_field_names=sorted(rng[0]),cpu_load_performed=True,gpu_forward_performed=False,semantic_reload_performed=False,owned_original_births_recorded=len(births),owned_original_births_still_present=after_resources['original_owned_births_present'],native_cgroup_exists=after_resources['native_cgroup_exists'],owned_gpu_contexts=after_resources['owned_gpu_contexts'],latest_head_receipt=latest,binding=checkpoint['binding'],pool_next_epoch=pool['next_epoch'],pool_cursor=pool['cursor'],pending_epoch=8,pending_head_update_file_bytes=partial_updates)
terminal=dict(status='ACTUAL_STOP3408_TERMINAL_ORIGINAL_LOG_FULL_SHA_PASS',observed_utc=utc(),job_id=3408,run_id=p['run_id'],base=str(b),native=native,native_source=native_source,pinned_saved_terminal_summary_path=p['saved_summary_path'],pinned_saved_terminal_summary_sha256=p['saved_summary_sha256'],cleanup=cleanup,original_cleanup_not_rewritten=True,logs=log_proofs,log_files=len(log_proofs),log_bytes=sum(v['bytes']for v in log_proofs),all_original_logs_full_sha_and_stat_stable=True,resource_observations=[before_resources,after_resources],checkpoint_sha256=cp_sha,no_original_logs_or_models_downloaded=True)
once(terminal_path,terminal);once(proof_path,proof)
print(json.dumps(dict(status='REMOTE_CPU_AND_TERMINAL_PROOFS_COMPLETE',terminal_path=str(terminal_path),terminal_sha256=digest(terminal_path),cpu_proof_path=str(proof_path),cpu_proof_sha256=digest(proof_path),checkpoint_sha256=cp_sha,checkpoint_bytes=cp_before.st_size,completed_epoch=7,actual_adam_steps_per_rank=[700]*8,optimizer_step_values=sorted(set(steps)),optimizer_parameter_states=len(states),model_parameter_dtypes=model_dtypes,optimizer_moment_dtypes=sorted(moment_dtypes),rank_rng=list(range(8)),log_files=len(log_proofs),log_bytes=sum(v['bytes']for v in log_proofs),original_owned_births_recorded=len(births),actual_owned_births_present=0,native_cgroup_exists=False,owned_gpu_contexts=[],original_cleanup_preserved=cleanup,pool_next_epoch=pool['next_epoch'],pool_cursor=pool['cursor'],pending_epoch=8,pending_head_adam_updates_observed=0,gpu_forward_performed=False,large_payload_transferred_to_Mac=False,observed_utc=utc())),flush=True)
'''.replace('PAYLOAD',repr(payload)).replace('else0','else 0')
compile(body,'<remote CPU proof>','exec')
wrapper="import os; body="+repr(body)+"; env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1'); os.execve('/dataset/shaoshitong/Search-G1-common8/env/bin/python',['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',body],env)"
sys.path.insert(0,str(REPO))
from reproduction.h1008_qprobe.management_transport import remote
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
transport=CWD/'work'/('prefix20-cp7-terminal-cpu-transport-'+stamp+'.json')
output=remote(wrapper,transport,control_path=None)
summary=json.loads(output)
summary.update(transport_receipt=str(transport),transport_receipt_sha256=hashlib.sha256(transport.read_bytes()).hexdigest(),local_adapter_path=str(Path(__file__)),local_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=CWD/'work'/('prefix20-cp7-terminal-cpu-summary-'+stamp+'.json')
with path.open('x')as f:json.dump(summary,f,indent=2,sort_keys=True)
print(json.dumps(dict(local_summary=str(path),local_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),summary=summary)),flush=True)
