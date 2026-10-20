"""Read actual same-card CP64 continuation, without touching the writer."""
import datetime,json,time,os
from pathlib import Path
from .control import remote,OUT
BASE=os.environ.get('B_SHARED_OBSERVE_BASE','/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp64-h1005-20261019-r3')
SCRIPT=r'''
import pathlib,json,time,hashlib
p=pathlib.Path(BASE)
def read(f):
 for attempt in range(12):
  try:return json.loads(f.read_text())
  except FileNotFoundError:return None
  except json.JSONDecodeError as error:
   if attempt==11:raise RuntimeError('OBSERVATION_JSON_DECODE_RETRIES_EXHAUSTED '+str(f))from error
   time.sleep(0.25)
def physical(f):
 s=str(f);prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
 return pathlib.Path('/mnt/data/shaoshitong')/s[len(prefix):]if s.startswith(prefix)else pathlib.Path(s)
def identity(f):
 st=f.stat();return dict(dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
def reusable(row,complete):
 cache=row.get('observation_cache',{})
 try:
  return cache.get('complete_identity')==identity(complete)and len(cache['ledger_identities'])==8 and all(identity(physical(r['path']))==r['identity']for r in cache['ledger_identities'])
 except(OSError,KeyError):return False
config=read(p/'preparation/launch-config.json')
execution_node=config['expected_execution_node']
if execution_node not in ('h100-5','h100-8','h100-10'):raise ValueError('Require activated allowed shared node; H10011 excluded')
resume_outer=int(config.get('resume_joint',{}).get('completed_outer',64))
stage=read(p/'logs/pipeline-status.json');n=max(resume_outer+1,(stage or{}).get('outer',resume_outer+1));root=p/'outers'/('outer-%06d'%n)
progress={str(f.relative_to(p)):read(f)for f in root.rglob('*progress*.json')if f.stat().st_size<15000}
latest_joint=read(p/'latest-joint.json')
committed_outer=(latest_joint or{}).get('outer',resume_outer)
facts=[]
for i in range(resume_outer+1,min(committed_outer,200)+1):
 complete=p/'outers'/('outer-%06d'%i)/'complete.json'
 cached=CACHED_FACTS.get(i)
 if cached and reusable(cached,complete):
  facts.append(dict(cached,reused_immutable_fact=True));continue
 j=read(complete)
 if not j:continue
 ranks=[];ledger_identities=[]
 for r in j['actor']['rank_receipts']:
  f=physical(r['ledger_path']);before=identity(f);raw=f.read_bytes();records=[json.loads(s)for s in raw.splitlines()];restore=next(x for x in records if x['event']=='ACTUAL_RESTORE')
  if identity(f)!=before:raise RuntimeError('Sealed actor ledger changed during observation: '+str(f))
  ledger_identities.append(dict(path=str(f),identity=before))
  ranks.append(dict(rank=r['rank'],restored_outer=restore['restored_completed_outer'],adam_before=restore['actual_adam_before'],scheduler_before=restore['restored_scheduler'],writes=sum(x['event']=='ACTUAL_ADAM_WRITE'for x in records),opportunities=sum(x['event']=='OPPORTUNITY'for x in records),ledger_sha_matches=hashlib.sha256(raw).hexdigest()==r['ledger_sha256']))
 facts.append(dict(outer=i,actor=j['actor']['actual_adam_steps_per_rank'],probe=j['probe']['actual_adam_steps_per_rank'],selection_metrics=j['selection_metrics'],phase_timings=j['phase_timings'],ranks=ranks,reused_immutable_fact=False,observation_cache=dict(complete_identity=identity(complete),ledger_identities=ledger_identities,actual_full_read_verified_unix=time.time())))
active=[]
for f in(p/'actor/outers').glob('outer-%04d-attempt-*/rank-*-updates.jsonl'%n):
 rows=[json.loads(s)for s in f.read_bytes().splitlines(keepends=True)if s.strip()and s.endswith(b'\n')];r=next((x for x in rows if x['event']=='ACTUAL_RESTORE'),None);w=[x for x in rows if x['event']=='ACTUAL_ADAM_WRITE']
 active.append(dict(rank=r['rank']if r else None,restore={k:r.get(k)for k in ('restored_completed_outer','actual_adam_before','restored_scheduler','actual_model_master_dtypes','actual_learning_rates','actual_global_micro')}if r else None,writes=len(w),opportunities=sum(x['event']=='OPPORTUNITY'for x in rows),last_adam=w[-1]['actual_adam_steps']if w else None))
logs=[]
for f in sorted((p/'logs').glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:4]:
 with f.open('rb')as q:q.seek(max(0,f.stat().st_size-3000));tail=q.read().decode(errors='replace')
 logs.append(dict(path=str(f.relative_to(p)),tail=tail,bytes=f.stat().st_size,mtime=f.stat().st_mtime))
print(json.dumps(dict(observed_unix=time.time(),node=execution_node,base=BASE,resume_outer=resume_outer,stage=stage,complete_outer=committed_outer,facts=facts,active_actor=active,progress=progress,receipt=read(p/'preparation/submit-receipt.json'),cleanup=read(p/'logs/cleanup-status.json'),logs=logs)))
'''

def observe():
 f=OUT/('shared-follow-'+str(time.time_ns())+'-transport.json')
 cached={}
 try:
  last=json.loads((OUT/'SHARED_RESUME64_LATEST.json').read_text())
  if last.get('base')==BASE:cached={row['outer']:row for row in last['facts']if row.get('observation_cache')}
 except(OSError,ValueError,KeyError):pass
 v=json.loads(remote('BASE='+repr(BASE)+'\nCACHED_FACTS='+repr(cached)+'\n'+SCRIPT,f,target='h100-10',established_controller=True))
 receipt=v.get('receipt');leader=receipt['leader']if receipt else None
 if leader:
  code='WHO='+repr(leader)+'\n'+r'''
import pathlib,json,subprocess,time
p=pathlib.Path('/proc')/str(WHO['pid']);st=(p/'stat').read_text()if p.exists()else None
same=bool(st and int(st.rsplit(')',1)[1].split()[19])==WHO['start_ticks']and st.rsplit(')',1)[1].split()[0]!='Z')
try:
 query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 gpu=dict(gpu=query.stdout,gpu_query_returncode=query.returncode,gpu_query_stderr=query.stderr,gpu_observation_valid=query.returncode==0)
except subprocess.TimeoutExpired:
 gpu=dict(gpu=None,gpu_query_returncode=None,gpu_query_stderr='NVML_QUERY_TIMEOUT',gpu_observation_valid=False)
print(json.dumps(dict(observed_unix=time.time(),same_birth=same,stat=st,**gpu)))
'''
  v['actual_node_health']=json.loads(remote(code,OUT/('shared-follow-health-'+str(time.time_ns())+'.json'),target=v['node'],established_controller=True))
 dest=f.with_name(f.name.replace('-transport','-actual'));dest.write_text(json.dumps(v,ensure_ascii=False,indent=2));v['evidence']=str(dest)
 (OUT/'SHARED_RESUME64_LATEST.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
 alive=bool(v.get('actual_node_health',{}).get('same_birth'))and not bool(v['cleanup'])
 observed200=v.get('complete_outer')==200
 status='SHARED_NATIVE_200_OBSERVED_AWAITING_AUDIT'if observed200 else('SHARED_NATIVE_CONTINUING'if alive else'SHARED_NATIVE_EXITED_NEEDS_DIAGNOSIS')
 if alive and not v.get('actual_node_health',{}).get('gpu_observation_valid',True):status='SHARED_NATIVE_GPU_OBSERVATION_UNKNOWN_NEEDS_CHECK'
 root=OUT.parents[1];s=json.loads((root/'outputs/STATUS.json').read_text());s.update(status=status,actual_native_running=alive,actual_training_running=alive and not observed200,actual_complete_outer=v['complete_outer'],current_stage=(v['stage']or{}).get('stage'),node=v['node'],ip={'h100-5':'10.33.0.5','h100-8':'10.33.0.8','h100-10':'10.33.0.18'}[v['node']],world_size=8,resume_job_id=None,cancelled_pending_job=3674,base=receipt['command'][-1]if receipt else None,physical_base=BASE,source_commit=receipt['source_commit']if receipt else None,native_leader=leader,training_started=bool(receipt),owned_births_all_gone=bool((v['cleanup']or{}).get('owned_births_all_gone')),execution_node_pending=False,current_evidence=str(dest),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (root/'outputs/STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 for target in (OUT,root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
  (target/'SHARED_RESUME64_STATUS.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
 print(json.dumps(dict(time=v['observed_unix'],complete=v['complete_outer'],stage=v['stage'],active=[(r['rank'],r['writes'],r['last_adam'])for r in v['active_actor']],progress={k:r.get('completed_trajectories',r.get('scored_trajectories'))for k,r in v['progress'].items()},health=v.get('actual_node_health'),cleanup=v['cleanup']),ensure_ascii=False),flush=True)
 return v

if __name__=='__main__':observe()
