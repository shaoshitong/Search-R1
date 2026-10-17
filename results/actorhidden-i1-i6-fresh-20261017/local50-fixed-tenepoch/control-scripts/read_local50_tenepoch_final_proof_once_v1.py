from pathlib import Path
import datetime,json,hashlib
from reproduction.h1008_qprobe.established_controller_transport import remote
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');state=json.loads((root/'work/local-probe-fit50-tenepoch-state.json').read_text());tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
inner=r'''
import pathlib,json,hashlib,os,sys,subprocess,glob,time
b=pathlib.Path(BASE);job=JOB;sys.path.insert(0,str(b/'repo'))
from reproduction.common8.native_allocation import fields
from reproduction.prefix_only_correctness_ab.launch import birth_states
proof=b/'preparation/local50-tenepoch-final-independent-cpu-original-records-once-v1.json'
if proof.exists():
 print(json.dumps(dict(existing=True,proof_path=str(proof),proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),proof=json.loads(proof.read_text()))));raise SystemExit
read=lambda p:json.loads(p.read_text())
r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
native=fields(r.stdout);assert native['JobId']==job and native['JobState']=='COMPLETED' and native['ExitCode']=='0:0',native
assert read(b/'preparation/submit-receipt.json')['job_id']==job
assert read(b/'logs/pipeline-complete.json')['train_complete']['actual_adam_per_rank']==[1200]*8
ledger=read(b/'logs/process-ledger.json');births=[v for j in ledger['jobs'].values()for v in j['ledger'].values()]
checks=[]
for attempt in range(2):
 present=birth_states(births);groups=glob.glob('/sys/fs/cgroup/**/job_'+job+'*',recursive=True)
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20);assert gpu.returncode==0
 owned=sorted(set(gpu.stdout.split()).intersection(str(v['pid'])for v in births))
 assert not present and not groups and not owned,(present,groups,owned)
 checks.append(dict(checked_unix=time.time(),actual_births_present=present,cgroup_paths=groups,owned_gpu_pids=owned))
 if attempt==0:time.sleep(5)
records=[]
for p in sorted((b/'logs').glob('*.log')):
 with p.open('rb')as f:
  before=os.fstat(f.fileno());h=hashlib.sha256()
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  after=os.fstat(f.fileno())
 sig=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns)
 assert sig(before)==sig(after)==sig(p.stat())
 records.append(dict(path=str(p),dev=before.st_dev,inode=before.st_ino,bytes=before.st_size,sha256=h.hexdigest()))
checkpoints=[];orders=set()
for epoch in range(1,11):
 c=read(b/'train'/('epoch-%02d-complete.json'%epoch));assert c['epoch']==epoch and c['actual_adam_per_rank']==[200+epoch*100]*8
 p=pathlib.Path(c['checkpoint']);assert p.stat().st_size==c['checkpoint_bytes']
 planpath=b/'train'/('epoch-%02d-order.json'%epoch);plan=read(planpath);ids=[qid for step in plan for qid in step['question_indices']];assert len(ids)==len(set(ids))==5120
 orders.add(hashlib.sha256(planpath.read_bytes()).hexdigest());checkpoints.append(c)
assert len(orders)==10
p=pathlib.Path(checkpoints[-1]['checkpoint']);before=p.stat();h=hashlib.sha256()
with p.open('rb')as f:
 for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
assert h.hexdigest()==checkpoints[-1]['checkpoint_sha256'];assert sig(before)==sig(p.stat())
import torch
saved=torch.load(p,map_location='cpu',weights_only=False)
assert saved['epoch']==10 and saved['actual_adam_per_rank']==[1200]*8 and saved['learning_rate']==1e-5
assert len(saved['model'])==len(saved['optimizer']['state'])==14 and len(saved['rank_rng'])==8
assert all(v.dtype==torch.float32 for v in saved['model'].values())
assert all(float(v['step'])==1200 and v['exp_avg'].dtype==v['exp_avg_sq'].dtype==torch.float32 for v in saved['optimizer']['state'].values())
assert all(v['lr']==1e-5 for v in saved['optimizer']['param_groups'])
updates=[]
for rank in range(8):
 p=b/'train'/('rank-%d-updates.jsonl'%rank);rows=[json.loads(line)for line in p.read_text().splitlines()]
 assert [v['actual_adam']for v in rows]==list(range(201,1201))
 for epoch in range(1,11):
  rows_e=[v for v in rows if v['epoch']==epoch];assert [v['step']for v in rows_e]==list(range(1,101))and sum(v['local_trajectories']for v in rows_e)==3200
 updates.append(dict(rank=rank,new_writes=len(rows),final_adam=1200,ledger_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
v=dict(status='LOCAL50_TEN_EPOCH_ACTUAL_COMPLETE_NATIVE_EXIT0_AND_FINAL_CPU_READBACK',job_id=job,native=native,
 actor_updated=False,formal200_resumed=False,release_checks=checks,original_logs=records,original_births=len(births),
 checkpoints=checkpoints,all_ten_group_orders_distinct=True,rank_updates=updates,
 final_checkpoint_independent_full_sha256=h.hexdigest(),final_model_state_tensors=14,final_adam_steps=1200,
 final_checkpoint_actual_cpu_load=True,gpu_forward_performed=False,checkpoint_other_than_final_independent_hash_performed=False,
 checked_unix=time.time())
with proof.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(dict(existing=False,proof_path=str(proof),proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),proof=v)))
'''.replace('BASE',repr(state['base'])).replace('JOB',repr(state['current_job_id']))
script='import subprocess,os\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2");r=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=env,text=True,capture_output=True,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)\n'
value=json.loads(remote(script,str(root/'work'/('local50-tenepoch-final-proof-'+tag+'.transport.json'))));out=root/'work'/('local50-tenepoch-final-proof-'+tag+'.json');out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items()if k!='proof'},indent=2));print('native',value['proof']['native']['JobState'],'finalAdam',value['proof']['final_adam_steps'],'originalLogs',len(value['proof']['original_logs']));print('smallproof',str(out),'SHA',hashlib.sha256(out.read_bytes()).hexdigest())
