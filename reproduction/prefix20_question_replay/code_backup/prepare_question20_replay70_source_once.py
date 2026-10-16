"""Freeze only the explicitly stopped source and inspect remote cached batches."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys
W=Path(__file__).parent
R=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
sys.path.insert(0,str(R))
from reproduction.h1008_qprobe.management_transport import remote
B='/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-4bc6838-lr3e5'
BODY=r'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
import torch
B=Path(BASE);sys.path.insert(0,str(B/'repo'))
from reproduction.common8.native_allocation import fields
from reproduction.prefix_only_correctness_ab.launch import birth_states
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def once(p,x):
 with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
P=B/'preparation';I=P/'question20-replay70-source-freeze-once-v1-intent.json';O=P/'question20-replay70-source-freeze-once-v1-result.json'
if I.exists():
 print(json.dumps(dict(status='EXISTING_SOURCE_FREEZE_NO_REPLAY',result=read(O)if O.exists()else None)));raise SystemExit
native_raw=subprocess.check_output(['scontrol','show','job','3409','-o'],text=True,timeout=30)
n=fields(native_raw)
assert n['JobId']=='3409' and n['JobState'] in ('CANCELLED','FAILED','COMPLETED','TIMEOUT')
assert n['Command']==str(P/'native-launch.sh') and n['NodeList']=='h100-8'
c=read(P/'launch-config.json');assert n['Comment']=='g1-prefix20-online:'+c['run_id']
ledger=read(B/'logs/process-ledger.json');births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
present=birth_states(births);assert not present,'Own original process births remain'
cg=list(Path('/sys/fs/cgroup').glob('**/job_3409'))+list(Path('/sys/fs/cgroup').glob('**/job3409'));assert not cg
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=30).split()
assert not(set(gpu)&{str(r['pid'])for r in births})
latest=read(B/'train/latest.json');ep=int(latest['epoch']);assert latest['actual_adam_steps_per_rank']==[ep*100]*8
assert latest['actual_optimizer_learning_rates']==[3e-5]
assert read(B/'train/epochs'/('epoch-%04d'%ep)/'complete.json')['sha256']==latest['sha256']
once(I,dict(job_id=3409,source_base=str(B),native=n,latest=latest,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
cp=B/'train/latest.pt';frozen=P/('question20-replay70-source-epoch%04d.pt'%ep)
os.link(cp,frozen);before=frozen.stat();actual=sha(frozen);assert actual==latest['sha256']
saved=torch.load(frozen,map_location='cpu',weights_only=False)
assert saved['epoch']==ep and saved['actual_adam_steps_per_rank']==[ep*100]*8
assert len(saved['model'])==len(saved['optimizer']['state'])==14
for v in saved['model'].values():assert v.dtype==torch.float32
for row in saved['optimizer']['state'].values():
 assert int(row['step'])==ep*100
 assert row['exp_avg'].dtype==row['exp_avg_sq'].dtype==torch.float32
assert [r['rank']for r in saved['rank_rng']]==list(range(8))
assert all(r['actual_adam_steps']==ep*100 for r in saved['rank_rng'])
assert frozen.stat()==before
old=Path(c['restore']['source_base']);sources=[dict(source_data_epoch=1,kind='cached_inputs',inputs=c['cached_inputs'],split='train')]
for i in range(2,71):
 p=(old if i<=7 else B)/'epochs'/('epoch-%04d'%i)/'complete.json'
 q=read(p);assert q['epoch']==i and q['questions']==5120 and q['trajectories']==25600
 assert all(Path(r['path']).stat().st_size==r['bytes']for r in q['shards'])
 sources.append(dict(source_data_epoch=i,kind='rollout_complete',rollout_complete=str(p),sha256=sha(p),split='train'))
cached=Path(c['cached_inputs']);scan=read(cached/'scan-complete.json')
first=json.loads((cached/'index.jsonl').open().readline())
logs=[]
for p in sorted((B/'logs').rglob('*')):
 if not p.is_file()or p.is_symlink():continue
 st=p.stat();h=sha(p);assert p.stat()==st
 logs.append(dict(path=str(p),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,sha256=h))
v=os.statvfs(B)
result=dict(status='STOPPED_SOURCE_LATEST_FULL_SHA_CPU_STATE_AND_REPLAY70_INPUTS_READY',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_base=str(B),source_job_id=3409,native_raw=native_raw,native=n,owned_births_recorded=len(births),owned_births_present=0,owned_gpu_pids=[],native_cgroup_exists=False,cleanup=read(B/'logs/cleanup-status.json'),source_epoch=ep,source_checkpoint=str(frozen),source_checkpoint_sha256=actual,source_checkpoint_bytes=before.st_size,actual_adam_steps_per_rank=[ep*100]*8,actual_optimizer_learning_rates=[3e-5],model_tensors=14,optimizer_states=14,rank_rng_count=8,independent_full_sha_performed=True,cpu_load_performed=True,gpu_forward_performed=False,source_model_shapes={k:list(v.shape)for k,v in saved['model'].items()},binding=saved['binding'],training_config=c,replay_sources=sources,test_source=dict(kind='cached_inputs',inputs=c['cached_inputs'],split='test'),cached_scan=scan,cached_first_row_metadata={k:v for k,v in first.items()if k not in ('question','prompt','gold')},original_terminal_logs=logs,ordinary_available_bytes=v.f_bavail*v.f_frsize,physical_free_bytes=v.f_bfree*v.f_frsize,large_payload_transferred_to_Mac=False)
once(O,result)
print(json.dumps(dict(status=result['status'],result_path=str(O),result_sha256=sha(O),result=result),allow_nan=False))
'''.replace('BASE',repr(B))
script='import subprocess,os,json; r=subprocess.run('+repr(['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',BODY])+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1"),capture_output=True,text=True,timeout=300); assert r.returncode==0,r.stderr; print(r.stdout)'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out=json.loads(remote(script,W/('question20-replay70-source-'+stamp+'-transport.json'),control_path=None,target='h100-8'))
p=W/('question20-replay70-source-'+stamp+'.json');p.write_text(json.dumps(out,indent=2)+'\n')
result=out.get('result')or{}
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status=out['status'],source_epoch=result.get('source_epoch'),source_checkpoint=result.get('source_checkpoint'),source_checkpoint_sha256=result.get('source_checkpoint_sha256'),ordinary_available_bytes=result.get('ordinary_available_bytes'),cached_scan=result.get('cached_scan'),cached_first_row_metadata=result.get('cached_first_row_metadata')),allow_nan=False))
