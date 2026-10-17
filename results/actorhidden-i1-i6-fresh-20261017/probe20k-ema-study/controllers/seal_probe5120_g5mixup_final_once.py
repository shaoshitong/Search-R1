from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
inner=r'''
import pathlib,json,subprocess,hashlib,time,sys
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-fit-20261017-0bdddc20840b-r48-q5120-g5-mixup-lr5e5');p=b/'preparation/g5mixup5120-final-native-state-release-once.json'
if p.exists():print(p.read_text());raise SystemExit
job=json.loads((b/'preparation/submit-receipt.json').read_text())['job_id']
native=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=20).stdout
assert 'JobState=COMPLETED'in native and 'ExitCode=0:0'in native and str(b/'preparation/native-probe20k-fit.sh')in native
sys.path.insert(0,str(b/'repo'));from reproduction.prefix_only_correctness_ab.launch import birth_states
ledger=json.loads((b/'logs/process-ledger.json').read_text());births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
release=[]
for _ in range(2):
 remaining=birth_states(births);gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 own_gpu=[int(pid.strip())for pid in gpu.splitlines()if pid.strip().isdecimal()and int(pid.strip())in{r['pid']for r in births}]
 release.append(dict(remaining_original_births=remaining,owned_gpu=own_gpu,cgroups=[str(x)for x in pathlib.Path('/sys/fs/cgroup').rglob('job_'+job)],checked_unix=time.time()));time.sleep(1)
assert all(not r['remaining_original_births']and not r['owned_gpu']and not r['cgroups']for r in release)
def digest(file):
 h=hashlib.sha256()
 with file.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
logs=[]
for file in(b/'logs').rglob('*.log'):
 st=file.stat();h=digest(file);after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 logs.append(dict(path=str(file),inode=st.st_ino,bytes=st.st_size,sha256=h))
import torch
arms={};c=json.loads((b/'preparation/launch-config.json').read_text())
for arm in c['probe20k_study']['arms']:
 root=b/'arms'/arm['name'];complete=json.loads((root/'complete.json').read_text());assert complete['epochs']==5 and complete['actual_adam_per_rank']==[700]*8
 r=json.loads((root/'latest.json').read_text());file=pathlib.Path(r['checkpoint']);st=file.stat();h=digest(file);assert h==r['checkpoint_sha256']and st.st_size==r['checkpoint_bytes']
 saved=torch.load(file,map_location='cpu',weights_only=False)
 assert saved['epoch']==5 and saved['actual_adam_per_rank']==[700]*8 and saved['ema_updates']==500 and saved['ema_decay']==.99 and saved['run_id']==c['run_id']
 assert saved['train_questions']==5120 and saved['selected_data_blocks']==[0]and saved['updates_per_epoch']==100
 assert len(saved['rank_rng'])==8 and saved['model'].keys()==saved['ema_model'].keys()
 steps=[int(v['step'].item())for v in saved['optimizer']['state'].values()];assert len(steps)==14 and set(steps)=={700}
 ranks=[]
 for rank in range(8):
  rows=[json.loads(line)for line in(root/('rank-%d-updates.jsonl'%rank)).open()]
  assert len(rows)==500 and rows[-1]['actual_adam']==700
  ranks.append(dict(rank=rank,actual_logged_updates=len(rows),last=rows[-1]))
 points=[json.loads((root/('epoch-%02d-complete.json'%e)).read_text())for e in range(1,6)]
 after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 arms[arm['name']]=dict(final_checkpoint=str(file),bytes=st.st_size,independent_full_sha256=h,cpu_state_load=True,model_and_ema_tensors=len(saved['model']),optimizer_states=len(steps),eight_rng_present=True,actual_adam_per_rank=saved['actual_adam_per_rank'],ema_updates=saved['ema_updates'],rank_update_journals=ranks,normal_savepoints=points)
 del saved
v=dict(base=str(b),job_id=job,native=native,original_birth_count=len(births),original_births_all_gone=True,no_live_own_gpu=True,release=release,original_inode_logs=logs,arms=arms,all_requested_g5mixup5_complete=True,formal_actor_resumed=False,checked_unix=time.time())
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work')/('probe5120-g5mixup-final-state-release-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arms={k:{key:a[key]for key in ('actual_adam_per_rank','ema_updates','independent_full_sha256','cpu_state_load')}for k,a in v['arms'].items()},released=True)))
