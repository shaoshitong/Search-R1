from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
inner=r'''
import pathlib,json,subprocess,hashlib,time,sys,os
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-fit-20261017-0bdddc20840b-r48');p=b/'preparation/check-removal-resume-terminal-cpu-once.json'
if p.exists():print(p.read_text());raise SystemExit
native=subprocess.run(['scontrol','show','job','3470','-o'],capture_output=True,text=True,timeout=20).stdout
assert 'JobState=CANCELLED'in native and str(b/'preparation/native-probe20k-fit.sh')in native
sys.path.insert(0,str(b/'repo'));from reproduction.prefix_only_correctness_ab.launch import birth_states
ledger=json.loads((b/'logs/process-ledger.json').read_text());births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
release=[]
for _ in range(2):
 remaining=birth_states(births);gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 own_gpu=[int(pid.strip())for pid in gpu.splitlines()if pid.strip().isdecimal()and int(pid.strip())in{r['pid']for r in births}]
 release.append(dict(remaining_original_births=remaining,owned_gpu=own_gpu,cgroups=[str(x)for x in pathlib.Path('/sys/fs/cgroup').rglob('job_3470')],checked_unix=time.time()));time.sleep(1)
assert all(not r['remaining_original_births']and not r['owned_gpu']and not r['cgroups']for r in release)
logs=[]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for file in(b/'logs').rglob('*.log'):
 st=file.stat();h=digest(file);after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 logs.append(dict(path=str(file),inode=st.st_ino,device=st.st_dev,bytes=st.st_size,sha256=h))
import torch
arms={};c=json.loads((b/'preparation/launch-config.json').read_text())
for arm in c['probe20k_study']['arms']:
 root=b/'arms'/arm['name'];last=root/'latest.json'
 if not last.exists():arms[arm['name']]=None;continue
 r=json.loads(last.read_text());file=pathlib.Path(r['checkpoint']);st=file.stat();actualsha=digest(file);assert actualsha==r['checkpoint_sha256']and st.st_size==r['checkpoint_bytes']
 saved=torch.load(file,map_location='cpu',weights_only=False);expected=200+400*r['epoch']
 assert saved['epoch']==r['epoch']and saved['actual_adam_per_rank']==[expected]*8 and saved['run_id']==c['run_id']and saved['arm']==arm
 assert len(saved['rank_rng'])==8 and saved['ema_updates']==400*r['epoch']and saved['ema_decay']==.99
 assert saved['model'].keys()==saved['ema_model'].keys()
 assert all(v.dtype==torch.float32 for v in saved['model'].values()if v.is_floating_point())
 assert all(v.dtype==torch.float32 for v in saved['ema_model'].values()if v.is_floating_point())
 steps=[int(v['step'].item())for v in saved['optimizer']['state'].values()];assert len(steps)==14 and set(steps)=={expected}
 after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 arms[arm['name']]=dict(r,independent_full_sha256=actualsha,cpu_state_load=True,ema_updates=saved['ema_updates'],model_tensors=len(saved['model']),optimizer_states=len(steps),eight_rng_present=True)
 del saved
value=dict(base=str(b),study_id=c['run_id'],native=native,original_birth_count=len(births),original_inode_logs=logs,release=release,
 arms=arms,original_births_all_gone=True,no_live_own_gpu=True,producer_cleanup=json.loads((b/'logs/cleanup-status.json').read_text()),
 controlled_stop_for_human_requested_loop_check_removal=True,whole_study_paused=False,checked_unix=time.time())
with p.open('x')as f:json.dump(value,f,indent=2)
print(json.dumps(value))
'''
script='import os,subprocess\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES="");r=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=env,text=True,capture_output=True,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work')/('probe20k3470-resume-terminal-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arms=v['arms'],original_birth_count=v['original_birth_count'],released=v['original_births_all_gone'])))
