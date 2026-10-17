from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
script=r'''
import pathlib,json,subprocess,hashlib,time,sys
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-fit-20261017-0bdddc20840b-r48-no-loop-checks-resume1');p=b/'preparation/human-stop-A-run-BC5-terminal-once.json'
if p.exists():print(p.read_text());raise SystemExit
native=subprocess.run(['scontrol','show','job','3471','-o'],capture_output=True,text=True,timeout=20).stdout
assert 'JobState=CANCELLED'in native and str(b/'preparation/native-probe20k-fit.sh')in native
sys.path.insert(0,str(b/'repo'));from reproduction.prefix_only_correctness_ab.launch import birth_states
ledger=json.loads((b/'logs/process-ledger.json').read_text());births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
release=[]
for _ in range(2):
 remaining=birth_states(births);gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 own_gpu=[int(pid.strip())for pid in gpu.splitlines()if pid.strip().isdecimal()and int(pid.strip())in{r['pid']for r in births}]
 release.append(dict(remaining_original_births=remaining,owned_gpu=own_gpu,cgroups=[str(x)for x in pathlib.Path('/sys/fs/cgroup').rglob('job_3471')],checked_unix=time.time()));time.sleep(1)
assert all(not r['remaining_original_births']and not r['owned_gpu']and not r['cgroups']for r in release)
logs=[]
for file in(b/'logs').rglob('*.log'):
 st=file.stat();h=hashlib.sha256()
 with file.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 logs.append(dict(path=str(file),inode=st.st_ino,bytes=st.st_size,sha256=h.hexdigest()))
arms={}
for name in ('data_ema','data_ema_group2','data_ema_mixup'):
 root=b/'arms'/name;arms[name]={k:json.loads((root/(k+'.json')).read_text())if(root/(k+'.json')).exists()else None for k in ('latest','progress','complete')}
v=dict(base=str(b),job_id='3471',native=native,original_birth_count=len(births),original_births_all_gone=True,no_live_own_gpu=True,release=release,original_inode_logs=logs,arms=arms,
 scope='Human stops baseline A; continue only B/C for five matched epochs; no actor/PPO/new rollout',checked_unix=time.time())
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
p=Path('work')/('probe20k3471-human-stop-terminal-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arms=v['arms'],original_birth_count=v['original_birth_count'],released=True)))
