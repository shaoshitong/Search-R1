from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
inner=r'''
import pathlib,json,subprocess,hashlib,time,sys
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe-dist-divkl-resplit5120-15360-20261018-0bdddc20840b');p=b/'preparation/failed3485-zero-new-adam-originals-once.json'
if p.exists():print(p.read_text());raise SystemExit
native=subprocess.run(['scontrol','show','job','3485','-o'],capture_output=True,text=True,timeout=20).stdout
assert 'JobState=FAILED'in native and str(b/'preparation/native-probe20k-fit.sh')in native
sys.path.insert(0,str(b/'repo'));from reproduction.prefix_only_correctness_ab.launch import birth_states
ledger=json.loads((b/'logs/process-ledger.json').read_text());births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
release=[]
for _ in range(2):
 remaining=birth_states(births);gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 own_gpu=[int(pid.strip())for pid in gpu.splitlines()if pid.strip().isdecimal()and int(pid.strip())in{r['pid']for r in births}]
 release.append(dict(remaining_original_births=remaining,owned_gpu=own_gpu,cgroups=[str(x)for x in pathlib.Path('/sys/fs/cgroup').rglob('job_3485')],checked_unix=time.time()));time.sleep(1)
assert all(not r['remaining_original_births']and not r['owned_gpu']and not r['cgroups']for r in release)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
logs=[]
for file in(b/'logs').rglob('*.log'):
 st=file.stat();h=digest(file);after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 logs.append(dict(path=str(file),inode=st.st_ino,bytes=st.st_size,sha256=h))
c=json.loads((b/'preparation/launch-config.json').read_text());arms={}
for arm in c['probe20k_study']['arms']:
 root=b/'arms'/arm['name'];rows=[json.loads(line)for file in root.glob('rank-*-updates.jsonl')for line in file.open()]
 assert not rows and not(root/'latest.json').exists()
 split=root/'question-split.json';v=json.loads(split.read_text());tr=set(v['train']['question_ids']);te=set(v['test']['question_ids'])
 assert len(tr)==5120 and len(te)==15360 and not(tr&te)
 arms[arm['name']]=dict(actual_new_adam=0,complete_epochs=0,split_sha256=digest(split),train_questions=5120,test_questions=15360)
v=dict(base=str(b),job_id='3485',native=native,original_birth_count=len(births),original_births_all_gone=True,no_live_own_gpu=True,release=release,original_inode_logs=logs,arms=arms,no_actual_new_adam=True,error='group_predictions old test-size whitelist rejected15360 before training',checked_unix=time.time())
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=100);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work')/('probe-resplit3485-native-terminal-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),released=v['original_births_all_gone'],arms=v['arms'])))
