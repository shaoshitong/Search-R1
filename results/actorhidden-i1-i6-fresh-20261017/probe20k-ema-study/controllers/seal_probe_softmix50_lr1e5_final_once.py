from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
inner=r'''
import pathlib,json,subprocess,hashlib,time,sys
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe-dist-softmix50-lr1e5-resplit5120-15360-20261018-0bdddc20840b');p=b/'preparation/softmix50-lr1e5-final-native-state-release-once.json'
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
 root=b/'arms'/arm['name'];complete=json.loads((root/'complete.json').read_text());assert complete['epochs']==10 and complete['actual_adam_per_rank']==[1200]*8
 r=json.loads((root/'latest.json').read_text());file=pathlib.Path(r['checkpoint']);st=file.stat();h=digest(file);assert h==r['checkpoint_sha256']and st.st_size==r['checkpoint_bytes']
 saved=torch.load(file,map_location='cpu',weights_only=False)
 assert saved['arm']==arm and arm['objective']=='group_vector_dist_softmix_kl'and arm['soft_mixed_weight']==50 and arm['diversity_weight']==0 and arm['trajectory_weight']==0 and arm['mean_weight']==0
 assert saved['epoch']==10 and saved['actual_adam_per_rank']==[1200]*8 and saved['ema_updates']==1000 and saved['ema_decay']==.99 and saved['run_id']==c['run_id']
 assert saved['learning_rate']==1e-5 and all(g['lr']==1e-5 for g in saved['optimizer']['param_groups'])and saved['arm']['soft_target_by_K']==[0,.9,1,1,.9,0]
 assert saved['train_questions']==5120 and saved['selected_data_blocks']==[0,1,2,3]and saved['test_questions']==15360 and saved['question_split']['seed']==20261018and saved['updates_per_epoch']==100
 assert len(saved['rank_rng'])==8 and saved['model'].keys()==saved['ema_model'].keys()
 assert all(v.dtype==torch.float32 for k in ('model','ema_model')for v in saved[k].values()if torch.is_tensor(v)and v.is_floating_point())
 assert all(v.dtype==torch.float32 for state in saved['optimizer']['state'].values()for key,v in state.items()if key!='step'and torch.is_tensor(v))
 steps=[int(v['step'].item())for v in saved['optimizer']['state'].values()];assert len(steps)==14 and set(steps)=={1200}
 ranks=[]
 for rank in range(8):
  rows=[json.loads(line)for line in(root/('rank-%d-updates.jsonl'%rank)).open()]
  assert len(rows)==1000 and rows[-1]['actual_adam']==1200
  ranks.append(dict(rank=rank,actual_logged_updates=len(rows),last=rows[-1]))
 points=[json.loads((root/('epoch-%02d-complete.json'%e)).read_text())for e in range(1,11)]
 after=file.stat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 split_path=root/'question-split.json';split=json.loads(split_path.read_text());train_q=set(split['train']['question_ids']);test_q=set(split['test']['question_ids'])
 assert len(train_q)==5120 and len(test_q)==15360 and not(train_q&test_q)and len(train_q|test_q)==20480
 assert saved['dataset_manifest_sha256']==split['train_manifest_sha256']and saved['test_manifest_sha256']==split['test_manifest_sha256']
 arms[arm['name']]=dict(question_split_sha256=digest(split_path),split_seed=20261018,train_questions=5120,test_questions=15360,question_overlap=0,final_checkpoint=str(file),bytes=st.st_size,independent_full_sha256=h,cpu_state_load=True,model_and_ema_tensors=len(saved['model']),optimizer_states=len(steps),eight_rng_present=True,actual_adam_per_rank=saved['actual_adam_per_rank'],ema_updates=saved['ema_updates'],rank_update_journals=ranks,normal_savepoints=points)
 del saved
v=dict(base=str(b),job_id=job,native=native,original_birth_count=len(births),original_births_all_gone=True,no_live_own_gpu=True,release=release,original_inode_logs=logs,arms=arms,all_requested_softmix50_lr1e5_complete=True,formal_actor_resumed=False,checked_unix=time.time())
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)'
p=Path('work')/('probe5120-test15360-softmix50-lr1e5-final-state-release-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');v=json.loads(remote(script,str(p.with_suffix('.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arms={k:{key:a[key]for key in ('actual_adam_per_rank','ema_updates','independent_full_sha256','cpu_state_load')}for k,a in v['arms'].items()},released=True)))
