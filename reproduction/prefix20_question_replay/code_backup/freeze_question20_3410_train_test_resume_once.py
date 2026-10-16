"""Freeze the real latest replay boundary for the requested paired metrics."""
from pathlib import Path
import datetime,hashlib,json,sys
W=Path(__file__).parent
R=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
sys.path.insert(0,str(R))
from reproduction.h1008_qprobe.management_transport import remote
B='/mnt/data/shaoshitong/search-g1-question20-replay70-20261016-5534d25'
BODY=r'''
from pathlib import Path
import json,datetime,hashlib,os,subprocess,sys
import torch
B=Path(BASE);sys.path.insert(0,str(B/'repo'))
from reproduction.common8.native_allocation import fields
from reproduction.prefix_only_correctness_ab.launch import birth_states
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def sig(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def once(p,x):
 with p.open('x')as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
P=B/'preparation';I=P/'train-test-mixed-resume-freeze-once-v1-intent.json';O=P/'train-test-mixed-resume-freeze-once-v1-result.json'
if I.exists():
 print(json.dumps(dict(status='EXISTING_RESUME_FREEZE_NO_REPLAY',proof=read(O)if O.exists()else None,proof_path=str(O),proof_sha256=sha(O)if O.exists()else None)));raise SystemExit
c=read(P/'launch-config.json');raw=subprocess.check_output(['scontrol','show','job','3410','-o'],text=True,timeout=30);n=fields(raw)
assert n['JobId']=='3410'and n['JobState']in('CANCELLED','FAILED','COMPLETED','TIMEOUT')
assert n['Command']==str(P/'native-launch.sh')and n['Comment']=='g1-q20-replay70:'+c['run_id']
ledger=read(B/'logs/process-ledger.json');births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
assert not birth_states(births)
assert not(list(Path('/sys/fs/cgroup').glob('**/job_3410'))+list(Path('/sys/fs/cgroup').glob('**/job3410')))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=30).split()
assert not(set(gpu)&{str(r['pid'])for r in births})
latest=read(B/'train/latest.json');i=int(latest['replay_iteration']);expected=[7100+i*100]*8
assert latest['actual_adam_steps_per_rank']==expected and latest['actual_optimizer_learning_rates']==[3e-6]
complete=read(B/'train/iterations'/('iteration-%04d'%i)/'complete.json');assert complete['sha256']==latest['sha256']
once(I,dict(native=n,latest=latest,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
frozen=P/('train-test-mixed-source-replay%04d.pt'%i);os.link(B/'train/latest.pt',frozen)
before=frozen.stat();actual=sha(frozen);assert actual==latest['sha256']
saved=torch.load(frozen,map_location='cpu',weights_only=False)
assert saved['replay_iteration']==i and saved['source_epoch']==71 and saved['actual_adam_steps_per_rank']==expected
assert saved['binding']['question_hidden_passed_to_head']and saved['binding']['question_max_tokens']==273
assert saved['binding']['learning_rate']==3e-6 and len(saved['model'])==len(saved['optimizer']['state'])==14
assert saved['model']['output.weight'].shape==(1,600064)
assert all(x.dtype==torch.float32 for x in saved['model'].values())
assert [r['rank']for r in saved['rank_rng']]==list(range(8))
for r in saved['rank_rng']:assert r['actual_adam_steps']==expected[0]
for state in saved['optimizer']['state'].values():
 assert int(state['step'])==expected[0]and state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
assert sig(before)==sig(frozen.stat())
test=B/'fixed-test/features';testsha=sha(test/'manifest.json');assert testsha==saved['binding']['test_manifest_sha256']
logs=[]
for path in sorted((B/'logs').rglob('*')):
 if not path.is_file()or path.is_symlink():continue
 st=path.stat();digest=sha(path);assert sig(st)==sig(path.stat())
 logs.append(dict(path=str(path),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,sha256=digest))
proof=dict(status='LATEST_REPLAY_FULL_SHA_CPU_RESUME_STATE_PASS',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_base=str(B),source_run_id=c['run_id'],native_job_id=3410,native_state=n['JobState'],native_exit_code=n['ExitCode'],native_raw=raw,owned_births_recorded=len(births),owned_births_present=0,native_cgroup_exists=False,owned_gpu_pids=[],completed_iteration=i,checkpoint_path=str(frozen),checkpoint_sha256=actual,checkpoint_bytes=before.st_size,latest_receipt_path=str(B/'train/latest.json'),latest_receipt_sha256=sha(B/'train/latest.json'),run_claim_path=str(B/'train/run-claim.json'),run_claim_sha256=sha(B/'train/run-claim.json'),question_max_proof_path=str(P/'question-max.json'),question_max_proof_sha256=sha(P/'question-max.json'),question_max_tokens=273,fixed_test_features_path=str(test),fixed_test_manifest_sha256=testsha,actual_adam_steps_per_rank=expected,actual_learning_rate=3e-6,model_parameter_tensors=14,optimizer_states=14,rank_rng_count=8,independent_full_sha_performed=True,cpu_load_performed=True,gpu_forward_performed=False,original_terminal_logs=logs,source_config=c,source_metrics=read(B/'train/iterations'/('iteration-%04d'%i)/'metrics.json'),large_payload_transferred_to_Mac=False)
once(O,proof);print(json.dumps(dict(status=proof['status'],proof_path=str(O),proof_sha256=sha(O),proof=proof),allow_nan=False))
'''.replace('BASE',repr(B))
script='import subprocess,os; r=subprocess.run('+repr(['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',BODY])+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1"),capture_output=True,text=True,timeout=180); assert r.returncode==0,r.stderr; print(r.stdout)'
t=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
o=json.loads(remote(script,W/('question20-train-test-resume-source-'+t+'-transport.json'),control_path=None,target='h100-8'))
p=W/('question20-train-test-resume-source-'+t+'.json');p.write_text(json.dumps(o,indent=2)+'\n')
v=o.get('proof')or{}
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status=o['status'],source_completed_iteration=v.get('completed_iteration'),checkpoint_sha256=v.get('checkpoint_sha256'),actual_adam_steps_per_rank=v.get('actual_adam_steps_per_rank'),fixed_test_manifest_sha256=v.get('fixed_test_manifest_sha256')),allow_nan=False))
