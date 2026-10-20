# Guarded one-time CPU merge using the immutable published c8b02d3b22b6 runtime.
import pathlib,json,hashlib,subprocess,time,os,fcntl
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=B/'algorithm-improvement-20261021/training-extraG5-128-20261021/materialized'
read=lambda p:json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
def fp(p):return dict(path=str(p),bytes=pathlib.Path(p).stat().st_size,sha256=sha(p))
def atomic(p,v):
 t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 t.replace(p)
c=read(b/'config.json');assert c['source_commit']=='c8b02d3b22b6a3267481e7c632b2f7113b018e93'
assert c['actor_updates_authorized'] is False and c['extra_response_indices']==list(range(5,10))
if (b/'bank/complete.json').exists():
 print(json.dumps(dict(status='EXISTING_G10_COMPLETE_DO_NOT_REMERGE',complete=fp(b/'bank/complete.json'),epoch=time.time())));raise SystemExit(0)
receipts=[]
for i in range(4):
 p=b/'sampling'/('shard-%02d'%i)/'extra-g5-complete.json'
 if not p.exists():
  print(json.dumps(dict(status='WAIT_FOR_ALL_FOUR_REAL_EXTRA_G5_RECEIPTS',ready_shards=len(receipts),epoch=time.time())));raise SystemExit(0)
 r=read(p);assert r['status']=='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED' and r['run_id']==c['run_id'] and r['questions']==32 and r['trajectories']==160
 receipts.append(fp(p))
root=pathlib.Path(c['code_root']);assert root.name=='code-c8b02d3b22b6'
for relative,digest in c['code_files_sha256'].items():assert sha(root/relative)==digest,relative
bank=b/'bank';bank.mkdir(exist_ok=True)
with (bank/'merge-control.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (bank/'complete.json').exists():
  print(json.dumps(dict(status='EXISTING_G10_COMPLETE_DO_NOT_REMERGE',complete=fp(bank/'complete.json'),epoch=time.time())));raise SystemExit(0)
 intent=bank/'merge-launch-intent.json'
 if intent.exists():
  v=read(intent)
  print(json.dumps(dict(status='EXISTING_MERGE_INTENT_INSPECT_NEVER_RELAUNCH',intent=v,complete_exists=(bank/'complete.json').exists(),epoch=time.time())));raise SystemExit(0)
 runtime=root/'reproduction/operations/qwen7b_warmup/runtime.sh'
 cmd=['flock','-n',str(bank/'merge-run.lock'),'bash',str(runtime),'-u','-m','reproduction.operations.qwen7b_warmup.extra_g5_merge','--base',str(b)]
 v=dict(status='ONE_CPU_MERGE_LAUNCH_INTENT',source_commit=c['source_commit'],code_root=str(root),runtime=fp(runtime),shard_receipts=receipts,command=cmd,actor_updates=0,epoch=time.time())
 atomic(intent,v)
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (bank/'extra-g5-merge.log').open('ab') as log:
  proc=subprocess.Popen(cmd,env=env,cwd=str(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:ticks=pathlib.Path('/proc/%d/stat'%proc.pid).read_text().rsplit(')',1)[1].split()[19]
 except OSError:ticks=None
 v.update(status='ONE_CPU_MERGE_STARTED_VERIFY_COMPLETION',pid=proc.pid,start_ticks=ticks)
 atomic(bank/'merge-controller.json',v)
 print(json.dumps(v))
