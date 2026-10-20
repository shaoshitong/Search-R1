import pathlib,json,hashlib,fcntl,os,subprocess,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=B/'algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized'
c=json.loads((b/'config.json').read_text());root=pathlib.Path(c['code_root'])
assert c['source_commit']=='2b293d6e6a0be5a8b99f976b3ac0c2b15e19ee40'
assert c['extra_response_indices']==[5,6,7,8,9]
p=root/'reproduction/operations/qwen7b_warmup/extra_g5_merge.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='d2eab1026193f486c44ab444a1efe42c408d1946536a0403523c6a8384094df6'
with (b/'g10-bank-merge.control.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=b/'bank/complete.json'
 if not out.exists():
  for i in range(4):
   own=json.loads((b/'sampling'/('shard-%02d'%i)/'extra-g5-complete.json').read_text())
   assert own['status']=='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED' and own['questions']==1632
  env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with (b/'g10-bank-merge.log').open('ab') as log:
   r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.extra_g5_merge','--base',str(b)],env=env,stdout=log,stderr=log,timeout=600)
  assert r.returncode==0,'G10 merge failed; see remote log; do not train'
 v=json.loads(out.read_text());assert v['questions']==6528 and v['trajectories']==65280
 print(json.dumps(dict(status=v['status'],questions=v['questions'],trajectories=v['trajectories'],sha256=hashlib.sha256(out.read_bytes()).hexdigest(),complete=str(out),epoch=time.time())))
