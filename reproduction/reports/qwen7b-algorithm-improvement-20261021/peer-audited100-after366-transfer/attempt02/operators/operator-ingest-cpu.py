inputs={'base': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-delta-import-runtime-audited100-after366-20261021-636015f5-a02', 'target': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-delta-import-runtime-audited100-after366-20261021-636015f5-a02', 'receiver': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/audited100-after366-20261021-636015f5-a02', 'ingest': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited100-after366-20261021-636015f5-a02', 'commit': 'd252e9ed943fc6993184d77b45103eeb4ee8bdc2', 'bundle_sha256': '52757e5b4782ef4dbcb9c36eacde9f2d8d3c4de3c0f2c2cd5d45cb1bfbc3378b', 'bundle_bytes': 96648, 'chunks': 24, 'cpu_mode': 'ingest'}

import base64,fcntl,gzip,hashlib,io,json,os,pathlib,shlex,subprocess,tarfile,time
b=pathlib.Path(inputs['base']);assert str(b)==inputs['target'] and b.resolve()==b
code=b/('code-'+inputs['commit'][:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
 q.replace(p)
def lock():
 b.mkdir(parents=True,exist_ok=True);f=(b/'operator.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def publication():
 p=read(b/'source-publication.json');root=pathlib.Path(p['code_root']);assert root.parent==b
 for name,v in p['files'].items():assert fp(root/name)==dict(v,path=str(root/name)),name
 return p,root
def command(args,timeout=25):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert r.returncode==0,(r.stdout[-1000:],r.stderr[-2000:]);return r.stdout.strip()
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def runtime(root,args):
 return ['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m']+args


guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
args=runtime(root,['unittest','reproduction.operations.qwen7b_warmup.test_peer_newseed_ingest','reproduction.operations.qwen7b_warmup.test_peer_newseed_replay'])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
smoke="from reproduction.prefix20_online.rollout import strict_label; from reproduction.operations.qwen7b_warmup.test_peer_newseed_ingest import CharTokenizer; x=list(map(ord,'<answer>Paris</answer>')); assert strict_label(CharTokenizer(),x,[1]*len(x),{'target':['Paris']})==1; assert strict_label(CharTokenizer(),x,[0]*len(x),{'target':['Paris']})==0; print('UNCHANGED_PRODUCTION_STRICT_LABEL_IMPORT_AND_MASK_PASS')"
r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',smoke],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
(b/'actual-production-strict-EM-smoke.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
atomic(b/'cpu-qualification.json',dict(status='ACTUAL_REMOTE_CPU_11_TESTS_PASSED',log=fp(b/'actual-cpu-tests.log'),production_strict_label_smoke=fp(b/'actual-production-strict-EM-smoke.log'),model_tensors_loaded=False,feature_capture_started=False))
print(json.dumps(read(b/'cpu-qualification.json')))
