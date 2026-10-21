import sys,json,hashlib,subprocess,os,time
from pathlib import Path
x=json.load(sys.stdin);b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/optimizations')/('selected-head-'+x['commit']);b.mkdir(parents=True,exist_ok=True)
for n,v in x['files'].items():
 p=b/n;p.parent.mkdir(exist_ok=True,parents=True);assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256']
 if p.exists():assert p.read_text()==v['text']
 else:p.write_text(v['text'])
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
(b/'search_g1/__init__.py').touch()
env=dict(os.environ,PYTHONPATH=str(b),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TRANSFORMERS_OFFLINE='1')
s=time.time();r=subprocess.run(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-m','unittest','search_g1.test_selected_qwen_head','-v'],cwd=b,env=env,capture_output=True,text=True,timeout=40)
out=dict(epoch=time.time(),elapsed=time.time()-s,commit=x['commit'],files={n:v['sha256']for n,v in x['files'].items()},returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,model_weights_downloaded=False,GPU_allocations=0,training_source_changed=False)
(b/'actual-cpu-tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
