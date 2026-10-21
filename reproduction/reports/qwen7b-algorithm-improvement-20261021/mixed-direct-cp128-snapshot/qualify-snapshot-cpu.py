import base64,hashlib,importlib.util,io,json,shlex,subprocess,sys,tarfile,zlib
from pathlib import Path
C=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5');R=C/'work/qwen7b-probe-warmup-20261019';P='reproduction/operations/qwen7b_warmup/';commit=sys.argv[1];model='8c0ccaf4eda6d97d46c9f0ea82620aa1a6d9a973'
spec=importlib.util.spec_from_file_location('ctl',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
base='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-cp128-eval-cpu-qualification-v1';code=base+'/code-'+commit[:12];old='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1/code-3b56e583efe2'
def call(script,t=50):
 r=ctl.run('python3 -c '+shlex.quote(script),account='mou176',source_address=ip,timeout=t)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-2000:]+r.stdout.decode(errors='replace')[-3500:])
 return r.stdout.decode()
files={}
for n in('evaluate_mixed_direct_checkpoint.py','test_evaluate_mixed_direct_checkpoint.py'):files[P+n]=subprocess.check_output(['git','show',commit+':'+P+n],cwd=R)
for n in('joint_mixed_direct_pretrain.py','mixed_group_checkpoint.py','test_joint_mixed_direct_pretrain.py'):files[P+n]=subprocess.check_output(['git','show','e2fd6e7fbe4fbd6d62fc86923565c01765d02794:'+P+n],cwd=R)
files[P+'mixed_direct_evaluation.py']=subprocess.check_output(['git','show','7380f3bfa82e4d699a617590d25d56bf4ef8b5ad:'+P+'mixed_direct_evaluation.py'],cwd=R)
files[P+'native_mixed_group_classifier.py']=subprocess.check_output(['git','show',model+':'+P+'native_mixed_group_classifier.py'],cwd=R)
files[P+'balanced_hard_g10_sampler.py']=subprocess.check_output(['git','show','c98ddcd7861b7023f3efa0d82d3c1fbea0664a13:'+P+'balanced_hard_g10_sampler.py'],cwd=R)
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w')as tar:
 for n,b in files.items():
  info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
raw=zlib.compress(out.getvalue());parts=[raw[o:o+6144]for o in range(0,len(raw),6144)];sha=hashlib.sha256(raw).hexdigest()
for i,chunk in enumerate(parts):
 s="import pathlib,base64,hashlib;root=pathlib.Path(%r);root.mkdir(parents=True,exist_ok=True);p=root/('part-%%03d'%%%d);raw=base64.b64decode(%r);assert not p.exists()or p.read_bytes()==raw;p.write_bytes(raw);print('codepart',%d,len(raw))"%(base,i,base64.b64encode(chunk).decode(),i)
 print(call(s).strip(),flush=True)
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),source_commit=model if n.endswith('native_mixed_group_classifier.py')else'c98ddcd7861b7023f3efa0d82d3c1fbea0664a13'if n.endswith('balanced_hard_g10_sampler.py')else'7380f3bfa82e4d699a617590d25d56bf4ef8b5ad'if n.endswith('mixed_direct_evaluation.py')else'e2fd6e7fbe4fbd6d62fc86923565c01765d02794'if n.endswith(('joint_mixed_direct_pretrain.py','mixed_group_checkpoint.py'))else commit)for n,b in files.items()}
stage="""import pathlib,hashlib,shutil,tarfile,io,zlib,json
root=pathlib.Path(%r);code=pathlib.Path(%r);raw=b''.join((root/('part-%%03d'%%i)).read_bytes()for i in range(%d));assert hashlib.sha256(raw).hexdigest()==%r
if not code.exists():shutil.copytree(%r,str(code))
manifest=%r
with tarfile.open(fileobj=io.BytesIO(zlib.decompress(raw)))as t:
 for m in t:
  assert m.name in manifest and m.isfile();payload=t.extractfile(m).read();assert hashlib.sha256(payload).hexdigest()==manifest[m.name]['sha256'];p=code/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists()or p.read_bytes()!=payload:
   if p.exists():p.chmod(p.stat().st_mode|128)
   p.write_bytes(payload)
p=root/'source-publication.json';v=dict(source_commit=%r,model_source_commit=%r,base_runtime_source='3b56e583efe2469ac9108873c31de7144a5201e6',code_root=str(code),files=manifest);p.write_text(json.dumps(v,sort_keys=True));print(json.dumps(dict(status='CODE_ONLY_IMMUTABLE_SYNTHETIC_FIXTURE_STAGED',code_root=str(code),publication=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))))
"""%(base,code,len(parts),sha,old,manifest,commit,model)
print(call(stage).strip(),flush=True)
run="""import pathlib,subprocess,os,json,time,hashlib,resource
root=pathlib.Path(%r);code=pathlib.Path(%r);env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2');log=root/'cpu-fixtures.log';start=time.time()
def limits():
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_CPU,(45,45))
with log.open('wb')as f:r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','reproduction.operations.qwen7b_warmup.test_evaluate_mixed_direct_checkpoint','-v'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=48,preexec_fn=limits)
v=dict(status='PASSED'if r.returncode==0 else'FAILED',returncode=r.returncode,seconds=time.time()-start,CPU=2,GPU=0,synthetic_only=True,no_real_tensor_data_read=True,peak_child_RSS_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,log=dict(path=str(log),bytes=log.stat().st_size,sha256=hashlib.sha256(log.read_bytes()).hexdigest()),source_publication=dict(path=str(root/'source-publication.json'),sha256=hashlib.sha256((root/'source-publication.json').read_bytes()).hexdigest()));(root/'cpu-qualification.json').write_text(json.dumps(v,sort_keys=True));print(json.dumps(v));print(log.read_text()[-4500:]);assert r.returncode==0
"""%(base,code)
result=call(run,55);print(result,flush=True);O=C/'outputs/qwen7b-warmup-20261019';(O/'mixed-direct-cp128-eval-cpu-qualification-console.txt').write_text(result)
