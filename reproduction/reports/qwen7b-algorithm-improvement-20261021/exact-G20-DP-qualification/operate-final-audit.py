"""Published code-only stage, then explicit final-only CPU audit; no GPU action."""
import argparse, base64, hashlib, importlib.util, json, shlex, subprocess
from pathlib import Path
from reproduction.operations.qwen7b_warmup.deploy_confirmation_materialization import make_bundle

C = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
R = C / 'work/qwen7b-probe-warmup-20261019'
A = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
TARGET = A + '/g20-exact-uniform15504-dp-final-audit-v1'
parser = argparse.ArgumentParser(); parser.add_argument('--stage-only', action='store_true'); parser.add_argument('--source-commit'); args = parser.parse_args()
source_receipt = Path(__file__).with_name('final-audit-stage-source.json')
pinned = args.source_commit or (json.loads(source_receipt.read_text())['source_commit'] if source_receipt.exists() else 'HEAD')
commit = subprocess.check_output(['git', 'rev-parse', pinned], cwd=R, text=True).strip()
remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/codex/qwen7b-probe-warmup-20261019'], cwd=R, text=True).split()[0]
subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=R, check=True)
paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=R, text=True).splitlines()
cache = {}
def load(name):
    if name not in cache:cache[name] = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=R)
    return cache[name]
module = 'reproduction/operations/qwen7b_warmup/audit_exact_g20_dp_final.py'
assert (R/module).read_bytes() == load(module)
data, manifest = make_bundle(paths, load, commit, seeds=(module,), resource_files=(), manifest_name='_source.json')
spec = importlib.util.spec_from_file_location('dp_final_ctl', C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
COMMON = r'''
import base64,fcntl,hashlib,io,json,os,pathlib,subprocess,tarfile,time
b=pathlib.Path(inputs['base']);b.mkdir(exist_ok=True);root=b/('code-'+inputs['commit'][:12]);root.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');t.replace(p)
guard=(b/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
'''
def call(body, extra=None):
    values = dict(base=TARGET, commit=commit); values.update(extra or {})
    command = 'python3 -c ' + shlex.quote('inputs=' + repr(values) + '\n' + COMMON + '\n' + body)
    assert len(command.encode()) < 16000
    done = ctl.run(command, account='mou176', source_address=address, timeout=55)
    if done.returncode:raise RuntimeError('Final CPU outcome unconfirmed; inspect before repeating: ' + done.stdout.decode()[-2000:] + done.stderr.decode()[-1000:])
    return json.loads(done.stdout.decode().splitlines()[-1])
for i, start in enumerate(range(0, len(data), 4096)):
    chunk = data[start:start+4096]
    call("p=root/('part-%04d'%inputs['i']);x=base64.b64decode(inputs['data']);assert hashlib.sha256(x).hexdigest()==inputs['sha256']\nif p.exists():assert p.read_bytes()==x\nelse:p.write_bytes(x)\nprint(json.dumps(dict(chunk=inputs['i'])))",dict(i=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest()))
body = r'''
data=b''.join((root/('part-%04d'%i)).read_bytes() for i in range(inputs['chunks']));assert hashlib.sha256(data).hexdigest()==inputs['sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 raw=tar.extractfile('_source.json').read();assert hashlib.sha256(raw).hexdigest()==inputs['manifest_sha'];manifest=json.loads(raw);files=manifest['files'];assert {m.name for m in tar.getmembers()}==set(files)|{'_source.json'}
 for name in set(files)|{'_source.json'}:
  rel=pathlib.Path(name);member=tar.getmember(name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts;raw=tar.extractfile(member).read()
  if name in files:assert len(raw)==files[name]['bytes'] and hashlib.sha256(raw).hexdigest()==files[name]['sha256']
  p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
pub=dict(commit=inputs['commit'],remote_sha=inputs['commit'],code_root=str(root),files=files);p=b/'source-publication.json'
if p.exists():assert read(p)==pub,'Existing final audit source preserved'
else:atomic(p,pub)
for name,v in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==v['sha256']
print(json.dumps(dict(status='FINAL_CPU_AUDIT_SOURCE_STAGED_NO_MODEL_NO_GPU',source_publication=pub)))
'''
staged = call(body, dict(chunks=(len(data)+4095)//4096,sha256=hashlib.sha256(data).hexdigest(),manifest_sha=manifest['source_manifest_sha256']))
source_receipt.write_text(json.dumps(dict(source_commit=commit,remote_sha=remote,staged=staged),indent=2)+'\n')
if args.stage_only:
    print(json.dumps(staged)); raise SystemExit(0)
body = r'''
source=pathlib.Path(inputs['training']);assert (source/'complete.json').exists() and read(source/'gpu-memory.json')['exit_normally']
if (b/'complete.json').exists():print(json.dumps(dict(status='EXISTING_FINAL_CPU_AUDIT_PRESERVED',result=read(b/'complete.json'))))
else:
 assert not (b/'intent.json').exists(),'Unknown prior final audit, no blind retry';atomic(b/'intent.json',dict(epoch=time.time(),CPU_limit=2,memory_cap_bytes=3*1024**3,GPU_requested=0))
 wrapper="import os,resource,runpy;os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,300));runpy.run_module('reproduction.operations.qwen7b_warmup.audit_exact_g20_dp_final',run_name='__main__')"
 env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (b/'cpu.log').open('ab') as log:r=subprocess.run(['python3','-u','-c',wrapper],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
 atomic(b/'exit.json',dict(returncode=r.returncode,epoch=time.time()));assert r.returncode==0,(b/'cpu.log').read_text()[-2500:]
 result=read(source/'independent-final-audit.json');atomic(b/'complete.json',result)
 print(json.dumps(dict(status='ACTUAL_FINAL_CPU_METRIC_AUDIT_COMPLETE',result=result,exit=read(b/'exit.json'))))
'''
result = call(body, dict(training=A+'/g20-exact-uniform15504-dp-v1'))
Path(__file__).with_name('final-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result), flush=True)
