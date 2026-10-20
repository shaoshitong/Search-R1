"""Stage published auditor only; one CPU overlap step in owned G20 allocation."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

C = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
R = C / 'work/qwen7b-probe-warmup-20261019'
BASE = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-exact-uniform15504-dp-v1'
COMMIT = '2590708a77893dc6bab774485ea6e613f0375864'
FILE = 'reproduction/operations/qwen7b_warmup/audit_exact_g20_dp_checkpoint.py'
raw = subprocess.check_output(['git', 'show', COMMIT + ':' + FILE], cwd=R)
remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/codex/qwen7b-probe-warmup-20261019'], cwd=R, text=True).split()[0]
subprocess.run(['git', 'merge-base', '--is-ancestor', COMMIT, remote], cwd=R, check=True)
spec = importlib.util.spec_from_file_location('dp_cp_control', C / 'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
COMMON = r'''
import base64,fcntl,hashlib,json,os,pathlib,subprocess,time
b=pathlib.Path(inputs['base']);root=b/('audit-code-'+inputs['commit'][:12]);root.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');t.replace(p)
f=(b/'audit-operator.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (b/'job.id').read_text().strip()=='743485'
'''
def call(body, extra=None):
    values = dict(base=BASE, commit=COMMIT); values.update(extra or {})
    command = 'python3 -c ' + shlex.quote('inputs=' + repr(values) + '\n' + COMMON + '\n' + body)
    assert len(command.encode()) < 16000
    done = ctl.run(command, account='mou176', source_address=address, timeout=55)
    if done.returncode:
        raise RuntimeError('Audit outcome unconfirmed; no blind repeated step: ' + done.stdout.decode()[-2500:] + done.stderr.decode()[-1200:])
    return json.loads(done.stdout.decode().splitlines()[-1])
for i, start in enumerate(range(0, len(raw), 4096)):
    chunk = raw[start:start + 4096]
    call("p=root/('part-%04d'%inputs['index']);x=base64.b64decode(inputs['chunk']);assert hashlib.sha256(x).hexdigest()==inputs['sha256']\nif p.exists():assert p.read_bytes()==x\nelse:p.write_bytes(x)\nprint(json.dumps(dict(chunk=inputs['index'])))", dict(index=i, chunk=base64.b64encode(chunk).decode(), sha256=hashlib.sha256(chunk).hexdigest()))
body = r'''
data=b''.join((root/('part-%04d'%i)).read_bytes() for i in range(inputs['chunks']));assert hashlib.sha256(data).hexdigest()==inputs['sha256'];script=root/'audit.py'
if script.exists():assert script.read_bytes()==data
else:script.write_bytes(data);script.chmod(0o444)
pub=dict(commit=inputs['commit'],remote_sha=inputs['remote_sha'],path=str(script),bytes=len(data),sha256=inputs['sha256']);p=root/'source-publication.json'
if p.exists():assert read(p)==pub
else:atomic(p,pub)
receipt=b/'independent-cp-000001.json'
if receipt.exists():print(json.dumps(dict(status='EXISTING_INDEPENDENT_CP1_AUDIT_PRESERVED',result=read(receipt))))
else:
 assert not (b/'independent-cp1-step-intent.json').exists(),'Unknown prior audit step; inspect before another attempt'
 assert read(b/'progress.json')['cursor']>=1 and (b/'checkpoints/cursor-000001.pt').exists()
 state=subprocess.check_output(['squeue','-h','-j','743485','-o','%i|%u|%T|%N'],universal_newlines=True).strip();assert state.startswith('743485|mou176|RUNNING|'),state
 source=pathlib.Path(read(b/'source-publication.json')['code_root'])
 args=['srun','--jobid=743485','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G','--gres=gpu:0','--time=00:05:00','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','QWEN7B_CODE_ROOT='+str(source),'bash',str(source/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u',str(script),'--cursor','1']
 atomic(b/'independent-cp1-step-intent.json',dict(args=args,audit_source=pub,epoch=time.time(),extra_GPU_requested=0,existing_allocation='743485'))
 with (b/'independent-cp1-step.log').open('ab') as log:
  try:done=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=45)
  except subprocess.TimeoutExpired:atomic(b/'independent-cp1-step-unknown.json',dict(status='CPU_STEP_TIMEOUT_NO_BLIND_REPEAT',epoch=time.time()));raise
 atomic(b/'independent-cp1-step-exit.json',dict(returncode=done.returncode,epoch=time.time(),extra_GPU_requested=0));assert done.returncode==0,(b/'independent-cp1-step.log').read_text()[-2500:]
 result=read(receipt);assert result['status']=='EXACT_G20_DP_INDEPENDENT_FULL_SHA_AND_CPU_SIX_COMPONENT_RELOAD_PASSED'
 print(json.dumps(dict(status='ACTUAL_OWNED_CPU_STEP_CP1_AUDIT_PASS',result=result,step_exit=read(b/'independent-cp1-step-exit.json'),source=pub)))
'''
result = call(body, dict(chunks=(len(raw) + 4095) // 4096, sha256=hashlib.sha256(raw).hexdigest(), remote_sha=remote))
Path(__file__).with_name('first-checkpoint-independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
