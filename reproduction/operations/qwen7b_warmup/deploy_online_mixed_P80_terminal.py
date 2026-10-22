"""Stage isolated terminal-only CPU P80 epoch3 verifier; never starts it."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-P80-terminal-CPU-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-P80-terminal-CPU-acceptance-20261023-v1'
FILES=['accept_online_mixed_epoch_boundary.py','accept_online_mixed_P80_terminal.py','deploy_online_mixed_P80_terminal.py','native_mixed_group_classifier.py','peer_long_prefix_diagnostic.py','runtime.sh']
COMMON='''import pathlib,json,hashlib,base64,os,subprocess
b=pathlib.Path(v['base']);assert b.parent.is_dir();b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,value):
 raw=(json.dumps(value,sort_keys=True)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
COPY='''prior=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-P80-handoff-CPU-acceptance-20261023-v1/code-74178e17320d')
assert prior.is_dir()
for name,ref in v['files'].items():
 raw=base64.b64decode(v['payload'][name])if name in v['payload']else(prior/name).read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 for part in code.glob(name+'.part-*'):part.unlink()
'''
SEAL='''for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for item in parts:item.unlink()
publication=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files']);once(b/'source-publication.json',publication)
entry=code/'acceptance.sh';text='#!/usr/bin/env bash'+chr(10)+'set -euo pipefail'+chr(10)+'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 QWEN7B_CODE_ROOT='+repr(str(code))+chr(10)+'exec bash '+repr(str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'))+' -u -m reproduction.operations.qwen7b_warmup.accept_online_mixed_P80_terminal "$@"'+chr(10)
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');result=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c','import reproduction.operations.qwen7b_warmup.accept_online_mixed_epoch_boundary; import reproduction.operations.qwen7b_warmup.accept_online_mixed_P80_terminal; print("P80_TERMINAL_IMPORT_NO_CP_OPEN_PASS")'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert result.returncode==0 and'P80_TERMINAL_IMPORT_NO_CP_OPEN_PASS'in result.stdout,result.stderr[-1000:]
ready=dict(status='P80_TERMINAL_E3_CPU_VERIFIER_STAGED_NOT_STARTED',publication=proof(b/'source-publication.json'),validator=proof(code/'reproduction/operations/qwen7b_warmup/accept_online_mixed_P80_terminal.py'),reused_boundary_validator=proof(code/'reproduction/operations/qwen7b_warmup/accept_online_mixed_epoch_boundary.py'),CPU_entry=proof(entry),runtime=str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),code_root=str(code),CPU_import_passed=True,checkpoint_opened=False,Slurm_mutated=False,training_paused=False,actual_epoch_boundary_acceptance_not_claimed=True)
once(b/'staged.json',ready);print(json.dumps(ready))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args();actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,actual],cwd=R)
 payload={P+name:subprocess.check_output(['git','show',a.commit+':'+P+name],cwd=R)for name in FILES};files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for name,raw in payload.items()}
 spec=importlib.util.spec_from_file_location('P80_terminal_private_direct',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(script,**more):
  v=dict(base=BASE,commit=a.commit,branch=BRANCH,files=files,**more);packed=base64.b64encode(zlib.compress(('v='+repr(v)+'\n'+COMMON+script).encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;out=ctl.run(cmd,account='mou176',source_address=ip,timeout=45)
  if out.returncode:raise RuntimeError(out.stdout[-1200:]+out.stderr[-1000:])
  return json.loads(out.stdout.splitlines()[-1])
 minimal={name:base64.b64encode(raw).decode()for name,raw in payload.items()if name.endswith(('accept_online_mixed_P80_terminal.py','deploy_online_mixed_P80_terminal.py'))}
 ready=call(COPY+SEAL,payload=minimal);dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/P80-terminal-CPU-acceptance';dest.mkdir(exist_ok=True,parents=True);(dest/'actual-stage.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready))

for script in(COMMON,PART,COPY,SEAL):compile(script,'P80_handoff_stage','exec')
if __name__=='__main__':main()
