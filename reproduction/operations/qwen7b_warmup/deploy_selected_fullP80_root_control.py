"""Publish/stage three CPU control files only; never actually bind/freeze/submit."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-fullP80-selected-root-control-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-selected-root-control-20261023-v1'
FILES=('control_selected_fullP80_once.py','test_control_selected_fullP80_once.py','deploy_selected_fullP80_root_control.py')
RULE='reproduction/reports/qwen7b-algorithm-improvement-20261021/frozen-full169028-MSE-final4096-scoring/PREDECLARED-FINAL-CONFIRMATION.json'
RULE_SHA='8f4fead8f40a7e14f7ddaffb2950bea3f3e405d435c544388519c507e7ce3a27'
COMMON='''import pathlib,json,base64,hashlib,subprocess,os,fcntl
b=pathlib.Path(v['base']);assert b.parent.is_dir(),'Actual existing mount required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=pathlib.Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable source-stage artifact differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];once(p,raw);print(json.dumps(dict(stored=True)))
'''
STAGE='''with(b/'source-stage.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,ref in v['files'].items():
  p=code/name;parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];compile(raw,name,'exec');once(p,raw);p.chmod(0o444)
  for part in parts:part.unlink()
 rule=base64.b64decode(v['rule']);assert hashlib.sha256(rule).hexdigest()==v['rule_sha'];once(b/'PREDECLARED-FINAL-CONFIRMATION.json',rule)
 publication=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files'],prospective_confirmation_rule=proof(b/'PREDECLARED-FINAL-CONFIRMATION.json'));once(b/'source-publication.json',(json.dumps(publication,sort_keys=True,separators=(',',':'))+chr(10)).encode())
 # Use the already published training environment without importing its model.
 a=b.parent;training=a/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3';training_pub=json.loads((training/'source-publication.json').read_bytes());runtime=pathlib.Path(training_pub['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh'
 entry=b/'control.sh';text=chr(10).join(['#!/usr/bin/env bash','set -euo pipefail','export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 QWEN7B_CODE_ROOT='+shlex.quote(str(code)),'exec bash '+shlex.quote(str(runtime))+' -u '+shlex.quote(str(code/'reproduction/operations/qwen7b_warmup/control_selected_fullP80_once.py'))+' "$@"','']);once(entry,text.encode());subprocess.check_call(['bash','-n',str(entry)])
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 test=subprocess.run(['bash',str(runtime),'-m','reproduction.operations.qwen7b_warmup.test_control_selected_fullP80_once'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert test.returncode==0 and'Ran 5 tests'in test.stderr,test.stderr[-2000:]
 help_result=subprocess.run(['bash',str(entry),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert help_result.returncode==0 and all(x in help_result.stdout for x in('bind-audit','submit-audit','freeze-final','submit-final')),help_result.stderr[-1500:]
 assert not any((b/n).exists()for n in('audit-bound.json','candidate-freeze.json','audit-submit-intent.json','final-submit-intent.json','audit-submission.json','final-submission.json')),'Stage/help must not bind, freeze or submit'
 result=dict(status='FULLP80_SELECTED_ROOT_CONTROL_SOURCE_STAGED_ONLY',source_commit=v['commit'],publication=proof(b/'source-publication.json'),entry=proof(entry),controller=proof(code/'reproduction/operations/qwen7b_warmup/control_selected_fullP80_once.py'),code_root=str(code),CPU_metadata_tests=5,CPU_metadata_tests_passed=True,help_passed=True,bash_syntax_passed=True,prospective_rule=proof(b/'PREDECLARED-FINAL-CONFIRMATION.json'),real_binding_attempted=False,candidate_frozen=False,Slurm_submitted=False,checkpoint_tensor_or_final_data_opened=False)
 once(b/'staged.json',(json.dumps(result,sort_keys=True,separators=(',',':'))+chr(10)).encode());print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R)
 sources={P+n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in FILES};files={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in sources.items()};rule=subprocess.check_output(['git','show',a.commit+':'+RULE],cwd=R);assert hashlib.sha256(rule).hexdigest()==RULE_SHA
 spec=importlib.util.spec_from_file_location('private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  v=dict(base=BASE,commit=a.commit,branch=BRANCH,files=files,rule=base64.b64encode(rule).decode(),rule_sha=RULE_SHA,**extra);script='v='+repr(v)+'\nimport shlex\n'+COMMON+body;compile(script,'rootcontrol-stage','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;reply=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if reply.returncode:raise RuntimeError(reply.stdout[-1000:]+reply.stderr[-2400:])
  return json.loads(reply.stdout.strip().splitlines()[-1])
 for name,raw in sources.items():
  for index,offset in enumerate(range(0,len(raw),6144)):
   chunk=raw[offset:offset+6144];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
 result=call(STAGE);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/fullP80-selected-root-control';out.mkdir(exist_ok=True,parents=True);(out/'actual-stage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(PART,STAGE):compile(COMMON+body,'rootcontrol-stage','exec')
if __name__=='__main__':main()
