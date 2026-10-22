"""Publish and source-stage root binder/control only; never invokes the control."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];ROOT=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-selected-trainB-root-control-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/selected-state-fixed-trainB2048-unified-20261023-v2'
FILES=['bind_submit_selected_trainB_once.py','test_bind_submit_selected_trainB_once.py','deploy_selected_trainB_root_control.py']
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,fcntl,time
b=pathlib.Path(v['base']);assert(b/'staged.json').is_file();code=b/('root-control-code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Root-control immutable stage differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
STAGE=r'''guard=(b/'root-control-stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);scorer=b/'code-ab8060368cc4';assert proof(b/'source-publication.json')['sha256']=='61eaa510f24f6830304ddcd81aed8090e75de7b9e7683404d0ee276f44dc067c'and proof(b/'scoring.sbatch')['sha256']=='07be86fa0369e79187d0193b42df12c616bf1febb309468611a6a29443b1b498'
for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(exist_ok=True,parents=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(q.read_bytes()for q in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 compile(raw,name,'exec')
 for q in parts:q.unlink()
once(code/'source-publication.json',dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files'],bound_scorer_publication=proof(b/'source-publication.json'),bound_scorer_entry=proof(b/'scoring.sbatch')))
control=code/'reproduction/operations/qwen7b_warmup/bind_submit_selected_trainB_once.py';entry=code/'root-bind-and-submit-once.sh';text='#!/usr/bin/env bash\nset -euo pipefail\nexport QWEN7B_CODE_ROOT='+str(scorer)+'\nexport CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1\n[[ "$(sha256sum '+str(control)+' | cut -d " " -f1)" == "'+proof(control)['sha256']+'" ]]\nexec bash '+str(scorer/'reproduction/operations/qwen7b_warmup/runtime.sh')+' -u '+str(control)+' "$@"\n'
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text);entry.chmod(0o555)
subprocess.check_call(['bash','-n',str(entry)])
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run(['bash',str(scorer/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.test_bind_submit_selected_trainB_once'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert r.returncode==0 and'Ran 5 tests'in r.stderr,r.stderr[-1200:]
r=subprocess.run(['bash',str(entry),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert r.returncode==0 and'--action {bind,submit}'in r.stdout,r.stderr[-1200:]
result=dict(status='FIXED_TRAIN_B_ROOT_BIND_SUBMIT_CONTROL_SOURCE_STAGED_NOT_EXECUTED',time=time.time(),source_commit=v['commit'],publication=proof(code/'source-publication.json'),entry=proof(entry),control=proof(control),scorer_publication=proof(b/'source-publication.json'),scorer_entry=proof(b/'scoring.sbatch'),CPU_tests_passed=5,actual_help_and_bash_syntax_passed=True,default_action='bind',explicit_submit_action_required=True,Zseed2_actual_CPU_coordinator_gate=True,durable_descriptor_and_submit_intent=True,same_name_command_owner_and_comment_reconciliation=True,unknown_submission_never_retried=True,actual_binding_not_invoked=True,model_or_checkpoint_payload_or_real_predictions_read=False,Slurm_submitted=False,GPU_started=False,final4096_read=False,training_changed=False);once(code/'staged.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(time=time.time(),source_commit=v['commit'],Slurm_submitted=False,actual_binding_not_invoked=True,final4096_read=False)
for name in('source-publication.json','staged.json'):
 if(code/name).exists():out[name]=dict(proof=proof(code/name),value=json.loads((code/name).read_bytes()))
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','status'),default='stage');a=p.parse_args();tip=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,tip],cwd=R);payload={P+n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in FILES};files={k:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for k,raw in payload.items()};spec=importlib.util.spec_from_file_location('root_trainB_control_private',ROOT/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  v=dict(base=BASE,commit=a.commit,branch=BRANCH,files=files,**extra);source='v='+repr(v)+'\n'+COMMON+body;compile(source,'trainB_control_stage','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-1800:]+r.stderr[-1800:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for index,start in enumerate(range(0,len(raw),6144)):
    chunk=raw[start:start+6144];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(STAGE)
 else:result=call(STATUS)
 dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/selected-state-fixed-trainB/root-control';dest.mkdir(exist_ok=True,parents=True);(dest/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(PART,STAGE,STATUS):compile(COMMON+body,'root_trainB_control_stage_static','exec')
if __name__=='__main__':main()
