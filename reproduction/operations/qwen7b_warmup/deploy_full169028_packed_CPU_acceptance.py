"""Start or observe one locked CPU validator; never alters a GPU Slurm job."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-packed-cpu-acceptance-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-full169028-variance-CPU-acceptance-20261022-v1'
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,fcntl,time
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def atomic(p,x):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,sort_keys=True)+chr(10));q.replace(p)
def live():
 p=b/'controller.json'
 if not p.exists():return False
 c=read(p);proc=pathlib.Path('/proc/%d'%c['pid'])
 return proc.exists()and'accept_full169028_packed_variance.py'in(proc/'cmdline').read_bytes().decode(errors='replace')and(proc/'stat').read_text().rsplit(')',1)[1].split()[19]==c['start_ticks']
'''
STAGE='''def stage():
 raw=base64.b64decode(v['source']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];code=b/('validator-'+v['commit'][:12]);code.mkdir(exist_ok=True);path=code/'accept_full169028_packed_variance.py'
 if path.exists():assert path.read_bytes()==raw
 else:path.write_bytes(raw);path.chmod(0o444)
 compile(raw,str(path),'exec');suite=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1');pub=read(suite/'source-publication.json');runtime=pathlib.Path(pub['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh'
 for name in('runtime.sh','packed_variance_probe.py'):
  rel='reproduction/operations/qwen7b_warmup/'+name;assert proof(pathlib.Path(pub['code_root'])/rel)==dict(path=str(pathlib.Path(pub['code_root'])/rel),**pub['files'][rel])
 env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 entry=code/'acceptance.sh';text='#!/usr/bin/env bash'+chr(10)+'set -euo pipefail'+chr(10)+'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 QWEN7B_CODE_ROOT='+repr(pub['code_root'])+chr(10)+'exec bash '+repr(str(runtime))+' -u '+repr(str(path))+' --source-sha '+v['sha256']+chr(10)
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)])
 ready=dict(status='THREE_FULL169028_VARIANCE_TERMINAL_ONLY_CPU_ACCEPTOR_STAGED_NOT_STARTED',source=proof(path),entry=proof(entry),commit=v['commit'],source_publication=proof(suite/'source-publication.json'),terminal_job_required='COMPLETED_0_0',running_checkpoint_opened=False,CPU_only=True,no_Slurm_mutation=True)
 publication=code/'staged.json'
 if publication.exists():assert read(publication)==ready
 else:atomic(publication,ready)
 return path,runtime,env,ready
'''
START='''guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
if live():print(json.dumps(dict(status='EXISTING_ACTUAL_CPU_VALIDATOR_REUSED',controller=read(b/'controller.json'))))
else:
 lock=(b/'acceptance.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
 path,runtime,env,ready=stage()
 if(b/'run-complete.json').exists():
  old=read(b/'run-complete.json');archive=b/('prior-run-complete-'+hashlib.sha256((b/'run-complete.json').read_bytes()).hexdigest()[:12]+'.json')
  if not archive.exists():atomic(archive,old)
 log=(b/'acceptance.log').open('ab');p=subprocess.Popen(['bash',str(runtime),'-u',str(path),'--source-sha',v['sha256']],env=env,stdout=log,stderr=log,start_new_session=True)
 atomic(b/'launch-receipt.json',dict(pid=p.pid,source=proof(path),commit=v['commit'],source_publication=ready['source_publication'],CPU_only=True,no_Slurm_mutation=True,time=time.time()))
 print(json.dumps(dict(status='LOCKED_SEQUENTIAL_CPU_VALIDATOR_STARTED_NOT_ACCEPTED',launch=read(b/'launch-receipt.json'))))
'''
STATUS='''result=dict(time=time.time(),actual_validator_identity_live=live())
for name in('controller.json','progress.json','run-complete.json','failure.json','launch-receipt.json'):
 if(b/name).exists():result[name[:-5]]=read(b/name)
result['accepted']={loss:read(b/loss/'accepted.json')for loss in('mse','mae','z_mse_anchor')if(b/loss/'accepted.json').exists()}
if(b/'acceptance.log').exists():
 with(b/'acceptance.log').open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-1800));result['log_tail']=f.read().decode(errors='replace')
print(json.dumps(result))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','start','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0]
 subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,actual],cwd=R)
 raw=subprocess.check_output(['git','show',a.commit+':'+P+'accept_full169028_packed_variance.py'],cwd=R)
 spec=importlib.util.spec_from_file_location('pilot_accept_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 v=dict(base=BASE,commit=a.commit,source=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest());body=STATUS if a.action=='status'else STAGE+(START if a.action=='start'else'print(json.dumps(stage()[3]))\n');script='v='+repr(v)+'\n'+COMMON+body;compile(script,'CPU_acceptance_remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=35)
 if r.returncode:raise RuntimeError(r.stdout[-1600:]+r.stderr[-1000:])
 result=json.loads(r.stdout.splitlines()[-1]);dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance-full169028-CPU-acceptance';dest.mkdir(exist_ok=True,parents=True);(dest/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


for script in(COMMON,STAGE,START,STATUS):compile(script,'CPU_acceptance_static','exec')
if __name__=='__main__':main()
