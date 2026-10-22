"""Stage two exact fullMSE terminal workers; Root alone calls start."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/q7b-fullP80-MSE-terminal-CPU-coordinators-20261023';NAME='fullP80_MSE_terminal_CPU_coordinator.py'
REMOTE='''import pathlib,json,base64,zlib,hashlib,fcntl,os,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');assert A.is_dir();root=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';root.mkdir(exist_ok=True);code=root/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);accept=A/'full169028-L15P80-MSE-terminal-CPU-acceptance-20261023-v1';oldcode=accept/'code-693a81dfbbae';validator=oldcode/'accept_full169028_L15P80_MSE_checkpoint.py'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable source/config changed'
 else:p.write_bytes(raw);p.chmod(0o444)
def ticks(pid):
 try:return pathlib.Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
assert proof(validator)['sha256']=='40e308e6a3c65bc0132c7b249939059775f4513d867576e8df74fa7aabc61448';raw=zlib.decompress(base64.b64decode(v['source']));assert hashlib.sha256(raw).hexdigest()==v['sha256'];compile(raw,'fullMSE_CPU_coordinator','exec');worker=code/v['name'];once(worker,raw);results=[]
for seed,plan,job,entrysha in [(9137,2026102227,'750189','9e03c902ac561917bb71fe839c37f47386659a03792ac3ed05c3eb1a64383e27'),(19137,2026102302,'750190','f648fc7fc8159f3677ca91bb52bc06f9c8fc163a22798c2a326fe7acbf9ce577')]:
 if v['seed']is not None and v['seed']!=seed:continue
 base=A/('full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%(seed,plan));assert str(read(base/'root-submission.json')['job_id'])==job;out=root/('job-'+job);out.mkdir(exist_ok=True);entry=oldcode/('seed%d.sh'%seed);assert proof(entry)['sha256']==entrysha;config=code/('job-'+job+'.json');once(config,(json.dumps(dict(out=str(out),base=str(base),job_id=job,entry=proof(entry),validator=proof(validator),accepted=str(accept/base.name/'epoch-10/accepted.json')),sort_keys=True)+chr(10)).encode());coordinator_entry=code/('job-'+job+'.sh');text=chr(10).join(['#!/usr/bin/env bash','set -euo pipefail','export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1','exec python3 -u '+repr(str(worker))+' --config '+repr(str(config))+' --config-sha '+proof(config)['sha256'],'']);once(coordinator_entry,text.encode());subprocess.check_call(['bash','-n',str(coordinator_entry)]);r=dict(status='FULL_MSE_DURABLE_TERMINAL_CPU_COORDINATOR_STAGED_NOT_STARTED',job_id=job,init_seed=seed,train_plan_seed=plan,source_commit=v['commit'],worker=proof(worker),config=proof(config),coordinator_entry=proof(coordinator_entry),existing_CPU_entry=proof(entry),existing_CPU_validator=proof(validator),accepted_path=str(accept/base.name/'epoch-10/accepted.json'),shared_CPU_lock_owned_inside_immutable693_child=True,model_or_checkpoint_read=False,Slurm_mutated=False,final4096_read=False)
 if v['action']=='start':
  launch_guard=(out/'launch.lock').open('a+');fcntl.flock(launch_guard,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=read(out/'controller.json')if(out/'controller.json').exists()else None;launch=read(out/'launch.json')if(out/'launch.json').exists()else None
  if controller and ticks(controller['pid'])==controller['start_ticks']:r.update(status='EXISTING_COORDINATOR_REUSED',controller=controller)
  elif launch and ticks(launch['pid'])==launch['start_ticks']:r.update(status='EXISTING_LAUNCH_REUSED',launch=launch)
  elif(out/'complete.json').exists():r.update(status='EXISTING_EXACT_TERMINAL_CPU_RESULT_REUSED',complete=read(out/'complete.json'))
  else:
   assert not(out/'failure.json').exists(),'Existing failure must be diagnosed, not restarted';unknown=[]
   for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
     if p.stat().st_uid!=os.getuid():continue
     argv=(p/'cmdline').read_bytes().split(bytes([0]))
     if(str(worker).encode()in argv and str(config).encode()in argv)or(str(validator).encode()in argv and str(base).encode()in argv)or str(coordinator_entry).encode()in argv:unknown.append(dict(pid=int(p.name),start_ticks=ticks(p.name)))
    except(FileNotFoundError,PermissionError,ProcessLookupError):pass
   assert not unknown,'Existing same-job coordinator/CPU reader identity unresolved: '+repr(unknown)
   with(out/'worker.log').open('ab')as stream:child=subprocess.Popen(['bash',str(coordinator_entry)],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,close_fds=True)
   launched=dict(pid=child.pid,start_ticks=ticks(child.pid),config=proof(config),time=time.time());(out/'launch.json').write_text(json.dumps(launched,sort_keys=True)+chr(10));r.update(status='DURABLE_FULL_MSE_TERMINAL_CPU_COORDINATOR_STARTED_NOT_ACCEPTED',launch=launched)
 for name in('launch.json','controller.json','progress.json','active-child.json','complete.json','failure.json'):
  if(out/name).exists():r[name]=read(out/name)
 results.append(r)
receipt=dict(status='FULL_MSE_TWO_TERMINAL_CPU_COORDINATORS_SOURCE_STAGE',source_commit=v['commit'],results=results,checkpoint_or_model_read=False,Slurm_mutated=False,final4096_read=False)
if v['action']=='stage':once(code/('staged-'+str(v['seed'])+'.json'),(json.dumps(receipt,sort_keys=True)+chr(10)).encode())
print(json.dumps(receipt))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--seed',type=int,choices=(9137,19137));p.add_argument('--action',choices=('stage','start','status'),default='stage');a=p.parse_args()
 if a.action=='start'and a.seed is None:p.error('Root starts one explicit seed/job at a time')
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);raw=subprocess.check_output(['git','show',a.commit+':'+P+NAME],cwd=R);v=dict(commit=a.commit,name=NAME,source=base64.b64encode(zlib.compress(raw)).decode(),sha256=hashlib.sha256(raw).hexdigest(),action=a.action,seed=a.seed);script='v='+repr(v)+'\n'+REMOTE;compile(script,'fullMSEcoordinator_stage','exec');command='python3 -c '+shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(script.encode())).decode())+')))');assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();reply=ctl.run(command,account='mou176',source_address=ip,timeout=35)
 if reply.returncode:raise RuntimeError(reply.stdout[-1800:]+reply.stderr[-1800:])
 result=json.loads(reply.stdout);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/fullP80-MSE-terminal-CPU-coordinators';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+('-'+str(a.seed)if a.seed else'')+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'fullMSEcoordinator_stage','exec')
if __name__=='__main__':main()
