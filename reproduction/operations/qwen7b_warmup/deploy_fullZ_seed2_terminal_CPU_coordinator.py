"""Publish/start only749935 metadata wait; never submit Slurm or load a CP."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/q7b-full169028-Z-seed2-CPU-acceptance-20261023';NAME='fullZ_seed2_terminal_CPU_coordinator.py'
REMOTE='''import pathlib,json,base64,zlib,hashlib,fcntl,os,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');assert A.is_dir();b=A/'fullZ-seed2-749935-terminal-CPU-coordinator-20261023-v1';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);base=A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1';accept=A/'full169028-Z-seed2-terminal-CPU-acceptance-20261023-v1';entry=accept/'code-0f051920d4de/seed19137.sh';validator=entry.parent/'accept_full169028_Z_seed2_checkpoint.py'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable source/config differs'
 else:p.write_bytes(raw);p.chmod(0o444)
def ticks(pid):
 try:return pathlib.Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
assert str(read(base/'root-submission.json')['job_id'])=='749935';assert proof(entry)['sha256']=='57c9c889ed80a7153da71bcc58bc67fa15e705aec86d0219128035de414e18dd';assert proof(validator)['sha256']=='682d8d6c8be4de3174c70acf9cb1287088137b27563b9f50036918254e0b6a44'
raw=zlib.decompress(base64.b64decode(v['source']));assert hashlib.sha256(raw).hexdigest()==v['sha256'];compile(raw,'Z_coordinator','exec');worker=code/v['name'];once(worker,raw);config=code/'config.json';once(config,(json.dumps(dict(out=str(b),base=str(base),job_id='749935',entry=proof(entry),validator=proof(validator),accepted=str(accept/'epoch10/accepted.json')),sort_keys=True)+chr(10)).encode());receipt=dict(status='FULL_Z_SEED2_DURABLE_TERMINAL_CPU_COORDINATOR_STAGED',source_commit=v['commit'],worker=proof(worker),config=proof(config),model_or_checkpoint_read=False,Slurm_mutated=False,final4096_read=False)
if v['action']=='start':
 guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=read(b/'controller.json')if(b/'controller.json').exists()else None;launch=read(b/'launch.json')if(b/'launch.json').exists()else None
 if controller and ticks(controller['pid'])==controller['start_ticks']:receipt.update(status='EXISTING_COORDINATOR_REUSED',controller=controller)
 elif launch and ticks(launch['pid'])==launch['start_ticks']:receipt.update(status='EXISTING_LAUNCH_REUSED',launch=launch)
 elif(b/'complete.json').exists():receipt.update(status='EXISTING_CPU_ACCEPTANCE_REUSED',complete=read(b/'complete.json'))
 else:
  assert not(b/'failure.json').exists(),'Existing failure requires diagnosis, not restart'
  unknown=[]
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    if p.stat().st_uid!=os.getuid():continue
    argv=(p/'cmdline').read_bytes().split(bytes([0]))
    if any(x in argv for x in(str(worker).encode(),str(validator).encode(),str(entry).encode())):unknown.append(dict(pid=int(p.name),start_ticks=ticks(p.name)))
   except(FileNotFoundError,PermissionError,ProcessLookupError):pass
  assert not unknown,'Existing actual coordinator/validator identities must be resolved first: '+repr(unknown)
  with(b/'worker.log').open('ab')as stream:child=subprocess.Popen(['python3','-u',str(worker),'--config',str(config),'--config-sha',proof(config)['sha256']],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,close_fds=True)
  launched=dict(pid=child.pid,start_ticks=ticks(child.pid),time=time.time(),config=proof(config));(b/'launch.json').write_text(json.dumps(launched,sort_keys=True)+chr(10));receipt.update(status='DURABLE_Z_SEED2_COORDINATOR_STARTED_NOT_ACCEPTED',launch=launched)
for name in('controller.json','progress.json','active-child.json','complete.json','failure.json'):
 if(b/name).exists():receipt[name]=read(b/name)
print(json.dumps(receipt))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=['stage','start','status'],default='stage');a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);raw=subprocess.check_output(['git','show',a.commit+':'+P+NAME],cwd=R);v=dict(commit=a.commit,name=NAME,source=base64.b64encode(zlib.compress(raw)).decode(),sha256=hashlib.sha256(raw).hexdigest(),action=a.action);script='v='+repr(v)+'\n'+REMOTE;compile(script,'remoteZcoordinator','exec');command='python3 -c '+shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(script.encode())).decode())+')))');assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();reply=ctl.run(command,account='mou176',source_address=ip,timeout=35)
 if reply.returncode:raise RuntimeError(reply.stdout[-1800:]+reply.stderr[-1800:])
 result=json.loads(reply.stdout);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-Z-seed2-CPU-acceptance';(out/('actual-coordinator-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'remoteZcoordinator','exec')
if __name__=='__main__':main()
