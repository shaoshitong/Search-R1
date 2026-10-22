"""Publish/stage/start a durable serial driver of immutable76637 CPU entries."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-selected-CPU-coordination-20261023'
REMOTE='''import pathlib,json,hashlib,base64,zlib,subprocess,os,time,fcntl
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');assert A.is_dir()
b=A/'selected-checkpoint-CPU-coordinator-20261023-v1';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable coordinator config/source changed'
 else:p.write_bytes(raw);p.chmod(0o444)
def ticks(pid):
 try:return pathlib.Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except (FileNotFoundError,ProcessLookupError):return None
raw=zlib.decompress(base64.b64decode(v['source']));assert hashlib.sha256(raw).hexdigest()==v['sha256'];compile(raw,'coordinator','exec');worker=code/'selected_checkpoint_cpu_coordinator.py';once(worker,raw)
old=A/'matched-window-terminal-CPU-acceptance-20261023-v1';entries=old/'code-76637b36e33a';assert proof(entries/'accept_matched_window_checkpoint.py')['sha256']=='d0fedb7c71ac44eeaa14ec3c9418b2fdfdaed43e8534d29b59ad83f5e8ef3fcc'
targets=[]
for suite,cell,epoch in [('online-packed-global64-window2018-L15P80-8192-20261023-v1','current64',5),('online-packed-matched-L15P80-8192-20261023-v1','mse-M5',7),('online-packed-global64-window2018-L15P80-8192-20261023-v1','window2018',10)]:
 base=A/suite/cell;assert base.is_dir()and(base/'root-submission.json').is_file();targets.append(dict(base=str(base),entry=proof(entries/(cell+'.sh')),epoch=epoch,accepted=str(old/cell/('epoch-%02d'%epoch)/'accepted.json')))
config=code/'config.json';once(config,(json.dumps(dict(out=str(b),targets=targets),sort_keys=True)+'\\n').encode());receipt=dict(status='DURABLE_SERIAL_EXISTING_CPU_VALIDATORS_STAGED',source_commit=v['commit'],source=proof(worker),config=proof(config),shared_mutex_owned_by_each_child=True,Slurm_mutated=False,final4096_read=False)
if v['action']=='start':
 guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 controller=read(b/'controller.json')if(b/'controller.json').exists()else None
 if controller and ticks(controller['pid'])==controller['start_ticks']:
  receipt.update(status='EXISTING_DURABLE_COORDINATOR_REUSED',controller=controller)
 elif(b/'complete.json').exists():receipt.update(status='EXISTING_ALL_THREE_SELECTED_ACCEPTANCES_REUSED',complete=read(b/'complete.json'))
 else:
  assert not(b/'failure.json').exists(),'Prior coordinator failure requires diagnosis, not blind restart'
  prior=read(b/'launch.json')if(b/'launch.json').exists()else None
  assert not prior or ticks(prior['pid'])!=prior['start_ticks'],'Existing launch process alive; do not duplicate'
  # A timed-out SSH call is not evidence of failure. Inspect actual child identities.
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    if p.stat().st_uid!=os.getuid():continue
    argv=(p/'cmdline').read_bytes().split(bytes([0]))
    if str(entries/'accept_matched_window_checkpoint.py').encode()in argv:raise RuntimeError('Existing immutable76637 validator still live; wait rather than duplicate')
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  with(b/'worker.log').open('ab')as stream:child=subprocess.Popen(['python3','-u',str(worker),'--config',str(config),'--config-sha',proof(config)['sha256']],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  launched=dict(pid=child.pid,start_ticks=ticks(child.pid),time=time.time(),config=proof(config));(b/'launch.json').write_text(json.dumps(launched,sort_keys=True)+'\\n');receipt.update(status='DURABLE_SERIAL_COORDINATOR_STARTED_NOT_ACCEPTED',launch=launched)
for n in('controller.json','progress.json','active-child.json','results-so-far.json','complete.json','failure.json'):
 if(b/n).exists():receipt[n]=read(b/n)
print(json.dumps(receipt))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','start','status'),default='stage');a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);raw=subprocess.check_output(['git','show',a.commit+':'+P+'selected_checkpoint_cpu_coordinator.py'],cwd=R);v=dict(commit=a.commit,source=base64.b64encode(zlib.compress(raw)).decode(),sha256=hashlib.sha256(raw).hexdigest(),action=a.action);script='v='+repr(v)+'\n'+REMOTE;compile(script,'remote_coordinator','exec');command='python3 -c '+shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(script.encode())).decode())+')))');assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=ctl.run(command,account='mou176',source_address=address,timeout=35)
 if r.returncode:raise RuntimeError(repr(r.stdout[-2000:])+repr(r.stderr[-2000:]))
 result=json.loads(r.stdout);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/matched-window-CPU-acceptance';out.mkdir(exist_ok=True,parents=True);(out/('actual-selected-coordinator-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'remote_coordinator','exec')
if __name__=='__main__':main()
