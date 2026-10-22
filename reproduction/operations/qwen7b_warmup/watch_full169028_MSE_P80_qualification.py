"""Bounded local two-v3 source qualification, never submits or touches checkpoints.

Wait for actual producer acceptance, then run the published immutable qualifier,
record actual remote ready SHA and exit. Root alone may submit Slurm afterward.
"""
import base64,fcntl,json,os,pathlib,subprocess,sys,tempfile,time
REPO=pathlib.Path(__file__).resolve().parents[3];R=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5';BRANCH='codex/q7b-full169028-L15P80-MSE-ownG5-20261023';SOURCE='7a8ff6ae8a33324846d85f77ffc28e9ab699c028';DEPLOY='reproduction/operations/qwen7b_warmup/deploy_full169028_MSE_P80.py';CONTROL=REPO.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py'
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def publish(message):
 parent=subprocess.check_output(['git','rev-parse',BRANCH],cwd=REPO,text=True).strip();fd,index=tempfile.mkstemp(prefix='full-v3-ready-index-');os.close(fd);os.unlink(index);env=dict(os.environ,GIT_INDEX_FILE=index)
 try:
  subprocess.check_call(['git','read-tree',parent],cwd=REPO,env=env);names=['actual-two-source-stage-v3-handoff.json','qualification-watch-latest.json','actual-v3-ready-proofs.json','qualification-watch-failure.json'];paths=[R/n for n in names if(R/n).exists()];paths+=[R/seed/'actual-qualify.json'for seed in ['seed1','seed2']if(R/seed/'actual-qualify.json').exists()];subprocess.check_call(['git','add',*[str(p.relative_to(REPO))for p in paths]],cwd=REPO,env=env);tree=subprocess.check_output(['git','write-tree'],cwd=REPO,env=env,text=True).strip()
  if tree==subprocess.check_output(['git','rev-parse',parent+'^{tree}'],cwd=REPO,text=True).strip():return parent
  commit=subprocess.check_output(['git','commit-tree',tree,'-p',parent,'-m',message],cwd=REPO,env=env,text=True).strip();subprocess.check_call(['git','update-ref','refs/heads/'+BRANCH,commit,parent],cwd=REPO);subprocess.check_call(['git','push','origin',BRANCH],cwd=REPO);assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0]==commit;return commit
 finally:pathlib.Path(index).unlink(missing_ok=True)
def main():
 guard=(R/'qualification-watch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);code=subprocess.check_output(['git','show',SOURCE+':'+DEPLOY],cwd=REPO);packed=base64.b64encode(code).decode();driver="import base64;__file__="+repr(str(REPO/DEPLOY))+";exec(compile(base64.b64decode("+repr(packed)+"),__file__,'exec'))";started=time.time();last=None
 def invoke(action,seed):
  p=subprocess.run([sys.executable,'-c',driver,'--action',action,'--commit',SOURCE,'--replica',seed],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
  if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-2000:])
  return json.loads(p.stdout.splitlines()[-1])
 while time.time()-started<4*3600:
  try:
   state=invoke('status','seed1');phase='WAITING_ACTUAL_PHYSICAL_FIXTURE_ACCEPTANCE';results={}
   if state['fixture_complete_exists']and state['fixture_accepted_exists']:
    for seed in ['seed1','seed2']:
     q=invoke('qualify',seed);results[seed]=q
     if q['status']!='FULL169028_L15P80_MSE_OWNG5_ACCEPTED_FIXTURE_READY_NOT_SUBMITTED':phase=q['status'];break
    if len(results)==2 and all(v['status']=='FULL169028_L15P80_MSE_OWNG5_ACCEPTED_FIXTURE_READY_NOT_SUBMITTED'for v in results.values()):
     p=subprocess.run([sys.executable,str(CONTROL),str(R/'read-v3-ready-proofs.py'),'--read-only','--timeout','30'],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=80)
     if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-2000:])
     proofs=json.loads(p.stdout.splitlines()[-1]);assert all(v['ready']for v in proofs['replicas'].values());atomic(R/'actual-v3-ready-proofs.json',proofs);handoff=json.loads((R/'actual-two-source-stage-v3-handoff.json').read_text())
     for seed,v in proofs['replicas'].items():assert v['root']==handoff['replicas'][seed]['root'];handoff['replicas'][seed]['ready']=v['ready']
     handoff.update(status='TWO_IMMUTABLE_V3_FULL_MSE_SEED_INPUTS_QUALIFIED_ROOT_MAY_SUBMIT_ONCE',actual_ready_proofs_time=proofs['time']);atomic(R/'actual-two-source-stage-v3-handoff.json',handoff);event=dict(time=time.time(),status=handoff['status'],observer_pid=os.getpid(),source_commit=SOURCE,Slurm_submitted=False,final4096_read=False);atomic(R/'qualification-watch-latest.json',event);c=publish('Record actual physical/P20-qualified ready proofs for both full MSE seeds');print(json.dumps(dict(event=event,backup_commit=c)),flush=True);return
   event=dict(time=time.time(),status=phase,observer_pid=os.getpid(),source_commit=SOURCE,fixture_complete_exists=state['fixture_complete_exists'],fixture_accepted_exists=state['fixture_accepted_exists'],Slurm_submitted=False,final4096_read=False);atomic(R/'qualification-watch-latest.json',event)
   if phase!=last:print(json.dumps(event),flush=True);publish('Record full two-seed immutable source qualification waiting state');last=phase
  except BaseException as e:
   error=str(e);atomic(R/'qualification-watch-transport-latest.json',dict(time=time.time(),status='CONTROL_READ_UNKNOWN_READY_NOT_INFERRED',error=error,source_commit=SOURCE,remote_jobs_unmodified=True))
   if any(s in error.lower()for s in('connection closed','connection reset','timed out','broken pipe','connect to host')):time.sleep(75);continue
   raise
  time.sleep(75)
 raise RuntimeError('Bounded4h qualifier expired; no Slurm submission attempted')
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(R/'qualification-watch-failure.json',dict(time=time.time(),error=repr(e),source_commit=SOURCE,remote_jobs_unmodified=True,Slurm_submitted=False,final4096_read=False));raise
