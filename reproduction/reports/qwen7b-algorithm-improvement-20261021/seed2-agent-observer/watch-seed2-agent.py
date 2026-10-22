"""Read-only exact second-seed jobs; actual updates and later epochs, no submits."""
import fcntl,json,os,pathlib,subprocess,sys,tempfile,time
B=pathlib.Path(__file__).resolve().parent;REPO=B.parents[3];CONTROL=REPO.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py';BRANCH='codex/q7b-leading-seed2-readonly-observer-20261023'
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def publish(message):
 parent=subprocess.check_output(['git','rev-parse',BRANCH],cwd=REPO,text=True).strip();fd,index=tempfile.mkstemp(prefix='seed2-agent-index-');os.close(fd);os.unlink(index);env=dict(os.environ,GIT_INDEX_FILE=index)
 try:
  subprocess.check_call(['git','read-tree',parent],cwd=REPO,env=env);files=[p for p in B.glob('agent-*')if p.suffix in('.json','.jsonl','.md')];subprocess.check_call(['git','add',*[str(p.relative_to(REPO))for p in files]],cwd=REPO,env=env);tree=subprocess.check_output(['git','write-tree'],cwd=REPO,env=env,text=True).strip()
  if tree==subprocess.check_output(['git','rev-parse',parent+'^{tree}'],cwd=REPO,text=True).strip():return parent
  c=subprocess.check_output(['git','commit-tree',tree,'-p',parent,'-m',message],cwd=REPO,env=env,text=True).strip();subprocess.check_call(['git','update-ref','refs/heads/'+BRANCH,c,parent],cwd=REPO);subprocess.check_call(['git','push','origin',BRANCH],cwd=REPO);assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0]==c;return c
 finally:
  if pathlib.Path(index).exists():pathlib.Path(index).unlink()
def main():
 guard=(B/'agent-seed2-watch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);previous={};verified=set();seen=None;start=time.time()
 while time.time()-start<8*3600:
  p=subprocess.run([sys.executable,str(CONTROL),str(B/'read-seed2-agent.py'),'--read-only','--timeout','25'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  if p.returncode:
   error=(p.stdout+p.stderr)[-2400:];atomic(B/'agent-seed2-transport-latest.json',dict(time=time.time(),error=error,current_remote_state_unknown=True,old_values_stale=True,remote_failure_not_inferred=True))
   if any(t in error for t in('Operation not permitted','Permission denied','network access is disabled','permission profile')):raise RuntimeError(error)
   if not any(t in error.lower()for t in('connection closed','connection reset','timed out','connect to host','broken pipe')):raise RuntimeError(error)
   time.sleep(55);continue
  x=json.loads(p.stdout.splitlines()[-1]);atomic(B/'agent-seed2-latest.json',x);summary=dict(time=x['time'],observer_pid=os.getpid(),queue=x['queue'],accounting=x['accounting'],cells={},final4096_read=False)
  for cell,v in x['cells'].items():
   ranks=[v.get('progress-rank-%d.json'%r,{})for r in(0,1)];tail=v['last_six_real_attempts'];steps=[r.get('actual_Adam_updates',r.get('cursor'))for r in tail];continuous=len(steps)==6 and all(steps[i]+1==steps[i+1]for i in range(5));old=previous.get(cell);actual=[r.get('actual_Adam_updates',r.get('cursor',0))for r in ranks];controller=v.get('controller.json',{})
   if cell not in verified and old and continuous and min(actual)>=6 and all(ranks[i].get('cursor',0)>old[i].get('cursor',0)for i in(0,1)):
    verified.add(cell);atomic(B/('agent-seed2-'+cell+'-six-updates-verified.json'),dict(time=x['time'],prior_ranks=old,current_ranks=ranks,last_six_real_updates=tail,actual_two_fresh_samples_and_six_updates_and_next=True,controller=controller,final4096_read=False));print(json.dumps(dict(event='SECOND_TRAINING_SEED_REAL_UPDATE_PROGRESS_VERIFIED',cell=cell,time=x['time'],actual_updates=actual)),flush=True)
   if all(ranks):previous[cell]=ranks
   summary['cells'][cell]=dict(job_id=v['job_id'],actual_rank_epochs=[r.get('epoch')for r in ranks],actual_rank_cursors=[r.get('cursor')for r in ranks],actual_Adam_updates=actual,real_six_update_sequence=steps,real_six_updates_continuous=continuous,verified_two_samples_plus_six_updates=cell in verified,initial_head_sha256=controller.get('initial_head_sha256'),init_seed=controller.get('init_seed'),train_plan_seed=controller.get('train_plan_seed'),diagnostic_seed=controller.get('diagnostic_seed'),evaluation_epochs=sorted(v['epoch_metrics']),failures=v['failures'],checkpoint=v.get('latest.json'),complete=v.get('complete.json'))
  with(B/'agent-seed2-observations.jsonl').open('a')as f:f.write(json.dumps(summary)+'\n')
  atomic(B/'agent-seed2-watch-latest.json',summary);key={k:(tuple(v['evaluation_epochs']),tuple(sorted(v['failures'])),v['verified_two_samples_plus_six_updates'],bool(v['complete']))for k,v in summary['cells'].items()}
  if key!=seen:
   c=publish('Record exact two-seed real progress and actual epoch receipts');seen=key;print(json.dumps(dict(event='SECOND_SEED_ACTUAL_PHASE_PUBLISHED',commit=c,summary=summary)),flush=True)
  if any(v['failures']for v in summary['cells'].values())or any(s in x['accounting']['stdout']for s in('|FAILED|','|CANCELLED|','|TIMEOUT|','|OUT_OF_MEMORY|','|NODE_FAIL|')):print(json.dumps(dict(event='SECOND_SEED_ACTUAL_FAILURE_ROOT_HANDLES',summary=summary)),flush=True);return
  if all(v['complete']for v in summary['cells'].values())and all(any(line.split('|')[:3]==[v['job_id'],'COMPLETED','0:0']for line in x['accounting']['stdout'].splitlines())for v in summary['cells'].values()):print(json.dumps(dict(event='BOTH_SECOND_SEED_PRODUCERS_COMPLETE_CPU_ACCEPTANCE_ROOT_OWNED',summary=summary)),flush=True);return
  time.sleep(55)
 raise RuntimeError('Bounded8h observer expired; remote jobs unmodified')
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(B/'agent-seed2-watch-failure.json',dict(time=time.time(),observer_only=True,error=repr(e),remote_jobs_unmodified=True));raise
