"""Read-only full MSE follow-through, small epoch publication, terminal CPU gate.

No GPU, Slurm, parameter or final-holdout actions. Uses the already published
private control/poller and terminal-only shared locked CPU acceptor.
"""
import fcntl,json,os,pathlib,subprocess,sys,tempfile,time
B=pathlib.Path(__file__).resolve().parent;REPO=B.parents[3]
BRANCH='codex/q7b-full169028-mse-observer-20261023'
ACCEPT_COMMIT='799e75f3'

def atomic(p,v):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def publish(message):
 parent=subprocess.check_output(['git','rev-parse',BRANCH],cwd=REPO,text=True).strip();fd,index=tempfile.mkstemp(prefix='mse-epoch-index-');os.close(fd);os.unlink(index);env=dict(os.environ,GIT_INDEX_FILE=index)
 try:
  subprocess.check_call(['git','read-tree',parent],cwd=REPO,env=env)
  files=[p for p in B.glob('agent-*')if p.suffix in('.json','.jsonl','.md')]+[B/'MSE-EPOCHS.md']
  subprocess.check_call(['git','add',*[str(p.relative_to(REPO))for p in files if p.exists()]],cwd=REPO,env=env)
  tree=subprocess.check_output(['git','write-tree'],cwd=REPO,env=env,text=True).strip();oldtree=subprocess.check_output(['git','rev-parse',parent+'^{tree}'],cwd=REPO,text=True).strip()
  if tree==oldtree:return parent
  c=subprocess.check_output(['git','commit-tree',tree,'-p',parent,'-m',message],cwd=REPO,text=True).strip();subprocess.check_call(['git','update-ref','refs/heads/'+BRANCH,c,parent],cwd=REPO)
  subprocess.check_call(['git','push','origin',BRANCH],cwd=REPO);remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0]
  if remote!=c:raise RuntimeError('New epoch source/summary publication not remotely verified')
  return c
 finally:
  if pathlib.Path(index).exists():pathlib.Path(index).unlink()
def report(x):
 rows=['# Full169028 MSE: actual fixed-pool epoch metrics','', 'Existing MSE job749568 only. Each epoch uses the full169028-question pool, per-epoch independent shuffled order and own randomM3..10 trajectory subset. No final4096 read. Fixed train2048 and viewed independent validation2048 are held constant across epochs; raw updates EMA.9995. Variance loss is not binary-Mixed BCE. The fixed G5 classification threshold0.08 was specified before fitting and is not tuned.','', '| Epoch | Pool | State | Variance MSE | MAE | Spearman | Mixed BA % (G5) | FPR % (G5) | Top512 Mixed |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
 for key in sorted(k for k in x if k.startswith('evaluation-epoch-')):
  e=x[key]
  for pool in('fixed_train2048_G5','fixed_validation2048_G5','fixed_validation2048_M3','fixed_validation2048_M5','fixed_validation2048_M10'):
   for state in('raw','EMA'):
    v=e['metrics'][pool][state];g=v.get('predeclared_G5_classification_diagnostic',{});ba='—'if not g else'%.2f'%(100*g['balanced_accuracy']);fp='—'if not g else'%.2f'%(100*g['nonMixed_false_positive_rate']);t=v['Top512'];rows.append('| %d | %s | %s | %.6f | %.6f | %.4f | %s | %s | %d/512 (%.2f%%) |'%(e['epoch'],pool,state,v['variance_MSE'],v['variance_MAE'],v['variance_Spearman'],ba,fp,t['Mixed'],100*t['Mixed_fraction']))
 rows+=['','## Actual epoch coverage','', '| Epoch | Questions | Unique | Trajectory visits | Plan SHA256 |','|---:|---:|---:|---:|---|']
 for key in sorted(k for k in x if k.startswith('coverage-epoch-')):
  c=x[key];rows.append('| %d | %d | %d | %d | `%s` |'%(c['epoch'],c['questions'],c['unique_questions'],c['trajectory_visits'],c['plan_sha256']))
 rows+=['', 'Latest actual sample epoch%.6f. Each metric is copied from the completed epoch evaluation receipt; an epoch absent from the table is not claimed complete. Terminal model/Adam/RNG acceptance is separate.'%x['time']]
 (B/'MSE-EPOCHS.md').write_text('\n'.join(rows)+'\n')
def run():
 guard=(B/'agent-mse-watch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);seen=set();start=time.time();heartbeat=0
 while time.time()-start<8*3600:
  p=subprocess.run([sys.executable,str(B/'poll-mse-agent.py')],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-2400:])
  summary=json.loads(p.stdout.splitlines()[-1]);x=json.loads((B/'agent-observations-mse-latest.json').read_text());epochs={k for k in x if k.startswith('evaluation-epoch-')};report(x)
  current=dict(time=x['time'],job='749568',observer_pid=os.getpid(),latest_rank_cursors=[x.get('progress-rank-%d.json'%r,{}).get('cursor')for r in(0,1)],latest_rank_epochs=[x.get('progress-rank-%d.json'%r,{}).get('epoch')for r in(0,1)],completed_evaluation_epochs=sorted(epochs),failures=x['failures'],Slurm=x['accounting']['stdout'],final_read=False)
  atomic(B/'agent-mse-watch-latest.json',current)
  if epochs!=seen:
   c=publish('Record MSE actual fixed-pool evaluations '+','.join(sorted(epochs)));seen=epochs;print(json.dumps(dict(event='NEW_ACTUAL_EPOCH_METRICS_PUBLISHED',commit=c,**current)),flush=True)
  terminal=[row.split('|')for row in x['accounting']['stdout'].splitlines()if row.startswith('749568|')]
  if x['failures']or any(row[1].startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'))for row in terminal):
   c=publish('Record actual MSE failure without modifying training');print(json.dumps(dict(event='ACTUAL_FAILURE_NEEDS_ROOT',commit=c,**current)),flush=True);return
  if x.get('complete.json')and any(row[1:3]==['COMPLETED','0:0']for row in terminal):
   # Existing sequential validator has its own identity+CPU locks and checks
   # complete/COMPLETED0:0 again. It does not open any still-running loss arm.
   script=REPO/'reproduction/operations/qwen7b_warmup/deploy_full169028_packed_CPU_acceptance.py';p=subprocess.run([sys.executable,str(script),'--action','start','--commit',ACCEPT_COMMIT],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-2400:])
   launch=json.loads(p.stdout.splitlines()[-1]);atomic(B/'agent-mse-terminal-CPU-launch.json',launch);c=publish('Record MSE all10epochs terminal completion and locked CPU acceptance launch');print(json.dumps(dict(event='MSE_TERMINAL_CPU_VALIDATOR_LAUNCHED_NOT_YET_ACCEPTED',commit=c,**current)),flush=True)
   for attempt in range(40):
    time.sleep(15)
    status=subprocess.run([sys.executable,str(script),'--action','status','--commit',ACCEPT_COMMIT],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if status.returncode:raise RuntimeError((status.stdout+status.stderr)[-2400:])
    accepted=json.loads(status.stdout.splitlines()[-1]);atomic(B/'agent-mse-terminal-CPU-status.json',accepted)
    if'mse'in accepted['accepted']:
     atomic(B/'agent-mse-terminal-CPU-accepted.json',accepted['accepted']['mse']);c=publish('Accept actual MSE E10 SHA native model Adam sampler and two-rank RNG checkpoint');print(json.dumps(dict(event='MSE_ALL10_EPOCHS_AND_REAL_CPU_CHECKPOINT_ACCEPTANCE_COMPLETE',commit=c,receipt=accepted['accepted']['mse'],final_read=False)),flush=True);return
    if not accepted['actual_validator_identity_live']:
     raise RuntimeError('Terminal validator exited without actual MSE acceptance: '+str(accepted)[-1800:])
   raise RuntimeError('Terminal CPU acceptance bounded follow-up timed out; no checkpoint accepted claim')
  if time.time()-heartbeat>300:print(json.dumps(dict(event='READ_ONLY_PROGRESS',**current)),flush=True);heartbeat=time.time()
  time.sleep(55)
 raise RuntimeError('Bounded8hour observer expired; remote job was not modified')
if __name__=='__main__':
 try:run()
 except BaseException as e:
  atomic(B/'agent-mse-watch-failure.json',dict(time=time.time(),observer_only=True,error=type(e).__name__,message=str(e),remote_job_unmodified=True));raise
