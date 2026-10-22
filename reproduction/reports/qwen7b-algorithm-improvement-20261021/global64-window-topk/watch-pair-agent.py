"""One read-only observer for root-owned749889/749890; no Slurm mutations."""
import fcntl,json,os,pathlib,subprocess,sys,tempfile,time
B=pathlib.Path(__file__).resolve().parent;REPO=B.parents[3];CONTROL=REPO.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py';BRANCH='codex/q7b-global64-window-observer-20261023'
def atomic(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def publish(message):
 parent=subprocess.check_output(['git','rev-parse',BRANCH],cwd=REPO,text=True).strip();fd,index=tempfile.mkstemp(prefix='pair-observer-index-');os.close(fd);os.unlink(index);env=dict(os.environ,GIT_INDEX_FILE=index)
 try:
  subprocess.check_call(['git','read-tree',parent],cwd=REPO,env=env);paths=[p for p in B.glob('agent-pair-*')if p.suffix in('.json','.jsonl','.md')]+[B/'PAIR-EPOCHS.md'];subprocess.check_call(['git','add',*[str(p.relative_to(REPO))for p in paths if p.exists()]],cwd=REPO,env=env);tree=subprocess.check_output(['git','write-tree'],cwd=REPO,env=env,text=True).strip();old=subprocess.check_output(['git','rev-parse',parent+'^{tree}'],cwd=REPO,text=True).strip()
  if tree==old:return parent
  c=subprocess.check_output(['git','commit-tree',tree,'-p',parent,'-m',message],cwd=REPO,text=True).strip();subprocess.check_call(['git','update-ref','refs/heads/'+BRANCH,c,parent],cwd=REPO);subprocess.check_call(['git','push','origin',BRANCH],cwd=REPO);actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0];assert actual==c;return c
 finally:
  if pathlib.Path(index).exists():pathlib.Path(index).unlink()
def report(x):
 out=['# Actual matched global64 TopK window-pair epochs','', 'Root-owned jobs current64=749889, window2018=749890. No final4096 read. Primary Top512 uses the same viewed independent2048 ownG5; fixed train2048 is reported separately. Rank sigmoid scores are uncalibrated; BA threshold0.5 is predeclared and secondary.','', '| Cell | Epoch | Pool | State | BA% | AUC% | Top512 Mixed | FPR% | rho |','|---|---:|---|---|---:|---:|---:|---:|---:|']
 for cell,c in x['cells'].items():
  for key,e in sorted(c.get('epoch_metrics',{}).items()):
   for pool,states in e['metrics'].items():
    for state,v in states.items():out.append('| %s | %d | %s | %s | %.2f | %.2f | %d/512 (%.2f%%) | %.2f | %s |'%(cell,e['epoch'],pool,state,100*v['balanced_accuracy'],100*v['Mixed_AUC'],v['Top512']['Mixed'],100*v['Top512']['Mixed_fraction'],100*v['nonMixed_false_positive_rate'],'—'if v['variance_Spearman']is None else'%.4f'%v['variance_Spearman']))
 out+=['','Only actual epoch receipts appear above; process start and queue status are not training acceptance. Each epoch audit is preserved in the small latest JSON and covers only its exact training-cache window, not a future or final pool.'];(B/'PAIR-EPOCHS.md').write_text('\n'.join(out)+'\n')
def main():
 lock=(B/'agent-pair-watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();seen=None;last_good={};verified=set();transport=0;heartbeat=0
 while time.time()-start<8*3600:
  p=subprocess.run([sys.executable,str(CONTROL),str(B/'read-pair-agent.py'),'--read-only','--timeout','25'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  if p.returncode:
   message=(p.stdout+p.stderr)[-2400:]
   if any(t in message for t in('Operation not permitted','Permission denied','network access is disabled','permission profile')):raise RuntimeError(message)
   if not any(t in message.lower()for t in('connection closed','connect to host','connection reset','timed out','read timeout','broken pipe')):raise RuntimeError(message)
   transport+=1;atomic(B/'agent-pair-transport-latest.json',dict(time=time.time(),status='CURRENT_REMOTE_STATE_UNKNOWN_LAST_VALUES_STALE',attempts=transport,error=message,remote_job_failure_not_inferred=True));time.sleep(55);continue
  x=json.loads(p.stdout.splitlines()[-1]);atomic(B/'agent-pair-latest.json',x);report(x);summary=dict(time=x['time'],observer_pid=os.getpid(),accounting=x.get('accounting'),queue=x.get('queue'),cells={},final4096_read=False)
  for cell,c in x['cells'].items():
   ranks=[c.get('progress-rank-%d.json'%r,{})for r in(0,1)];tail=c.get('last_six_real_attempts',[])[-6:];steps=[v.get('actual_Adam_updates',0)for v in tail];consecutive=len(steps)==6 and all(steps[i+1]==steps[i]+1 for i in range(5));epochs=sorted(c.get('epoch_metrics',{}));audits=sorted(c.get('epoch_audits',{}));controller=c.get('controller.json',{});summary['cells'][cell]=dict(job_id=c.get('root-submission.json',{}).get('job_id'),rank_attempts=[v.get('cursor')for v in ranks],rank_Adam_updates=[v.get('actual_Adam_updates')for v in ranks],rank_epochs=[v.get('epoch')for v in ranks],last_six_actual_Adam_steps=steps,last_six_steps_consecutive=consecutive,N_eff=ranks[0].get('N_eff'),K=ranks[0].get('K'),history=ranks[0].get('detached_historical_questions'),initial_head_sha256=controller.get('initial_head_sha256'),evaluations=epochs,audits=audits,failures=c['failures'],checkpoint=c.get('latest.json'))
   old=last_good.get(cell)
   if old and cell not in verified and all(v.get('actual_Adam_updates',0)>=6 for v in ranks)and consecutive and any(ranks[i].get('cursor',0)>old[i].get('cursor',0)for i in(0,1)):
    verified.add(cell);atomic(B/('agent-pair-'+cell+'-six-updates-verified.json'),dict(time=x['time'],prior_actual_ranks=old,current_actual_ranks=ranks,last_six_real_attempts=tail,two_fresh_samples_and_six_actual_Adam_steps=True,Slurm_only_not_sufficient=True,final4096_read=False));print(json.dumps(dict(event='ACTUAL_PAIRED_CELL_SIX_ADAM_UPDATES_AND_NEXT_PROGRESS_VERIFIED',cell=cell,summary=summary['cells'][cell])),flush=True)
   if all(v for v in ranks):last_good[cell]=ranks
  with(B/'agent-pair-observations.jsonl').open('a')as f:f.write(json.dumps(summary)+'\n')
  atomic(B/'agent-pair-watch-latest.json',summary);key={cell:(tuple(c['evaluations']),tuple(c['audits']),tuple(sorted(c['failures'])),cell in verified)for cell,c in summary['cells'].items()}
  if key!=seen:
   commit=publish('Record actual global64 TopK pair phase/epoch progress and cache audit');seen=key;print(json.dumps(dict(event='PAIRED_PHASE_PUBLISHED',commit=commit,summary=summary)),flush=True)
  if any(c['failures']for c in summary['cells'].values())or any(t in x.get('accounting',{}).get('stdout','')for t in('|FAILED|','|CANCELLED|','|TIMEOUT|','|OUT_OF_MEMORY|','|NODE_FAIL|')):print(json.dumps(dict(event='ACTUAL_PAIR_FAILURE_ROOT_HANDLES',summary=summary)),flush=True);return
  if all(c.get('complete.json')for c in x['cells'].values()):print(json.dumps(dict(event='PAIR_PRODUCER_COMPLETE_NEEDS_INDEPENDENT_CHECKPOINT_ACCEPTANCE',summary=summary)),flush=True);return
  if time.time()-heartbeat>300:print(json.dumps(dict(event='PAIRED_READ_ONLY_PROGRESS',summary=summary)),flush=True);heartbeat=time.time()
  transport=0;time.sleep(55)
 raise RuntimeError('Bounded8hour observer expired; remote jobs unmodified')
if __name__=='__main__':
 try:main()
 except BaseException as e:atomic(B/'agent-pair-watch-failure.json',dict(time=time.time(),observer_only=True,error=type(e).__name__,message=str(e),remote_jobs_unmodified=True));raise
