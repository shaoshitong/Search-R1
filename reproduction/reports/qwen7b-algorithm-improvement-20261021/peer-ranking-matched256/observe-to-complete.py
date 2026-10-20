"""Finite read-only60s observer for exactly the two existing matched jobs.

No submission, retry, restart, mutation, cancellation, or escalation occurs.
"""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[4]
CONTROL=ROOT.parents[1]
OUT=pathlib.Path(__file__).parent/'actual-observer'
OUT.mkdir(exist_ok=True)
COMMIT='3c6faad3b821015b36b6c92ba51308898ac1b381'
started=time.monotonic();observations=0;finished=set()
while time.monotonic()-started<6*3600:
 for recipe in ('inter1','bothrank1'):
  if recipe in finished:continue
  args=[sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_rank_matched','--control-root',str(CONTROL),'--commit',COMMIT,'--recipe',recipe,'--action','status']
  p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=65)
  if p.returncode:
   x=dict(status='READ_FAILED_NOT_CURRENT_PROGRESS',recipe=recipe,time=time.time(),returncode=p.returncode,error=p.stderr[-1200:])
  else:
   x=json.loads(p.stdout.splitlines()[-1]);(OUT/(recipe+'-latest-good.json')).write_text(json.dumps(x,indent=2)+'\n')
  progress=x.get('progress',{});row=dict(recipe=recipe,read_status=x.get('status'),read_error=x.get('error'),time=x.get('observed_epoch',time.time()),job_id=x.get('job_id'),squeue=x.get('squeue'),sacct=x.get('sacct'),cursor=progress.get('cursor'),steps=progress.get('steps'),loss=progress.get('loss'),seconds=progress.get('seconds'),replay=x.get('next-update-replay',{}).get('status'),first_independent_CPU=x.get('first-cp-independent-audit',{}).get('status'),latest_cursor=x.get('latest',{}).get('cursor'),failure=x.get('failure'),CP_audit_failure=x.get('first-cp-audit-failure'),complete_ref=x.get('complete_ref'))
  with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
  print(json.dumps(row,sort_keys=True),flush=True);observations+=1
  terminal=x.get('sacct','')
  if x.get('failure') or x.get('first-cp-audit-failure') or any(s in terminal for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):finished.add(recipe)
  elif x.get('complete_ref') and '|COMPLETED|0:0|' in terminal:finished.add(recipe)
 if len(finished)==2:break
 time.sleep(60)
print(json.dumps(dict(status='FINITE_READ_ONLY_OBSERVER_EXITED',recipes_with_terminal_observations=sorted(finished),observations=observations,all_final_acceptance_still_requires_independent_audit=True)),flush=True)
