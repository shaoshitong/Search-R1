"""Finite current-run observation and four typed node acceptances; no schedule.

Never submits/restarts/cancels training or changes its priority/code/data. A
scientific negative is reported for the parent, never silently stops a run.
"""
import fcntl,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
guard=(HERE/'finite-milestone-observer.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.time();previous=None;accepted={};started_audits=set()
def invoke(args):
 p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if p.returncode:raise RuntimeError('Finite readonly operator needs attention; training unchanged: '+p.stderr[-2200:])
 return json.loads(p.stdout.splitlines()[-1])
while time.time()-started<11400:
 s=invoke(['python3','-m','reproduction.operations.qwen7b_warmup.deploy_direct_count_headlr','--action','status']);assert s.get('job_id')=='743809'
 p=s.get('progress',{});queue=s['squeue']['stdout'].strip();sacct=s['sacct']['stdout'].strip();cursor=p.get('cursor',0)
 row=dict(epoch=s['epoch'],job_id='743809',cursor=cursor,legacy_Adam=p.get('steps'),head_Adam=p.get('head_steps'),full_CP=s.get('latest',{}).get('cursor'),queue=queue,sacct=sacct,failure=s.get('failure'))
 with(HERE/'actual-milestone-observations.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
 (HERE/'actual-milestone-latest-good.json').write_text(json.dumps(s,indent=2)+'\n')
 key=(cursor//32,tuple(accepted),bool(s.get('failure')))
 if key!=previous:print(json.dumps(row),flush=True);previous=key
 main_rows=[line for line in sacct.splitlines()if line.split('|')[0]=='743809']
 if len(main_rows)!=1:raise RuntimeError('Exact allocation row unavailable; no inference/restart')
 if s.get('failure')or any(x in main_rows[0]for x in ['FAILED','OUT_OF_MEMORY','NODE_FAIL','TIMEOUT','CANCELLED']):raise RuntimeError('Actual training exit/error requires diagnosis, no automatic rollback: '+json.dumps(row))
 for k in (256,512,768,938):
  if k in accepted or 'milestone_%d_complete'%k not in s:continue
  if k not in started_audits:
   r=invoke(['python3',str(HERE/'typed-milestone-audit.py'),'--action','start','--cursor',str(k)]);started_audits.add(k);print(json.dumps(dict(stage='TYPED_AUDIT_STARTED_OR_EXISTING_PRESERVED',cursor=k,result=r)),flush=True)
  r=invoke(['python3',str(HERE/'typed-milestone-audit.py'),'--action','status','--cursor',str(k)])
  if r.get('exit',{}).get('returncode',0)!=0:raise RuntimeError('Typed CPU audit failed; no blind retry: '+json.dumps(r))
  if 'result' in r:
   assert r['result']['config']['sha256']=='71c3ee90f0232b889d4ced2d0cb50d69e852eb1ebbdbb55fdff03eaefdbbcc71'and r['result']['head_Adam_LR']==1e-4
   accepted[k]=r;print(json.dumps(dict(stage='TYPED_FIXED_MILESTONE_ACCEPTED',cursor=k,receipt=r['receipt'],metrics=r['result']['metrics'])),flush=True)
   (HERE/'actual-accepted-milestones.json').write_text(json.dumps({str(k):v for k,v in accepted.items()},indent=2)+'\n')
 if set(accepted)=={256,512,768,938}and '743809|COMPLETED|0:0|'in sacct and s.get('complete',{}).get('cursor')==938:
  result=dict(status='HEADLR1E4_MATCHED938_AND_ALL_TYPED_FIXED_MILESTONES_ACCEPTED',training_complete=s['complete'],milestones={str(k):v['receipt']for k,v in accepted.items()},epoch=time.time())
  (HERE/'actual-terminal-accepted.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);break
 time.sleep(45)
else:print(json.dumps(dict(status='FINITE_OBSERVATION_BOUND_REACHED_NO_MUTATION',seconds=time.time()-started)),flush=True)
