"""Finite current v2 capacity job observation; no schedule, submission or training edits."""
import fcntl,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
lock=(HERE/'finite-current-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.time();previous=None;status_errors=0
while time.time()-started<11400:
 p=subprocess.run(['python3',str(HERE/'prepare-screen-and-entry.py'),'--action','status'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if p.returncode:
  status_errors+=1
  with(HERE/'actual-readonly-query-failures.jsonl').open('a')as f:f.write(json.dumps(dict(epoch=time.time(),status='READONLY_QUERY_FAILED_NO_TRAINING_STATE_INFERENCE',attempt=status_errors,returncode=p.returncode,kind='optional_NFS_FileNotFound'if 'FileNotFoundError'in p.stderr else 'status_RPC_failed',last_good_preserved=True,training_mutated=False))+chr(10))
  if status_errors>=3:raise RuntimeError('Three readonly observations failed; training unchanged: '+p.stderr[-2000:])
  time.sleep(5);continue
 status_errors=0
 s=json.loads(p.stdout.splitlines()[-1]);assert s['job_id']!='743933' and s['submission']['config']['sha256']=='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767'
 cache=s.get('cache-progress',{});progress=s.get('progress',{});latest=s.get('latest',{})
 row=dict(epoch=s['epoch'],job_id=s['job_id'],stage=progress.get('stage'),cache_trajectories=cache.get('completed_trajectories',0),
  cache_complete=bool(s.get('cache-complete')),native_golden=s.get('cache-same-api-golden',{}).get('status'),
  head_cursor=progress.get('cursor',0),head_Adam=progress.get('head_Adam'),head_full_CP=latest.get('cursor'),
  first_full_update_replay=s.get('first-update-replay',{}).get('status'),failure=s.get('failure'),
  squeue=s['squeue']['stdout'].strip(),sacct=s['sacct']['stdout'].strip())
 with(HERE/'actual-progress-observations.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
 (HERE/'actual-current-latest-good.json').write_text(json.dumps(row,indent=2)+'\n')
 key=(row['cache_trajectories']//2560,row['head_cursor']//128,row['cache_complete'],bool(row['failure']))
 if key!=previous:print(json.dumps(row),flush=True);previous=key
 main_rows=[line for line in row['sacct'].splitlines()if line.split('|')[0]==s['job_id']]
 if len(main_rows)!=1:raise RuntimeError('Exact allocation accounting row unavailable; no inference/restart')
 if row['failure']or any(x in main_rows[0]for x in ('FAILED','OUT_OF_MEMORY','NODE_FAIL','TIMEOUT','CANCELLED')):
  raise RuntimeError('Actual exit/failure requires diagnosis; no blind restart: '+json.dumps(row))
 if s.get('complete'):
  print(json.dumps(dict(status='REAL_CAPACITY_COMPLETION_EXISTS_AWAITING_INDEPENDENT_METRICS',complete=s['complete'])),flush=True);break
 if (s['job_id']+'|COMPLETED|0:0|')in row['sacct']:
  print(json.dumps(dict(status='CURRENT_ALLOCATION_EXITED_WITHOUT_FULL20EPOCH_COMPLETE_PRESERVE_CACHE_HEAD_STATE',actual=row)),flush=True);break
 time.sleep(45)
else:print(json.dumps(dict(status='FINITE_CURRENT_OBSERVATION_BOUND_REACHED_NO_MUTATION',seconds=time.time()-started)),flush=True)
