"""Finite readonly745731 observer, once/minute; never submit/signal/delete."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
b=Path(__file__).parent
with(b/'finite-capture-observer.lock').open('a+')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (b/'finite-observer-controller.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time(),observed_job='745731',maximum_observations=120,readonly=True))+'\n')
 previous=None;errors=0
 for i in range(120):
  start=time.monotonic();r=subprocess.run([sys.executable,str(b/'observe-fixed9344-job.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=50)
  if r.returncode:
   errors+=1;print(json.dumps(dict(status='OBSERVER_READ_FAILED_NO_JOB_ACTION',attempt=i,error_tail=r.stderr.decode(errors='replace')[-1000:],consecutive=errors)),flush=True)
   if errors>=3:raise RuntimeError('Three observer reads failed; training state is not inferred')
  else:
   errors=0;v=json.loads((b/'actual-capture-latest.json').read_text());shards=v['shards'];compact=dict(epoch=v['epoch'],main=v['main_sacct'],shards={k:dict(rows=x.get('complete.json',{}).get('trajectories',x.get('progress.json',{}).get('trajectories',0)),chunks=x['complete_chunks'],bytes=x['actual_packed_bytes'],failed=x.get('failure.json'))for k,x in shards.items()},acceptance=v.get('bank-acceptance-progress.json'),bank_ready='bank/complete.json'in v)
   changed=json.dumps({k:x for k,x in compact.items()if k!='epoch'},sort_keys=True)
   if changed!=previous:print(json.dumps(compact),flush=True);previous=changed
   rows=[x.split('|')for x in v['main_sacct'].splitlines()if x.split('|')[0]=='745731'];assert len(rows)==1
   state=rows[0][1]
   if state not in('RUNNING','PENDING','COMPLETING','CONFIGURING'):
    if state=='COMPLETED'and compact['bank_ready']:(b/'finite-observer-terminal.json').write_text(json.dumps(dict(status='ACTUAL_FIXED9344_CAPTURE_AND_SOURCE_BOUND_BANK_TERMINAL',observation=v))+'\n');break
    (b/'finite-observer-attention.json').write_text(json.dumps(v)+'\n');raise RuntimeError('Actual main terminal without accepted feature bank: '+state)
  time.sleep(max(1,60-(time.monotonic()-start)))
 else:raise RuntimeError('Finite observation budget exhausted; no claim of completion')
