"""Serial read-only followup for root's exact submitted seed2 MSE-M5 job."""
import json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[5];C=R.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py'
last=None;deadline=time.time()+10*3600;first=None
while time.time()<deadline:
 r=subprocess.run(['python3',str(C),str(H/'read-live.py'),'--read-only','--timeout','25'],cwd=R,capture_output=True,text=True)
 if r.returncode:
  x=dict(time=time.time(),read_failed=True,stdout_tail=r.stdout[-1000:],stderr=r.stderr[-1500:],state_not_inferred=True)
  with(H/'read-errors.jsonl').open('a')as f:f.write(json.dumps(x)+'\n')
  if last!='READ_FAILED':print(json.dumps(x),flush=True);last='READ_FAILED'
  time.sleep(60);continue
 x=json.loads(r.stdout);tmp=H/'actual-latest.tmp';tmp.write_text(json.dumps(x,sort_keys=True)+'\n');tmp.replace(H/'actual-latest.json');files=x['files'];s=x['accounting']['stdout'];state=s.split('|')[1]if s else'UNKNOWN';p=[files.get('progress-rank-%d.json'%i)for i in(0,1)];epochs=sorted(k for k in files if k.startswith('evaluation-epoch-'));errors={k:v for k,v in files.items()if'failure-rank-'in k};summary=dict(time=x['time'],job_id='750001',state=state,ranks=p,errors=errors,evaluated_epochs=epochs,six_consecutive_updates_proven=x['six_consecutive_updates_proven'],actual_last_updates=x['last_actual_updates'],complete=files.get('complete.json'),latest=files.get('latest.json'),final4096_read=False)
 if all(p):
  if first is None:first=dict(time=x['time'],cursors=[v['cursor']for v in p])
  summary['first_two_rank_progress']=first;summary['second_separate_progress_proven']=x['time']>first['time']and all(v['cursor']>a for v,a in zip(p,first['cursors']))
 with(H/'observations.jsonl').open('a')as f:f.write(json.dumps(summary,sort_keys=True)+'\n')
 notice=(state,tuple(epochs),bool(errors),x['six_consecutive_updates_proven'],summary.get('second_separate_progress_proven'))
 if notice!=last:print(json.dumps(summary),flush=True);last=notice
 if errors or state in('FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','CANCELLED','COMPLETED'):print(json.dumps(dict(stop_reason='ACTUAL_OWNED_JOB_TERMINAL',state=state,CPU_acceptance_still_required=True)),flush=True);break
 time.sleep(60)
