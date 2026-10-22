import json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];C=R.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py';last=None;first=None;deadline=time.time()+5*3600
while time.time()<deadline:
 r=subprocess.run(['python3',str(C),str(H/'read-live.py'),'--read-only','--timeout','25'],capture_output=True,text=True,cwd=R)
 if r.returncode:
  with(H/'read-errors.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),error=r.stderr[-1200:],unknown=True))+'\n')
  time.sleep(45);continue
 x=json.loads(r.stdout);tmp=H/'actual-latest.tmp';tmp.write_text(json.dumps(x,sort_keys=True)+'\n');tmp.replace(H/'actual-latest.json');files=x['files'];p=files.get('P20-crosscheck-progress.json',{});summary=dict(time=x['time'],live=x['actual_owned_process_live'],progress=p,resources=x.get('process_resources'),failure=files.get('P20-crosscheck-failure.json'),complete=files.get('P20-crosscheck-complete.json'),mismatch=files.get('P20-crosscheck-mismatch.json'),final4096_read=False)
 if p:
  if first is None:first=p
  summary['first_progress']=first;summary['two_actual_samples']=p['time']>first['time']and(p.get('P20_train_rows',0)>first.get('P20_train_rows',0)or p.get('P80_train_rows_compared',0)>first.get('P80_train_rows_compared',0))
 with(H/'observations.jsonl').open('a')as f:f.write(json.dumps(summary,sort_keys=True)+'\n')
 notice=(p.get('stage'),p.get('P20_train_rows',0)//262144,p.get('P80_train_rows_compared',0)//262144,bool(summary['failure']))
 if notice!=last:print(json.dumps(summary),flush=True);last=notice
 if summary['failure']or summary['complete']or(not summary['live']and files.get('P20-crosscheck-controller.json')):print(json.dumps(dict(stop_reason='ACTUAL_CPU_AUDIT_TERMINAL')),flush=True);break
 time.sleep(30)
