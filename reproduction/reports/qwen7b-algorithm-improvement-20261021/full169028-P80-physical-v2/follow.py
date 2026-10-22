"""One serial, read-only observer; never restart an uncertain CPU producer."""
import json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];C=R.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py';last=None;first=None
deadline=time.time()+5*3600
while time.time()<deadline:
 r=subprocess.run(['python3',str(C),str(H/'read-live.py'),'--read-only','--timeout','25'],capture_output=True,text=True,cwd=R)
 if r.returncode:
  x=dict(time=time.time(),read_failed=True,stderr=r.stderr[-1400:],stdout_tail=r.stdout[-800:],state_not_inferred=True)
  with(H/'read-errors.jsonl').open('a')as f:f.write(json.dumps(x)+'\n')
  if last!='READ_FAILED':print(json.dumps(x),flush=True);last='READ_FAILED'
  time.sleep(45);continue
 x=json.loads(r.stdout);tmp=H/'actual-latest.tmp';tmp.write_text(json.dumps(x,sort_keys=True)+'\n');tmp.replace(H/'actual-latest.json');files=x['files'];p=files.get('progress.json',{});summary=dict(time=x['time'],live=x['actual_owned_process_live'],progress=p,failure=files.get('failure.json'),complete=files.get('complete.json'),accepted=files.get('fixture/accepted.json'),process_resources=x.get('process_resources'),final4096_read=False)
 if p:
  if first is None or ('unique_train_trajectories'in p)!=('unique_train_trajectories'in first):first=p
  summary['first_progress']=first;counter='unique_train_trajectories'if'unique_train_trajectories'in p else'physical_selected_rows';summary['two_actual_progress_samples']=p['time']>first['time']and p.get(counter,0)>first.get(counter,0)
  if p['time']>first['time']and'RAW_compressed_bytes_completed'in p and'RAW_compressed_bytes_completed'in first:summary['interval_RAW_bytes_per_second']=(p['RAW_compressed_bytes_completed']-first['RAW_compressed_bytes_completed'])/(p['time']-first['time'])
 with(H/'observations.jsonl').open('a')as f:f.write(json.dumps(summary,sort_keys=True)+'\n')
 notice=(p.get('source_files_completed',0)//64,bool(summary['failure']),bool(summary['accepted']))
 if notice!=last:print(json.dumps(summary),flush=True);last=notice
 if summary['failure']or summary['accepted']or(not summary['live']and files.get('controller.json')):print(json.dumps(dict(stop_reason='ACTUAL_CPU_PROCESS_ACCEPTED_OR_NEEDS_ATTENTION',never_auto_restarted=True)),flush=True);break
 time.sleep(30)
