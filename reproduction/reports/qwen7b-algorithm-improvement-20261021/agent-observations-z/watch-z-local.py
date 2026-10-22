"""Temporary read-only Z749577 metadata watcher; no remote mutations or CP reads."""
from pathlib import Path
import json,subprocess,time
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];CONTROL=REPO.parents[1]
last_epoch=max([int(p.stem.rsplit('-',1)[1])for p in HERE.glob('observed-through-evaluation-epoch-*.json')],default=-1);last_terminal=None
for attempt in range(340):
 started=time.time()
 p=subprocess.run(['python3',str(CONTROL/'outputs/qwen7b-warmup-20261019/campus-control.py'),str(HERE/'read-z-749577.py')],cwd=CONTROL,capture_output=True,text=True)
 if p.returncode:
  record=dict(time=time.time(),stage='CONTROL_READ_FAILED_CURRENT_TRAINING_STATE_UNKNOWN',last_success_epoch=last_epoch,detail=p.stderr[-1200:])
 else:
  try:value=json.loads([l for l in p.stdout.splitlines()if l.startswith('{')][-1])
  except Exception as error:
   record=dict(time=time.time(),stage='CONTROL_RESPONSE_PARSE_FAILED_STATE_UNKNOWN',detail=repr(error))
  else:
   (HERE/'latest-actual.json').write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
   progress=[value.get('progress-rank-%d.json'%rank,{})for rank in(0,1)]
   record=dict(time=value['time'],control_route='existing_approved_hkust3_campus_control',stage='ACTUAL_READ_ONLY_METADATA',sacct=value.get('sacct'),queue=value.get('queue'),ranks=[{k:p[k]for k in('epoch','cursor','epoch_questions_seen','training_loss','variance_MSE','variance_MAE','alignment','actual_global_batch','native_gradient_L2','head_gradient_L2')if k in p}for p in progress],coverage={k:v for k,v in value.items()if k.startswith('coverage-epoch-')},evaluated_epochs=[v['epoch']for k,v in value.items()if k.startswith('evaluation-epoch-')],latest=value.get('latest.json'),complete=value.get('complete.json'),new_writer_failures=value.get('new_writer_failures'),actual_log_error_tail=value.get('actual_log_error_tail'),consecutive_updates=[v.get('cursor')for v in value.get('last8_updates',[])])
   epoch=max(record['evaluated_epochs']or[-1])
   if epoch>last_epoch:
    (HERE/('observed-through-evaluation-epoch-%02d.json'%epoch)).write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n');last_epoch=epoch
   terminal=value.get('sacct','')
   if any(state in terminal for state in('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):
    last_terminal=terminal
 with(HERE/'observations.jsonl').open('a')as output:output.write(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n')
 print(json.dumps(record,sort_keys=True),flush=True)
 if last_terminal:break
 time.sleep(max(1,45-(time.time()-started)))
