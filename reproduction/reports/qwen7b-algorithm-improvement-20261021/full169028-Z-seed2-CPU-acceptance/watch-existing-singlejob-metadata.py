"""Read one already-running job/coordinator every50s; no start or mutation."""
from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[6];REPO=Path.cwd();R=Path(__file__).parent;reader=R/'read-existing-749935-terminal-and-CPU.py';control=REPO.parents[1]/'outputs/qwen7b-warmup-20261019/balanced-control.py'
while True:
 started=time.monotonic();r=subprocess.run(['python3',str(control),str(reader),'--timeout','35'],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
 if r.returncode:
  print(json.dumps(dict(stage='METADATA_QUERY_ERROR',time=time.time(),error=r.stderr[-1600:]+r.stdout[-1600:])),flush=True)
 else:
  j=json.loads(r.stdout);(R/'watch-existing-749935-latest.json').write_text(json.dumps(j,indent=2)+'\n');error=j['training_failures']or j['coordinator'].get('failure.json');done=('749935|COMPLETED|0:0|'in j['accounting']and'complete.json'in j['coordinator']and'accepted'in j);summary=dict(time=j['time'],accounting=j['accounting'],rank_cursors={k:v.get('cursor')for k,v in j['training'].items()if k!='complete.json'},epochs={k:v.get('epoch')for k,v in j['training'].items()if k!='complete.json'},coordinator_live=j['coordinator_actual_identity_live'],CPU_child_live=j.get('CPU_child_actual_identity_live'),CPU_stage=j.get('CPU_validator_progress',{}).get('stage'),accepted='accepted'in j,complete=done,error=error)
  with(R/'watch-existing-749935-metadata-observations.jsonl').open('a')as f:f.write(json.dumps(summary)+'\n')
  print(json.dumps(summary),flush=True)
  if error or done:
   (R/('ACTUAL-749935-E10-CPU-ACCEPTED.json'if done else'ACTUAL-749935-ERROR.json')).write_text(json.dumps(j,indent=2)+'\n');break
 time.sleep(max(0,50-(time.monotonic()-started)))
