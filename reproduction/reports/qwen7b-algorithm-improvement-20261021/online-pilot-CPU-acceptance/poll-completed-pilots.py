"""Local small metadata snapshot only; does not launch validation or jobs."""
import json,subprocess,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[4];control=repo.parents[1]/'outputs/qwen7b-warmup-20261019/balanced-control.py';source=Path(__file__).with_name('monitor-seven-cell-completion.py');r=subprocess.run([sys.executable,str(control),str(source),'--timeout','30'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-1500:])
x=json.loads(r.stdout.splitlines()[-1]);base=Path(__file__).parent;(base/'completion-current.json').write_text(json.dumps(x,indent=2)+'\n');compact=dict(time=x['time'],variants=[dict(variant=v['variant'],job=v.get('job_id'),state=v.get('sacct',[None,'UNKNOWN'])[1],cursor=v.get('progress-rank-0.json',{}).get('cursor'),complete=v['complete_exists'],accepted=v['accepted_exists'],accepted_unchanged=v.get('accepted_checkpoint_metadata_unchanged'),eligible=v['eligible_new_CPU_acceptance'])for v in x['variants']])
with(base/'completion-observations.jsonl').open('a')as f:f.write(json.dumps(compact)+'\n')
print(json.dumps(compact))
