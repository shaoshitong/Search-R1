from pathlib import Path
import json,datetime,hashlib,re
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3347-1791494104474180576');p=r/'calibration.log';last=[];events=[]
for line in p.open(errors='replace'):
 if 'val/em/all' in line and ('512' in line or '51713' in line):events.append({'before':last[-10:],'metric_prefix':line[:250]})
 # Only source/step events, no trajectory texts
 if any(x in line for x in ['global_step_','step=','step:', '512','FINAL','EVAL','checkpoint','validation']) and not any(x in line for x in ['<think>','<search>','<answer>','<information>']):last.append(line.strip()[:700])
 last=last[-10:]
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'validation_files':[str(x) for x in (r/'validation_trajectories').glob('*')],'events':events}))
