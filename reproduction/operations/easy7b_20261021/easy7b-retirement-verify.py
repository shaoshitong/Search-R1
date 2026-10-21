import json,time,os,subprocess
from pathlib import Path
p=Path('/tmp/approved-old-qprobe-checkpoints-20261021/plan.json');plan=json.loads(p.read_text());remaining=[x['path']for x in plan['files']if Path(x['path']).exists()];ledgers=[]
for f in p.parent.glob('unlink-*.jsonl'):ledgers += [json.loads(x)for x in f.read_text().splitlines()if x]
d=dict(epoch=time.time(),all32_absent=not remaining,remaining=remaining,unlink_records=len(ledgers),confirmed_released_bytes=sum(x['released_lastlink_allocated_bytes']for x in ledgers),raw_and_archives_preserved=True,archive_roots_exist=[])
for name in ['qprobe-teacher-warmup200-20261014','qprobe-teacher-warmup200-sigma3-20261015','qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017']:
 b=Path('/data/user/jxu957/experiments')/name;d['archive_roots_exist'].append(dict(base=str(b),logs=(b/'logs').is_dir(),archives=(b/'archives').exists(),raw_attempts=(b/'artifacts/question-probe/attempts').exists()))
for c in [Path('/tmp/easy7b-storage-current-20261021-21930'),Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/control/storage-current')]:
 for f in c.glob('result-*.json'):d[str(f)]=json.loads(f.read_text())
 for f in c.glob('claim-*.json'):
  x=json.loads(f.read_text());q=Path('/proc')/str(x['pid'])
  if x['uid']!=os.getuid():x['process_visibility']='unknown_different_UID_not_terminal_evidence'
  elif q.exists():
   st=(q/'stat').read_text().rsplit(')',1)[1].split();x['actual_state']=st[0];x['same_birth']=st[19]==x['birth'];x['children']=(q/'task'/str(x['pid'])/'children').read_text()
  else:x['own_process_exists']=False
  d[str(f)]=x
print(json.dumps(d))
