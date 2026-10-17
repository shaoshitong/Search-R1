"""Bounded read of periodic full-pool and fresh diagnostic metadata only.

No model load, tensor read, training edits, submission, or extra observer.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
p=argparse.ArgumentParser();p.add_argument('--outer',type=int,required=True);a=p.parse_args()
assert a.outer in range(10,200,10)
arms=json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())
remote=r'''
import json,time,hashlib
from pathlib import Path
ARMS=__ARMS__;N=__OUTER__
r={}
for arm,entry in ARMS.items():
 root=Path(entry['base'].replace('/mnt/data/sihe-share5','/mnt/shared'))/'outers'/('outer-%06d'%N)
 phases={}
 for role,folder in [('pool','full-pool'),('diagnostic_prefix','diagnostic/outers/outer-%06d/prefix'%N),('diagnostic_full','diagnostic/outers/outer-%06d/full'%N)]:
  rows=[]
  for rank in range(4):
   path=root/folder/('rank-%d-progress.json'%rank)
   if path.exists():
    d=json.loads(path.read_text());rows.append({k:d.get(k)for k in ['rank','outer','phase','stage','actor_version','completed_trajectories','expected_trajectories','active_trajectories','pending_generation','pending_retrieval','actual_sampled_tokens','elapsed_seconds','utc','device_binding','initialization_oom_attempts']})
  phases[role]=rows
 records={}
 for role,name,keys in [('pool_complete','full-pool/complete.json',['run_id','outer','actor_version','questions','trajectories','phase','selection_sha256','prefix_scan_sha256']),('actual_selection','probe-score/actual-em-selection.json',['actor_version','candidate_questions','selected_questions','outcome_labels_used','selection_mode','actual_mixed_candidate_count','actual_nonmixed_candidate_count','chosen_mixed_count','fallback_count','actual_candidate_correct_count_histogram']),('diagnostic_complete','diagnostic-complete.json',None),('probe_metrics','probe-update/metrics.json',None),('frozen_metrics','probe-frozen.json',None)]:
  path=root/name
  if path.exists():
   data=path.read_bytes();assert len(data)<2*1024**2
   d=json.loads(data);records[role]={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'value':{k:d.get(k)for k in keys}if keys else d}
 r[arm]={'phases':phases,'records':records}
print(json.dumps({'observed_unix':time.time(),'outer':N,'arms':r}))
'''.replace('__ARMS__',repr(arms)).replace('__OUTER__',str(a.outer))
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
r=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=remote,text=True,capture_output=True,timeout=65);assert r.returncode==0,r.stderr;d=json.loads(r.stdout);tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/('work/prefix5-ab-periodic%d-phases-%s.json'%(a.outer,tag));out.write_text(json.dumps(d,indent=2)+'\n');print('actual',str(out),'SHA',hashlib.sha256(out.read_bytes()).hexdigest())
for arm,v in d['arms'].items():
 print(arm,[(k,[(x['rank'],x['completed_trajectories'],x['expected_trajectories'],x['stage'])for x in rows])for k,rows in v['phases'].items()]);print('records',arm,{k:v['value']for k,v in v['records'].items()})
