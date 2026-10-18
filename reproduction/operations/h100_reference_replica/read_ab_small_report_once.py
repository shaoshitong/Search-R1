"""Collect completed A/B metrics and global probe loss curves, without tensors.

This is a one-shot reporting tool, not a watcher or training-side check.
Only metadata under each run is read; existing trainers are never modified.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from .readonly_management_route import MODES, run_readonly

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REMOTE = r'''
import json,hashlib,time
from pathlib import Path
ARMS=__ARMS__
THROUGH=__THROUGH__
result={};complete={};pending={}
def read_small(path):
 raw=path.read_bytes()
 if len(raw)>2*1024**2:raise ValueError('Report input is unexpectedly large: '+str(path))
 return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for arm,entry in ARMS.items():
 base=Path(entry['base'].replace('/mnt/data/sihe-share5','/mnt/shared'))
 latest,_=read_small(base/'latest-joint.json');complete[arm]=latest['outer'];pending[arm]=[];result[arm]={}
 for outer in range(1,THROUGH+1):
  root=base/'outers'/('outer-%06d'%outer)
  if not (root/'complete.json').exists():pending[arm].append(outer);continue
  joint,origin=read_small(root/'complete.json');actor=joint['actor'];probe=joint['probe_metrics']
  reward=actor.get('reward',{})
  item=dict(joint_metrics=dict(source=origin,value=dict(
   outer=outer,run_id=joint['run_id'],selection=joint.get('selection_metrics'),
   actor={key:actor.get(key)for key in ('actual_adam_steps_per_rank','new_adam_steps_per_rank',
    'scheduler_completed_outer','learning_rate_used','next_learning_rate','skipped_update_opportunities',
    'actual_global_micro','actual_local_micro','gradient_accumulation')},
   reward={key:reward.get(key)for key in ('score_mean','trajectories','version','grounding_manifest_sha256')},
   probe={key:probe[key]for key in ('status','new_adam_steps_per_rank','actual_adam_steps_per_rank',
    'ema_updates','learning_rate','ema_rate','sampling','before_update','after_update','train_loop_seconds')if key in probe},
   probe_actual_adam=joint['probe']['actual_adam_steps_per_rank'],
   phase_timings=joint['phase_timings'])))
  wall=root/'iteration-wall-time.json'
  if wall.exists():
   value,origin=read_small(wall);item['iteration_wall']=dict(origin,value=value)
  curve=root/'probe-update/global-loss-updates.jsonl'
  if curve.exists():
   raw=curve.read_bytes()
   if len(raw)>2*1024**2:raise ValueError('Unexpectedly large global loss curve')
   rows=[json.loads(line)for line in raw.splitlines()]
   item['loss_curve']=dict(path=str(curve),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),value=rows,
    report_rows_match_committed_probe_updates=len(rows)==probe['new_adam_steps_per_rank'])
  qa=root/'validation/complete.json'
  if qa.exists():
   value,origin=read_small(qa)
   item['qa']=dict(origin,value={key:value.get(key)for key in ('qa_count','em','f1','per_source',
    'actual_worker_exit_codes','evaluation_contract','actual_compute_dtype','question_source_sha256','actor_version')})
  result[arm][str(outer)]=item
print(json.dumps(dict(checked_unix=time.time(),requested_through=THROUGH,actual_complete_by_arm=complete,
 pending=pending,data=result,models_or_full_outputs_read=False,training_modified=False),allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--transport', choices=MODES, default='retained18')
    args = parser.parse_args()
    assert 1 <= args.through <= 200
    arms = json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())
    assert set(arms)=={'A','B'} and all(v['node']=='h100-11'for v in arms.values())
    spec = importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    script = REMOTE.replace('__ARMS__',repr(arms)).replace('__THROUGH__',str(args.through))
    response, transport = run_readonly(ctl,'h100-11','/usr/bin/python3 -',
                                      input_text=script,timeout=90,mode=args.transport)
    if response.returncode: raise RuntimeError(response.stderr or response.stdout)
    value = json.loads(response.stdout)
    value['observation_transport'] = transport
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = ROOT/('work/prefix5-ab-small-report-through%03d-%s.json'%(args.through,stamp))
    target.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(original=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        actual_complete_by_arm=value['actual_complete_by_arm'],pending=value['pending'],
        completed_report_rows={arm:len(rows)for arm,rows in value['data'].items()})))


if __name__ == '__main__':
    main()
