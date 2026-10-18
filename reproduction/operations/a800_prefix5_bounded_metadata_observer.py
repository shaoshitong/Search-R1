"""Read the running Prefix5 experiment without reloading unbounded history.

This entry is observation only. It sends code over the existing physical SSH
route and never installs files in, or mutates, the running training checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from . import a800_prefix5_fault_joint_recovery_control as previous

ROOT = previous.ROOT
HISTORY_START = "v['restored_complete_history']=[]\n"
HISTORY_END = "v['cpu_qualification_is_not_cuda_restoration']=True\n"
assert previous.OBSERVE.count(HISTORY_START) == 1
assert previous.OBSERVE.count(HISTORY_END) == 1
_required, _history = previous.OBSERVE.split(HISTORY_START, 1)
_old_history, _finish = _history.split(HISTORY_END, 1)

# The prefix retains the exact source, owned-birth, current four-rank ledger,
# native progress, latest-joint, candidate and existing QA checks. Only the
# optional recovery/history suffix changes. In particular read(), its 64 MiB
# cumulative guard, emit(), and its 128 KiB guard are unchanged.
REQUIRED_CHECKS_SHA256 = hashlib.sha256(_required.encode()).hexdigest()
BOUNDED_HISTORY = r'''
v['restored_complete_history']=[]
v['restored_complete_history_status']='recovery67..71 already accepted; no repeated startup reads'
optional_history_start_bytes=metadata_bytes
v['delivered_anchor_complete_window']=dict(status='unknown_no_same_execution_delivered_anchor')
after=p.get('after_step')
last=(v.get('latest_joint') or {}).get('outer')
if type(after) is int and type(last) is int and 0<=after<=last<=200:
 window=dict(after_step=after,through_step=last,rows=[],status='observed',
  maximum_new_outer_rows=5,anchor_row_status='delivered anchor remains local; not redundantly re-read',
  omitted=[],latest_joint_reused=False)
 start=max(after+1,last-4)
 for ordinal in range(after+1,start):
  window['omitted'].append(dict(outer=ordinal,reason='five_outer_history_limit'))
 def optional_document(root,name,ordinal):
  path=root/name
  if not path.exists():
   window['omitted'].append(dict(outer=ordinal,document=name,reason='missing_existing_metadata'))
   return None
  info=regular(path)
  remaining=MAX_METADATA-metadata_bytes
  if info.st_size>remaining:
   window['omitted'].append(dict(outer=ordinal,document=name,reason='metadata_read_budget',
    bytes=info.st_size,remaining_bytes=remaining))
   return None
  return document(root,name)
 # Prefer the newest facts when the optional history cannot all fit. The raw
 # latest joint was already read and checked in the unchanged required prefix.
 for ordinal in range(last,start-1,-1):
  root=b/'outers'/('outer-%06d'%ordinal)
  if ordinal==last:
   assert joint is not None and joint['outer']==ordinal
   digest=sha(root/'complete.json')
   assert digest==v['latest_joint']['proof']['sha256'],'Latest joint and immutable outer differ'
   value=joint
   window['latest_joint_reused']=True
  else:
   value=optional_document(root,'complete.json',ordinal)
   if value is None:continue
   digest=sha(root/'complete.json')
  assert value['run_id']==h['run_id'] and value['outer']==ordinal
  actor=value['actor'];assert actual_actor_boundary(actor,ordinal)
  rollout=value['rollout'];periodic=value['periodic_probe_update']
  row=dict(outer=ordinal,metadata_sha256=digest,
   actor=project(actor,('actual_adam_steps_per_rank','new_adam_steps_per_rank',
    'opportunities_processed','skipped_update_opportunities','scheduler_completed_outer')),
   reward_mean=actor.get('reward',{}).get('score_mean'),
   reward_grounding_round=actor.get('reward',{}).get('grounding_state',{}).get('round'),
   reward_grounding_version=actor.get('reward',{}).get('version'),
   selected_mixed_fraction=rollout.get('actual_selected_mixed_fraction'),
   selected_questions=rollout['questions'],selected_trajectories=rollout['trajectories'],
   periodic_probe_update=periodic,candidate_schedule=value.get('candidate_schedule'))
  if periodic:
   metrics=value['probe_metrics']
   row['probe_update']=dict(project(metrics,('outer','unselected_training_questions',
    'actual_learning_rate','new_adam_steps_per_rank','actual_adam_steps_per_rank',
    'train_loop_seconds')),before_update=metrics['before_update'],
    train=metrics['train'],test=metrics['test'])
   pool=optional_document(root/'full-pool','complete.json',ordinal)
   row['full_pool']=None if pool is None else project(pool,('questions','trajectories',
    'actual_selected_mixed_fraction'))
  window['rows'].append(row)
 window['rows'].sort(key=lambda item:item['outer'])
 if window['omitted']:window['status']='partial_existing_history_unknown'
 v['delivered_anchor_complete_window']=window
v['metadata_bytes_read']=metadata_bytes
v['readonly_metadata_budget']=dict(schema='bounded_optional_history_v1',
 maximum_bytes=MAX_METADATA,required_metadata_bytes=optional_history_start_bytes,
 final_metadata_bytes=metadata_bytes,maximum_new_outer_rows=5,
 required_checks_unchanged=True,read_guard_unchanged=True,output_guard_unchanged=True,
 omitted_history_is_unknown=True,training_mutations=False)
'''
OBSERVE = _required + BOUNDED_HISTORY + HISTORY_END + _finish
assert OBSERVE.startswith(_required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('observe',), required=True)
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--after-step', type=int)
    args = parser.parse_args()
    h = json.loads(args.handle.read_text())
    payload = dict(handle=h)
    anchor = json.loads((ROOT/'progress-checks/hourly-anchor.json').read_text())
    if args.after_step is not None:
        payload['after_step'] = args.after_step
    elif anchor.get('delivered') is True and all(
            anchor.get(k) == h[k] for k in ('run_id', 'execution_id')):
        payload['after_step'] = anchor['complete_outers']
    else:
        payload['after_step'] = h['resume_completed_outer']
    began = time.time()
    identity = dict(run_id=h['run_id'], execution_id=h['execution_id'],
                    source_commit=h['source_commit'], operator_path=str(Path(__file__).resolve()),
                    operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    required_checks_sha256=REQUIRED_CHECKS_SHA256)
    try:
        result = subprocess.run(previous.previous.route(OBSERVE),
            input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=65)
        receipt = dict(identity, action='observe', checked_unix=time.time(),
            elapsed_seconds=time.time()-began, exit_code=result.returncode,
            stdout=result.stdout.decode(), stderr=result.stderr.decode())
    except subprocess.TimeoutExpired as error:
        receipt = dict(identity, action='observe', checked_unix=time.time(),
            elapsed_seconds=time.time()-began, exit_code=None, timed_out=True,
            remote_action_outcome_unknown=True, no_training_mutation=True,
            stdout=(error.stdout or b'').decode(errors='replace'),
            stderr=(error.stderr or b'').decode(errors='replace'))
    out = ROOT/'preparation'/('a800-prefix5-bounded-observe-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    if receipt.get('timed_out') or receipt['exit_code']:
        raise RuntimeError('Inspect readonly receipt; training state remains unknown: '+str(out))
    if len(receipt['stdout'].encode()) > 128*1024:
        raise RuntimeError('Readonly output exceeds 128 KiB: '+str(out))
    print(json.dumps(dict(receipt=str(out), result=json.loads(receipt['stdout']))))


if __name__ == '__main__':
    main()
