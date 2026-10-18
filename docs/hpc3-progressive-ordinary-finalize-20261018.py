#!/usr/bin/env python3
"""Finalize one ordinary outer's metadata audit; never control training.

Run from this checkout, after real next-outer progress:
  python3 docs/hpc3-progressive-ordinary-finalize-20261018.py 64 --observer-session 4862

P10, milestone CPU payload readback, QA and final evaluation have separate
acceptance tools. This ordinary-outer journal audit cannot replace them.
Only scalar receipts and small audit reports cross the control connection.
"""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run

BASE = '/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3'
RUNTIME = '89c97ff2a5a60c7c3d6e1b0bf825d122f0ff874c'
REMOTE = r'''
import pathlib,json,hashlib,math,time
b=pathlib.Path(BASE);n=ctx['verified_outer'];root=b/'outers'/('outer-%06d'%n)
def read(p):return json.loads(p.read_text())
def sha(raw):return hashlib.sha256(raw).hexdigest()
j=read(root/'complete.json');prev=read(b/'outers'/('outer-%06d'%(n-1))/'complete.json')
a=j['actor'];selection=read(pathlib.Path(j['actor_selection_path']))
assert j['runtime_source_commit']==RUNTIME and not j['periodic_probe_update']
assert a['completed_outer']==n and a['complete_outer'] and a['scheduler_completed_outer']==n
assert not selection['outcome_labels_used']
from reproduction.prefix_selection200.progressive_prefix import boundary,actor_prefix
from reproduction.prefix_selection200.adaptive_candidates import derive
expected_probe=boundary(n);probe_outer=expected_probe['last_update_outer'];probe_steps=expected_probe['actual_adam_per_rank']
c=read(b/'preparation/launch-config.json');assert c['run_id']==j['run_id']
assert c['source_commit']==RUNTIME
actual_lr=1e-6*min((n-1)/57,1.)
assert j['probe']['completed_outer']==probe_outer
assert j['probe']['actual_adam_steps_per_rank']==[probe_steps]*4
assert j['probe_metrics']['new_adam_steps_per_rank']==0
assert selection['probe_before_checkpoint_sha256']==j['probe']['sha256']
assert j['rollout']['actor_version']==prev['actor']['actor_version']
prefix=read(root/'prefix/complete.json');features=[]
assert prefix['head_prefix_tokens']==actor_prefix(n)
assert prefix['scan_sha256']==j['rollout']['prefix_scan_sha256']==selection['prefix_scan_sha256']
questions=j['rollout']['scheduled_candidate_questions'];assert questions%4==0
for r in prefix['ranks']:
 assert r['actor_version']==prev['actor']['actor_version']
 assert r['actor_hidden_layer']==27 and r['actor_hidden_boundary']=='post_block_hidden_plus_residual'
 assert r['independent_encoder_forward'] is False and r['actual_compute_dtype']=='torch.bfloat16'
 assert r['trajectories']==questions*5//4 and r['actor_hidden_forward_calls']>0 and r['actor_hidden_captured_rows']>0
 features.append({k:r[k] for k in ('rank','actor_version','actor_hidden_layer','actor_hidden_boundary','trajectories','actor_hidden_forward_calls','actor_hidden_captured_rows','independent_encoder_forward','actual_compute_dtype')})
assert sorted(r['rank'] for r in features)==list(range(4))
reward=a['reward'];version=a['grounding_probe_version'];round_expected=min(150,((n-1)//50)*50)
assert a['grounding_state']['round']==round_expected and reward['actual_reward_world_size']==4
assert not reward['prefix_correctness_head_used_for_reward'] and len(reward['ranks'])==4
assert len(set(r['probe_url'] for r in reward['ranks']))==4
assert all(r['version']==version and not r['prefix_correctness_head_used_for_reward'] for r in reward['ranks'])
ranks=[]
for rank,r in enumerate(a['rank_receipts']):
 raw=pathlib.Path(r['ledger_path']).read_bytes();assert sha(raw)==r['ledger_sha256']
 events=[json.loads(line) for line in raw.splitlines() if line.strip()]
 restore=next(e for e in events if e.get('event')=='ACTUAL_RESTORE')
 writes=[e for e in events if e.get('event')=='ACTUAL_ADAM_WRITE']
 monitors=[e for e in events if e.get('event')=='POLICY_MONITOR']
 opportunities=[e for e in events if e.get('event')=='OPPORTUNITY']
 before=prev['actor']['actual_adam_steps_per_rank'][rank];after=a['actual_adam_steps_per_rank'][rank]
 skipped=a['skipped_update_opportunities'][rank];new=a['new_adam_steps_per_rank'][rank]
 assert restore['restored_completed_outer']==n-1 and restore['restored_scheduler']==n-1
 assert restore['actual_adam_before']==before and restore['actual_model_master_dtypes']==['torch.float32']
 assert set(restore['actual_learning_rates'])=={actual_lr}
 assert [e['actual_adam_steps'] for e in writes]==list(range(before+1,after+1))
 assert len(writes)==new and new+skipped==20 and new>=1
 assert {e['actual_learning_rate'] for e in writes}=={actual_lr}
 assert len(monitors)==40 and len(opportunities)==20 and r['scheduler_advances']==1
 assert r['scheduler_completed_outer']==n
 ranks.append(dict(rank=rank,restored_outer=n-1,restored_adam=before,restored_scheduler=n-1,
  master_dtypes=restore['actual_model_master_dtypes'],actual_learning_rate=actual_lr,
  first_actual_adam=writes[0]['actual_adam_steps'],last_actual_adam=writes[-1]['actual_adam_steps'],
  new_actual_updates=new,skipped=skipped,monitor_callbacks=len(monitors),
  fresh_observations=r['fresh_monitor_observations'],reused_observations=r['reused_monitor_observations'],
  all_monitor_allowed=all(e['allowed'] for e in monitors),opportunity_statuses=r['opportunity_statuses'],
  ledger_sha256=r['ledger_sha256']))
assert len(ranks)==4
nb=read(b/'outers'/('outer-%06d'%(n+1))/'candidate-budget.json')
mixed=j['rollout']['actual_selected_mixed_questions']
assert nb['previous_outer']==n and nb['previous_questions']==questions
assert nb['previous_selected_mixed_questions']==mixed
assert nb['previous_joint_sha256']==sha((root/'complete.json').read_bytes())
if nb['mode']=='fixed_probe_update':assert nb['questions']==2560 and (n+1)%10==0 and n+1<200
else:
 assert nb['mode']=='adaptive'
 den=1.1 if mixed==512 else mixed/512
 assert nb['questions']==max(512,min(5120,4*math.ceil(questions/(4*(den+1e-6)))))
assert nb==derive(c,n+1)
report=dict(ctx,status='ORDINARY_NATIVE_OUTER_AND_ACTUAL_NEXT_PROGRESS_ACCEPTED',
 run_id=j['run_id'],runtime_commit=RUNTIME,actor={k:a[k] for k in ('completed_outer','actual_adam_steps_per_rank','new_adam_steps_per_rank','scheduler_completed_outer','skipped_update_opportunities')},
 probe=j['probe'],native_rank_evidence=ranks,current_actor_feature_capture=features,
 selected_mixed=mixed,selected_mixed_fraction=j['rollout']['actual_selected_mixed_fraction'],
 candidate_questions=questions,next_budget=nb,grounding_round=round_expected,grounding_version=version,
 wall_seconds=j['observed_unix']-prev['observed_unix'],phase_timings=j['phase_timings'],
 checkpoint_independent_CPU_payload_scan_performed=False,full_goal_complete=False,goal_status='active')
p=b/'preparation'/('operator-native-outer-%06d-%s.json'%(n,ctx['operator_file_sha256'][:12]))
if p.exists():
 report=read(p);assert report['verified_outer']==n and report['operator_file_sha256']==ctx['operator_file_sha256']
else:p.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
raw=p.read_bytes();print(json.dumps(dict(report=report,mirror=dict(path=str(p),sha256=sha(raw),verified=True))))
'''


def save(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temporary = path.with_name(path.name + '.ordinary-finalize.tmp')
    temporary.write_text(raw)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outer', type=int)
    parser.add_argument('--observer-session', type=int, required=True)
    args = parser.parse_args()
    n = args.outer
    if not 51 <= n < 200 or n % 10 == 0:
        parser.error('Only ordinary progressive outers51..199; P10/milestones need their separate acceptance')
    from reproduction.hpc3_prefix5.observe import query_for
    address = subprocess.check_output(['ipconfig','getifaddr','en0'], text=True).strip()
    result = run(PYTHON+' -c '+shlex.quote(query_for(BASE)), account='mou176', source_address=address, timeout=25)
    if result.returncode:raise RuntimeError('Metadata read failed, not training terminal: '+result.stderr.decode()[-1000:])
    d=json.loads(result.stdout)
    if d['complete_outer']<n:
        print(json.dumps(dict(status='PENDING_NATIVE_JOINT',outer=n)));return
    assert d['job']=='733342' and d['source_publication']==RUNTIME
    assert '|RUNNING|' in d['queue'] and not d['error_lines'] and not d['failed_jobs']
    actual_next_complete = d['complete_outer'] >= n+1
    counts = [r.get('completed_trajectories', 0) for r in d.get('prefix', []) if r]
    positive = (d.get('status') or {}).get('outer')==n+1 and len(counts)==4 and min(counts)>0
    if not (actual_next_complete or positive):
        print(json.dumps(dict(status='PENDING_ACTUAL_NEXT_PROGRESS', outer=n)));return
    source = pathlib.Path(__file__)
    relative = str(source.relative_to(ROOT))
    commit = subprocess.check_output(['git','log','-1','--format=%H','--',relative], cwd=ROOT, text=True).strip()
    if not commit:raise RuntimeError('Publish this operator before its first remote execution')
    ctx = dict(verified_outer=n,epoch=d['epoch'],job=733342,queue=d['queue'],
               observed_complete_outer=d['complete_outer'],observed_stage=d['status'],
               next_outer_already_complete=actual_next_complete,actual_next_prefix_counts=counts,
               observer_session=args.observer_session,operator_source_commit=commit,
               operator_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    code = "import os;os.environ['DYNRPO_PROGRESSIVE_PREFIX']='cp50_prefix20_40_80_three_epoch_v1';os.environ['DYNRPO_PREFIX_TOKENS']="+repr(str(20 if n<=100 else 40 if n<=150 else 80))+"\nimport sys;sys.path.insert(0,"+repr(BASE+'/repo')+")\nBASE="+repr(BASE)+'\nRUNTIME='+repr(RUNTIME)+'\nctx='+repr(ctx)+'\n'+REMOTE
    address = subprocess.check_output(['ipconfig','getifaddr','en0'], text=True).strip()
    r = run(PYTHON+' -c '+shlex.quote(code), account='mou176', source_address=address, timeout=25)
    if r.returncode:raise RuntimeError('Ordinary audit/mirror failed; no training action taken: '+r.stdout.decode()[-1200:]+r.stderr.decode()[-1200:])
    value = next(json.loads(line) for line in reversed(r.stdout.decode().splitlines()) if line.lstrip().startswith('{'));report=value['report'];mirror=value['mirror']
    raw = (json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
    assert hashlib.sha256(raw).hexdigest()==mirror['sha256']
    path = OUTPUT/('progressive-ordinary-outer-%06d-finalization.json'%n);path.write_bytes(raw)
    save(OUTPUT/('progressive-ordinary-outer-%06d-mirror.json'%n), mirror)
    for name in ('current.json','progressive-goal-acceptance-ledger.json'):
        path_state=OUTPUT/name;s=json.loads(path_state.read_text()) if path_state.exists() else {}
        s.update(checked_unix=d['epoch'],goal_status='active',full_goal_complete=False,
                 status='PROGRESSIVE_TRAINING_RUNNING_WITH_VERIFIED_NATIVE_JOINT',job=d['job'],base=BASE,
                 runtime_source_commit=RUNTIME,pipeline=dict(status=d['status'],complete_outer=d['complete_outer'],
                     actor=d['actor'],probe=d['probe'],errors=d['error_lines']),
                 actual_complete_outer=d['complete_outer'],
                 actual_completed_outers=d['complete_outer'],
                 latest_native_acceptance=str(path),latest_native_accepted_outer=n,
                 last_goal_turn_classification='progress',last_turn_classification='progress',
                 observer_session=args.observer_session)
        if name=='current.json':s['formal_outers']=d['complete_outer']
        save(path_state,s)
    print(json.dumps(dict(status=report['status'],verified_outer=n,
          actual_adam_per_rank=report['actor']['actual_adam_steps_per_rank'],
          mixed=report['selected_mixed'],wall_seconds=report['wall_seconds'],
          next_questions=report['next_budget']['questions'],mirror=mirror)))


if __name__=='__main__':main()
