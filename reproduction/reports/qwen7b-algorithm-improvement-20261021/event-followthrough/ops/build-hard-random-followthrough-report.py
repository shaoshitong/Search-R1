"""Summarize only persisted small, actual control/audit observations."""
import hashlib,json,pathlib,time
O=pathlib.Path(__file__).resolve().parent;P=O/'reports'
def read(p):return json.loads(p.read_text())
def fp(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
selection=read(P/'training-enrichment-independent-audit-final.json')
acceptance=read(P/'hard-random477-independent-cp32-audit.json')
assert selection['status'].endswith('INDEPENDENT_PASS') and acceptance['status'].endswith('INDEPENDENT_PASS')
snapshots=[read(P/('hard-random477-actual-%d.json'%n)) for n in (5,6,7)]
variants={}
for name in ('targeted','random_control'):
 rows=[]
 for state in snapshots:
  v=state['variants'][name];progress=v['progress.json']
  assert 'failure.json' not in v and progress['actor_updates']==0 and progress['steps']==14418+progress['cursor']
  rows.append(dict(observed_epoch=state['observed_epoch'],cursor=progress['cursor'],steps=progress['steps']))
 assert all(b['cursor']>a['cursor'] for a,b in zip(rows,rows[1:]))
 intervals=[dict(elapsed_seconds=b['observed_epoch']-a['observed_epoch'],
    actual_accepted_updates=b['cursor']-a['cursor'],
    seconds_per_update=(b['observed_epoch']-a['observed_epoch'])/(b['cursor']-a['cursor']))
    for a,b in zip(rows,rows[1:])]
 v=acceptance['variants'][name]
 assert v['latest_checkpoint']['cursor']>=32 and v['accepted_complete_updates']>32
 variants[name]=dict(job_id=v['job_id'],actual_progress_observations=rows,measured_intervals=intervals,
    accepted_complete_updates_at_independent_audit=v['accepted_complete_updates'],
    actual_adam_steps_at_independent_audit=v['actual_adam_steps'],first_checkpoint=v['first_checkpoint'],
    latest_complete_checkpoint=v['latest_checkpoint'],first_update_double_replay_passed=True,
    real_next_updates_after_CP32=True)
summary=dict(status='E4_HARD_RANDOM_ACTUAL_GPU_REPLAY_FULL_CP32_AND_CONTINUED_UPDATES_VERIFIED',
    observed_epoch=time.time(),scoring_questions=32768,scoring_trajectories=163840,
    selection_complete=selection['selection_complete'],variants=variants,
    common_start=dict(source_epoch=4,source_adam_steps=14418,raw_EMA_Adam_preserved=True,
        original_four_rank_RNG_archived=True,canonical_source_RNG_rank=0,four_rank_bitexact_resume_claim=False),
    controls=dict(questions_per_branch=22896,common_core_questions=6528,supplement_questions=16368,
        global_question_batch=48,update_budget=477,epochs=1,lr=5e-6,ema_decay=.99,
        trajectory_BCE_weight=1,three_event_NLL_weight=1,inter_group_rank_weight=1,
        same_macro_class_counts=True,Mixed_internal_K_distribution_matched=False,
        untouched_final_read=False,actor_updates=0),
    scheduler=dict(partition='acd_u',qos='normal',gpus_per_branch=1,dependency=None),
    complete_training_and_evaluation=False,claim_of_improved_Mixed_accuracy=False,
    audits={p.name:fp(p) for p in [P/'training-enrichment-independent-audit-final.json',
        P/'hard-random477-independent-first-audit-final.json',P/'hard-random477-independent-cp32-audit.json']})
path=P/'hard-random477-followthrough-final.json';path.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(status=summary['status'],summary=fp(path),variants={k:dict(job_id=v['job_id'],
    updates=v['accepted_complete_updates_at_independent_audit'],saved_cursor=v['latest_complete_checkpoint']['cursor']) for k,v in variants.items()},
    training_complete=False)))
