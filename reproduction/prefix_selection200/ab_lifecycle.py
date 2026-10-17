"""Joint commit of independently clocked frozen/EMA probes and original actor."""
import os
import time
import json
import math
from collections import defaultdict
from pathlib import Path

from reproduction.prefix20_online.launch import read, sha, once, publish
from . import ab_policy as policy
from .contract import WORLD
from .storage import actual_actor_boundary


def boundary(config, probe, outer):
    return policy.validate_receipt(config, probe, outer)


def selection_metrics(root,selection_path):
    groups=read(selection_path)['selected'];labels=defaultdict(list)
    for line in (root/'full/own-em.jsonl').open():
        row=json.loads(line);labels[str(row['qid'])].append(row['strict_em'])
    if len(groups)!=512 or len(labels)!=512 or any(len(rows)!=5 for rows in labels.values()):
        raise ValueError('Selected metrics require exactly512 authentic labeled G5')
    if {str(g['qid']) for g in groups}!=set(labels):raise ValueError('Selected metrics label identities differ')
    counts=[sum(labels[str(g['qid'])]) for g in groups]
    return dict(questions=512,mixed=sum(0<k<5 for k in counts),all_wrong=counts.count(0),all_correct=counts.count(5),
        mixed_fraction=sum(0<k<5 for k in counts)/512,all_wrong_fraction=counts.count(0)/512,
        all_correct_fraction=counts.count(5)/512,mean_selection_q=math.fsum(g['q'] for g in groups)/512,
        no_uncontinued_candidate_accuracy_claim=True)


def check_update(config, joint, update):
    outer = joint['outer']; probe = joint['probe']
    boundary(config,probe,outer)
    if policy.mode(config)==policy.FROZEN:
        if update['status']!='PROBE_FROZEN_NO_OPTIMIZER_UPDATE' or update['new_adam_steps_per_rank']!=0:
            raise ValueError('Frozen arm cannot fit the selection probe')
    elif (update['status']!=policy.COMPLETE or update['checkpoint']!=probe
          or update['new_adam_steps_per_rank'] not in (0,policy.updates(outer))
          or (update['new_adam_steps_per_rank']==0 and update['sampling']['status']!='SKIPPED_MISSING_CLASS')
          or update['ema_updates']!=probe['actual_adam_steps_per_rank'][0]):
        raise ValueError('Balanced arm must complete one pass or record absent class')
    if update['outer']!=outer or update['actual_adam_steps_per_rank']!=probe['actual_adam_steps_per_rank']:
        raise ValueError('Probe update actual boundary differs')


def commit_outer(base,c,outer,root,actor_selection_path,timings,grounding_state):
    from .candidate_schedule import consumed_candidates_through, schedule_metadata
    from .periodic import is_probe_update
    from . import grounding
    actor,probe=read(base/'actor/latest.json'),read(base/'probe/latest.json')
    update_path=root/('probe-update/complete.json' if policy.mode(c)==policy.BALANCED else 'probe-frozen.json')
    update=read(update_path)
    joint=dict(run_id=c['run_id'],execution_id=c.get('execution_id'),outer=outer,actor=actor,probe=probe)
    check_update(c,joint,update)
    if not actual_actor_boundary(actor,outer) or actor['scheduler_completed_outer']!=outer:
        raise ValueError('Actor original20-opportunity boundary incomplete')
    previous=read(base/'latest-joint.json')['probe'] if outer>1 else dict(actual_adam_steps_per_rank=[0]*WORLD)
    if any(a-b!=update['new_adam_steps_per_rank'] for a,b in zip(probe['actual_adam_steps_per_rank'],previous['actual_adam_steps_per_rank'])):
        raise ValueError('Actual probe writes do not match previous joint')
    if policy.mode(c)==policy.BALANCED and sha(root/'probe-update/metrics.json')!=update['metrics_sha256']:
        raise ValueError('Probe metrics differ from complete receipt')
    pool=read(base/'question-pool/state.json');consumed=consumed_candidates_through(c,outer)
    if (pool['next_outer']!=outer+1 or pool['consumed_candidates']!=consumed
            or pool['cycle']*len(pool['order'])+pool['cursor']!=consumed):
        raise ValueError('Joint pool boundary differs from actual candidates')
    pool_snapshot=root/'pool-state-at-completion.json';once(pool_snapshot,pool)
    probe_snapshot=root/'probe-state-at-completion.pt';os.link(base/'probe/latest.pt',probe_snapshot)
    state=grounding.checked_state(c,grounding_state,require_adopted=True)
    if actor['grounding_state']!=state:raise ValueError('Actor grounding adopted state differs')
    grounding_snapshot=root/'grounding-state-at-completion.json';once(grounding_snapshot,state)
    joint.update(grounding=state,grounding_snapshot=str(grounding_snapshot),
        grounding_snapshot_sha256=sha(grounding_snapshot),
        grounding_refresh_due=outer if outer in grounding.REFIT_ROUNDS else None,
        probe_snapshot=str(probe_snapshot),pool_snapshot=str(pool_snapshot),pool_sha256=sha(pool_snapshot),
        selection_sha256=sha(actor_selection_path),actor_selection_path=str(actor_selection_path),
        actor_selection_sha256=sha(actor_selection_path),predicted_selection_sha256=sha(root/'probe-score/selection.json'),
        actor_selection_outcome_labels_used=is_probe_update(outer),rollout=read(root/'full/complete.json'),
        probe_metrics=update,periodic_probe_update=is_probe_update(outer),
        selection_probe_experiment=policy.mode(c),probe_boundary=boundary(c,probe,outer),
        selection_metrics=selection_metrics(root,actor_selection_path),
        source_commit=c['source_commit'],observed_unix=time.time(),candidate_schedule=schedule_metadata(c,outer),
        phase_timings={k:v for k,v in timings.items() if k.startswith('outer-%06d-'%outer)})
    once(root/'complete.json',joint);publish(base/'latest-joint.json',joint)
    return joint
