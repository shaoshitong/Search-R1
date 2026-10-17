"""Assign the approved arms to prepared slots, then use the original once launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from reproduction.prefix20_online.launch import read, sha, once
from . import ab_policy as policy


def prepare(template,staged_repo,commit,slot):
    from .launch import validate_config
    c=read(template);base=Path(c['base']);staged=Path(staged_repo)
    c.update(source_commit=commit,selection_probe_experiment=policy.FROZEN if slot=='A' else policy.BALANCED,
        probe_group_loss_weight=1. if slot=='A' else 5.,probe_ema_rate=None if slot=='A' else .99,
        experiment_assignment_pending=False,preparation_only=False,
        ab_goal='frozen_warmup_vs_current_outer_balanced_ema_200_v1',
        approved_prepared_template_sha256=sha(template),reference_replica_baseline_commit='7679bb8ced2c13922fa9f5c9050ed0ce7fa54dde')
    validate_config(c,base)
    config_path=base/'preparation/launch-config.json'
    if config_path.exists():
        if read(config_path)!=c:raise ValueError('Existing assignment differs; never overwrite/relaunch')
        return dict(status='EXISTING_ASSIGNED_ARM_NO_REPLAY',slot=slot,base=str(base),config_sha256=sha(config_path))
    # A partial assignment is reconciled explicitly, not deleted/replayed.
    base.mkdir(exist_ok=False)
    for name in ('repo','preparation','logs'):(base/name).mkdir()
    once(base/'preparation/assignment-intent.json',dict(slot=slot,commit=commit,template=str(template),
        template_sha256=sha(template),created_unix=time.time()))
    files=[]
    for source in sorted(staged.rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(staged)
        if '__pycache__' in relative.parts:continue
        if source.is_symlink():raise ValueError('Staged sources must be regular files')
        target=base/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,target)
        expected=sha(source);actual=sha(target)
        if actual!=expected:raise ValueError('Assigned source file full SHA differs')
        files.append(dict(path=str(relative),sha256=actual))
    once(base/'preparation/assigned-source-full-sha.json',dict(commit=commit,files=files))
    once(config_path,c)
    result=dict(status='APPROVED_AB_ARM_PREPARED_NOT_YET_SUBMITTED',slot=slot,base=str(base),
        run_id=c['run_id'],commit=commit,files=len(files),config_sha256=sha(config_path),
        cards=c['physical_gpu_indices'],probe_mode=policy.mode(c))
    once(base/'preparation/assignment-complete.json',result)
    return result


def observe(base,verified=0):
    from .launch import observe as original
    from .a800_runtime import query
    from .storage import actual_actor_boundary
    base=Path(base);v=original(base)
    config=read(base/'preparation/launch-config.json')
    joint=read(base/'latest-joint.json') if (base/'latest-joint.json').exists() else {}
    n=joint.get('outer',0);facts=[]
    for outer in range(verified+1,n+1):
        root=base/'outers'/('outer-%06d'%outer);j=read(root/'complete.json');actor=j['actor'];head=j['probe']
        if not actual_actor_boundary(actor,outer):raise ValueError('Original actor journal receipt incomplete')
        policy.validate_receipt(config,head,outer)
        ranks=[]
        for row in sorted(actor['rank_receipts'],key=lambda r:r['rank']):
            path=Path(row['ledger_path'])
            if sha(path)!=row['ledger_sha256']:raise ValueError('Actual actor ledger SHA differs')
            events=[json.loads(line) for line in path.open()]
            writes=[e for e in events if e['event']=='ACTUAL_ADAM_WRITE']
            before=row['actual_adam_steps']-row['new_adam_steps']
            if [e['actual_adam_steps'] for e in writes]!=list(range(before+1,row['actual_adam_steps']+1)):
                raise ValueError('Actual returned Adam writes not sequential')
            ranks.append(dict(rank=row['rank'],writes=len(writes),actual_adam_steps=row['actual_adam_steps'],ledger_sha256=sha(path)))
        update=j['probe_metrics'];probe_ranks=[]
        if update['new_adam_steps_per_rank']:
            for rank in range(4):
                path=root/'probe-update'/('rank-%d-updates.jsonl'%rank)
                rows=[json.loads(line) for line in path.open()]
                if (len(rows)!=update['new_adam_steps_per_rank'] or rows[-1]['actual_adam_steps']!=head['actual_adam_steps_per_rank'][rank]
                        or any(r['ema_updates']!=r['actual_adam_steps'] for r in rows)):
                    raise ValueError('Actual online/EMA probe clock differs')
                probe_ranks.append(dict(rank=rank,writes=len(rows),last_actual=rows[-1]['actual_adam_steps'],ledger_sha256=sha(path)))
        facts.append(dict(outer=outer,actor=ranks,probe=probe_ranks,probe_actual=head['actual_adam_steps_per_rank'],
            ema_updates=head.get('ema_updates',0),selection_metrics=j.get('selection_metrics'),phase_timings=j['phase_timings'],
            probe_metrics={k:update[k] for k in ('new_adam_steps_per_rank','sampling','before_update','after_update','train_loop_seconds') if k in update}))
    # Avoid downloading full indices, hidden, tensors or trajectory predictions.
    qa={}
    for outer in (5,50,100,150,200):
        path=base/'outers'/('outer-%06d'%outer)/'validation/complete.json'
        if path.exists():
            data=read(path);qa[str(outer)]={k:val for k,val in data.items() if k in ('status','count','qa_count','exact_match','em','f1','metrics','aggregate','questions') and not isinstance(val,list)}
    return dict(base=str(base),run_id=config['run_id'],mode=policy.mode(config),observed_unix=time.time(),
        native=query(base),complete_outer=n,actor_adam=joint.get('actor',{}).get('actual_adam_steps_per_rank'),
        probe_adam=joint.get('probe',{}).get('actual_adam_steps_per_rank'),stage=v['stage'],
        prefix=v.get('current_prefix'),continuation=v.get('current_continuation'),facts=facts,qa=qa,
        cleanup=v['cleanup'],pipeline_complete=v['pipeline_complete'],gpu=v['gpu_actual'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','submit','observe'))
    p.add_argument('--template');p.add_argument('--staged-repo');p.add_argument('--commit');p.add_argument('--slot',choices=('A','B'))
    p.add_argument('--base');p.add_argument('--verified',type=int,default=0);a=p.parse_args()
    if a.phase=='prepare':result=prepare(a.template,a.staged_repo,a.commit,a.slot)
    elif a.phase=='submit':
        from .launch import submit
        result=submit(a.base)
    else:result=observe(a.base,a.verified)
    print(json.dumps(result,allow_nan=False))
