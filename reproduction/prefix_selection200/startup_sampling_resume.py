"""Preserve actual first-round sampling after a zero-committed actor failure."""
import json
import os
from pathlib import Path
import time
import uuid

from reproduction.prefix20_online.launch import read,sha,once


def validate_reused_sampling(base,c):
    base=Path(base);spec=c['startup_sampling_resume'];root=base/'outers/outer-000001'
    if spec['actor_completed_outer']!=0 or spec['probe_actual_adam']!=[0]*4:
        raise ValueError('This adapter is only for the untouched first actor boundary')
    for phase,item in spec['phases'].items():
        path=root/phase/item['marker']
        if sha(path)!=item['sha256']:raise ValueError('Imported sampling phase marker changed')
        v=read(path)
        if v['run_id']!=c['run_id'] or v['outer']!=1:
            raise ValueError('Imported sampling belongs to another run/outer')
    full=read(root/'full/complete.json');score=read(root/'probe-score/manifest.json')
    selection=root/'probe-score/selection.json'
    selected=read(selection);score_complete=read(root/'probe-score/complete.json')
    if (full['actor_version']!=c['run_id']+'-official0' or full['trajectories']!=2560
            or score['actor_version']!=full['actor_version']
            or sha(selection)!=score['selection_sha256']
            or full['actor_selection_sha256']!=sha(selection)
            or selected['outcome_labels_used'] is not False or selected['selected_questions']!=512
            or score_complete['manifest_sha256']!=sha(root/'probe-score/manifest.json')
            or score['probe_before_checkpoint_sha256']!=read(base/'probe/latest.json')['sha256']):
        raise ValueError('Actual actor0 continuation/head provenance changed')
    return selection


def prepare(source,staged_repo,commit,slot):
    from .ab_control import prepare as assign
    from .pool import state_sha
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    from reproduction.a800.owned_processes import same
    source=Path(source);old=read(source/'preparation/launch-config.json')
    receipt_path=source/'preparation/actor-memory-repair-receipt.json'
    if receipt_path.exists():return read(receipt_path)
    if (source/'actor/latest.json').exists() or (source/'latest-joint.json').exists():
        raise ValueError('Do not discard an actually committed actor boundary')
    head=read(source/'probe/latest.json')
    if head['completed_outer']!=0 or head['actual_adam_steps_per_rank']!=[0]*4:
        raise ValueError('No online probe update can be reset')
    ledger=read(source/'logs/process-ledger.json');cleanup=read(source/'logs/cleanup-status.json')
    births=[r for job in ledger['jobs'].values() for r in job['ledger'].values()]
    if not cleanup['cleaned'] or birth_states(births) or same(read(source/'preparation/submit-receipt.json')['leader']):
        raise ValueError('Original native/owned writers must actually be gone')
    attempts=[];saw_oom=False
    for directory in sorted((source/'actor/outers').glob('outer-0001-attempt-*')):
        counts=[]
        for rank in range(4):
            path=directory/('rank-%d-updates.jsonl'%rank)
            rows=[json.loads(line)for line in path.open()] if path.exists() else []
            counts.append(sum(r.get('event')=='ACTUAL_ADAM_WRITE' for r in rows))
            failure=directory/('rank-%d-failure.json'%rank)
            if failure.exists():saw_oom |= read(failure).get('actual_cuda_oom') is True
        attempts.append(dict(path=str(directory),returned_unsaved_writes_per_rank=counts))
    if not saw_oom:raise ValueError('Require actual native CUDA OOM evidence for this memory repair')
    phases={};root=source/'outers/outer-000001'
    for phase,marker in (('prefix','complete.json'),('probe-score','complete.json'),('full','complete.json')):
        path=root/phase/marker;v=read(path)
        if v['run_id']!=old['run_id'] or v['outer']!=1:raise ValueError('Real sampling phases differ')
        phases[phase]=dict(marker=marker,sha256=sha(path))
    pool=read(source/'question-pool/state.json');reserved=read(root/'pool-reservation.json')
    if pool['next_outer']!=2 or pool['consumed_candidates']!=2560 or state_sha(pool)!=reserved['after_sha256']:
        raise ValueError('Preserve the actual pending first reservation exactly')
    logs=[]
    for path in sorted((source/'logs').glob('*.log')):
        st=path.stat();logs.append(dict(path=str(path),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,sha256=sha(path)))
    proof=dict(status='ACTUAL_ACTOR_OOM_FIRST_SAMPLING_RETAINED',run_id=old['run_id'],
        source_base=str(source),actor_completed_outer=0,probe_actual_adam=[0]*4,
        discarded_unsaved_attempts=attempts,phases=phases,pool_consumed=2560,
        original_logs_full_sha=logs,owned_births_actual_gone=True,cleanup=cleanup,observed_unix=time.time(),
        new_rollout_or_gpu_forward=False,partial_updates_stitched=False)
    once(source/'preparation/actor-memory-fault-proof.json',proof)
    base=Path(str(source)+'-actor-memory-repair1')
    def rebase(v):
        if isinstance(v,dict):return {k:rebase(x)for k,x in v.items()}
        if isinstance(v,list):return [rebase(x)for x in v]
        if isinstance(v,str)and v.startswith(str(source)):return str(base)+v[len(str(source)):]
        return v
    c=rebase(old);c.update(base=str(base),execution_id=uuid.uuid4().hex,
        actor_global_micro_batch_size=8,retriever_phase_residency='rollout_only',
        startup_sampling_resume=dict(source_base=str(source),actor_completed_outer=0,probe_actual_adam=[0]*4,
            fault_proof_sha256=sha(source/'preparation/actor-memory-fault-proof.json'),phases=phases))
    template=source/'preparation/actor-memory-repair-config-template.json';once(template,c)
    result=assign(template,staged_repo,commit,slot)
    # All source/target files are on the same existing NFS filesystem. Link the
    # authentic complete sampling, not newly generated replacement trajectories.
    for relative in ('probe','question-pool','outers/outer-000001/prefix','outers/outer-000001/probe-score','outers/outer-000001/full'):
        original=source/relative;target=base/relative;target.mkdir(parents=True,exist_ok=False)
        for parent,dirs,names in os.walk(original):
            for name in dirs:(target/Path(parent).relative_to(original)/name).mkdir()
            for name in names:
                p=Path(parent)/name
                if p.is_symlink():raise ValueError('Only original regular immutable sampling inputs')
                os.link(p,target/p.relative_to(original))
    for name in ('pool-reservation.json','selection.json','questions.json'):
        os.link(root/name,base/'outers/outer-000001'/name)
    validate_reused_sampling(base,read(base/'preparation/launch-config.json'))
    result.update(reused_real_sampling=True,original_actor_completed_outer=0,
        abandoned_unsaved_attempts=attempts,retriever_released_during_actor=True,
        global_micro=8,local_micro=2,accumulation=16,new_rollout_performed=False)
    once(receipt_path,result);return result
