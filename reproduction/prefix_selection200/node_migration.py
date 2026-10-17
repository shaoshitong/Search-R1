"""Move an uncommitted first outer to another authorized H100 node.

Reuse the exact saved sampling and source paths through a private mount
namespace. Never import unsaved optimizer writes or modify another task.
"""
import json
import os
from pathlib import Path
import time
import uuid

from reproduction.prefix20_online.launch import read, sha, once


def prepare_first(source, staged_repo, commit, slot, target_uuids):
    from .ab_control import prepare
    from .startup_sampling_resume import validate_reused_sampling
    from .pool import state_sha
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    from reproduction.a800.owned_processes import same
    source=Path(source);old=read(source/'preparation/launch-config.json')
    receipt=source/'preparation/migrate-h10011-receipt.json'
    if receipt.exists():return read(receipt)
    if (source/'latest-joint.json').exists() or (source/'actor/latest.json').exists():
        raise ValueError('A completed boundary requires full joint recovery, never first-outer reset')
    stop=read(source/'preparation/migrate-h10011-human-stop.json')
    if stop['joint_at_request']!=0:raise ValueError('Not an uncommitted first outer')
    cleanup=read(source/'logs/cleanup-status.json');ledger=read(source/'logs/process-ledger.json')
    births=[r for j in ledger['jobs'].values()for r in j['ledger'].values()]
    if not cleanup['cleaned'] or birth_states(births) or same(read(source/'preparation/submit-receipt.json')['leader']):
        raise ValueError('Source writers must be gone before migration')
    head=read(source/'probe/latest.json')
    if head['completed_outer']!=0 or head['actual_adam_steps_per_rank']!=[0]*4:
        raise ValueError('Do not reset a trained probe')
    validate_reused_sampling(source,old)
    root=source/'outers/outer-000001';pool=read(source/'question-pool/state.json')
    if pool['next_outer']!=2 or pool['consumed_candidates']!=2560 or state_sha(pool)!=read(root/'pool-reservation.json')['after_sha256']:
        raise ValueError('Preserve the real reservation')
    attempts=[]
    for d in sorted((source/'actor/outers').glob('outer-0001-attempt-*')):
        counts=[]
        for rank in range(4):
            p=d/('rank-%d-updates.jsonl'%rank)
            rows=[json.loads(l)for l in p.open()]if p.exists()else []
            counts.append(sum(x.get('event')=='ACTUAL_ADAM_WRITE'for x in rows))
        attempts.append(dict(path=str(d),discarded_unsaved_writes_per_rank=counts))
    logs=[]
    for p in sorted((source/'logs').glob('*.log')):
        st=p.stat();logs.append(dict(path=str(p),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,sha256=sha(p)))
    proof=dict(status='HUMAN_NODE_MIGRATION_SOURCE_STOPPED',source=str(source),target_node='h100-11',
        cleanup=cleanup,original_logs_full_sha=logs,discarded_unsaved_attempts=attempts,
        committed_actor_outer=0,probe_adam=[0]*4,partial_updates_stitched=False,checked_unix=time.time())
    once(source/'preparation/migrate-h10011-source-proof.json',proof)
    base=Path(str(source)+'-h10011-r1')
    def rebase(v):
        if isinstance(v,dict):return {k:rebase(x)for k,x in v.items()}
        if isinstance(v,list):return [rebase(x)for x in v]
        if isinstance(v,str)and v.startswith(str(source)):return str(base)+v[len(str(source)):]
        return v
    c=rebase(old);ids=c['physical_gpu_indices'];uuids=[target_uuids[i]for i in ids]
    if len(set(target_uuids))!=8:raise ValueError('Need the actual target node eight UUIDs')
    c.update(base=str(base),execution_id=uuid.uuid4().hex,physical_gpu_uuids=uuids,auxiliary_gpu_uuids=uuids,
        expected_execution_node='h100-11',actor_global_micro_batch_size=4,
        node_migration=dict(source=str(source),source_stop_proof_sha256=sha(source/'preparation/migrate-h10011-source-proof.json'),
            target='h100-11',private_mount_namespace=True,model_through_mac=False))
    template=source/'preparation/migrate-h10011-template.json';once(template,c)
    result=prepare(template,staged_repo,commit,slot)
    for relative in ('probe','question-pool','outers/outer-000001/prefix','outers/outer-000001/probe-score','outers/outer-000001/full'):
        original=source/relative;target=base/relative;target.mkdir(parents=True,exist_ok=False)
        for parent,dirs,names in os.walk(original):
            for name in dirs:(target/Path(parent).relative_to(original)/name).mkdir()
            for name in names:
                p=Path(parent)/name
                if p.is_symlink():raise ValueError('Only regular immutable sampling files')
                os.link(p,target/p.relative_to(original))
    for name in ('pool-reservation.json','selection.json','questions.json'):
        os.link(root/name,base/'outers/outer-000001'/name)
    validate_reused_sampling(base,read(base/'preparation/launch-config.json'))
    result.update(node='h100-11',reused_real_sampling=True,complete_outer=0,
        global_micro=4,local_micro=1,accumulation=32,source_stop_proof=proof)
    once(receipt,result);return result
