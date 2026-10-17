"""Rebind failed zero-write starts to an existing exact corpus/Arrow cache.

No training process is signalled. Original evidence and initialized head bytes
are retained; the selected raw/EMA warmup state is hardlinked, never reset.
"""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

from reproduction.prefix20_online.launch import read,sha,once


def prepare_retry(source,staged_repo,commit,slot,cache,output_parent=None,fault_kind="cache_space"):
    from reproduction.prefix_selection200.ab_control import prepare
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    from reproduction.prefix20_online.retriever import prepare_corpus
    from reproduction.a800.owned_processes import same, identity
    import datasets
    source=Path(source);cache=Path(cache);old=read(source/'preparation/launch-config.json')
    output=source/'preparation/zero-outer-cache-repair-receipt.json'
    if output.exists():return read(output)
    if fault_kind not in ('cache_space','kv_profile_capacity'):raise ValueError('Explicit actual startup fault required')
    if ((source/'actor/latest.json').exists() or (source/'latest-joint.json').exists()
            or (fault_kind=='cache_space' and (source/'question-pool/state.json').exists())):
        raise ValueError('This adapter may not discard any actor/pool/probe iteration')
    head=read(source/'probe/latest.json')
    if head['completed_outer']!=0 or head['actual_adam_steps_per_rank']!=[0]*4:
        raise ValueError('Only actual untouched warmup state is reusable here')
    cleanup=read(source/'logs/cleanup-status.json');ledger=read(source/'logs/process-ledger.json')
    births=[r for job in ledger['jobs'].values() for r in job['ledger'].values()]
    if not cleanup['cleaned'] or birth_states(births):
        raise ValueError('Original owned processes must actually be gone')
    leader=read(source/'preparation/submit-receipt.json')['leader']
    leader_state=identity(leader['pid']) if same(leader) else None
    if leader_state is not None and leader_state['state']!='Z':
        raise ValueError('Old native leader still runs; do not duplicate it')
    error=(source/'logs/retriever.log').read_text()
    if fault_kind=='cache_space' and 'OSError: Not enough disk space' not in error:
        raise ValueError('Require the actual HF cache-space fault, not an inferred failure')
    if fault_kind=='kv_profile_capacity':
        prefix=source/'outers/outer-000001/prefix'
        errors=[p.read_text(errors='replace') for p in prefix.glob('rank-*.log')]
        if not any('No available memory for the cache blocks' in t for t in errors):
            raise ValueError('Require the actual zero-KV-block constructor failure')
        for p in prefix.glob('rank-*-progress.json'):
            progress=read(p)
            if progress.get('actual_sampled_tokens',0) or progress.get('completed_trajectories',0):
                raise ValueError('This adapter cannot discard already generated trajectories')
        if list(prefix.glob('rank-*-complete.json')):
            raise ValueError('This adapter is restricted to pre-generation capacity faults')
    corpus,corpus_proof=prepare_corpus(old['retrieval_corpus'],cache)
    infos=list((cache/'arrow').glob('**/dataset_info.json'))
    if len(infos)!=1:raise ValueError('Use one already complete exact JSON Arrow dataset')
    info=read(infos[0]);expected=info['splits']['train']['num_examples']
    data=datasets.load_dataset('json',data_files=str(corpus),split='train',num_proc=4,cache_dir=str(cache/'arrow'))
    if len(data)!=expected or not data.cache_files:raise ValueError('Existing corpus cache is incomplete')
    for row in data.cache_files:
        p=Path(row['filename'])
        if not p.is_file() or cache not in p.parents:raise ValueError('Unexpected corpus cache asset')
    logs=[]
    log_paths=list((source/'logs').glob('*.log'))
    if fault_kind=='kv_profile_capacity':log_paths+=list((source/'outers/outer-000001/prefix').glob('rank-*.log'))
    for path in sorted(log_paths):
        st=path.stat();logs.append(dict(path=str(path),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,sha256=sha(path)))
    proof=dict(status='ACTUAL_ZERO_OUTER_HF_CACHE_FAULT_OWNED_GONE_EXISTING_CORPUS_REUSED',
        source_base=str(source),fault_kind=fault_kind,original_logs_full_sha=logs,cleanup=cleanup,second_actual_owned_birth_check=[],
        old_native_leader_state=leader_state,
        actor_actual_writes=0,probe_actual_writes=0,pool_consumed=0,original_head=head,
        corpus_cache=str(cache),corpus_provenance=corpus_proof,documents=len(data),
        arrow_files=[dict(path=row['filename'],bytes=Path(row['filename']).stat().st_size)for row in data.cache_files],
        checked_unix=time.time(),new_rollout_or_gpu_forward=False)
    once(source/'preparation/zero-outer-cache-fault-proof.json',proof)
    from reproduction.prefix_selection200.h100_replica_hardware import SHARED_PARENT
    if output_parent is not None and str(output_parent)!=SHARED_PARENT:
        raise ValueError('Use the explicitly inventoried existing H100-10 shared namespace only')
    suffix='-cache-repair1' if fault_kind=='cache_space' else '-kv-profile-repair1'
    base=(Path(output_parent)/(source.name+suffix) if output_parent else Path(str(source)+suffix))
    def rebase(value):
        if isinstance(value,dict):return {k:rebase(v)for k,v in value.items()}
        if isinstance(value,list):return [rebase(v)for v in value]
        if isinstance(value,str) and value.startswith(str(source)):return str(base)+value[len(str(source)):]
        return value
    c=rebase(old);c.update(base=str(base),execution_id=uuid.uuid4().hex,
        retrieval_data_cache=str(cache),zero_outer_cache_fault_source=str(source),
        zero_outer_cache_fault_proof_sha256=sha(source/'preparation/zero-outer-cache-fault-proof.json'))
    if fault_kind=='kv_profile_capacity':
        c.update(rollout_max_num_seqs=256,rollout_max_batched_tokens=8192,
                 actual_kv_profile_capacity_repair=True,
                 unchanged_generation_budget_and_sampling_parameters=True)
    template=source/'preparation/cache-repair-config-template.json';once(template,c)
    prepared=prepare(template,staged_repo,commit,slot)
    (base/'probe').mkdir()
    import hashlib
    copied={}
    for name in ('latest.pt','latest.json','head-outer-0000.pt','initialization-complete.json','initialization-intent.json'):
        src=source/'probe'/name;dst=base/'probe'/name;st=src.stat();key=(st.st_dev,st.st_ino)
        if key in copied:
            os.link(copied[key],dst)
        elif st.st_dev==dst.parent.stat().st_dev:
            os.link(src,dst);copied[key]=dst
        else:
            h=hashlib.sha256()
            with src.open('rb') as inp,dst.open('xb') as out:
                for chunk in iter(lambda:inp.read(8*1024**2),b''):
                    h.update(chunk);out.write(chunk)
                out.flush();os.fsync(out.fileno())
            if sha(dst)!=h.hexdigest():raise ValueError('Remote source/target initialization state SHA differs')
            copied[key]=dst
    if read(base/'probe/latest.json')!=head:raise ValueError('Reused initial model state differs')
    if fault_kind=='kv_profile_capacity':
        target=base/'question-pool';target.mkdir()
        for name in ('binding.json','state.json'):
            os.link(source/'question-pool'/name,target/name)
        before=read(target/'state.json');reservation=read(source/'outers/outer-000001/pool-reservation.json')
        from reproduction.prefix_selection200.pool import state_sha
        if (before['next_outer']!=2 or before['consumed_candidates']!=2560
                or state_sha(before)!=reservation['after_sha256']):
            raise ValueError('Preserve the exact original first2560 candidate reservation')
        root=base/'outers/outer-000001';root.mkdir(parents=True)
        for name in ('pool-reservation.json','selection.json','questions.json'):
            os.link(source/'outers/outer-000001'/name,root/name)
    result=dict(prepared,source_base=str(source),cache=str(cache),fault_kind=fault_kind,reused_warmup_actual_adam=[0]*4,
        original_head_sha256=head['sha256'],cache_documents=len(data),submitted=False)
    once(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--staged-repo',required=True)
    p.add_argument('--commit',required=True);p.add_argument('--slot',choices=('A','B'),required=True);p.add_argument('--cache',required=True);p.add_argument('--output-parent');p.add_argument('--fault-kind',choices=('cache_space','kv_profile_capacity'),default='cache_space')
    a=p.parse_args();print(json.dumps(prepare_retry(a.source,a.staged_repo,a.commit,a.slot,a.cache,a.output_parent,a.fault_kind)))
