"""Recover a zero-Adam first-prefix OOM with a conservative per-process KV cap."""
import os,pathlib,time,uuid
from reproduction.prefix20_online.launch import once,read,sha
from reproduction.a800.owned_processes import same
from reproduction.prefix_only_correctness_ab.launch import birth_states

def prepare(source,target,canonical,commit):
    source=pathlib.Path(source);target=pathlib.Path(target);old=read(source/'preparation/launch-config.json')
    ledger=read(source/'logs/process-ledger.json');births=[row for job in ledger['jobs'].values()for row in job['ledger'].values()]
    remaining=birth_states(births);assert not[r for r in remaining if r['state']!='Z']
    assert not same(read(source/'preparation/submit-receipt.json')['leader'])
    assert not(source/'actor').exists()and not(source/'latest-joint.json').exists()
    phase=source/'outers/outer-000001/prefix';logs=list(phase.glob('rank-*.log'))
    assert any(any(kind in f.read_text()for kind in ['torch.OutOfMemoryError: CUDA out of memory','No available memory for the cache blocks'])for f in logs)
    assert not(phase/'complete.json').exists()
    head=read(source/'probe/latest.json');assert head['actual_adam_steps_per_rank']==[0]*4
    def remap(v):
        if isinstance(v,str):return canonical+v[len(old['base']):]if v==old['base']or v.startswith(old['base']+'/')else v
        if isinstance(v,dict):return {k:remap(x)for k,x in v.items()}
        if isinstance(v,list):return [remap(x)for x in v]
        return v
    c=remap(old);c.update(source_commit=commit,execution_id=uuid.uuid4().hex,rollout_utilization_cap=.13,evaluation_utilization_cap=.13,retrieval_data_cache=old['retrieval_data_cache'],rollout_num_gpu_blocks_override=2048,evaluation_num_gpu_blocks_override=2048,rollout_max_batched_tokens=4096,rollout_max_num_seqs=64,evaluation_max_batched_tokens=4096,evaluation_max_num_seqs=64,evaluation_enforce_eager=True)
    # Preserve the consumed first reservation; never draw another candidate set.
    (target/'question-pool').mkdir()
    for name in ['binding.json','state.json']:os.link(source/'question-pool'/name,target/'question-pool'/name)
    root=target/'outers/outer-000001';root.mkdir(parents=True)
    for name in ['pool-reservation.json','selection.json','questions.json']:os.link(source/'outers/outer-000001'/name,root/name)
    (target/'probe').mkdir()
    for name in ['latest.pt','latest.json','head-outer-0000.pt','initialization-complete.json','initialization-intent.json']:os.link(source/'probe'/name,target/'probe'/name)
    assert sha(target/'probe/latest.pt')==head['sha256']
    once(target/'preparation/launch-config.json',c)
    proof=dict(source_base=str(source),run_id=c['run_id'],source_leader_gone=True,remaining_nonexecuting_zombies=remaining,actor_writes=0,probe_writes=0,pool_preserved_sha256=sha(target/'question-pool/state.json'),reservation_preserved_sha256=sha(root/'pool-reservation.json'),same_initial_probe_sha256=head['sha256'],failed_partial_generation_retained=True,partial_trajectories_not_stitched=True,old_partial_progress=[read(f)for f in phase.glob('rank-*-progress.json')],original_logs_full_sha=[dict(path=str(f),sha256=sha(f))for f in logs],only_memory_cap_changed=True,rollout_utilization_cap=.13,evaluation_utilization_cap=.13,created_unix=time.time())
    once(target/'preparation/zero-adam-partial-prefix-kv-repair.json',proof)
    return proof
