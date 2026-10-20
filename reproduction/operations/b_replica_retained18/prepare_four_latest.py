"""Prepare CP181 four-rank continuation; retain the complete eight-rank source."""
import argparse
from pathlib import Path
import time
import uuid
from reproduction.prefix_selection200.diverse_resume import read,sha,once
from reproduction.prefix_selection200.diverse_four_resume import MODE,STATUS,UNCHANGED,DISPATCH,prepare


def main():
    p=argparse.ArgumentParser()
    for name in('source','base','commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--outer',required=True,type=int);a=p.parse_args();s=Path(a.source);b=Path(a.base)
    assert a.outer==181
    original=read(s/'preparation/outer-000181-launch-config.json')
    def rebase(v):
        if isinstance(v,dict):return{k:rebase(x)for k,x in v.items()}
        if isinstance(v,list):return[rebase(x)for x in v]
        if isinstance(v,str)and(v==str(s)or v.startswith(str(s)+'/')):return str(b)+v[len(str(s)):]
        return v
    c=rebase(original)
    c.update(base=str(b),execution_id=uuid.uuid4().hex,source_commit=a.commit,
        expected_execution_node='UNASSIGNED',preparation_only=True,training_authorized=False,
        experiment_assignment_pending=False,world_size=4,actor_global_micro_batch_size=4,
        execution_hardware_layout='h100_four_rank_colocated_v1',physical_gpu_indices=[0,1,2,3],
        auxiliary_gpu_indices=[0,1,2,3],physical_gpu_uuids=original['physical_gpu_uuids'][:4],
        auxiliary_gpu_uuids=original['physical_gpu_uuids'][:4],retrieval_gpu_shards=4,
        grounding_physical_gpu=3,grounding_physical_gpu_by_rank=[0,1,2,3],grounding_refit_visible_device=3,
        retrieval_port=19710,search_url='http://127.0.0.1:19710/retrieve',grounding_probe_base_port=19720)
    c['resume_joint']=dict(adapter='b_diverse_4to8_cp50_v1',recovery_mode=MODE,source_base=str(s),completed_outer=181)
    for key in('resume_cached_sampling','runtime_compatibility','startup_sampling_resume'):c.pop(key,None)
    runtime={name:sha(b/'repo/reproduction/prefix_selection200'/name)for name in UNCHANGED+DISPATCH}
    for name in UNCHANGED:
        assert runtime[name]==sha(s/'repo/reproduction/prefix_selection200'/name),name
    proof=dict(status=STATUS,runtime_commit=a.commit,binding_commit=original['actor_checkpoint_binding_commit'],
        source_base=str(s),source_commit=original['source_commit'],runtime_sha256=runtime,
        scope='three narrow hardware/imported-boundary dispatches; actor/probe/objective/sampler scientific code unchanged',
        human_authorization='允许改4卡，保留并重分片完整状态',created_unix=time.time())
    ref=b/'preparation/runtime-compatibility.json';once(ref,proof)
    c['runtime_compatibility']=dict(path=str(ref),sha256=sha(ref));once(b/'preparation/launch-config.json',c)
    result=prepare(s,b,c,s/'preparation/recovery181-cpu/full-state-proof.json')
    from reproduction.prefix_selection200.launch import validate_config
    validate_config(c,b)
    ready=dict(status='LATEST_NATIVE_JOINT_PREPARED_AWAITING_TARGET',outer=181,next_outer=182,base=str(b),
        world_size=4,global_micro=4,local_micro=1,accumulation=32,prefix_tokens=80,
        actor=result['actor_adam'],probe=result['probe_adam'],optimizer_reset=False,training_started=False,
        source_commit=a.commit,ready_unix=time.time())
    once(b/'preparation/native-recovery-complete.json',ready)
    print(__import__('json').dumps(ready),flush=True)


if __name__=='__main__':main()
