"""Retire only Actor177 of the stopped CP178 source after successor180 commits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--current',required=True);args=p.parse_args()
    parent=Path('/mnt/data/shaoshitong');current=Path(args.current)
    assert current.parent==parent and current.name=='search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp178-shared5-recovery-20261020-r1'
    source=parent/'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp174-retained18-recovery-20261020-r1'
    def read(f):return json.loads(f.read_text())
    def physical(s):
        prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
        return parent/s[len(prefix):]if s.startswith(prefix)else Path(s)
    def references(x,root):
        if isinstance(x,dict):return any(references(v,root)for v in x.values())
        if isinstance(x,list):return any(references(v,root)for v in x)
        return physical(x).is_relative_to(root)if isinstance(x,str)and x.startswith('/')else False
    j=read(current/'latest-joint.json');config=read(current/'preparation/launch-config.json')
    assert j['outer']>=180 and physical(config['resume_joint']['source_base'])==source
    assert not(current/'logs/cleanup-status.json').exists()
    stage=read(current/'logs/pipeline-status.json');active=current/'preparation'/('outer-%06d-launch-config.json'%stage['outer'])
    live=[j,config]
    if active.exists():live.append(read(active))
    anchor=read(source/'latest-joint.json');proof=read(source/'preparation/recovery178-cpu/full-state-proof.json')
    assert anchor['outer']==178 and anchor['run_id']==j['run_id']
    assert read(source/'logs/cleanup-status.json')['owned_births_all_gone']
    assert proof['outer']==178 and proof['status']=='DIVERSE_NATIVE_FULL_STATE_CPU_PASS'
    assert hashlib.sha256((source/'latest-joint.json').read_bytes()).hexdigest()==proof['joint_sha256']
    protected={physical(r['path'])for r in proof['files']}
    old=read(source/'outers/outer-000177/complete.json');a=old['actor'];root=physical(a['state_dir'])
    assert root.parent==source/'actor/outers'and a['completed_outer']==177 and old['run_id']==j['run_id']
    assert not references(live,root)
    # A successor's direct resume must have consumed the retained178 state.
    complete=read(current/'outers/outer-000180/complete.json')
    assert complete['actor']['completed_outer']==180
    counts=complete['actor']['actual_adam_steps_per_rank'];before=anchor['actor']['actual_adam_steps_per_rank']
    assert len(counts)==len(before)==8 and all(after>previous for after,previous in zip(counts,before))
    assets=[(physical(r['path']),r)for r in a['rank_states']]
    hf=physical(a['hf_model_dir']);assets += [(hf/k,r)for k,r in a['hf_files'].items()if k.endswith(('.safetensors','.bin'))]
    assert len(a['rank_states'])==8 and len(assets)==12
    plan=[]
    for f,r in assets:
        assert f.is_relative_to(root)and f not in protected and not f.is_symlink()
        st=f.stat();assert st.st_size==r['bytes'] and st.st_nlink==1
        plan.append(dict(path=str(f),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns,producer_sha256=r['sha256'],contents_rehashed=False))
    value=dict(started_unix=time.time(),current_outer=j['outer'],run_id=j['run_id'],source=str(source),retired_outer=177,retained_source_anchor=178,assets=plan,milestone_or_latest_checkpoint_deleted=False)
    audit=current/'preparation/retired-predecessor177-prune-intent.json'
    with audit.open('x')as h:json.dump(value,h)
    assert not(current/'logs/cleanup-status.json').exists()
    for r in plan:
        f=Path(r['path']);st=f.stat();assert(st.st_size,st.st_ino,st.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']);f.unlink()
    value.update(status='RETIRED_SAME_EXPERIMENT_PREDECESSOR177_PRUNED',completed_unix=time.time(),removed_bytes=sum(r['bytes']for r in plan),data_free_bytes=shutil.disk_usage(current).free,raw_and_logs_retained=True)
    with(current/'preparation/retired-predecessor177-pruned.json').open('x')as h:json.dump(value,h)
    with(source/'outers/outer-000177/retired-model-state-pruned.json').open('x')as h:json.dump(dict(current_native=str(current),audit=str(audit),outer=177,native_weight_paths_retired=True,history_and_raw_evidence_retained=True),h)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
