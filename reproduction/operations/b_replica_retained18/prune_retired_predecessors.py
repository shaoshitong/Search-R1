"""Apply rotating retention to two stopped ancestors of this same B execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--current',required=True);args=p.parse_args()
    current=Path(args.current);parent=Path('/mnt/data/shaoshitong')
    assert current.parent==parent and current.name=='search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp174-retained18-recovery-20261020-r1'
    names=['search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp171-shared-recovery-20261020-r1',
           'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp64-h1005-20261019-r3']
    def read(f):return json.loads(f.read_text())
    def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
    def physical(s):
        prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
        return parent/s[len(prefix):]if s.startswith(prefix)else Path(s)
    j=read(current/'latest-joint.json');assert j['outer']>=177
    assert not(current/'logs/cleanup-status.json').exists()
    live=[j,read(current/'preparation/launch-config.json')]
    stage=read(current/'logs/pipeline-status.json');active=current/'preparation'/('outer-%06d-launch-config.json'%stage['outer'])
    if active.exists():live.append(read(active))
    archive=read(Path('/dataset/shaoshitong/search-g1-retired-anchor174-20261020-r1/relocation-complete.json'))
    assert archive['status']=='RETIRED_ACTOR_RELOCATED_SAME_HOST_SHA_VERIFIED'and archive['run_id']==j['run_id']
    assert all(Path(r['archived']).stat().st_size==r['bytes']for r in archive['assets'])
    def referenced(x,root):
        if isinstance(x,dict):return any(referenced(v,root)for v in x.values())
        if isinstance(x,list):return any(referenced(v,root)for v in x)
        return physical(x).is_relative_to(root)if isinstance(x,str)and x.startswith('/')else False
    expected=physical(live[1]['resume_joint']['source_base']);plan=[];anchors=[]
    for name in names:
        source=parent/name;assert source==expected
        c=read(source/'preparation/launch-config.json');expected=physical(c['resume_joint']['source_base'])
        latest=read(source/'latest-joint.json');anchor=latest['outer'];n=anchor-1
        assert read(source/'logs/cleanup-status.json')['owned_births_all_gone']
        assert latest['run_id']==j['run_id']and j['outer']>=anchor+3
        assert n in(170,173)and n not in(0,5,50,100,150,200)
        q=read(source/('preparation/recovery%d-cpu/full-state-proof.json'%anchor))
        assert q['outer']==anchor and q['status']=='DIVERSE_NATIVE_FULL_STATE_CPU_PASS'
        assert q['joint_sha256']==sha(source/'latest-joint.json')
        protected={physical(r['path'])for r in q['files']}
        old=read(source/'outers'/('outer-%06d'%n)/'complete.json');a=old['actor'];root=physical(a['state_dir'])
        assert root.parent==source/'actor/outers'and a['completed_outer']==n and old['run_id']==j['run_id']
        assert not referenced(live,root)
        assets=[(physical(r['path']),r)for r in a['rank_states']]
        hf=physical(a['hf_model_dir']);assets += [(hf/k,r)for k,r in a['hf_files'].items()if k.endswith(('.safetensors','.bin'))]
        assert len(a['rank_states'])==8 and len(assets)>=9
        for f,r in assets:
            assert f.is_relative_to(root)and f not in protected and not f.is_symlink()
            st=f.stat();assert st.st_size==r['bytes']
            plan.append(dict(path=str(f),outer=n,bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns,
                links=st.st_nlink,producer_sha256=r['sha256'],contents_rehashed=False))
        anchors.append(dict(source=str(source),retained_anchor=anchor,retired_predecessor=n))
    assert len({r['path']for r in plan})==len(plan)
    audit=current/'preparation/retired-predecessor170-173-prune-intent.json'
    value=dict(started_unix=time.time(),current_outer=j['outer'],run_id=j['run_id'],anchors=anchors,assets=plan,
        milestone_or_latest_checkpoint_deleted=False,trajectories_and_logs_deleted=False)
    with audit.open('x')as h:json.dump(value,h)
    assert read(current/'latest-joint.json')==j and not(current/'logs/cleanup-status.json').exists()
    for r in plan:
        f=Path(r['path']);st=f.stat();assert(st.st_size,st.st_ino,st.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns'])
        f.unlink()
    value.update(status='RETIRED_SAME_EXPERIMENT_NONMILESTONE_PREDECESSORS_PRUNED',completed_unix=time.time(),
        estimated_reclaimed_bytes=sum(r['bytes']for r in plan if r['links']==1),
        current_native_and_174_backup_untouched=True,data_free_bytes=shutil.disk_usage(current).free)
    with(current/'preparation/retired-predecessor170-173-pruned.json').open('x')as h:json.dump(value,h)
    for row in anchors:
        marker=Path(row['source'])/'outers'/('outer-%06d'%row['retired_predecessor'])/'retired-model-state-pruned.json'
        with marker.open('x')as h:json.dump(dict(current_native=str(current),audit=str(audit),
            outer=row['retired_predecessor'],native_weight_paths_retired=True,history_and_raw_evidence_retained=True),h)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
