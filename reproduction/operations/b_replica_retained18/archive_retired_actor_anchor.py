"""Relocate a retired recovery actor on retained18, after verified local copies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--current',required=True)
    parser.add_argument('--archive',required=True)
    args=parser.parse_args()
    source,current,archive=map(Path,(args.source,args.current,args.archive))
    parent=Path('/mnt/data/shaoshitong')
    assert socket.gethostname()=='h100-10' and source.parent==current.parent==parent
    assert all(x.name.startswith('search-g1-prefix-selection200-')for x in(source,current))
    assert archive.parent==Path('/dataset/shaoshitong') and archive.name.startswith('search-g1-retired-anchor')
    def read(p):return json.loads(p.read_text())
    def physical(s):
        prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
        return parent/s[len(prefix):]if s.startswith(prefix)else Path(s)
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb')as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()
    old=read(source/'latest-joint.json');n=old['outer'];assert n not in(0,5,50,100,150,200)
    assert read(source/'logs/cleanup-status.json')['owned_births_all_gone']
    j=read(current/'latest-joint.json');assert j['run_id']==old['run_id']and j['outer']>=n+2
    config=read(current/'preparation/launch-config.json')
    assert physical(config['resume_joint']['source_base'])==source
    assert config['resume_joint']['completed_outer']==n
    assert not(current/'logs/cleanup-status.json').exists()
    actor_dir=physical(old['actor']['state_dir']);assert actor_dir.parent==source/'actor/outers'
    live=[j,config]
    stage=read(current/'logs/pipeline-status.json');active=current/'preparation'/('outer-%06d-launch-config.json'%stage['outer'])
    if active.exists():live.append(read(active))
    def references(x):
        if isinstance(x,dict):return any(references(v)for v in x.values())
        if isinstance(x,list):return any(references(v)for v in x)
        return isinstance(x,str)and physical(x).is_relative_to(actor_dir)if isinstance(x,str)and x.startswith('/')else False
    assert not references(live),'Active native state/config still references the retired actor'
    # The current complete state and its actually restored predecessor remain.
    for row in j['actor']['rank_receipts']:
        f=physical(row['ledger_path']);assert digest(f)==row['ledger_sha256']
        events=[json.loads(s)for s in f.read_text().splitlines()]
        restored=next(r for r in events if r['event']=='ACTUAL_RESTORE')
        assert restored['restored_completed_outer']>n
    for row in j['actor']['rank_states']:
        assert physical(row['path']).stat().st_size==row['bytes']
    assets=[(physical(row['path']),row)for row in old['actor']['rank_states']]
    hf=physical(old['actor']['hf_model_dir'])
    assets += [(hf/name,row)for name,row in old['actor']['hf_files'].items()if name.endswith(('.safetensors','.bin'))]
    assert len(old['actor']['rank_states'])==8 and len(assets)>=9
    assert shutil.disk_usage(archive.parent).free>sum(row['bytes']for _,row in assets)
    archive.mkdir(exist_ok=False);plan=[]
    for f,row in assets:
        assert f.is_relative_to(actor_dir)and not f.is_symlink()and f.stat().st_size==row['bytes']
        st=f.stat();target=archive/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        with f.open('rb')as inp,target.open('xb')as out:shutil.copyfileobj(inp,out,8*1024**2)
        assert target.stat().st_size==row['bytes']and digest(target)==row['sha256']
        plan.append(dict(original=str(f),archived=str(target),bytes=st.st_size,inode=st.st_ino,
            mtime_ns=st.st_mtime_ns,sha256=row['sha256']))
    proof=dict(status='RETIRED_ACTOR_ALL_COPIES_SHA_VERIFIED',old_outer=n,new_outer=j['outer'],
        run_id=j['run_id'],verified_unix=time.time(),assets=plan,other_native_states_deleted=False)
    (archive/'verified-copy.json').write_text(json.dumps(proof))
    # Recheck the live boundary and exact source identities before releasing space.
    assert read(current/'latest-joint.json')==j and not(current/'logs/cleanup-status.json').exists()
    assert not references([read(current/'latest-joint.json'),read(active)if active.exists()else config])
    intent=source/'preparation'/('retired-actor-%d-relocation-intent.json'%n)
    with intent.open('x')as h:json.dump(proof,h)
    for row in plan:
        f=Path(row['original']);st=f.stat()
        assert(st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns'])
        f.unlink()
    proof.update(status='RETIRED_ACTOR_RELOCATED_SAME_HOST_SHA_VERIFIED',completed_unix=time.time(),
        original_actor_weight_paths_retired=True,restore_requires_copying_archived_files_to_original_paths=True,
        active_latest_and_previous_native_untouched=True,all_milestone_checkpoints_untouched=True,
        data_volume_free_bytes=shutil.disk_usage(source).free,archive_volume_free_bytes=shutil.disk_usage(archive).free)
    with(source/'preparation'/('retired-actor-%d-relocation-complete.json'%n)).open('x')as h:json.dump(proof,h)
    (archive/'relocation-complete.json').write_text(json.dumps(proof))
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
