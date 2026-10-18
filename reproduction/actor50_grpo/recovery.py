"""Exact zero-write51 recovery: retain actor50, probe0, RNG and all sampled G5s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .bootstrap import read,sha,once,publish,digest,SOURCE_JOINT_SHA


def recover(base):
    base=Path(base);c=read(base/'preparation/launch-config.json');parent=Path(c['recovery_sampling']['source_base'])
    old=read(parent/'preparation/launch-config.json');assert old['run_id']==c['run_id']
    cleanup=read(parent/'logs/cleanup-status.json');assert cleanup['cleaned'] and cleanup['owned_births_all_gone']
    assert read(parent/'logs/exit.json')['success'] is False
    joint=read(parent/'latest-joint.json');assert joint['outer']==50
    actor=read(parent/'actor/latest.json');probe=read(parent/'probe/latest.json')
    assert actor==joint['actor'] and actor['actual_adam_steps_per_rank']==[1000]*4
    assert probe==joint['probe'] and probe['completed_outer']==50 and probe['actual_adam_steps_per_rank']==[0]*4
    attempt=c['recovery_sampling']['zero_write_attempt'];root=parent/'outers/outer-000051'
    ledgers=[]
    for rank in range(4):
        path=parent/'actor/outers'/('outer-0051-attempt-'+attempt)/('rank-%d-updates.jsonl'%rank)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert sum(row['event']=='ACTUAL_ADAM_WRITE' for row in rows)==0
        restored=next(row for row in rows if row['event']=='ACTUAL_RESTORE')
        assert restored['actual_adam_before']==1000 and restored['restored_completed_outer']==50
        ledgers.append(dict(rank=rank,path=str(path),sha256=sha(path),returned_writes=0))
    proof=read(parent/'preparation/branch-qualified.json')
    for row in proof['actor50_files']:
        s=Path(row['path']).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns'])
    assert proof['source_joint_sha256']==SOURCE_JOINT_SHA
    prefix=read(root/'prefix/complete.json');score=read(root/'probe-score/manifest.json');full=read(root/'full/complete.json')
    assert prefix['questions']==2560 and prefix['trajectories']==12800 and prefix['head_prefix_tokens']==20
    assert score['count']==12800 and score['head_max_tokens']==293 and score['probe_before_checkpoint_sha256']==probe['sha256']
    assert score['prefix_scan_sha256']==prefix['scan_sha256'] and full['prefix_scan_sha256']==prefix['scan_sha256']
    assert full['questions']==512 and full['trajectories']==2560 and score['selection_sha256']==full['selection_sha256']
    assert full['actor_version']==actor['actor_version'] and full['actor_version']==score['actor_version']
    selected=read(root/'probe-score/selection.json');assert selected['outcome_labels_used'] is False and selected['selected_questions']==512
    # Stream SHA on remote only; no tensor or trajectory payload is copied to Mac.
    linked=[]
    def link(source,target):
        source=Path(source);target=Path(target);assert source.resolve()==source and source.is_file() and not source.is_symlink()
        target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
        linked.append(dict(source=str(source),target=str(target),bytes=source.stat().st_size,sha256=sha(source)))
    for phase in ('prefix','probe-score','full'):
        files=sorted(path for path in (root/phase).rglob('*') if path.is_file())
        for i,path in enumerate(files):
            publish(base/'preparation/prepare-progress.json',dict(stage='VERIFYING_ZERO_WRITE51_SAMPLING_FILES',phase=phase,file=i,total=len(files),checked_unix=time.time()))
            link(path,base/'outers/outer-000051'/phase/path.relative_to(root/phase))
    for name in ('pool-reservation.json','selection.json','questions.json','actor-binding.json'):
        link(root/name,base/'outers/outer-000051'/name)
    # Start from the complete50 pool. prepare() applies the SAME sealed51 reservation once.
    once(base/'actor/latest.json',actor);once(base/'question-pool/state.json',read(joint['pool_snapshot']))
    once(base/'question-pool/binding.json',read(parent/'question-pool/binding.json'))
    assert sha(parent/'probe/latest.pt')==probe['sha256']
    link(parent/'probe/latest.pt',base/'probe/latest.pt');link(parent/'probe/latest.pt',base/'probe/head-outer-0050.pt')
    once(base/'probe/latest.json',probe)
    link(parent/'probe/initialization-complete.json',base/'probe/initialization-complete.json')
    for n in range(1,51):link(parent/'outers'/('outer-%06d'%n)/'complete.json',base/'outers'/('outer-%06d'%n)/'complete.json')
    once(base/'latest-joint.json',joint)
    record=dict(status='COMPLETE50_PLUS_ZERO_WRITE51_SAMPLING_IMPORTED',checked_unix=time.time(),source_base=str(parent),
        source_execution_id=old['execution_id'],source_commit=old['source_commit'],
        new_actor_updates_lost_or_reexecuted=0,new_probe_updates_lost_or_reexecuted=0,
        actor50_preserved=True,probe0_bytes_preserved=True,rank_Adam=[1000]*4,source_seed_probe_sha256=probe['sha256'],
        all_source_owned_writers_gone=True,ledgers=ledgers,files=linked,
        phase_hashes={phase:sha(base/'outers/outer-000051'/phase/('manifest.json' if phase=='probe-score' else 'complete.json')) for phase in ('prefix','probe-score','full')},
        actor_binding_sha256=sha(base/'outers/outer-000051/actor-binding.json'))
    once(base/'preparation/recovery-import.json',record)
    metadata=[base/x for x in ('actor/latest.json','probe/latest.json','question-pool/state.json','question-pool/binding.json','latest-joint.json','preparation/launch-config.json','preparation/recovery-import.json')]
    q=dict(proof,status='ACTOR50_WARMUP_EMA_BRANCH_CPU_QUALIFIED',source_commit=c['source_commit'],checked_unix=time.time(),
        recovery_import_sha256=sha(base/'preparation/recovery-import.json'),
        sealed_metadata=[dict(path=str(p),sha256=sha(p)) for p in metadata],
        actor_CUDA_restore='PENDING_REAL_RETRY51_WITH_SAME_STATE')
    once(base/'preparation/branch-qualified.json',q)
    return record


def inherited_phase(base,root,phase,c):
    marker=root/phase/('manifest.json' if phase=='probe-score' else 'complete.json')
    if not marker.exists():return False
    proof=read(Path(base)/'preparation/recovery-import.json')
    if root.name!='outer-000051' or proof['source_base']!=c['recovery_sampling']['source_base']:
        raise ValueError('Only the explicitly qualified zero-write51 event is inherited')
    if sha(marker)!=proof['phase_hashes'][phase]:raise ValueError('Inherited sampling metadata changed')
    return True


def inherited_actor_binding(base,root,c,actual):
    if root.name!='outer-000051':return False
    proof=read(Path(base)/'preparation/recovery-import.json');path=root/'actor-binding.json'
    assert sha(path)==proof['actor_binding_sha256'] and read(path)==actual
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    try:
        recover(base);once(base/'preparation/prepare-recover-exit.json',dict(success=True,checked_unix=time.time()))
    except BaseException as error:
        once(base/'preparation/prepare-recover-exit.json',dict(success=False,checked_unix=time.time(),error=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
