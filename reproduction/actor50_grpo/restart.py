"""Human-authorized new experiment from actor50 and the preserved warmup56 seed.

Reuses sealed source qualification, never sampled51 data or superseded writes.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .bootstrap import read, sha, once, publish, SOURCE_BASE, SOURCE_JOINT_SHA, validate_config


def restart(base):
    base=Path(base);c=read(base/'preparation/launch-config.json')
    parent=Path(c['restart_from_actor50']['source_base']);source=Path(SOURCE_BASE)
    old=read(parent/'preparation/launch-config.json');proof=read(parent/'preparation/branch-qualified.json')
    assert read(parent/'logs/cleanup-status.json')['owned_births_all_gone']
    stopped=read(parent/'preparation/human-restart-mixed30.json')
    assert stopped['intent']['stop_execution_id']==old['execution_id']
    assert sha(source/'outers/outer-000050/complete.json')==SOURCE_JOINT_SHA
    for row in proof['actor50_files']:
        s=Path(row['path']).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns'])
    joint=read(source/'outers/outer-000050/complete.json');actor=joint['actor']
    assert actor['actual_adam_steps_per_rank']==[1000]*4 and actor['scheduler_completed_outer']==50
    initial=read(parent/'probe/initialization-complete.json');probe=initial['receipt']
    assert probe['completed_outer']==50 and probe['actual_adam_steps_per_rank']==[0]*4
    cp=parent/'probe/head-outer-0050.pt';assert cp.stat().st_size==probe['bytes'] and sha(cp)==probe['sha256']
    warmup=read(base/'preparation/warmup-parent-handle.json');wb=Path(warmup['base'])
    latest=read(wb/'probe/latest.json');handoff=read(wb/'preparation/human-handoff-to-grpo50.json')
    assert latest==handoff['latest'] and latest['sha256']==c['probe_source_sha256'] and latest['completed_epochs']==56
    validate_config(c,base)
    once(base/'actor/latest.json',actor);once(base/'question-pool/state.json',read(joint['pool_snapshot']))
    once(base/'question-pool/binding.json',read(source/'question-pool/binding.json'))
    (base/'probe').mkdir();os.link(cp,base/'probe/latest.pt');os.link(cp,base/'probe/head-outer-0050.pt')
    once(base/'probe/latest.json',probe);once(base/'probe/initialization-complete.json',initial)
    start=dict(joint,execution_id=c['execution_id'],source_commit=c['source_commit'],probe=probe,
        probe_snapshot=str(base/'probe/head-outer-0050.pt'),branch_start=True,periodic_probe_update=False,
        probe_boundary=dict(last_update_outer=50,actual_adam_per_rank=0),
        probe_metrics=dict(status='PRESERVED_WARMUP56_EMA_MODEL_ONLY_SEED',source_epoch=56,test=latest['metrics']['after']['test']),
        source_lineage=dict(actor50_joint_sha256=SOURCE_JOINT_SHA,warmup_checkpoint_sha256=latest['sha256'],
            source_actor50_unchanged=True,new_online_probe_Adam0=True,human_restart=True,
            superseded_execution=old['execution_id'],old51_sampling_reused=False))
    once(base/'latest-joint.json',start)
    for n in range(1,51):
        src=source/'outers'/('outer-%06d'%n)/'complete.json';dest=base/'outers'/('outer-%06d'%n)/'complete.json'
        dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest)
    record=dict(status='HUMAN_RESTART_FROM50_SAME_WARMUP56_NO_SAMPLING_REUSE',checked_unix=time.time(),
        source_base=str(parent),source_execution_id=old['execution_id'],source_joint_sha256=SOURCE_JOINT_SHA,
        actor50_files_unchanged=True,source_full_SHA_finite_qualification_reused=True,
        actor_actual_Adam_per_rank=[1000]*4,scheduler_completed_outer=50,probe_initial_sha256=probe['sha256'],
        online_probe_Adam_per_rank=[0]*4,pool_consumed=230400,new_next_outer=51,
        source_owned_writers_gone=True,old_partial_or_completed_updates_imported=False,
        old_sampling_events_imported=False,original_outputs_preserved=True)
    once(base/'preparation/restart-import.json',record)
    metadata=[base/x for x in ('actor/latest.json','probe/latest.json','question-pool/state.json',
        'question-pool/binding.json','latest-joint.json','preparation/launch-config.json','preparation/restart-import.json')]
    q=dict(proof,status='ACTOR50_WARMUP_EMA_BRANCH_CPU_QUALIFIED',source_commit=c['source_commit'],checked_unix=time.time(),
        sealed_metadata=[dict(path=str(p),sha256=sha(p)) for p in metadata],
        restart_import_sha256=sha(base/'preparation/restart-import.json'),CUDA_restore='PENDING_NEW_ACTOR51')
    once(base/'preparation/branch-qualified.json',q);return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    try:
        result=restart(base);once(base/'preparation/prepare-restart-exit.json',dict(success=True,checked_unix=time.time(),status=result['status']))
    except BaseException as e:
        once(base/'preparation/prepare-restart-exit.json',dict(success=False,checked_unix=time.time(),error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
