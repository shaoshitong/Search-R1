"""Prepare the bound native allocation, then the extra frozen teacher warmup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def storage_admission(base,model):
    """Fresh account usage, not the multi-petabyte filesystem's free space."""
    from reproduction.common8.disk_admission import checkpoint_floor
    from search_g1.qprobe_warmup import write_once
    root=Path('/data/user/jxu957/experiments')
    print(json.dumps(dict(stage='hpc3_personal_storage_measurement_started',
        account_experiment_root=str(root),checked_unix=time.time(),slurm_job_id=os.environ.get('SLURM_JOB_ID'))),flush=True)
    result=subprocess.run(['du','-s','-B1',str(root)],capture_output=True,text=True,timeout=1800)
    if result.returncode or result.stderr.strip():raise ValueError('PERSONAL_STORAGE_USAGE_NOT_VERIFIED: '+result.stderr[-2000:])
    measured,path=result.stdout.strip().split('\t');measured=int(measured)
    if Path(path)!=root or measured<=0:raise ValueError('PERSONAL_STORAGE_AUDIT_INVALID')
    checkpoint=checkpoint_floor(model)
    # Model shards, FP32 Adam moments and a complete exported model; reserve all
    # five milestones, plus features, teacher cache, head/replay and rollout audits.
    reserve=4*len(checkpoint['save_steps'])*checkpoint['per_save_allowance_bytes']+96*2**30
    authorization=base/'preparation/personal-storage-authorization.json'
    limit=10**12
    if authorization.is_file():
        policy=json.loads(authorization.read_text())
        if policy.get('budget_bytes')!=2*10**12 or policy.get('scope')!='jxu957_native4_qprobe_only':
            raise ValueError('PERSONAL_STORAGE_AUTHORIZATION_DIFFERS')
        limit=policy['budget_bytes']
    proof=dict(account_experiment_root=str(root),measured_allocated_bytes=measured,
        measurement_stdout=result.stdout,measurement_stderr=result.stderr,
        checkpoint_upper_allowance_bytes=4*len(checkpoint['save_steps'])*checkpoint['per_save_allowance_bytes'],
        other_run_allowance_bytes=96*2**30,remaining_run_allowance_bytes=reserve,
        personal_budget_bytes=limit,projected_bytes=measured+reserve,
        budget_authorization=str(authorization) if authorization.is_file() else None,
        actual_filesystem_hard_quota_verified=False,
        accepted=measured+reserve<=limit,checked_unix=time.time(),
        scope='fresh prelaunch personal storage floor; actual growth still requires monitoring')
    write_once(base/'logs/storage-budget-admission.json',proof)
    if not proof['accepted']:raise ValueError('PERSONAL_1TB_STORAGE_RESERVATION_EXCEEDED')
    print(json.dumps(dict(stage='hpc3_personal_storage_admitted',**proof)),flush=True)
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();base=Path(a.base).resolve()
    from reproduction.hpc3_qprobe.control import own_base
    own_base(base)
    features=json.loads((base/'preparation/features-ready.json').read_text())
    if features['actual_visible_cuda_count']!=1 or features['completed_training_outers']!=0:
        raise ValueError('Fixed feature preparation is incomplete or mixed with actor training')
    from reproduction.common8.prepare_run import prepare
    model=Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725')
    storage_admission(base,model)
    binding=base/'preparation/checkpoint-restoration.json'
    restore=None
    if binding.is_file():
        value=json.loads(binding.read_text())
        from search_g1.full_state_restore import digest,source
        if value['source_receipt_sha256']!=digest(value['source_receipt']):
            raise ValueError('Explicit restoration source changed')
        _,receipt,_,_,_=source(value['source_receipt'])
        if receipt['world_size']!=4 or receipt['step']!=5 or value['target']!=str(base):
            raise ValueError('Native4 checkpoint5 restoration differs')
        restore=value['source_receipt']
    run=prepare(base,model,base/'preparation/official-model-manifest.json',a.commit,
        rollout_memory_utilization=0.1,
        local_retrieval_assets='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki',
        execution_site='hpc3',restore_receipt=restore)
    from reproduction.common8.qprobe_bind import bind
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    if restore is None:
        run=bind(base,base/'question-features',retriever_version=sha_json(ASSETS),
            exploration_questions=64,seed=20261013,teacher_warmup_cache=base/'teacher-warmup-cache')
    from reproduction.common8.configuration import resolved
    resolved(base)
    # The existing pipeline owns the one durable resource admission. Preparing
    # a second admission here would collide with its immutable receipt.
    print(json.dumps(dict(stage='hpc3_question_probe_configuration_prepared',run_id=run['run_id'],
        extra_frozen_teacher_warmup_outers=1,target_successful_training_outers=200,
        actual_training_outers_completed=0 if restore is None else run['full_state_restore']['completed_outer'],
        historical_teacher_reused=restore is not None,actual_cuda_restore_pending=restore is not None,
        actual_allocation_admission_pending=True)),flush=True)


if __name__=='__main__':main()
