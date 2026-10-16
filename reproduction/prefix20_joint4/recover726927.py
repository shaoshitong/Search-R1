"""Fresh native4 target for the observed pre-training GPU inventory failure."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha,atomic
from reproduction.prefix20_joint4.control import (
    BASE,ASSETS,own_base,publish,verify_source,bootstrap,assets_ready,storage_admission,
)
from reproduction.prefix20_joint4.storage_complete import validate_upper_bound

FAILED=Path(BASE)
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016')
OLD_COMMIT='1d1ba467717c3208fbbe0c4ce5a3ce31e84d93fc'
JOB='726927'
EVIDENCE_SHA='10010b262eb62dfb7e8223e9e71cfeb1ff835031c04c9bdc2c37eacc5d480141'


def pristine_failure():
    if os.getuid()!=14090 or (FAILED/'job.id').read_text().strip()!=JOB:
        raise ValueError('PREFIX20_RECOVERY_ORIGINAL_OWNER_JOB_DIFFERS')
    original=FAILED/'preparation/recovery726927-original-evidence.json'
    if sha(original)!=EVIDENCE_SHA:raise ValueError('PREFIX20_RECOVERY_ORIGINAL_EVIDENCE_SHA_DIFFERS')
    prior=json.loads(original.read_text())
    if prior['bound_source']!=OLD_COMMIT or not prior['gpu_job_lock_free'] or any(prior['forbidden'].values()):
        raise ValueError('PREFIX20_RECOVERY_PRIOR_PIPELINE_WORK')
    submitted=json.loads((FAILED/'preparation/training-submission.json').read_text())
    prepared=json.loads((FAILED/'logs/prefix20-configuration-prepared.json').read_text())
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    expected=dict(JobId=JOB,JobState='RUNNING',Account='jxu957',QOS='formal-user',Partition='emergency_acd',
        UserId='jxu957(14090)',TresPerNode='gres/gpu:4',NumNodes='1',NumCPUs='48',
        Command=str(FAILED/'source'/OLD_COMMIT/'reproduction/prefix20_joint4/train.sh'),WorkDir=str(FAILED/'repo'))
    binding=prepared['actual_slurm_job_binding']
    if (any(binding.get(k)!=v for k,v in expected.items()) or submitted['job']!=JOB or
            submitted['returncode']!=0 or submitted['source_commit']!=OLD_COMMIT or
            run['source_commit']!=OLD_COMMIT or run['run_id']!=prior['run_id'] or run['completed_updates']!=0):
        raise ValueError('PREFIX20_RECOVERY_ORIGINAL_SUBMISSION_BINDING_DIFFERS')
    terminal=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Account,QOS,JobName,Start,End,NodeList'],capture_output=True,text=True,timeout=30)
    expected_line=JOB+'|FAILED|1:0|jxu957|14090|jxu957|formal-user|qprobe-prefix20-joint200|2026-10-16T20:52:55|2026-10-16T20:54:13|ACD1-32'
    if terminal.returncode or expected_line not in terminal.stdout.splitlines() or not any(
            x.startswith(JOB+'.extern|COMPLETED|0:0|') for x in terminal.stdout.splitlines()):
        raise ValueError('PREFIX20_RECOVERY_ACTUAL_TERMINAL_ACCOUNTING_DIFFERS')
    queue=subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i'],capture_output=True,text=True,timeout=20)
    if queue.returncode or JOB in queue.stdout.split():raise ValueError('PREFIX20_RECOVERY_SOURCE_QUEUE_NOT_TERMINAL')
    forbidden=['logs/process-ledger.json','logs/training-launch-claim.json','logs/search-g1-common8-training.log',
        'logs/pipeline-status.json','artifacts/question-probe','checkpoints']
    if any((FAILED/x).exists() for x in forbidden) or any((FAILED/'artifacts').iterdir()):
        raise ValueError('PREFIX20_RECOVERY_ACTOR_PROBE_OR_CHECKPOINT_WORK_EXISTS')
    text=(FAILED/('logs/training-'+JOB+'.log')).read_text()
    if ('minor_number' not in text or 'returned non-zero exit status 2' not in text or
            'native_allocation.py' not in text):
        raise ValueError('PREFIX20_RECOVERY_FAILURE_IS_NOT_OBSERVED_QUERY_ERROR')
    with (FAILED/'gpu-job.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source,_=verify_source(FAILED,OLD_COMMIT)
    bound=json.loads((FAILED/'preparation/run-source.json').read_text())
    if bound['source_commit']!=OLD_COMMIT:raise ValueError('PREFIX20_RECOVERY_OLD_REPO_BINDING_DIFFERS')
    for name,row in bound['files'].items():
        if sha(FAILED/'repo'/name)!=row['sha256']:raise ValueError('PREFIX20_RECOVERY_OLD_RUNTIME_CHANGED')
    return dict(job=JOB,run_id=run['run_id'],terminal=terminal.stdout,queue=queue.stdout,
        gpu_job_lock_free=True,completed_outers=0,actor_probe_pipeline_never_started=True,
        original_evidence=dict(path=str(original),sha256=EVIDENCE_SHA),old_source=str(source),
        original_run_manifest_sha256=sha(FAILED/'logs/run-manifest.json'),
        original_training_log_sha256=sha(FAILED/('logs/training-'+JOB+'.log')),checked_unix=time.time())


def relocate_global_storage(original_base,target,additional_prelaunch_reserve_bytes=2*2**30):
    """Copy actual global observation bytes; preserve its original timestamps.

    No new walk or fresher measurement is claimed. Reserve another 2GiB for
    this fresh target's input/source copies, beyond the unchanged run reserve.
    """
    source=original_base/'preparation/personal-storage-observation.json'
    original=json.loads(source.read_text());validate_upper_bound(original,original_base)
    directory=target/'preparation/copied-global-storage-observation';directory.mkdir()
    def raw(row,tag):
        value=copy.deepcopy(row)
        for key in ('stdout','stderr'):
            old=Path(row[key+'_file']);new=directory/(tag+'.'+key)
            shutil.copyfile(old,new)
            if sha(new)!=row[key+'_sha256'] or new.read_text()!=row[key]:
                raise ValueError('PREFIX20_RECOVERY_STORAGE_RAW_BYTES_CHANGED')
            value[key+'_file']=str(new)
        return value
    root_original=Path(original['original_root_receipt_file'])
    root=raw(json.loads(root_original.read_text()),'root')
    root['relocated_observation_provenance']=dict(original_path=str(root_original),original_sha256=sha(root_original))
    root_path=directory/'root-receipt.json';publish(root_path,root)
    proof=copy.deepcopy(original)
    for key in ('stdout_file','stderr_file'):proof[key]=root[key]
    proof['original_root_receipt_file']=str(root_path);proof['original_root_receipt_sha256']=sha(root_path)
    proof['permission_subtree_complements']=[raw(row,'part-'+str(i)) for i,row in enumerate(original['permission_subtree_complements'])]
    if additional_prelaunch_reserve_bytes not in (2*2**30,4*2**30):
        raise ValueError('PREFIX20_RECOVERY_CUMULATIVE_RESERVE_INVALID')
    proof['additional_prelaunch_reserve_bytes']=additional_prelaunch_reserve_bytes
    proof['relocation_provenance']=dict(original_aggregate_path=str(source),original_aggregate_sha256=sha(source),
        original_observation_timestamps_preserved=True,new_root_du_performed=False,copied_unix=time.time(),
        semantics='same still-fresh global account measurement; absolute evidence paths relocated; explicitly bounded cumulative recovery allowance')
    validate_upper_bound(proof,target)
    publish(target/'preparation/personal-storage-observation.json',proof)
    return dict(original_checked_unix=proof['checked_unix'],allocated_bytes_upper_bound=proof['allocated_bytes_upper_bound'],
        additional_prelaunch_reserve_bytes=proof['additional_prelaunch_reserve_bytes'],new_root_du_performed=False)


def prepare(commit):
    os.umask(0o007);own_base(TARGET)
    if TARGET.exists():raise ValueError('PREFIX20_RECOVERY_TARGET_ALREADY_ATTEMPTED_INSPECT')
    actual=pristine_failure();source,manifest=verify_source(FAILED,commit)
    validate_upper_bound(json.loads((FAILED/'preparation/personal-storage-observation.json').read_text()),FAILED)
    claim=FAILED/'preparation/recovery726927-target-claim.json'
    if claim.exists():raise ValueError('PREFIX20_RECOVERY_ALREADY_ATTEMPTED_INSPECT')
    publish(claim,dict(target=str(TARGET),source_commit=commit,original=actual,created_unix=time.time()))
    TARGET.mkdir();(TARGET/'preparation').mkdir()
    source_target=TARGET/'source'/commit;shutil.copytree(source,source_target)
    verify_source(TARGET,commit)
    prepared=bootstrap(TARGET,commit)
    root=TARGET/'assets/prefix20';root.mkdir()
    copied=[]
    for name,(size,digest) in ASSETS.items():
        original=FAILED/'assets/prefix20'/name;target=root/name
        if original.is_symlink() or original.stat().st_size!=size or sha(original)!=digest:
            raise ValueError('PREFIX20_RECOVERY_PRETRAINED_INPUT_CHANGED')
        shutil.copyfile(original,target)
        with target.open('rb') as f:os.fsync(f.fileno())
        if target.stat().st_size!=size or sha(target)!=digest:raise ValueError('PREFIX20_RECOVERY_INPUT_COPY_SHA_DIFFERS')
        copied.append(dict(path=str(target),bytes=size,sha256=digest))
    storage_origin=relocate_global_storage(FAILED,TARGET)
    admission=storage_admission(TARGET);assets=assets_ready(TARGET)
    value=dict(status='PREFIX20_PRISTINE_STARTUP_FAILURE_FRESH_TARGET_PREPARED',target=str(TARGET),
        source_commit=commit,recovery_of=JOB,original=actual,assets=assets,storage=admission,
        copied_global_observation=storage_origin,source_all_files_verified=len(manifest['files']),
        old_source_and_logs_untouched=True,training_submitted=False,actual_gpu_acceptance_pending=True,checked_unix=time.time())
    publish(TARGET/'preparation/recovery-from726927.json',value)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.commit)))


if __name__=='__main__':main()
