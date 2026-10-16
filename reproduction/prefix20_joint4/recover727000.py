"""Fresh start after the observed shared-filesystem Triton preflight failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4.control import ASSETS,own_base,publish,verify_source,bootstrap,assets_ready,storage_admission,validate_slurm_job
from reproduction.prefix20_joint4.recover726927 import relocate_global_storage

FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016')
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016')
JOB='727000'
RUN='8a53f94d112147229e7f94e4629f5316'
OLD_COMMIT='7a154d92fe2368d024ee09bb77ce8b48f787f1b1'
EVIDENCE={
    'gpu-preflight.log':'398747892519d5cf6848ae2e360b6e657ca7ec0f421ba750081ca1c3de492183',
    'pipeline-status.json':'37c8efd1b87393d924e426a40c13da65be5e344b76b3f451606e5c568f86f329',
    'cleanup-status.json':'5e10c84f4abe6e73d9c6ea70a9629886a016873077f1471f4c5e2f5c022dcccd',
    'process-ledger.json':'b5442b8631bcdcc38876b5825be15df6bb4f424fcce6faf852d610f265ab2abd',
    'native-allocation-admission.json':'e40906655e3f374943890eec6bb9bd69de4dc36f6757d17cbe1d8c50d24c8667',
}


def validate_ledger(value):
    if value.get('run_id')!=RUN or set(value.get('jobs',{}))!={'gpu-preflight'}:
        raise ValueError('PREFIX20_RECOVERY727000_PROJECT_TRAINING_PROCESS_EXISTS')
    row=value['jobs']['gpu-preflight']
    if row.get('state')!='STOPPED' or row.get('exit_code')!=1 or row['leader']['uid']!=14090:
        raise ValueError('PREFIX20_RECOVERY727000_PREFLIGHT_NOT_STOPPED1')


def pristine_failure():
    if os.getuid()!=14090 or (FAILED/'job.id').read_text().strip()!=JOB:
        raise ValueError('PREFIX20_RECOVERY727000_OWNER_JOB_DIFFERS')
    submitted=json.loads((FAILED/'preparation/training-submission.json').read_text())
    prepared=json.loads((FAILED/'logs/prefix20-configuration-prepared.json').read_text())
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    validate_slurm_job(FAILED,OLD_COMMIT,submitted,prepared['actual_slurm_job_binding'])
    if (submitted.get('returncode')!=0 or submitted.get('job')!=JOB or run['run_id']!=RUN or
            run['source_commit']!=OLD_COMMIT or run['completed_updates']!=0 or prepared['completed_outers']!=0):
        raise ValueError('PREFIX20_RECOVERY727000_ORIGINAL_RUN_DIFFERS')
    result=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Account,QOS,JobName,Start,End,NodeList'],capture_output=True,text=True,timeout=30)
    line='727000|FAILED|1:0|jxu957|14090|jxu957|formal-user|qprobe-prefix20-joint200|2026-10-16T21:22:40|2026-10-16T21:24:28|ACD1-26'
    if result.returncode or line not in result.stdout.splitlines() or not any(
            row.startswith(JOB+'.extern|COMPLETED|0:0|') for row in result.stdout.splitlines()):
        raise ValueError('PREFIX20_RECOVERY727000_ACTUAL_TERMINAL_DIFFERS')
    queue=subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i'],capture_output=True,text=True,timeout=20)
    if queue.returncode or JOB in queue.stdout.split():raise ValueError('PREFIX20_RECOVERY727000_STILL_QUEUED')
    for name,digest in EVIDENCE.items():
        if sha(FAILED/'logs'/name)!=digest:raise ValueError('PREFIX20_RECOVERY727000_FAILURE_EVIDENCE_CHANGED')
    validate_ledger(json.loads((FAILED/'logs/process-ledger.json').read_text()))
    allocation=json.loads((FAILED/'logs/native-allocation-admission.json').read_text())
    if not allocation['accepted'] or allocation['run_id']!=RUN or allocation['slurm_job_id']!=JOB or allocation['training_world_size']!=4:
        raise ValueError('PREFIX20_RECOVERY727000_NATIVE4_PROOF_DIFFERS')
    log=(FAILED/'logs/gpu-preflight.log').read_text()
    if 'triton_.cubin' not in log or '__triton_launcher.so' not in log or 'No such file or directory' not in log:
        raise ValueError('PREFIX20_RECOVERY727000_NOT_OBSERVED_JIT_CACHE_FAILURE')
    forbidden=['logs/training-launch-claim.json','logs/search-g1-common8-training.log','artifacts/question-probe','checkpoints']
    if any((FAILED/name).exists() for name in forbidden) or any((FAILED/'artifacts').iterdir()):
        raise ValueError('PREFIX20_RECOVERY727000_ACTOR_PROBE_ROLLOUT_WORK_EXISTS')
    with (FAILED/'gpu-job.lock').open('r+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_source(FAILED,OLD_COMMIT)
    bound=json.loads((FAILED/'preparation/run-source.json').read_text())
    if bound['source_commit']!=OLD_COMMIT:raise ValueError('PREFIX20_RECOVERY727000_OLD_SOURCE_BINDING_DIFFERS')
    for name,row in bound['files'].items():
        if sha(FAILED/'repo'/name)!=row['sha256']:raise ValueError('PREFIX20_RECOVERY727000_RUNTIME_SOURCE_CHANGED')
    return dict(job=JOB,run_id=RUN,source_commit=OLD_COMMIT,completed_outers=0,
        actor_probe_pipeline_never_started=True,native4_allocation_previously_accepted=True,
        terminal_accounting=result.stdout,queue=queue.stdout,failure_file_sha256=EVIDENCE,
        gpu_job_lock_free=True,checked_unix=time.time())


def prepare(commit):
    os.umask(0o007);own_base(TARGET)
    claim=FAILED/'preparation/recovery727000-target-claim.json'
    if TARGET.exists() or claim.exists():raise ValueError('PREFIX20_RECOVERY727000_ALREADY_ATTEMPTED_INSPECT')
    actual=pristine_failure();source,manifest=verify_source(FAILED,commit)
    storage_admission(FAILED)
    publish(claim,dict(target=str(TARGET),source_commit=commit,original=actual,created_unix=time.time()))
    TARGET.mkdir();(TARGET/'preparation').mkdir()
    shutil.copytree(source,TARGET/'source'/commit);verify_source(TARGET,commit);bootstrap(TARGET,commit)
    root=TARGET/'assets/prefix20';root.mkdir()
    for name,(size,digest) in ASSETS.items():
        old=FAILED/'assets/prefix20'/name;new=root/name
        if old.is_symlink() or old.stat().st_size!=size or sha(old)!=digest:raise ValueError('PREFIX20_RECOVERY727000_INPUT_CHANGED')
        shutil.copyfile(old,new)
        with new.open('rb') as stream:os.fsync(stream.fileno())
        if new.stat().st_size!=size or sha(new)!=digest:raise ValueError('PREFIX20_RECOVERY727000_COPIED_INPUT_DIFFERS')
    observation=relocate_global_storage(FAILED,TARGET)
    value=dict(status='PREFIX20_JIT_PREFLIGHT_FAILURE_FRESH_TARGET_PREPARED',target=str(TARGET),
        source_commit=commit,recovery_of=JOB,original=actual,assets=assets_ready(TARGET),
        storage=storage_admission(TARGET),copied_global_observation=observation,
        source_all_files_verified=len(manifest['files']),old_source_and_logs_untouched=True,
        training_submitted=False,actual_gpu_acceptance_pending=True,checked_unix=time.time())
    publish(TARGET/'preparation/recovery-from727000.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.commit)))


if __name__=='__main__':main()
