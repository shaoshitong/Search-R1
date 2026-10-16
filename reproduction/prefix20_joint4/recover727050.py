"""Fresh start after the observed omitted BF16 sampling-journal model-smoke contract."""
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

FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016')
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016')
JOB='727050'
RUN='ba89b8b5b8c847ea8d9c29e72f37957f'
OLD_COMMIT='9b34d3fe15795c4c143b19c08c10aef20a2400ab'
EVIDENCE={'model-initial.log': '3581c3aa13ca5e480c2bda167ebab1c14b2fc0b222439d44a487de8ea8578c3e', 'gpu-preflight.log': '6e8fa009764785e0c681d9c8ddf53e995efc802470d29f7d81588e33941ee2c1', 'pipeline-status.json': 'fc2152ed2674d781e7c7a30ba1118dcac04eb2eb2f51652876dfd5be63e28f31', 'cleanup-status.json': 'fd5b522a8826ea698cd2253e4a085234bb99d0cd2dec4ce7ea6e197377f481fb', 'process-ledger.json': 'ef012f6fa6f13ed1aef62d0c4574a827e84f86acb26f31ca7ab2f97d2c839810', 'native-allocation-admission.json': '6b0422cb8d553cc6ab8ba59a80117529abcfc3624877e6a0f9d0d3f376ae87b5'}


def validate_ledger(value):
    if value.get('run_id')!=RUN or set(value.get('jobs',{}))!={'gpu-preflight','model-initial'}:
        raise ValueError('PREFIX20_RECOVERY727050_PROJECT_TRAINING_PROCESS_EXISTS')
    for name,code in [('gpu-preflight',0),('model-initial',1)]:
        row=value['jobs'][name]
        if row.get('state')!='STOPPED' or row.get('exit_code')!=code or row['leader']['uid']!=14090:
            raise ValueError('PREFIX20_RECOVERY727050_SMOKE_NOT_STOPPED')



def pristine_failure():
    if os.getuid()!=14090 or (FAILED/'job.id').read_text().strip()!=JOB:
        raise ValueError('PREFIX20_RECOVERY727050_OWNER_JOB_DIFFERS')
    submitted=json.loads((FAILED/'preparation/training-submission.json').read_text())
    prepared=json.loads((FAILED/'logs/prefix20-configuration-prepared.json').read_text())
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    validate_slurm_job(FAILED,OLD_COMMIT,submitted,prepared['actual_slurm_job_binding'])
    if (submitted.get('returncode')!=0 or submitted.get('job')!=JOB or run['run_id']!=RUN or
            run['source_commit']!=OLD_COMMIT or run['completed_updates']!=0 or prepared['completed_outers']!=0):
        raise ValueError('PREFIX20_RECOVERY727050_ORIGINAL_RUN_DIFFERS')
    result=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Account,QOS,JobName,Start,End,NodeList'],capture_output=True,text=True,timeout=30)
    line='727050|FAILED|1:0|jxu957|14090|jxu957|formal-user|qprobe-prefix20-joint200|2026-10-16T21:42:25|2026-10-16T21:46:08|ACD1-32'
    if result.returncode or line not in result.stdout.splitlines() or not any(
            row.startswith(JOB+'.extern|COMPLETED|0:0|') for row in result.stdout.splitlines()):
        raise ValueError('PREFIX20_RECOVERY727050_ACTUAL_TERMINAL_DIFFERS')
    queue=subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i'],capture_output=True,text=True,timeout=20)
    if queue.returncode or JOB in queue.stdout.split():raise ValueError('PREFIX20_RECOVERY727050_STILL_QUEUED')
    for name,digest in EVIDENCE.items():
        if sha(FAILED/'logs'/name)!=digest:raise ValueError('PREFIX20_RECOVERY727050_FAILURE_EVIDENCE_CHANGED')
    validate_ledger(json.loads((FAILED/'logs/process-ledger.json').read_text()))
    allocation=json.loads((FAILED/'logs/native-allocation-admission.json').read_text())
    if not allocation['accepted'] or allocation['run_id']!=RUN or allocation['slurm_job_id']!=JOB or allocation['training_world_size']!=4:
        raise ValueError('PREFIX20_RECOVERY727050_NATIVE4_PROOF_DIFFERS')
    log=(FAILED/'logs/model-initial.log').read_text()
    if ('COMMON8_MODEL_PARITY_REJECTED=' not in log or '0.7373237609863281' not in log or
            '1.2512952089309692' not in log):
        raise ValueError('PREFIX20_RECOVERY727050_NOT_OBSERVED_CROSS_BACKEND_FAILURE')
    forbidden=['logs/training-launch-claim.json','logs/search-g1-common8-training.log','artifacts/question-probe','checkpoints']
    if any((FAILED/name).exists() for name in forbidden) or any((FAILED/'artifacts').iterdir()):
        raise ValueError('PREFIX20_RECOVERY727050_ACTOR_PROBE_ROLLOUT_WORK_EXISTS')
    with (FAILED/'gpu-job.lock').open('r+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_source(FAILED,OLD_COMMIT)
    bound=json.loads((FAILED/'preparation/run-source.json').read_text())
    if bound['source_commit']!=OLD_COMMIT:raise ValueError('PREFIX20_RECOVERY727050_OLD_SOURCE_BINDING_DIFFERS')
    for name,row in bound['files'].items():
        if sha(FAILED/'repo'/name)!=row['sha256']:raise ValueError('PREFIX20_RECOVERY727050_RUNTIME_SOURCE_CHANGED')
    return dict(job=JOB,run_id=RUN,source_commit=OLD_COMMIT,completed_outers=0,
        actor_probe_pipeline_never_started=True,native4_allocation_previously_accepted=True,
        terminal_accounting=result.stdout,queue=queue.stdout,failure_file_sha256=EVIDENCE,
        gpu_job_lock_free=True,checked_unix=time.time())


def prelaunch_footprint(source):
    # Conservatively count all own fresh/failed trees, including bytes already
    # covered by the original global measurement. Shared env/model links are
    # not followed. The extra 2GiB is cumulative, never renewed per recovery.
    bases=[Path('/data/user/jxu957/experiments')/name for name in (
        'qprobe-teacher-warmup200-prefix20-joint4-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016')]
    r=subprocess.run(['du','-s','-B1',*[str(p) for p in bases]],capture_output=True,text=True,timeout=60)
    if r.returncode or r.stderr:raise ValueError('PREFIX20_RECOVERY727050_BOUNDED_DU_FAILED')
    rows=[line.split('\t') for line in r.stdout.splitlines()]
    if len(rows)!=3 or [row[1] for row in rows]!=[str(p) for p in bases] or not all(row[0].isdecimal() for row in rows):
        raise ValueError('PREFIX20_RECOVERY727050_BOUNDED_DU_PATHS_DIFFER')
    # Two full source copies and the three real input copies, plus metadata.
    manifest=json.loads((source/'source-publication.json').read_text())
    next_alloc=sum(ASSETS[name][0] for name in ASSETS)+2*sum(row['bytes'] for row in manifest['files'].values())+64*2**20
    projected=sum(int(row[0]) for row in rows)+next_alloc
    if projected>2*2**30:raise ValueError('PREFIX20_RECOVERY727050_CUMULATIVE_EXTRA_RESERVE_EXCEEDED')
    return dict(checked_unix=time.time(),du_stdout=r.stdout,existing_bases_allocated_bytes=sum(int(row[0]) for row in rows),
        next_target_conservative_allowance_bytes=next_alloc,projected_cumulative_prelaunch_bytes=projected,
        cumulative_extra_allowance_bytes=2*2**30,full_root_measurement_repeated=False)


def prepare(commit):
    os.umask(0o007);own_base(TARGET)
    claim=FAILED/'preparation/recovery727050-target-claim.json'
    if TARGET.exists() or claim.exists():raise ValueError('PREFIX20_RECOVERY727050_ALREADY_ATTEMPTED_INSPECT')
    actual=pristine_failure();source,manifest=verify_source(FAILED,commit)
    storage_admission(FAILED)
    footprint=prelaunch_footprint(source)
    publish(claim,dict(target=str(TARGET),source_commit=commit,original=actual,created_unix=time.time()))
    TARGET.mkdir();(TARGET/'preparation').mkdir()
    shutil.copytree(source,TARGET/'source'/commit);verify_source(TARGET,commit);bootstrap(TARGET,commit)
    root=TARGET/'assets/prefix20';root.mkdir()
    for name,(size,digest) in ASSETS.items():
        old=FAILED/'assets/prefix20'/name;new=root/name
        if old.is_symlink() or old.stat().st_size!=size or sha(old)!=digest:raise ValueError('PREFIX20_RECOVERY727050_INPUT_CHANGED')
        shutil.copyfile(old,new)
        with new.open('rb') as stream:os.fsync(stream.fileno())
        if new.stat().st_size!=size or sha(new)!=digest:raise ValueError('PREFIX20_RECOVERY727050_COPIED_INPUT_DIFFERS')
    observation=relocate_global_storage(FAILED,TARGET)
    value=dict(status='PREFIX20_BOUND_BF16_CHECKER_FAILURE_FRESH_TARGET_PREPARED',target=str(TARGET),
        source_commit=commit,recovery_of=JOB,original=actual,assets=assets_ready(TARGET),
        cumulative_prelaunch_footprint=footprint,
        storage=storage_admission(TARGET),copied_global_observation=observation,
        source_all_files_verified=len(manifest['files']),old_source_and_logs_untouched=True,
        training_submitted=False,actual_gpu_acceptance_pending=True,checked_unix=time.time())
    publish(TARGET/'preparation/recovery-from727050.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.commit)))


if __name__=='__main__':main()
