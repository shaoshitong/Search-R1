"""Fresh start after the observed missing original G1 CPU dependencies and legacy-kernel cleanup."""
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

FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016')
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016')
JOB='727148'
RUN='ebcff3f7fff6499e9f121153dc29ebab'
OLD_COMMIT='f6e2f268128a888a6abbc7489a02871fe28c0178'
EVIDENCE={'initial-calibration.log': '0dfa4894a4f1ed0ef50206f9d67a437f971ff07a12605f37a4698bd329db6dbb', 'pipeline-status.json': '9f019a717613de20fddb38a993846f51bf3c60757b03a74e608945a1da53de39', 'process-ledger.json': 'e9ac9218e43227759f56a7f75f3e67a3f20bcc3aed6c7040850afd74d2d01980', 'native-allocation-admission.json': 'c7689ce66d756f59320adeab3e3e2ed714bc61d326597616a3f673edc382fc06'}


def validate_ledger(value):
    expected={'gpu-preflight':('STOPPED',0),'model-initial':('STOPPED',0),
        'retrieval-stage':('STOPPED',0),'retriever':('CLEANUP_FAILED',None),
        'initial-calibration':('STOPPED',1)}
    if value.get('run_id')!=RUN or set(value.get('jobs',{}))!=set(expected):
        raise ValueError('PREFIX20_RECOVERY727148_PROJECT_TRAINING_PROCESS_EXISTS')
    for name,(state,code) in expected.items():
        row=value['jobs'][name]
        if row.get('state')!=state or row.get('exit_code')!=code or row['leader']['uid']!=14090:
            raise ValueError('PREFIX20_RECOVERY727148_OBSERVED_ROLES_DIFFER')
    if 'pidfd_open' not in value['jobs']['retriever'].get('cleanup_error',''):
        raise ValueError('PREFIX20_RECOVERY727148_CLEANUP_FAILURE_DIFFERS')



def pristine_failure():
    if os.getuid()!=14090 or (FAILED/'job.id').read_text().strip()!=JOB:
        raise ValueError('PREFIX20_RECOVERY727148_OWNER_JOB_DIFFERS')
    submitted=json.loads((FAILED/'preparation/training-submission.json').read_text())
    prepared=json.loads((FAILED/'logs/prefix20-configuration-prepared.json').read_text())
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    validate_slurm_job(FAILED,OLD_COMMIT,submitted,prepared['actual_slurm_job_binding'])
    if (submitted.get('returncode')!=0 or submitted.get('job')!=JOB or run['run_id']!=RUN or
            run['source_commit']!=OLD_COMMIT or run['completed_updates']!=0 or prepared['completed_outers']!=0):
        raise ValueError('PREFIX20_RECOVERY727148_ORIGINAL_RUN_DIFFERS')
    result=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Account,QOS,JobName,Start,End,NodeList'],capture_output=True,text=True,timeout=30)
    line='727148|FAILED|1:0|jxu957|14090|jxu957|formal-user|qprobe-prefix20-joint200|2026-10-16T22:16:44|2026-10-16T22:26:18|ACD1-25'
    if result.returncode or line not in result.stdout.splitlines() or not any(
            row.startswith(JOB+'.extern|COMPLETED|0:0|') for row in result.stdout.splitlines()):
        raise ValueError('PREFIX20_RECOVERY727148_ACTUAL_TERMINAL_DIFFERS')
    queue=subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i'],capture_output=True,text=True,timeout=20)
    if queue.returncode or JOB in queue.stdout.split():raise ValueError('PREFIX20_RECOVERY727148_STILL_QUEUED')
    for name,digest in EVIDENCE.items():
        if sha(FAILED/'logs'/name)!=digest:raise ValueError('PREFIX20_RECOVERY727148_FAILURE_EVIDENCE_CHANGED')
    validate_ledger(json.loads((FAILED/'logs/process-ledger.json').read_text()))
    allocation=json.loads((FAILED/'logs/native-allocation-admission.json').read_text())
    if not allocation['accepted'] or allocation['run_id']!=RUN or allocation['slurm_job_id']!=JOB or allocation['training_world_size']!=4:
        raise ValueError('PREFIX20_RECOVERY727148_NATIVE4_PROOF_DIFFERS')
    log=(FAILED/'logs/initial-calibration.log').read_text()
    if "ModuleNotFoundError: No module named 'joblib'" not in log:
        raise ValueError('PREFIX20_RECOVERY727148_NOT_OBSERVED_CPU_PROBE_FAILURE')
    smoke=json.loads((FAILED/'logs/model-smoke-initial/complete.json').read_text())
    if smoke.get('accepted') is not True or smoke['run_id']!=RUN or len(smoke['ranks'])!=4 or smoke['actual_grpo_updates']!=0:
        raise ValueError('PREFIX20_RECOVERY727148_INITIAL_MODEL_PROOF_DIFFERS')
    forbidden=['logs/training-launch-claim.json','logs/search-g1-common8-training.log','artifacts/question-probe','checkpoints']
    if any((FAILED/name).exists() for name in forbidden) or any((FAILED/'artifacts').iterdir()):
        raise ValueError('PREFIX20_RECOVERY727148_ACTOR_PROBE_ROLLOUT_WORK_EXISTS')
    with (FAILED/'gpu-job.lock').open('r+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_source(FAILED,OLD_COMMIT)
    bound=json.loads((FAILED/'preparation/run-source.json').read_text())
    if bound['source_commit']!=OLD_COMMIT:raise ValueError('PREFIX20_RECOVERY727148_OLD_SOURCE_BINDING_DIFFERS')
    for name,row in bound['files'].items():
        if sha(FAILED/'repo'/name)!=row['sha256']:raise ValueError('PREFIX20_RECOVERY727148_RUNTIME_SOURCE_CHANGED')
    return dict(job=JOB,run_id=RUN,source_commit=OLD_COMMIT,completed_outers=0,
        actor_probe_pipeline_never_started=True,native4_allocation_previously_accepted=True,
        terminal_accounting=result.stdout,queue=queue.stdout,failure_file_sha256=EVIDENCE,
        gpu_job_lock_free=True,checked_unix=time.time())


def prelaunch_footprint(source):
    # Conservatively count all own fresh/failed trees, including bytes already
    # covered by the original global measurement. Shared env/model links are
    # not followed. The extra 4GiB is cumulative, never renewed per recovery.
    bases=[Path('/data/user/jxu957/experiments')/name for name in (
        'qprobe-teacher-warmup200-prefix20-joint4-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016')]
    r=subprocess.run(['du','-s','-B1',*[str(p) for p in bases]],capture_output=True,text=True,timeout=60)
    if r.returncode or r.stderr:raise ValueError('PREFIX20_RECOVERY727148_BOUNDED_DU_FAILED')
    rows=[line.split('\t') for line in r.stdout.splitlines()]
    if len(rows)!=4 or [row[1] for row in rows]!=[str(p) for p in bases] or not all(row[0].isdecimal() for row in rows):
        raise ValueError('PREFIX20_RECOVERY727148_BOUNDED_DU_PATHS_DIFFER')
    # Two full source copies and the three real input copies, plus metadata.
    manifest=json.loads((source/'source-publication.json').read_text())
    next_alloc=sum(ASSETS[name][0] for name in ASSETS)+2*sum(row['bytes'] for row in manifest['files'].values())+64*2**20
    projected=sum(int(row[0]) for row in rows)+next_alloc
    if projected>4*2**30:raise ValueError('PREFIX20_RECOVERY727148_CUMULATIVE_EXTRA_RESERVE_EXCEEDED')
    return dict(checked_unix=time.time(),du_stdout=r.stdout,existing_bases_allocated_bytes=sum(int(row[0]) for row in rows),
        next_target_conservative_allowance_bytes=next_alloc,projected_cumulative_prelaunch_bytes=projected,
        cumulative_extra_allowance_bytes=4*2**30,full_root_measurement_repeated=False)


def prepare(commit):
    os.umask(0o007);own_base(TARGET)
    claim=FAILED/'preparation/recovery727148-target-claim.json'
    if TARGET.exists() or claim.exists():raise ValueError('PREFIX20_RECOVERY727148_ALREADY_ATTEMPTED_INSPECT')
    actual=pristine_failure();source,manifest=verify_source(FAILED,commit)
    storage_admission(FAILED)
    footprint=prelaunch_footprint(source)
    publish(claim,dict(target=str(TARGET),source_commit=commit,original=actual,created_unix=time.time()))
    TARGET.mkdir();(TARGET/'preparation').mkdir()
    shutil.copytree(source,TARGET/'source'/commit);verify_source(TARGET,commit);bootstrap(TARGET,commit)
    root=TARGET/'assets/prefix20';root.mkdir()
    for name,(size,digest) in ASSETS.items():
        old=FAILED/'assets/prefix20'/name;new=root/name
        if old.is_symlink() or old.stat().st_size!=size or sha(old)!=digest:raise ValueError('PREFIX20_RECOVERY727148_INPUT_CHANGED')
        shutil.copyfile(old,new)
        with new.open('rb') as stream:os.fsync(stream.fileno())
        if new.stat().st_size!=size or sha(new)!=digest:raise ValueError('PREFIX20_RECOVERY727148_COPIED_INPUT_DIFFERS')
    observation=relocate_global_storage(FAILED,TARGET,additional_prelaunch_reserve_bytes=4*2**30)
    cleanup_files={name:sha(FAILED/'logs'/name) for name in
        ('run-manifest.json','process-ledger.json','retrieval-staging-claim.json','corpus-cache-claim.json')}
    if cleanup_files['process-ledger.json']!=EVIDENCE['process-ledger.json']:
        raise ValueError('PREFIX20_RECOVERY727148_CLEANUP_LEDGER_CHANGED')
    publish(TARGET/'preparation/failed727148-cleanup-evidence-sha.json',dict(failed_base=str(FAILED),
        failed_run=RUN,source_commit=commit,files=cleanup_files))
    publish(TARGET/'preparation/failed727148-deferred-node-cleanup.json',dict(failed_base=str(FAILED),failed_job=JOB,failed_run=RUN,node='ACD1-25',recovery_base=str(TARGET),source_commit=commit,before_any_cuda=True))
    value=dict(status='PREFIX20_CPU_PROBE_RUNTIME_FAILURE_FRESH_TARGET_PREPARED',target=str(TARGET),
        source_commit=commit,recovery_of=JOB,original=actual,assets=assets_ready(TARGET),
        cumulative_prelaunch_footprint=footprint,
        storage=storage_admission(TARGET),copied_global_observation=observation,
        source_all_files_verified=len(manifest['files']),old_source_and_logs_untouched=True,
        training_submitted=False,actual_gpu_acceptance_pending=True,old_node_cleanup_pending_before_any_cuda=True,checked_unix=time.time())
    publish(TARGET/'preparation/recovery-from727148.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.commit)))


if __name__=='__main__':main()
