"""Once-only four-GPU launch; epoch68 supplies weights, never teacher work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from reproduction.hpc3_qprobe.control import prepare as prepare_environment
from reproduction.prefix20_joint4.assets import sha

BASE='/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-20261016'
MODEL='/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725'
PERSONAL_ROOT='/data/user/jxu957/experiments'
ASSETS={
    'head.safetensors':(308473196,'535c7bfee07374dde88b7d7f9e4f08ac874d46088ef961f15ea6ab7dd849a560'),
    'probe-config.json':(1940,'be7456474ad62f3c874a4fb176d9858fa8cef385077e96247a3e8c795e3347d7'),
    'eligible-question-pool.json':(41242673,'ae13710254e7ae77cf5531b62a755ec2000bc3fba20051bbafaff1f6714aa8fc'),
}


def publish(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial-'+str(time.time_ns()))
    try:
        with temporary.open('x') as out:json.dump(value,out,indent=2);out.flush();os.fsync(out.fileno())
        os.link(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary.exists():temporary.unlink()


def own_base(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or not re.fullmatch(
            r'/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-[A-Za-z0-9._-]+',str(path)):
        raise ValueError('PREFIX20_DEDICATED_REMOTE_BASE_REQUIRED')
    return path


def verify_source(base,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('PREFIX20_FULL_SOURCE_COMMIT_REQUIRED')
    source=base/'source'/commit
    proof=json.loads((source/'source-publication.json').read_text())
    if proof['source_commit']!=commit or not proof['fresh_source_all_files_sha_verified']:
        raise ValueError('PREFIX20_SOURCE_NOT_PUBLISHED_AND_DEPLOYED')
    for name,record in proof['files'].items():
        path=source/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('PREFIX20_UNSAFE_SOURCE_MEMBER')
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('PREFIX20_DEPLOYED_SOURCE_SHA_DIFFERS')
    return source,proof


def bind_repo(base,commit):
    """Loaded native modules must really live at base/repo, not a symlink."""
    source,proof=verify_source(base,commit);repo=base/'repo'
    receipt=base/'preparation/run-source.json'
    if repo.is_symlink():raise ValueError('PREFIX20_RUNTIME_REPO_CANNOT_BE_SYMLINK')
    if repo.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if old['source_commit']!=commit:
            forbidden=[base/'job.id',base/'preparation/training-submit-claim.json',
                base/'preparation/training-submission.json',base/'logs/run-manifest.json',
                base/'logs/training-launch-claim.json',base/'logs/process-ledger.json',base/'gpu-job.lock']
            if any(p.exists() for p in forbidden):raise ValueError('PREFIX20_CANNOT_PROMOTE_CLAIMED_RUNTIME_SOURCE')
            for name,record in old['files'].items():
                path=repo/name
                if path.is_symlink() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                    raise ValueError('PREFIX20_UNSTARTED_OLD_RUNTIME_SOURCE_CHANGED')
            retired=base/('repo.prelaunch-retired-'+old['source_commit'])
            if retired.exists():raise ValueError('PREFIX20_OLD_SOURCE_RETIREMENT_ALREADY_EXISTS_INSPECT')
            repo.rename(retired)
            receipt.rename(receipt.with_name('run-source-retired-'+old['source_commit']+'.json'))
            publish(base/'preparation'/('source-promotion-'+commit+'.json'),dict(
                old_source_commit=old['source_commit'],new_source_commit=commit,retired_repo=str(retired),
                no_job_or_training_claim=True,old_all_files_sha_verified=True,checked_unix=time.time()))
    if not repo.exists():
        stage=base/('repo.partial-'+commit);stage.mkdir(exist_ok=False)
        for name,record in proof['files'].items():
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
            raw=(source/name).read_bytes()
            with target.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
            target.chmod((source/name).stat().st_mode & 0o777)
            if sha(target)!=record['sha256']:raise ValueError('PREFIX20_RUNTIME_COPY_SHA_DIFFERS')
        stage.rename(repo)
        publish(receipt,dict(source_commit=commit,source_root=str(source),runtime_repo=str(repo),
            files=proof['files'],fresh_runtime_copy_sha_verified=True,checked_unix=time.time()))
    bound=json.loads(receipt.read_text())
    if bound['source_commit']!=commit or bound['files']!=proof['files']:
        raise ValueError('PREFIX20_EXISTING_RUNTIME_COPY_BOUND_TO_OTHER_SOURCE')
    for name,record in proof['files'].items():
        path=repo/name
        if path.is_symlink() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('PREFIX20_BOUND_RUNTIME_SOURCE_CHANGED')
    return bound


def assets_ready(base):
    """Only three selected new inputs; the original dataset/model stay read-only."""
    root=base/'assets/prefix20';files=[]
    for name,(size,digest) in ASSETS.items():
        path=root/name
        if not path.is_file() or any(p.is_symlink() for p in [path,*path.parents]):
            raise ValueError('PREFIX20_SELECTED_ASSET_MISSING_OR_SYMLINK')
        if path.stat().st_size!=size or sha(path)!=digest:raise ValueError('PREFIX20_SELECTED_ASSET_SHA_DIFFERS')
        files.append(dict(path=str(path),bytes=size,sha256=digest))
    from search_g1.prefix20_data import TRAIN_SHA
    if sha(base/'data/train.parquet')!=TRAIN_SHA:raise ValueError('PREFIX20_CANONICAL_TRAIN_CHANGED')
    return dict(files=files,canonical_train_sha256=TRAIN_SHA,old_optimizer_loaded=False,
        teacher_warmup_required=False,checked_unix=time.time())


def storage_plan(observation,checkpoint,budget,*,now=None):
    now=time.time() if now is None else now
    if (observation.get('role')!='prefix20_prelaunch_personal_storage_observation' or
            observation.get('account_experiment_root')!=PERSONAL_ROOT or
            now-observation['checked_unix']>21600 or observation['checked_unix']>now+60):
        raise ValueError('PREFIX20_FRESH_PERSONAL_STORAGE_OBSERVATION_REQUIRED')
    raw=observation['stdout'].strip().split('\t')
    if len(raw)!=2 or raw[1]!=PERSONAL_ROOT or not raw[0].isdecimal():
        raise ValueError('PREFIX20_PERSONAL_STORAGE_MEASUREMENT_INVALID')
    measured=int(raw[0]);complete=bool(observation.get('measurement_complete') and
        observation['returncode']==0 and not observation['stderr'].strip())
    if observation.get('measurement_kind')=='permission_partition_allocated_byte_upper_bound':
        measured=observation['allocated_bytes_upper_bound']
        complete=observation['measurement_complete'] and observation['returncode']==1
    # Full actor export plus FSDP Adam/RNG at all five checkpoints, with a
    # separate conservative allowance for four-rank probe and original traces.
    actor_complete=3*checkpoint['per_save_allowance_bytes']
    probe_complete=4*(3*ASSETS['head.safetensors'][0]+64*2**20)
    per_pair=actor_complete+probe_complete
    source_copies=len(checkpoint['save_steps']);restore_copies=1
    encoder_dependency=checkpoint['per_save_allowance_bytes']
    recovery_extra=observation.get('additional_prelaunch_reserve_bytes',0)
    if type(recovery_extra) is not int or recovery_extra not in (0,2*2**30,4*2**30):
        raise ValueError('PREFIX20_RECOVERY_PRELAUNCH_RESERVE_DIFFERS')
    reserve=(source_copies+restore_copies)*per_pair+encoder_dependency+96*2**30+recovery_extra
    if type(budget) is not int or budget not in (2*10**12,25*10**11):
        raise ValueError('PREFIX20_EXPLICIT_PERSONAL_BUDGET_REQUIRED')
    return dict(account_experiment_root=PERSONAL_ROOT,measured_allocated_bytes=measured,
        measurement_complete=complete,measurement_is_lower_bound=not complete,
        remaining_run_allowance_bytes=reserve,personal_budget_bytes=budget,
        per_complete_actor_probe_checkpoint_allowance_bytes=per_pair,
        hpc3_source_checkpoint_copies=source_copies,hpc3_simultaneous_restore_target_copies=restore_copies,
        hpc3_archive_copies=0,hkust3_archive_copies=source_copies,
        hpc3_restore_reward_encoder_allowance_bytes=encoder_dependency,
        hkust3_archive_encoder_dependency_allowance_bytes=2*encoder_dependency,
        hkust3_archive_allowance_bytes=source_copies*per_pair+2*encoder_dependency,
        other_run_allowance_bytes=96*2**30,other_run_allowance_measured=False,
        recovery_prelaunch_extra_allowance_bytes=recovery_extra,
        placement_policy='five_HPC3_sources_five_HKUST3_archives_one_simultaneous_HPC3_restore_target',
        projected_bytes=measured+reserve,actual_filesystem_hard_quota_verified=False,
        accepted=complete and measured+reserve<=budget,checked_unix=now)


def storage_admission(base):
    from reproduction.common8.disk_admission import checkpoint_floor
    path=base/'preparation/personal-storage-authorization.json';budget=2*10**12
    if path.exists():
        auth=json.loads(path.read_text())
        if auth.get('scope')!='jxu957_prefix20_joint4_200' or not auth.get('explicit_user_authorization'):
            raise ValueError('PREFIX20_PERSONAL_BUDGET_AUTHORIZATION_DIFFERS')
        budget=auth['budget_bytes']
    observed=json.loads((base/'preparation/personal-storage-observation.json').read_text())
    if (observed.get('observer_uid'),observed.get('observing_user')) not in ((14090,'jxu957'),(21930,'mou176')):
        raise ValueError('PREFIX20_STORAGE_OBSERVER_IDENTITY_DIFFERS')
    if observed.get('observer_source_sha256')!=sha(Path(__file__).with_name('storage_observe.py')):
        raise ValueError('PREFIX20_STORAGE_OBSERVER_SOURCE_DIFFERS')
    if observed.get('measurement_kind'):
        from reproduction.prefix20_joint4.storage_complete import validate_upper_bound
        validate_upper_bound(observed,base)
    for key in ('stdout','stderr'):
        path=Path(observed[key+'_file'])
        if not path.is_relative_to(base/'preparation') or path.is_symlink() or sha(path)!=observed[key+'_sha256']:
            raise ValueError('PREFIX20_STORAGE_OBSERVATION_FILE_SHA_DIFFERS')
        if path.read_text()!=observed[key]:raise ValueError('PREFIX20_STORAGE_OBSERVATION_BYTES_DIFFERS')
    proof=storage_plan(observed,checkpoint_floor(Path(MODEL)),budget)
    if not proof['accepted']:raise ValueError('PREFIX20_PERSONAL_STORAGE_RESERVATION_NOT_ADMITTED')
    return proof


def bootstrap(base,commit):
    base=own_base(base);environment=prepare_environment(base)
    source=bind_repo(base,commit)
    return dict(environment=environment,source_commit=commit,source=source,
        training_job_submitted=False,extra_teacher_warmup_updates=0)


def launch_command(base,commit):
    command=['sbatch','--parsable','--partition=emergency_acd','--qos=formal-user','--account=jxu957',
        '--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=7-00:00:00',
        '--job-name=qprobe-prefix20-joint200','--export=NONE',
        '--chdir='+str(base/'repo'),
        '--output='+str(base/'logs/training-%j.log'),'--error='+str(base/'logs/training-%j.log')]
    if base==Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016'):
        permit=json.loads((base/'preparation/failed727148-deferred-node-cleanup.json').read_text())
        if permit.get('source_commit')!=commit or permit.get('node')!='ACD1-25' or permit.get('before_any_cuda') is not True:
            raise ValueError('PREFIX20_RECOVERY727148_BOUND_NODE_CLEANUP_PERMIT_REQUIRED')
        command.append('--nodelist=ACD1-25')
    return command+[str(base/'source'/commit/'reproduction/prefix20_joint4/train.sh'),str(base),commit]


def submit(base,commit):
    base=own_base(base);bootstrap(base,commit)
    intent=base/'preparation/training-submit-claim.json';receipt=base/'preparation/training-submission.json'
    if intent.exists() or receipt.exists() or (base/'job.id').exists():
        raise ValueError('PREFIX20_SUBMISSION_ALREADY_ATTEMPTED_INSPECT_ACTUAL_JOB')
    if any((base/'logs'/name).exists() for name in ('run-manifest.json','training-launch-claim.json','process-ledger.json')):
        raise ValueError('PREFIX20_RUN_ALREADY_PREPARED_OR_CLAIMED')
    assets=assets_ready(base);storage=storage_admission(base);command=launch_command(base,commit)
    publish(intent,dict(source_commit=commit,command=command,assets=assets,storage=storage,created_unix=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(source_commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        command=command,checked_unix=time.time(),training_world_size=4,teacher_warmup_updates=0)
    if result.returncode==0:
        job=result.stdout.strip().split(';')[0]
        if not job.isdecimal():raise ValueError('PREFIX20_INVALID_SLURM_SUBMISSION_ID_INSPECT_CLAIM')
        value['job']=job
        with (base/'job.id').open('x') as out:out.write(job+'\n');out.flush();os.fsync(out.fileno())
    publish(receipt,value)
    if result.returncode:raise RuntimeError('PREFIX20_SUBMISSION_FAILED_INSPECT_SAVED_RECEIPT')
    return value


def validate_slurm_job(base,commit,submitted,actual):
    from reproduction.common8.stage_retrieval import memory_bytes
    expected=dict(JobId=submitted['job'],JobState='RUNNING',Account='jxu957',QOS='formal-user',
        Partition='emergency_acd',JobName='qprobe-prefix20-joint200',NumNodes='1',NumCPUs='48',
        Command=str(base/'source'/commit/'reproduction/prefix20_joint4/train.sh'),WorkDir=str(base/'repo'))
    if submitted['source_commit']!=commit or any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError('PREFIX20_ACTUAL_SLURM_JOB_BINDING_DIFFERS')
    if actual.get('UserId')!='jxu957(14090)' or actual.get('TresPerNode') not in ('gres/gpu:4','gres:gpu:4'):
        raise ValueError('PREFIX20_ACTUAL_SLURM_USER_OR_GPU_COUNT_DIFFERS')
    if memory_bytes(actual['TRES'])<512*2**30:raise ValueError('PREFIX20_ACTUAL_SLURM_MEMORY_TOO_SMALL')
    return dict(expected,UserId=actual['UserId'],TresPerNode=actual['TresPerNode'],TRES=actual['TRES'])


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('bootstrap','submit'))
    p.add_argument('--base',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    value=bootstrap(a.base,a.commit) if a.mode=='bootstrap' else submit(a.base,a.commit)
    print(json.dumps(value))


if __name__=='__main__':main()
