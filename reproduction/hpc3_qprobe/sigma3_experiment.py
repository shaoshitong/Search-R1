"""Isolated, once-only four-GPU loss ablation; never changes the legacy run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015')
LEGACY = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014')
TEACHER = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014/teacher-warmup-cache')
LOSS = dict(loss_mode='sigma3', mean_loss_weight=0., std_loss_weight=1., auxiliary_count_loss_weight=.1)


def validate_settings(value, base):
    if Path(base).resolve() != BASE or value.get('base') != str(BASE):
        raise ValueError('SIGMA3_NEW_ROOT_BINDING_DIFFERS')
    if value.get('head_overrides') != LOSS or value.get('fresh_official_actor') is not True:
        raise ValueError('SIGMA3_LOSS_OR_FRESH_ACTOR_DIFFERS')
    if value.get('hidden_size') != 64 or value.get('teacher_origin') != str(TEACHER):
        raise ValueError('SIGMA3_CAPACITY_OR_TEACHER_ORIGIN_DIFFERS')
    return value


def isolated_base():
    if BASE.is_symlink() or BASE.resolve()!=BASE:
        raise ValueError('SIGMA3_OUTPUT_ROOT_MUST_NOT_ALIAS_ANOTHER_RUN')
    return BASE


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def stage_and_submit(commit):
    from reproduction.hpc3_qprobe.control import prepare, link, publish
    from search_g1.qprobe_warmup import TeacherCache, STATUS
    from search_g1.qprobe_features import FeatureCache
    if os.getuid()!=14090 or len(commit)!=40:
        raise ValueError('SIGMA3_ACTUAL_OWNER_AND_SOURCE_REQUIRED')
    isolated_base()
    source=ROOT/'source'/commit
    verification=ROOT/'preparation'/('source-'+commit+'-verification.json')
    proof=json.loads(verification.read_text())
    if proof['commit']!=commit or proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS':
        raise ValueError('SIGMA3_SOURCE_NOT_PUBLISHED')
    for name, sha in proof['files'].items():
        if _sha(source/name)!=sha:raise ValueError('SIGMA3_SOURCE_BYTES_DIFFER: '+name)
    if any((BASE/'preparation'/name).exists() for name in ('training-submit-claim.json','training-submission.json')):
        raise ValueError('SIGMA3_ALREADY_SUBMITTED_INSPECT_ORIGINAL')
    prepare(BASE)
    link(BASE/'repo', source)
    (BASE/'source').mkdir(exist_ok=True)
    link(BASE/'source'/commit,source)
    shutil.copy2(verification,BASE/'preparation'/verification.name)
    for name in ('features-ready.json','official-model-manifest.json'):
        origin=LEGACY/'preparation'/name
        target=BASE/'preparation'/name
        if target.exists():
            if _sha(target)!=_sha(origin):raise ValueError('SIGMA3_REUSED_PREPARATION_DIFFERS')
        else:shutil.copy2(origin,target)
    link(BASE/'question-features',ROOT/'question-features')
    features=FeatureCache(BASE/'question-features');features.verify_all()
    if features.manifest['count']!=169028:raise ValueError('SIGMA3_FEATURE_COUNT_DIFFERS')
    identity=json.loads((TEACHER/'identity.json').read_text())
    if identity['training_world_size']!=4:raise ValueError('SIGMA3_TEACHER_TOPOLOGY_DIFFERS')
    # The raw teacher cache is copied entirely on remote shared storage. Its
    # lock/metadata are private; no original teacher writer or head is reused.
    target=BASE/'teacher-warmup-cache'
    if target.is_symlink() or target.resolve()!=target:
        raise ValueError('SIGMA3_TEACHER_DIRECTORY_MUST_BE_PRIVATE')
    target.mkdir(exist_ok=True)
    names=['identity.json','complete.json']+['chunk-%02d.json'%i for i in range(10)]+['rollout-%02d.pt'%i for i in range(10)]
    copied=[]
    for name in names:
        origin=TEACHER/name;dest=target/name
        if not origin.is_file() or origin.is_symlink():raise ValueError('SIGMA3_TEACHER_FILE_MISSING: '+name)
        if dest.is_symlink():raise ValueError('SIGMA3_TEACHER_TARGET_IS_SYMLINK')
        if dest.exists() and (not stat.S_ISREG(dest.stat().st_mode) or dest.stat().st_nlink!=1 or (dest.stat().st_dev,dest.stat().st_ino)==(origin.stat().st_dev,origin.stat().st_ino)):
            raise ValueError('SIGMA3_TEACHER_TARGET_IS_NOT_INDEPENDENT')
        if not dest.exists():
            partial=dest.with_suffix(dest.suffix+'.partial')
            if partial.exists() or partial.is_symlink():raise ValueError('SIGMA3_PARTIAL_COPY_REQUIRES_INSPECTION')
            shutil.copyfile(origin,partial)
            with partial.open('rb') as f:os.fsync(f.fileno())
            partial.rename(dest)
        if _sha(dest)!=_sha(origin):raise ValueError('SIGMA3_TEACHER_COPY_SHA_DIFFERS: '+name)
        copied.append(dict(name=name,bytes=dest.stat().st_size,sha256=_sha(dest)))
        print(json.dumps(dict(stage='sigma3_teacher_copy_verified',name=name,files=len(copied),epoch=time.time())),flush=True)
    cache=TeacherCache(target,identity)
    try:receipt,groups,chunks=cache.verify_complete()
    finally:cache.close()
    if receipt['status']!=STATUS or len(groups)!=5120:raise ValueError('SIGMA3_TEACHER_NOT_COMPLETE')
    settings=dict(base=str(BASE),head_overrides=LOSS,hidden_size=64,fresh_official_actor=True,
                  teacher_origin=str(TEACHER),target_successful_outers=200,
                  fixed_validation=[5,50,100,150,200],no_update201=True)
    validate_settings(settings,BASE)
    publish(BASE/'preparation/sigma3-experiment.json',settings)
    publish(BASE/'preparation/teacher-copy-verification.json',dict(origin=str(TEACHER),target=str(target),
        files=copied,questions=5120,trajectories=25600,receipt=receipt,checked_unix=time.time()))
    publish(BASE/'preparation/personal-storage-authorization.json',dict(budget_bytes=2*10**12,scope='jxu957_native4_qprobe_only',
        reason='existing user authorization of remote 2TB maximum; real prelaunch usage measured on allocation'))
    for name in ('checkpoint-restoration.json','initial-head.pt'):
        if (BASE/'preparation'/name).exists():raise ValueError('SIGMA3_MUST_NOT_RESTORE_LEGACY_STATE')
    # Existing servers use fixed ports, so prevent this separate experiment
    # from sharing an allocation node with another active question-probe run.
    queue=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%j|%T|%N|%Z'],text=True,timeout=20)
    excluded={'ACD1-31','ACD1-67'}
    for line in queue.splitlines():
        row=line.split('|')
        if len(row)==5 and row[4]==str(BASE):raise ValueError('SIGMA3_EXISTING_QUEUE_JOB_REQUIRES_RECONCILIATION')
        if len(row)==5 and 'qprobe' in row[1] and row[2]=='RUNNING':
            excluded.update(subprocess.check_output(['scontrol','show','hostnames',row[3]],text=True,timeout=10).split())
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=emergency_acd',
        '--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=7-00:00:00',
        '--exclude='+','.join(sorted(excluded)),'--job-name=qprobe-sigma3-200','--export=NONE','--chdir='+str(BASE),
        '--output='+str(BASE/'logs/training-%j.log'),'--error='+str(BASE/'logs/training-%j.log'),
        str(source/'reproduction/hpc3_qprobe/train.sh'),str(BASE),commit]
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=30)
    publish(BASE/'preparation/training-test-only.json',dict(command=command,exit=test.returncode,stdout=test.stdout,stderr=test.stderr,epoch=time.time()))
    if test.returncode:raise ValueError('SIGMA3_SCHEDULER_TEST_FAILED')
    publish(BASE/'preparation/training-submit-claim.json',dict(commit=commit,command=command,epoch=time.time(),
        independent_experiment=True,legacy_job_untouched='719225'))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0]
        if not value['job'].isdecimal():raise ValueError('SIGMA3_INVALID_SUBMITTED_JOB')
    publish(BASE/'preparation/training-submission.json',value)
    if result.returncode:raise ValueError('SIGMA3_SUBMISSION_FAILED_SAVED_RECEIPT')
    return value


def worker(commit):
    from reproduction.hpc3_qprobe.control import publish
    try:
        result=stage_and_submit(commit)
        publish(BASE/'preparation/sigma3-stage-complete.json',dict(result=result,epoch=time.time()))
        print(json.dumps(result),flush=True)
    except BaseException as e:
        publish(BASE/'preparation/sigma3-stage-failure.json',dict(type=type(e).__name__,error=str(e),epoch=time.time()))
        raise


def same_birth(left, right):
    return bool(left and right and all(left.get(k)==right.get(k) for k in ('pid','start_ticks','sid','uid')))


def accepted_stage(out):
    controller=out.get('sigma3-stage-controller.json',{})
    child=out.get('sigma3-stage-child.json',{})
    wait=out.get('sigma3-stage-wait.json',{})
    complete=out.get('sigma3-stage-complete.json',{})
    return bool(controller and child and wait and complete and not out.get('sigma3-stage-failure.json')
        and not out.get('sigma3-supervisor-failure.json') and wait.get('actual_wait_exit_code')==0
        and wait.get('complete_receipt_exists') is True
        and controller.get('commit')==child.get('commit')==wait.get('commit')==complete.get('result',{}).get('commit')
        and same_birth(wait.get('child_birth'),child.get('birth'))
        and same_birth(wait.get('supervisor_birth'),child.get('supervisor_birth'))
        and same_birth(wait.get('supervisor_birth'),controller.get('birth')))


def supervisor(commit):
    from reproduction.hpc3_qprobe.control import publish
    from reproduction.a800.owned_processes import identity
    isolated_base()
    source=ROOT/'source'/commit
    command=[str(LEGACY/'env/bin/python'),'-u','-m','reproduction.hpc3_qprobe.sigma3_experiment','worker','--commit',commit]
    child=None;birth=None
    try:
        child=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL)
        birth=identity(child.pid)
        publish(BASE/'preparation/sigma3-stage-child.json',dict(pid=child.pid,birth=birth,command=command,
            supervisor_birth=identity(os.getpid()),commit=commit,epoch=time.time()))
        code=child.wait()
        result=dict(actual_wait_exit_code=code,child_birth=birth,supervisor_birth=identity(os.getpid()),
            commit=commit,epoch=time.time(),complete_receipt_exists=(BASE/'preparation/sigma3-stage-complete.json').is_file())
        publish(BASE/'preparation/sigma3-stage-wait.json',result)
        if code:raise SystemExit(code)
        return result
    except BaseException as error:
        # Popen retains this direct child's identity until wait reaps it.
        # Only this child is signalled; any submitted Slurm job is reconciled
        # from its once claim, never cancelled or submitted again here.
        if child is not None and child.returncode is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        publish(BASE/'preparation/sigma3-supervisor-failure.json',dict(error=str(error),
            type=type(error).__name__,actual_child_wait=None if child is None else child.returncode,
            child_birth=birth,commit=commit,epoch=time.time()))
        raise


def start(commit):
    from reproduction.hpc3_qprobe.control import publish
    from reproduction.a800.owned_processes import identity
    if os.getuid()!=14090 or not commit or len(commit)!=40:raise ValueError('SIGMA3_OWNER_AND_COMMIT_REQUIRED')
    isolated_base()
    prep=BASE/'preparation';prep.mkdir(parents=True,exist_ok=True);(BASE/'logs').mkdir(exist_ok=True)
    claim=prep/'sigma3-stage-claim.json'
    publish(claim,dict(commit=commit,epoch=time.time(),base=str(BASE)))
    source=ROOT/'source'/commit
    log=(BASE/'logs/sigma3-stage.log').open('ab')
    env=dict(os.environ,QPROBE_SOURCE=str(source),PYTHONPATH=str(source),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    child=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u','-m','reproduction.hpc3_qprobe.sigma3_experiment','supervisor','--commit',commit],
        cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    birth=identity(child.pid)
    value=dict(pid=child.pid,birth=birth,commit=commit,base=str(BASE),epoch=time.time())
    publish(prep/'sigma3-stage-controller.json',value)
    return value


def inspect():
    out=dict(epoch=time.time(),base=str(BASE))
    for name in ('sigma3-stage-controller.json','sigma3-supervisor-failure.json','sigma3-stage-child.json','sigma3-stage-wait.json','sigma3-stage-complete.json','sigma3-stage-failure.json','training-submission.json'):
        p=BASE/'preparation'/name
        if p.is_file():out[name]=json.loads(p.read_text())
    from reproduction.a800.owned_processes import identity
    controller=out.get('sigma3-stage-controller.json')
    if controller:
        try:
            actual=identity(controller['pid'])
            out['supervisor_actual_live_birth']=same_birth(actual,controller['birth']) and actual['state']!='Z'
        except (ProcessLookupError,FileNotFoundError):out['supervisor_actual_live_birth']=False
    out['stage_accepted']=accepted_stage(out)
    for p in [BASE/'logs/sigma3-stage.log']+sorted((BASE/'logs').glob('training-*.log')):
        if p.is_file():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-10000));tail=f.read().decode(errors='replace')
            out[p.name]=dict(bytes=p.stat().st_size,tail=tail)
    r=out.get('training-submission.json',{})
    if r.get('job'):
        out['accounting']=subprocess.check_output(['sacct','-nP','-j',r['job'],'-X','--format=JobID,State,ExitCode,UID,AllocTRES,NodeList'],text=True,timeout=20)
        out['queue']=subprocess.check_output(['squeue','-h','-j',r['job'],'-o','%i|%T|%R'],text=True,timeout=20)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','worker','supervisor','inspect'));p.add_argument('--commit');a=p.parse_args()
    print(json.dumps(inspect() if a.mode=='inspect' else (start(a.commit) if a.mode=='start' else supervisor(a.commit) if a.mode=='supervisor' else worker(a.commit))),flush=True)
