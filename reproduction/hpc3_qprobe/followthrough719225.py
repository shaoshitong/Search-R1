"""Once owner-only followthrough for the unchanged native4 producer719225.

This CPU supervisor never starts/restarts training or edits its source. It
waits for real pairs, retains archive children through the existing controls,
then requests the existing restore-only final audit and full CPU goal consumer.
Unknown claims are inspected, never replayed. Live queue handles are waits,
not acceptance. All output is remote, small and explicit about those limits.
"""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as control

ROOT=control.ROOT
BASE=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
JOB='719225'
PRODUCER='1843babdd96b387dd10d91e3207c9ea9489b6941'
RUN='c3856315d3f345d7920ebf5725a083b4'
DIRECTORY=ROOT/'followthrough'/JOB
TERMINAL_FAILURES={'FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
REGISTRATION_WAIT={}
PAIR_READY_TIMEOUT=10.
PAIR_READY_POLL=.05


def need(ok,message):
    if not ok:raise ValueError('QPROBE_FOLLOWTHROUGH719225_'+message)


def read(path):return control.read(Path(path))


def _pair_snapshot(path):
    """Read a canonical owner file without following a raced-in link/device."""
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink(),
         'PAIR_RECEIPT_UNSAFE_PATH')
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid==control.OWNER,
         'PAIR_RECEIPT_UNSAFE_FILE_OR_OWNER')
    flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
    with os.fdopen(os.open(path,flags),'rb') as stream:
        opened=os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode) and opened.st_uid==control.OWNER and
             (opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino),
             'PAIR_RECEIPT_UNSAFE_OPEN_IDENTITY')
        content=stream.read()
        after=os.fstat(stream.fileno())
    current=path.lstat()
    need(stat.S_ISREG(current.st_mode) and current.st_uid==control.OWNER and
         path.resolve()==path and not path.is_symlink(),'PAIR_RECEIPT_UNSAFE_PATH')
    identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
    if not identity(before)==identity(opened)==identity(after)==identity(current):return None
    return identity(current),content


def pair_ready(step,*,timeout=PAIR_READY_TIMEOUT,poll=PAIR_READY_POLL):
    """Wait before any new claim for two complete, identical JSON snapshots.

    The unchanged producer publishes with open('x') before JSON/fsync finishes.
    A visible path alone is not ready. Only this bounded pre-claim observation
    retries; existing claims, child outcomes and scientific gates never do.
    """
    need(type(step) is int and step in control.STEPS,'PAIR_READY_STEP_INVALID')
    path=BASE/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json'
    deadline=time.monotonic()+timeout;previous=None
    while True:
        snapshot=_pair_snapshot(path)
        if snapshot is not None:
            try:value=json.loads(snapshot[1])
            except (json.JSONDecodeError,UnicodeDecodeError):previous=None
            else:
                need(isinstance(value,dict),'PAIR_RECEIPT_JSON_OBJECT_REQUIRED')
                if previous==snapshot:
                    need(time.monotonic()<deadline,'PAIR_RECEIPT_READY_TIMEOUT_INSPECT')
                    return value
                previous=snapshot
        else:previous=None
        remaining=deadline-time.monotonic()
        need(remaining>0,'PAIR_RECEIPT_READY_TIMEOUT_INSPECT')
        time.sleep(min(poll,remaining))


def publish_state(value):
    temporary=DIRECTORY/'state.partial'
    with temporary.open('x') as stream:
        json.dump(dict(value,checked_unix=time.time()),stream,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(DIRECTORY/'state.json')
    fd=os.open(DIRECTORY,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def fixed_binding():
    """Bind the queued job without requiring a nonexistent live manifest."""
    control.owner();control.caps()
    submission=read(BASE/'preparation/training-submission.json')
    claim=read(BASE/'preparation/training-submit-claim.json')
    preflight=read(BASE/'preparation/configuration-preflight.json')
    need(submission['job']==JOB and submission['returncode']==0 and
         submission['commit']==claim['commit']==preflight['source_commit']==PRODUCER and
         submission['recovery_of_failed_job']=='718650' and
         preflight['status']=='ACTUAL_REMOTE_HYDRA_SAVED5_NATIVE4_CONFIGURATION_PASS' and
         preflight['training_world_size']==4 and preflight['completed_outer']==5 and preflight['next_outer']==6,
         'ACTUAL_NATIVE4_SUBMISSION_OR_PREFLIGHT_DIFFERS')
    need(claim['command'][-3:]==[str(ROOT/'source'/PRODUCER/'reproduction/hpc3_qprobe/train.sh'),str(BASE),PRODUCER] and
         all(value in claim['command'] for value in ('--account=jxu957','--qos=formal-user',
             '--partition=emergency_acd','--gres=gpu:4','--cpus-per-task=48','--mem=512G',
             '--exclude=ACD1-31,ACD1-67','--chdir='+str(BASE))), 'PRODUCER_ENTRY_OR_RESOURCES_CHANGED')
    source_receipt=BASE/'preparation/checkpoint-restoration.json'
    restore=read(source_receipt)
    need(restore['source_receipt_sha256']==claim['source_receipt_sha256']==preflight['source_receipt_sha256']==
         control.digest(restore['source_receipt']), 'SAVED5_SOURCE_ADMISSION_CHANGED')
    return dict(job=JOB,base=str(BASE),producer_commit=PRODUCER,run_id=RUN,
        submission_sha256=control.digest(BASE/'preparation/training-submission.json'),
        claim_sha256=control.digest(BASE/'preparation/training-submit-claim.json'),
        preflight_sha256=control.digest(BASE/'preparation/configuration-preflight.json'),
        restore_binding_sha256=control.digest(source_receipt))


def accounting(job=JOB,*,allow_missing=False):
    raw=subprocess.check_output(['sacct','-nP','-X','-j',job,
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition'],text=True,timeout=30)
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    if not rows and allow_missing:return None
    need(len(rows)==1 and rows[0][0]==job and rows[0][3:]==
         ['jxu957','14090','jxu957','formal-user','emergency_acd'], 'ACTUAL_OWNER_ACCOUNTING_DIFFERS')
    return dict(job=job,state=rows[0][1].split()[0],exit_code=rows[0][2],raw=raw,checked_unix=time.time())


def queued_detail():
    from reproduction.common8.native_allocation import fields
    raw=subprocess.check_output(['scontrol','show','job',JOB],text=True,timeout=30)
    row=fields(raw)
    need(row['JobId']==JOB and row['UserId']=='jxu957(14090)' and row['Account']=='jxu957' and
         row['QOS']=='formal-user' and row['Partition']=='emergency_acd' and row['NumCPUs']=='48' and
         row['MinMemoryNode']=='512G' and row['NumNodes'] in ('1','1-1') and
         row['TresPerNode'] in ('gres/gpu:4','gres:gpu:4') and row['WorkDir']==str(BASE) and
         row['Command']==str(ROOT/'source'/PRODUCER/'reproduction/hpc3_qprobe/train.sh'),
         'ACTUAL_QUEUED_WORKDIR_SOURCE_RESOURCE_DIFFERS')
    return raw


def final_registration_wait(audit):
    """Bounded wait for known sbatch job's DB row, never fabricate its state."""
    job=audit['job'];claim=read(ROOT/'final-control'/RUN/'final-audit/submit-claim.json')
    from reproduction.common8.native_allocation import fields
    raw=subprocess.check_output(['scontrol','show','job','-d',job],text=True,timeout=30);actual=fields(raw)
    need(claim['audit_base']==audit['audit_base'] and actual['JobId']==job and
         actual['UserId']=='jxu957(14090)' and actual['Account']=='jxu957' and actual['QOS']=='formal-user' and
         actual['Partition']=='emergency_acd' and actual['WorkDir']==claim['audit_base'] and
         actual['Command']==claim['command'][-5] and actual['JobName']=='qprobe-final200audit' and
         actual['NumCPUs']=='48' and actual['MinMemoryNode']=='512G' and actual['NumNodes'] in ('1','1-1') and
         actual['TresPerNode'] in ('gres/gpu:4','gres:gpu:4','gres:gpu:h100:4') and
         actual['JobState'] in ('PENDING','RUNNING','COMPLETING','COMPLETED'),
         'KNOWN_NATIVE_AUDIT_REGISTRATION_JOB_DIFFERS')
    first=REGISTRATION_WAIT.setdefault(job,time.monotonic())
    need(time.monotonic()-first<300,'KNOWN_NATIVE_AUDIT_DB_REGISTRATION_TIMEOUT_INSPECT')
    return dict(phase='ACTUAL_BOUND_AUDIT_WAITING_SACCT_REGISTRATION',job=job,actual_job_detail=raw,
                actual_native_restore_accepted=False)


def start(commit):
    binding=fixed_binding();repository,publication=control.source(commit);actual=accounting()
    need(actual['state'] in ('PENDING','RUNNING','COMPLETING','COMPLETED') and
         (actual['state']!='COMPLETED' or actual['exit_code']=='0:0'),'PRODUCER_ALREADY_FAILED')
    detail=queued_detail() if actual['state'] in ('PENDING','RUNNING') else None
    need(not DIRECTORY.exists(),'EXISTING_CONTROL_INSPECT_DO_NOT_REPLAY')
    DIRECTORY.mkdir(parents=True,mode=0o770)
    command=[str(BASE/'env/bin/python'),'-m','reproduction.hpc3_qprobe.followthrough719225',
             'supervise','--commit',commit]
    control.publish(DIRECTORY/'claim.json',dict(schema_version=1,binding=binding,source=publication,
        command=command,cwd=str(repository),actual_initial_accounting=actual,actual_initial_job_detail=detail,
        created_unix=time.time(),training_submission_permitted=False,
        actor_model_reload_permitted=False,direct_gpu_access_permitted=False,
        native_final_restore_only_submission_permitted=True))
    env=dict(os.environ,QPROBE_SOURCE=str(repository),CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',
             **{key:'1' for key in control.CAPS})
    with (DIRECTORY/'output.log').open('x') as output:
        child=subprocess.Popen(command,cwd=repository,env=env,stdin=subprocess.DEVNULL,
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    birth=control.process(child.pid)
    need(birth['uid']==control.OWNER and birth['sid']==birth['pid'],'DETACHED_ACTUAL_OWNER_BIRTH_DIFFERS')
    result=dict(actual_process=control.birth(birth),commit=commit,control=str(DIRECTORY),checked_unix=time.time())
    control.publish(DIRECTORY/'process.json',result)
    return result


def one_tick(commit,verified_archives):
    """Advance only from original real producer evidence, not anticipated work."""
    actual=accounting()
    if actual['state'] in TERMINAL_FAILURES or (actual['state']=='COMPLETED' and actual['exit_code']!='0:0'):
        control.publish(DIRECTORY/'producer-terminal-failure.json',actual)
        raise ValueError('QPROBE_FOLLOWTHROUGH719225_ACTUAL_PRODUCER_FAILED_DIAGNOSE_NO_RESTART')
    if actual['state']=='PENDING':
        return dict(phase='ACTUAL_PRODUCER_QUEUE_WAIT',accounting=actual,actual_job_detail=queued_detail(),
                    actual_restore_accepted=False)
    need(actual['state'] in ('RUNNING','COMPLETING','COMPLETED'),'UNKNOWN_PRODUCER_STATE_INSPECT')
    if not (BASE/'logs/run-manifest.json').is_file() or not (BASE/'logs/native-allocation-admission.json').is_file():
        need(actual['state']!='COMPLETED','FINISHED_PRODUCER_MISSING_NATIVE_BINDING')
        return dict(phase='ACTUAL_PRODUCER_STARTUP_WAIT',accounting=actual,actual_restore_accepted=False)
    run,binding,_=control.producer(BASE)
    need(run['run_id']==RUN and run['source_commit']==PRODUCER and binding['producer_job']==JOB,
         'ACTUAL_NATIVE_PRODUCER_CHANGED')
    components=ROOT/'final-control'/RUN
    archives={}
    for step in control.STEPS:
        pair=BASE/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json'
        directory=components/('archive-'+str(step))
        if not pair.exists() and not pair.is_symlink():
            need(actual['state']!='COMPLETED','COMPLETED_PRODUCER_MISSING_PAIR_'+str(step))
            break
        if not directory.exists():
            pair_ready(step)  # No claim/Popen can exist while the receipt is partial.
            control.launch_archive(BASE,commit,step)
        observed=control.inspect_archive(BASE,step,full=False)
        if observed.get('actual_task_exit_code') not in (None,0):
            raise ValueError('QPROBE_FOLLOWTHROUGH719225_ACTUAL_ARCHIVE_FAILED_'+str(step))
        if observed.get('status')=='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0':
            key=observed['transport_sha256']
            if verified_archives.get(step)!=key:
                verified=control.inspect_archive(BASE,step,full=True)
                need(verified.get('fresh_full_sha') is not None,'ACTUAL_FRESH_ARCHIVE_SHA_MISSING')
                verified_archives[step]=key
            archives[str(step)]=dict(status=observed['status'],transport_path=observed['transport_path'],transport_sha256=key,
                actual_child_exit=0,fresh_full_sha_verified=True)
        else:
            need(observed.get('runner_alive') is True,'ARCHIVE_UNKNOWN_OUTCOME_INSPECT_'+str(step))
            archives[str(step)]=dict(status=observed['status'],actual_payload_bytes=observed['actual_payload_bytes'],
                actual_payload_files=observed['actual_payload_files'],runner_alive=True,actual_child_exit=None)
            break  # Serialize reservations/copy admission and lower producer IO.
    complete_archives=sum(row['status']=='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0' for row in archives.values())
    if actual['state']!='COMPLETED' or complete_archives!=5:
        return dict(phase='ACTUAL_PRODUCER_OR_ARCHIVE_WAIT',accounting=actual,archives=archives,
                    actual_final_native_restore_accepted=False)
    control.producer(BASE,complete=True)
    if not (components/'audit-controller').exists() and not (components/'final-audit').exists():
        control.start_audit(BASE,commit)
    audit=control.inspect_audit(BASE)
    cpu=audit.get('cpu_controller') or {}
    if cpu.get('actual_task_exit_code') not in (None,0):
        raise ValueError('QPROBE_FOLLOWTHROUGH719225_FINAL_CPU_PREFLIGHT_FAILED')
    if not audit.get('job'):
        need(audit['status']!='SUBMIT_FAILED_DO_NOT_REPLAY','ACTUAL_FINAL_SUBMISSION_FAILED_NO_REPLAY')
        need(cpu.get('runner_alive') is True or cpu.get('task_alive') is True,
             'FINAL_SUBMISSION_UNKNOWN_INSPECT_NO_REPLAY')
        return dict(phase='ACTUAL_FINAL_SUBMIT_CPU_WAIT',accounting=actual,archives=archives,audit=audit)
    need(audit['status'] in ('ACTUAL_SLURM_SUBMITTED','ACTUAL_SLURM_RECONCILED'),
         'KNOWN_FINAL_JOB_REQUIRES_ACTUAL_SUBMIT_OR_RECONCILE')
    native=accounting(audit['job'],allow_missing=True)
    if native is None:return final_registration_wait(audit)
    REGISTRATION_WAIT.pop(audit['job'],None)
    if native['state'] in TERMINAL_FAILURES or (native['state']=='COMPLETED' and native['exit_code']!='0:0'):
        control.publish(DIRECTORY/'final-audit-terminal-failure.json',native)
        raise ValueError('QPROBE_FOLLOWTHROUGH719225_ACTUAL_NATIVE_FINAL_AUDIT_FAILED')
    if native['state']!='COMPLETED':
        return dict(phase='ACTUAL_FINAL_NATIVE_RESTORE_WAIT',accounting=actual,archives=archives,audit=native)
    accepted=BASE/'logs/qprobe-final-goal-acceptance.json'
    consumer_binding=DIRECTORY/'final-consumer-binding.json'
    if not accepted.exists():
        from reproduction.hpc3_qprobe.verify_goal import verify
        verify(BASE,audit['audit_base'])
        control.publish(consumer_binding,dict(actual_process=control.birth(control.process(os.getpid())),
            helper_commit=commit,audit_base=audit['audit_base'],goal_sha256=control.digest(accepted),
            actual_native_audit_accounting=native,full_cpu_consumer_returned=True,checked_unix=time.time()))
    consumed=read(consumer_binding)
    need(consumed['actual_process']==control.birth(control.process(os.getpid())) and
         consumed['helper_commit']==commit and consumed['audit_base']==audit['audit_base'] and
         consumed['goal_sha256']==control.digest(accepted) and consumed['full_cpu_consumer_returned'] is True,
         'EXISTING_GOAL_RECEIPT_NOT_THIS_ACTUAL_CPU_CONSUMER')
    proof=read(accepted)
    need(proof['status']=='QPROBE_FULL200_TEACHER_HEAD_FIXED_VALIDATION_ARCHIVE_NATIVE4_GOAL_PASS' and
         proof['run_id']==RUN and proof['completed_outer']==200 and proof['update201_performed'] is False,
         'FINAL_CPU_GOAL_CONSUMER_NOT_ACCEPTED')
    return dict(phase='ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS',accounting=actual,archives=archives,
        actual_final_native_restore_accepted=True,final_goal_path=str(accepted),final_goal_sha256=control.digest(accepted),
        full_objective_accepted=True)


def supervise(commit):
    code=0;error=None;result=None
    birth=control.birth(control.process(os.getpid()))
    try:
        deadline=time.monotonic()+15
        while not (DIRECTORY/'process.json').is_file():
            need(time.monotonic()<deadline,'PARENT_PROCESS_BINDING_MISSING_INSPECT_NO_REPLAY')
            time.sleep(.05)
        claim=read(DIRECTORY/'claim.json');record=read(DIRECTORY/'process.json')
        need(birth==record['actual_process'] and claim['binding']==fixed_binding() and
             claim['source']['commit']==commit and claim['cwd']==str(Path.cwd().resolve()),'ACTUAL_SUPERVISOR_BINDING_DIFFERS')
        control.source(commit);verified={}
        control.publish(DIRECTORY/'started.json',dict(actual_process=birth,checked_unix=time.time()))
        while True:
            result=one_tick(commit,verified);publish_state(result)
            if result.get('full_objective_accepted'):break
            time.sleep(60)
    except Exception as exc:
        import traceback
        traceback.print_exc();code=1;error=repr(exc)
        publish_state(dict(phase='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS',error=error,
                           previous_actual_state=result,full_objective_accepted=False))
    control.publish(DIRECTORY/'result.json',dict(exit_code=code,error=error,result=result,
        actual_process=birth,checked_unix=time.time(),training_submission_performed=False,
        actor_model_loaded=False,direct_gpu_device_access_performed=False,
        native_final_restore_only_submission_permitted=True))
    return code


def inspect():
    control.owner();claim=read(DIRECTORY/'claim.json');record=read(DIRECTORY/'process.json')
    need(claim['binding']==fixed_binding(),'FIXED_INTENT_CHANGED')
    return dict(actual_live_birth=control.live(record['actual_process']),process=record,
        state=read(DIRECTORY/'state.json') if (DIRECTORY/'state.json').is_file() else None,
        result=read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').is_file() else None,
        checked_unix=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','supervise','inspect'));p.add_argument('--commit')
    a=p.parse_args();control.owner();control.caps()
    if a.mode=='supervise':sys.exit(supervise(a.commit))
    print(json.dumps(start(a.commit) if a.mode=='start' else inspect()),flush=True)
