"""One typed CPU handoff for actual9c archive5 failure before any copy.

Never changes the actor/source or submits training. Old follower remains in
place; failed archive control is atomically preserved once. A root-observed
fresh successful owner du receipt is required, but never replaces the archive
worker's independent fresh storage gate. Unknown outcomes cannot be replayed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as control
from reproduction.hpc3_qprobe import followthrough_sigma3 as original
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader

ROOT=control.ROOT
BASE=original.BASE
RUN=original.RUN
JOB=original.JOB
PRODUCER=original.PRODUCER
OLD='9c34720ecd564dfe902127d0167474d27b115cfe'
DIRECTORY=ROOT/'followthrough/721485-archive5-recovery-20261015'
INTENT=ROOT/'final-control'/RUN/'archive5-recovery-first-9c'
ARCHIVE=ROOT/'final-control'/RUN/'archive-5'
PRESERVED=ROOT/'final-control'/RUN/'failed-archive-5-first-9c'
OLD_FOLLOW=ROOT/'followthrough/721485'
PAYLOAD=ROOT/'archives'/RUN/'global_step_5'
CLAIM_SHA='cf38e6d5d92af430cf327ee315d2a20e13f4260a214b67dc4ea63866669eb8c2'
PAIR_SHA='eb13a22502399c9cdbe458c5cbd73e8f07ebdfd9a347f2ef55508d3224d58339'
OLD_CONTROLLER=dict(pid=1045047,start_ticks=75858599,sid=1045047,uid=14090)
OLD_RUNNER=dict(pid=2149540,start_ticks=76216560,sid=2149540,uid=14090)
OLD_CHILD=dict(pid=2155258,start_ticks=76218684,sid=2149540,uid=14090)
FINISHED=1792052826.0763557
STOPPED=1792052865.3746393
ERROR="ValueError('QPROBE_FOLLOWTHROUGH_SIGMA3_ACTUAL_ARCHIVE_FAILED_5')"
STATUS='SIGMA3_ARCHIVE5_FIRST_9C_PRE_COPY_FAILURE_PRESERVED_ONCE'
MODULE='reproduction.hpc3_qprobe.recover_sigma3_archive5'


def need(ok,message):control.need(ok,'SIGMA3_ARCHIVE5_RECOVERY: '+message)


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def snapshot(folder):
    """Complete small evidence fileset; links/devices/unsafe owners refuse."""
    folder=Path(folder)
    need(folder.is_absolute() and folder.resolve()==folder and not folder.is_symlink() and
         stat.S_ISDIR(folder.lstat().st_mode) and folder.stat().st_uid==control.OWNER,'unsafe evidence directory')
    result={}
    for path in sorted(folder.rglob('*')):
        st=path.lstat()
        need(path.resolve()==path and not path.is_symlink() and st.st_uid==control.OWNER,'unsafe evidence member')
        if stat.S_ISDIR(st.st_mode):continue
        need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size<=32*2**20,'unsafe or unbounded evidence file')
        _,proof=reader.read_file(path)
        result[str(path.relative_to(folder))]=dict(bytes=proof['bytes'],sha256=proof['sha256'],
            inode=st.st_ino,device=st.st_dev,mode=stat.S_IMODE(st.st_mode),uid=st.st_uid,gid=st.st_gid)
    need(result,'empty evidence directory')
    return result


def _read(folder,name):return control.read(Path(folder)/name)


def failure_evidence(archive=None):
    """Only the observed held1 + exact storage exception + dead3 births."""
    archive=ARCHIVE if archive is None else Path(archive)
    need(archive in (ARCHIVE,PRESERVED),'failure evidence path differs')
    _,old_source=reader.publication(OLD)
    run,producer,_=control.producer(BASE)
    need(run['run_id']==RUN and run['source_commit']==PRODUCER and producer['producer_job']==JOB and
         not run.get('full_state_restore'),'unchanged fresh producer binding differs')
    pair_path,pair=control.pair(BASE,run,5)
    need(pair['sha256']==PAIR_SHA,'actual original pair5 changed')
    claim=_read(archive,'submit-claim.json');runner=_read(archive,'runner-binding.json')
    child=_read(archive,'task-binding.json');wait=_read(archive,'worker-result.json')
    need(control.digest(archive/'submit-claim.json')==CLAIM_SHA and claim['source']==old_source and
         claim['producer']==producer and claim['pair']==pair and claim['step']==5 and
         claim['schema_version']==1 and claim['role']=='detached_cpu_pair_archive' and
         claim['controller']==OLD_CONTROLLER and claim['command']==control.command(ROOT/'source'/OLD,
             'archive-worker',ARCHIVE/'submit-claim.json',claim['token']) and
         all(claim[k] is False for k in ('training_permitted','direct_gpu_device_access_permitted',
                                        'native_gpu_audit_submission_permitted')),'original archive intent differs')
    need(runner['actual_process']==OLD_RUNNER and runner['controller']==OLD_CONTROLLER and
         child['actual_process']==OLD_CHILD and child['supervisor']==OLD_RUNNER and
         wait['actual_task_birth']==OLD_CHILD and wait['actual_supervisor']==OLD_RUNNER and
         wait['actual_task_exit_code']==1 and type(wait['actual_task_exit_code']) is int and
         wait['status']=='CPU_CHILD_FAILED_OR_MISSING_RECEIPT' and wait['role']==claim['role'] and
         wait['finished_unix']==FINISHED and all(v['token']==claim['token'] and v['claim_sha256']==CLAIM_SHA
             for v in (runner,child,wait)),'original held failure/birth differs')
    need(not any(control.live(v) for v in (OLD_CONTROLLER,OLD_RUNNER,OLD_CHILD)),
         'original follower/runner/child live or unknown; no recovery')
    need(not (archive/'task-result.json').exists() and not (archive/'task-result.json').is_symlink(),
         'original task result exists; not the known pre-copy failure')
    task_log,task_record=reader.read_file(archive/'task.log',capture=True)
    task_text=task_log.decode('utf-8')
    expected_frame='  File "'+str(ROOT/'source'/OLD/'reproduction/hpc3_qprobe/archive_completed_pair.py')+'", line 102, in archive\n'
    need(task_text.count('Traceback (most recent call last):')==1 and expected_frame in task_text and
         "    if measured.returncode or measured.stderr.strip():raise ValueError('Fresh owner storage measurement failed')\n" in task_text and
         task_text.endswith('ValueError: Fresh owner storage measurement failed\n'),
         'not the exact original storage measurement exception')
    process=_read(OLD_FOLLOW,'process.json');result=_read(OLD_FOLLOW,'result.json')
    state=_read(OLD_FOLLOW,'state.json');fclaim=_read(OLD_FOLLOW,'claim.json')
    need(process['actual_process']==result['actual_process']==OLD_CONTROLLER and process['commit']==OLD and
         process['control']==str(OLD_FOLLOW) and result['exit_code']==1 and type(result['exit_code']) is int and
         result['error']==state['error']==ERROR and result['checked_unix']==STOPPED and
         state['phase']=='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS' and state['full_objective_accepted'] is False and
         fclaim['source']==old_source and fclaim['binding']==fixed_binding() and
         fclaim['cwd']==str(ROOT/'source'/OLD) and fclaim['command']==[str(BASE/'env/bin/python'),'-m',
             'reproduction.hpc3_qprobe.followthrough_sigma3','supervise','--commit',OLD] and
         all(result[k] is False for k in ('training_submission_performed','actor_model_loaded','direct_gpu_device_access_performed')),
         'original follower did not return the known archive5 failure')
    need(all(fclaim.get(k) is False for k in ('training_submission_permitted','actor_model_reload_permitted','direct_gpu_access_permitted')) and
         fclaim.get('native_final_restore_only_submission_permitted') is True and
         result.get('native_final_restore_only_submission_permitted') is True and
         state.get('previous_actual_state')==result['result'], 'original follower scope/state differs')
    need(not (OLD_FOLLOW/'final-consumer-binding.json').exists() and not (OLD_FOLLOW/'producer-terminal-failure.json').exists(),
         'different original follower terminal path')
    previous=result['result']
    need(previous['phase']=='ACTUAL_PRODUCER_OR_ARCHIVE_WAIT' and previous['accounting']['job']==JOB and
         previous['accounting']['state']=='RUNNING' and previous['accounting']['exit_code']=='0:0' and
         set(previous['archives'])=={'5'} and previous['archives']['5']['actual_payload_bytes']==0 and
         previous['archives']['5']['actual_payload_files']==0,'original failure did not precede copy')
    return dict(status=STATUS,producer=producer,source=old_source,pair=pair,
        archive_files=snapshot(archive),follower_files=snapshot(OLD_FOLLOW),
        actual_original_follower=OLD_CONTROLLER,actual_original_runner=OLD_RUNNER,actual_original_child=OLD_CHILD,
        original_held_exit_code=1,original_failure_cause='storage_measurement_failed; original stderr was not retained',
        original_finished_unix=FINISHED,original_follower_stopped_unix=STOPPED)


def no_payload():
    need(not PAYLOAD.exists() and not PAYLOAD.is_symlink(),'any original or partial archive payload forbids recovery')


def successful_du(path,*,now=None):
    value,record=reader.document(path);now=time.time() if now is None else now
    root=str(ROOT.parent)
    need(value.get('root')==root and value.get('uid')==control.OWNER and type(value.get('returncode')) is int and
         value['returncode']==0 and value.get('stderr')=='' and type(value.get('allocated_bytes')) is int and
         0<=value['allocated_bytes']<2*10**12 and
         value.get('command')==['du','-s','-B1',root] and
         value.get('stdout')==str(value['allocated_bytes'])+'\t'+root+'\n' and
         all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) for k in ('started_unix','completed_unix')) and
         value['started_unix']<=value['completed_unix']<=now and now-value['completed_unix']<=600,
         'fresh successful actual owner du receipt required')
    original=value.get('original_result') or {}
    actual,actual_record=reader.document(Path(original.get('path','')))
    need(actual_record['sha256']==original.get('sha256'),'actual du observer result SHA differs')
    observer_source=actual.get('source') or {}
    observer,publication=reader.load_fixed(observer_source.get('commit'),
        'storage_observer_sigma3_archive5_failure','_unused_control')
    observed=observer.inspect()
    need(Path(original['path'])==observer.CONTROL/'result.json' and
         observed['status']=='ACTUAL_DU_COMPLETED0_USAGE_MEASURED' and
         observed['actual_live_birth'] is False and observed['actual_du_live_birth'] is False and
         observed['result']==actual and actual['allocated_bytes']==value['allocated_bytes'] and
         actual['stdout']==value['stdout'] and actual['stderr']==value['stderr'] and
         actual['actual_du_exit_code']==value['returncode'] and
         actual['started_unix']==value['started_unix'] and actual['completed_unix']==value['completed_unix'] and
         observer_source['commit']==publication['commit'], 'actual held du observer admission differs')
    return record


def consume(commit):
    """Read-only original SHA + actual moved files + exclusive migration proof."""
    repository,source=control.source(commit)
    need(Path(__file__).resolve()==repository/'reproduction/hpc3_qprobe/recover_sigma3_archive5.py',
         'loaded recovery module differs from its immutable source')
    intent=_read(INTENT,'intent.json');migration=_read(INTENT,'migration.json')
    need(intent['source']==source and intent['failure']['status']==STATUS and
         intent['original_archive']==str(ARCHIVE) and intent['preserved_archive']==str(PRESERVED) and
         intent['new_follower']==str(DIRECTORY) and intent['training_submission_permitted'] is False and
         migration['intent_sha256']==control.digest(INTENT/'intent.json') and migration['status']==STATUS and
         migration['before_files']==migration['after_files']==intent['failure']['archive_files'],
         'recovery intent/migration incomplete or different')
    need(failure_evidence(PRESERVED)==intent['failure'],'original failure bytes changed after handoff')
    # Rehash du evidence; its original admission freshness is recorded, not
    # required forever. Every actual archive worker still runs its own du.
    row=intent['du_receipt'];du,du_record=reader.document(Path(row['path']))
    need(du_record['sha256']==row['sha256'] and du_record['bytes']==row['bytes'] and
         du['completed_unix']<=intent['created_unix'] and intent['created_unix']-du['completed_unix']<=600,
         'du receipt/admission-time freshness changed')
    successful_du(row['path'],now=intent['created_unix'])
    return dict(status=STATUS,source=source,intent=dict(path=str(INTENT/'intent.json'),sha256=control.digest(INTENT/'intent.json')),
        migration=dict(path=str(INTENT/'migration.json'),sha256=control.digest(INTENT/'migration.json')),
        failure_fingerprint=hashlib.sha256(json.dumps(intent['failure'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_follower=OLD_CONTROLLER,original_archive_child_exit=1,original_failure_preserved=True,
        training_source=PRODUCER,producer_job=JOB,run_id=RUN)


def core():
    """Private module instance: no mutation of the old imported follower."""
    path=Path(__file__).with_name('followthrough_sigma3.py')
    spec=importlib.util.spec_from_file_location('_sigma3_archive5_recovered_followthrough',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.DIRECTORY=DIRECTORY
    module.control=control
    return module


def start(commit,du_receipt):
    control.owner();control.caps()
    # Existing intent, including any crash/unknown migration, is never replayed.
    need(not INTENT.exists() and not INTENT.is_symlink() and not DIRECTORY.exists() and
         not DIRECTORY.is_symlink() and not PRESERVED.exists() and not PRESERVED.is_symlink(),
         'existing recovery intent/follower/preservation; inspect only, never replay')
    repository,source=control.source(commit)
    need(Path(__file__).resolve()==repository/'reproduction/hpc3_qprobe/recover_sigma3_archive5.py',
         'loaded recovery module differs from its immutable source')
    need(commit!=OLD and re.fullmatch('[0-9a-f]{40}',commit),'new explicit immutable recovery source required')
    failure=failure_evidence();no_payload();du=successful_du(du_receipt)
    # Bind source compatibility before making a recovery intent.
    run,_,accounting=control.producer(BASE)
    actual=[row.split('|') for row in accounting.splitlines() if row]
    need(len(actual)==1 and actual[0][0]==JOB and actual[0][1] in ('RUNNING','COMPLETING','COMPLETED') and
         (actual[0][1]!='COMPLETED' or actual[0][2]=='0:0'),'actual producer is not healthy; no CPU handoff')
    publication=control.read(Path(source['publication_path']))
    for name,sha in run['source_sha256'].items():
        if name.startswith(('verl/','search_g1/','search_r1/','reproduction/common8/')) and not Path(name).name.startswith('test_'):
            need(publication['files'].get(name)==sha,'recovery science source differs: '+name)
    INTENT.mkdir(mode=0o770)
    control.publish(INTENT/'intent.json',dict(source=source,failure=failure,du_receipt=du,original_archive=str(ARCHIVE),
        preserved_archive=str(PRESERVED),new_follower=str(DIRECTORY),created_unix=time.time(),training_submission_permitted=False))
    need(failure_evidence()==failure,'failure changed before atomic preservation');no_payload()
    need(not PRESERVED.exists() and not PRESERVED.is_symlink(),'preserved target unexpectedly appeared; unknown no replay')
    os.rename(ARCHIVE,PRESERVED);fsync_dir(ARCHIVE.parent)
    after=snapshot(PRESERVED)
    need(after==failure['archive_files'] and failure_evidence(PRESERVED)==failure,'atomic preservation changed original bytes')
    control.publish(INTENT/'migration.json',dict(status=STATUS,intent_sha256=control.digest(INTENT/'intent.json'),
        before_files=failure['archive_files'],after_files=after,completed_unix=time.time()))
    reference=consume(commit)
    DIRECTORY.mkdir(mode=0o770)
    binding=fixed_binding();command=[str(BASE/'env/bin/python'),'-m',MODULE,'supervise','--commit',commit]
    control.publish(DIRECTORY/'claim.json',dict(schema_version=1,binding=binding,source=source,recovery=reference,
        command=command,cwd=str(repository),created_unix=time.time(),training_submission_permitted=False,
        actor_model_reload_permitted=False,direct_gpu_access_permitted=False,native_final_restore_only_submission_permitted=True))
    env=dict(os.environ,QPROBE_SOURCE=str(repository),CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',
             **{key:'1' for key in control.CAPS})
    with (DIRECTORY/'output.log').open('x') as out:
        child=subprocess.Popen(command,cwd=repository,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    actual=control.birth(control.process(child.pid))
    need(actual['uid']==control.OWNER and actual['sid']==actual['pid'],'new detached follower owner/birth differs')
    result=dict(actual_process=actual,commit=commit,control=str(DIRECTORY),checked_unix=time.time())
    control.publish(DIRECTORY/'process.json',result);return result


def supervise(commit):
    control.owner();control.caps()
    deadline=time.monotonic()+15
    while not (DIRECTORY/'process.json').exists():
        need(time.monotonic()<deadline,'new parent birth missing; unknown no replay');time.sleep(.05)
    need(_read(DIRECTORY,'claim.json')['recovery']==consume(commit),'new follower lost original typed handoff')
    # Existing generic loop keeps all five actual archive/full SHA/native4 and
    # full CPU gates. Its commands always use this new explicit source commit.
    return core().supervise(commit)


def fixed_binding():return core().fixed_binding()


def inspect():
    claim=_read(DIRECTORY,'claim.json')
    need(claim['recovery']==consume(claim['source']['commit']),'recovered follower failure binding changed')
    return core().inspect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('start','supervise','inspect','consume'))
    p.add_argument('--commit',required=True);p.add_argument('--du-receipt');a=p.parse_args();control.owner();control.caps()
    if a.mode=='supervise':sys.exit(supervise(a.commit))
    value=start(a.commit,a.du_receipt) if a.mode=='start' else consume(a.commit) if a.mode=='consume' else inspect()
    print(json.dumps(value,allow_nan=False),flush=True)
