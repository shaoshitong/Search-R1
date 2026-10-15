"""One typed CPU handoff for actual B894 archive100 permission failure before any copy.

Never changes the actor/source or submits training. Original live RAW is untouched; old follower remains in
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
OLD='b89495c8ec1f2a7fb0d5cd70d89974c16c801cff'
RAW_SOURCE='f979d53cdb0d348e87c5877d84403252e85cd784'
DIRECTORY=ROOT/'followthrough/721485-archive100-recovery-20261016'
INTENT=ROOT/'final-control'/RUN/'archive100-recovery-b894'
ARCHIVE=ROOT/'final-control'/RUN/'archive-100'
PRESERVED=ROOT/'final-control'/RUN/'failed-archive-100-b894'
OLD_FOLLOW=ROOT/'followthrough/721485-archive50-recovery-second-20261015'
PAYLOAD=ROOT/'archives'/RUN/'global_step_100'
CLAIM_SHA='a11da7a6c0ffe5d2223bfa502e4c7bc6c322fd75388b80b7c3dcbfc10bbe3ba0'
PAIR_SHA='0e3cce18c38dc862f2eeb864627956cad0c5996fbeced584cee9be9e6ce652f8'
OLD_CONTROLLER=dict(pid=58027,start_ticks=78951603,sid=58027,uid=14090)
OLD_RUNNER=dict(pid=3233317,start_ticks=80612923,sid=3233317,uid=14090)
OLD_CHILD=dict(pid=3238003,start_ticks=80615197,sid=3233317,uid=14090)
FINISHED=1792096634.8975205
STOPPED=1792096648.5083373
ERROR="ValueError('QPROBE_FOLLOWTHROUGH_SIGMA3_ACTUAL_ARCHIVE_FAILED_100')"
ARCHIVE50_CLAIM_SHA='943239779ce4bf0a85ae52034763ecdbb49854a2f487b379e01f082b469cbd83'
ARCHIVE50_TRANSPORT_SHA='413ef94f42a4a360e632e69225ac2910bd06056d26da1d9673185b6f2088fe76'
DU_SOURCE='a2e4934b7375e8202eae1a1e46f3c326d116c0b0'
STATUS='SIGMA3_ARCHIVE100_B894_PRE_COPY_FAILURE_PRESERVED_ONCE'
MODULE='reproduction.hpc3_qprobe.recover_sigma3_archive100'


def need(ok,message):control.need(ok,'SIGMA3_ARCHIVE100_RECOVERY: '+message)


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


def service():
    old,_=reader.load_fixed(OLD,'recover_sigma3_archive50_second','control')
    return old.service()


def inherited_archive50(producer,old_source):
    """Keep the original actual B894 claim and held0 identity; never retag it."""
    folder=ARCHIVE.parent/'archive-50'
    names=('submit-claim.json','runner-binding.json','task-binding.json','worker-result.json','task-result.json')
    values=[_read(folder,name) for name in names]
    claim=values[0]
    need(control.digest(folder/names[0])==ARCHIVE50_CLAIM_SHA and
         claim['source']==old_source and claim['controller']==OLD_CONTROLLER,
         'original successful50 claim SHA/source/controller changed')
    held=reader.validate_archive_documents(*values,step=50,binding=producer,publication=old_source,
        allowed_controllers=[OLD_CONTROLLER],alive=control.live)
    need(held['ready'] is True,'original successful50 actual held0 runner not terminal')
    transport=ROOT/'archives'/RUN/'global_step_50/complete.json'
    need(control.digest(transport)==ARCHIVE50_TRANSPORT_SHA and
         values[4]['archive']['path']==str(transport) and
         values[4]['archive']['sha256']==ARCHIVE50_TRANSPORT_SHA,
         'original successful50 transport differs')
    return dict(path=str(folder/names[0]),sha256=ARCHIVE50_CLAIM_SHA,source=old_source,
        controller=OLD_CONTROLLER,worker_result_sha256=held['worker_result_sha256'],
        task_result_sha256=held['task_result_sha256'],transport_sha256=ARCHIVE50_TRANSPORT_SHA)


def failure_evidence(archive=None):
    """Only this actual held1 storage failure with all three dead births."""
    service().host_guard()
    archive=ARCHIVE if archive is None else Path(archive)
    need(archive in (ARCHIVE,PRESERVED),'failure evidence path differs')
    _,old_source=reader.publication(OLD)
    run,producer,_=control.producer(BASE)
    pair_path,pair=control.pair(BASE,run,100)
    need(pair['sha256']==PAIR_SHA and run['run_id']==RUN and run['source_commit']==PRODUCER and
         producer['producer_job']==JOB and not run.get('full_state_restore'),'unchanged native4 producer/pair100 differs')
    claim=_read(archive,'submit-claim.json');runner=_read(archive,'runner-binding.json')
    child=_read(archive,'task-binding.json');wait=_read(archive,'worker-result.json')
    need(control.digest(archive/'submit-claim.json')==CLAIM_SHA and claim['source']==old_source and
         claim['producer']==producer and claim['pair']==pair and claim['step']==100 and
         claim['role']=='detached_cpu_pair_archive' and claim['schema_version']==1 and
         claim['controller']==OLD_CONTROLLER and claim['command']==control.command(ROOT/'source'/OLD,
             'archive-worker',ARCHIVE/'submit-claim.json',claim['token']) and
         all(claim[k] is False for k in ('training_permitted','direct_gpu_device_access_permitted',
                                        'native_gpu_audit_submission_permitted')),'original archive100 intent differs')
    need(control.digest(archive/'worker-result.json')=='b6666674f40f37fab3c61a531f758c6c31389b9515d8bca3914e4f8abcb2bab9',
         'original held1 wait receipt bytes differ')
    need(runner['actual_process']==OLD_RUNNER and runner['controller']==OLD_CONTROLLER and
         child['actual_process']==OLD_CHILD and child['supervisor']==OLD_RUNNER and
         wait['actual_task_birth']==OLD_CHILD and wait['actual_supervisor']==OLD_RUNNER and
         type(wait['actual_task_exit_code']) is int and wait['actual_task_exit_code']==1 and
         wait['status']=='CPU_CHILD_FAILED_OR_MISSING_RECEIPT' and wait['role']==claim['role'] and
         wait['finished_unix']==FINISHED and all(v['token']==claim['token'] and v['claim_sha256']==CLAIM_SHA
             for v in (runner,child,wait)),'original held1/source/birth differs')
    need(not any(control.live(v) for v in (OLD_CONTROLLER,OLD_RUNNER,OLD_CHILD)),
         'original follower/runner/child live or unknown; no recovery')
    need(not (archive/'task-result.json').exists() and not (archive/'task-result.json').is_symlink(),
         'original task result exists; not pre-copy failure')
    task_log,task_record=reader.read_file(archive/'task.log',capture=True)
    need(task_record['sha256']=='9dae4a8e309c0a10d33d36679bdfd62988a8073dd02f6abfd7e46e956e32c2ee',
         'not the actual preserved archive100 failure log')
    event=json.loads(task_log.splitlines()[0])
    expected_stderr=''.join("du: cannot read directory '"+str(ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016-r6/tmp'/name)+"': Permission denied\n"
         for name in ('torchelastic_c_fap2h7','torchelastic_edd78rvc'))
    need(event['stage']=='archive_owner_storage_measurement' and event['run_id']==RUN and event['step']==100 and
         event['attempt']==1 and event['actual_returncode']==1 and event['accepted'] is False and
         event['started_unix']==1792096555.2718759 and event['completed_unix']==1792096634.4604366 and
         event['stderr']==expected_stderr and event['stdout']=='1526125326336\t'+str(ROOT.parent)+'\n' and
         b'allocated=measured_owner_storage' in task_log and b'Fresh owner storage measurement failed: rc=1' in task_log,
         'not the observed storage-admission exception before payload creation')
    old,_=reader.load_fixed(OLD,'recover_sigma3_archive50_second','control')
    observed=old.inspect();fclaim=_read(OLD_FOLLOW,'claim.json');result=observed['result'];state=observed['state']
    need(observed['actual_live_birth'] is False and observed['process']['actual_process']==OLD_CONTROLLER and
         result['actual_process']==OLD_CONTROLLER and result['exit_code']==1 and
         result['error']==state['error']==ERROR and result['checked_unix']==STOPPED and
         state['phase']=='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS' and
         state['previous_actual_state']==result['result'] and fclaim['source']==old_source and
         fclaim['binding']==fixed_binding() and fclaim['recovery']==old.consume(OLD) and
         all(result[k] is False for k in ('training_submission_performed','actor_model_loaded',
                                         'direct_gpu_device_access_performed')),'original follower terminal differs')
    previous=result['result'];five=previous['archives']['5']
    need(set(previous['archives'])=={'5','50','100'} and five['actual_child_exit']==0 and
         five['fresh_full_sha_verified'] is True and five['transport_sha256']==
         '39e2edbaf0f92871aacb303011671a002e3ad2530fb43ab01959df134bb87b76' and
         previous['archives']['50']['actual_child_exit']==0 and
         previous['archives']['50']['fresh_full_sha_verified'] is True and
         previous['archives']['50']['transport_sha256']==ARCHIVE50_TRANSPORT_SHA and
         previous['archives']['100']['actual_payload_bytes']==0 and
         previous['archives']['100']['actual_payload_files']==0,'accepted archive5/50 and failed100 prefix differs')
    # Only immutable anchors of the live RAW controller; never snapshot/modify its suffix.
    raw_folder=ROOT/'observations/service-outage-721485-after49-20261015/raw-control'
    anchors={name:dict(path=str(raw_folder/name),sha256=control.digest(raw_folder/name))
             for name in ('claim.json','process.json')}
    return dict(status=STATUS,producer=producer,source=old_source,pair=pair,
        archive_files=snapshot(archive),follower_files=snapshot(OLD_FOLLOW),
        inherited_archive50=inherited_archive50(producer,old_source),
        observer_recovery=old.consume(OLD)['observer_recovery'],
        prior_archive50_second_recovery=old.consume(OLD),raw_immutable_anchors=anchors,
        actual_original_follower=OLD_CONTROLLER,actual_original_runner=OLD_RUNNER,actual_original_child=OLD_CHILD,
        original_held_exit_code=1,original_failure_cause='actual DU Permission denied; zero-copy admission failure',
        original_finished_unix=FINISHED,original_follower_stopped_unix=STOPPED)


def no_payload():
    need(not PAYLOAD.exists() and not PAYLOAD.is_symlink(),'any original or partial archive payload forbids recovery')


def successful_du(path,*,now=None):
    """Consume the exact explicitly selected generic-observer slot, never slot1 by accident."""
    path=Path(path)
    base=ROOT/'observations/storage-sigma3-after-actual-denial-repair-20261016'
    need(path.is_absolute() and path.resolve()==path and path.name=='result.json' and
         path.parent.parent==base and re.fullmatch(r'attempt-(?:[1-9]|1[0-6])',path.parent.name),
         'DU receipt is not an explicit authorized slot path')
    slot=int(path.parent.name.split('-')[1])
    value,record=reader.document(path);now=time.time() if now is None else now
    filename='storage_observer_sigma3_after_actual_denial_repair'
    source=value.get('source') or {}
    need(source.get('commit')==DU_SOURCE and Path(source.get('path',''))==ROOT/'source'/DU_SOURCE/'reproduction/hpc3_qprobe'/(filename+'.py'),
         'only exact published generic observer source accepted')
    observer,_=reader.load_fixed(DU_SOURCE,filename,'_unused_control')
    observer.SLOT=slot;observer.CONTROL=path.parent
    service().host_guard();actual=observer.inspect()
    need(actual.get('control')==str(path.parent) and actual['result']==value and
         actual['status']=='ACTUAL_DU_COMPLETED0_USAGE_MEASURED' and
         actual['actual_live_birth'] is False and actual['actual_du_live_birth'] is False and
         type(value['actual_du_exit_code']) is int and value['actual_du_exit_code']==0 and value['stderr']=='' and
         value['completed_unix']>FINISHED and value['completed_unix']<=now and now-value['completed_unix']<=600 and
         type(value['allocated_bytes']) is int and 0<=value['allocated_bytes']<2*10**12,
         'fresh actual held0 full owner DU is required')
    return record


def consume(commit):
    """Read-only original SHA + actual moved files + exclusive migration proof."""
    repository,source=control.source(commit)
    need(Path(__file__).resolve()==repository/'reproduction/hpc3_qprobe/recover_sigma3_archive100.py',
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
        observer_recovery=intent['failure']['observer_recovery'],
        prior_archive50_second_recovery=intent['failure']['prior_archive50_second_recovery'],
        inherited_archive50=intent['failure']['inherited_archive50'],
        legacy_archive5_recovery=intent['failure']['observer_recovery']['legacy_archive5_recovery'],
        training_source=PRODUCER,producer_job=JOB,run_id=RUN)


def core():
    """Private module instance: no mutation of the old imported follower."""
    path=Path(__file__).with_name('followthrough_sigma3.py')
    spec=importlib.util.spec_from_file_location('_sigma3_archive100_recovered_followthrough',path)
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
    need(Path(__file__).resolve()==repository/'reproduction/hpc3_qprobe/recover_sigma3_archive100.py',
         'loaded recovery module differs from its immutable source')
    need(commit!=OLD and re.fullmatch('[0-9a-f]{40}',commit),'new explicit immutable recovery source required')
    failure=failure_evidence();no_payload();du=successful_du(du_receipt)
    raw,_=reader.load_fixed(RAW_SOURCE,'recover_sigma3_raw49','ctl');raw_status=raw.inspect(RAW_SOURCE)
    need(raw_status['status'] in ('ACTUAL_READONLY_AUDIT_LIVE',raw.STATUS),'original RAW not healthy; no restart')
    for name in ('archive-150','archive-200','audit-controller','final-audit'):
        need(not (ARCHIVE.parent/name).exists(),'unexpected later component; no replay')
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
    return core().supervise(commit,initial_verified_archives={5:'39e2edbaf0f92871aacb303011671a002e3ad2530fb43ab01959df134bb87b76',50:ARCHIVE50_TRANSPORT_SHA})


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
