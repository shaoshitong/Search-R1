"""Once typed handoff for the actual 5f pre-copy archive150 storage failure.

Preserves all original failure bytes and the stopped follower in place. Only
the failed canonical150 control is renamed once; new150/200 copies use full
two-owner storage coverage without chmod. The existing native4 producer and
RAW supervisor are never changed, restarted or replaced.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as control
from reproduction.hpc3_qprobe import followthrough_sigma3 as original
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader
from reproduction.hpc3_qprobe import recovery_lineage_stage as lineage
from reproduction.hpc3_qprobe.recover_sigma3_archive100_fourth import snapshot,fsync_dir
from reproduction.hpc3_qprobe import owner_partition_storage as storage

ROOT=control.ROOT
BASE=original.BASE
RUN=original.RUN
JOB=original.JOB
PRODUCER=original.PRODUCER
OLD='5f0065b98d1239c994addc19fd0b7e32f506d7fd'
RAW_SOURCE='f979d53cdb0d348e87c5877d84403252e85cd784'
DIRECTORY=ROOT/'followthrough/721485-archive150-recovery-first-20261016'
INTENT=ROOT/'final-control'/RUN/'archive150-recovery-first-5f'
ARCHIVE=ROOT/'final-control'/RUN/'archive-150'
PRESERVED=ROOT/'final-control'/RUN/'failed-archive-150-first-5f'
OLD_FOLLOW=ROOT/'followthrough/721485-archive100-recovery-fourth-20261016'
PAYLOAD=ROOT/'archives'/RUN/'global_step_150'
CLAIM_SHA='6dca59f3f96c069dfd24e624b3ee91371d9f9532a7177cd7d4c374879bc25917'
PAIR_SHA='b77dee6a76d980a1d55da57e88f8d54bceeb7bfbe3c88ed28ef54818181c625d'
OLD_CONTROLLER=dict(pid=560250,start_ticks=81264116,sid=560250,uid=14090)
OLD_RUNNER=dict(pid=3556013,start_ticks=82587058,sid=3556013,uid=14090)
OLD_CHILD=dict(pid=3560715,start_ticks=82589365,sid=3556013,uid=14090)
FINISHED=1792116692.8873627
STOPPED=1792116701.0708904
ERROR="ValueError('QPROBE_FOLLOWTHROUGH_SIGMA3_ACTUAL_ARCHIVE_FAILED_150')"
ARCHIVE100_CLAIM_SHA='06e7a3a265ef3a6184ba6f71eabab69ab12b40a29701137dccfc1287395a6d39'
ARCHIVE100_TRANSPORT_SHA='5f113a12cd245201e7a842d9d59035909e3ed6f92c7793b04c19088ee49b528e'
TRANSPORTS={5:'39e2edbaf0f92871aacb303011671a002e3ad2530fb43ab01959df134bb87b76',
    50:'413ef94f42a4a360e632e69225ac2910bd06056d26da1d9673185b6f2088fe76',100:ARCHIVE100_TRANSPORT_SHA}
STATUS='SIGMA3_ARCHIVE150_FIRST_5F_PRE_COPY_FAILURE_PRESERVED_ONCE'
MODULE='reproduction.hpc3_qprobe.recover_sigma3_archive150_first'
RELATIVE='reproduction/hpc3_qprobe/recover_sigma3_archive150_first.py'
FIXTURE=Path(__file__).with_name('archive150_first_failure_fixture.json')


def need(ok,message):control.need(ok,'SIGMA3_ARCHIVE150_RECOVERY: '+message)


def read(folder,name):return control.read(Path(folder)/name)


def pinned_snapshot(folder,original_folder):
    """All 24 real failure files, including full stderr and failed DU bytes."""
    expected=json.loads(FIXTURE.read_bytes())[str(original_folder)]
    actual=snapshot(folder)
    need(set(actual)==set(expected),'original failure complete fileset differs')
    for name,row in actual.items():
        need(all(row[key]==expected[name][key] for key in ('inode','bytes','sha256')),
             'original failure bytes/inode differ: '+name)
        raw=base64.b64decode(expected[name]['raw_base64'],validate=True)
        need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],
             'published actual-failure fixture differs')
    return actual


def inherited_archive100(producer):
    """Bind original 5f/controller560250100; never retag it as the new source."""
    folder=ARCHIVE.parent/'archive-100';_,source=reader.publication(OLD)
    names=('submit-claim.json','runner-binding.json','task-binding.json','worker-result.json','task-result.json')
    values=[read(folder,name) for name in names];claim=values[0]
    need(control.digest(folder/names[0])==ARCHIVE100_CLAIM_SHA and claim['source']==source and
         claim['controller']==OLD_CONTROLLER,'original successful100 claim/source/controller differs')
    held=reader.validate_archive_documents(*values,step=100,binding=producer,publication=source,
        allowed_controllers=[OLD_CONTROLLER],alive=control.live)
    need(held['ready'] is True,'original100 actual held0 missing')
    transport=ROOT/'archives'/RUN/'global_step_100/complete.json'
    need(control.digest(transport)==ARCHIVE100_TRANSPORT_SHA and values[4]['archive']['path']==str(transport) and
         values[4]['archive']['sha256']==ARCHIVE100_TRANSPORT_SHA and
         held['worker_result_sha256']=='1446b5142990c22b49fee1bf26e535e75082afb0b077a31bad4cf5df7f03fb02' and
         held['task_result_sha256']=='a479ce9c89fe68715d8c30c9a9cf01eeb2bd8f171caa99c8df4b676a3936a762',
         'original successful100 held/transport bytes differ')
    return dict(path=str(folder/names[0]),sha256=ARCHIVE100_CLAIM_SHA,source=source,controller=OLD_CONTROLLER,
        worker_result_sha256=held['worker_result_sha256'],task_result_sha256=held['task_result_sha256'],
        transport_sha256=ARCHIVE100_TRANSPORT_SHA)


def failure_evidence(archive=None):
    archive=ARCHIVE if archive is None else Path(archive)
    need(archive in (ARCHIVE,PRESERVED),'failure scope differs')
    storage.host_guard()
    prior=lineage.consume(OLD,'recover_sigma3_archive100_fourth')
    _,source=reader.publication(OLD);run,producer,_=control.producer(BASE)
    _,pair=control.pair(BASE,run,150)
    need(pair['sha256']==PAIR_SHA and run['run_id']==RUN and run['source_commit']==PRODUCER and
         producer['producer_job']==JOB and not run.get('full_state_restore'),'native4 producer/pair150 changed')
    archive_files=pinned_snapshot(archive,ARCHIVE);follower_files=pinned_snapshot(OLD_FOLLOW,OLD_FOLLOW)
    du_folder=storage.INITIAL_FAILURE.parent;du_files=pinned_snapshot(du_folder,du_folder)
    claim=read(archive,'submit-claim.json');runner=read(archive,'runner-binding.json')
    child=read(archive,'task-binding.json');wait=read(archive,'worker-result.json')
    need(control.digest(archive/'submit-claim.json')==CLAIM_SHA and claim['source']==source and
         claim['producer']==producer and claim['pair']==pair and claim['step']==150 and
         claim['role']=='detached_cpu_pair_archive' and claim['schema_version']==1 and
         claim['controller']==OLD_CONTROLLER and claim['storage_wait_protocol']=='actual-full-du-repair-v1' and
         claim['command']==control.command(ROOT/'source'/OLD,'archive-worker',ARCHIVE/'submit-claim.json',claim['token']) and
         all(claim[k] is False for k in ('training_permitted','direct_gpu_device_access_permitted',
                                        'native_gpu_audit_submission_permitted')),'original failed150 claim differs')
    need(runner['actual_process']==OLD_RUNNER and runner['controller']==OLD_CONTROLLER and
         child['actual_process']==OLD_CHILD and child['supervisor']==OLD_RUNNER and
         wait['actual_task_birth']==OLD_CHILD and wait['actual_supervisor']==OLD_RUNNER and
         type(wait['actual_task_exit_code']) is int and wait['actual_task_exit_code']==1 and
         wait['status']=='CPU_CHILD_FAILED_OR_MISSING_RECEIPT' and wait['role']==claim['role'] and
         wait['finished_unix']==FINISHED and all(v['token']==claim['token'] and v['claim_sha256']==CLAIM_SHA
             for v in (runner,child,wait)),'original150 held1/birth differs')
    need(not any(control.live(v) for v in (OLD_CONTROLLER,OLD_RUNNER,OLD_CHILD)),'old follower/runner/child still live')
    need(not (archive/'task-result.json').exists() and not (archive/'task-result.json').is_symlink(),
         'original150 result exists; not a pre-copy failure')
    roots,initial=storage.initial_frontier()
    exception=read(archive,'storage-measurement-1-exception.json')
    held=read(archive,'storage-measurement-1-held.json')
    need(exception['cleanup_pending'] is False and held['actual_observer_wait_exit_code']==0 and
         'reported directory outside authorized temporary scope' in exception['error'] and
         any('/artifacts/behavior-traces' in str(path) for path in roots),
         'not the actual private-root storage refusal with reaped observer')
    result=read(OLD_FOLLOW,'result.json');state=read(OLD_FOLLOW,'state.json');fclaim=read(OLD_FOLLOW,'claim.json')
    need(read(OLD_FOLLOW,'process.json')['actual_process']==OLD_CONTROLLER and result['actual_process']==OLD_CONTROLLER and
         result['exit_code']==1 and result['error']==state['error']==ERROR and result['checked_unix']==STOPPED and
         state['phase']=='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS' and
         state['previous_actual_state']==result['result'] and fclaim['source']==source and
         fclaim['recovery']==prior and fclaim['binding']==fixed_binding() and
         all(result[k] is False for k in ('training_submission_performed','actor_model_loaded',
                                         'direct_gpu_device_access_performed')),'original fourth follower terminal differs')
    previous=result['result']['archives']
    need(set(previous)=={'5','50','100','150'} and all(previous[str(step)]['actual_child_exit']==0 and
         previous[str(step)]['fresh_full_sha_verified'] is True and
         previous[str(step)]['transport_sha256']==sha for step,sha in TRANSPORTS.items()) and
         previous['150']['actual_payload_bytes']==previous['150']['actual_payload_files']==0,
         'successful archive5/50/100 and failed150 prefix differs')
    raw=ROOT/'observations/service-outage-721485-after49-20261015/raw-control'
    anchors={name:dict(path=str(raw/name),sha256=control.digest(raw/name)) for name in ('claim.json','process.json')}
    return dict(status=STATUS,producer=producer,source=source,pair=pair,archive_files=archive_files,
        follower_files=follower_files,du_files=du_files,initial_failed_du=initial,raw_immutable_anchors=anchors,
        prior_archive100_fourth_recovery=prior,inherited_archive100=inherited_archive100(producer),
        actual_original_follower=OLD_CONTROLLER,actual_original_runner=OLD_RUNNER,actual_original_child=OLD_CHILD,
        original_held_exit_code=1,original_finished_unix=FINISHED,original_follower_stopped_unix=STOPPED,
        original_failure_cause='actual complete DU permission refusal; private trace roots; zero payload')


def no_payload():
    need(not PAYLOAD.exists() and not PAYLOAD.is_symlink(),'any original or partial150 payload forbids recovery')


def consume(commit):
    repo,source=control.source(commit)
    need(Path(__file__).resolve()==repo/RELATIVE,'loaded150 recovery source differs')
    intent=read(INTENT,'intent.json');migration=read(INTENT,'migration.json')
    need(intent['source']==source and intent['failure']['status']==STATUS and intent['original_archive']==str(ARCHIVE) and
         intent['preserved_archive']==str(PRESERVED) and intent['new_follower']==str(DIRECTORY) and
         intent['training_submission_permitted'] is False and migration['intent_sha256']==control.digest(INTENT/'intent.json') and
         migration['status']==STATUS and migration['before_files']==migration['after_files']==intent['failure']['archive_files'],
         '150 handoff intent/migration incomplete or different')
    need(failure_evidence(PRESERVED)==intent['failure'],'preserved original150 evidence changed')
    prior=intent['failure']['prior_archive100_fourth_recovery']
    result=dict(status=STATUS,source=source,intent=dict(path=str(INTENT/'intent.json'),sha256=control.digest(INTENT/'intent.json')),
        migration=dict(path=str(INTENT/'migration.json'),sha256=control.digest(INTENT/'migration.json')),
        failure_fingerprint=hashlib.sha256(json.dumps(intent['failure'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_follower=OLD_CONTROLLER,original_archive_child_exit=1,original_failure_preserved=True,
        prior_archive100_fourth_recovery=prior,inherited_archive100=intent['failure']['inherited_archive100'],
        training_source=PRODUCER,producer_job=JOB,run_id=RUN,storage_protocol=storage.PROTOCOL)
    for key in ('observer_recovery','prior_archive50_second_recovery','inherited_archive50','prior_archive100_recovery',
                'prior_archive100_second_recovery','prior_archive100_third_recovery','legacy_archive5_recovery'):
        result[key]=prior[key]
    return result


def core():
    spec=importlib.util.spec_from_file_location('_sigma3_archive150_first_followthrough',Path(__file__).with_name('followthrough_sigma3.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.DIRECTORY=DIRECTORY
    class ArchiveControl:
        def __getattr__(self,name):return getattr(control,name)
        def launch_archive(self,*args,**kwargs):
            need(len(args)>=3 and args[2] in (150,200),'new follower may only claim150/200')
            return control.launch_archive(*args,owner_partition_protocol=True,**kwargs)
    module.control=ArchiveControl();return module


def start(commit):
    control.owner();control.caps()
    need(all(not p.exists() and not p.is_symlink() for p in (INTENT,DIRECTORY,PRESERVED)),
         'existing150 intent/follower/preservation; inspect only, never replay')
    repo,source=control.source(commit)
    need(Path(__file__).resolve()==repo/RELATIVE and commit!=OLD and re.fullmatch('[0-9a-f]{40}',commit),
         'new explicit immutable150 recovery source required')
    failure=failure_evidence();no_payload()
    raw,_=reader.load_fixed(RAW_SOURCE,'recover_sigma3_raw49','ctl');observed=raw.inspect(RAW_SOURCE)
    need(observed['status'] in ('ACTUAL_READONLY_AUDIT_LIVE',raw.STATUS),'original RAW not healthy; no restart')
    for name in ('archive-200','audit-controller','final-audit'):
        need(not (ARCHIVE.parent/name).exists() and not (ARCHIVE.parent/name).is_symlink(),'later component exists; no replay')
    run,_,accounting=control.producer(BASE);rows=[line.split('|') for line in accounting.splitlines() if line]
    need(len(rows)==1 and rows[0][0]==JOB and rows[0][1] in ('RUNNING','COMPLETING','COMPLETED') and
         (rows[0][1]!='COMPLETED' or rows[0][2]=='0:0'),'actual producer is not healthy')
    publication=control.read(Path(source['publication_path']))
    for name,sha in run['source_sha256'].items():
        if name.startswith(('verl/','search_g1/','search_r1/','reproduction/common8/')) and not Path(name).name.startswith('test_'):
            need(publication['files'].get(name)==sha,'science source changed: '+name)
    INTENT.mkdir(mode=0o770)
    control.publish(INTENT/'intent.json',dict(source=source,failure=failure,original_archive=str(ARCHIVE),
        preserved_archive=str(PRESERVED),new_follower=str(DIRECTORY),created_unix=time.time(),training_submission_permitted=False))
    need(failure_evidence()==failure,'failure changed before atomic preservation');no_payload()
    need(not PRESERVED.exists() and not PRESERVED.is_symlink(),'preservation appeared; unknown no replay')
    os.rename(ARCHIVE,PRESERVED);fsync_dir(ARCHIVE.parent)
    after=snapshot(PRESERVED)
    need(after==failure['archive_files'] and failure_evidence(PRESERVED)==failure,'atomic preservation changed original150 bytes')
    control.publish(INTENT/'migration.json',dict(status=STATUS,intent_sha256=control.digest(INTENT/'intent.json'),
        before_files=failure['archive_files'],after_files=after,completed_unix=time.time()))
    reference=consume(commit);DIRECTORY.mkdir(mode=0o770)
    command=[str(BASE/'env/bin/python'),'-m',MODULE,'supervise','--commit',commit]
    control.publish(DIRECTORY/'claim.json',dict(schema_version=1,binding=fixed_binding(),source=source,recovery=reference,
        command=command,cwd=str(repo),created_unix=time.time(),training_submission_permitted=False,
        actor_model_reload_permitted=False,direct_gpu_access_permitted=False,native_final_restore_only_submission_permitted=True))
    env=dict(os.environ,QPROBE_SOURCE=str(repo),PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1',
             **{key:'1' for key in control.CAPS})
    with (DIRECTORY/'output.log').open('x') as output:
        child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    actual=control.birth(control.process(child.pid))
    need(actual['uid']==control.OWNER and actual['sid']==actual['pid'],'new detached150 follower birth differs')
    result=dict(actual_process=actual,commit=commit,control=str(DIRECTORY),checked_unix=time.time())
    control.publish(DIRECTORY/'process.json',result);return result


def supervise(commit):
    control.owner();control.caps();deadline=time.monotonic()+15
    while not (DIRECTORY/'process.json').exists():
        need(time.monotonic()<deadline,'new150 parent birth missing; unknown no replay');time.sleep(.05)
    need(read(DIRECTORY,'claim.json')['recovery']==consume(commit),'new150 follower typed handoff changed')
    private=core();results={}
    # Re-read every source/archive/target byte once before inheriting the old
    # verified prefix. These scans are detached, with explicit progress states.
    for step,sha in sorted(TRANSPORTS.items()):
        private.publish_state(dict(phase='ACTUAL_INHERITED_ARCHIVE_FULL_SHA_READBACK',step=step,
            prior_completed=list(results),training_actions=0,permission_changes=0,full_objective_accepted=False))
        actual=control.inspect_archive(BASE,step,full=True)
        need(actual['status']=='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0' and actual['transport_sha256']==sha and
             actual.get('fresh_full_sha') is not None,'inherited archive full three-way SHA missing')
        results[step]=actual['fresh_full_sha']
    control.publish(DIRECTORY/'inherited-archives-full-sha.json',dict(source_commit=commit,run_id=RUN,
        transports=TRANSPORTS,readbacks=results,checked_unix=time.time(),training_actions=0,permission_changes=0))
    return private.supervise(commit,initial_verified_archives=TRANSPORTS)


def fixed_binding():return core().fixed_binding()


def inspect():
    claim=read(DIRECTORY,'claim.json')
    need(claim['recovery']==consume(claim['source']['commit']),'new150 follower handoff changed')
    return core().inspect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('start','supervise','inspect','consume','preflight'))
    p.add_argument('--commit',required=True);a=p.parse_args();control.owner();control.caps()
    repo,_=control.source(a.commit);need(Path(__file__).resolve()==repo/RELATIVE,'loaded150 CLI source differs')
    if a.mode=='supervise':sys.exit(supervise(a.commit))
    value=start(a.commit) if a.mode=='start' else consume(a.commit) if a.mode=='consume' else failure_evidence() if a.mode=='preflight' else inspect()
    print(json.dumps(value,allow_nan=False),flush=True)
