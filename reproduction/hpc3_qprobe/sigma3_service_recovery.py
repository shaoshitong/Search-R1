"""Typed once handoff for the observed Login1 accounting outage after RAW49.

Original CPU controllers, their claims/results, and archive5 stay in place.
No training or archive claim is replayed. This helper never loads an actor.
"""
import hashlib
import functools
import json
import os
from pathlib import Path
import socket
import stat
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader

ROOT=ctl.ROOT
BASE=reader.BASE
RUN=reader.RUN
JOB=reader.JOB
RAW_OLD=reader.RAW
FOLLOW_OLD='09ccff26c5b764cc4c6a285c1e8f57a8e9b07726'
RAW_ORIGINAL=reader.RAW_DIRECTORY
FOLLOW_ORIGINAL=ROOT/'followthrough/721485-archive5-recovery-20261015'
DIRECTORY=ROOT/'observations/service-outage-721485-after49-20261015'
RAW_DIRECTORY=DIRECTORY/'raw-control'
FOLLOW_DIRECTORY=DIRECTORY/'followthrough-control'
RAW_BIRTH=dict(pid=1171759,start_ticks=75909729,sid=1171759,uid=14090)
FOLLOW_BIRTH=dict(pid=2688346,start_ticks=76403669,sid=2688346,uid=14090)
HOST=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]')
STATUS='SIGMA3_ACTUAL_ACCOUNTING_OUTAGE_RAW49_FOLLOW5_HANDOFF'
RAW_ERROR="Command '['sacct', '-nP', '-X', '-j', '721449', '--format=JobID,State,ExitCode,UID,NodeList']' returned non-zero exit status 1."
FOLLOW_ERROR="CalledProcessError(1, ['sacct', '-nP', '-X', '-j', '721485', '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition'])"


def host_guard():
    actual=dict(hostname=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid_namespace=os.readlink('/proc/self/ns/pid'))
    ctl.need(actual==HOST,'UNKNOWN_HOST_IDENTITY_MISMATCH; death cannot be inferred on another host')
    return actual


def snapshot(folder):
    """Only bounded metadata in these already terminal controller namespaces."""
    folder=Path(folder);ctl.need(folder.resolve()==folder and not folder.is_symlink(),'unsafe service evidence root')
    result={};total=0
    for p in sorted(folder.rglob('*')):
        s=p.lstat();ctl.need(p.resolve()==p and not p.is_symlink() and s.st_uid==ctl.OWNER,'unsafe service evidence member')
        if stat.S_ISDIR(s.st_mode):continue
        ctl.need(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=32*2**20,'unbounded service evidence member')
        total+=s.st_size;ctl.need(total<=128*2**20,'service evidence exceeds metadata budget')
        result[str(p.relative_to(folder))]=dict(bytes=s.st_size,sha256=ctl.digest(p),inode=s.st_ino,
            device=s.st_dev,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode))
    ctl.need(result,'empty original controller evidence')
    return result


@functools.lru_cache(maxsize=1)
def old_raw():
    return reader.load_fixed(RAW_OLD,'raw_audit_followthrough_sigma3','ctl')


def failure_evidence():
    host_guard();raw,raw_source=old_raw();observed=raw.inspect(RAW_OLD)
    ctl.need(observed['status']=='ACTUAL_READONLY_AUDIT_STOPPED_EVIDENCE_RETAINED' and
        observed['actual_live_birth'] is False and observed['actual_child_live_birth'] is False and
        observed['process']==RAW_BIRTH and observed['accepted_attempt_receipts']==49,
        'original RAW49 is not this terminal dead supervisor; never replay')
    terminal=observed['result'];state=observed['progress'];claim=ctl.read(RAW_ORIGINAL/'claim.json')
    ctl.need(terminal['error_type']=='CalledProcessError' and terminal['error']==RAW_ERROR and
        terminal['completed_unix']==1792075504.6165059 and terminal['completed_attempt_audits']==49 and
        terminal['source']==claim['source']==raw_source and state['successful_outers']==49,
        'not the observed post49 accounting-query failure')
    accepted=sorted((RAW_ORIGINAL/'accepted').glob('*.json'))
    ctl.need([p.name for p in accepted]==['%06d.json'%i for i in range(1,50)],'original accepted49 namespace differs')
    tasks=raw.actual_tasks(claim['parts']);records=[]
    for i,p in enumerate(accepted):
        row=ctl.read(p);task=row['task']
        ctl.need(task==tasks[i] and task['step']==i+1 and task['complete_outer'],'original49 task prefix differs')
        measured=raw.consume_task(task,RAW_ORIGINAL/'attempts'/task['attempt_id'],RAW_BIRTH)
        ctl.need(measured==row,'original held0 receipt/output bytes changed')
        records.append(row)
    ctl.need({p.name for p in (RAW_ORIGINAL/'attempts').iterdir()}=={r['task']['attempt_id'] for r in records},
        'original unaccepted/in-flight attempt exists; no recovery')
    process=ctl.read(FOLLOW_ORIGINAL/'process.json');result=ctl.read(FOLLOW_ORIGINAL/'result.json')
    follow_state=ctl.read(FOLLOW_ORIGINAL/'state.json');follow_claim=ctl.read(FOLLOW_ORIGINAL/'claim.json')
    ctl.need(process['actual_process']==result['actual_process']==FOLLOW_BIRTH and not ctl.live(FOLLOW_BIRTH) and
        process['commit']==follow_claim['source']['commit']==FOLLOW_OLD and result['exit_code']==1 and
        result['error']==follow_state['error']==FOLLOW_ERROR and result['checked_unix']==1792075543.145685 and
        follow_state['phase']=='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS' and
        follow_state['previous_actual_state']==result['result'] and
        all(result[k] is False for k in ('training_submission_performed','actor_model_loaded','direct_gpu_device_access_performed')),
        'original follower is not this actual dead accounting failure')
    archives=result['result']['archives'];five=archives.get('5',{})
    ctl.need(set(archives)=={'5'} and five.get('actual_child_exit')==0 and five.get('fresh_full_sha_verified') is True and
        five.get('transport_sha256')=='39e2edbaf0f92871aacb303011671a002e3ad2530fb43ab01959df134bb87b76',
        'original archive5 accepted prefix differs')
    components=ROOT/'final-control'/RUN
    for name in ['archive-'+str(s) for s in (50,100,150,200)]+['audit-controller','final-audit']:
        ctl.need(not (components/name).exists() and not (components/name).is_symlink(),
            'later original component exists; this specific handoff cannot replay it')
    archive=components/'archive-5';archive_claim=ctl.read(archive/'submit-claim.json')
    ctl.need(archive_claim['controller']==FOLLOW_BIRTH and archive_claim['source']['commit']==FOLLOW_OLD and
        ctl.digest(five['transport_path'])==five['transport_sha256'],'archive5 original creator/transport differs')
    # The old archive5 handoff stays independently frozen, including failed9c.
    preserved={str(p):snapshot(p) for p in (
        components/'archive5-recovery-first-9c',components/'failed-archive-5-first-9c',ROOT/'followthrough/721485')}
    return dict(status=STATUS,host=HOST,raw_source=raw_source,raw_process=RAW_BIRTH,
        raw_prefix_records=records,raw_prefix_sha256=raw.canonical_sha(records),parts=claim['parts'],
        raw_files=snapshot(RAW_ORIGINAL),follow_files=snapshot(FOLLOW_ORIGINAL),archive5_files=snapshot(archive),
        legacy_preserved_files=preserved,follow_source=follow_claim['source'],follow_process=FOLLOW_BIRTH,
        follow_binding=follow_claim['binding'],verified_archives={'5':five['transport_sha256']},
        existing_archive_claims={'5':dict(path=str(archive/'submit-claim.json'),sha256=ctl.digest(archive/'submit-claim.json'),
            source=archive_claim['source'],controller=archive_claim['controller'])})


def stable_failure(failure):
    """Recheck immutable predecessor bytes; later NEW components are allowed."""
    host_guard();ctl.need(not ctl.live(RAW_BIRTH) and not ctl.live(FOLLOW_BIRTH),'predecessor birth is live/unknown')
    for folder,key in ((RAW_ORIGINAL,'raw_files'),(FOLLOW_ORIGINAL,'follow_files'),
                       (ROOT/'final-control'/RUN/'archive-5','archive5_files')):
        ctl.need(snapshot(folder)==failure[key],'original terminal evidence changed: '+str(folder))
    for folder,expected in failure['legacy_preserved_files'].items():
        ctl.need(snapshot(folder)==expected,'original archive5 recovery evidence changed')
    raw,_=old_raw();tasks=raw.actual_tasks(failure['parts'])
    for i,record in enumerate(failure['raw_prefix_records']):
        task=record['task'];ctl.need(task==tasks[i],'original49 ledger prefix changed')
        ctl.need(raw.consume_task(task,RAW_ORIGINAL/'attempts'/task['attempt_id'],RAW_BIRTH)==record,
            'original49 wait/output changed')
    ctl.need(raw.canonical_sha(failure['raw_prefix_records'])==failure['raw_prefix_sha256'],
        'original49 prefix fingerprint differs')


def consume(commit, *, verify_legacy=False):
    host_guard();_,source=ctl.source(commit)
    intent=ctl.read(DIRECTORY/'intent.json');handoff=ctl.read(DIRECTORY/'handoff.json')
    ctl.need(intent['source']==handoff['source']==source and intent['status']==handoff['status']==STATUS and
        handoff['intent_sha256']==ctl.digest(DIRECTORY/'intent.json') and
        intent['training_submission_permitted'] is False and intent['failure']['host']==HOST,
        'service recovery exclusive intent/source differs')
    stable_failure(intent['failure'])
    if verify_legacy:
        legacy,_=reader.load_fixed(FOLLOW_OLD,'recover_sigma3_archive5','control')
        ctl.need(legacy.consume(FOLLOW_OLD)==intent['legacy_archive5_recovery'],'original archive5 handoff changed')
    return dict(status=STATUS,source=source,intent=dict(path=str(DIRECTORY/'intent.json'),sha256=ctl.digest(DIRECTORY/'intent.json')),
        handoff=dict(path=str(DIRECTORY/'handoff.json'),sha256=ctl.digest(DIRECTORY/'handoff.json')),
        failure_fingerprint=hashlib.sha256(json.dumps(intent['failure'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_raw_source=RAW_OLD,original_raw_process=RAW_BIRTH,original_raw_accepted=49,
        original_follow_source=FOLLOW_OLD,original_follow_process=FOLLOW_BIRTH,
        legacy_archive5_recovery=intent['legacy_archive5_recovery'],training_performed=False)


def prepare(commit):
    ctl.owner();ctl.caps();host_guard();repository,source=ctl.source(commit)
    if DIRECTORY.exists():
        return consume(commit)  # Partial/unknown intent refuses, never relaunches.
    run,binding,_=ctl.producer(BASE)
    ctl.need(run['run_id']==RUN and binding['producer_job']==JOB and binding['producer_commit']==reader.PRODUCER,
        'current native4 producer binding differs')
    publication=ctl.read(source['publication_path'])
    for name,sha in run['source_sha256'].items():
        if name.startswith(('verl/','search_g1/','search_r1/','reproduction/common8/')) and not Path(name).name.startswith('test_'):
            ctl.need(publication['files'].get(name)==sha,'recovery scientific source differs: '+name)
    failure=failure_evidence()
    ctl.need(dict(binding,sigma3_settings_sha256=ctl.digest(BASE/'preparation/sigma3-experiment.json'))==failure['follow_binding'],
        'current producer differs from original controller binding')
    legacy,_=reader.load_fixed(FOLLOW_OLD,'recover_sigma3_archive5','control')
    legacy_proof=legacy.consume(FOLLOW_OLD)
    DIRECTORY.mkdir(mode=0o770)
    ctl.publish(DIRECTORY/'intent.json',dict(status=STATUS,source=source,failure=failure,
        legacy_archive5_recovery=legacy_proof,created_unix=time.time(),training_submission_permitted=False))
    ctl.publish(DIRECTORY/'handoff.json',dict(status=STATUS,source=source,
        intent_sha256=ctl.digest(DIRECTORY/'intent.json'),created_unix=time.time()))
    return consume(commit)


def inherited():
    return ctl.read(DIRECTORY/'intent.json')['failure']
