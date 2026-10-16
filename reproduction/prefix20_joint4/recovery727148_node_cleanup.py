"""Before-CUDA cleanup of the exact failed727148 node staging namespace.

No process is signalled: Slurm must already have reaped every recorded reader.
Only the original exact-inode tmpfs asset copy is removed; originals stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from reproduction.prefix20_joint4.control import publish,validate_slurm_job
from reproduction.a800.owned_processes import identity
from reproduction.common8.native_allocation import fields

BASE=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016')
JOB='727148'
RUN='ebcff3f7fff6499e9f121153dc29ebab'
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016')


def execute(recovery_base,commit):
    if os.getuid()!=14090 or socket.gethostname().split('.')[0]!='ACD1-25':
        raise ValueError('PREFIX20_CLEANUP727148_ACTUAL_OWN_NODE_REQUIRED')
    target=Path(recovery_base)
    if target!=TARGET or Path(__file__).resolve()!=target/'repo/reproduction/prefix20_joint4/recovery727148_node_cleanup.py':
        raise ValueError('PREFIX20_CLEANUP727148_BOUND_FRESH_RUNTIME_REQUIRED')
    permit=json.loads((target/'preparation/failed727148-deferred-node-cleanup.json').read_text())
    if permit!=dict(failed_base=str(BASE),failed_job=JOB,failed_run=RUN,node='ACD1-25',recovery_base=str(TARGET),source_commit=commit,before_any_cuda=True):
        raise ValueError('PREFIX20_CLEANUP727148_RECOVERY_PERMIT_DIFFERS')
    current_job=os.environ.get('SLURM_JOB_ID','')
    if not current_job.isdecimal() or current_job==JOB:raise ValueError('PREFIX20_CLEANUP727148_FRESH_JOB_REQUIRED')
    result=subprocess.run(['scontrol','show','job','-o',current_job],capture_output=True,text=True,timeout=30)
    actual=fields(result.stdout)
    submitted=json.loads((target/'preparation/training-submission.json').read_text())
    if result.returncode or actual.get('NodeList')!='ACD1-25' or submitted['job']!=current_job:
        raise ValueError('PREFIX20_CLEANUP727148_NOT_BOUND_FRESH_NODE')
    validate_slurm_job(target,commit,submitted,actual)
    terminal=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Start,End,NodeList'],capture_output=True,text=True,timeout=30)
    expected='727148|FAILED|1:0|jxu957|14090|2026-10-16T22:16:44|2026-10-16T22:26:18|ACD1-25'
    if terminal.returncode or expected not in terminal.stdout.splitlines() or not any(
            line.startswith(JOB+'.extern|COMPLETED|0:0|') for line in terminal.stdout.splitlines()):
        raise ValueError('PREFIX20_CLEANUP727148_ORIGINAL_TERMINAL_DIFFERS')
    ledger_file=BASE/'logs/process-ledger.json';ledger=json.loads(ledger_file.read_text())
    sealed=json.loads((target/'preparation/failed727148-cleanup-evidence-sha.json').read_text())
    if (sealed.get('failed_base')!=str(BASE) or sealed.get('failed_run')!=RUN or sealed.get('source_commit')!=commit or
            hashlib.sha256(ledger_file.read_bytes()).hexdigest()!='e9ac9218e43227759f56a7f75f3e67a3f20bcc3aed6c7040850afd74d2d01980'):
        raise ValueError('PREFIX20_CLEANUP727148_SEALED_ORIGINAL_LEDGER_DIFFERS')
    if ledger['run_id']!=RUN:raise ValueError('PREFIX20_CLEANUP727148_LEDGER_RUN_DIFFERS')
    records={r['pid']:r for job in ledger['jobs'].values() for r in job['ledger'].values()}
    observations=[]
    for pid,record in records.items():
        try:current=identity(pid)
        except (FileNotFoundError,ProcessLookupError):current=None
        matched=current is not None and all(current[k]==record[k] for k in ('pid','start_ticks','sid','uid'))
        if matched and current['state']!='Z':raise ValueError('PREFIX20_CLEANUP727148_RECORDED_READER_STILL_LIVE')
        observations.append(dict(pid=pid,recorded_start_ticks=record['start_ticks'],same_birth=matched,
                                 state=None if current is None else current['state']))
    old_sessions={r['sid'] for r in records.values()}
    def no_old_session_members():
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():continue
            try:
                if path.stat().st_uid!=os.getuid():continue
                row=identity(int(path.name))
            except (FileNotFoundError,ProcessLookupError):continue
            if row['uid']==os.getuid() and row['sid'] in old_sessions and row['state']!='Z':
                raise ValueError('PREFIX20_CLEANUP727148_OLD_SESSION_READER_STILL_LIVE')
    no_old_session_members()
    directory=target/'preparation/failed727148-node-cleanup'
    directory.mkdir(exist_ok=False)
    shadow=directory/'cleanup-shadow';(shadow/'logs').mkdir(parents=True)
    copies={}
    for name in ('run-manifest.json','process-ledger.json','retrieval-staging-claim.json','corpus-cache-claim.json'):
        original=BASE/'logs'/name
        if not original.exists():
            if name=='corpus-cache-claim.json':continue
            raise ValueError('PREFIX20_CLEANUP727148_ORIGINAL_CLAIM_MISSING')
        raw=original.read_bytes();copies[name]=hashlib.sha256(raw).hexdigest()
        if sealed['files'].get(name)!=copies[name]:
            raise ValueError('PREFIX20_CLEANUP727148_ORIGINAL_DIFFERS_FROM_QUEUED_BINDING')
        with (shadow/'logs'/name).open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
    from reproduction.common8.temporary_cache import cleanup as cleanup_cache
    from reproduction.common8.stage_retrieval import cleanup
    cache_cleaned=cleanup_cache(shadow) if (shadow/'logs/corpus-cache-claim.json').exists() else None
    cleaned=cleanup(shadow)
    no_old_session_members()
    if Path(cleaned['root']).exists():raise ValueError('PREFIX20_CLEANUP727148_OWN_TMPFS_STILL_EXISTS')
    for name,digest in copies.items():
        if hashlib.sha256((BASE/'logs'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('PREFIX20_CLEANUP727148_ORIGINAL_EVIDENCE_CHANGED')
    value=dict(status='FAILED727148_ACTUAL_NODE_READERS_ABSENT_AND_OWN_TMPFS_CLEANED',base=str(BASE),
        failed_job=JOB,run_id=RUN,fresh_training_job=current_job,node=socket.gethostname(),uid=os.getuid(),
        recorded_readers=observations,processes_signalled=0,gpu_contexts_created=0,
        actual_reserved_training_gpus=4,additional_gpu_allocation=0,original_terminal=terminal.stdout,
        original_evidence_sha256=copies,old_session_ids=sorted(old_sessions),old_sessions_absent=True,
        original_ledger_sha256=hashlib.sha256(ledger_file.read_bytes()).hexdigest(),
        corpus_cache_cleanup=cache_cleaned,cleanup=cleaned,
        shared_original_retrieval_untouched=True,checked_unix=time.time())
    publish(directory/'complete.json',value)
    publish(target/'preparation/failed727148-node-cleanup-accepted.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recovery-base',required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args();print(json.dumps(execute(args.recovery_base,args.commit)),flush=True)
