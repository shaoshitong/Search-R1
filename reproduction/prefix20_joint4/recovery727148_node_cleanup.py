"""CPU-only cleanup of the exact failed727148 node staging namespace.

No process is signalled: Slurm must already have reaped every recorded reader.
Only the original exact-inode tmpfs asset copy is removed; originals stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from reproduction.prefix20_joint4.control import publish
from reproduction.a800.owned_processes import identity
from reproduction.common8.native_allocation import fields

BASE=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016')
JOB='727148'
RUN='ebcff3f7fff6499e9f121153dc29ebab'


def execute():
    if os.getuid()!=14090 or socket.gethostname().split('.')[0]!='ACD1-25':
        raise ValueError('PREFIX20_CLEANUP727148_ACTUAL_OWN_NODE_REQUIRED')
    cpu_job=os.environ.get('SLURM_JOB_ID','')
    if not cpu_job.isdecimal() or cpu_job==JOB:raise ValueError('PREFIX20_CLEANUP727148_CPU_JOB_REQUIRED')
    result=subprocess.run(['scontrol','show','job','-o',cpu_job],capture_output=True,text=True,timeout=30)
    actual=fields(result.stdout)
    if (result.returncode or actual.get('UserId')!='jxu957(14090)' or actual.get('JobState')!='RUNNING' or
            actual.get('JobName')!='qprobe-prefix20-727148-cleanup' or actual.get('NumCPUs')!='1' or
            any(int(n)!=0 for n in re.findall(r'gres/gpu(?:[^=, ]*)?=([0-9]+)',actual.get('TRES','')))):
        raise ValueError('PREFIX20_CLEANUP727148_NOT_BOUND_CPU_ONLY')
    terminal=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Start,End,NodeList'],capture_output=True,text=True,timeout=30)
    expected='727148|FAILED|1:0|jxu957|14090|2026-10-16T22:16:44|2026-10-16T22:26:18|ACD1-25'
    if terminal.returncode or expected not in terminal.stdout.splitlines() or not any(
            line.startswith(JOB+'.extern|COMPLETED|0:0|') for line in terminal.stdout.splitlines()):
        raise ValueError('PREFIX20_CLEANUP727148_ORIGINAL_TERMINAL_DIFFERS')
    ledger_file=BASE/'logs/process-ledger.json';ledger=json.loads(ledger_file.read_text())
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
    directory=BASE/'preparation/failed727148-node-cleanup'
    directory.mkdir(exist_ok=False)
    claim=BASE/'logs/retrieval-staging-claim.json'
    before=claim.read_bytes()
    with (directory/'original-retrieval-staging-claim.json').open('xb') as stream:
        stream.write(before);stream.flush();os.fsync(stream.fileno())
    from reproduction.common8.stage_retrieval import cleanup
    cleaned=cleanup(BASE)
    value=dict(status='FAILED727148_ACTUAL_NODE_READERS_ABSENT_AND_OWN_TMPFS_CLEANED',base=str(BASE),
        failed_job=JOB,run_id=RUN,cpu_job=cpu_job,node=socket.gethostname(),uid=os.getuid(),
        recorded_readers=observations,processes_signalled=0,gpu_allocations=0,original_terminal=terminal.stdout,
        original_ledger_sha256=hashlib.sha256(ledger_file.read_bytes()).hexdigest(),
        original_staging_claim_sha256=hashlib.sha256(before).hexdigest(),cleanup=cleaned,
        shared_original_retrieval_untouched=True,checked_unix=time.time())
    publish(directory/'complete.json',value)
    return value


if __name__=='__main__':
    print(json.dumps(execute()),flush=True)
