"""Replace only a held, never-started owned job after publishing a source repair.

This is not a training resume. A RUNNING job is never cancelled by this tool.
The accepted continuous probe is retained by hard link and fresh full SHA.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .common import BASE, DEFAULT_BASE, own, read, sha, publish


def validate_pending(fields, *, job, source, uid, require_held=False):
    if (fields.get('JobId') != job or
            fields.get('UserId','').split('(')[-1] != str(uid)+')' or
            fields.get('JobName') != 'dynrpo-periodic5-formal' or
            fields.get('WorkDir') != str(source)):
        raise ValueError('PENDING_REPLACEMENT_JOB_IDENTITY_DIFFERS')
    if fields.get('JobState') != 'PENDING':
        raise ValueError('PENDING_REPLACEMENT_NEVER_CANCELS_STARTED_JOB')
    if require_held and fields.get('Priority') != '0':
        raise ValueError('PENDING_REPLACEMENT_REQUIRES_OWN_HELD_JOB')


def inspect(job):
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
    return dict(re.findall(r'(\w+)=([^ ]+)',raw))


def never_started(source):
    for name in ('logs/run-manifest.json','logs/training-launch-claim.json','logs/actual-updates.jsonl'):
        if (source/name).exists():
            raise ValueError('PENDING_REPLACEMENT_SOURCE_HAS_REAL_RUNTIME_STATE')
    if list((source/'logs').glob('optimizer-rank-*.jsonl')) or list((source/'checkpoints').glob('**/manifest.json')):
        raise ValueError('PENDING_REPLACEMENT_SOURCE_HAS_REAL_STUDENT_STATE')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('hold','prepare','retire'))
    p.add_argument('--source-base',required=True,type=Path);p.add_argument('--job',required=True)
    args=p.parse_args();source=args.source_base;own(BASE)
    pattern=re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?'
    if (not args.job.isdecimal() or source==BASE or source.resolve()!=source or source.is_symlink() or
            re.fullmatch(pattern,str(source)) is None or source.stat().st_uid!=os.getuid()):
        raise ValueError('PENDING_REPLACEMENT_REQUIRES_DISTINCT_OWN_NAMESPACE')
    submission=read(source/'preparation/formal-submission.json')
    if submission['job']!=args.job:
        raise ValueError('PENDING_REPLACEMENT_SOURCE_RECEIPT_DIFFERS')
    with (source/'preparation/formal-submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=inspect(args.job)
        validate_pending(before,job=args.job,source=source,uid=os.getuid(),require_held=args.action!='hold')
        never_started(source)
        if args.action=='hold':
            subprocess.run(['scontrol','hold',args.job],check=True)
            after=inspect(args.job)
            validate_pending(after,job=args.job,source=source,uid=os.getuid(),require_held=True)
            print(json.dumps(dict(status='OWN_NEVER_STARTED_JOB_HELD_FOR_PUBLISHED_REPAIR',
                job=args.job,source_base=str(source),target_base=str(BASE),actual_job=after)))
            return
        ready=read(source/'warmup/ready.json');head=source/'warmup/continued-head.pt'
        if ready['head_adam_steps']!=400 or sha(head)!=ready['head_sha256']:
            raise ValueError('PENDING_REPLACEMENT_ACCEPTED_PROBE400_CHANGED')
        if args.action=='prepare':
            for name in ('continued-head.pt','ready.json','warmup-metrics.json'):
                origin,target=source/'warmup'/name,BASE/'warmup'/name
                if target.exists():
                    if sha(target)!=sha(origin):raise ValueError('PENDING_REPLACEMENT_TARGET_STATE_DIFFERS')
                elif name.endswith('.pt'):os.link(origin,target)
                else:shutil.copyfile(origin,target)
            receipt=dict(source_base=str(source),source_job=args.job,source_commit=submission['commit'],
                target_commit=read(BASE/'preparation/source-publication.json')['commit'],
                source_was_held_pending=True,student_updates_before_replacement=0,
                inherited_probe_adam_steps=400,inherited_probe_sha256=ready['head_sha256'],
                probe_optimizer_reset=False,student_prefix_repeated=False,
                reason='Repair native retriever querying eight CUDA devices on a four-GPU allocation',
                checked_unix=time.time())
            path=BASE/'preparation/pending-replacement.json'
            if path.exists():
                existing=read(path)
                for key in receipt:
                    if key!='checked_unix' and existing[key]!=receipt[key]:raise ValueError('PENDING_REPLACEMENT_RECEIPT_DIFFERS')
            else:publish(path,receipt,once=True)
            print(json.dumps(dict(status='HELD_PENDING_REPLACEMENT_PROBE400_FULL_SHA_PASS',**receipt)))
            return
        receipt=read(BASE/'preparation/pending-replacement.json')
        if (receipt['source_base']!=str(source) or receipt['source_job']!=args.job or
                receipt['inherited_probe_sha256']!=sha(BASE/'warmup/continued-head.pt') or
                receipt['target_commit']!=read(BASE/'preparation/source-publication.json')['commit']):
            raise ValueError('PENDING_REPLACEMENT_TARGET_NOT_AUTHENTICATED')
        if (BASE/'preparation/formal-submission.json').exists():
            raise ValueError('PENDING_REPLACEMENT_TARGET_ALREADY_SUBMITTED')
        validate_pending(inspect(args.job),job=args.job,source=source,uid=os.getuid(),require_held=True)
        never_started(source)
        subprocess.run(['scancel',args.job],check=True)
        deadline=time.monotonic()+20
        while True:
            queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split()
            accounting=subprocess.check_output(['sacct','-nP','-X','-j',args.job,'--format=JobID,State,ExitCode,User'],text=True).strip()
            rows=[line.split('|') for line in accounting.splitlines() if line]
            if args.job not in queue and len(rows)==1 and rows[0][0]==args.job and rows[0][1].split()[0]=='CANCELLED':break
            if time.monotonic()>deadline:raise ValueError('PENDING_REPLACEMENT_RETIRE_OUTCOME_REQUIRES_RECONCILIATION')
            time.sleep(1)
        publish(BASE/'preparation/pending-source-retired.json',dict(source_job=args.job,
            source_base=str(source),actual_accounting=accounting,student_updates=0,
            accepted_probe_sha256=ready['head_sha256'],checked_unix=time.time()),once=True)
        print(json.dumps(dict(status='OWN_HELD_PENDING_SOURCE_TERMINAL_CANCELLED_ZERO_STUDENT_UPDATES',
            job=args.job,actual_accounting=accounting)))


if __name__=='__main__':main()
