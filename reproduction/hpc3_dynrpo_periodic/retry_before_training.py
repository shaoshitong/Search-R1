"""Preserve a qualified probe after an authenticated zero-student-update failure."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import pwd

from .common import BASE,DEFAULT_BASE,own,read,sha,publish


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-base',required=True,type=Path)
    p.add_argument('--job',required=True);args=p.parse_args();own(BASE)
    source=args.source_base
    pattern=re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?'
    if (source==BASE or source.resolve()!=source or source.is_symlink() or
            re.fullmatch(pattern,str(source)) is None or source.stat().st_uid!=os.getuid()):
        raise ValueError('Pretraining retry must preserve an owned independent prior execution')
    record=read(source/'preparation/formal-submission.json')
    if record['job']!=args.job or not args.job.isdecimal():
        raise ValueError('Source job identity differs')
    rows=subprocess.check_output(['sacct','-nP','-X','-j',args.job,'--format=JobID,State,ExitCode,User'],text=True).strip().splitlines()
    if len(rows)!=1 or rows[0].split('|')[0]!=args.job or rows[0].split('|')[1].split()[0] not in ('FAILED','OUT_OF_MEMORY','TIMEOUT'):
        raise ValueError('Source is not an authenticated failed terminal execution')
    if rows[0].split('|')[3] != pwd.getpwuid(os.getuid()).pw_name:
        raise ValueError('Source failed job belongs to another account')
    active=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split()
    if args.job in active:
        raise ValueError('Source job is still active')
    for name in ('training-launch-claim.json','actual-updates.jsonl'):
        if (source/'logs'/name).exists():
            raise ValueError('Training may have started; require actual full-state recovery instead')
    if list((source/'logs').glob('optimizer-rank-*.jsonl')) or list((source/'checkpoints').glob('**/training_state/manifest.json')):
        raise ValueError('Actual student updates/state exist; never reinitialize this prefix')
    cleanup=read(source/'logs/cleanup-status.json')
    if cleanup.get('cleaned') is not True:
        raise ValueError('Prior owned processes have not been reconciled')
    ready=read(source/'warmup/ready.json');original=source/'warmup/continued-head.pt'
    if ready['head_adam_steps']!=400 or sha(original)!=ready['head_sha256']:
        raise ValueError('The accepted existing probe was not preserved intact')
    for name in ('continued-head.pt','ready.json','warmup-metrics.json'):
        target=BASE/'warmup'/name;origin=source/'warmup'/name
        if target.exists():
            if sha(target)!=sha(origin):raise ValueError('Retry warmup reference differs')
        elif name.endswith('.pt'):
            os.link(origin,target)  # Same remote filesystem; no tensor copy or reset.
        else:
            shutil.copyfile(origin,target)
    publish(BASE/'preparation/zero-update-retry.json',dict(source_base=str(source),source_job=args.job,
        actual_source_terminal=rows[0],source_run_id=read(source/'logs/run-manifest.json')['run_id'],
        student_updates_before_failure=0,inherited_probe_adam_steps=400,
        inherited_probe_sha256=ready['head_sha256'],probe_optimizer_reset=False,
        scope='retry before the first student update; not a training checkpoint resume',checked_unix=time.time()),once=True)
    print(json.dumps(dict(stage='ZERO_STUDENT_UPDATE_RETRY_PROBE400_PRESERVED',base=str(BASE))))


if __name__=='__main__':main()
