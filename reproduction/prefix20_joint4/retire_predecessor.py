"""Retire exactly the authorized, inactive predecessor CP10/100 tensor files."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.prefix20_joint4.control import own_base, publish

OLD = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014')
RUN = 'c3856315d3f345d7920ebf5725a083b4'
NAMES = {'model-0000%d-of-00003.safetensors'%i for i in (1,2,3)} | {
    'training_state/rank-%d.pt'%i for i in range(4)} | {
    'training_state/driver.pt', 'training_state/question-probe-state.pt',
    'training_state/question-probe-state.pt.head.pt'}


def validate_plan(plan):
    if (plan.get('kind') != 'authorized_previous_qprobe_checkpoint_retirement' or
            plan.get('source_base') != str(OLD) or plan.get('run_id') != RUN or
            not plan.get('authorization') or not plan.get('metadata_logs_evaluations_retained')):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_AUTHORIZATION_BINDING_DIFFERS')
    expected = {str(OLD/'checkpoints/search-g1-common8/actor'/('global_step_'+str(step))/name)
                for step in (10,100) for name in NAMES}
    rows = plan['files']
    if len(rows) != 20 or {row['path'] for row in rows} != expected:
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_EXACT_FILE_SET_DIFFERS')
    if any(row['symlink'] or row['uid'] != 14090 or row['nlink'] != 1 or
           type(row['bytes']) is not int or row['bytes'] <= 0 for row in rows):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_UNSAFE_SOURCE_RECORD')
    if (sum(row['bytes'] for row in rows) != plan['logical_bytes'] or
            sum(row['allocated'] for row in rows) != plan['allocated_file_bytes']):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_BYTE_TOTAL_DIFFERS')


def identity(row):
    path = Path(row['path'])
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_UNSAFE_PATH')
    stat = path.stat()
    actual = dict(bytes=stat.st_size, allocated=stat.st_blocks*512, inode=stat.st_ino,
                  nlink=stat.st_nlink, mtime_ns=stat.st_mtime_ns, uid=stat.st_uid, gid=stat.st_gid)
    if any(actual[key] != row[key] for key in actual):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_FILE_CHANGED')


def inactive():
    run = json.loads((OLD/'logs/run-manifest.json').read_text())
    state = json.loads((OLD/'logs/pipeline-status.json').read_text())
    if run['run_id'] != RUN or state['run_id'] != RUN or state['stage'] != 'FAILED':
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_NOT_TERMINAL')
    queue = subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i|%u|%T|%Z|%o'],
                           capture_output=True,text=True,timeout=30)
    if queue.returncode or str(OLD) in queue.stdout or any(
            line.split('|')[0] in ('718650','719225') for line in queue.stdout.splitlines()):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_STILL_QUEUED_OR_REFERENCED')
    result = subprocess.run(['sacct','-nP','-j','718650,719225',
        '--format=JobID,State,ExitCode,User,UID,Account,Start,End,NodeList'],
        capture_output=True,text=True,timeout=30)
    required = {
        '718650|FAILED|1:0|jxu957|14090|jxu957|2026-10-14T20:13:05|2026-10-14T22:20:47|ACD1-1',
        '719225|CANCELLED by 14090|0:0|jxu957|14090|jxu957|2026-10-15T00:10:34|2026-10-15T14:36:47|ACD1-56'}
    if result.returncode or not required <= set(result.stdout.splitlines()) or not all(
            any(line.startswith(job+'.extern|COMPLETED|0:0|') for line in result.stdout.splitlines())
            for job in ('718650','719225')):
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_ACTUAL_ACCOUNTING_DIFFERS')
    return dict(accounting=result.stdout, queue=queue.stdout, pipeline_state=state,
                scope='old job718650 failed; later job719225 human-cancelled; current native4 job untouched')


def retire(base, plan_path, expected_sha):
    base = own_base(base);plan_path = Path(plan_path)
    if os.getuid() != 14090 or plan_path.parent != base/'preparation' or plan_path.is_symlink():
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_OWNER_AND_PLAN_REQUIRED')
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_PLAN_SHA_DIFFERS')
    plan = json.loads(raw);validate_plan(plan)
    claim = base/'preparation/retire-old10-100-claim.json'
    receipt = base/'preparation/retire-old10-100-complete.json'
    if claim.exists() or receipt.exists():
        raise ValueError('PREFIX20_RETIRE_PREDECESSOR_ALREADY_ATTEMPTED_INSPECT')
    with (OLD/'gpu-job.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal = inactive()
        for row in plan['files']:identity(row)
        publish(claim,dict(plan_file=str(plan_path),plan_sha256=expected_sha,
            authorization=plan['authorization'],terminal=terminal,files=plan['files'],started_unix=time.time()))
        journal = base/'preparation/retire-old10-100-deleted.jsonl'
        with journal.open('x') as stream:
            for row in plan['files']:
                identity(row);Path(row['path']).unlink()
                stream.write(json.dumps(dict(row,deleted_unix=time.time()))+'\n')
                stream.flush();os.fsync(stream.fileno())
        for parent in {Path(row['path']).parent for row in plan['files']}:
            fd = os.open(parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        if any(Path(row['path']).exists() for row in plan['files']):
            raise ValueError('PREFIX20_RETIRE_PREDECESSOR_PATH_REAPPEARED')
        value = dict(status='AUTHORIZED_PREDECESSOR_CP10_CP100_TENSORS_RETIRED',
            files=20,logical_bytes=plan['logical_bytes'],allocated_file_bytes=plan['allocated_file_bytes'],
            all_deleted_paths_absent=True,old_checkpoint_metadata_logs_evaluations_retained=True,
            current_prefix20_run_untouched=True,plan_sha256=expected_sha,journal=str(journal),
            checked_unix=time.time())
        publish(receipt,value)
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',required=True);parser.add_argument('--plan',required=True)
    parser.add_argument('--plan-sha',required=True);args = parser.parse_args()
    print(json.dumps(retire(args.base,args.plan,args.plan_sha)))


if __name__ == '__main__':main()
