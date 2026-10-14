"""Retained once-only CPU supervisor for the bounded actual15 trace readback.

No training, Slurm submission, model forward, signals or control replacement.
An observation deadline never restarts this child. Unknown launch stays unknown.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl

BASE = ctl.ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
DIRECTORY = ctl.ROOT/'observations/actor-trace15-719225-20261015'
MODULE = 'reproduction.hpc3_qprobe.actor_trace15_control719225'
ENTRY = 'reproduction/hpc3_qprobe/actor_trace15_readonly719225.py'
RAW_SHA = '717e0a81b02acbc38572d54b3b527acc962666110095105f97f14a10e9782953'


def validate(commit):
    ctl.owner()
    ctl.caps()
    ctl.need(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'no bytecode writes required')
    return ctl.source(commit)


def validate_completed(result, process, child, child_live, output, claim):
    ctl.need(isinstance(process, dict) and isinstance(child, dict) and
             all(type(row.get(k)) is int and row[k] > 0 for row in (process, child)
                 for k in ('pid', 'start_ticks', 'sid', 'uid')) and
             process['uid'] == child['uid'] == ctl.OWNER and process['sid'] == process['pid'] and
             child['sid'] == process['sid'] and process['pid'] != child['pid'] and
             result['process'] == process and result['child'] == child and 'worker_error' not in result and
             result['actual_child_exit_code'] == 0 and not child_live and result['stderr']['bytes'] == 0,
             'actual CPU child must have bound birth/waited0/empty stderr')
    ctl.need(output['status'] == 'ACTUAL_RAW15_PRIVATE_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL' and
             output['step'] == 15 and output['job'] == '719225' and output['raw_file']['sha256'] == RAW_SHA and
             output['source_commit'] == claim['producer']['producer_commit'] and
             output['source_manifest']['sha256'] == claim['producer']['source_manifest_sha256'] and
             output['allocation_file']['sha256'] == claim['producer']['native_allocation_sha256'] and
             output['actual_private_trace_byte_equal'] is True and
             output['actual_selected_logprob_projection_equal'] is True and
             output['hf_forward_performed'] is False and output['model_weights_loaded'] is False and
             output['generated_trajectories'] == output['actor_or_head_updates_performed'] ==
             output['evidence_files_written'] == 0, 'actual CPU reader result differs')


def inspect(commit):
    repository, publication = validate(commit)
    if not DIRECTORY.exists():
        return dict(status='NOT_STARTED', directory=str(DIRECTORY), source=publication)
    ctl.need(DIRECTORY.resolve() == DIRECTORY and DIRECTORY.stat().st_uid == ctl.OWNER,
             'unsafe trace control directory')
    claim = ctl.read(DIRECTORY/'claim.json')
    ctl.need(claim['source'] == publication and claim['expected_raw_sha256'] == RAW_SHA and
             claim['step'] == 15 and claim['producer']['base'] == str(BASE) and
             claim['producer']['producer_job'] == '719225', 'actual retained trace claim differs')
    process = ctl.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    child = ctl.read(DIRECTORY/'child.json') if (DIRECTORY/'child.json').exists() else None
    result = ctl.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None
    process_live = process is not None and ctl.live(process)
    child_live = child is not None and ctl.live(child)
    status = 'ACTUAL_TRACE_CPU_SUPERVISOR_LIVE' if process_live else 'UNKNOWN_NO_REPLAY'
    if result is not None:
        ctl.need(result['source'] == publication and result['claim_sha256'] == ctl.digest(DIRECTORY/'claim.json')
                 and result['process'] == process and result['child'] == child and
                 result['completed_unix'] >= result['started_unix'] >= claim['created_unix'],
                 'actual waited trace result binding differs')
        for name in ('stdout', 'stderr'):
            path = DIRECTORY/('audit.'+name)
            ctl.need(result[name]['path'] == str(path) and ctl.digest(path) == result[name]['sha256'] and
                     path.stat().st_size == result[name]['bytes'], 'actual CPU output SHA/bytes differ')
        status = result['status']
        if status == 'ACTUAL_TRACE_CPU_CHILD_COMPLETED0':
            output = ctl.read(DIRECTORY/'audit.stdout')
            validate_completed(result, process, child, child_live, output, claim)
            result = dict(result, readback=output)
    return dict(status=status, actual_process_live_birth=process_live,
                actual_child_live_birth=child_live, process=process, child=child,
                claim=claim, result=result, checked_unix=time.time())


def start(commit):
    repository, publication = validate(commit)
    if DIRECTORY.exists():
        return inspect(commit)
    run, binding, _ = ctl.producer(BASE)
    ctl.need(binding['producer_job'] == '719225' and binding['producer_commit'] ==
             '1843babdd96b387dd10d91e3207c9ea9489b6941', 'original source/native allocation differs')
    DIRECTORY.parent.mkdir(mode=0o770, exist_ok=True)
    ctl.need(DIRECTORY.parent.resolve() == DIRECTORY.parent and DIRECTORY.parent.stat().st_uid == ctl.OWNER,
             'actual JXU-owned observation namespace required')
    DIRECTORY.mkdir(mode=0o770)
    command = [sys.executable, '-m', MODULE, '--commit', commit, '--worker']
    ctl.publish(DIRECTORY/'claim.json', dict(source=publication, producer=binding,
                expected_raw_sha256=RAW_SHA, step=15, command=command, created_unix=time.time(),
                parent=ctl.birth(ctl.process(os.getpid()))))
    with (DIRECTORY/'supervisor.log').open('x') as logfile:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=logfile,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                                 cwd=str(repository))
    for _ in range(50):
        if (DIRECTORY/'process.json').exists() or child.poll() is not None:
            break
        time.sleep(0.2)
    return inspect(commit)


def worker(commit):
    repository, publication = validate(commit)
    claim = ctl.read(DIRECTORY/'claim.json')
    ctl.need(claim['source'] == publication and claim['expected_raw_sha256'] == RAW_SHA,
             'worker claim/source differs')
    process = ctl.birth(ctl.process(os.getpid()))
    ctl.need(process['uid'] == ctl.OWNER and process['sid'] == process['pid'], 'detached CPU birth differs')
    ctl.publish(DIRECTORY/'process.json', process)
    started = time.time()
    child = None
    actual_child = None
    worker_error = None
    with (DIRECTORY/'audit.stdout').open('x') as stdout, (DIRECTORY/'audit.stderr').open('x') as stderr:
        try:
            child = subprocess.Popen([sys.executable, str(repository/ENTRY)], stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, cwd=str(repository), close_fds=True)
            actual_child = ctl.birth(ctl.process(child.pid))
            ctl.need(actual_child['uid'] == ctl.OWNER and actual_child['sid'] == process['sid'],
                     'direct CPU child owner/session differs')
            ctl.publish(DIRECTORY/'child.json', actual_child)
        except Exception as error:
            worker_error = dict(error_type=type(error).__name__, message=str(error))
        finally:
            # Retain and reap the original direct diagnostic even if its
            # metadata publication fails. No observer deadline causes replay.
            code = child.wait() if child is not None else None
        stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    outputs = {name: dict(path=str(DIRECTORY/('audit.'+name)),
                          sha256=ctl.digest(DIRECTORY/('audit.'+name)),
                          bytes=(DIRECTORY/('audit.'+name)).stat().st_size) for name in ('stdout', 'stderr')}
    value = dict(source=publication, claim_sha256=ctl.digest(DIRECTORY/'claim.json'),
                 process=process, child=actual_child, actual_child_exit_code=code,
                 started_unix=started, completed_unix=time.time(), **outputs)
    if worker_error is not None:
        value['worker_error'] = worker_error
    value['status'] = ('ACTUAL_TRACE_CPU_CHILD_COMPLETED0' if code == 0 and worker_error is None and outputs['stderr']['bytes'] == 0
                       else 'ACTUAL_TRACE_CPU_CHILD_FAILED_EVIDENCE_RETAINED')
    ctl.publish(DIRECTORY/'result.json', value)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--start', action='store_true')
    g.add_argument('--inspect', action='store_true')
    g.add_argument('--worker', action='store_true')
    a = p.parse_args()
    os.umask(0o007)
    print(json.dumps(start(a.commit) if a.start else worker(a.commit) if a.worker else inspect(a.commit)), flush=True)
