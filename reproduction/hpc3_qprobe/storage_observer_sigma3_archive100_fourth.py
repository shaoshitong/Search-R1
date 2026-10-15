"""Fresh once-only observation slots, using the unchanged canonical DU worker.

This is an isolated namespace, not a retry of any existing measurement. The
private core keeps the same complete-output, held-child and bounded pure-ENOENT
checks. It never accepts a partial total or repairs permissions itself.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
RELATIVE='reproduction/hpc3_qprobe/storage_observer_sigma3_archive100_fourth.py'
CORE_RELATIVE='reproduction/hpc3_qprobe/storage_observer_sigma3_after_actual_denial_repair.py'
PREFIX='storage-sigma3-archive100-fourth-20261016-attempt-'


def configure(core,slot,entry):
    if type(slot) is not int or not 1<=slot<=64:
        raise ValueError('Explicit fresh measurement slot must be 1..64')
    core.__file__=str(entry)
    core.RELATIVE_SOURCE=RELATIVE
    core.SLOT=slot
    core.CONTROL=ROOT/'observations'/(PREFIX+Path(entry).parents[2].name+'-'+str(slot))
    return core


def load_core(slot):
    entry=Path(__file__).absolute();repo=entry.parents[2]
    if entry.resolve()!=entry or repo.parent!=ROOT/'source' or not re.fullmatch('[0-9a-f]{40}',repo.name):
        raise ValueError('Immutable published observer required')
    # Check both the entry and the reused worker before executing either.
    receipt=ROOT/'preparation'/('source-'+repo.name+'-verification.json')
    before=receipt.stat();raw=receipt.read_bytes();after=receipt.stat();proof=json.loads(raw)
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Source publication changed during read')
    if before.st_uid!=14090 or proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or proof['commit']!=proof['published_remote_sha'] or proof['commit']!=repo.name:
        raise ValueError('Source publication identity differs')
    for name in (RELATIVE,CORE_RELATIVE):
        path=repo/name
        if path.resolve()!=path or path.stat().st_uid!=14090 or hashlib.sha256(path.read_bytes()).hexdigest()!=proof['files'][name]:
            raise ValueError('Published observer source bytes differ')
    spec=importlib.util.spec_from_file_location('_sigma3_fourth_private_du_worker',repo/CORE_RELATIVE)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    return configure(core,slot,entry)


def start_held(core):
    """Retain the real observer Popen; the archive parent must wait/reap it."""
    original=core.subprocess;children=[]
    class PrivateSubprocess:
        def __getattr__(self,name):return getattr(original,name)
        def Popen(self,*args,**kwargs):
            child=original.Popen(*args,**kwargs);children.append(child);return child
    core.subprocess=PrivateSubprocess()
    try:actual=core.start()
    except BaseException as error:
        cleanup=[reap_held(child) for child in children]
        record=dict(phase='OBSERVER_START_EXCEPTION_HELD_CLEANUP',error=repr(error),
                    children=cleanup,cleanup_pending=any(x['cleanup_pending'] for x in cleanup),checked_unix=time.time())
        try:core.publish(core.CONTROL/'parent-launch-error.json',record)
        except BaseException:print(json.dumps(record),flush=True)
        raise
    finally:core.subprocess=original
    if len(children)!=1:
        cleanup=[reap_held(child) for child in children]
        print(json.dumps(dict(phase='UNEXPECTED_OBSERVER_CHILD_COUNT',children=cleanup,
                              cleanup_pending=any(x['cleanup_pending'] for x in cleanup))),flush=True)
        raise ValueError('Fresh observer launch did not yield exactly one held child; no replay')
    return actual,children[0]


def reap_held(child,timeout=1860):
    """Wait for this direct observer, including its bounded DU cleanup path."""
    try:code=child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:code=None
    return dict(pid=child.pid,actual_observer_wait_exit_code=code,
                cleanup_pending=code is None,finished_unix=time.time())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('start','inspect','worker'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--slot',type=int,choices=range(1,65),required=True)
    args=parser.parse_args();os.umask(0o007);core=load_core(args.slot)
    print(json.dumps(core.worker() if args.worker else core.start() if args.start else core.inspect()),flush=True)


if __name__=='__main__':main()
