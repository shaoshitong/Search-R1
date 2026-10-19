"""Owned direct parent records its actual training child's waitpid exit status.

No GPU tests, signals, model changes or inferred exits. This must be the process
which calls submit; an existing detached leader is never adopted or relaunched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def once(path,value):
    with Path(path).open('x')as f:
        json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())


def run(base):
    base=Path(base);prep=base/'preparation'
    if (prep/'submit-intent.json').exists():
        raise ValueError('Existing native intent cannot be adopted or replayed')
    sys.path.insert(0,str(base/'repo'))
    from reproduction.prefix_selection200.launch import submit
    from reproduction.a800.owned_processes import identity
    receipt=submit(base);child=receipt['leader']
    once(prep/'native-wait-supervisor-armed.json',dict(
        supervisor=identity(os.getpid()),native_receipt=receipt,
        submitted_child_parent=os.getpid(),armed_unix=time.time(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    while True:
        try:pid,status=os.waitpid(child['pid'],0);break
        except InterruptedError:continue
    if pid!=child['pid']:raise RuntimeError('Unexpected child wait result')
    code=os.waitstatus_to_exitcode(status)
    once(prep/'native-actual-waitpid-exit.json',dict(
        status='ACTUAL_OWN_CHILD_WAITPID_EXIT_CAPTURED',native_receipt=receipt,
        native_leader=child,actual_waitpid=pid,actual_wait_status=status,
        actual_returncode=code,actual_native_exit_zero=code==0,
        termination_signal=-code if code<0 else None,captured_unix=time.time(),
        final_outputs_and_owned_release_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    run(p.parse_args().base)
