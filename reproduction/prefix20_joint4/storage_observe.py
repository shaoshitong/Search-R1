"""Owned, bounded full-root du with durable child identity, byte logs and exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha,atomic
from reproduction.prefix20_joint4.control import own_base,PERSONAL_ROOT,publish


def identity(pid):
    value=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(value[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def observe(base,commit,*,publish_canonical=True):
    os.umask(0o007);base=own_base(base);preparation=base/'preparation'
    with (preparation/'storage-full-observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        directory=preparation/('storage-observation-'+str(time.time_ns()));directory.mkdir()
        start=time.time();stdout=directory/'du.stdout';stderr=directory/'du.stderr';timed_out=False
        with stdout.open('xb') as out,stderr.open('xb') as error:
            child=subprocess.Popen(['du','-s','-B1',PERSONAL_ROOT],stdout=out,stderr=error)
            claim=dict(observer=identity(os.getpid()),child=identity(child.pid),command=['du','-s','-B1',PERSONAL_ROOT],
                started_unix=start,source_commit=commit,hostname=socket.gethostname(),timeout_seconds=1800)
            publish(directory/'claim.json',claim)
            while child.poll() is None:
                if time.time()-start>1800:child.kill();timed_out=True;break
                atomic(directory/'progress.json',dict(observer=claim['observer'],child=claim['child'],
                    stdout_bytes=stdout.stat().st_size,stderr_bytes=stderr.stat().st_size,checked_unix=time.time()))
                time.sleep(5)
            code=child.wait();out.flush();error.flush();os.fsync(out.fileno());os.fsync(error.fileno())
        output=stdout.read_text();errors=stderr.read_text()
        proof=dict(role='prefix20_prelaunch_personal_storage_observation',account_experiment_root=PERSONAL_ROOT,
            observing_user=pwd.getpwuid(os.getuid()).pw_name,observer_uid=os.getuid(),observer_source_sha256=sha(Path(__file__)),
            source_commit=commit,hostname=socket.gethostname(),claim=claim,returncode=code,timed_out=timed_out,
            stdout=output,stderr=errors,stdout_file=str(stdout),stderr_file=str(stderr),
            stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),measurement_complete=code==0 and not timed_out and not errors.strip(),
            started_unix=start,checked_unix=time.time(),training_job_submitted=False)
        publish(directory/'result.json',proof)
        # A refresh may require a second account to measure denied subtrees.
        # Keep the last complete canonical proof visible until that aggregate
        # is validated and atomically published by storage_complete.
        if publish_canonical:
            atomic(preparation/'personal-storage-observation.json',proof)
        print(json.dumps(proof),flush=True)
        return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--no-publish-canonical',action='store_true',
        help='retain the current canonical proof while recording new raw evidence; complete publishes the validated aggregate')
    a=p.parse_args()
    observe(a.base,a.commit,publish_canonical=not a.no_publish_canonical)


if __name__=='__main__':main()
