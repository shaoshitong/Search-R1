"""Once CPU followthrough continuation after the observed accounting outage."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as control
from reproduction.hpc3_qprobe import sigma3_service_recovery as recovery
from reproduction.hpc3_qprobe import followthrough_sigma3 as original

BASE=original.BASE
ROOT=original.ROOT
RUN=original.RUN
JOB=original.JOB
PRODUCER=original.PRODUCER
DIRECTORY=recovery.FOLLOW_DIRECTORY
MODULE='reproduction.hpc3_qprobe.recover_sigma3_followthrough_service'
_service=None


def service():
    global _service
    if _service is None or _service.ctl is not control:
        path=Path(__file__).with_name('sigma3_service_recovery.py')
        spec=importlib.util.spec_from_file_location('_sigma3_follow_service_handoff',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ctl=control;_service=module
    return _service


def fixed_binding():return core().fixed_binding()


def consume(commit):return service().consume(commit)


def core():
    path=Path(__file__).with_name('followthrough_sigma3.py')
    spec=importlib.util.spec_from_file_location('_sigma3_service_recovered_followthrough',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.DIRECTORY=DIRECTORY;module.control=control
    return module


def start(commit):
    repository,source=control.source(commit);reference=service().prepare(commit)
    if DIRECTORY.exists():return inspect()
    binding=fixed_binding();actual=core().accounting()
    control.need(actual['state'] in ('RUNNING','COMPLETING','COMPLETED') and
        (actual['state']!='COMPLETED' or actual['exit_code']=='0:0'),'actual producer failed; never restart training')
    DIRECTORY.mkdir(mode=0o770)
    command=[str(BASE/'env/bin/python'),'-m',MODULE,'supervise','--commit',commit]
    control.publish(DIRECTORY/'claim.json',dict(schema_version=1,binding=binding,source=source,recovery=reference,
        command=command,cwd=str(repository),actual_initial_accounting=actual,created_unix=time.time(),
        training_submission_permitted=False,actor_model_reload_permitted=False,direct_gpu_access_permitted=False,
        native_final_restore_only_submission_permitted=True))
    with (DIRECTORY/'output.log').open('x') as output:
        child=subprocess.Popen(command,cwd=repository,stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    birth=control.process(child.pid)
    control.need(birth['uid']==control.OWNER and birth['sid']==birth['pid'],'actual detached follower birth differs')
    control.publish(DIRECTORY/'process.json',dict(actual_process=control.birth(birth),commit=commit,
        control=str(DIRECTORY),checked_unix=time.time()))
    return inspect()


def supervise(commit):
    reference=consume(commit);claim=control.read(DIRECTORY/'claim.json')
    control.need(claim['recovery']==reference,'recovered follower handoff differs')
    # Carry only the original held0/full-SHA proof; final200 rereads all archives.
    verified={int(k):v for k,v in service().inherited()['verified_archives'].items()}
    return core().supervise(commit,initial_verified_archives=verified)


def inspect():
    claim=control.read(DIRECTORY/'claim.json')
    control.need(claim['recovery']==consume(claim['source']['commit']),'recovered follower handoff changed')
    record=control.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    return dict(actual_live_birth=bool(record and control.live(record['actual_process'])),process=record,
        state=control.read(DIRECTORY/'state.json') if (DIRECTORY/'state.json').exists() else None,
        result=control.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None,
        checked_unix=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','supervise','inspect'));p.add_argument('--commit')
    a=p.parse_args();control.owner();control.caps()
    if a.mode=='supervise':sys.exit(supervise(a.commit))
    print(__import__('json').dumps(start(a.commit) if a.mode=='start' else inspect()),flush=True)
