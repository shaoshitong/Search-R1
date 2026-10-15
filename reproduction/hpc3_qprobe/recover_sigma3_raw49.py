"""Continue only unclaimed RAW attempts after the authentic original49."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import raw_audit_followthrough_sigma3 as runtime
from reproduction.hpc3_qprobe import sigma3_service_recovery as recovery

BASE=recovery.BASE
DIRECTORY=recovery.RAW_DIRECTORY
MODULE='reproduction.hpc3_qprobe.recover_sigma3_raw49'
STATUS=runtime.STATUS
birth_ok=runtime.birth_ok
canonical_sha=runtime.canonical_sha
actual_tasks=runtime.actual_tasks
_service=None


def service():
    """Bind adjacent dependency to this pinned controller, including later readers."""
    global _service
    if _service is None or _service.ctl is not ctl:
        path=Path(__file__).with_name('sigma3_service_recovery.py')
        spec=importlib.util.spec_from_file_location('_sigma3_RAW49_service_handoff',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ctl=ctl;_service=module
    return _service


def validate(commit):
    ctl.owner();ctl.caps()
    ctl.need(os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','recovered RAW bytecode writes forbidden')
    repository,source=ctl.source(commit)
    ctl.need(Path(__file__).resolve()==repository/'reproduction/hpc3_qprobe/recover_sigma3_raw49.py',
        'recovered RAW loaded source differs')
    return repository,source


def core():
    path=Path(__file__).with_name('raw_audit_followthrough_sigma3.py')
    spec=importlib.util.spec_from_file_location('_sigma3_recovered_RAW49_runtime',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.DIRECTORY=DIRECTORY;module.ctl=ctl
    return module


def task_directory(task):
    root=recovery.RAW_ORIGINAL if task['attempt_index']<=49 else DIRECTORY
    return root/'attempts'/task['attempt_id']


def consume_task(task,directory,supervisor):
    ctl.need(Path(directory)==task_directory(task),'RAW task assigned to another source namespace')
    if task['attempt_index']<=49:
        old,_=service().old_raw()
        return old.consume_task(task,directory,recovery.RAW_BIRTH)
    return core().consume_task(task,directory,supervisor)


def accepted_paths():
    old=sorted((recovery.RAW_ORIGINAL/'accepted').glob('*.json'))
    new=sorted((DIRECTORY/'accepted').glob('*.json'))
    ctl.need([p.name for p in old]==['%06d.json'%i for i in range(1,50)] and
        [p.name for p in new]==['%06d.json'%i for i in range(50,50+len(new))],
        'RAW original49/new suffix receipts overlap or have a gap')
    return old+new


def inspect(commit):
    _,source=validate(commit);reference=service().consume(commit)
    if not DIRECTORY.exists():return dict(status='NOT_STARTED',source=source,recovery=reference)
    claim=ctl.read(DIRECTORY/'claim.json');ctl.need(claim['source']==source and claim['recovery']==reference,
        'recovered RAW claim/source/handoff differs')
    process=ctl.read(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    progress=ctl.read(DIRECTORY/'state.json') if (DIRECTORY/'state.json').exists() else None
    final=ctl.read(DIRECTORY/'result.json') if (DIRECTORY/'result.json').exists() else None
    paths=accepted_paths();records=[ctl.read(p) for p in paths];tasks=actual_tasks(claim['parts'])
    ctl.need([r['task'] for r in records]==tasks[:len(records)],'recovered RAW receipts do not match actual attempt prefix')
    latest=consume_task(records[-1]['task'],task_directory(records[-1]['task']),process) if records else None
    if records:ctl.need(latest==records[-1],'recovered RAW latest original wait/output differs')
    if final:
        ctl.need(final['source']==source and final['process']==process and
            final['claim_sha256']==ctl.digest(DIRECTORY/'claim.json'),'recovered RAW terminal source/birth differs')
        if final['status']==STATUS:
            consumed=[consume_task(r['task'],task_directory(r['task']),process) for r in records]
            ctl.need(consumed==records==final['records'] and [r['task'] for r in records]==tasks and
                [r['task']['step'] for r in records if r['task']['complete_outer']]==list(range(1,201)) and
                canonical_sha(records)==final['record_list_sha256'] and len(records)==final['normal_return_attempts'],
                'recovered all200 actual held receipts differ')
    child=(progress or {}).get('child')
    return dict(status=final['status'] if final else ('ACTUAL_READONLY_AUDIT_LIVE' if process and ctl.live(process) else 'UNKNOWN_NO_REPLAY'),
        process=process,actual_live_birth=bool(process and ctl.live(process)),actual_child_live_birth=bool(child and ctl.live(child)),
        accepted_attempt_receipts=len(records),progress=progress,latest_completed_audit=latest,
        result={k:v for k,v in final.items() if k not in ('records','formal_attempt_audits')} if final else None,
        source=source,recovery=reference,checked_unix=time.time())


def start(commit):
    repository,source=validate(commit);reference=service().prepare(commit)
    if DIRECTORY.exists():return inspect(commit)
    parts=core().plan();ctl.need(parts==service().inherited()['parts'],'recovered RAW producer plan differs')
    DIRECTORY.mkdir(mode=0o770);(DIRECTORY/'attempts').mkdir();(DIRECTORY/'accepted').mkdir()
    command=[sys.executable,'-m',MODULE,'worker','--commit',commit]
    ctl.publish(DIRECTORY/'claim.json',dict(source=source,parts=parts,recovery=reference,command=command,
        created_unix=time.time(),training_control_permitted=False,goal_complete=False))
    with (DIRECTORY/'supervisor.log').open('x') as output:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            cwd=str(repository),close_fds=True,start_new_session=True)
    for _ in range(50):
        if (DIRECTORY/'process.json').exists() or child.poll() is not None:break
        time.sleep(.2)
    return inspect(commit)


def worker(commit):
    service().consume(commit)
    return core().worker(commit,initial_records=service().inherited()['raw_prefix_records'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','worker','inspect'));p.add_argument('--commit',required=True)
    a=p.parse_args();os.umask(0o007)
    print(__import__('json').dumps(start(a.commit) if a.mode=='start' else worker(a.commit) if a.mode=='worker' else inspect(a.commit)),flush=True)
