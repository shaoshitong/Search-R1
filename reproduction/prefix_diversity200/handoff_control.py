"""Deploy and arm one reviewed source handoff outside the frozen trainer tree."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_guard(successor):
    successor=Path(successor);config=read(successor/'preparation/launch-config.json')
    spec=config['resume_joint'];source=Path(spec['source_base'])
    plan_source=successor/'repo/reproduction/prefix_diversity200/i1i6_evidence/CALIBRATION_HANDOFF_PLAN.json'
    plan=read(plan_source);previous=read(source/'preparation/launch-config.json')
    if (not spec['planned_calibration_handoff'] or spec['completed_outer']!=plan['target_complete_outer']
            or str(source)!=plan['source_base'] or str(successor)!=plan['successor_base']
            or str(spec['source_job'])!=str(plan['source_job']) or config['run_id']!=plan['run_id']
            or previous['run_id']!=plan['run_id'] or previous['source_commit']!=plan['source_commit']
            or sha(plan_source)!=spec['planned_handoff_plan_sha256']
            or sha(source/'preparation/launch-config.json')!=plan['source_config_sha256']):
        raise ValueError('Source, successor and published plan differ')
    directory=source/'preparation/calibration-handoff-v1'
    if directory.exists():raise FileExistsError('Existing guard preparation needs reconciliation, never overwrite')
    directory.mkdir(mode=0o700)
    module=successor/'repo/reproduction/prefix_diversity200/calibration_handoff_guard.py'
    expected=read(successor/'preparation/source-files.json')['reproduction/prefix_diversity200/calibration_handoff_guard.py']
    if sha(module)!=expected:raise ValueError('Guard differs from published source manifest')
    (directory/'plan.json').write_bytes(plan_source.read_bytes())
    (directory/'guard.py').write_bytes(module.read_bytes())
    sys.path.insert(0,str(successor/'repo'))
    from reproduction.prefix_diversity200.calibration_handoff_guard import once
    value=dict(status='REVIEWED_GUARD_STAGED_NOT_ARMED',directory=str(directory),
        plan_sha256=sha(directory/'plan.json'),guard_sha256=sha(directory/'guard.py'),
        successor_config_sha256=sha(successor/'preparation/launch-config.json'),
        successor=str(successor),source_job=spec['source_job'],source_code_modified=False,
        observed_unix=time.time())
    once(directory/'prepare-receipt.json',value)
    return value


def arm_guard(successor):
    successor=Path(successor);config=read(successor/'preparation/launch-config.json');spec=config['resume_joint']
    source=Path(spec['source_base']);directory=source/'preparation/calibration-handoff-v1'
    prep=read(directory/'prepare-receipt.json');plan=read(directory/'plan.json')
    sys.path.insert(0,str(successor/'repo'))
    from reproduction.prefix_diversity200.calibration_handoff_guard import once,native,check_native,exact_dependency
    job=str(read(successor/'preparation/submit-receipt.json')['job_id'])
    waiting=native(job);check_native(waiting,successor,config['run_id'],job)
    current=native(str(spec['source_job']));check_native(current,source,config['run_id'],str(spec['source_job']))
    if (str(prep['source_job'])!=str(spec['source_job']) or str(plan['source_job'])!=str(spec['source_job'])
            or config['run_id']!=plan['run_id']
            or current['JobState']!='RUNNING' or waiting['JobState']!='PENDING'
            or not exact_dependency(waiting['Dependency'],str(spec['source_job']))
            or sha(directory/'plan.json')!=spec['planned_handoff_plan_sha256']
            or sha(directory/'guard.py')!=prep['guard_sha256']
            or sha(successor/'preparation/launch-config.json')!=prep['successor_config_sha256']):
        raise ValueError('Exact pending dependency or published guard identity differs')
    if (source/'latest-joint.json').exists() and read(source/'latest-joint.json')['outer']>plan['target_complete_outer']:
        raise ValueError('Planned source boundary already passed; leave the source running')
    arm=dict(plan_sha256=spec['planned_handoff_plan_sha256'],successor_job=job,
        successor_config_sha256=prep['successor_config_sha256'],created_unix=time.time())
    unit='g1-calibration-handoff-'+str(spec['source_job'])+'-to-'+job
    command=['systemd-run','--unit='+unit,'--property=CPUQuota=20%','--property=MemoryMax=128M',
        '--property=Nice=19','--property=TasksMax=32','--property=RuntimeMaxSec=5400',
        '--property=StandardOutput=append:'+str(directory/'unit.log'),
        '--property=StandardError=append:'+str(directory/'unit.log'),
        '--setenv=CUDA_VISIBLE_DEVICES=','/usr/bin/python3',str(directory/'guard.py'),
        '--directory',str(directory)]
    once(directory/'arm.json',arm)
    once(directory/'unit-intent.json',dict(unit=unit,command=command,arm=arm,created_unix=time.time()))
    result=subprocess.run(command,text=True,capture_output=True,timeout=30)
    receipt=dict(unit=unit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        status='ARMED' if result.returncode==0 else 'ARM_OUTCOME_REQUIRES_RECONCILIATION',observed_unix=time.time())
    once(directory/'unit-receipt.json',receipt)
    if result.returncode:raise RuntimeError(json.dumps(receipt))
    return receipt
