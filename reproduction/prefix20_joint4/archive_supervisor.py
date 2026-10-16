"""Held, once-only CPU archive operations; no training/model/GPU calls.

Runs on pinned HPC3 ACD-Login-1, or launches the independent HKUST3 controller.
Every long operation has a persistent public plan, exclusive claim, process
birth, worker stdout/stderr and an actual subprocess.wait() exit receipt.
An SSH observation timeout never authorizes another snapshot/accept writer.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback

try:
    from . import checkpoint_archive as core
except ImportError:
    import checkpoint_archive as core

HK_ROOT = Path('/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016')
HPC_IDENTITY = dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
                    pid_namespace='pid:[4026531836]')
KNOWN_HOSTS = '/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts'
PEER = 'jxu957@10.120.48.31'
BUNDLE_NAMES = ('checkpoint_archive.py', 'archive_supervisor.py', 'archive_controller.py')


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return dict(host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        pid_namespace=os.readlink('/proc/self/ns/pid'), pid=pid, start_ticks=int(fields[19]),
        sid=int(fields[3]), uid=Path('/proc/%d' % pid).stat().st_uid, state=fields[0])


def same_birth(birth, observe=identity):
    try:
        actual = observe(birth['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = ('host', 'boot_id', 'pid_namespace', 'pid', 'start_ticks', 'sid', 'uid')
    return all(actual[k] == birth[k] for k in fields) and actual.get('state') != 'Z'


def source_host():
    actual = identity()
    core.need(actual['uid'] == 14090 and {k: actual[{'hostname':'host'}.get(k, k)] for k in HPC_IDENTITY} == HPC_IDENTITY,
              'SOURCE_PINNED_LOGIN_IDENTITY_REQUIRED')


def bound_command(command):
    """The same strict kernel binding as the modern account bridge."""
    code = ('import json,os,socket,sys\nexpected=json.loads(' + repr(json.dumps(HPC_IDENTITY, sort_keys=True)) + ')\n'
        "with open('/proc/sys/kernel/random/boot_id') as f: boot=f.read().strip()\n"
        "actual=dict(hostname=socket.gethostname(),boot_id=boot,pid_namespace=os.readlink('/proc/self/ns/pid'))\n"
        'if actual!=expected:\n'
        " print(json.dumps(dict(status='UNKNOWN_HOST_IDENTITY_MISMATCH',actual=actual,expected=expected)),file=sys.stderr)\n"
        ' sys.exit(86)\n'
        "os.execv('/bin/bash',['bash','-c'," + repr(command) + '])\n')
    return 'python3 -B -c ' + shlex.quote(code)


def ssh_argv(command):
    return ['sshpass', '-e', 'ssh', '-T', '-oStrictHostKeyChecking=yes', '-oControlMaster=no', '-S', 'none',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn', '-oUserKnownHostsFile='+KNOWN_HOSTS,
        '-oConnectTimeout=8', '-oServerAliveInterval=15', '-oServerAliveCountMax=3', PEER, bound_command(command)]


def atomic(path, value):
    path = Path(path)
    core.safe_components(path.parent)
    temporary = path.with_name(path.name+'.partial-'+str(time.time_ns()))
    with temporary.open('xb') as out:
        out.write(core.document_bytes(value)); out.flush(); os.fsync(out.fileno())
    os.replace(str(temporary), str(path))
    core.fsync_directory(path.parent)


def validate_config(value):
    required = {'schema_version', 'base', 'training_commit', 'source_job', 'code_commit', 'code_manifest',
        'controller_root', 'hpc_identity', 'known_hosts', 'ssh_peer', 'poll_seconds',
        'source_wait_deadline_seconds', 'transport_stall_seconds'}
    core.need(isinstance(value, dict) and set(value) in (required,required|{'recovery_of'}) and value['schema_version'] == 1, 'CONTROLLER_CONFIG_SCHEMA_DIFFERS')
    core.own_base(value['base'])
    for key in ('training_commit', 'code_commit'):
        core.need(isinstance(value[key], str) and re.fullmatch('[0-9a-f]{40}', value[key]) is not None, 'CONTROLLER_FULL_COMMIT_REQUIRED')
    core.need(isinstance(value['source_job'], str) and value['source_job'].isdecimal(), 'CONTROLLER_SOURCE_JOB_REQUIRED')
    expected_root=HK_ROOT/'archive-controller'/('job-'+value['source_job'])
    recovery=value.get('recovery_of')
    if recovery is not None:
        core.need(isinstance(recovery,dict),'CONTROLLER_RECOVERY_BINDING_INVALID')
        collision=recovery.get('reason')=='terminal_archive_progress_timestamp_collision'
        fields=({'reason','old_code_commit','old_config_sha256','old_exit_sha256','old_result_sha256',
                 'source_snapshot_sha256','run_id'} if collision else
                {'reason','old_code_commit','old_config_sha256','old_exit_sha256','old_stop_request_sha256',
                 'source_no_operations_proof_path','source_no_operations_proof_sha256'})
        core.need(set(recovery)==fields and (collision or
            recovery.get('reason')=='source_system_python36_capture_output_rejected') and
            isinstance(recovery['old_code_commit'],str) and re.fullmatch('[0-9a-f]{40}',recovery['old_code_commit']) is not None and
            recovery['old_code_commit']!=value['code_commit'],'CONTROLLER_RECOVERY_BINDING_INVALID')
        for name in (('old_config_sha256','old_exit_sha256','old_result_sha256','source_snapshot_sha256') if collision else
                     ('old_config_sha256','old_exit_sha256','old_stop_request_sha256','source_no_operations_proof_sha256')):
            core.digest_text(recovery[name])
        if collision:core.binding(value['base'],value['training_commit'],recovery['run_id'],5)
        else:
            proof=core.absolute(recovery['source_no_operations_proof_path'])
            core.need(core.under(proof,Path(value['base'])/'preparation/archive-control'),'CONTROLLER_RECOVERY_SOURCE_PROOF_OUTSIDE_BASE')
        expected_root=expected_root/('recovery-'+recovery['old_code_commit'])
    core.need(value['controller_root'] == str(expected_root) and
        value['hpc_identity'] == HPC_IDENTITY and value['known_hosts'] == KNOWN_HOSTS and value['ssh_peer'] == PEER,
        'CONTROLLER_ROUTING_BINDING_DIFFERS')
    core.need(type(value['poll_seconds']) is int and 10 <= value['poll_seconds'] <= 300 and
        type(value['source_wait_deadline_seconds']) is int and 60 <= value['source_wait_deadline_seconds'] <= 14*86400 and
        type(value['transport_stall_seconds']) is int and 60 <= value['transport_stall_seconds'] <= 1800,
        'CONTROLLER_DEADLINES_INVALID')
    manifest = value['code_manifest']
    core.need(isinstance(manifest, dict) and set(manifest) == set(BUNDLE_NAMES), 'CONTROLLER_CODE_FILE_SET_DIFFERS')
    for row in manifest.values():
        core.need(isinstance(row, dict) and set(row) == {'bytes', 'sha256'} and type(row['bytes']) is int and row['bytes'] > 0,
            'CONTROLLER_CODE_FILE_RECORD_INVALID')
        core.digest_text(row['sha256'])
    return value


def source_python(config):
    validate_config(config)
    return str(Path(config['base'])/'env/bin/python')


def source_argv(config,*arguments):
    """Every HPC3 sidecar operation uses the bound training Python, never system3.6."""
    return [source_python(config),'-B']+list(arguments)


def code_root(config, source=False):
    return (Path(config['base'])/'preparation/archive-control/code'/config['code_commit'] if source else
            HK_ROOT/'archive-controller/code'/config['code_commit'])


def verify_bundle(directory, config):
    directory = Path(directory)
    core.safe_components(directory)
    core.exact_tree(directory, list(BUNDLE_NAMES)+['code-publication.json'])
    proof = core._read_json(directory/'code-publication.json')
    core.need(proof == dict(code_commit=config['code_commit'], files=config['code_manifest']), 'SIDECAR_CODE_PUBLICATION_DIFFERS')
    for name, row in config['code_manifest'].items():
        core.checked_file(directory/name, row['sha256'], expected_bytes=row['bytes'])
    return proof


def deploy_bundle(config, contents):
    """HPC3 small sidecar only; never overwrite source/, repo/ or training logs."""
    import base64
    source_host(); validate_config(config)
    root = code_root(config, source=True)
    core.need(isinstance(contents, dict) and set(contents) == set(BUNDLE_NAMES), 'SIDECAR_PAYLOAD_SET_DIFFERS')
    if root.exists():
        return dict(status='EXISTING_SIDECAR_FULL_SHA_PASS', code_root=str(root), publication=verify_bundle(root, config))
    core.safe_components(root.parent, require_exists=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    for name in BUNDLE_NAMES:
        raw = base64.b64decode(contents[name], validate=True)
        row = config['code_manifest'][name]
        core.need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'SIDECAR_PAYLOAD_SHA_DIFFERS')
        with (root/name).open('xb') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
    core.publish(root/'code-publication.json', dict(code_commit=config['code_commit'], files=config['code_manifest']))
    return dict(status='SIDECAR_FULL_SHA_PASS', code_root=str(root), publication=verify_bundle(root, config))


def operation_plan(config, run_id, step, kind, *, snapshot_sha=None, archive_wrapper=None):
    validate_config(config)
    core.binding(config['base'], config['training_commit'], run_id, step)
    core.need(kind in ('snapshot', 'accept'), 'OPERATION_KIND_INVALID')
    value = dict(schema_version=1, config=config, run_id=run_id, step=step, kind=kind,
        snapshot_sha256=snapshot_sha, archive_wrapper=archive_wrapper)
    if kind == 'snapshot':
        core.need(snapshot_sha is None and archive_wrapper is None, 'SNAPSHOT_OPERATION_HAS_ACCEPT_PAYLOAD')
    else:
        core.digest_text(snapshot_sha)
        core.need(isinstance(archive_wrapper, dict), 'ACCEPT_OPERATION_NEEDS_ARCHIVE_RECEIPT')
    return value


def validate_plan(plan):
    core.need(isinstance(plan, dict) and set(plan) == {'schema_version', 'config', 'run_id', 'step', 'kind',
        'snapshot_sha256', 'archive_wrapper'} and plan['schema_version'] == 1, 'OPERATION_PLAN_SCHEMA_DIFFERS')
    core.need(operation_plan(plan['config'], plan['run_id'], plan['step'], plan['kind'],
        snapshot_sha=plan['snapshot_sha256'], archive_wrapper=plan['archive_wrapper']) == plan, 'OPERATION_PLAN_DIFFERS')
    return plan


def operation_root(plan):
    validate_plan(plan)
    return Path(plan['config']['base'])/'preparation/archive-control/operations'/plan['config']['code_commit']/plan['run_id']/(
        'step-%d-%s' % (plan['step'], plan['kind']))


def tail_json(path):
    path = Path(path)
    if not path.exists():
        return None
    core.regular(path)
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END); size=stream.tell(); stream.seek(max(0, size-16384))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def operation_status(plan):
    source_host(); validate_plan(plan)
    root = operation_root(plan)
    source_root = Path(plan['config']['base'])/'preparation/checkpoint-archives'/('global_step_'+str(plan['step']))
    result = dict(operation_root=str(root), kind=plan['kind'], step=plan['step'], run_id=plan['run_id'], checked_unix=time.time(),
        source_snapshot_claim_present=(source_root/'claim.json').exists(),
        source_snapshot_present=(source_root/'source-snapshot.json').exists(),
        source_complete_present=(source_root/'complete.json').exists(),
        source_progress=tail_json(source_root/'source-progress.jsonl'),
        operation_progress=tail_json(root/'source-readback-progress.jsonl') if plan['kind']=='accept' else None)
    claim = root/'claim.json'
    if not claim.exists():
        if root.exists():result['status']='UNKNOWN_PARTIAL_OPERATION_DIRECTORY'
        else:result['status'] = 'UNSTARTED' if not result['source_snapshot_claim_present'] else 'EXISTING_SOURCE_SNAPSHOT_CLAIM_REQUIRES_INSPECTION'
        return result
    core.need(core._read_json(claim)['plan_sha256'] == core.document_sha(plan) and core._read_json(root/'plan.json') == plan,
              'EXISTING_OPERATION_BINDING_DIFFERS')
    completed = root/'operation-result.json'
    if completed.exists():
        value = core._read_json(completed)
        core.need(value['plan_sha256'] == core.document_sha(plan), 'OPERATION_RESULT_BINDING_DIFFERS')
        result.update(status='COMPLETED' if value['worker_returncode'] == 0 and value['accepted_output'] else 'FAILED',
            operation_result=value, operation_result_sha256=core.sha(completed))
        if result['status'] == 'COMPLETED':
            output = root/'worker-output.json'
            core.checked_file(output, value['worker_output_sha256'], expected_bytes=value['worker_output_bytes'])
            result['output'] = core._read_json(output)
        return result
    birth_file = root/'supervisor-birth.json'
    if not birth_file.exists():
        result['status'] = 'CLAIMED_BIRTH_PENDING_OR_UNKNOWN'
        return result
    born = core._read_json(birth_file)
    result.update(supervisor=born, status='RUNNING' if same_birth(born) else 'UNKNOWN_OPERATION_OUTCOME')
    return result


def launch_operation(plan):
    source_host(); validate_plan(plan)
    root = operation_root(plan)
    verify_bundle(code_root(plan['config'], source=True), plan['config'])
    if (root/'claim.json').exists():
        return operation_status(plan)  # Idempotent query; never launch a second writer.
    source_root=Path(plan['config']['base'])/'preparation/checkpoint-archives'/('global_step_'+str(plan['step']))
    if plan['kind']=='snapshot' and (source_root/'claim.json').exists() and not (source_root/'source-snapshot.json').exists():
        return operation_status(plan)
    core.safe_components(root.parent, require_exists=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    core.publish(root/'plan.json', plan)
    core.publish(root/'claim.json', dict(plan_sha256=core.document_sha(plan), launcher=identity(), created_unix=time.time()))
    script = code_root(plan['config'], source=True)/'archive_supervisor.py'
    with (root/'supervisor.log').open('xb') as log:
        child = subprocess.Popen(source_argv(plan['config'],str(script),'supervise','--root',str(root)),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    born = identity(child.pid)  # Bind before poll/wait can reap a rapid failure.
    core.publish(root/'launcher-observed-birth.json', born)
    return dict(status='DISPATCHED_SUPERVISOR', operation_root=str(root), supervisor=born,
                plan_sha256=core.document_sha(plan), checked_unix=time.time())


def supervise(root, *, observe=identity, worker_command=None):
    source_host()
    root = Path(root)
    plan = core._read_json(root/'plan.json'); validate_plan(plan)
    core.need(root == operation_root(plan), 'SUPERVISOR_ROOT_DIFFERS')
    verify_bundle(code_root(plan['config'], source=True), plan['config'])
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core.publish(root/'supervisor-birth.json', observe())
        command = worker_command or source_argv(plan['config'],str(code_root(plan['config'], source=True)/'archive_supervisor.py'),
                                     'operation-worker','--root',str(root))
        started=time.time()
        with (root/'worker.stdout').open('xb') as out, (root/'worker.stderr').open('xb') as error:
            child=subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=error, close_fds=True)
            worker=observe(child.pid); core.publish(root/'worker-birth.json', worker)
            code=child.wait()
            out.flush();error.flush();os.fsync(out.fileno());os.fsync(error.fileno())
        output=root/'worker-output.json';accepted=False;output_sha=None;output_bytes=None;validation_error=None
        if code == 0:
            try:
                value=core._read_json(output)
                accepted = value.get('kind') == plan['kind'] and value.get('run_id') == plan['run_id'] and value.get('step') == plan['step']
                if accepted:output_sha=core.sha(output);output_bytes=output.stat().st_size
            except Exception as error:
                validation_error=type(error).__name__
        result=dict(plan_sha256=core.document_sha(plan), worker=worker, worker_returncode=code,
            accepted_output=bool(accepted), worker_output_sha256=output_sha, worker_output_bytes=output_bytes,
            validation_error=validation_error, stdout_sha256=core.sha(root/'worker.stdout'), stderr_sha256=core.sha(root/'worker.stderr'),
            started_unix=started, checked_unix=time.time(), actual_cuda_restore_accepted=False)
        core.publish(root/'operation-result.json', result)
        return result


def operation_worker(root):
    source_host(); root=Path(root);plan=validate_plan(core._read_json(root/'plan.json'))
    core.need(root == operation_root(plan), 'WORKER_ROOT_DIFFERS')
    verify_bundle(code_root(plan['config'], source=True), plan['config'])
    config=plan['config'];base=Path(config['base']);step=plan['step'];run=plan['run_id'];commit=config['training_commit']
    source_snapshot=base/'preparation/checkpoint-archives'/('global_step_'+str(step))/'source-snapshot.json'
    if plan['kind'] == 'snapshot':
        if source_snapshot.exists():
            value=core._read_json(source_snapshot)
            core.validate_snapshot(value,base=base,commit=commit,run_id=run,step=step)
            result=dict(snapshot_file=str(source_snapshot),snapshot_sha256=core.sha(source_snapshot),snapshot=value,
                adopted_existing_snapshot=True,original_snapshot_checked_unix=value['checked_unix'])
        else:
            core.need(not (source_snapshot.parent/'claim.json').exists(), 'EXISTING_SNAPSHOT_CLAIM_NO_REPLAY')
            result=core.snapshot(base,step,commit=commit,run_id=run)
    else:
        # A complete receipt with no held-worker exit receipt is an unknown
        # interrupted operation. It requires inspection, never another accept.
        core.need(not (source_snapshot.parent/'complete.json').exists(), 'EXISTING_COMPLETE_NO_REPLAY')
        snapshot_value=core.load_snapshot(source_snapshot,plan['snapshot_sha256'],base=base,commit=commit,run_id=run,step=step)
        core.validate_archive_proof(plan['archive_wrapper'],snapshot_value,plan['snapshot_sha256'])
        imported=source_snapshot.parent/'archive-receipt-import.json'
        if imported.exists():
            core.need(core._read_json(imported)==plan['archive_wrapper'] and core.sha(imported)==core.document_sha(plan['archive_wrapper']),
                'EXISTING_IMPORTED_ARCHIVE_WRAPPER_DIFFERS')
        else:core.publish(imported,plan['archive_wrapper'])
        original=core.checked_file;checked_bytes=0;checked_files=0
        with (root/'source-readback-progress.jsonl').open('x') as progress:
            def measured(*args,**kwargs):
                nonlocal checked_bytes,checked_files
                value=original(*args,**kwargs);checked_bytes+=value['bytes'];checked_files+=1
                progress.write(json.dumps(dict(verified_bytes=checked_bytes,verified_files=checked_files,
                    path=str(args[0]),scope='actual successful full SHA reads during accept',checked_unix=time.time()))+'\n')
                progress.flush();os.fsync(progress.fileno());return value
            core.checked_file=measured
            try:
                result=core.accept_archive(base,source_snapshot,plan['snapshot_sha256'],plan['archive_wrapper'],
                    commit=commit,run_id=run,step=step)
            finally:core.checked_file=original
    value=dict(kind=plan['kind'],run_id=run,step=step,result=result,checked_unix=time.time())
    core.publish(root/'worker-output.json',value)
    return value


def stream_file(plan, index, snapshot_sha):
    """Read one exact source file; raw stdout bytes, actual exit0 only on SHA PASS."""
    source_host();validate_plan(plan)
    core.need(plan['kind']=='snapshot' and type(index) is int and index>=0,'STREAM_PLAN_INVALID')
    config=plan['config'];step=plan['step'];run=plan['run_id']
    snapshot_path=Path(config['base'])/'preparation/checkpoint-archives'/('global_step_'+str(step))/'source-snapshot.json'
    value=core.load_snapshot(snapshot_path,snapshot_sha,base=config['base'],commit=config['training_commit'],run_id=run,step=step)
    core.need(index<len(value['files']),'STREAM_INDEX_OUTSIDE_SNAPSHOT')
    row=value['files'][index];before=core.regular(row['source_path'])
    core.need(before==row['source_identity'],'STREAM_SOURCE_IDENTITY_CHANGED')
    descriptor=os.open(row['source_path'],os.O_RDONLY|getattr(os,'O_NOFOLLOW',0));count=0;digest=hashlib.sha256()
    try:
        info=os.fstat(descriptor)
        core.need((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)==tuple(before[k] for k in ('device','inode','bytes','mtime_ns')),
            'STREAM_OPENED_IDENTITY_CHANGED')
        while True:
            block=os.read(descriptor,2**20)
            if not block:break
            sys.stdout.buffer.write(block);digest.update(block);count+=len(block)
        sys.stdout.buffer.flush()
        core.need(count==row['bytes'] and digest.hexdigest()==row['sha256'] and core.regular(row['source_path'])==before,
                  'STREAM_FULL_SHA_OR_IDENTITY_CHANGED')
    finally:
        os.close(descriptor)


def probe_source(config):
    """Bound, small metadata/accounting observation; missing records stay unknown."""
    source_host();validate_config(config);base=Path(config['base']);job=config['source_job']
    core.need((base/'job.id').read_text().strip()==job,'SOURCE_JOB_ID_DIFFERS')
    submitted=core._read_json(base/'preparation/training-submission.json')
    core.need(submitted.get('job')==job and submitted.get('source_commit')==config['training_commit'] and
              submitted.get('returncode')==0,'SOURCE_SUBMISSION_DIFFERS')
    out=dict(source_job=job,base=str(base),training_commit=config['training_commit'],run_id=None,
             pair_receipts={},source_terminal=False,checked_unix=time.time())
    run_path=base/'logs/run-manifest.json';prepared_path=base/'logs/prefix20-configuration-prepared.json'
    if run_path.exists() and prepared_path.exists():
        run=core._read_json(run_path);prepared=core._read_json(prepared_path)
        born=prepared.get('actual_slurm_job_binding',{})
        expected=dict(JobId=job,JobState='RUNNING',Account='jxu957',QOS='formal-user',Partition='emergency_acd',
            UserId='jxu957(14090)',NumNodes='1',NumCPUs='48',JobName='qprobe-prefix20-joint200',
            Command=str(base/'source'/config['training_commit']/'reproduction/prefix20_joint4/train.sh'),WorkDir=str(base/'repo'))
        memory=re.search(r'(?:^|,)mem=([0-9]+)([KMGT])(?:,|$)',born.get('TRES',''))
        memory_bytes=int(memory.group(1))*1024**({'K':1,'M':2,'G':3,'T':4}[memory.group(2)]) if memory else 0
        core.need(all(born.get(k)==v for k,v in expected.items()) and born.get('TresPerNode') in ('gres/gpu:4','gres:gpu:4') and
            memory_bytes>=512*2**30 and run.get('question_probe',{}).get('probe_architecture')==core.ARCHITECTURE and
            run.get('source_commit')==config['training_commit'] and run.get('training_world_size')==4 and
            run.get('data_selection_mode')=='prefix20_probe' and run.get('target_updates')==200 and
            prepared.get('source_commit')==config['training_commit'] and prepared.get('run_id')==run.get('run_id'),
            'SOURCE_PREPARED_NATIVE4_BINDING_DIFFERS')
        run_id=run['run_id'];core.binding(base,config['training_commit'],run_id,5);out['run_id']=run_id
        out['run_manifest_sha256']=core.sha(run_path);out['prepared_sha256']=core.sha(prepared_path)
        out['pair_receipts']={str(step):core.pair_path(base,step).is_file() for step in core.STEPS}
    for name,command in (
        ('queue',['squeue','-h','-u','jxu957,mou176','-o','%i|%T']),
        ('accounting',['sacct','-nP','-j',job,'--format=JobID,State,ExitCode,User,UID,Account,QOS,JobName,Start,End,NodeList'])):
        try:
            value=subprocess.run(command,capture_output=True,text=True,timeout=25)
            out[name]=dict(returncode=value.returncode,stdout=value.stdout,stderr=value.stderr)
        except subprocess.TimeoutExpired:
            out[name]=dict(returncode=None,status='OBSERVATION_TIMEOUT')
    queue=out['queue'];account=out['accounting']
    if queue.get('returncode')==0 and account.get('returncode')==0 and not any(
        line.split('|')[0]==job for line in queue['stdout'].splitlines()):
        rows=[line.split('|') for line in account['stdout'].splitlines()]
        for row in rows:
            if (len(row)>=11 and row[0]==job and row[1] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','COMPLETED') and
                row[3:8]==['jxu957','14090','jxu957','formal-user','qprobe-prefix20-joint200'] and
                any(r[0]==job+'.extern' and r[1:3]==['COMPLETED','0:0'] for r in rows)):
                out['source_terminal']=True;out['source_terminal_row']=row
    return out


def controller_status(config):
    validate_config(config);root=Path(config['controller_root'])
    core.need(socket.gethostname()=='xielab3','CONTROLLER_HKUST3_REQUIRED')
    if not (root/'controller-launch-claim.json').exists():return dict(status='UNSTARTED',controller_root=str(root))
    claim=core._read_json(root/'controller-launch-claim.json')
    core.need(claim['config_sha256']==core.document_sha(config) and core._read_json(root/'config.json')==config,
              'CONTROLLER_EXISTING_BINDING_DIFFERS')
    result=root/'controller-result.json';exit_file=root/'controller-exit.json'
    if exit_file.exists():
        exited=core._read_json(exit_file)
        core.need(exited['config_sha256']==core.document_sha(config),'CONTROLLER_EXIT_BINDING_DIFFERS')
        value=core._read_json(result) if result.exists() else None
        accepted=exited['returncode']==0 and controller_result_accepted(config,value)
        return dict(status='COMPLETED' if accepted else 'FAILED',controller_root=str(root),result=value,
            result_sha256=core.sha(result) if value is not None else None,actual_exit=exited)
    birth=root/'controller-birth.json'
    if not birth.exists():return dict(status='CLAIMED_BIRTH_PENDING_OR_UNKNOWN',controller_root=str(root))
    actual=core._read_json(birth)
    supervisor_file=root/'controller-supervisor-birth.json'
    supervisor=core._read_json(supervisor_file) if supervisor_file.exists() else None
    alive=same_birth(actual) or (supervisor is not None and same_birth(supervisor))
    return dict(status=('FINISHING' if result.exists() else 'RUNNING') if alive else 'UNKNOWN_CONTROLLER_OUTCOME',controller_root=str(root),
                controller=actual,progress=core._read_json(root/'progress.json') if (root/'progress.json').exists() else None)


def controller_result_accepted(config,value):
    """An exit0/marker alone cannot turn an unknown or missing200 run complete."""
    if not isinstance(value,dict):return False
    try:
        core.binding(config['base'],config['training_commit'],value.get('run_id'),200)
    except ValueError:return False
    expected=dict(status='ALL_FIVE_CHECKPOINT_ARCHIVES_ACCEPTED',scope='checkpoint bytes and CPU archive transport only',
        source_job=config['source_job'],base=config['base'],training_commit=config['training_commit'],code_commit=config['code_commit'],
        steps=list(core.STEPS),actual_cuda_restore_accepted=False,final_native51713_accepted=False,training_job_changed=False,
        hpc3_archives_created=0,hpc3_restore_targets_created=0)
    return all(value.get(k)==v and type(value.get(k)) is type(v) for k,v in expected.items())


NO_OPERATIONS_PROGRAM = r'''
import hashlib,json,os,re,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
c=json.load(sys.stdin);b=Path(c['base']);r=c['recovery_of']
assert os.getuid()==14090 and str(b)==str(b.resolve())
assert b.parent==Path('/data/user/jxu957/experiments')
assert re.fullmatch(r'qprobe-teacher-warmup200-prefix20-[A-Za-z0-9._-]+',b.name)
def safe(p):
 assert p.is_absolute() and not any(x.is_symlink() for x in (p,*p.parents))
def read(p):
 safe(p)
 with p.open() as f:return json.load(f)
safe(b)
safe(b/'job.id');assert (b/'job.id').is_file() and (b/'job.id').read_text().strip()==c['source_job']
s=read(b/'preparation/training-submission.json')
assert s['job']==c['source_job'] and s['source_commit']==c['training_commit'] and s['returncode']==0
p=Path(r['source_no_operations_proof_path']);safe(p)
assert p.is_file() and b/'preparation/archive-control' in p.parents
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_no_operations_proof_sha256']
roots=[]
for p in (b/'preparation/archive-control/operations',b/'preparation/checkpoint-archives'):
 safe(p);exists=p.exists()
 assert not exists or (p.is_dir() and next(p.iterdir(),None) is None)
 roots.append(dict(path=str(p),exists=exists,empty=True))
with open('/proc/sys/kernel/random/boot_id') as f:boot=f.read().strip()
actual=dict(hostname=socket.gethostname(),boot_id=boot,pid_namespace=os.readlink('/proc/self/ns/pid'))
assert actual==c['hpc_identity']
print(json.dumps(dict(status='FRESH_SOURCE_NO_ARCHIVE_OPERATIONS_PASS',base=str(b),source_job=c['source_job'],
 training_commit=c['training_commit'],observer_uid=os.getuid(),identity=actual,roots=roots,
 prior_proof_sha256=r['source_no_operations_proof_sha256'],checked_unix=time.time(),run_manifest_may_exist=True)),flush=True)
'''


def fresh_no_source_operations(config, password):
    """Read-only fresh source check. Unknown observation never admits recovery."""
    validate_config(config)
    core.need(config.get('recovery_of') is not None,'RECOVERY_METADATA_REQUIRED')
    core.need(isinstance(password,str) and password and '\n' not in password,'RAM_PASSWORD_REQUIRED')
    environment=dict(os.environ,SSHPASS=password,PYTHONDONTWRITEBYTECODE='1')
    try:
        command=shlex.join(source_argv(config,'-c',NO_OPERATIONS_PROGRAM))
        try:
            result=subprocess.run(ssh_argv(command),input=core.document_bytes(config),env=environment,capture_output=True,timeout=45)
        except subprocess.TimeoutExpired:
            raise ValueError('RECOVERY_SOURCE_OBSERVATION_UNKNOWN_NO_LAUNCH') from None
    finally:environment.pop('SSHPASS',None)
    core.need(result.returncode==0,'RECOVERY_SOURCE_NO_OPERATIONS_OBSERVATION_FAILED_NO_LAUNCH')
    try:proof=json.loads(result.stdout)
    except (ValueError,UnicodeDecodeError):raise ValueError('RECOVERY_SOURCE_PROOF_INVALID') from None
    roots=[str(Path(config['base'])/p) for p in ('preparation/archive-control/operations','preparation/checkpoint-archives')]
    expected=dict(status='FRESH_SOURCE_NO_ARCHIVE_OPERATIONS_PASS',base=config['base'],source_job=config['source_job'],
        training_commit=config['training_commit'],observer_uid=14090,identity=HPC_IDENTITY,
        prior_proof_sha256=config['recovery_of']['source_no_operations_proof_sha256'],run_manifest_may_exist=True)
    core.need(isinstance(proof,dict) and all(proof.get(k)==v and type(proof.get(k)) is type(v) for k,v in expected.items()),
        'RECOVERY_FRESH_SOURCE_BINDING_DIFFERS')
    rows=proof.get('roots');now=time.time()
    core.need(isinstance(rows,list) and len(rows)==2 and all(isinstance(row,dict) and set(row)=={'path','exists','empty'} and
        row['path']==path and type(row['exists']) is bool and row['empty'] is True for row,path in zip(rows,roots)) and
        type(proof.get('checked_unix')) in (int,float) and abs(now-proof['checked_unix'])<=60,'RECOVERY_FRESH_SOURCE_PROOF_DIFFERS')
    return dict(returncode=result.returncode,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),proof=proof)


def completed_snapshot_observation(old, password, run_id):
    """Read the completed old held operation, without another source SHA walk."""
    core.need(isinstance(password,str) and password and '\n' not in password,'RAM_PASSWORD_REQUIRED')
    plan=operation_plan(old,run_id,5,'snapshot')
    command=shlex.join(source_argv(old,str(code_root(old,source=True)/'archive_supervisor.py'),'operation-status'))
    environment=dict(os.environ,SSHPASS=password,PYTHONDONTWRITEBYTECODE='1')
    try:
        try:
            result=subprocess.run(ssh_argv(command),input=core.document_bytes(plan),env=environment,capture_output=True,timeout=45)
        except subprocess.TimeoutExpired:
            raise ValueError('RECOVERY_SOURCE_SNAPSHOT_OBSERVATION_UNKNOWN_NO_LAUNCH') from None
    finally:environment.pop('SSHPASS',None)
    core.need(result.returncode==0,'RECOVERY_SOURCE_SNAPSHOT_OBSERVATION_FAILED_NO_LAUNCH')
    return json.loads(result.stdout)


def _completed_snapshot(observed, old, recovery):
    plan=operation_plan(old,recovery['run_id'],5,'snapshot')
    core.need(observed.get('status')=='COMPLETED' and observed.get('kind')=='snapshot' and
        observed.get('step')==5 and observed.get('run_id')==recovery['run_id'] and
        observed.get('operation_root')==str(operation_root(plan)) and observed.get('source_complete_present') is False,
        'RECOVERY_COMPLETED_SNAPSHOT_OPERATION_REQUIRED')
    result=observed['operation_result'];output=observed['output']
    core.need(result['worker_returncode']==0 and result['accepted_output'] is True and
        result['plan_sha256']==core.document_sha(plan) and core.document_sha(result)==observed['operation_result_sha256'] and
        core.document_sha(output)==result['worker_output_sha256'] and len(core.document_bytes(output))==result['worker_output_bytes'] and
        output['kind']=='snapshot' and output['run_id']==recovery['run_id'] and output['step']==5,
        'RECOVERY_SNAPSHOT_ACTUAL_WORKER_EXIT_OR_OUTPUT_DIFFERS')
    wrapped=output['result'];snapshot=wrapped['snapshot']
    core.need(wrapped['snapshot_sha256']==recovery['source_snapshot_sha256'] and
        core.document_sha(snapshot)==recovery['source_snapshot_sha256'] and wrapped['snapshot_file']==str(
            Path(old['base'])/'preparation/checkpoint-archives/global_step_5/source-snapshot.json'),
        'RECOVERY_EXISTING_SOURCE_SNAPSHOT_SHA_DIFFERS')
    core.validate_snapshot(snapshot,base=old['base'],commit=old['training_commit'],run_id=recovery['run_id'],step=5)
    return snapshot


def _partial_manifest(directory):
    """Bounded small failure evidence only; never read a checkpoint payload."""
    core.safe_components(directory);rows={}
    for path in directory.rglob('*'):
        core.safe_components(path)
        if path.is_dir():continue
        identity=core.regular(path)
        core.need(identity['bytes']<=4*2**20 and len(rows)<32,'RECOVERY_PARTIAL_EVIDENCE_NOT_SMALL')
        rows[str(path.relative_to(directory))]=dict(bytes=identity['bytes'],sha256=core.sha(path))
    core.need(sum(row['bytes'] for row in rows.values())<=16*2**20,'RECOVERY_PARTIAL_EVIDENCE_TOO_LARGE')
    return rows


def _rename_no_replace(source,destination):
    import ctypes
    library=ctypes.CDLL(None,use_errno=True);rename=getattr(library,'renameat2',None)
    core.need(rename is not None,'RECOVERY_ATOMIC_NOREPLACE_RENAME_UNAVAILABLE')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        error=ctypes.get_errno();raise OSError(error,os.strerror(error),str(destination))


def progress_collision_admission(config,password,old,old_root,exited,born,supervisor,source_observer):
    recovery=config['recovery_of'];run_id=recovery['run_id']
    core.checked_file(old_root/'controller-result.json',recovery['old_result_sha256'])
    failed=core._read_json(old_root/'controller-result.json')
    core.need(failed.get('status')=='ARCHIVE_CONTROLLER_INCOMPLETE' and failed.get('error_type')=='TypeError' and
        failed.get('error')=="type object got multiple values for keyword argument 'checked_unix'" and
        failed.get('completed_steps')==[] and failed.get('run_id')==run_id and failed.get('source_job')==config['source_job'] and
        failed.get('training_job_changed') is False,'RECOVERY_TIMESTAMP_COLLISION_FAILURE_REQUIRED')
    bound=core._read_json(old_root/'run-binding.json')
    core.need(all(bound.get(k)==v for k,v in dict(base=config['base'],source_job=config['source_job'],
        training_commit=config['training_commit'],run_id=run_id).items()),'RECOVERY_OLD_RUN_BINDING_DIFFERS')
    recorded=core._read_json(old_root/'step-5-snapshot-operation.json')
    snapshot=_completed_snapshot(recorded,old,recovery)
    fresh=(source_observer or (lambda c,p:completed_snapshot_observation(old,p,run_id)))(config,password)
    core.need(_completed_snapshot(fresh,old,recovery)==snapshot,'RECOVERY_FRESH_SOURCE_SNAPSHOT_DIFFERS')
    directory=core.ARCHIVES/run_id/'global_step_5';preserved=old_root/'failed-step5-transport'
    core.safe_components(preserved,require_exists=False)
    claim_file=old_root/'timestamp-collision-recovery-claim.json'
    core.need(not claim_file.exists() and not preserved.exists(),'RECOVERY_COLLISION_ALREADY_ATTEMPTED_INSPECT_NO_RETRY')
    files=_partial_manifest(directory)
    core.need(files['source-snapshot.json']['sha256']==recovery['source_snapshot_sha256'] and
        not any(name in files for name in ('transport-complete.json','archive-receipt.json','source-acceptance.json')),
        'RECOVERY_PARTIAL_ARCHIVE_SNAPSHOT_OR_COMPLETION_DIFFERS')
    archive_claim=core._read_json(directory/'archive-claim.json')
    core.need(archive_claim['run_id']==run_id and archive_claim['step']==5 and
        archive_claim['source_snapshot_sha256']==recovery['source_snapshot_sha256'] and archive_claim['code_commit']==old['code_commit'],
        'RECOVERY_PARTIAL_ARCHIVE_CLAIM_DIFFERS')
    core.publish(claim_file,dict(old_config_sha256=recovery['old_config_sha256'],old_exit_sha256=recovery['old_exit_sha256'],
        old_result_sha256=recovery['old_result_sha256'],source_snapshot_sha256=recovery['source_snapshot_sha256'],
        original_root=str(directory),preserved_root=str(preserved),files=files,run_id=run_id,checked_unix=time.time()))
    core.need(not same_birth(born) and not same_birth(supervisor),'RECOVERY_OLD_PROCESS_STILL_ALIVE')
    core.need(_partial_manifest(directory)==files,'RECOVERY_PARTIAL_CHANGED_AFTER_CLAIM')
    _rename_no_replace(directory,preserved);core.fsync_directory(directory.parent);core.fsync_directory(old_root)
    core.need(not directory.exists() and _partial_manifest(preserved)==files,'RECOVERY_PRESERVED_PARTIAL_SHA_DIFFERS')
    receipt=old_root/'failed-step5-transport-preservation.json'
    core.publish(receipt,dict(status='FAILED_STEP5_TRANSPORT_PRESERVED_FULL_SHA_PASS',original_root=str(directory),
        preserved_root=str(preserved),files=files,claim_sha256=core.sha(claim_file),checked_unix=time.time()))
    return dict(status='TERMINAL_TIMESTAMP_COLLISION_ARCHIVE_RECOVERY_ADMITTED',run_id=run_id,
        old_controller_root=str(old_root),new_controller_root=config['controller_root'],old_actual_exit=exited,
        old_births_dead=True,preserved_transport_receipt=str(receipt),preserved_transport_sha256=core.sha(receipt),
        source_snapshot_sha256=recovery['source_snapshot_sha256'],source_snapshot_adoption_required=True,
        existing_source_operation_sha256=core.sha(old_root/'step-5-snapshot-operation.json'),
        fresh_source_operation_result_sha256=fresh['operation_result_sha256'],
        training_job_changed=False,checked_unix=time.time())


def recovery_admission(config, password, *, source_observer=None):
    """Admit one proved terminal CPU failure after its two own births died.

    No signal or source mutation occurs here. The run manifest may now exist;
    only archive operation/snapshot namespaces must still be genuinely empty.
    """
    validate_config(config);recovery=config.get('recovery_of')
    if recovery is None:return None
    old_root=HK_ROOT/'archive-controller'/('job-'+config['source_job'])
    core.safe_components(old_root)
    core.checked_file(old_root/'config.json',recovery['old_config_sha256'])
    old=validate_config(core._read_json(old_root/'config.json'))
    core.need(old.get('recovery_of') is None and old['code_commit']==recovery['old_code_commit'] and
        old['controller_root']==str(old_root) and all(old[k]==config[k] for k in ('base','source_job','training_commit')),
        'RECOVERY_OLD_CONFIG_BINDING_DIFFERS')
    verify_bundle(code_root(old),old)
    claim=core._read_json(old_root/'controller-launch-claim.json')
    core.need(claim['config_sha256']==core.document_sha(old),'RECOVERY_OLD_CLAIM_DIFFERS')
    core.checked_file(old_root/'controller-exit.json',recovery['old_exit_sha256'])
    exited=core._read_json(old_root/'controller-exit.json')
    core.need(exited['config_sha256']==core.document_sha(old) and type(exited['returncode']) is int and exited['returncode']!=0,
        'RECOVERY_REQUIRES_BOUND_ACTUAL_NONZERO_EXIT')
    born=core._read_json(old_root/'controller-birth.json')
    supervisor=core._read_json(old_root/'controller-supervisor-birth.json')
    observed=core._read_json(old_root/'launcher-observed-birth.json')
    fields=('host','boot_id','pid_namespace','pid','start_ticks','sid','uid')
    core.need(all(exited['controller'][k]==born[k] and supervisor[k]==observed[k] for k in fields) and
        born['host']==supervisor['host']=='xielab3' and born['uid']==supervisor['uid']==os.getuid(),
        'RECOVERY_OLD_PROCESS_BIRTH_BINDING_DIFFERS')
    core.need(not same_birth(born) and not same_birth(supervisor),'RECOVERY_OLD_PROCESS_STILL_ALIVE')
    core.checked_file(old_root/'controller.log',exited['stdout_sha256'])
    if recovery['reason']=='terminal_archive_progress_timestamp_collision':
        return progress_collision_admission(config,password,old,old_root,exited,born,supervisor,source_observer)
    core.checked_file(old_root/'failed-controller-stop-request.json',recovery['old_stop_request_sha256'])
    core.need(not (old_root/'run-binding.json').exists() and not list(old_root.glob('step-*-operation.json')),
        'RECOVERY_OLD_CONTROLLER_HAS_RUN_OR_OPERATIONS')
    progress=core._read_json(old_root/'progress.json')
    core.need(progress.get('run_id') is None and progress.get('completed_steps')==[], 'RECOVERY_OLD_PROGRESS_NOT_EMPTY')
    errors=old_root/'observation-errors.jsonl';core.regular(errors)
    with errors.open('rb') as stream:
        stream.seek(0,os.SEEK_END);end=stream.tell();stream.seek(max(0,end-65536));tail=stream.read()
    known=False
    for line in tail.splitlines():
        try:error=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if (isinstance(error,dict) and error.get('mode')=='probe' and type(error.get('returncode')) is int and
            error['returncode']!=0 and isinstance(error.get('stderr'),str) and 'TypeError' in error['stderr'] and
            'capture_output' in error['stderr']):known=True
    core.need(known,'RECOVERY_KNOWN_PYTHON36_FAILURE_NOT_PROVED')
    source=(source_observer or fresh_no_source_operations)(config,password)
    return dict(status='FAILED_PYTHON36_CPU_CONTROLLER_RECOVERY_ADMITTED',old_controller_root=str(old_root),
        old_code_commit=old['code_commit'],old_config_sha256=recovery['old_config_sha256'],
        old_exit_sha256=recovery['old_exit_sha256'],old_stop_request_sha256=recovery['old_stop_request_sha256'],
        old_actual_exit=exited,old_controller_birth=born,old_supervisor_birth=supervisor,old_births_dead=True,
        fresh_source_no_operations=source,new_controller_root=config['controller_root'],
        training_job_changed=False,checked_unix=time.time())


def launch_controller(config, password):
    core.need(socket.gethostname()=='xielab3','CONTROLLER_HKUST3_REQUIRED');validate_config(config)
    root=Path(config['controller_root']);verify_bundle(code_root(config),config)
    if (root/'controller-launch-claim.json').exists():return controller_status(config)
    admitted=recovery_admission(config,password)
    core.safe_components(root.parent,require_exists=False);root.parent.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=False)
    if admitted is not None:core.publish(root/'recovery-admission.json',admitted)
    core.publish(root/'config.json',config)
    core.publish(root/'controller-launch-claim.json',dict(config_sha256=core.document_sha(config),launcher=identity(),created_unix=time.time()))
    core.need(isinstance(password,str) and password and '\n' not in password,'RAM_PASSWORD_REQUIRED')
    with (root/'controller-supervisor.log').open('xb') as out:
        child=subprocess.Popen([sys.executable,'-B',str(code_root(config)/'archive_supervisor.py'),'supervise-controller','--root',str(root)],
            stdin=subprocess.PIPE,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    born=identity(child.pid);core.publish(root/'launcher-observed-birth.json',born)
    child.stdin.write((json.dumps(dict(password=password))+'\n').encode());child.stdin.flush();child.stdin.close()
    return dict(status='DISPATCHED_CONTROLLER_SUPERVISOR',controller_root=str(root),controller_supervisor=born,
                credentials_persisted=False,training_job_changed=False,checked_unix=time.time())


def supervise_controller(root):
    core.need(socket.gethostname()=='xielab3','CONTROLLER_HKUST3_REQUIRED')
    root=Path(root);config=validate_config(core._read_json(root/'config.json'))
    core.need(root==Path(config['controller_root']),'CONTROLLER_SUPERVISOR_ROOT_DIFFERS')
    verify_bundle(code_root(config),config)
    secret=json.load(sys.stdin);password=secret.pop('password');secret=None
    core.need(isinstance(password,str) and password and '\n' not in password,'RAM_PASSWORD_REQUIRED')
    with (root/'controller-supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.publish(root/'controller-supervisor-birth.json',identity())
        started=time.time()
        with (root/'controller.log').open('xb') as out:
            child=subprocess.Popen([sys.executable,'-B',str(code_root(config)/'archive_controller.py'),'--config',str(root/'config.json')],
                stdin=subprocess.PIPE,stdout=out,stderr=subprocess.STDOUT,close_fds=True)
            born=identity(child.pid);core.publish(root/'supervisor-observed-controller-birth.json',born)
            child.stdin.write((json.dumps(dict(password=password))+'\n').encode());child.stdin.flush();child.stdin.close();password=None
            code=child.wait();out.flush();os.fsync(out.fileno())
        value=dict(config_sha256=core.document_sha(config),controller=born,returncode=code,
            stdout_sha256=core.sha(root/'controller.log'),started_unix=started,checked_unix=time.time(),credentials_persisted=False)
        core.publish(root/'controller-exit.json',value)
        return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('probe','launch-operation','operation-status','supervise','supervise-controller','operation-worker','stream'))
    parser.add_argument('--root');parser.add_argument('--index',type=int);parser.add_argument('--snapshot-sha');args=parser.parse_args()
    if args.mode=='supervise':value=supervise(args.root)
    elif args.mode=='supervise-controller':value=supervise_controller(args.root)
    elif args.mode=='operation-worker':value=operation_worker(args.root)
    else:
        payload=json.load(sys.stdin)
        if args.mode=='probe':value=probe_source(payload)
        elif args.mode=='launch-operation':value=launch_operation(payload)
        elif args.mode=='operation-status':value=operation_status(payload)
        else:stream_file(payload,args.index,args.snapshot_sha);return
    print(json.dumps(value,allow_nan=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # No credentials enter this source-side program or its durable plans.
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
