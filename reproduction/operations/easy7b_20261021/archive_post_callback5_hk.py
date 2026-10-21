"""One-shot HK CPU supplement for job745371 CP5; credentials enter stdin only.

Run as a saved file on HKUST3. This script does not detach itself or submit any
job. It never changes the healthy controller, BASE/repo, or the083 source pack.
"""
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
import uuid


HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
BOOTSTRAP_COMMIT = '0830581f16eee86d5a89100cda91b175de63635c'
BOOTSTRAP = HK/'source'/BOOTSTRAP_COMMIT
MODULE_COMMIT = 'b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d'
MODULE_SHA = '082201e54ab06626e877ea66e70f881ff9f6e69cc81a2eb03b6e98ced51b2c99'
MODULE_BYTES = 20774
MODULE_NAME = 'reproduction.easy2560_joint4.post_callback_archive'
TRAINING = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
JOB = '745371'
RUN = '232f20e112ac4a978c5aba52a0367195'
BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
STEP = 5
CORE = HK/'checkpoints'/RUN/'global_step_5'
CONTROL = HK/'post-callback-boundary-control'/('job-'+JOB)/'global_step_5'
HK_TOOL = HK/'post-callback-boundary-tools'/MODULE_COMMIT/'post_callback_archive.py'
HPC_TOOL = BASE/'preparation/post-callback-tools'/MODULE_COMMIT/'post_callback_archive.py'
SOURCE_SNAPSHOT = BASE/'preparation/post-callback-archives/global_step_5/source-snapshot.json'
HEALTHY_CLAIM = HK/'archive-controller'/('job-'+JOB)/'recovery-transport-r1-20261021/claim.json'
MAX_ENVELOPE = 256 * 1024


def need(condition, message):
    if not condition:
        raise ValueError('POST_CALLBACK5_ONCE_' + message)


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'CANONICAL_PATH_REQUIRED')
    return path


def sha(path):
    path = canonical(path); need(path.is_file(), 'REGULAR_FILE_REQUIRED')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def read_json(path, maximum=16*1024**2):
    path = canonical(path)
    need(path.is_file() and path.stat().st_size <= maximum, 'SMALL_JSON_REQUIRED')
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def fsync_directory(path):
    fd = os.open(str(canonical(path)), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_json(path, value):
    path = canonical(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def validate_envelope(value):
    need(isinstance(value, dict) and set(value) ==
         {'password', 'module_source', 'module_sha256', 'published_commit'}, 'ENVELOPE_FIELDS_DIFFER')
    password = value['password']
    need(isinstance(password, str) and 0 < len(password) <= 4096 and
         '\n' not in password and '\x00' not in password, 'RAM_PASSWORD_REQUIRED')
    source = value['module_source']
    need(isinstance(source, str), 'MODULE_SOURCE_REQUIRED')
    raw = source.encode('utf-8')
    need(len(raw) == MODULE_BYTES and value['published_commit'] == MODULE_COMMIT and
         value['module_sha256'] == MODULE_SHA == hashlib.sha256(raw).hexdigest(), 'PUBLISHED_MODULE_SHA_DIFFERS')
    return password, raw


def install_immutable(path, raw):
    path = canonical(path); path.parent.mkdir(parents=True, exist_ok=True)
    canonical(path.parent)
    if path.exists():
        need(path.stat().st_size == len(raw) and sha(path) == MODULE_SHA, 'EXISTING_TOOL_DIFFERS')
        return dict(path=str(path), bytes=len(raw), sha256=MODULE_SHA, reused=True)
    temporary = path.with_name('.'+path.name+'.partial-'+uuid.uuid4().hex)
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    need(sha(temporary) == MODULE_SHA, 'TOOL_WRITE_SHA_DIFFERS')
    os.link(str(temporary), str(path)); temporary.unlink(); fsync_directory(path.parent)
    need(sha(path) == MODULE_SHA, 'TOOL_READBACK_SHA_DIFFERS')
    return dict(path=str(path), bytes=len(raw), sha256=MODULE_SHA, reused=False)


def bootstrap():
    """Import only from the existing immutable083 package and verify its map."""
    publication = read_json(BOOTSTRAP/'source-publication.json')
    need(publication.get('source_commit') == BOOTSTRAP_COMMIT, 'BOOTSTRAP_PUBLICATION_DIFFERS')
    names = ('reproduction/easy2560_joint4/checkpoint_archive.py',
             'reproduction/easy2560_joint4/control.py',
             'reproduction/prefix20_joint4/archive_supervisor.py')
    for name in names:
        need(sha(BOOTSTRAP/name) == publication['files'][name]['sha256'], 'BOOTSTRAP_SOURCE_SHA_DIFFERS')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BOOTSTRAP))
    package = importlib.import_module('reproduction.easy2560_joint4')
    need(Path(package.__file__).resolve() == BOOTSTRAP/'reproduction/easy2560_joint4/__init__.py',
         'BOOTSTRAP_PACKAGE_SHADOWED')
    supervisor = importlib.import_module('reproduction.prefix20_joint4.archive_supervisor')
    need(Path(supervisor.__file__).resolve() == BOOTSTRAP/'reproduction/prefix20_joint4/archive_supervisor.py' and
         supervisor.PEER == 'jxu957@10.120.48.31' and supervisor.HPC_IDENTITY['hostname'] == 'ACD-Login-1',
         'PINNED_SSH_ENTRY_DIFFERS')
    return supervisor.ssh_argv


def ready_core():
    """Only existing small receipts are read; no core payload is modified."""
    source = read_json(CORE/'source-snapshot.json')
    accepted = read_json(CORE/'source-acceptance.json')
    received = read_json(CORE/'archive-receipt.json')
    complete = read_json(CORE/'checkpoint/complete.json')
    need(source.get('role') == 'easy2560_full_checkpoint_source_snapshot' and
         source.get('training_world_size') == 4 and source.get('question_probe_saved') is False,
         'NATIVE4_CORE_REQUIRED')
    for value in (source, accepted, received):
        need(all(value.get(k) == v for k, v in dict(base=str(BASE), run_id=RUN,
            source_commit=TRAINING, step=STEP).items()), 'FIXED_CORE_IDENTITY_DIFFERS')
    need(accepted.get('status') == 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS' and
         received.get('status') == 'EASY2560_ARCHIVE_RECEIVED_FRESH_SHA_FSYNC_PASS' and
         accepted.get('fresh_source_after_archive_sha_verified') is True and
         all(value.get('fresh_destination_sha_verified') is True and value.get('destination_fsynced') is True and
             value.get('actual_cuda_restore_accepted') is False for value in (accepted, received)),
         'CORE_NOT_ACTUALLY_ACCEPTED')
    snapshot_sha = sha(CORE/'source-snapshot.json')
    need(all(value.get('snapshot_sha256') == snapshot_sha and value.get('file_count') == len(source['files']) and
             value.get('total_bytes') == source['total_bytes'] for value in (accepted, received)) and
         complete.get('run_id') == RUN and complete.get('step') == STEP and
         complete.get('finite_all_saved_tensors') is True and
         sha(CORE/'checkpoint/complete.json') == source['files']['checkpoint/complete.json']['sha256'],
         'CORE_RECEIPT_OR_COMPLETE_SHA_DIFFERS')
    claim = read_json(HEALTHY_CLAIM)
    need(claim.get('base') == str(BASE) and claim.get('source_job') == JOB and
         claim.get('source_commit') == TRAINING, 'SOURCE_JOB_CLAIM_DIFFERS')
    return dict(source_snapshot_sha256=snapshot_sha, source_acceptance_sha256=sha(CORE/'source-acceptance.json'),
        archive_receipt_sha256=sha(CORE/'archive-receipt.json'),
        checkpoint_complete_sha256=sha(CORE/'checkpoint/complete.json'),
        existing_controller_claim_sha256=sha(HEALTHY_CLAIM))


def load_module():
    need(sha(HK_TOOL) == MODULE_SHA, 'HK_MODULE_SHA_DIFFERS')
    need(MODULE_NAME not in sys.modules, 'MODULE_ALREADY_LOADED')
    spec = importlib.util.spec_from_file_location(MODULE_NAME, str(HK_TOOL))
    need(spec is not None and spec.loader is not None, 'MODULE_IMPORT_SPEC_REQUIRED')
    module = importlib.util.module_from_spec(spec); sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    need(Path(module.__file__).resolve() == HK_TOOL and
         Path(module.core.__file__).resolve() == BOOTSTRAP/'reproduction/easy2560_joint4/checkpoint_archive.py',
         'MODULE_PACKAGE_SHADOWED')
    return module


def remote_install_code():
    """No secret is interpolated; module bytes arrive on this command's stdin."""
    return '''import hashlib,json,os,socket,stat,sys,uuid
from pathlib import Path
p=json.load(sys.stdin)
base=Path(%r); target=Path(%r)
assert os.getuid()==14090 and socket.gethostname()=='ACD-Login-1'
assert set(p)=={'module_source','module_sha256','published_commit'}
raw=p['module_source'].encode('utf-8')
assert p['published_commit']==%r and p['module_sha256']==%r
assert len(raw)==%d and hashlib.sha256(raw).hexdigest()==p['module_sha256']
run=json.loads((base/'logs/run-manifest.json').read_text())
allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
assert run['run_id']==%r and run['source_commit']==%r and run['training_world_size']==4
assert run['easy2560_selection'] is True and run.get('question_probe') is None
assert allocation['run_id']==run['run_id'] and allocation['slurm_job_id']==%r and allocation['accepted'] is True
assert target.is_absolute() and target.resolve()==target and not target.is_symlink()
target.parent.mkdir(parents=True,exist_ok=True)
assert target.parent.resolve()==target.parent
reused=target.exists()
if reused:
 assert target.is_file() and target.stat().st_size==len(raw) and hashlib.sha256(target.read_bytes()).hexdigest()==p['module_sha256']
else:
 tmp=target.with_name('.'+target.name+'.partial-'+uuid.uuid4().hex)
 fd=os.open(str(tmp),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as stream:
  stream.write(raw);stream.flush();os.fsync(stream.fileno())
 assert hashlib.sha256(tmp.read_bytes()).hexdigest()==p['module_sha256']
 os.link(str(tmp),str(target));tmp.unlink()
 fd=os.open(str(target.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
assert hashlib.sha256(target.read_bytes()).hexdigest()==p['module_sha256']
print(json.dumps(dict(path=str(target),sha256=p['module_sha256'],bytes=len(raw),reused=reused,
 source_job=allocation['slurm_job_id'],run_manifest_sha256=hashlib.sha256((base/'logs/run-manifest.json').read_bytes()).hexdigest(),
 allocation_sha256=hashlib.sha256((base/'logs/native-allocation-admission.json').read_bytes()).hexdigest())))
''' % (str(BASE), str(HPC_TOOL), MODULE_COMMIT, MODULE_SHA, MODULE_BYTES, RUN, TRAINING, JOB)


def remote_module_code(operation):
    need(operation in ('snapshot', 'emit'), 'READ_ONLY_SOURCE_OPERATION_REQUIRED')
    common = '''import hashlib,importlib,importlib.util,json,sys
from pathlib import Path
base=Path(%r); tool=Path(%r)
assert tool.resolve()==tool and not tool.is_symlink() and tool.is_file()
assert tool.stat().st_size==%d and hashlib.sha256(tool.read_bytes()).hexdigest()==%r
sys.dont_write_bytecode=True
sys.path.insert(0,str(base/'repo'))
package=importlib.import_module('reproduction.easy2560_joint4')
assert Path(package.__file__).resolve()==base/'repo/reproduction/easy2560_joint4/__init__.py'
name=%r
assert name not in sys.modules
spec=importlib.util.spec_from_file_location(name,str(tool))
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
assert Path(module.core.__file__).resolve()==base/'repo/reproduction/easy2560_joint4/checkpoint_archive.py'
''' % (str(BASE), str(HPC_TOOL), MODULE_BYTES, MODULE_SHA, MODULE_NAME)
    if operation == 'snapshot':
        return common + 'print(json.dumps(module.snapshot(base,5,%r),sort_keys=True))\n' % str(CORE)
    return common + 'module.emit(%r,sys.stdout.buffer)\n' % str(SOURCE_SNAPSHOT)


def remote_command(code):
    return ('CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ' +
            shlex.join([str(BASE/'env/bin/python'), '-B', '-c', code]))


def redact(value, password):
    text = str(value)
    return text.replace(password, '[REDACTED]') if password else text


def checked_json(ssh_argv, code, env, *, payload=None):
    command = ssh_argv(remote_command(code))
    kwargs = dict(env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    if payload is None: kwargs['stdin'] = subprocess.DEVNULL
    else: kwargs['input'] = json.dumps(payload)
    result = subprocess.run(command, **kwargs)
    need(result.returncode == 0, 'SOURCE_COMMAND_FAILED_RC_%d: %s' %
         (result.returncode, redact(result.stderr[-2000:], env.get('SSHPASS'))))
    need(len(result.stdout.encode()) <= 64*1024, 'SOURCE_RESULT_TOO_LARGE')
    return json.loads(result.stdout)


def run(envelope):
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    canonical(Path(__file__))
    password, raw = validate_envelope(envelope)
    envelope.pop('password'); envelope.pop('module_source')
    ssh_argv = bootstrap()
    before = ready_core()  # No writes or claims before the actual core ACK.
    CONTROL.mkdir(parents=True, exist_ok=True); canonical(CONTROL)
    lock_fd = os.open(str(CONTROL/'operation.lock'), os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    env = None; child = None; lock = os.fdopen(lock_fd, 'a')
    with lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((CONTROL/name).exists() for name in ('claim.json', 'result.json', 'failure.json')),
             'ONCE_CLAIM_EXISTS_INSPECT_NO_REPLAY')
        birth = dict(pid=os.getpid(), uid=os.getuid(), hostname=socket.gethostname(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        write_json(CONTROL/'claim.json', dict(base=str(BASE), source_job=JOB, run_id=RUN, step=STEP,
            training_commit=TRAINING, module_commit=MODULE_COMMIT, module_sha256=MODULE_SHA,
            bootstrap_commit=BOOTSTRAP_COMMIT, core_before=before, birth=birth,
            runner_source_sha256=sha(Path(__file__)), credentials_persisted=False,
            training_runtime_modified=False, jobs_submitted=0, checked_unix=time.time()))
        try:
            hk_install = install_immutable(HK_TOOL, raw)
            write_json(CONTROL/'hk-tool.json', hk_install)
            module = load_module()
            binding = module.core_binding(CORE)
            need(binding['base'] == str(BASE) and binding['run_id'] == RUN and binding['step'] == STEP and
                 binding['source_commit'] == TRAINING, 'LOADED_CORE_BINDING_DIFFERS')
            env = dict(os.environ, SSHPASS=password, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            payload = dict(module_source=raw.decode('utf-8'), module_sha256=MODULE_SHA, published_commit=MODULE_COMMIT)
            installed = checked_json(ssh_argv, remote_install_code(), env, payload=payload)
            need(installed.get('path') == str(HPC_TOOL) and installed.get('sha256') == MODULE_SHA and
                 installed.get('bytes') == MODULE_BYTES and installed.get('source_job') == JOB, 'HPC_TOOL_RESULT_DIFFERS')
            write_json(CONTROL/'hpc-tool.json', installed)
            saved = checked_json(ssh_argv, remote_module_code('snapshot'), env)
            need(saved.get('snapshot') == str(SOURCE_SNAPSHOT) and saved.get('file_count') == 8 and
                 0 < saved.get('total_bytes', 0) <= 256*1024**2 and
                 saved.get('phase', {}).get('sampling_rng_stage') == 'post_QA5', 'SOURCE_SNAPSHOT_RESULT_DIFFERS')
            write_json(CONTROL/'source-snapshot-result.json', saved)
            with (CONTROL/'emit.stderr.log').open('xb') as error:
                child = subprocess.Popen(ssh_argv(remote_command(remote_module_code('emit'))), env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=error)
                transfer = module.receive(CORE, child.stdout)
                child.stdout.close()
                rc = child.wait(timeout=120)
            write_json(CONTROL/'transfer.json', dict(transfer, emit_returncode=rc,
                source_emit_completed=rc == 0, checked_unix=time.time()))
            need(rc == 0, 'SOURCE_EMIT_EXIT_NONZERO_%d' % rc)
            need(ready_core() == before and module.core_binding(CORE) == binding, 'CORE_CHANGED_DURING_SUPPLEMENT')
            receipt = module.accept(CORE, transfer['stage'])
            need(receipt.get('status') == module.STATUS and receipt.get('binding') == binding and
                 receipt.get('file_count') == 8 and receipt.get('actual_cuda_restore_accepted') is False,
                 'SUPPLEMENT_ACCEPTANCE_DIFFERS')
            result = dict(status='EASY7B_CP5_POST_CALLBACK_ARCHIVE_ONCE_COMPLETE', base=str(BASE), source_job=JOB,
                run_id=RUN, step=STEP, training_commit=TRAINING, module_commit=MODULE_COMMIT,
                module_sha256=MODULE_SHA, core_before=before, core_after=ready_core(),
                supplementary_receipt=receipt, emit_returncode=0, credentials_persisted=False,
                training_runtime_modified=False, healthy_controller_modified=False,
                jobs_submitted=0, checked_unix=time.time())
            write_json(CONTROL/'result.json', result)
            print(json.dumps(result, sort_keys=True), flush=True)
            return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)
            write_json(CONTROL/'failure.json', dict(error_type=type(error).__name__,
                error=redact(error, password)[-2000:], source_job=JOB, step=STEP,
                credentials_persisted=False, staging_preserved=True,
                training_runtime_modified=False, jobs_submitted=0, checked_unix=time.time()))
            raise RuntimeError('POST_CALLBACK5_ONCE_FAILED_INSPECT_PRESERVED_FAILURE') from None
        finally:
            if env is not None: env.pop('SSHPASS', None)
            password = None


def main():
    raw = sys.stdin.buffer.read(MAX_ENVELOPE+1)
    need(0 < len(raw) <= MAX_ENVELOPE, 'SMALL_STDIN_ENVELOPE_REQUIRED')
    envelope = json.loads(raw); raw = None
    run(envelope)


if __name__ == '__main__': main()
