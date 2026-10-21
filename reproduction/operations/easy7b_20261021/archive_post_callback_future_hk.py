"""Independent CPU companion for the four remaining Easy7B callback archives.

The core controller and the completed CP5 supplement are read-only dependencies.
Each milestone has one writer claim; a failed or interrupted writer is never
automatically replayed. The eight state files and grounding closure have separate
receipts. Neither receipt represents an actual CUDA restore.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid


HELPER_COMMIT = '9a5025a2471c5a3cf4476a0f5981e1c99af8dc06'
HELPER_SHA = 'a2a32b86f30d3864187c1a9e94a2b96252d6fd7e532f54498896a7d614b9e2aa'
HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
HELPER = HK/'post-callback-boundary-launch-tools'/HELPER_COMMIT/'archive_post_callback5_hk.py'
ROOT = HK/'post-callback-boundary-control/job-745371/future-50-200'
CONTROLLER = HK/'archive-controller/job-745371/recovery-transport-r1-20261021'
STEPS = (50, 100, 150, 200)
DEPENDENCIES_NAME = 'reproduction.easy2560_joint4.post_callback_dependencies'
MAX_ENVELOPE = 384*1024


def need(condition, message):
    if not condition: raise ValueError('POST_CALLBACK_FUTURE_' + message)


def frozen_helper():
    need(HELPER.is_absolute() and HELPER.resolve() == HELPER and not HELPER.is_symlink(), 'HELPER_CANONICAL_REQUIRED')
    raw = HELPER.read_bytes()
    need(len(raw) == 18957 and hashlib.sha256(raw).hexdigest() == HELPER_SHA, 'FROZEN_HELPER_SHA_DIFFERS')
    # This is a private module instance. No existing module or file is mutated.
    spec = importlib.util.spec_from_file_location('_future_private_frozen_cp5_' + uuid.uuid4().hex, str(HELPER))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    need(module.JOB == '745371' and module.RUN == '232f20e112ac4a978c5aba52a0367195' and
         module.MODULE_COMMIT == 'b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d', 'FROZEN_HELPER_IDENTITY_DIFFERS')
    return module


def configure_step(helper, step):
    need(step in STEPS, 'FUTURE_MILESTONE_REQUIRED')
    helper.STEP = step
    helper.CORE = HK/'checkpoints'/helper.RUN/('global_step_' + str(step))
    helper.CONTROL = ROOT/('global_step_' + str(step))
    helper.SOURCE_SNAPSHOT = helper.BASE/'preparation/post-callback-archives'/('global_step_' + str(step))/'source-snapshot.json'


def validate_envelope(helper, envelope):
    old = {'password', 'module_source', 'module_sha256', 'published_commit'}
    need(isinstance(envelope, dict) and set(envelope) == old | {
        'dependencies_source', 'dependencies_sha256', 'dependencies_commit'}, 'ENVELOPE_FIELDS_DIFFER')
    password, boundary_raw = helper.validate_envelope({key: envelope[key] for key in old})
    value = envelope['dependencies_source']
    need(isinstance(value, str), 'DEPENDENCIES_SOURCE_REQUIRED')
    raw = value.encode('utf-8'); commit = envelope['dependencies_commit']; digest = envelope['dependencies_sha256']
    need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit) is not None and
         isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None and
         0 < len(raw) <= 128*1024 and hashlib.sha256(raw).hexdigest() == digest,
         'PUBLISHED_DEPENDENCIES_SHA_DIFFERS')
    return password, boundary_raw, dict(raw=raw, commit=commit, sha256=digest,
        hk_tool=HK/'post-callback-dependency-tools'/commit/'post_callback_dependencies.py',
        hpc_tool=helper.BASE/'preparation/post-callback-tools'/commit/'post_callback_dependencies.py')


def process_observation(birth):
    """A known missing/reused/Z birth is terminal; unreadable /proc is unknown."""
    try:
        process = Path('/proc')/str(birth['pid'])
        fields = (process/'stat').read_text().rsplit(')', 1)[1].split()
        actual = dict(pid=birth['pid'], start_ticks=int(fields[19]), state=fields[0], uid=process.stat().st_uid,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        same = all(actual[key] == birth[key] for key in ('pid', 'start_ticks', 'uid', 'boot_id'))
        return dict(state='LIVE' if same and actual['state'] != 'Z' else 'TERMINAL', actual=actual, same_birth=same)
    except (FileNotFoundError, ProcessLookupError):
        return dict(state='TERMINAL', actual=None, same_birth=False)
    except OSError as error:
        return dict(state='UNKNOWN', actual=None, error_type=type(error).__name__)


def current_birth():
    return dict(pid=os.getpid(), uid=os.getuid(),
        start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def controller_observation(helper):
    records = {}
    for name in ('claim.json', 'progress.json', 'failure.json', 'result.json'):
        path = CONTROLLER/name
        if path.exists(): records[name] = helper.read_json(path)
    claim = records.get('claim.json')
    if claim is None: return dict(state='UNKNOWN', records=records, reason='CLAIM_NOT_OBSERVED')
    need(claim.get('base') == str(helper.BASE) and claim.get('source_job') == helper.JOB and
         claim.get('source_commit') == helper.TRAINING, 'CORE_CONTROLLER_IDENTITY_DIFFERS')
    birth = claim.get('birth')
    need(isinstance(birth, dict), 'CORE_CONTROLLER_BIRTH_REQUIRED')
    process = process_observation(birth)
    terminal = 'failure.json' in records or 'result.json' in records or process['state'] == 'TERMINAL'
    return dict(state='TERMINAL' if terminal else process['state'], birth=birth, process=process, records=records)


def progress(helper, **values):
    value = dict(values, source_job=helper.JOB, run_id=helper.RUN, training_commit=helper.TRAINING,
        training_modified=False, healthy_controller_modified=False, jobs_submitted=0, checked_unix=time.time())
    temporary = ROOT/('.progress.partial-' + uuid.uuid4().hex)
    helper.write_json(temporary, value); os.replace(temporary, ROOT/'progress.json'); helper.fsync_directory(ROOT)
    print(json.dumps(value, sort_keys=True), flush=True)


def wait_for_core(helper, completed, poll_seconds=60, sleep=time.sleep):
    """Wait locally for all actual core ACK files before any source SSH."""
    names = ('source-snapshot.json', 'source-acceptance.json', 'archive-receipt.json', 'checkpoint/complete.json')
    while True:
        observation = controller_observation(helper)
        if all((helper.CORE/name).is_file() for name in names):
            return helper.ready_core(), observation
        if observation['state'] == 'TERMINAL':
            progress(helper, phase='DEPENDENCY_TERMINAL', step=helper.STEP, completed_steps=completed,
                     core_controller=observation, training_failure_inferred=False)
            raise RuntimeError('DEPENDENCY_TERMINAL_CORE_CONTROLLER_BEFORE_ACTUAL_ACK')
        progress(helper, phase='WAITING_CORE_ACTUAL_ACK', step=helper.STEP, completed_steps=completed,
                 core_controller=observation, source_ssh_called=False)
        sleep(poll_seconds)


def install_dependency(helper, data):
    path = helper.canonical(data['hk_tool']); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.stat().st_size == len(data['raw']) and helper.sha(path) == data['sha256'], 'EXISTING_DEPENDENCY_TOOL_DIFFERS')
    else:
        tmp = path.with_name('.' + path.name + '.partial-' + uuid.uuid4().hex)
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        with os.fdopen(fd, 'wb') as stream: stream.write(data['raw']); stream.flush(); os.fsync(stream.fileno())
        need(helper.sha(tmp) == data['sha256'], 'DEPENDENCY_WRITE_SHA_DIFFERS')
        os.link(tmp, path); tmp.unlink(); helper.fsync_directory(path.parent)
    return dict(path=str(path), bytes=len(data['raw']), sha256=helper.sha(path), source_commit=data['commit'])


def load_dependencies(helper, data):
    need(helper.sha(data['hk_tool']) == data['sha256'] and DEPENDENCIES_NAME not in sys.modules, 'DEPENDENCY_IMPORT_DIFFERS')
    spec = importlib.util.spec_from_file_location(DEPENDENCIES_NAME, str(data['hk_tool']))
    module = importlib.util.module_from_spec(spec); sys.modules[DEPENDENCIES_NAME] = module; spec.loader.exec_module(module)
    return module


def dependency_install_code(helper, data):
    # Reuse the frozen validated installer with this public module's explicit
    # identities substituted in the generated command, not in the helper module.
    code = helper.remote_install_code()
    return (code.replace(repr(str(helper.HPC_TOOL)), repr(str(data['hpc_tool'])))
        .replace(repr(helper.MODULE_COMMIT), repr(data['commit']))
        .replace(repr(helper.MODULE_SHA), repr(data['sha256']))
        .replace('len(raw)==%d' % helper.MODULE_BYTES, 'len(raw)==%d' % len(data['raw'])))


def source_loader(helper, data=None):
    tools = [(helper.MODULE_NAME, helper.HPC_TOOL, helper.MODULE_SHA, helper.MODULE_BYTES)]
    if data is not None: tools.append((DEPENDENCIES_NAME, data['hpc_tool'], data['sha256'], len(data['raw'])))
    code = "import hashlib,importlib,importlib.util,json,sys\nfrom pathlib import Path\n"
    code += 'base=Path(%r)\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(base/"repo"))\n' % str(helper.BASE)
    code += 'package=importlib.import_module("reproduction.easy2560_joint4")\n'
    code += 'assert Path(package.__file__).resolve()==base/"repo/reproduction/easy2560_joint4/__init__.py"\n'
    for name, path, digest, size in tools:
        code += ('name=%r;tool=Path(%r)\nassert tool.resolve()==tool and not tool.is_symlink() and tool.is_file()\n'
            'assert tool.stat().st_size==%d and hashlib.sha256(tool.read_bytes()).hexdigest()==%r\n'
            'assert name not in sys.modules\nspec=importlib.util.spec_from_file_location(name,str(tool))\n'
            'module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)\n'
            % (name, str(path), size, digest))
    return code


def source_operation(helper, operation, dependencies=None):
    need(operation in ('snapshot', 'emit'), 'SOURCE_OPERATION_DIFFERS')
    code = source_loader(helper, dependencies)
    if operation == 'snapshot':
        return code + 'print(json.dumps(module.snapshot(base,%d,%r),sort_keys=True))\n' % (helper.STEP, str(helper.CORE))
    snapshot = (helper.SOURCE_SNAPSHOT if dependencies is None else
        helper.BASE/'preparation/post-callback-dependencies'/('global_step_' + str(helper.STEP))/'source-snapshot.json')
    return code + 'module.emit(%r,sys.stdout.buffer)\n' % str(snapshot)


def boundary_observation_code(helper):
    folder = helper.BASE/'preparation/rolling-checkpoints'/('global_step_' + str(helper.STEP))
    return '''import json
from pathlib import Path
folder=Path(%r);path=folder/'boundary/manifest.json';ready=path.is_file() and (folder/'full-manifest.json').is_file()
if ready:
 value=json.loads(path.read_bytes())
 assert value['run_id']==%r and value['source_commit']==%r and value['completed_outer']==%d
 assert value['callbacks_complete'] is True and value['probe_round']==%d
print(json.dumps(dict(ready=ready)))
''' % (str(folder), helper.RUN, helper.TRAINING, helper.STEP, min(helper.STEP//50, 3)*50)


def observe_source(helper, ssh_argv, env, *, attempts=3, retry_seconds=15, sleep=time.sleep):
    """Only this read-only observation has bounded transport retries."""
    for attempt in range(1, attempts+1):
        try:
            result = subprocess.run(ssh_argv(helper.remote_command(boundary_observation_code(helper))),
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            need(len(result.stdout.encode()) <= 64*1024, 'SOURCE_OBSERVATION_TOO_LARGE')
            value = json.loads(result.stdout)
            need(set(value) == {'ready'} and type(value['ready']) is bool, 'SOURCE_OBSERVATION_SCHEMA_DIFFERS')
            return value
        if result is not None and result.returncode not in (5, 255):
            raise RuntimeError('SOURCE_OBSERVATION_NONTRANSPORT_RC_%d: %s' %
                (result.returncode, helper.redact(result.stderr[-2000:], env.get('SSHPASS'))))
        if attempt < attempts: sleep(retry_seconds)
    return dict(ready=False, transport_unknown=True)


def wait_for_boundary(helper, ssh_argv, env, completed, poll_seconds=60, sleep=time.sleep):
    while True:
        actual = observe_source(helper, ssh_argv, env)
        if actual['ready']: return actual
        controller = controller_observation(helper)
        if not actual.get('transport_unknown') and controller['state'] == 'TERMINAL':
            progress(helper, phase='DEPENDENCY_TERMINAL', step=helper.STEP, completed_steps=completed,
                source_observation=actual, core_controller=controller, core_actual_ack=True,
                training_failure_inferred=False)
            raise RuntimeError('DEPENDENCY_TERMINAL_CORE_CONTROLLER_BEFORE_POST_CALLBACK_BOUNDARY')
        progress(helper, phase='WAITING_POST_CALLBACK_BOUNDARY', step=helper.STEP, completed_steps=completed,
            source_observation=actual, core_controller=controller, core_actual_ack=True, training_failure_inferred=False)
        sleep(poll_seconds)


def transfer_once(helper, ssh_argv, env, module, name, dependencies=None):
    child = None
    try:
        saved = helper.checked_json(ssh_argv, source_operation(helper, 'snapshot', dependencies), env)
        helper.write_json(helper.CONTROL/(name+'-source-snapshot.json'), saved)
        if dependencies is None:
            need(saved.get('file_count') == 8 and saved.get('phase') == module.phase(helper.STEP), 'BOUNDARY_PHASE_DIFFERS')
        with (helper.CONTROL/(name+'-emit.stderr.log')).open('xb') as error:
            child = subprocess.Popen(ssh_argv(helper.remote_command(source_operation(helper, 'emit', dependencies))),
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=error)
            received = module.receive(helper.CORE, child.stdout); child.stdout.close(); rc = child.wait(timeout=120)
        helper.write_json(helper.CONTROL/(name+'-transfer.json'), dict(received, emit_returncode=rc, source_emit_completed=rc == 0))
        need(rc == 0, name.upper()+'_EMIT_EXIT_NONZERO_%d' % rc)
        receipt = module.accept(helper.CORE, received['stage'])
        need(receipt.get('status') == module.STATUS and receipt.get('actual_cuda_restore_accepted') is False,
             name.upper()+'_ACCEPTANCE_DIFFERS')
        helper.write_json(helper.CONTROL/(name+'-result.json'), receipt)
        return receipt
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)


def inspect_closure(helper, dependencies):
    value = dependencies.inspect(helper.CORE, verify_files=True)
    need(value.get('status') == dependencies.STATUS and value.get('all_manifest_dependencies_covered') is True and
         value.get('fresh_transferred_dependency_sha_checked') is True and
         value.get('covered_core_sha_ack_verified') is True and value.get('core_covered_payload_rehashed') is False and
         all(value.get('binding', {}).get(k) == v for k, v in dict(base=str(helper.BASE), run_id=helper.RUN,
             source_commit=helper.TRAINING, step=helper.STEP, core_archive=str(helper.CORE)).items()),
         'GROUNDING_DEPENDENCY_CLOSURE_NOT_ACTUALLY_ACCEPTED')
    return value


def verify_completed_five(helper, boundary):
    core = HK/'checkpoints'/helper.RUN/'global_step_5'
    receipt = helper.read_json(core/'post-callback-boundary/boundary-archive-receipt.json')
    need(receipt.get('status') == boundary.STATUS and receipt.get('file_count') == 8 and
         receipt.get('phase') == boundary.phase(5) and receipt.get('binding') == boundary.core_binding(core) and
         receipt.get('fresh_destination_sha_verified') is True and receipt.get('destination_fsynced') is True and
         receipt.get('actual_cuda_restore_accepted') is False,
         'COMPLETED_CP5_RECEIPT_DIFFERS')
    return dict(step=5, skipped=True, writer_replayed=False,
        receipt_sha256=helper.sha(core/'post-callback-boundary/boundary-archive-receipt.json'))


def launch_prerequisites():
    """Read-only launch evidence; future CP50 does not have to exist yet."""
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    helper = frozen_helper(); helper.bootstrap()
    observation = controller_observation(helper)
    need(observation['state'] == 'LIVE', 'EXISTING_CORE_CONTROLLER_NOT_ACTUALLY_LIVE')
    before = helper.ready_core()
    receipt = helper.read_json(helper.CORE/'post-callback-boundary/boundary-archive-receipt.json')
    need(receipt.get('status') == 'EASY2560_POST_CALLBACK_BOUNDARY_ARCHIVE_SHA_FSYNC_PASS' and
         receipt.get('file_count') == 8 and receipt.get('fresh_destination_sha_verified') is True and
         receipt.get('destination_fsynced') is True and receipt.get('actual_cuda_restore_accepted') is False and
         receipt.get('phase', {}).get('sampling_rng_stage') == 'post_QA5' and
         all(receipt.get('binding', {}).get(k) == v for k, v in dict(base=str(helper.BASE), run_id=helper.RUN,
             source_commit=helper.TRAINING, step=5, core_archive=str(helper.CORE)).items()),
         'COMPLETED_CP5_BOUNDARY_REQUIRED')
    return dict(core5=before, completed5_boundary_receipt_sha256=helper.sha(
        helper.CORE/'post-callback-boundary/boundary-archive-receipt.json'), core_controller=observation,
        helper_commit=HELPER_COMMIT, helper_sha256=HELPER_SHA, future_checkpoints_required=False,
        source_ssh_called=False, training_modified=False, checked_unix=time.time())


def install_source_tools(helper, ssh_argv, env, raw, data):
    installed = []
    for module_raw, commit, digest, target, code in (
        (raw, helper.MODULE_COMMIT, helper.MODULE_SHA, helper.HPC_TOOL, helper.remote_install_code()),
        (data['raw'], data['commit'], data['sha256'], data['hpc_tool'], dependency_install_code(helper, data))):
        value = helper.checked_json(ssh_argv, code, env, payload=dict(module_source=module_raw.decode(),
            published_commit=commit, module_sha256=digest))
        need(value.get('path') == str(target) and value.get('sha256') == digest and
             value.get('bytes') == len(module_raw) and value.get('source_job') == helper.JOB,
             'SOURCE_PUBLIC_TOOL_READBACK_DIFFERS')
        installed.append(value)
    return installed


def run(envelope, poll_seconds=60):
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    need(type(poll_seconds) in (int, float) and 1 <= poll_seconds <= 3600, 'POLL_SECONDS_INVALID')
    helper = frozen_helper(); password, raw, data = validate_envelope(helper, envelope)
    envelope.clear(); ssh_argv = helper.bootstrap(); env = None; completed = []
    ROOT.mkdir(parents=True, exist_ok=True); helper.canonical(ROOT)
    with (ROOT/'operation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((ROOT/name).exists() for name in ('claim.json', 'result.json', 'failure.json')), 'ONCE_CLAIM_EXISTS_NO_REPLAY')
        birth = current_birth()
        helper.write_json(ROOT/'claim.json', dict(base=str(helper.BASE), source_job=helper.JOB, run_id=helper.RUN,
            training_commit=helper.TRAINING, requested_steps=list(STEPS), helper_commit=HELPER_COMMIT, helper_sha256=HELPER_SHA,
            module_commit=helper.MODULE_COMMIT, dependencies_commit=data['commit'], dependencies_sha256=data['sha256'],
            runner_source_sha256=helper.sha(Path(__file__).resolve()), birth=birth, credentials_persisted=False,
            training_modified=False, healthy_controller_modified=False, jobs_submitted=0, checked_unix=time.time()))
        try:
            helper.install_immutable(helper.HK_TOOL, raw); boundary = helper.load_module()
            install_dependency(helper, data); dependencies = load_dependencies(helper, data)
            skipped = verify_completed_five(helper, boundary); helper.write_json(ROOT/'completed-five-read-only.json', skipped)
            env = dict(os.environ, SSHPASS=password, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            # Qualify only the NEW dependency module on the already accepted CP5.
            # The b87 eight-file writer is never invoked again for step5.
            qualification_before = helper.ready_core()
            helper.CONTROL = ROOT/'qualification-global_step_5'
            helper.CONTROL.mkdir(parents=True, exist_ok=True)
            need(not any(helper.CONTROL.iterdir()), 'QUALIFICATION_WRITER_ALREADY_ATTEMPTED_NO_REPLAY')
            helper.write_json(helper.CONTROL/'claim.json', dict(step=5, source_job=helper.JOB, run_id=helper.RUN,
                core_before=qualification_before, birth=birth, new_dependencies_only=True, boundary_writer_replayed=False))
            helper.write_json(helper.CONTROL/'source-tools.json', install_source_tools(helper, ssh_argv, env, raw, data))
            qualification_receipt = transfer_once(helper, ssh_argv, env, dependencies, 'dependencies', data)
            qualification_closure = inspect_closure(helper, dependencies)
            need(helper.ready_core() == qualification_before, 'CORE5_CHANGED_DURING_DEPENDENCY_QUALIFICATION')
            helper.write_json(helper.CONTROL/'result.json', dict(step=5,
                status='EASY7B_CP5_NEW_DEPENDENCIES_ONLY_QUALIFIED', dependencies_receipt=qualification_receipt,
                dependency_closure=qualification_closure, boundary_writer_replayed=False,
                future_new_head_transfer_actually_tested=False, actual_cuda_restore_accepted=False))
            for step in STEPS:
                configure_step(helper, step); before, observed = wait_for_core(helper, completed, poll_seconds)
                helper.CONTROL.mkdir(parents=True, exist_ok=True); helper.canonical(helper.CONTROL)
                need(not any(helper.CONTROL.iterdir()), 'STEP_WRITER_ALREADY_ATTEMPTED_NO_REPLAY')
                helper.write_json(helper.CONTROL/'claim.json', dict(step=step, source_job=helper.JOB, run_id=helper.RUN,
                    core_before=before, core_controller=observed, birth=birth, phase=boundary.phase(step), writer_attempt=1))
                wait_for_boundary(helper, ssh_argv, env, completed, poll_seconds)
                progress(helper, phase='BOUNDARY_WRITER_ONCE', step=step, completed_steps=completed)
                boundary_receipt = transfer_once(helper, ssh_argv, env, boundary, 'boundary')
                need(helper.ready_core() == before, 'CORE_CHANGED_DURING_BOUNDARY')
                progress(helper, phase='DEPENDENCIES_WRITER_ONCE', step=step, completed_steps=completed,
                    boundary_accepted=True, grounding_closure_accepted=False)
                dependencies_receipt = transfer_once(helper, ssh_argv, env, dependencies, 'dependencies', data)
                closure = inspect_closure(helper, dependencies)
                need(helper.ready_core() == before, 'CORE_CHANGED_DURING_DEPENDENCIES')
                helper.write_json(helper.CONTROL/'result.json', dict(step=step, source_job=helper.JOB, run_id=helper.RUN,
                    status='EASY7B_POST_CALLBACK_STATE_AND_DEPENDENCIES_ACCEPTED', phase=boundary.phase(step),
                    boundary_receipt=boundary_receipt, dependencies_receipt=dependencies_receipt, dependency_closure=closure,
                    actual_cuda_restore_accepted=False, actual_model_optimizer_rng_dataequal_accepted=False,
                    credentials_persisted=False, training_modified=False, healthy_controller_modified=False,
                    jobs_submitted=0, checked_unix=time.time()))
                completed.append(step); progress(helper, phase='MILESTONE_SUPPLEMENTS_ACCEPTED', step=step, completed_steps=completed)
            result = dict(status='EASY7B_FUTURE_POST_CALLBACK_COMPANION_COMPLETE', completed_steps=completed,
                actual_cuda_restore_accepted=False, credentials_persisted=False, training_modified=False,
                healthy_controller_modified=False, jobs_submitted=0, checked_unix=time.time())
            helper.write_json(ROOT/'result.json', result); return result
        except BaseException as error:
            helper.write_json(ROOT/'failure.json', dict(status=('DEPENDENCY_TERMINAL' if 'DEPENDENCY_TERMINAL' in str(error) else 'FAILED'),
                error_type=type(error).__name__, error=helper.redact(error, password)[-2000:], completed_steps=completed,
                source_job=helper.JOB, staging_preserved=True, automatic_writer_replay=False, training_failure_inferred=False,
                credentials_persisted=False, training_modified=False, jobs_submitted=0, checked_unix=time.time()))
            raise RuntimeError('POST_CALLBACK_FUTURE_FAILED_INSPECT_PRESERVED_FAILURE') from None
        finally:
            if env is not None: env.pop('SSHPASS', None)
            password = None


def main():
    raw = sys.stdin.buffer.read(MAX_ENVELOPE+1)
    need(0 < len(raw) <= MAX_ENVELOPE, 'SMALL_STDIN_ENVELOPE_REQUIRED')
    envelope = json.loads(raw); raw = None
    print(json.dumps(run(envelope), sort_keys=True), flush=True)


if __name__ == '__main__': main()
