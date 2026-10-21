"""CPU-only identity capture and natural terminal proof for one finite model.

Never signals, drains, launches models, touches primary GPU leases, or probes
CUDA/HTTP. Only its dedicated NAS10 capture/proof files are written. Slot
release probes acquire existing POSIX lock files nonblocking and close them.
"""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time

RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
BASE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
PHYSICAL = Path('/mnt/data/shaoshitong') / BASE.name
REGISTRY_SHA = '923fc27162d119dd1c413308487cee195ec3388cad827a2a8a98963f9900bf3e'
RENEWAL_SHA = '6db06d63bcf836d1911540d8a6f66f9419fa96f714bf178bf94bcc6b73ee7af4'
DIRECTORY = 'finite-instance-terminal-v1'
STATUS = 'OWN_FINITE_MODEL_APPLICATION_COMPLETE_RELEASED_V1'
CAPTURE = 'OWN_FINITE_MODEL_IDENTITY_CAPTURED_V1'
PROC = Path('/proc')
CGROUP = Path('/sys/fs/cgroup')
MAX_JSON = 512 * 1024


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:raise EvidenceError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


class View:
    """One previously selected NAS10 view; never retry after a SHA mismatch."""
    def __init__(self, root):self.root = Path(root)

    def relative(self, path):
        path = Path(path)
        require(path.is_absolute() and '..' not in path.parts, 'Noncanonical NAS10 path')
        for base in (BASE, PHYSICAL):
            if path.is_relative_to(base):return path.relative_to(base)
        raise EvidenceError('Path leaves this exact same-run NAS10 base')

    def path(self, path):return self.root / self.relative(path)
    def record(self, path):return str(PHYSICAL / self.relative(path))

    def metadata(self, path, optional=False):
        target = self.path(path)
        try:
            with os.fdopen(os.open(target, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                before = os.fstat(stream.fileno())
                require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_JSON, 'Expected bounded regular JSON metadata')
                raw = stream.read(MAX_JSON + 1);after = os.fstat(stream.fileno());named = target.stat()
            keys = lambda s:(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            require(keys(before) == keys(after) == keys(named) and len(raw) == before.st_size, 'Metadata changed during read')
            return dict(path=self.record(path), sha256=digest(raw), bytes=len(raw), device=before.st_dev,
                inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns, value=json.loads(raw))
        except FileNotFoundError:
            if optional:return None
            raise


def immutable(path, value):
    path = Path(path);raw = encoded(value)
    temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def query(unit):
    fields = ('Id', 'LoadState', 'MainPID', 'ActiveState', 'SubState', 'ControlGroup', 'Result',
              'ExecMainCode', 'ExecMainStatus', 'InvocationID')
    result = subprocess.run(['systemctl', 'show', unit, *[arg for key in fields for arg in ('-p', key)]],
        text=True, capture_output=True, timeout=10)
    return dict(returncode=result.returncode, properties=dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line),
        stderr=result.stderr[:2048], epoch=time.time())


def process(pid):
    try:
        folder = PROC / str(pid);fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), birth=fields[19], state=fields[0], cgroup=(folder / 'cgroup').read_text())
    except (FileNotFoundError, ProcessLookupError):return None


def members(group):
    result = {}
    for folder in PROC.glob('[0-9]*'):
        actual = process(int(folder.name))
        if actual and any(value == group or value.startswith(group + '/')
                for value in (line.split(':', 2)[-1] for line in actual['cgroup'].splitlines())):
            result[str(actual['pid'])] = actual['birth']
    return result


def validate_selection(value):
    require(value['schema_version'] == 1 and value['run_id'] == RUN_ID and value['instance'] == 'r1', 'Selection run/instance differs')
    require(value['registry_sha256'] == REGISTRY_SHA and value['renewal_plan_sha256'] == RENEWAL_SHA, 'Unreviewed registry/renewal plan')
    require(value['node'] in ('h100-5', 'h100-10') and value['unit'].startswith('sst-q7b-g5-')
        and re.fullmatch(r'[a-z0-9_.@-]+\.service', value['unit'])
        and value['unit'].endswith('-r1-' + RUN_ID[:12] + '.service'), 'Selection is not an exact finite r1 model')
    require(re.fullmatch(r'[a-z0-9-]+', value['directory']) and re.fullmatch(r'[a-z0-9-]+', value['slot']), 'Selection directory/slot is noncanonical')
    require(value['owner_unit'] != value['unit'] and re.fullmatch(r'[0-9a-f]{64}', value['source_bundle_sha256']), 'Owner/bundle identity differs')
    require(value['submit_intent'] == str(BASE / 'preparation' / value['directory'] / 'submit-intent.json'), 'Selection intent directory differs')
    return value


def root_for(selection):return PHYSICAL / 'preparation' / DIRECTORY / 'instances' / selection['node'] / selection['unit']


def native(selection, capture=None):
    result = query(selection['unit']);props = result['properties']
    require(result['returncode'] == 0 and props.get('Id') == selection['unit'], 'Native observation unavailable or wrong unit')
    load = props.get('LoadState')
    require(load in ('loaded', 'not-found'), 'Native load state is unknown')
    if load == 'not-found':return result, 'PURGED', None
    invocation = props.get('InvocationID')
    if capture and invocation and invocation != '0' * 32:
        require(invocation == capture['binding']['invocation_id'], 'Native invocation differs from captured r1')
    if props.get('ActiveState') == 'failed':return result, 'FAILED', None
    if props.get('ActiveState') in ('active', 'activating', 'deactivating', 'reloading'):
        return result, 'ACTIVE', None
    require(props.get('ActiveState') == 'inactive' and props.get('MainPID') == '0', 'Native terminal state is not established')
    status = None
    if props.get('ExecMainCode') in ('1', 'exited') and str(props.get('ExecMainStatus', '')).isdigit():
        status = int(props['ExecMainStatus'])
    if status not in (None, 0) or props.get('Result') not in ('success', ''):return result, 'FAILED', status
    return result, 'TERMINAL', status


def capture_identity(view, selection, code_identity):
    observed, state, _ = native(selection)
    if state != 'ACTIVE':return dict(status='ROOT_ATTENTION_CAPTURE_REQUIRES_ACTIVE_ORIGINAL_MODEL', native=observed)
    props = observed['properties'];pid = int(props.get('MainPID', '0'));group = '/system.slice/' + selection['unit']
    actual = process(pid)
    require(pid > 1 and props.get('ActiveState') == 'active' and props.get('ControlGroup') == group
        and actual and actual['state'] not in ('Z', 'X') and members(group).get(str(pid)) == actual['birth'], 'Active original controller/cgroup differs')
    invocation = props.get('InvocationID', '')
    require(re.fullmatch(r'[0-9a-f]{32}', invocation) and invocation != '0' * 32, 'Actual invocation ID was not captured')
    boot = (PROC / 'sys/kernel/random/boot_id').read_text().strip()
    require(re.fullmatch(r'[0-9a-fA-F-]{36}', boot), 'Actual boot ID was not captured')
    original = view.metadata(selection['submit_intent']);intent = original['value'];manifest = intent['manifest']
    require(intent['unit'] == manifest['unit'] == selection['unit'] and manifest['role'] == 'slot'
        and manifest['instance'] == 'r1' and manifest['slot'] == selection['slot']
        and manifest['source_bundle_sha256'] == selection['source_bundle_sha256'], 'Original submit intent/instance/bundle differs')
    prefix = 'source-slot' if selection['node'] == 'h100-5' else 'relay-slot'
    jobs = list(view.path(BASE / 'pool/jobs' / ('job-%s-%s-%d' % (prefix, selection['slot'], pid))).glob('attempt-*/allocation.json'))
    if not jobs:return dict(status='PENDING_CAPTURE_ALLOCATION_NOT_READY', native=observed)
    require(len(jobs) == 1, 'Original model has multiple allocation attempts')
    allocation_path = BASE / jobs[0].relative_to(view.root)
    record = view.metadata(allocation_path);allocation = record['value']
    require(allocation['run_id'] == RUN_ID and allocation['native_unit'] == selection['unit'] and allocation['node'] == selection['node']
        and allocation['controller_pid'] == pid and allocation['native_cgroup'] == group
        and allocation['source_owner']['unit'] == selection['owner_unit'], 'Original allocation owner/controller differs')
    require(allocation['model_indices'] == selection['model_indices']
        and allocation['tensor_parallel_size'] == selection['tensor_parallel_size'], 'Original model topology differs')
    evidence = view.metadata(allocation['allocation_evidence'])
    require(evidence['sha256'] == allocation['allocation_evidence_sha256']
        and evidence['value']['main_pid'] == pid and evidence['value']['unit'] == selection['unit']
        and evidence['value']['cgroup'] == group and evidence['value']['node'] == selection['node'], 'Allocation evidence differs')
    owner_ready = view.metadata(allocation['source_owner_ready'])
    require(owner_ready['sha256'] == allocation['source_owner_ready_sha256']
        and owner_ready['value']['owner'] == allocation['source_owner'], 'Borrowed owner-ready binding differs')
    context = intent['context'];ready_sha = context.get('owner_ready_sha256', context.get('downstream_owner_ready_sha256'))
    require(ready_sha == owner_ready['sha256'], 'Original submit context belongs to another owner-ready')
    owner = allocation['source_owner'];lease_root = BASE / 'pool/gpu-slot-leases' / RUN_ID / ('%d-%s' % (owner['pid'], owner['birth']))
    declared = allocation['slot_lease_paths'];require(isinstance(declared, dict) and declared, 'Original model declares no slot leases')
    leases = []
    for raw in declared.values():
        relative = view.relative(raw);canonical = BASE / relative
        require(canonical.is_relative_to(lease_root) and canonical.suffix == '.lock', 'Lease is not an original owner slot lease')
        info = view.path(raw).lstat();require(stat.S_ISREG(info.st_mode), 'Original lease is not a regular file')
        leases.append(dict(path=view.record(raw), device=info.st_dev, inode=info.st_ino))
    require(len({row['path'] for row in leases}) == len(leases), 'Duplicate declared slot lease')
    births = members(group)
    borrowed_pids = {owner['pid'], allocation['local_relay_process']['pid']}
    require(births.get(str(pid)) == actual['birth'] and not borrowed_pids.intersection(map(int, births)), 'Borrowed process was included as an owned child')
    final, state, _ = native(selection);final_process = process(pid)
    require(state == 'ACTIVE' and final['properties']['MainPID'] == str(pid)
        and final['properties']['InvocationID'] == invocation and final_process
        and final_process['birth'] == actual['birth'], 'Original controller changed during capture')
    binding = dict(selection=selection, code_identity=code_identity, controller_pid=pid, controller_birth=actual['birth'],
        cgroup=group, boot_id=boot, invocation_id=invocation, original_owned_births=births,
        attempt=view.record(allocation_path.parent), job_id=allocation['job_id'],
        original_metadata={name:{key:value for key,value in row.items() if key != 'value'} for name,row in
            (('submit_intent', original), ('allocation', record), ('allocation_evidence', evidence), ('owner_ready', owner_ready))},
        slot_leases=sorted(leases, key=lambda x:x['path']), borrowed_processes=sorted(borrowed_pids))
    return dict(status=CAPTURE, schema_version=1, binding=binding, binding_sha256=digest(encoded(binding)), native=final,
        model_processes_modified=False, borrowed_processes_modified=False, epoch=time.time())


def validate_capture(view, selection, code_identity, capture):
    require(capture['status'] == CAPTURE and capture['schema_version'] == 1, 'Wrong capture status/schema')
    binding = capture['binding']
    require(binding['selection'] == selection and binding['code_identity'] == code_identity
        and capture['binding_sha256'] == digest(encoded(binding)), 'Capture belongs to another exact instance/binding')
    require(binding['original_owned_births'].get(str(binding['controller_pid'])) == binding['controller_birth'], 'Captured controller birth binding differs')
    require(binding['slot_leases'] and len({x['path'] for x in binding['slot_leases']}) == len(binding['slot_leases']), 'Captured slot lease declaration differs')
    for expected in binding['original_metadata'].values():
        current = view.metadata(expected['path'])
        require(all(current[key] == expected[key] for key in expected), 'Pinned original metadata changed: ' + expected['path'])
    return binding


def application(view, binding):
    job = Path(binding['attempt'])
    failures = [view.metadata(job / name, optional=True) for name in ('failure.json', 'sampler-failure.json')]
    require(not any(failures), 'Original model has a final attempt failure')
    complete = view.metadata(job / 'complete.json', optional=True)
    sampler = view.metadata(job / 'sampler-exit.json', optional=True)
    require(complete and sampler, 'Original terminal model lacks complete/sampler-exit metadata')
    a, b = complete['value'], sampler['value']
    for value in (a, b):
        require(value['run_id'] == RUN_ID and value['job_id'] == binding['job_id'], 'Application terminal run/job differs')
        counts = [value.get(key) for key in ('completed_own_batches', 'questions', 'trajectories')]
        require(all(type(x) is int and x >= 0 for x in counts) and counts[2] == counts[1] * 5, 'Application whole-G5 counts differ')
    require(a['controller_status'] == 'OWN_RELAY_CONSUMER_FINISHED' and a['sampler_returncode'] == 0
        and a['own_children_reaped'] is True and a['foreign_processes_signalled'] is False
        and a['upstream_retriever_signalled'] is False, 'Application did not complete its own clean shutdown')
    require(b['status'] in ('POOL_BATCH_BOUNDARY_TIME_LIMIT', 'POOL_INPUT_SCAN_FINISHED', 'POOL_REMAINING_BATCHES_LOCKED_BY_OTHER_JOBS'),
        'Application was not a natural finite sampler completion')
    require(all(a[key] == b[key] for key in ('status', 'completed_own_batches', 'questions', 'trajectories')), 'Controller/sampler terminal counts differ')
    return dict(complete=complete, sampler_exit=sampler)


def release(view, binding):
    remaining = {pid:birth for pid,birth in binding['original_owned_births'].items()
        if (process(int(pid)) or {}).get('birth') == birth}
    current_members = members(binding['cgroup'])
    folder = CGROUP / binding['cgroup'].lstrip('/')
    kernel_members = sorted({pid for path in folder.rglob('cgroup.procs') for pid in path.read_text().split()}) if folder.exists() else []
    acquired, busy = [], []
    for expected in binding['slot_leases']:
        path = view.path(expected['path'])
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (expected['device'], expected['inode']), 'Original slot lease inode changed')
            try:fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):raise
                busy.append(expected['path'])
            else:acquired.append(dict(expected, probe='POSIX_WRITE_ACQUIRED_NONBLOCKING_THEN_CLOSED'))
        finally:os.close(fd)
    return dict(epoch=time.time(), original_births_gone=not remaining, remaining_original_births=remaining,
        own_cgroup_empty=not current_members and not kernel_members, remaining_cgroup_members=current_members,
        kernel_cgroup_pids=kernel_members, slot_leases_free=not busy, slot_leases_acquired=acquired, busy_slot_leases=busy)


def released(value):return value['original_births_gone'] is True and value['own_cgroup_empty'] is True and value['slot_leases_free'] is True


def journal(binding, unit):
    boot = binding['boot_id'].replace('-', '')
    argv = ['journalctl', '--no-pager', '-o', 'json', '-n', '32', '_BOOT_ID=' + boot,
        '_SYSTEMD_INVOCATION_ID=' + binding['invocation_id'], '+', '_BOOT_ID=' + boot,
        'INVOCATION_ID=' + binding['invocation_id'], 'UNIT=' + unit]
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=8)
        rows = []
        for line in result.stdout.splitlines():
            try:value = json.loads(line)
            except ValueError:continue
            if value.get('_BOOT_ID', '').replace('-', '') != binding['boot_id'].replace('-', ''):continue
            if binding['invocation_id'] not in (value.get('_SYSTEMD_INVOCATION_ID'), value.get('INVOCATION_ID')):continue
            if unit not in (value.get('_SYSTEMD_UNIT'), value.get('UNIT')):continue
            rows.append({key:str(value[key])[:256] for key in ('__REALTIME_TIMESTAMP', '_BOOT_ID', '_SYSTEMD_INVOCATION_ID',
                'INVOCATION_ID', '_SYSTEMD_UNIT', 'UNIT', 'MESSAGE_ID', 'RESULT', 'EXIT_CODE', 'EXIT_STATUS') if key in value})
        return dict(returncode=result.returncode, matched_same_invocation=rows, stdout_sha256=digest(result.stdout.encode()),
            stdout_bytes=len(result.stdout.encode()), stderr=result.stderr[:1024], required_for_pass=False)
    except (OSError, subprocess.TimeoutExpired) as error:return dict(available=False, error=str(error), required_for_pass=False)


def validate_proof(view, selection, capture_record, proof):
    capture = capture_record['value'];binding = capture['binding']
    require(proof['status'] == STATUS and proof['schema_version'] == 1 and proof['unit'] == selection['unit']
        and proof['instance'] == selection['instance'] and proof['node'] == selection['node'] and proof['run_id'] == RUN_ID,
        'Cached proof belongs to another model instance')
    require(proof['capture_sha256'] == capture_record['sha256'] and proof['binding_sha256'] == capture['binding_sha256']
        and proof['binding'] == binding, 'Cached proof allocation/birth binding differs')
    app = application(view, binding)
    require(proof['application'] == app, 'Cached application terminal metadata differs')
    observations = proof['release_observations']
    require(len(observations) == 2 and observations[1]['epoch'] - observations[0]['epoch'] >= 3, 'Cached proof lacks two timed releases')
    for value in observations:
        require(released(value) and not value['remaining_original_births'] and not value['remaining_cgroup_members']
            and not value['kernel_cgroup_pids'] and not value['busy_slot_leases'], 'Cached proof release fields are not empty')
        expected = [dict(row, probe='POSIX_WRITE_ACQUIRED_NONBLOCKING_THEN_CLOSED') for row in binding['slot_leases']]
        require(value['slot_leases_acquired'] == expected, 'Cached proof original lease paths/inodes differ')
    require(proof['borrowed_processes_modified'] is False and proof['primary_gpu_locks_modified'] is False
        and proof['signals_sent'] is False, 'Cached proof is not an observer-only terminal proof')
    properties = proof['native']['properties']
    require(proof['native']['returncode'] == 0 and properties.get('Id') == selection['unit']
        and properties.get('LoadState') in ('loaded', 'not-found'), 'Cached native observation names another unit or unknown state')
    if properties['LoadState'] == 'not-found':
        require(proof['native_exit_status'] is None, 'Purged native status was fabricated')
    else:
        require(properties.get('MainPID') == '0' and properties.get('ActiveState') == 'inactive'
            and proof['native_exit_status'] in (None, 0), 'Cached native terminal observation is not successful/unknown')
    return proof


def operate(action, selection, view, code_identity):
    """Return evidence/status; only capture and successful collect write here."""
    try:
        validate_selection(selection)
        require(os.geteuid() == 0 and socket.gethostname().split('.')[0] == selection['node'], 'Wrong observer node/uid')
        require(view.metadata(BASE / 'config.json')['value']['run_id'] == RUN_ID, 'NAS10 view belongs to another run')
        root = root_for(selection);capture_path = root / 'capture.json';complete_path = root / 'complete.json'
        saved = view.metadata(capture_path, optional=True)
        if action == 'capture' and not saved:
            value = capture_identity(view, selection, code_identity)
            if value['status'] != CAPTURE:return value
            target = view.path(root);target.mkdir(parents=True, exist_ok=True)
            with (target / 'capture.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not view.path(capture_path).exists():immutable(view.path(capture_path), value)
            saved = view.metadata(capture_path)
        if not saved:
            observed, state, _ = native(selection)
            return dict(status='PENDING_NOT_CAPTURED' if state == 'ACTIVE' else 'ROOT_ATTENTION_IDENTITY_CAPTURE_MISSING', native=observed)
        binding = validate_capture(view, selection, code_identity, saved['value'])
        observed, state, native_exit = native(selection, saved['value'])
        if state == 'ACTIVE':
            actual = process(binding['controller_pid'])
            require(observed['properties']['MainPID'] == str(binding['controller_pid'])
                and actual and actual['birth'] == binding['controller_birth'], 'Active model no longer matches original captured birth')
            return dict(status='FINITE_INSTANCE_CAPTURED_ACTIVE' if action == 'capture' else 'PENDING_MODEL_ACTIVE',
                capture=saved['path'], capture_sha256=saved['sha256'], binding_sha256=saved['value']['binding_sha256'], native=observed)
        if state == 'FAILED':return dict(status='ROOT_ATTENTION_NATIVE_MODEL_FAILED', native=observed, native_exit_status=native_exit)
        app = application(view, binding)
        if action != 'collect':return dict(status='APPLICATION_COMPLETE_RELEASE_NOT_COLLECTED', native=observed,
            native_exit_status=native_exit, capture_sha256=saved['sha256'], application=app)
        existing = view.metadata(complete_path, optional=True)
        if existing:validate_proof(view, selection, saved, existing['value'])
        first = release(view, binding)
        if not released(first):return dict(status='PENDING_MODEL_RESOURCE_RELEASE', native=observed, release_observation=first)
        if existing:return dict(status=STATUS, proof=existing['path'], proof_sha256=existing['sha256'],
            cached_exact_binding_validated=True, fresh_release_observation=first,
            native_exit_status=existing['value']['native_exit_status'], native=observed)
        history = journal(binding, selection['unit'])
        for entry in history.get('matched_same_invocation', []):
            failure_results = ('exit-code', 'signal', 'core-dump', 'timeout', 'watchdog', 'oom-kill', 'resources', 'failed', 'start-limit-hit', 'protocol')
            require(entry.get('RESULT') not in failure_results
                and (entry.get('EXIT_STATUS') in (None, '', '0')
                    or entry.get('EXIT_CODE') not in ('exited', 'killed', 'dumped')),
                'Same-invocation journal records a native failure')
        time.sleep(3)
        final_native, final_state, final_exit = native(selection, saved['value'])
        require(final_state in ('PURGED', 'TERMINAL'), 'Native state changed during terminal collection')
        second = release(view, binding)
        if not released(second):return dict(status='PENDING_MODEL_RESOURCE_RELEASE', native=final_native, release_observation=second)
        require(second['epoch'] - first['epoch'] >= 3, 'Actual release observation interval is below three seconds')
        require(application(view, binding) == app, 'Application metadata changed between release observations')
        value = dict(status=STATUS, schema_version=1, run_id=RUN_ID, node=selection['node'], unit=selection['unit'], instance=selection['instance'],
            capture=saved['path'], capture_sha256=saved['sha256'], binding_sha256=saved['value']['binding_sha256'], binding=binding,
            application=app, native=final_native, native_exit_status=final_exit,
            native_exit_basis='PURGED_NATIVE_METADATA_UNAVAILABLE' if final_state == 'PURGED' else 'ACTUAL_TERMINAL_NATIVE_PROPERTIES',
            same_invocation_journal=history, release_observations=[first, second], signals_sent=False,
            borrowed_processes_modified=False, primary_gpu_locks_modified=False, owner_and_retriever_retained=True,
            original_partial_outputs_retained=True, global_goal_complete=False, epoch=time.time())
        validate_proof(view, selection, saved, value)
        with (view.path(root) / 'collect.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not view.path(complete_path).exists():immutable(view.path(complete_path), value)
        sealed = view.metadata(complete_path);validate_proof(view, selection, saved, sealed['value'])
        return dict(status=STATUS, proof=sealed['path'], proof_sha256=sealed['sha256'],
            native_exit_status=sealed['value']['native_exit_status'], native_exit_basis=sealed['value']['native_exit_basis'],
            actual_release_observations=2, global_goal_complete=False)
    except (EvidenceError, KeyError, ValueError) as error:
        return dict(status='ROOT_ATTENTION_FINITE_INSTANCE_EVIDENCE_INVALID', error=str(error), action=action)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(status='OBSERVATION_UNKNOWN_NO_NATIVE_EXIT_INFERRED', error=str(error), action=action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'status', 'collect'))
    parser.add_argument('--selection', required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--view-root', required=True)
    parser.add_argument('--code-identity', required=True)
    args = parser.parse_args();raw = Path(args.selection).read_bytes()
    require(digest(raw) == args.selection_sha256, 'Selection SHA differs')
    print(json.dumps(operate(args.action, json.loads(raw), View(args.view_root), args.code_identity), sort_keys=True))


if __name__ == '__main__':main()
