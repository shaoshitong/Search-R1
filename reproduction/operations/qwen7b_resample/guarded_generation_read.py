"""Read-only native/metadata snapshot; invoked only by an explicit follower.

No HTTP, CUDA, process launches, locks, signals or writes. REGISTRY, NODE and
HISTORY_PROOFS are supplied by the caller; the source is sent over transport.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

PROC = Path('/proc')
MAX_JSON = 512 * 1024


class Reader:
    def __init__(self, base, physical, views):
        self.base, self.physical = Path(base), Path(physical)
        self.views = list(dict.fromkeys(map(Path, views)))

    def relative(self, path):
        path = Path(path)
        if '..' in path.parts:raise ValueError('Noncanonical metadata path')
        for root in (self.base, self.physical, *self.views):
            if path.is_relative_to(root):return path.relative_to(root)
        raise ValueError('Metadata is outside the same-run NAS10 root')

    def path(self, path):
        relative = self.relative(path)
        return next((root / relative for root in self.views if (root / relative).exists()), self.views[0] / relative)

    def record(self, path):return str(self.physical / self.relative(path))

    def metadata(self, path, limit=MAX_JSON):
        path = self.path(path)
        try:
            before = path.stat()
            if before.st_size > limit:raise ValueError('JSON metadata exceeds size bound')
            raw = path.read_bytes();after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Metadata changed during read; retry next observation')
            return dict(path=self.record(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns, value=json.loads(raw))
        except FileNotFoundError:return None

    def read(self, path):return (self.metadata(path) or {}).get('value')


def process(pid):
    try:
        root = PROC / str(pid);fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, birth=fields[19], state=fields[0], cgroup=(root / 'cgroup').read_text())
    except (FileNotFoundError, ProcessLookupError, PermissionError):return dict(pid=pid, present=False)


def phases(reader, allocation, children):
    result = []
    group = allocation.get('native_cgroup')
    for pid in children.get('own_process_groups', []):
        actual = process(pid)
        if not group or actual.get('state') in ('Z', 'X') or not any(line.split(':', 2)[-1] == group for line in actual.get('cgroup', '').splitlines()):continue
        try:descriptors = list((PROC / str(pid) / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):continue
        for fd in descriptors:
            try:
                target = os.readlink(fd)
                relative = reader.relative(target)
                if relative.parts[:2] != ('pool', 'batch-locks') or len(relative.parts) != 4 or relative.suffix != '.lock':continue
                detail = (PROC / str(pid) / 'fdinfo' / fd.name).read_text()
                if not any(line.startswith('lock:') and 'POSIX' in line and 'WRITE' in line and str(pid) in line.split() for line in detail.splitlines()):continue
                shard, batch = relative.parts[2], relative.stem
                root = reader.physical / 'sampling' / shard / 'batches' / batch
                result.append(dict(child=actual, fd=int(fd.name), lock_target=reader.record(target), batch=batch, shard=shard,
                    fdinfo_sha256=hashlib.sha256(detail.encode()).hexdigest(), whole_complete=reader.metadata(root / 'complete.json'),
                    phases={phase: reader.metadata(root / 'outers/outer-000001' / phase / 'rank-0-progress.json') for phase in ('prefix', 'full')}))
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):continue
    return result


def attempt(reader, job, allocation):
    children = reader.read(job / 'children.json') or {}
    progress = reader.metadata(job / 'progress.json')
    latest = ((progress or {}).get('value') or {}).get('latest_input')
    latest_seal = None
    if latest:
        relative = reader.relative(latest)
        if len(relative.parts) == 3 and relative.parts[0] == 'inputs':
            latest_seal = reader.metadata(reader.physical / 'sampling' / relative.parts[1] / 'batches' / Path(latest).stem / 'complete.json')
    return dict(directory=reader.record(job), allocation=allocation, progress=progress, latest_own_seal=latest_seal,
        failure=reader.metadata(job / 'failure.json'), complete=reader.metadata(job / 'complete.json'),
        owner_ready=reader.metadata(job / 'owner-ready.json') or reader.metadata(job / 'relay-owner-ready.json'),
        retrieval_ready=reader.metadata(job / 'retriever-ready.json'),
        model_initialized=[reader.metadata(p) for p in sorted((job / 'sampler-launches').glob('launch-*/model-initialized.json'))],
        launch_failures=[reader.metadata(p) for p in sorted((job / 'sampler-launches').glob('launch-*/failure.json'))],
        typed_initialization_ooms=[reader.metadata(p) for p in sorted((job / 'sampler-launches').glob('launch-*/typed-model-initialization-oom.json'))],
        active_phases=phases(reader, allocation['value'], children))


def sealed(reader, run_id):
    seen, latest, errors = set(), [], []
    count = questions = trajectories = 0
    for path in sorted(reader.path(reader.physical / 'sampling').glob('shard-*/batches/batch-*/complete.json')):
        try:
            item = reader.metadata(path);value = item['value'];key = (item['device'], item['inode'])
            if key in seen:raise ValueError('Duplicate sealed inode')
            seen.add(key)
            if value.get('run_id') != run_id or value.get('batch') != path.parent.name or not all(type(value.get(k)) is int and value[k] > 0 for k in ('questions', 'trajectories')) or value['trajectories'] != value['questions'] * 5:
                raise ValueError('Whole-G5 seal identity/count mismatch')
            count += 1;questions += value['questions'];trajectories += value['trajectories']
            latest.append(dict(path=item['path'], sha256=item['sha256'], questions=value['questions'], trajectories=value['trajectories'], epoch=value.get('epoch')))
        except (OSError, ValueError) as error:errors.append(dict(path=reader.record(path), error=str(error)))
    return dict(sealed_batches=count, questions=questions, trajectories=trajectories,
        latest=sorted(latest, key=lambda x: x['epoch'] or 0)[-8:], errors=errors, independent_full_audit_performed=False)


def historical_index(reader, pins, run_id):
    index, proofs, errors, seen = {}, [], [], set()
    pending = list(pins)
    while pending:
        pin = pending.pop()
        if (pin['path'], pin['sha256']) in seen:continue
        seen.add((pin['path'], pin['sha256']))
        try:
            item = reader.metadata(pin['path'], 4 * 1024 * 1024)
            if not item or item['sha256'] != pin['sha256']:raise ValueError('Historical proof SHA differs')
            proof = item['value'];observations = proof.get('release_observations') or []
            if proof.get('run_id') != run_id or len(observations) < 2 or observations[-1]['epoch'] - observations[0]['epoch'] < 3 or not all(
                    x.get('original_births_gone') and x.get('own_cgroups_empty') and not x.get('remaining_known_pids')
                    and not x.get('remaining_original_births') and not x.get('own_gpu_contexts') for x in observations):
                raise ValueError('Historical proof lacks actual terminal release observations')
            for entry in proof.get('original_files', []):
                relative = reader.relative(entry['path'])
                if len(relative.parts) == 5 and relative.parts[:2] == ('pool', 'jobs') and relative.name in ('failure.json', 'sampler-failure.json'):
                    index[(reader.record(entry['path']), entry['sha256'])] = dict(proof_path=item['path'], proof_sha256=item['sha256'], proof_status=proof['status'])
            proofs.append(dict(path=item['path'], sha256=item['sha256'], status=proof['status']))
            if proof.get('remote_node10_proof') and proof.get('remote_node10_proof_sha256'):
                pending.append(dict(path=proof['remote_node10_proof'], sha256=proof['remote_node10_proof_sha256']))
            if proof.get('source_plan') and proof.get('source_plan_sha256'):
                plan = reader.metadata(proof['source_plan'], 4 * 1024 * 1024)
                if not plan or plan['sha256'] != proof['source_plan_sha256']:raise ValueError('Historical source plan SHA differs')
                plan = plan['value']
                if plan.get('prior_failed_proof_path'):pending.append(dict(path=plan['prior_failed_proof_path'], sha256=plan['prior_failed_proof_sha256']))
        except (OSError, ValueError, KeyError) as error:errors.append(dict(pin=pin, error=str(error)))
    return index, proofs, errors


def failure_census(reader, registry, pins, run_id):
    index, proofs, errors = historical_index(reader, pins, run_id)
    rows = []
    current = {row['unit'] for row in registry}
    for job in sorted(reader.path(reader.physical / 'pool/jobs').glob('job-*/attempt-*')):
        failure = reader.metadata(job / 'failure.json') or reader.metadata(job / 'sampler-failure.json')
        if not failure:continue
        allocation = reader.read(job / 'allocation.json') or {};fact = failure['value']
        row = dict(path=failure['path'], sha256=failure['sha256'], native_unit=allocation.get('native_unit'),
            error_type=fact.get('error_type'), original_status=fact.get('status'), classification='UNKNOWN_FAILURE_REQUIRES_ROOT')
        historical = index.get((failure['path'], failure['sha256']))
        if allocation.get('native_unit') not in current and historical:row.update(classification='HISTORICAL_ATTENTION_TERMINAL_PROVEN', **historical)
        elif allocation.get('native_unit') not in current:
            resolution = reader.metadata(job / 'resolution.json')
            if resolution:
                value = resolution['value']
                if value.get('status') == 'RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT' and value.get('original_failure_sha256') == failure['sha256'] and value.get('replacement_new_complete_batches', 0) >= 2:
                    proof = reader.metadata(value['terminal_proof'], 4 * 1024 * 1024)
                    if proof and proof['sha256'] == value.get('terminal_proof_sha256'):
                        row.update(classification='HISTORICAL_RECOVERY_RECEIPT_RETAINED', proof_path=proof['path'], proof_sha256=proof['sha256'], resolution_sha256=resolution['sha256'])
        rows.append(row)
    return dict(failures=rows, proofs=proofs, proof_errors=errors,
        total_original_failures=len(rows), historical_attention=sum(x['classification'] == 'HISTORICAL_ATTENTION_TERMINAL_PROVEN' for x in rows),
        historical_recovered=sum(x['classification'] == 'HISTORICAL_RECOVERY_RECEIPT_RETAINED' for x in rows))


def context_proofs(reader, registry):
    # NAS10 is shared: the node10 bank census must also follow the source5
    # owner's actual predecessor context, without querying a remote PID.
    result = []
    for row in registry:
        if row['role'] == 'model':continue
        context = reader.read(reader.physical / 'preparation' / row['directory'] / 'submit-context.json') or {}
        if context.get('predecessor_proof') and context.get('predecessor_sha256'):
            result.append(dict(path=context['predecessor_proof'], sha256=context['predecessor_sha256']))
    return result


def observe(registry, node, base, physical, pins, run_id):
    assert socket.gethostname().split('.')[0] == node
    selected = [row for row in registry if row['node'] == node]
    props = ('Id', 'LoadState', 'MainPID', 'ActiveState', 'SubState', 'ControlGroup', 'Result', 'ExecMainStatus')
    query = subprocess.run(['systemctl', 'show', *[x['unit'] for x in selected], *[arg for name in props for arg in ('-p', name)]], text=True, capture_output=True, timeout=20)
    native = {}
    for block in query.stdout.strip().split('\n\n'):
        row = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
        if row.get('Id'):native[row['Id']] = row
    views = []
    for row in selected:
        state = native.get(row['unit']) or {};pid = int(state.get('MainPID', '0'))
        actual = process(pid) if pid else {}
        if pid and state.get('ControlGroup') and any(x.split(':', 2)[-1] == state['ControlGroup'] for x in actual.get('cgroup', '').splitlines()):
            views.append(PROC / str(pid) / 'root' / Path(base).relative_to('/'))
    views += [Path(physical), Path(base)]
    views = [path for path in views if path.exists()]
    if not views:
        return dict(status='READ_ONLY_GUARDED_GENERATION', node=node, run_id=run_id, epoch=time.time(),
            native_query_returncode=query.returncode, native_query_stderr=query.stderr, units={x['key']: dict(unit=x['unit'], native=native.get(x['unit'], {}), attempts=[]) for x in selected}, nas_view_unavailable=True)
    reader = Reader(base, physical, views)
    allocations = []
    for job in sorted(reader.path(reader.physical / 'pool/jobs').glob('job-*/attempt-*')):
        allocation = reader.metadata(job / 'allocation.json')
        if allocation:allocations.append((job, allocation))
    units = {}
    for row in selected:
        root = reader.path(reader.physical / 'preparation' / row['directory'])
        manifest = reader.read(root / row['source_bundle_sha256'] / 'manifest.json')
        units[row['key']] = dict(unit=row['unit'], native=native.get(row['unit'], {}),
            submitted=(root / 'submit-intent.json').exists(),
            staged_manifest_matches=bool(manifest and manifest.get('unit') == row['unit'] and manifest.get('source_bundle_sha256') == row['source_bundle_sha256']),
            attempts=[attempt(reader, job, item) for job, item in allocations if item['value'].get('native_unit') == row['unit']],
            owner_ready=reader.metadata(root / 'owner-ready.json'))
    result = dict(status='READ_ONLY_GUARDED_GENERATION', node=node, run_id=run_id, epoch=time.time(),
        native_query_returncode=query.returncode, native_query_stderr=query.stderr, units=units)
    if node == 'h100-10':result.update(global_sealed=sealed(reader, run_id),
        historical=failure_census(reader, registry, [*pins, *context_proofs(reader, registry)], run_id))
    return result


if __name__ == '__main__':
    print(json.dumps(observe(REGISTRY, NODE, BASE, PHYSICAL, HISTORY_PROOFS, RUN_ID)))
