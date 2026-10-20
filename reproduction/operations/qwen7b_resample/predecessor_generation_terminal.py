"""Seal one explicit own predecessor cohort before a future generation rollover.

This tool is not a scheduler or an expiry timer. A later operator supplies a
fresh, immutable, node-local plan with exact original PIDs, births, allocations
and dependency order. It never discovers and stops arbitrary matching units.
Run outside those units' cgroups, already inside the old source mount namespace
on h100-5, so that NAS10 remains accessible after the old owner exits. Existing
intent is observation-only; an interrupted command is never blindly replayed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time

RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
BASE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
PHYSICAL = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
SOURCE = dict(node='h100-5', unit='sst-q7b-g5-sharedvram-v6-70188b8e50fd.service',
              pid=1490214, birth='6807965', retriever_pid=1490465, retriever_birth='6808057')
CONSUMER = dict(node='h100-10', unit='sst-q7b-g5-consumer10-v5-70188b8e50fd.service',
                pid=2371322, birth='1958624331')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def birth(pid):
    try:
        fields = Path('/proc/%d/stat' % int(pid)).read_text().rsplit(')', 1)[1].split()
        return fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def query(unit):
    value = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'MainPID',
        '-p', 'ActiveState', '-p', 'ControlGroup', '-p', 'Result', '-p', 'ExecMainStatus'],
        text=True, timeout=20)
    return dict(line.split('=', 1) for line in value.splitlines() if '=' in line)


def physical(path):
    path = Path(path)
    if path.is_relative_to(BASE):
        return str(PHYSICAL / path.relative_to(BASE))
    assert path.is_relative_to(PHYSICAL)
    return str(path)


def validate_plan(plan):
    assert plan['run_id'] == RUN_ID and plan['node'] in ('h100-5', 'h100-10')
    expected = SOURCE if plan['node'] == 'h100-5' else CONSUMER
    assert plan['source'] == {key: expected[key] for key in ('unit', 'pid', 'birth')}
    rows = plan['ordered_units']
    assert rows and len({row['unit'] for row in rows}) == len(rows)
    assert (rows[-1]['unit'], rows[-1]['pid'], str(rows[-1]['birth'])) == (expected['unit'], expected['pid'], expected['birth'])
    assert len({int(row['pid']) for row in rows}) == len(rows)
    positions = {row['unit']: i for i, row in enumerate(rows)}
    for row in rows:
        assert row['unit'].startswith('sst-q7b-') and row['unit'].endswith(RUN_ID[:12] + '.service')
        assert int(row['pid']) > 1 and str(row['birth']).isdigit()
        assert row['node'] == plan['node'], 'Never check another node PID locally'
        for dependency in row.get('borrows_units', []):
            assert dependency in positions and positions[row['unit']] < positions[dependency], 'Stop borrowers before owners'
        for path in row.get('metadata_roots', []):
            path = Path(path)
            assert path.is_absolute() and '..' not in path.parts
            assert path.is_relative_to(BASE) or path.is_relative_to(PHYSICAL)
            assert not any(part in ('sampling', 'cache', 'assets', 'code') for part in path.relative_to(BASE if path.is_relative_to(BASE) else PHYSICAL).parts)
        assert bool(row.get('allocation_path')) == bool(row.get('allocation_sha256'))
        if row.get('allocation_path'):
            path = Path(row['allocation_path'])
            assert path.is_absolute() and '..' not in path.parts
            assert path.is_relative_to(BASE / 'pool/jobs') or path.is_relative_to(PHYSICAL / 'pool/jobs')
    assert plan.get('original_partial_outputs_retained') is True
    assert plan.get('old_loops_have_no_drain_hook') is True
    if plan['node'] == 'h100-5':
        assert plan.get('remote_node10_proof') and plan.get('remote_node10_proof_sha256')
    return expected


def owned(cgroup):
    found = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            paths = [line.split(':', 2)[-1] for line in (path / 'cgroup').read_text().splitlines()]
            if any(value == cgroup or value.startswith(cgroup + '/') for value in paths):
                value = birth(path.name)
                if value is not None:
                    found[path.name] = value
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return found


def publish(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    temporary = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def terminal_hash(path):
    """Read the original regular inode once, after its unit is actually terminal."""
    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode)
        assert identity(before) == identity(path.stat())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=physical(path), bytes=before.st_size, sha256=digest.hexdigest(),
                device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                ctime_ns=before.st_ctime_ns)


def metadata_paths(rows):
    found = set()
    for row in rows:
        for raw in row.get('metadata_roots', []):
            root = Path(raw)
            if root.exists():
                for path in root.rglob('*'):
                    if (path.is_file() and not path.is_symlink()
                            and path.suffix in ('.json', '.log', '.txt', '.err', '.out')
                            and not any(part in ('cache', 'sampling', 'features', 'states', 'retrieval-cache') for part in path.relative_to(root).parts)):
                        found.add(path)
    return sorted(found)


def release(rows, known):
    units = []
    for row in rows:
        observed = query(row['unit'])
        assert observed['MainPID'] == '0' and observed['ActiveState'] in ('inactive', 'failed')
        assert not owned('/system.slice/' + row['unit'])
        folder = Path('/sys/fs/cgroup/system.slice') / row['unit']
        if folder.exists():
            assert not {pid for path in folder.rglob('cgroup.procs') for pid in path.read_text().split()}
        units.append(dict(unit=row['unit'], observed=observed))
    remaining = {pid: ticks for pid, ticks in known.items() if birth(pid) == ticks}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    own_gpu = sorted(set(apps.split()) & set(known))
    assert not remaining and not own_gpu
    return dict(epoch=time.time(), original_births_gone=True, own_cgroups_empty=True,
                remaining_original_births=remaining, own_gpu_contexts=own_gpu, units=units,
                foreign_signalled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert sha(args.plan) == args.plan_sha256
    plan = read(args.plan); source = validate_plan(plan)
    assert os.geteuid() == 0 and socket.gethostname().split('.')[0] == plan['node']
    assert read(BASE / 'config.json')['run_id'] == RUN_ID
    root = Path(args.plan).parent
    assert root.is_relative_to(BASE / 'preparation') or root.is_relative_to(PHYSICAL / 'preparation')
    with (root / 'terminal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete, intent = root / 'complete.json', root / 'stop-intent.json'
        if complete.exists():
            value = read(complete)
            assert value['status'] == 'OWN_PREDECESSOR_GENERATION_TERMINAL'
            print(json.dumps(dict(status=value['status'], proof=physical(complete), sha256=sha(complete), existing_proof_read_no_replay=True)))
            return
        if intent.exists():
            print(json.dumps(dict(status='EXISTING_INTENT_OBSERVE_ONLY_NO_STOP_REPLAY',
                intent=read(intent), units=[dict(unit=row['unit'], observed=query(row['unit'])) for row in plan['ordered_units']])))
            return
        rows = plan['ordered_units']; captured = []; known = {}
        for row in rows:
            observed = query(row['unit'])
            assert observed['Id'] == row['unit'] and observed['MainPID'] == str(row['pid'])
            assert observed['ActiveState'] == 'active' and birth(row['pid']) == str(row['birth'])
            group = '/system.slice/' + row['unit']; assert observed['ControlGroup'] == group
            assert group not in Path('/proc/self/cgroup').read_text(), 'Terminal producer must survive all predecessors'
            births = owned(group); assert births.get(str(row['pid'])) == str(row['birth'])
            if row.get('allocation_path'):
                assert sha(row['allocation_path']) == row['allocation_sha256']
                allocation = read(row['allocation_path'])
                assert allocation['controller_pid'] == row['pid'] and allocation['native_unit'] == row['unit']
                assert allocation['node'] == plan['node'] and allocation.get('run_id', RUN_ID) == RUN_ID
            assert not set(known) & set(births)
            known.update(births); captured.append(dict(**row, observed=observed, original_owned_births=births))
        if plan['node'] == 'h100-5':
            assert known.get(str(SOURCE['retriever_pid'])) == SOURCE['retriever_birth']
            assert os.stat('/proc/self/ns/mnt').st_ino == os.stat('/proc/%d/ns/mnt' % SOURCE['pid']).st_ino
            proof = read(plan['remote_node10_proof'])
            assert sha(plan['remote_node10_proof']) == plan['remote_node10_proof_sha256']
            assert proof['status'] == 'OWN_PREDECESSOR_GENERATION_TERMINAL' and proof['node'] == 'h100-10' and proof['run_id'] == RUN_ID
            assert (proof['source_unit'], proof['source_pid'], str(proof['source_birth'])) == (CONSUMER['unit'], CONSUMER['pid'], CONSUMER['birth'])
            assert proof['original_owned_births'].get(str(CONSUMER['pid'])) == CONSUMER['birth']
            assert any(row['unit'] == CONSUMER['unit'] and row['pid'] == CONSUMER['pid']
                       and str(row['birth']) == CONSUMER['birth'] for row in proof['units'])
        if not args.execute:
            print(json.dumps(dict(status='READ_ONLY_COHORT_PLAN_VERIFIED_NO_STOP_INTENT', captured=captured, epoch=time.time())))
            return
        proposed = dict(status='OWN_COHORT_CONTROLLED_ROLLOVER_INTENT', run_id=RUN_ID,
            plan=plan, plan_sha256=args.plan_sha256, captured=captured, original_owned_births=known,
            source_code_sha256=sha(__file__), foreign_signalled=False, epoch=time.time())
        publish(intent, proposed)
        receipts = []
        for row in rows:
            observed = query(row['unit'])
            if observed['MainPID'] != '0':
                assert observed['MainPID'] == str(row['pid']) and birth(row['pid']) == str(row['birth'])
                result = subprocess.run(['systemctl', 'stop', row['unit']], text=True,
                    capture_output=True, timeout=65)
                receipt = dict(unit=row['unit'], returncode=result.returncode, stdout=result.stdout,
                               stderr=result.stderr, observed=query(row['unit']), epoch=time.time())
            else:
                receipt = dict(unit=row['unit'], already_terminal=True, observed=observed, epoch=time.time())
            receipts.append(receipt); publish(root / ('stop-receipt-%03d.json' % len(receipts)), receipt)
            # Never cut a borrowed service after its consumer's stop failed.
            terminal = receipt['observed']
            assert terminal['MainPID'] == '0' and terminal['ActiveState'] in ('inactive', 'failed')
            assert not owned('/system.slice/' + row['unit'])
        observations = [release(rows, known)]
        time.sleep(3); observations.append(release(rows, known))
        records = [terminal_hash(path) for path in metadata_paths(rows)]
        assert records, 'Actual original terminal logs and metadata must be retained'
        value = dict(status='OWN_PREDECESSOR_GENERATION_TERMINAL', run_id=RUN_ID, node=plan['node'],
            source_unit=source['unit'], source_pid=source['pid'], source_birth=source['birth'],
            units=captured, original_owned_births=known, original_files=records,
            original_bytes=sum(row['bytes'] for row in records), release_observations=observations,
            original_partial_outputs_retained=True, old_loops_had_no_drain_hook=True,
            no_zero_partial_loss_claim=True, source_plan=physical(args.plan), source_plan_sha256=args.plan_sha256,
            stop_intent=physical(intent), stop_intent_sha256=sha(intent), stop_receipts=receipts,
            remote_node10_proof=plan.get('remote_node10_proof'), remote_node10_proof_sha256=plan.get('remote_node10_proof_sha256'),
            source_code_sha256=sha(__file__), foreign_signalled=False, global_goal_complete=False, epoch=time.time())
        publish(complete, value)
        print(json.dumps(dict(status=value['status'], proof=physical(complete), sha256=sha(complete),
            original_files=len(records), original_bytes=value['original_bytes'], actual_release_observations=2,
            global_goal_complete=False)))


if __name__ == '__main__':
    main()
