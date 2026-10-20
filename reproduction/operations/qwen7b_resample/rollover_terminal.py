"""Observe an explicit replacement cohort's real terminal states; never signal.

A live capture binds original births/allocations before draining. The collector
can wait in the old mount namespace outside all predecessor cgroups, then seals
two real releases and original metadata inodes. Old V5 failures stay failures.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
try:
    import predecessor_generation_terminal as core
    import rollover_paths as paths
except ModuleNotFoundError:
    from . import predecessor_generation_terminal as core
    from . import rollover_paths as paths

STATUS = 'OWN_BOUND_COHORT_TERMINAL_RELEASED_V1'
FAILED = 'OWN_NODE10_NATIVE_FAILURES_ORIGINAL_SHA_RELEASE_PASS'
RUN_ID, BASE, PHYSICAL = core.RUN_ID, core.BASE, core.PHYSICAL
SOURCE_MODELS = (
    'source5-extra-gpu1-v3', 'source5-extra-gpu2-v3',
    'source5-tp1-replacement-gpu3-v1', 'source5-tp1-replacement-gpu4-v1',
    'source5-tp1-replacement-gpu6-v1', 'source5-tp1-replacement-gpu7-v1',
    'source5-extra-retrieval-v1', 'sharedvram-v6')
PRIMARY_INSTANCES = {
    'gpu1-tp1': 'v2', 'gpu2-tp1': 'v3', 'gpu3-tp1': 'v2',
    'gpu5-tp1': 'v3', 'tp4-0-4-6-7': 'v3'}


def unit(name):return 'sst-q7b-g5-' + name + '-' + RUN_ID[:12] + '.service'


def readable(path):return paths.readable(path, BASE, PHYSICAL)
def read(path):return paths.read(path, BASE, PHYSICAL)
def sha(path):return paths.sha(path, BASE, PHYSICAL)


def primary_instances(value, slots):
    """Bind explicit instances, never infer a replacement from a failed unit."""
    assert isinstance(value, dict) and set(value) == set(slots) == set(PRIMARY_INSTANCES)
    assert all(re.fullmatch(r'v(?:[2-9]|[1-9][0-9]+)', instance) for instance in value.values())
    return value


def validate_plan(value):
    assert value['schema_version'] == 2 and value['run_id'] == RUN_ID
    node, rows = value['node'], value['ordered_units']
    assert node in ('h100-5', 'h100-10') and rows
    assert len({row['unit'] for row in rows}) == len(rows) == len({row['pid'] for row in rows})
    positions = {row['unit']: i for i, row in enumerate(rows)}
    for row in rows:
        assert row['node'] == node and row['unit'].startswith('sst-q7b-') and row['unit'].endswith(RUN_ID[:12] + '.service')
        assert row['pid'] > 1 and str(row['birth']).isdigit()
        assert row['original_owned_births'].get(str(row['pid'])) == str(row['birth'])
        assert all(str(pid).isdigit() and str(birth).isdigit() for pid, birth in row['original_owned_births'].items())
        for dependency in row.get('borrows_units', []):assert dependency in positions and positions[row['unit']] < positions[dependency]
        assert row['allocation_path'] and re.fullmatch(r'[0-9a-f]{64}', row['allocation_sha256'])
        assert row['metadata_roots']
        for path in [row['allocation_path'], *row['metadata_roots']]:
            p = Path(path)
            assert p.is_absolute() and '..' not in p.parts and (p.is_relative_to(BASE) or p.is_relative_to(PHYSICAL))
            assert not any(part in ('sampling', 'assets', 'cache') for part in p.relative_to(BASE if p.is_relative_to(BASE) else PHYSICAL).parts)
    source = value['source']
    assert all(rows[-1][key] == source[key] for key in ('unit', 'pid', 'birth'))
    assert value['original_partial_outputs_retained'] and value['collector_signals_no_processes']
    if node == 'h100-10':
        assert value['cohort_kind'] == 'legacy-cpu-and-primary-mixed'
        ready = value['cpu_owner_ready']
        assert ready['status'] == 'LEGACY_SOURCE_RELAY_OWNER_READY' and ready['role'] == 'current-oldV6-source'
        assert ready['owner']['unit'] == source['unit'] and ready['owner']['pid'] == source['pid'] and ready['owner']['birth'] == source['birth']
        instances = primary_instances(value['primary_instances'], ready['slots'])
        expected = {unit('primary10-' + name + '-' + instance) for name, instance in instances.items()}
        assert {row['unit'] for row in rows[:-1]} == expected
        assert source['unit'] == unit('cpu-relay10-v1')
        assert all(row['borrows_units'] == [source['unit']] for row in rows[:-1])
        assert value['prior_failed_proof_status'] == FAILED and value['prior_failed_proof_path'] and value['prior_failed_proof_sha256']
    else:
        assert value['cohort_kind'] == 'source5-v6-eight-models'
        assert source == dict(unit=core.SOURCE['unit'], pid=1490214, birth='6807965')
        assert rows[-1]['original_owned_births'].get('1490465') == '6808057'
        expected = {unit(name) for name in SOURCE_MODELS} | {'sst-q7b-retrieval-relay5-v1-' + RUN_ID[:12] + '.service'}
        assert {row['unit'] for row in rows} == expected
        assert value['source_mount_ns_inode'] > 0 and value['source_net_ns_inode'] > 0
        assert value['remote_node10_plan_path'] and value['remote_node10_plan_sha256'] and value['remote_node10_proof_path']
    return value


def verify_allocation(row):
    assert sha(row['allocation_path']) == row['allocation_sha256']
    allocation = read(row['allocation_path'])
    if row.get('allocation_kind') == 'retained-upstream-ready':
        assert allocation['role'] == 'upstream' and allocation['status'] == 'RELAY_LISTENING'
        assert allocation['process']['pid'] == row['pid'] and str(allocation['process']['birth']) == row['birth']
        assert '/system.slice/' + row['unit'] in allocation['process']['cgroup']
        assert allocation['source_pid'] == 1490465 and allocation['source_birth'] == '6808057'
        return
    assert allocation['controller_pid'] == row['pid'] and allocation['native_unit'] == row['unit']
    assert allocation['node'] == row['node'] and allocation.get('run_id', RUN_ID) == RUN_ID


def captured_row(name, node, allocation, metadata_roots, borrows=(), allocation_kind='model-or-cpu-owner'):
    props = core.query(name);pid = int(props['MainPID']);birth = core.birth(pid)
    assert props['Id'] == name and props['ActiveState'] == 'active' and pid > 1 and str(birth).isdigit()
    group = '/system.slice/' + name
    assert props['ControlGroup'] == group and group not in Path('/proc/self/cgroup').read_text()
    births = core.owned(group);assert births.get(str(pid)) == birth
    row = dict(unit=name, node=node, pid=pid, birth=birth, borrows_units=list(borrows),
        allocation_path=str(allocation), allocation_sha256=sha(allocation),
        metadata_roots=list(map(str, metadata_roots)), original_owned_births=births, captured_unit=props,
        allocation_kind=allocation_kind)
    verify_allocation(row)
    return row


def verify_prior_failed(plan):
    assert sha(plan['prior_failed_proof_path']) == plan['prior_failed_proof_sha256']
    value = read(plan['prior_failed_proof_path'])
    assert value['status'] == FAILED and value['node'] == 'h100-10' and value['run_id'] == RUN_ID
    assert value['predecessor_compatibility_status'] == 'OWN_PREDECESSOR_GENERATION_TERMINAL'
    assert value['termination_kind'] == 'OBSERVED_NATIVE_FAILURE_NOT_CONTROLLED_STOP' and value['controlled_stop'] is False
    assert len(value['release_observations']) >= 2
    assert value['release_observations'][-1]['epoch'] - value['release_observations'][0]['epoch'] >= 3
    assert all(not row['remaining_known_pids'] and not row['own_gpu_contexts']
        and not row['loopback_28831_listeners'] and row['own_cgroups_empty'] for row in value['release_observations'])


def metadata(rows):
    visible_rows = [dict(row, metadata_roots=[str(readable(path)) for path in row['metadata_roots']]) for row in rows]
    paths = set(core.metadata_paths(visible_rows))
    for row in rows:
        for raw in row['metadata_roots']:
            root = readable(raw)
            if root.exists():
                paths.update(path for path in root.rglob('*') if path.is_file() and not path.is_symlink()
                    and path.suffix in ('.py', '.sh') and not any(x in ('cache', 'sampling', 'assets') for x in path.relative_to(root).parts))
    return sorted(paths)


def release(plan):
    rows = plan['ordered_units'];known = {}
    for row in rows:
        assert not set(known) & set(row['original_owned_births'])
        known.update(row['original_owned_births'])
    result = core.release(rows, known)
    if plan['node'] == 'h100-10':
        # A successful local bind is a fresh absence check; it never stops or
        # replaces another listener. The socket is closed immediately.
        with socket.socket() as probe:probe.bind(('127.0.0.1', 28831))
        result['loopback_28831_unbound'] = True
    return result


def verify_proof(path, digest, expected_node, expected_plan_sha=None):
    assert sha(path) == digest
    value = read(path)
    assert value['status'] == STATUS and value['schema_version'] == 2 and value['node'] == expected_node and value['run_id'] == RUN_ID
    assert value['collector_signals_no_processes'] and value['controlled_stop'] is False
    assert value['termination_kind'] == 'OBSERVED_TERMINAL_STATES_NO_SIGNALS_BY_COLLECTOR'
    assert sha(value['source_plan']) == value['source_plan_sha256']
    plan = validate_plan(read(value['source_plan']))
    if expected_plan_sha is not None:assert value['source_plan_sha256'] == expected_plan_sha
    assert value['source'] == plan['source'] and value['ordered_units'] == plan['ordered_units']
    assert value['original_owned_births'] == {pid: birth for row in plan['ordered_units'] for pid, birth in row['original_owned_births'].items()}
    observations = value['release_observations']
    assert len(observations) >= 2 and observations[-1]['epoch'] - observations[0]['epoch'] >= 3
    assert all(row['original_births_gone'] and row['own_cgroups_empty'] and not row['remaining_original_births']
        and not row['own_gpu_contexts'] for row in observations)
    expected_units = {row['unit'] for row in plan['ordered_units']}
    assert all({row['unit'] for row in observation['units']} == expected_units
        and all(row['observed']['MainPID'] == '0' and row['observed']['ActiveState'] in ('inactive', 'failed')
            for row in observation['units']) for observation in observations)
    if expected_node == 'h100-10':
        assert all(row['loopback_28831_unbound'] for row in observations)
        verify_prior_failed(plan)
    else:
        assert value['source']['pid'] == 1490214 and value['source']['birth'] == '6807965'
        assert sha(value['remote_node10_proof']) == value['remote_node10_proof_sha256']
        verify_proof(value['remote_node10_proof'], value['remote_node10_proof_sha256'], 'h100-10', plan['remote_node10_plan_sha256'])
    return value


def verify_source_proof(path, digest, base, run_id):
    assert str(base) == str(BASE) and run_id == RUN_ID
    value = verify_proof(path, digest, 'h100-5')
    # Recheck only this node. Never interpret a remote node10 PID locally.
    plan = read(value['source_plan'])
    release(plan)
    return value


def collect(plan_path, plan_sha, watch_seconds=0):
    plan_path = readable(plan_path)
    assert sha(plan_path) == plan_sha
    plan = validate_plan(read(plan_path))
    assert socket.gethostname().split('.')[0] == plan['node'] and os.geteuid() == 0
    assert read(BASE / 'config.json')['run_id'] == RUN_ID
    assert 0 <= watch_seconds <= 7200
    if plan['node'] == 'h100-5':
        assert os.stat('/proc/self/ns/mnt').st_ino == plan['source_mount_ns_inode']
        assert os.stat('/proc/self/ns/net').st_ino == plan['source_net_ns_inode']
    else:
        assert sha(plan['cpu_owner_ready_path']) == plan['cpu_owner_ready_sha256']
        assert read(plan['cpu_owner_ready_path']) == plan['cpu_owner_ready']
        assert plan['ordered_units'][-1]['allocation_sha256'] == plan['cpu_owner_ready']['allocation_sha256']
    for row in plan['ordered_units']:
        assert '/system.slice/' + row['unit'] not in Path('/proc/self/cgroup').read_text()
        verify_allocation(row)
    root = Path(plan_path).parent;complete = root / 'complete.json'
    with (root / 'collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete.exists():return verify_proof(complete, sha(complete), plan['node'])
        deadline = time.monotonic() + watch_seconds
        while True:
            try:
                remote = None
                if plan['node'] == 'h100-10':verify_prior_failed(plan)
                else:
                    remote = plan['remote_node10_proof_path']
                    verify_proof(remote, sha(remote), 'h100-10', plan['remote_node10_plan_sha256'])
                first = release(plan)
                break
            except (AssertionError, FileNotFoundError):
                if time.monotonic() >= deadline:raise
                time.sleep(2)
        time.sleep(3);second = release(plan)
        originals = [core.terminal_hash(path) for path in metadata(plan['ordered_units'])]
        assert originals
        value = dict(status=STATUS, schema_version=2, run_id=RUN_ID, node=plan['node'], source=plan['source'],
            source_unit=plan['source']['unit'], source_pid=plan['source']['pid'], source_birth=plan['source']['birth'],
            ordered_units=plan['ordered_units'], original_owned_births={pid: birth for row in plan['ordered_units'] for pid, birth in row['original_owned_births'].items()},
            release_observations=[first, second], original_files=originals, original_bytes=sum(row['bytes'] for row in originals),
            original_partial_outputs_retained=True, collector_signals_no_processes=True, controlled_stop=False,
            termination_kind='OBSERVED_TERMINAL_STATES_NO_SIGNALS_BY_COLLECTOR', global_goal_complete=False,
            source_plan=core.physical(plan_path), source_plan_sha256=plan_sha,
            remote_node10_proof=core.physical(remote) if remote else None, remote_node10_proof_sha256=sha(remote) if remote else None,
            source_code_sha256=sha(__file__), epoch=time.time())
        core.publish(complete, value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True);parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--collect-terminal', action='store_true');parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--log-file')
    args = parser.parse_args()
    if args.log_file:
        log = readable(Path(args.log_file).parent) / Path(args.log_file).name
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, 1);os.dup2(fd, 2);os.close(fd)
    assert sha(args.plan) == args.plan_sha256
    if not args.collect_terminal:
        plan = validate_plan(read(args.plan))
        print(json.dumps(dict(status='READ_ONLY_BOUND_COHORT_PLAN_NO_SIGNALS', plan=plan)));return
    value = collect(args.plan, args.plan_sha256, args.watch_seconds)
    print(json.dumps(dict(status=value['status'], proof=core.physical(Path(args.plan).parent / 'complete.json'),
                         sha256=sha(Path(args.plan).parent / 'complete.json'), collector_signals_no_processes=True)))


if __name__ == '__main__':main()
