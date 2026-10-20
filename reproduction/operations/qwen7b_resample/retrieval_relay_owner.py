"""Independent bounded CPU relay owners for one future retrieval generation.

Only downstream owns h100-10's primary UUID leases. Models borrow declared
slots; upstream never owns the source GPU processes. Secrets remain remote.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
try:
    import retrieval_only_owner as source
    import retrieval_remote_guard_owner as guard
    import retrieval_relay as relay
except ModuleNotFoundError:
    from . import retrieval_only_owner as source, retrieval_remote_guard_owner as guard, retrieval_relay as relay

shared = source.shared
SLOTS = {
    'tp4-0-3': dict(indices=[0, 1, 2, 3], tp=4, seqs=8, tokens=4096, blocks=1024, reserve=.125),
    'tp4-4-7': dict(indices=[4, 5, 6, 7], tp=4, seqs=8, tokens=4096, blocks=1024, reserve=.125),
    **{'gpu%d-tp1' % index: dict(indices=[index], tp=1, seqs=8, tokens=4096, blocks=1024, reserve=.5)
       for index in (1, 2, 3, 5)},
}
DEFAULT_SLOTS = ('tp4-0-3', 'tp4-4-7')


def same_process(expected):
    actual = source.process(expected['pid'])
    shared.require(all(actual[key] == expected[key] for key in ('birth', 'cgroup', 'netns_inode', 'mntns_inode')),
                   'Original CPU/source process identity differs')
    return actual


def bound_read(pool, path, digest):
    shared.require(pool.sha(path) == digest, 'Immutable ready/allocation SHA differs')
    return pool.read(path)


def source_ready(pool, args, local=False):
    value = bound_read(pool, args.source_owner_ready, args.source_owner_ready_sha256)
    shared.require(value['status'] == 'RETRIEVAL_ONLY_OWNER_READY' and value['run_id'] == pool.RUN_ID,
                   'Future retrieval owner identity differs')
    allocation = bound_read(pool, value['allocation_path'], value['allocation_sha256'])
    shared.require(allocation['owner'] == value['owner'] and allocation['run_id'] == pool.RUN_ID
        and allocation['allocation_role'] == 'RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER', 'Source owner allocation differs')
    contract = value['remote_relay_guard']
    shared.require(contract == allocation['remote_relay_guard'] and contract['contract'] == guard.CONTRACT
        and contract['code_sha256'] == pool.sha(Path(guard.__file__))
        and contract['dependency_path'] == str(guard.dependency_path(args.base, pool.RUN_ID, value['owner'])),
        'Source owner lacks the published relay lifetime guard')
    shared.require(not (Path(args.source_owner_ready).parent / 'drain-request.json').exists(), 'Source owner is draining')
    if local:
        same_process(value['owner']);same_process(value['retriever'])
    return value


def own_evidence(pool, args, node):
    shared.require(socket.gethostname().split('.')[0] == node, 'CPU relay owner is on the wrong node')
    evidence = pool.read(args.allocation_evidence)
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    actual = source.process(os.getpid())
    shared.require(evidence['node'] == node and evidence['main_pid'] == os.getpid()
        and props.get('Id') == evidence['unit'] and props.get('MainPID') == str(os.getpid())
        and props.get('ActiveState') == 'active' and props.get('ControlGroup') == evidence['cgroup']
        and evidence['cgroup'].endswith('/' + evidence['unit'])
        and any(row.split(':', 2)[-1] == evidence['cgroup'] for row in actual['cgroup'].splitlines()),
        'CPU relay owner native unit/PID/cgroup differs')
    shared.require(actual['mntns_inode'] != os.stat('/proc/1/ns/mnt').st_ino
        and actual['netns_inode'] == os.stat('/proc/1/ns/net').st_ino,
        'CPU relay owner requires private retained mounts and host cluster network')
    return evidence, dict(actual, unit=evidence['unit'])


def downstream_dependency(base, run_id, upstream):
    return guard.dependency_path(base, run_id, upstream['source_owner']).parent / (
        '%d-%s' % (upstream['owner']['pid'], upstream['owner']['birth'])) / 'downstream.lock'


def check_terminal(pool, path):
    job = Path(path).parent
    shared.require(not (job / 'complete.json').exists() and not (job / 'failure.json').exists(),
                   'Original CPU relay owner has a terminal receipt')


def verify_predecessor(pool, args):
    proof = bound_read(pool, args.predecessor_proof, args.predecessor_sha256)
    shared.require(proof['status'] == 'OWN_PREDECESSOR_GENERATION_TERMINAL'
        and proof['run_id'] == pool.RUN_ID and proof['node'] == 'h100-10'
        and proof['original_owned_births'] and proof['units'], 'Independent node10 predecessor release proof differs')
    for pid, birth in proof['original_owned_births'].items():
        try:
            actual = source.process(int(pid))
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
        shared.require(actual['birth'] != str(birth), 'Original node10 process is still live')
    for row in proof['units']:
        detail = subprocess.check_output(['systemctl', 'show', row['unit'], '-p', 'Id', '-p', 'MainPID', '-p', 'ActiveState'], text=True, timeout=20)
        props = dict(line.split('=', 1) for line in detail.splitlines() if '=' in line)
        shared.require(props.get('Id') == row['unit'] and props.get('MainPID') == '0'
            and props.get('ActiveState') in ('inactive', 'failed'), 'Original node10 unit is not terminal')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=20)
    shared.require(not set(proof['original_owned_births']) & set(apps.split()), 'Original node10 GPU context remains')
    return dict(path=str(args.predecessor_proof), sha256=args.predecessor_sha256)


def lease_busy(pool, path):
    fd = pool.try_lock(path)
    if fd is None:
        return True
    os.close(fd)
    return False


def health(pool, value):
    shared.require(value['ready'] and value['run_id'] == pool.RUN_ID
        and value['gpu_shards'] == 4 and value['logical_flat_shards'] == 64
        and value['encoder_dtype'] == value['index_dtype'] == 'float16'
        and value['original_global_ids'] and value['endpoint'] == pool.URL,
        'Transparent original full FP16 retrieval health differs')


def controller(pool, args):
    base = Path(args.base)
    pool.configuration(base)
    root = source_ready(pool, args, local=args.role == 'upstream')
    evidence, owner = own_evidence(pool, args, 'h100-5' if args.role == 'upstream' else 'h100-10')
    hard_end = evidence['unit_hard_end_epoch']
    jobdir = base / 'pool/jobs' / ('job-relay-owner-%s-%d' % (args.role, os.getpid())) / ('attempt-' + str(time.time_ns()))
    jobdir.mkdir(parents=True)
    child = stream = None
    locks = []
    try:
        upstream = None
        if args.role == 'upstream':
            address = ipaddress.ip_address(args.listen_host)
            shared.require(address.version == 4 and address.is_private and not address.is_loopback,
                           'Upstream requires an explicit private cluster IPv4 listener')
            dependency = guard.dependency_path(base, pool.RUN_ID, root['owner'])
            fd = pool.try_lock(dependency)
            shared.require(fd is not None, 'This source generation already has a CPU upstream owner')
            locks.append(fd)
            source_ready(pool, args, local=True)
            deadline = min(time.time() + args.owner_seconds, root['deadline_epoch'] - 900, hard_end - 600)
        else:
            upstream = bound_read(pool, args.upstream_owner_ready, args.upstream_owner_ready_sha256)
            shared.require(upstream['status'] == 'RETRIEVAL_RELAY_OWNER_READY' and upstream['role'] == 'upstream'
                and upstream['run_id'] == pool.RUN_ID and upstream['source_owner'] == root['owner']
                and upstream['source_owner_ready_sha256'] == args.source_owner_ready_sha256,
                'Independent upstream generation differs')
            up_allocation = bound_read(pool, upstream['allocation_path'], upstream['allocation_sha256'])
            shared.require(up_allocation['owner'] == upstream['owner'] and up_allocation['source_owner'] == root['owner'],
                           'Upstream allocation differs')
            raw_upstream = bound_read(pool, upstream['transport_ready_path'], upstream['transport_ready_sha256'])
            shared.require(raw_upstream['role'] == 'upstream' and raw_upstream['status'] == 'RELAY_LISTENING'
                and raw_upstream['source_pid'] == root['retriever']['pid']
                and raw_upstream['source_birth'] == root['retriever']['birth']
                and raw_upstream['source_netns_inode'] == root['retriever']['netns_inode']
                and raw_upstream['code_sha256'] == pool.sha(args.relay_script), 'Published upstream transport identity differs')
            check_terminal(pool, upstream['job_ready_path'])
            shared.require(not (Path(upstream['job_ready_path']).parent / 'drain-request.json').exists(), 'Upstream is draining')
            dependency = downstream_dependency(base, pool.RUN_ID, upstream)
            fd = pool.try_lock(dependency)
            shared.require(fd is not None, 'This upstream generation already has a downstream CPU owner')
            locks.append(fd)
            predecessor = verify_predecessor(pool, args)
            declared = pool.read(args.slot_manifest)
            shared.require(set(DEFAULT_SLOTS) <= set(declared) <= set(SLOTS)
                and all(row == SLOTS[name] for name, row in declared.items()),
                'Node10 owner accepts only its explicitly declared finite model slots')
            inventory = {row['index']: row for row in pool.nvml().values()}
            shared.require(set(inventory) == set(range(8)), 'Node10 eight-GPU mapping differs')
            primary = {}
            for uuid in sorted(row['uuid'] for row in inventory.values()):
                fd = pool.try_lock(base / 'pool/gpu-locks' / (uuid + '.lock'))
                shared.require(fd is not None, 'A cooperating queued/native task holds primary UUID ' + uuid)
                locks.append(fd)
                info = os.fstat(fd)
                primary[uuid] = dict(fd=fd, device=info.st_dev, inode=info.st_ino)
            slots = {name: dict(row, cuda_uuids=[inventory[index]['uuid'] for index in row['indices']]) for name, row in declared.items()}
            deadline = min(time.time() + args.owner_seconds, upstream['deadline_epoch'] - 300,
                           root['deadline_epoch'] - 1200, hard_end - 300)
        consumer_deadline = min(root['consumer_deadline_epoch'], deadline - 300,
                                upstream['consumer_deadline_epoch'] if upstream else float('inf'))
        shared.require(consumer_deadline > time.time() + 1320, 'Too little source lifetime for model admission')
        binding = dict(run_id=pool.RUN_ID, allocation_role='RETRIEVAL_RELAY_CPU_OWNER', role=args.role,
            owner=owner, source_owner=root['owner'], source_retriever=root['retriever'],
            source_owner_ready=str(args.source_owner_ready), source_owner_ready_sha256=args.source_owner_ready_sha256,
            source_owner_allocation_sha256=root['allocation_sha256'], controller_pid=os.getpid(),
            native_unit=evidence['unit'], native_cgroup=evidence['cgroup'], model_processes_owned=False,
            deadline_epoch=deadline, consumer_deadline_epoch=consumer_deadline, hard_end_epoch=hard_end,
            dependency_path=str(dependency), allocation_evidence=str(args.allocation_evidence),
            allocation_evidence_sha256=pool.sha(args.allocation_evidence), slots={}, gpu_lock_proofs={})
        if upstream:
            binding.update(slots=slots, gpu_lock_proofs=primary, leased_indices=list(range(8)),
                leased_cuda_uuids=[inventory[index]['uuid'] for index in range(8)], predecessor=predecessor,
                upstream_owner_ready=str(args.upstream_owner_ready), upstream_owner_ready_sha256=args.upstream_owner_ready_sha256)
        allocation = jobdir / 'allocation.json'
        pool.publish(allocation, binding)
        raw_ready = jobdir / 'transport-ready.json'
        command = [sys.executable, '-u', args.relay_script, args.role, '--token-file', args.token_file,
                   '--ready-file', str(raw_ready)]
        if args.role == 'upstream':
            command += ['--listen-host', args.listen_host, '--listen-port', '0', '--source-pid', str(root['retriever']['pid']),
                        '--source-birth', root['retriever']['birth'], '--source-netns-inode', str(root['retriever']['netns_inode'])]
        else:
            command += ['--listen-host', '127.0.0.1', '--listen-port', '28831',
                        '--upstream-ready-file', upstream['transport_ready_path']]
        stream = (jobdir / 'relay.log').open('ab')
        child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        pool.publish(jobdir / 'children.json', dict(own_process_groups=[child.pid], source_gpu_processes_owned=False))
        while not raw_ready.exists():
            shared.require(child.poll() is None and time.time() < consumer_deadline - 1200, 'Owned CPU relay initialization failed')
            time.sleep(.2)
        transport = pool.read(raw_ready)
        shared.require(transport['status'] == 'RELAY_LISTENING' and transport['role'] == args.role
            and transport['source_pid'] == root['retriever']['pid'] and transport['source_birth'] == root['retriever']['birth']
            and transport['source_netns_inode'] == root['retriever']['netns_inode']
            and transport['code_sha256'] == pool.sha(args.relay_script) and transport['process']['pid'] == child.pid,
            'Actual transparent relay process/source/code proof differs')
        child_proof = source.process(child.pid)
        shared.require(all(child_proof[key] == transport['process'][key] for key in ('pid', 'birth', 'cgroup', 'netns_inode'))
            and child_proof['cgroup'] == owner['cgroup'], 'Owned CPU relay child cgroup/identity differs')
        shared.require(transport['listener_netns_inode'] == owner['netns_inode']
            and child_proof['mntns_inode'] == owner['mntns_inode']
            and child_proof['netns_inode'] == (root['retriever']['netns_inode'] if args.role == 'upstream' else owner['netns_inode']),
            'Actual CPU relay listener/backend namespace differs')
        if upstream:
            shared.require(transport['upstream_proof']['sha256'] == pool.sha(upstream['transport_ready_path'])
                and transport['listen_url'] == 'http://127.0.0.1:28831', 'Downstream listener/upstream proof differs')
            with urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=5) as response:
                actual_health = json.load(response)
            health(pool, actual_health)
            shared.require(actual_health['pid'] == root['retriever']['pid'], 'Original retrieval PID differs')
        proof = dict(binding, status='RETRIEVAL_RELAY_OWNER_READY', allocation_path=str(allocation),
            allocation_sha256=pool.sha(allocation), relay=child_proof, transport_ready_path=str(raw_ready),
            transport_ready_sha256=pool.sha(raw_ready), job_ready_path=str(jobdir / 'relay-owner-ready.json'),
            source_processes_owned=False, model_processes_owned=False, epoch=time.time())
        relay.publish_ready(jobdir / 'relay-owner-ready.json', proof)
        relay.publish_ready(args.ready_file, proof)
        print(json.dumps(dict(status=proof['status'], role=args.role, ready_file=args.ready_file,
                             ready_sha256=pool.sha(args.ready_file))), flush=True)
        own_drain = jobdir / 'drain-owner.json'
        source_drain = Path(args.source_owner_ready).parent / 'drain-request.json'
        up_drain = Path(upstream['job_ready_path']).parent / 'drain-request.json' if upstream else None
        while time.time() < deadline and not own_drain.exists() and not source_drain.exists() and not (up_drain and up_drain.exists()):
            shared.require(child.poll() is None, 'Owned CPU relay actually exited')
            if upstream:
                check_terminal(pool, upstream['job_ready_path'])
            else:
                same_process(root['owner']);same_process(root['retriever'])
            time.sleep(1)
        if own_drain.exists():
            request = pool.read(own_drain)
            shared.require(request['status'] == 'DRAIN_RETRIEVAL_OWNER' and request['run_id'] == pool.RUN_ID
                and request['owner_pid'] == owner['pid'] and request['owner_allocation_sha256'] == proof['allocation_sha256'],
                'CPU relay owner drain binding differs')
        for marker, expected_owner, expected_allocation in (
                (source_drain, root['owner'], root['allocation_sha256']),
                (up_drain, upstream['owner'] if upstream else None, upstream['allocation_sha256'] if upstream else None)):
            if marker is not None and marker.exists():
                request = pool.read(marker)
                shared.require(request['status'] == 'DRAIN_RETRIEVAL_OWNER' and request['run_id'] == pool.RUN_ID
                    and request['owner_pid'] == expected_owner['pid'] and request['owner_allocation_sha256'] == expected_allocation,
                    'Original source/upstream drain binding differs')
        pool.publish(jobdir / 'drain-request.json', dict(status='DRAIN_RETRIEVAL_OWNER', run_id=pool.RUN_ID,
            owner_pid=owner['pid'], owner_allocation_sha256=proof['allocation_sha256'], epoch=time.time()))
        while True:
            busy = source.busy_slots(pool, base, binding) if upstream else (
                [dict(relay_dependency='downstream')] if lease_busy(pool, downstream_dependency(base, pool.RUN_ID, proof)) else [])
            if not busy:
                break
            shared.require(child.poll() is None and time.time() < min(hard_end - 120, root['deadline_epoch'] - 120),
                           'CPU relay hard limit reached before dependent leases drained; partial work remains possible')
            pool.publish(jobdir / 'draining.json', dict(active_dependents=busy, epoch=time.time()))
            time.sleep(1)
        pool.stop_group(child)
        pool.publish(jobdir / 'complete.json', dict(status='CPU_RELAY_OWNER_DRAINED', run_id=pool.RUN_ID,
            role=args.role, dependent_leases_released=True, source_processes_signalled=False,
            model_processes_owned=False, relay_returncode=child.returncode, global_goal_complete=False, epoch=time.time()))
        child = None
    except BaseException as error:
        pool.publish(jobdir / 'failure.json', dict(status='CPU_RELAY_OWNER_FAILED_OR_INTERRUPTED', error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), source_processes_signalled=False, global_goal_complete=False))
        raise
    finally:
        if child is not None:pool.stop_group(child)
        if stream is not None:stream.close()
        for fd in reversed(locks):os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('upstream', 'downstream'))
    for key in ('base', 'source-owner-ready', 'source-owner-ready-sha256', 'allocation-evidence', 'token-file', 'ready-file'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--pool-worker', default=str(Path(__file__).with_name('pool_worker.py')))
    parser.add_argument('--relay-script', default=str(Path(__file__).with_name('retrieval_relay.py')))
    parser.add_argument('--owner-seconds', type=int, default=86400)
    parser.add_argument('--listen-host')
    parser.add_argument('--upstream-owner-ready')
    parser.add_argument('--upstream-owner-ready-sha256')
    parser.add_argument('--predecessor-proof')
    parser.add_argument('--predecessor-sha256')
    parser.add_argument('--slot-manifest')
    args = parser.parse_args()
    shared.require(sys.platform == 'linux' and 1800 < args.owner_seconds <= 86400, 'Bounded remote Linux CPU owner only')
    def terminate(signum, frame):raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate);signal.signal(signal.SIGINT, terminate)
    controller(shared.load_pool(args.pool_worker), args)


if __name__ == '__main__':
    main()
