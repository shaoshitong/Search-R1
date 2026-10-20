"""Independent CPU downstream for the still-live, explicitly old V6 source.

Owns only one HTTP listener, no GPU lease or model process. Source5 stays
untouched; its early failure is a real dependency failure, never empty search.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import urllib.request

ROLE = 'current-oldV6-source'
TP1 = {'gpu%d-tp1' % i: dict(indices=[i], tp=1, seqs=64, tokens=8192, blocks=4096, reserve=1.0) for i in (1, 2, 3, 5)}
TOPOLOGIES = {
    'tp2-pairs': dict(TP1, **{'tp2-0-4': dict(indices=[0, 4], tp=2, seqs=16, tokens=4096, blocks=1024, reserve=.5),
        'tp2-6-7': dict(indices=[6, 7], tp=2, seqs=16, tokens=4096, blocks=1024, reserve=.5)}),
    'tp4': dict(TP1, **{'tp4-0-4-6-7': dict(indices=[0, 4, 6, 7], tp=4, seqs=8, tokens=4096, blocks=512, reserve=.125)}),
}


def require(condition, message):
    if not condition:raise ValueError(message)


def process(pid):
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    require(fields[0] not in ('Z', 'X'), 'Original process is terminal')
    return dict(pid=pid, birth=fields[19], cgroup=Path('/proc/%d/cgroup' % pid).read_text(),
        netns_inode=os.stat('/proc/%d/ns/net' % pid).st_ino, mntns_inode=os.stat('/proc/%d/ns/mnt' % pid).st_ino)


def lease_path(base, run_id, owner, slot):
    return Path(base) / 'pool/relay-live-slots' / run_id / ('%d-%s' % (owner['pid'], owner['birth'])) / (slot + '.lock')


def source_binding(pool, args):
    require(pool.sha(args.source_context) == args.source_context_sha256
        and pool.sha(args.upstream_ready) == args.upstream_ready_sha256, 'Retained source/upstream SHA differs')
    context, upstream = pool.read(args.source_context), pool.read(args.upstream_ready)
    require(context['source_main_pid'] == 1490214 and context['source_pid'] == 1490465
        and context['source_unit'] == 'sst-q7b-g5-sharedvram-v6-' + pool.RUN_ID[:12] + '.service',
        'This CPU owner is limited to the explicitly retained old V6 source')
    require(upstream['status'] == 'RELAY_LISTENING' and upstream['role'] == 'upstream'
        and upstream['source_pid'] == context['source_pid'] and upstream['source_birth'] == context['source_birth']
        and upstream['source_netns_inode'] == context['source_netns_inode']
        and upstream['code_sha256'] == pool.sha(Path(args.relay_script).with_name('retrieval_relay.py')),
        'Retained original upstream transport/source identity differs')
    token = Path(args.token_file).lstat()
    require(stat.S_ISREG(token.st_mode) and stat.S_IMODE(token.st_mode) == 0o600 and token.st_uid == os.geteuid(),
            'Retained token must be an owned regular 0600 file')
    return context, upstream


def verify_release(pool, args):
    require(pool.sha(args.predecessor_proof) == args.predecessor_sha256, 'Node10 terminal/release SHA differs')
    proof = pool.read(args.predecessor_proof)
    expected_units = {'sst-q7b-g5-' + name + '-' + pool.RUN_ID[:12] + '.service' for name in
        ('consumer10-v5', 'consumer10-parallel-v1', 'consumer10-tp1-gpu1-v1', 'consumer10-tp1-gpu2-v1')}
    require(proof['status'] == 'OWN_NODE10_NATIVE_FAILURES_ORIGINAL_SHA_RELEASE_PASS'
        and proof['predecessor_compatibility_status'] == 'OWN_PREDECESSOR_GENERATION_TERMINAL'
        and proof['node'] == 'h100-10' and proof['run_id'] == pool.RUN_ID
        and {row['unit'] for row in proof['units']} == expected_units
        and proof['original_owned_births'] and proof['original_partial_outputs_retained'], 'Node10 failed-cohort terminal/release proof differs')
    observations = proof['release_observations']
    require(len(observations) >= 2 and observations[-1]['epoch'] - observations[0]['epoch'] >= 3
        and all(row['original_births_gone'] and row['own_cgroups_empty'] and not row['remaining_known_pids']
            and not row['own_gpu_contexts'] and not row['loopback_28831_listeners'] for row in observations),
        'Actual failed-cohort release was not observed twice')
    for pid in proof['pids_without_recorded_historical_birth']:
        require(not Path('/proc/%d' % int(pid)).exists(), 'Original PID without historical birth must still be absent')
    for pid, birth in proof['original_owned_births'].items():
        try:actual = process(int(pid))
        except (FileNotFoundError, ProcessLookupError, ValueError):continue
        require(actual['birth'] != str(birth), 'Original node10 generation process remains live')
    for row in proof['units']:
        detail = subprocess.check_output(['systemctl', 'show', row['unit'], '-p', 'Id', '-p', 'MainPID', '-p', 'ActiveState'], text=True, timeout=15)
        props = dict(line.split('=', 1) for line in detail.splitlines() if '=' in line)
        require(props.get('Id') == row['unit'] and props.get('MainPID') == '0' and props.get('ActiveState') in ('inactive', 'failed'),
                'Original node10 unit is not actually terminal')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    require(not set(apps.split()) & set(proof['original_owned_births']), 'An original node10 GPU context remains')


def own_binding(pool, args):
    require(socket.gethostname().split('.')[0] == 'h100-10', 'Independent CPU downstream is restricted to h100-10')
    evidence = pool.read(args.allocation_evidence)
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    actual = process(os.getpid())
    require(evidence['node'] == 'h100-10' and evidence['main_pid'] == os.getpid()
        and props.get('Id') == evidence['unit'] and props.get('MainPID') == str(os.getpid())
        and props.get('ActiveState') == 'active' and props.get('ControlGroup') == evidence['cgroup']
        and evidence['cgroup'].endswith('/' + evidence['unit'])
        and any(line.split(':', 2)[-1] == evidence['cgroup'] for line in actual['cgroup'].splitlines()), 'CPU owner unit/PID/cgroup differs')
    require(actual['netns_inode'] == os.stat('/proc/1/ns/net').st_ino
        and actual['mntns_inode'] != os.stat('/proc/1/ns/mnt').st_ino, 'CPU owner requires hostNET/private retained mounts')
    return evidence, dict(actual, unit=evidence['unit'])


def source_health(pool, source_pid):
    with urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=60) as response:value = json.load(response)
    require(value['ready'] and value['pid'] == source_pid and value['run_id'] == pool.RUN_ID
        and value['gpu_shards'] == 4 and value['logical_flat_shards'] == 64
        and value['encoder_dtype'] == value['index_dtype'] == 'float16' and value['original_global_ids']
        and value['endpoint'] == pool.URL, 'Actual original full FP16 retrieval health differs')
    return value


def busy_slots(pool, base, owner, slots):
    busy = []
    for name in slots:
        fd = pool.try_lock(lease_path(base, pool.RUN_ID, owner, name))
        if fd is None:busy.append(name)
        else:os.close(fd)
    return busy


def controller(pool, args):
    base = Path(args.base);pool.configuration(base)
    context, upstream = source_binding(pool, args)
    evidence, owner = own_binding(pool, args)
    verify_release(pool, args)
    slots = pool.read(args.slot_manifest)
    require(any(slots == rows for rows in TOPOLOGIES.values()), 'CPU owner accepts only an explicit disjoint finite slot topology')
    with socket.socket() as check:check.bind(('127.0.0.1', 28831))
    deadline = min(time.time() + args.owner_seconds, context['source_sampling_deadline_epoch'] - 900,
                   evidence['unit_hard_end_epoch'] - 300)
    require(deadline > time.time() + 1800, 'Too little old source lifetime for an independent CPU relay')
    job = base / 'pool/jobs' / ('job-legacy-relay-owner-' + str(os.getpid())) / ('attempt-' + str(time.time_ns()))
    job.mkdir(parents=True)
    allocation = job / 'allocation.json';transport = job / 'transport-ready.json'
    binding = dict(status='LEGACY_SOURCE_RELAY_OWNER_ALLOCATION', role=ROLE, node='h100-10', run_id=pool.RUN_ID,
        owner=owner, controller_pid=os.getpid(), native_unit=evidence['unit'], native_cgroup=evidence['cgroup'],
        source_context_path=str(args.source_context), source_context_sha256=args.source_context_sha256,
        upstream_ready_path=str(args.upstream_ready), upstream_ready_sha256=args.upstream_ready_sha256,
        source_context=context, source_upstream_process=upstream['process'], token_file=str(args.token_file),
        token_file_sha256=pool.sha(args.token_file), slots=slots, deadline_epoch=deadline, consumer_deadline_epoch=deadline - 600,
        hard_end_epoch=evidence['unit_hard_end_epoch'], source_may_exit_early=True,
        source_gpu_processes_owned=False, model_processes_owned=False, gpu_primary_leases_owned=False,
        allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence))
    pool.publish(allocation, binding)
    child = stream = None
    try:
        command = [sys.executable, '-u', args.relay_script, 'downstream', '--listen-host', '127.0.0.1', '--listen-port', '28831',
            '--token-file', args.token_file, '--upstream-ready-file', args.upstream_ready, '--ready-file', str(transport),
            '--max-workers', '8', '--max-queued', '128', '--queue-wait', '45', '--request-budget', '59', '--listen-backlog', '256', '--timeout', '50',
            '--stats-file', str(job / 'backpressure-stats.json')]
        stream = (job / 'relay.log').open('ab')
        child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        pool.publish(job / 'children.json', dict(own_process_groups=[child.pid], source_gpu_processes_owned=False, model_processes_owned=False))
        while not transport.exists():
            require(child.poll() is None and time.time() < deadline - 1200, 'Independent downstream failed before ready')
            time.sleep(.2)
        raw = pool.read(transport);actual = process(child.pid)
        require(raw['status'] == 'RELAY_LISTENING' and raw['role'] == 'downstream' and raw['listen_url'] == 'http://127.0.0.1:28831'
            and raw['source_pid'] == context['source_pid'] and raw['source_birth'] == context['source_birth']
            and raw['source_netns_inode'] == context['source_netns_inode']
            and raw['upstream_proof']['sha256'] == args.upstream_ready_sha256
            and raw['code_sha256'] == pool.sha(args.relay_script)
            and raw['transport_core_sha256'] == pool.sha(Path(args.relay_script).with_name('retrieval_relay.py'))
            and raw['bounds']['max_workers'] == 8 and raw['bounds']['timeout'] == 50
            and raw['backpressure']['max_queued'] == 128 and raw['backpressure']['queue_wait'] == 45
            and raw['backpressure']['request_budget'] == 59
            and raw['backpressure']['fixed_worker_pool'] and raw['backpressure']['no_empty_retrieval_fallback']
            and all(actual[key] == raw['process'][key] for key in ('pid', 'birth', 'cgroup', 'netns_inode'))
            and actual['cgroup'] == owner['cgroup'] and actual['netns_inode'] == owner['netns_inode']
            and actual['mntns_inode'] == owner['mntns_inode'], 'Actual owned downstream/source/code proof differs')
        health = source_health(pool, context['source_pid'])
        ready = dict(binding, status='LEGACY_SOURCE_RELAY_OWNER_READY', relay=actual,
            allocation_path=str(allocation), allocation_sha256=pool.sha(allocation), job_ready_path=str(job / 'owner-ready.json'),
            owner_drain_path=str(job / 'drain-owner.json'),
            transport_ready_path=str(transport), transport_ready_sha256=pool.sha(transport),
            code_sha256=pool.sha(Path(__file__)), ready_epoch=time.time())
        pool.publish(job / 'health.json', health)
        require(not Path(args.ready_file).exists(), 'Independent CPU owner ready receipt already exists')
        pool.publish(job / 'owner-ready.json', ready);pool.publish(args.ready_file, ready)
        print(json.dumps(dict(status=ready['status'], ready_file=args.ready_file, ready_sha256=pool.sha(args.ready_file))), flush=True)
        marker = job / 'drain-owner.json'
        while time.time() < deadline and not marker.exists():
            require(child.poll() is None, 'Independent downstream actually exited')
            source_health(pool, context['source_pid'])
            time.sleep(5)
        if marker.exists():
            request = pool.read(marker)
            require(request['status'] == 'DRAIN_RETRIEVAL_OWNER' and request['run_id'] == pool.RUN_ID
                and request['owner_pid'] == owner['pid'] and request['owner_allocation_sha256'] == ready['allocation_sha256'], 'CPU relay drain identity differs')
        drain = dict(status='DRAIN_RETRIEVAL_OWNER', run_id=pool.RUN_ID,
            owner_pid=owner['pid'], owner_allocation_sha256=ready['allocation_sha256'], epoch=time.time())
        if not marker.exists():pool.publish(marker, drain)
        pool.publish(job / 'drain-request.json', drain)
        while True:
            busy = busy_slots(pool, base, owner, slots)
            if not busy:break
            require(child.poll() is None and time.time() < min(context['source_sampling_deadline_epoch'] - 120,
                evidence['unit_hard_end_epoch'] - 120), 'CPU relay hard/source deadline reached while models still hold live leases')
            source_health(pool, context['source_pid'])
            pool.publish(job / 'draining.json', dict(active_model_slots=busy, epoch=time.time()))
            time.sleep(5)
        pool.stop_group(child)
        pool.publish(job / 'complete.json', dict(status='LEGACY_SOURCE_CPU_RELAY_DRAINED', role=ROLE, run_id=pool.RUN_ID,
            model_live_leases_released=True, source_gpu_processes_signalled=False, models_signalled=False,
            relay_returncode=child.returncode, global_goal_complete=False, epoch=time.time()))
        child = None
    except BaseException as error:
        pool.publish(job / 'failure.json', dict(status='LEGACY_SOURCE_CPU_RELAY_FAILED', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), source_gpu_processes_signalled=False, models_signalled=False,
            global_goal_complete=False, partial_batches_may_remain=True, epoch=time.time()))
        raise
    finally:
        if child is not None:pool.stop_group(child)
        if stream is not None:stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'source-context', 'source-context-sha256', 'upstream-ready', 'upstream-ready-sha256',
                 'token-file', 'allocation-evidence', 'predecessor-proof', 'predecessor-sha256', 'slot-manifest', 'ready-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--owner-seconds', type=int, default=14400)
    parser.add_argument('--pool-worker', default=str(Path(__file__).with_name('pool_worker.py')))
    parser.add_argument('--relay-script', default=str(Path(__file__).with_name('retrieval_relay_backpressure.py')))
    args = parser.parse_args()
    require(sys.platform == 'linux' and 1800 < args.owner_seconds <= 14400, 'Bounded remote Linux CPU owner only')
    spec = importlib.util.spec_from_file_location('legacy_cpu_relay_pool', args.pool_worker)
    pool = importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)
    def terminate(signum, frame):raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate);signal.signal(signal.SIGINT, terminate)
    controller(pool, args)


if __name__ == '__main__':main()
