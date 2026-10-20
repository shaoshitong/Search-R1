"""CPU-only future relay ownership, finite-slot binding and opaque transport."""
import argparse
import ast
import contextlib
import fcntl
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import retrieval_relay_generation_control as control
from . import retrieval_remote_guard_control as guarded_control

DIRECTORY = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):spec.loader.exec_module(module)
    return module


RUNTIME = load('cpu_future_relay_owner', 'retrieval_relay_owner.py')
SLOT = load('cpu_future_relay_slot', 'relay_slot_consumer_worker.py')


def read(path):return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:os.close(fd);return None
    return fd


class RelayGenerationTests(unittest.TestCase):
    def test_guard_waits_a_real_independent_posix_dependency_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);owner = dict(pid=900, birth='111')
            path = RUNTIME.guard.dependency_path(base, control.RUN_ID, owner)
            path.parent.mkdir(parents=True)
            program = "import fcntl,os,sys;fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600);fcntl.lockf(fd,fcntl.LOCK_EX);print('LOCKED',flush=True);sys.stdin.readline()"
            pool = SimpleNamespace(try_lock=lock, RUN_ID=control.RUN_ID)
            with subprocess.Popen([sys.executable, '-u', '-c', program, str(path)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
                self.assertEqual(child.stdout.readline().strip(), 'LOCKED')
                self.assertEqual(RUNTIME.guard.busy_slots(pool, base, dict(owner=owner, slots={})),
                                 [dict(relay_dependency='upstream', owner_pid=900)])
                child.communicate('\n', timeout=5)
            self.assertEqual(RUNTIME.guard.busy_slots(pool, base, dict(owner=owner, slots={})), [])

    @contextlib.contextmanager
    def fixture(self, role):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);run = control.RUN_ID
            root_owner = dict(pid=900, birth='111', unit='future-source.service', cgroup='0::/system.slice/future-source.service\n', netns_inode=22, mntns_inode=33)
            retriever = dict(pid=901, birth='112', cgroup=root_owner['cgroup'], netns_inode=22, mntns_inode=33)
            source_path = base / 'source/owner-ready.json';source_allocation = source_path.parent / 'allocation.json'
            contract = dict(contract=RUNTIME.guard.CONTRACT, code_sha256=sha(RUNTIME.guard.__file__),
                dependency_path=str(RUNTIME.guard.dependency_path(base, run, root_owner)))
            source_binding = dict(allocation_role='RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER', run_id=run,
                owner=root_owner, remote_relay_guard=contract)
            publish(source_allocation, source_binding)
            root = dict(status='RETRIEVAL_ONLY_OWNER_READY', run_id=run, owner=root_owner, retriever=retriever,
                allocation_path=str(source_allocation), allocation_sha256=sha(source_allocation),
                remote_relay_guard=contract, deadline_epoch=55000, consumer_deadline_epoch=50000)
            publish(source_path, root)
            upstream_owner = dict(pid=700, birth='113', unit='future-upstream.service',
                cgroup='0::/system.slice/future-upstream.service\n', netns_inode=11, mntns_inode=77)
            upstream_path = base / 'upstream/relay-owner-ready.json';upstream_allocation = upstream_path.parent / 'allocation.json'
            upstream_transport = upstream_path.parent / 'transport-ready.json'
            upstream_relay = dict(pid=701, birth='114', cgroup=upstream_owner['cgroup'], netns_inode=22)
            raw_upstream = dict(status='RELAY_LISTENING', role='upstream', source_pid=901, source_birth='112', source_netns_inode=22,
                listen_url='http://10.0.0.5:4123', listen=dict(host='10.0.0.5', port=4123), process=upstream_relay,
                code_sha256=sha(DIRECTORY / 'retrieval_relay.py'))
            publish(upstream_transport, raw_upstream)
            publish(upstream_allocation, dict(owner=upstream_owner, source_owner=root_owner))
            upstream = dict(status='RETRIEVAL_RELAY_OWNER_READY', role='upstream', run_id=run, owner=upstream_owner,
                source_owner=root_owner, source_owner_ready_sha256=sha(source_path), source_retriever=retriever,
                allocation_path=str(upstream_allocation), allocation_sha256=sha(upstream_allocation),
                transport_ready_path=str(upstream_transport), transport_ready_sha256=sha(upstream_transport),
                job_ready_path=str(upstream_path), deadline_epoch=49000, consumer_deadline_epoch=48000)
            publish(upstream_path, upstream)
            unit = 'future-%s-owner.service' % role;group = '/system.slice/' + unit
            own = dict(pid=123, birth='115', cgroup='0::' + group + '\n', netns_inode=11, mntns_inode=111)
            child_proof = dict(pid=500, birth='116', cgroup=own['cgroup'], netns_inode=22 if role == 'upstream' else 11, mntns_inode=111)
            evidence = base / 'evidence.json'
            publish(evidence, dict(node='h100-5' if role == 'upstream' else 'h100-10', main_pid=123, unit=unit,
                cgroup=group, unit_show='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (unit, group),
                unit_hard_end_epoch=100000, captured_epoch=1000, resource_limits=dict(runtime_max_seconds=90000)))
            predecessor = base / 'node10-release.json'
            publish(predecessor, dict(status='OWN_PREDECESSOR_GENERATION_TERMINAL', run_id=run, node='h100-10',
                original_owned_births={'10': 'old-node10-birth'}, units=[dict(unit='retired-node10.service')]))
            slots = base / 'slots.json';publish(slots, {name: RUNTIME.SLOTS[name] for name in RUNTIME.DEFAULT_SLOTS})
            args = argparse.Namespace(base=str(base), role=role, source_owner_ready=str(source_path),
                source_owner_ready_sha256=sha(source_path), allocation_evidence=str(evidence),
                owner_seconds=4000, listen_host='10.0.0.5', upstream_owner_ready=str(upstream_path),
                upstream_owner_ready_sha256=sha(upstream_path), predecessor_proof=str(predecessor),
                predecessor_sha256=sha(predecessor), slot_manifest=str(slots),
                relay_script=str(DIRECTORY / 'retrieval_relay.py'), token_file=str(base / 'remote-only-secret'),
                ready_file=str(base / 'published/owner-ready.json'))
            Path(args.ready_file).parent.mkdir()
            clock, sleeps, stopped, calls, descriptors = [1000], [], [], [], []
            model_busy = [True]
            pool = SimpleNamespace(read=read, publish=publish, sha=sha, RUN_ID=run,
                URL='http://127.0.0.1:28831/retrieve', configuration=lambda _: {},
                nvml=lambda: {'GPU-%d' % i: dict(index=i, uuid='GPU-%d' % i, free_bytes=22 * 1024 ** 3) for i in range(8)})

            def lease(path):
                if model_busy[0] and ((role == 'downstream' and 'gpu-slot-leases' in str(path) and path.name == 'GPU-0.lock')
                    or (role == 'upstream' and path.name == 'downstream.lock')):
                    return None
                fd = lock(path)
                if fd is not None:descriptors.append(fd)
                return fd

            pool.try_lock = lease
            class RelayProcess:
                pid = 500
                returncode = None
                def poll(self):return self.returncode
            child = RelayProcess()
            def spawn(command, **kwargs):
                calls.append(command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                raw_ready = Path(command[command.index('--ready-file') + 1])
                raw = dict(status='RELAY_LISTENING', role=role, source_pid=901, source_birth='112', source_netns_inode=22,
                    process={k: child_proof[k] for k in ('pid', 'birth', 'cgroup', 'netns_inode')}, listener_netns_inode=11,
                    code_sha256=sha(args.relay_script), upstream_proof=dict(sha256=sha(upstream_transport)),
                    listen_url='http://10.0.0.5:4123' if role == 'upstream' else 'http://127.0.0.1:28831')
                publish(raw_ready, raw)
                return child
            def stop(process):
                self.assertFalse(model_busy[0], 'The owner must wait for its dependent lease')
                stopped.append(process.pid);process.returncode = 0
            pool.stop_group = stop
            def sleep(seconds):
                sleeps.append(clock[0])
                if clock[0] >= 5000:model_busy[0] = False;clock[0] += 1
                else:clock[0] += 2000
            processes = {123: own, 500: child_proof, 900: root_owner, 901: retriever}
            observed_pids = []
            def process(pid):
                observed_pids.append(pid)
                if pid == 10:raise FileNotFoundError('retired local process')
                return dict(processes[pid])
            def command(argv, **kwargs):
                if argv[0] == 'systemctl':return 'Id=retired-node10.service\nMainPID=0\nActiveState=inactive\n'
                self.assertEqual(argv[0], 'nvidia-smi');return '9999\n'
            real_stat = os.stat
            def stat(path, *a, **k):
                if str(path) in ('/proc/1/ns/mnt', '/proc/1/ns/net'):
                    return SimpleNamespace(st_ino=1 if str(path).endswith('/mnt') else 11)
                return real_stat(path, *a, **k)
            health = dict(ready=True, pid=901, run_id=run, gpu_shards=4, logical_flat_shards=64,
                encoder_dtype='float16', index_dtype='float16', original_global_ids=True, endpoint=pool.URL)
            with patch('os.getpid', return_value=123), patch('socket.gethostname', return_value='h100-5' if role == 'upstream' else 'h100-10'), \
                 patch('os.stat', side_effect=stat), patch.object(RUNTIME.source, 'process', side_effect=process), \
                 patch.object(RUNTIME.subprocess, 'Popen', side_effect=spawn), patch.object(RUNTIME.subprocess, 'check_output', side_effect=command), \
                 patch.object(RUNTIME.time, 'time', side_effect=lambda: clock[0]), patch.object(RUNTIME.time, 'sleep', side_effect=sleep), \
                 patch.object(RUNTIME.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(json.dumps(health).encode())):
                yield SimpleNamespace(base=base, args=args, pool=pool, sleeps=sleeps, stopped=stopped, calls=calls,
                    descriptors=descriptors, model_busy=model_busy, observed_pids=observed_pids, source_path=source_path, clock=clock)

    def test_both_cpu_owners_wait_for_dependents_and_never_own_source_models(self):
        for role in ('upstream', 'downstream'):
            with self.subTest(role=role), self.fixture(role) as f, contextlib.redirect_stdout(io.StringIO()):
                publish(f.base / 'already-finished-model.json', dict(sampler_returncode=0))
                RUNTIME.controller(f.pool, f.args)
                self.assertEqual(len(f.calls), 1)
                self.assertNotIn('--sampler-loop', f.calls[0])
                self.assertGreaterEqual(len(f.sleeps), 3)
                self.assertEqual(f.stopped, [500])
                ready = read(f.args.ready_file);job = Path(ready['job_ready_path']).parent
                self.assertEqual(ready['source_owner']['pid'], 900)
                self.assertFalse(ready['model_processes_owned'])
                self.assertTrue(read(job / 'complete.json')['dependent_leases_released'])
                if role == 'downstream':
                    self.assertEqual(len(ready['gpu_lock_proofs']), 8)
                    self.assertEqual(set(ready['slots']), {'tp4-0-3', 'tp4-4-7'})
                    self.assertFalse({900, 901, 700} & set(f.observed_pids), 'Never inspect source5 /proc PIDs on node10')
                for fd in set(f.descriptors):
                    with self.assertRaises(OSError):os.fstat(fd)

    def test_changed_source_ready_is_rejected_before_start_or_locks(self):
        with self.fixture('downstream') as f:
            value = read(f.source_path);value['retriever']['pid'] = 1490465;publish(f.source_path, value)
            with self.assertRaisesRegex(ValueError, 'SHA differs'):RUNTIME.controller(f.pool, f.args)
            self.assertEqual(f.calls, [])
            self.assertEqual(f.descriptors, [])

    def test_new_bundles_have_distinct_intents_and_guarded_source_entry(self):
        units = []
        for role, slot in (('upstream', None), ('downstream', None), ('slot', 'tp4-0-3'), ('slot', 'tp4-4-7')):
            _, files, manifest = control.bundle(role, slot=slot)
            units.append(manifest['unit']);ast.parse(files['relay_generation_bootstrap.py'])
            self.assertNotIn(b'1490465', files['relay_generation_bootstrap.py'])
            self.assertNotIn(b'2371322', files['relay_generation_bootstrap.py'])
        self.assertEqual(len(units), len(set(units)))
        _, _, defaults = control.bundle('downstream')
        self.assertEqual(set(defaults['finite_node10_slots']), set(RUNTIME.DEFAULT_SLOTS))
        _, _, optional = control.bundle('downstream', enabled_slots=tuple(RUNTIME.SLOTS))
        self.assertEqual(set(optional['finite_node10_slots']) - set(RUNTIME.DEFAULT_SLOTS),
                         {'gpu1-tp1', 'gpu2-tp1', 'gpu3-tp1', 'gpu5-tp1'})
        ast.parse(control.REMOTE)
        _, files, manifest = guarded_control.bundle()
        self.assertIn(b"str(stage/'retrieval_remote_guard_owner.py')", files['generation_bootstrap.py'])
        self.assertEqual(manifest['remote_relay_guard'], RUNTIME.guard.CONTRACT)

    def test_model_slot_finishes_without_owning_or_releasing_the_cpu_relay(self):
        with self.fixture('downstream') as f:
            original_sleep = RUNTIME.time.sleep
            original_process = RUNTIME.source.process
            completed = []
            def exercise_model(seconds):
                if not completed:
                    ready = read(f.args.ready_file);slot = ready['slots']['tp4-4-7']
                    f.model_busy[0] = False
                    unit = 'future-independent-model.service';group = '/system.slice/' + unit
                    process = dict(pid=124, birth='new-model-birth', cgroup='0::' + group + '\n', netns_inode=11, mntns_inode=111)
                    evidence = f.base / 'model-evidence.json'
                    publish(evidence, dict(node='h100-10', main_pid=124, unit=unit, cgroup=group,
                        unit_show='Id=%s\nMainPID=124\nActiveState=active\nControlGroup=%s\n' % (unit, group),
                        captured_epoch=1000, unit_hard_end_epoch=29800,
                        resource_limits=dict(runtime_max_seconds=28800)))
                    upstream = read(ready['upstream_owner_ready'])
                    args = argparse.Namespace(base=str(f.base), reuse_local_relay_ready=f.args.ready_file,
                        reuse_local_relay_sha256=sha(f.args.ready_file), reuse_local_relay_unit=ready['owner']['unit'],
                        allocation_evidence=str(evidence), upstream_ready=upstream['transport_ready_path'],
                        relay_script=f.args.relay_script, pool_worker=str(DIRECTORY / 'draining_pool_worker.py'),
                        model_gpus='4,5,6,7', tp=4, model_max_seqs=slot['seqs'], model_batch_tokens=slot['tokens'],
                        model_kv_block_cap=slot['blocks'], model_reserve_gib=slot['reserve'],
                        run_seconds=28500, batch_reserve_seconds=1200, handoff_marker=None)
                    f.pool.handoff = lambda *a: {}
                    primary = read(ready['allocation_path'])['gpu_lock_proofs']
                    def held(path, owner, proof):
                        self.assertEqual(owner['pid'], 123)
                        actual = os.fstat(proof['fd'])
                        self.assertEqual((actual.st_dev, actual.st_ino), (path.stat().st_dev, path.stat().st_ino))
                        return dict(proof)
                    class Sampler:
                        pid = 600
                        returncode = 0
                        def poll(self):return self.returncode
                    def spawn(command, **kwargs):
                        self.assertIn('--sampler-loop', command)
                        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '4,5,6,7')
                        job = Path(command[command.index('--job-dir') + 1])
                        self.assertEqual(kwargs['env']['Q7B_POOL_INTERNAL'], str(job))
                        self.assertEqual(float(command[command.index('--deadline') + 1]), ready['consumer_deadline_epoch'])
                        self.assertTrue(Path(command[command.index('--launch-dir') + 1]).is_dir())
                        publish(job / 'sampler-exit.json', dict(completed_own_batches=1, questions=128, trajectories=640))
                        return Sampler()
                    with patch.dict(os.environ, Q7B_RELAY_SLOT_NAME='tp4-4-7'), patch('os.getpid', return_value=124), \
                         patch.object(RUNTIME.source, 'process', side_effect=lambda pid: dict(process) if pid == 124 else original_process(pid)), \
                         patch.object(SLOT.source_slot, 'held_lock', side_effect=held), \
                         patch.object(SLOT.subprocess, 'check_output', return_value=json.dumps(dict(actual_cuda_runtime_uuids=slot['cuda_uuids']))), \
                         patch.object(SLOT.subprocess, 'Popen', side_effect=spawn), \
                         patch.object(SLOT.consumer, 'admit', side_effect=SLOT.admit), \
                         patch.object(SLOT.consumer, 'borrowed_relay', side_effect=SLOT.borrowed_relay), \
                         contextlib.redirect_stdout(io.StringIO()):
                        SLOT.consumer.controller(f.pool, args)
                    for proof in primary.values():os.fstat(proof['fd'])
                    self.assertEqual(f.stopped, [600], 'A completed model must leave CPU relay 500 alive')
                    job = next((f.base / 'pool/jobs').glob('job-relay-slot-*/attempt-*'))
                    self.assertEqual(read(job / 'complete.json')['trajectories'], 640)
                    completed.append(job)
                original_sleep(seconds)
            with patch.object(RUNTIME.time, 'sleep', side_effect=exercise_model), contextlib.redirect_stdout(io.StringIO()):
                RUNTIME.controller(f.pool, f.args)
            self.assertEqual(len(completed), 1)
            self.assertEqual(f.stopped, [600, 500])

    def test_generated_bootstraps_execute_with_finite_runtime_and_borrow_flags(self):
        for role, slot in (('upstream', None), ('downstream', None), ('slot', 'tp4-0-3'), ('slot', 'tp4-4-7')):
            with self.subTest(role=role, slot=slot), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                with patch.object(control, 'BASE', str(base)):
                    identity, files, manifest = control.bundle(role, slot=slot)
                stage = base / 'preparation' / manifest['directory'] / identity;stage.mkdir(parents=True)
                for name, raw in files.items():(stage / name).write_bytes(raw)
                publish(stage / 'manifest.json', manifest)
                up_path = base / 'upstream-owner-ready.json';publish(up_path, dict(transport_ready_path='raw-upstream.json'))
                down_path = base / 'downstream-owner-ready.json'
                publish(down_path, dict(slots=RUNTIME.SLOTS, consumer_deadline_epoch=50000, owner=dict(unit='cpu-owner.service'),
                    upstream_owner_ready=str(up_path), job_ready_path=str(base / 'job/relay-owner-ready.json')))
                context = dict(source_owner_ready='future-source.json', source_owner_ready_sha256='a' * 64,
                    upstream_owner_ready=str(up_path), upstream_owner_ready_sha256=sha(up_path),
                    downstream_owner_ready=str(down_path), downstream_owner_ready_sha256=sha(down_path),
                    predecessor_proof='terminal-node10.json', predecessor_sha256='b' * 64, listen_host='10.0.0.5')
                publish(stage.parent / 'submit-context.json', context)
                unit = manifest['unit'];group = '/system.slice/' + unit
                memory = 103079215104 if role == 'slot' else 1073741824
                detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=%d\nActiveEnterTimestampMonotonic=1000000000\n' % (unit, group, memory)
                real_stat, real_text = os.stat, Path.read_text
                def stat(path, *a, **k):
                    if str(path).startswith('/proc/') and '/ns/' in str(path):
                        return SimpleNamespace(st_ino=2 if str(path) == '/proc/self/ns/mnt' else 1)
                    return real_stat(path, *a, **k)
                def text(path, *a, **k):
                    return '0::' + group + '\n' if str(path) == '/proc/self/cgroup' else real_text(path, *a, **k)
                namespace = dict(__file__=str(stage / 'relay_generation_bootstrap.py'))
                try:
                    with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                         patch('socket.gethostname', return_value='h100-5' if role == 'upstream' else 'h100-10'), \
                         patch('os.stat', side_effect=stat), patch.object(Path, 'read_text', text), \
                         patch('time.time', return_value=2000), patch('time.monotonic', return_value=1500), \
                         patch('subprocess.check_output', return_value=detail), patch('os.dup2'), patch('os.chdir'), \
                         patch('builtins.print'), patch('os.execvpe') as execute:
                        exec(compile(files['relay_generation_bootstrap.py'], namespace['__file__'], 'exec'), namespace)
                    _, argv, env = execute.call_args.args
                    flag = lambda name: argv[argv.index(name) + 1]
                    if role == 'slot':
                        self.assertEqual(flag('--run-seconds'), '28210')
                        self.assertEqual(flag('--reuse-local-relay-ready'), str(down_path))
                        self.assertEqual(flag('--upstream-ready'), 'raw-upstream.json')
                        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3' if slot == 'tp4-0-3' else '4,5,6,7')
                        self.assertNotIn('--token-file', argv)
                    else:
                        self.assertEqual(flag('--source-owner-ready'), 'future-source.json')
                        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                finally:
                    if 'log' in namespace:namespace['log'].close()

    def test_real_http_bytes_status_scores_and_auth_are_unchanged(self):
        seen = []
        response_bytes = b'{"result":[{"score":-0.12345678912345678,"id":7}],"raw":"\\u4e2d"}'
        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *a):pass
            def do_POST(self):
                seen.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(207);self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(response_bytes)));self.end_headers();self.wfile.write(response_bytes)
            def do_GET(self):
                self.send_response(200);self.send_header('Content-Length', '2');self.end_headers();self.wfile.write(b'{}')
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            token_file = Path(folder) / 'remote-secret';token_file.write_text(secrets.token_hex(32));token_file.chmod(0o600)
            token = RUNTIME.relay.read_token(token_file)
            backend = HTTPServer(('127.0.0.1', 0), Backend)
            upstream = RUNTIME.relay.RelayHTTPServer(('127.0.0.1', 0), backend.server_address, expected_token=token)
            downstream = RUNTIME.relay.RelayHTTPServer(('127.0.0.1', 0), upstream.server_address, outgoing_token=token)
            for server in (backend, upstream, downstream):
                thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
                stack.callback(server.server_close);stack.callback(server.shutdown)
            body = b'{"queries":["binary\\u0000\\u4e2d"],"topk":3}'
            connection = http.client.HTTPConnection(*downstream.server_address, timeout=3)
            connection.request('POST', '/retrieve', body=body, headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual((response.status, response.getheader('Content-Type'), response.read()),
                             (207, 'application/json; charset=utf-8', response_bytes));connection.close()
            self.assertEqual(seen, [('/retrieve', body)])
            denied = http.client.HTTPConnection(*upstream.server_address, timeout=3)
            denied.request('GET', '/health', headers={RUNTIME.relay.AUTH_HEADER: 'incorrect-token'})
            response = denied.getresponse();self.assertEqual(response.status, 403);response.read();denied.close()
            connection = http.client.HTTPConnection(*downstream.server_address, timeout=3)
            connection.request('GET', '/health');response = connection.getresponse()
            self.assertEqual((response.status, response.read()), (200, b'{}'));connection.close()


if __name__ == '__main__':unittest.main()
