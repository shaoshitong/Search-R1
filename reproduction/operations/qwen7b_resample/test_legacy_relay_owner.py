"""Pure CPU lifecycle and real file/SHA fixtures for the old-source CPU owner."""
import argparse
import ast
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import legacy_relay_owner as owner
from . import legacy_relay_control as control

DIRECTORY = Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LegacyOwnerTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, fail_health=False):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);run = control.RUN_ID;unit = 'cpu-independent.service';group = '/system.slice/' + unit
            process = dict(pid=123, birth='own-birth', cgroup='0::' + group + '\n', netns_inode=11, mntns_inode=111)
            child_proof = dict(process, pid=500, birth='relay-birth')
            context = base / 'retained/source-context.json'
            publish(context, dict(source_main_pid=1490214, source_main_birth='old-root-birth', source_pid=1490465,
                source_birth='old-retriever-birth', source_netns_inode=22, source_unit='sst-q7b-g5-sharedvram-v6-' + run[:12] + '.service',
                source_sampling_deadline_epoch=20000))
            upstream = context.parent / 'upstream-ready.json'
            publish(upstream, dict(status='RELAY_LISTENING', role='upstream', source_pid=1490465,
                source_birth='old-retriever-birth', source_netns_inode=22,
                code_sha256=sha(DIRECTORY / 'retrieval_relay.py'), process=dict(pid=1654561, birth='old-upstream-birth')))
            token = context.parent / 'relay-secret';token.write_text(secrets.token_hex(32));token.chmod(0o600)
            proof = base / 'failed-node10-proof.json'
            units = [dict(unit='sst-q7b-g5-' + name + '-' + run[:12] + '.service') for name in
                ('consumer10-v5', 'consumer10-parallel-v1', 'consumer10-tp1-gpu1-v1', 'consumer10-tp1-gpu2-v1')]
            observation = dict(original_births_gone=True, own_cgroups_empty=True, remaining_known_pids=[],
                own_gpu_contexts=[], loopback_28831_listeners=[])
            publish(proof, dict(status='OWN_NODE10_NATIVE_FAILURES_ORIGINAL_SHA_RELEASE_PASS',
                predecessor_compatibility_status='OWN_PREDECESSOR_GENERATION_TERMINAL', node='h100-10', run_id=run,
                units=units, original_owned_births={'10': 'old-node10-birth'}, pids_without_recorded_historical_birth=[11],
                original_partial_outputs_retained=True, release_observations=[dict(observation, epoch=1), dict(observation, epoch=4)]))
            evidence = base / 'evidence.json'
            publish(evidence, dict(node='h100-10', unit=unit, main_pid=123, cgroup=group, unit_hard_end_epoch=10000,
                unit_show='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (unit, group)))
            slots = base / 'slots.json';publish(slots, owner.TOPOLOGIES['tp4'])
            args = argparse.Namespace(base=str(base), source_context=str(context), source_context_sha256=sha(context),
                upstream_ready=str(upstream), upstream_ready_sha256=sha(upstream), token_file=str(token),
                allocation_evidence=str(evidence), predecessor_proof=str(proof), predecessor_sha256=sha(proof),
                slot_manifest=str(slots), relay_script=str(DIRECTORY / 'retrieval_relay_backpressure.py'),
                ready_file=str(base / 'control/owner-ready.json'), owner_seconds=4000)
            clock, calls, stopped, health_calls = [1000], [], [], []
            busy = [True]
            pool = SimpleNamespace(read=read, sha=sha, publish=publish, RUN_ID=run,
                configuration=lambda _: {}, URL='http://127.0.0.1:28831/retrieve')
            def lock(path):
                if busy[0] and path.name == 'gpu1-tp1.lock':return None
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600);fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB);return fd
            pool.try_lock = lock
            class Relay:
                pid = 500
                returncode = None
                def poll(self):return self.returncode
            child = Relay()
            def spawn(command, **kwargs):
                calls.append(command);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                raw = dict(status='RELAY_LISTENING', role='downstream', listen_url='http://127.0.0.1:28831',
                    source_pid=1490465, source_birth='old-retriever-birth', source_netns_inode=22,
                    upstream_proof=dict(sha256=sha(upstream)), code_sha256=sha(args.relay_script),
                    transport_core_sha256=sha(DIRECTORY / 'retrieval_relay.py'), process=child_proof,
                    bounds=dict(max_workers=8, timeout=50), backpressure=dict(max_queued=128, queue_wait=45, request_budget=59,
                        fixed_worker_pool=True, no_empty_retrieval_fallback=True))
                publish(command[command.index('--ready-file') + 1], raw);return child
            def stop(p):stopped.append(p.pid);p.returncode = 0
            pool.stop_group = stop
            def sleep(seconds):
                if clock[0] >= 5000:busy[0] = False;clock[0] += 1
                else:clock[0] += 2000
            def query(command, **kwargs):
                if command[0] == 'systemctl':return 'Id=%s\nMainPID=0\nActiveState=failed\n' % command[2]
                self.assertEqual(command[0], 'nvidia-smi');return '9999\n'
            def live(pid):
                self.assertNotIn(pid, (1490214, 1490465, 1654561), 'Do not inspect source5 PIDs in node10 /proc')
                if pid == 10:raise FileNotFoundError('old original is absent')
                return dict(process if pid == 123 else child_proof)
            def health(*a, **k):
                health_calls.append(clock[0])
                if fail_health and len(health_calls) >= 3:raise OSError('actual upstream disconnected')
                return io.BytesIO(json.dumps(dict(ready=True, pid=1490465, run_id=run, gpu_shards=4, logical_flat_shards=64,
                    encoder_dtype='float16', index_dtype='float16', original_global_ids=True, endpoint=pool.URL)).encode())
            real_stat = os.stat
            def stat(path, *a, **k):
                if str(path) in ('/proc/1/ns/net', '/proc/1/ns/mnt'):return SimpleNamespace(st_ino=11 if str(path).endswith('/net') else 1)
                return real_stat(path, *a, **k)
            class Probe:
                def __enter__(self):return self
                def __exit__(self, *a):pass
                def bind(self, address):self_address = address;assert self_address == ('127.0.0.1', 28831)
            with patch('os.getpid', return_value=123), patch('socket.gethostname', return_value='h100-10'), \
                 patch('os.stat', side_effect=stat), patch.object(owner, 'process', side_effect=live), \
                 patch.object(owner.subprocess, 'check_output', side_effect=query), patch.object(owner.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(owner.socket, 'socket', return_value=Probe()), patch.object(owner.urllib.request, 'urlopen', side_effect=health), \
                 patch.object(owner.time, 'time', side_effect=lambda: clock[0]), patch.object(owner.time, 'sleep', side_effect=sleep):
                yield SimpleNamespace(base=base, pool=pool, args=args, calls=calls, stopped=stopped, busy=busy)

    def test_independent_cpu_lifetime_waits_model_live_lease(self):
        with self.fixture() as f, contextlib.redirect_stdout(io.StringIO()):
            publish(f.base / 'already-completed-model.json', dict(trajectories=640))
            owner.controller(f.pool, f.args)
            ready = read(f.args.ready_file);job = Path(ready['job_ready_path']).parent
            self.assertEqual((ready['deadline_epoch'], ready['consumer_deadline_epoch']), (5000, 4400))
            self.assertEqual(ready['status'], 'LEGACY_SOURCE_RELAY_OWNER_READY')
            self.assertEqual(ready['token_file_sha256'], sha(f.args.token_file))
            self.assertEqual(ready['transport_ready_sha256'], sha(ready['transport_ready_path']))
            self.assertEqual(len(ready['slots']), 5)
            self.assertEqual(ready['slots']['tp4-0-4-6-7']['blocks'], 512)
            self.assertFalse(ready['gpu_primary_leases_owned'])
            self.assertFalse((f.base / 'pool/gpu-locks').exists())
            self.assertTrue((job / 'draining.json').exists())
            self.assertEqual(read(ready['owner_drain_path'])['owner_allocation_sha256'], ready['allocation_sha256'])
            self.assertTrue(read(job / 'complete.json')['model_live_leases_released'])
            self.assertEqual(f.stopped, [500]);self.assertEqual(len(f.calls), 1)

    def test_actual_source_disconnect_is_terminal_without_empty_success(self):
        with self.fixture(fail_health=True) as f, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, 'upstream disconnected'):owner.controller(f.pool, f.args)
            ready = read(f.args.ready_file);job = Path(ready['job_ready_path']).parent
            failure = read(job / 'failure.json')
            self.assertFalse((job / 'complete.json').exists())
            self.assertFalse(failure['models_signalled']);self.assertFalse(failure['source_gpu_processes_signalled'])
            self.assertTrue(failure['partial_batches_may_remain']);self.assertEqual(f.stopped, [500])

    def test_unproven_second_release_is_rejected_before_listener(self):
        with self.fixture() as f:
            proof = read(f.args.predecessor_proof);proof['release_observations'][1]['remaining_known_pids'] = [10]
            publish(f.args.predecessor_proof, proof);f.args.predecessor_sha256 = sha(f.args.predecessor_proof)
            with self.assertRaisesRegex(ValueError, 'observed twice'):owner.controller(f.pool, f.args)
            self.assertEqual(f.calls, [])

    def test_bootstrap_uses_remote_retained_paths_and_no_gpu_command(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            with patch.object(control, 'BASE', str(base)):
                identity, files, manifest = control.bundle('tp4')
            stage = base / 'preparation' / manifest['directory'] / identity;stage.mkdir(parents=True)
            for name, raw in files.items():(stage / name).write_bytes(raw)
            publish(stage / 'manifest.json', manifest)
            context = dict(source_context='old-source-context.json', source_context_sha256='a' * 64,
                upstream_ready='old-upstream-ready.json', upstream_ready_sha256='b' * 64, token_file='old-remote-token',
                predecessor_proof='actual-node10-failure-proof.json', predecessor_sha256='c' * 64)
            publish(stage.parent / 'submit-context.json', context)
            unit = manifest['unit'];group = '/system.slice/' + unit
            detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=1073741824\nActiveEnterTimestampMonotonic=1000000000\n' % (unit, group)
            real_stat, real_text = os.stat, Path.read_text
            def stat(path, *a, **k):
                if str(path).startswith('/proc/') and '/ns/' in str(path):return SimpleNamespace(st_ino=2 if str(path) == '/proc/self/ns/mnt' else 1)
                return real_stat(path, *a, **k)
            def text(path, *a, **k):return '0::' + group + '\n' if str(path) == '/proc/self/cgroup' else real_text(path, *a, **k)
            namespace = dict(__file__=str(stage / 'legacy_cpu_bootstrap.py'))
            try:
                with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                     patch('socket.gethostname', return_value='h100-10'), patch('os.stat', side_effect=stat), \
                     patch.object(Path, 'read_text', text), patch('subprocess.check_output', return_value=detail), \
                     patch('time.time', return_value=2000), patch('time.monotonic', return_value=1500), \
                     patch('os.dup2'), patch('builtins.print'), patch('os.execvpe') as execute:
                    exec(compile(files['legacy_cpu_bootstrap.py'], namespace['__file__'], 'exec'), namespace)
                _, argv, env = execute.call_args.args
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(argv[argv.index('--token-file') + 1], 'old-remote-token')
                self.assertIn('legacy_relay_owner.py', ' '.join(argv));self.assertNotIn('sampler', ' '.join(argv))
                evidence = read(argv[argv.index('--allocation-evidence') + 1])
                self.assertEqual(evidence['unit_hard_end_epoch'], 15900)
                ast.parse(control.REMOTE)
            finally:
                if 'log' in namespace:namespace['log'].close()


if __name__ == '__main__':unittest.main()
