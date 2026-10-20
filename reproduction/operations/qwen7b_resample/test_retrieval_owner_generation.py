"""CPU owner lifecycle and finite-slot admission; no GPU or remote calls."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import retrieval_only_control as control

DIRECTORY = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):
        spec.loader.exec_module(module)
    return module


OWNER = load('cpu_retrieval_owner', 'retrieval_only_owner.py')
SLOT = load('cpu_source_slot', 'source_slot_consumer_worker.py')


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


class RetrievalOwnerTests(unittest.TestCase):
    def test_owner_lifetime_does_not_end_with_model_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);run = control.RUN_ID
            pool = SimpleNamespace(read=read, publish=publish, sha=sha, try_lock=lock,
                RUN_ID=run, URL='http://127.0.0.1:28831/retrieve', handoff=lambda *a: {},
                configuration=lambda _: dict(base=str(base), retrieval_index='index', retrieval_corpus='corpus', retrieval_data_cache='cache'))
            uuids = ['GPU-%d' % i for i in (3, 4, 6, 7, 0, 1, 2, 5)]
            fds = [lock(base / 'pool/gpu-locks' / (uuid + '.lock')) for uuid in uuids]
            binding = dict(model_indices=[3, 4, 6, 7], retrieval_indices=[0, 1, 2, 5],
                retrieval_cuda_uuids=uuids[4:], retrieval_cuda_visible_devices='0,1,2,5', actual_cuda_runtime_uuids=uuids)
            slots = base / 'slots.json';publish(slots, {name: OWNER.SLOTS[name] for name in OWNER.DEFAULT_SLOTS})
            evidence = base / 'evidence.json';publish(evidence, dict(unit='owner-cpu.service', captured_epoch=1000,
                resource_limits=dict(runtime_max_seconds=90000)))
            args = argparse.Namespace(base=str(base), handoff_marker='fixture', slot_manifest=str(slots),
                allocation_evidence=str(evidence), owner_seconds=30400, encoder='encoder',
                retriever_script='original-retriever.py', retrieval_weights='15,9,20,20')
            clock, sleeps, stopped, commands = [1000], [], [], []
            publish(base / 'old-model-complete.json', dict(sampler_returncode=0))

            class Retriever:
                pid = 500
                returncode = None
                def poll(self):return self.returncode

            retriever = Retriever()
            def spawn(command, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1,2,5')
                commands.append(command);return retriever
            def sleep(seconds):
                sleeps.append(clock[0]);clock[0] += 10000
            def stop(p):
                stopped.append(p.pid);p.returncode = 0
            pool.stop_group = stop
            health = dict(ready=True, pid=500, run_id=run, actual_cuda_uuids=uuids[4:], gpu_shards=4,
                logical_flat_shards=64, encoder_dtype='float16', index_dtype='float16',
                original_global_ids=True, endpoint=pool.URL)
            proof = lambda pid: dict(pid=pid, birth=str(pid), cgroup='0::/system.slice/owner-cpu.service\n', netns_inode=11, mntns_inode=12)
            with patch.object(OWNER.shared, 'admit', return_value=(binding, fds)), \
                 patch.object(OWNER, 'process', side_effect=proof), patch.object(OWNER.subprocess, 'Popen', side_effect=spawn), \
                 patch.object(OWNER.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(json.dumps(health).encode())), \
                 patch.object(OWNER.time, 'time', side_effect=lambda: clock[0]), \
                 patch.object(OWNER.time, 'sleep', side_effect=sleep), contextlib.redirect_stdout(io.StringIO()):
                OWNER.controller(pool, args)
            self.assertGreaterEqual(len(sleeps), 3, 'An already completed model must not retire the index')
            self.assertEqual(len(commands), 1)
            self.assertIn('original-retriever.py', commands[0])
            self.assertNotIn('--sampler-loop', commands[0])
            self.assertEqual(stopped, [500])
            job = next((base / 'pool/jobs').glob('job-retrieval-owner-*/attempt-*'))
            ready = read(job / 'owner-ready.json')
            self.assertFalse(ready['model_processes_owned'])
            self.assertEqual(read(job / 'allocation.json')['model_engine_count'], 0)
            self.assertEqual(read(job / 'allocation.json')['model_indices'], [])
            self.assertEqual(ready['consumer_deadline_epoch'], ready['deadline_epoch'] - 1800)
            self.assertTrue(read(job / 'complete.json')['model_leases_released'])
            self.assertFalse(read(job / 'complete.json')['global_goal_complete'])
            for fd in fds:
                with self.assertRaises(OSError):os.fstat(fd)

    def test_bundles_are_independent_and_optional_slots_require_declaration(self):
        owner_id, files, manifest = control.bundle()
        slot_id, _, consumer = control.bundle('slot', slot='gpu2-tp1')
        import ast
        ast.parse(control.REMOTE)
        ast.parse(files['generation_bootstrap.py'])
        self.assertNotEqual(owner_id, slot_id)
        self.assertNotEqual(manifest['unit'], consumer['unit'])
        self.assertNotIn('gpu4-tp1', manifest['slots'])
        self.assertEqual(manifest['owner_hard_seconds'], 90000)
        self.assertNotIn('sbatch', control.REMOTE)

    def test_generated_bootstraps_bind_units_deadlines_and_the_drain_wrapper(self):
        for role, slot_name in (('owner', None), ('slot', 'gpu2-tp1')):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                with patch.object(control, 'BASE', str(base)):
                    identity, files, manifest = control.bundle(role, slot=slot_name)
                stage = base / 'preparation' / manifest['directory'] / identity
                stage.mkdir(parents=True)
                for name, value in files.items():(stage / name).write_bytes(value)
                publish(stage / 'manifest.json', manifest)
                ready_path = base / 'owner/owner-ready.json'
                publish(ready_path, dict(slots={'gpu2-tp1': OWNER.SLOTS['gpu2-tp1']}, consumer_deadline_epoch=50000,
                    owner=dict(unit='independent-owner.service')))
                publish(stage.parent / 'submit-context.json', dict(owner_ready=str(ready_path), owner_ready_sha256=sha(ready_path)))
                unit = manifest['unit'];group = '/system.slice/' + unit
                memory = 171798691840 if role == 'owner' else 103079215104
                detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=%d\nActiveEnterTimestampMonotonic=1000000000\n' % (unit, group, memory)
                original_text = Path.read_text
                def text(path, *a, **k):
                    return '0::' + group + '\n' if str(path) == '/proc/self/cgroup' else original_text(path, *a, **k)
                namespace = {'__file__': str(stage / 'generation_bootstrap.py')}
                try:
                    with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                         patch('socket.gethostname', return_value='h100-5'), patch.object(Path, 'read_text', text), \
                         patch('time.time', return_value=2000), patch('time.monotonic', return_value=1500), \
                         patch('subprocess.check_output', return_value=detail), patch('os.chdir'), patch('os.dup2'), \
                         patch('builtins.print'), patch('os.execvpe') as execute:
                        exec(compile(files['generation_bootstrap.py'], namespace['__file__'], 'exec'), namespace)
                    _, argv, env = execute.call_args.args
                    if role == 'owner':
                        self.assertIn('retrieval_only_owner.py', ' '.join(argv))
                        self.assertIn('--net', argv)
                    else:
                        flag = lambda key: argv[argv.index(key) + 1]
                        self.assertEqual(flag('--tp'), '1')
                        self.assertEqual(flag('--run-seconds'), '28210')
                        self.assertTrue(flag('--pool-worker').endswith('/draining_pool_worker.py'))
                        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
                        self.assertEqual(env['Q7B_SLOT_OWNER_DRAIN'], str(ready_path.parent / 'drain-request.json'))
                finally:
                    if 'log' in namespace:namespace['log'].close()


class SourceSlotTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);run = control.RUN_ID;owner_unit = 'cpu-independent-owner.service'
            owner = dict(pid=900, birth='111', unit=owner_unit, cgroup='0::/system.slice/' + owner_unit + '\n', netns_inode=11, mntns_inode=12)
            retriever = dict(pid=901, birth='112', cgroup=owner['cgroup'], netns_inode=11, mntns_inode=12)
            own_unit = 'cpu-model-slot.service';own_group = '/system.slice/' + own_unit
            own = dict(pid=123, birth='113', cgroup='0::' + own_group + '\n', netns_inode=11, mntns_inode=12)
            master = base / 'pool/gpu-locks/GPU-2.lock';master.parent.mkdir(parents=True);master.write_bytes(b'original untouched primary')
            info = master.stat();slot = dict(OWNER.SLOTS['gpu2-tp1'], cuda_uuids=['GPU-2'])
            allocation = base / 'owner/allocation.json'
            publish(allocation, dict(allocation_role='RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER', run_id=run, owner=owner,
                slots={'gpu2-tp1': slot}, gpu_lock_proofs={'GPU-2': dict(fd=94, device=info.st_dev, inode=info.st_ino)}))
            path = base / 'owner/owner-ready.json'
            value = dict(status='RETRIEVAL_ONLY_OWNER_READY', run_id=run, owner=owner, retriever=retriever,
                allocation_path=str(allocation), allocation_sha256=sha(allocation), slots={'gpu2-tp1': slot}, consumer_deadline_epoch=50000)
            publish(path, value)
            upstream = path.parent / 'retriever-source-ready.json'
            publish(upstream, dict(role='upstream', source_pid=901, listen_url='http://127.0.0.1:28831'))
            evidence = base / 'evidence.json'
            publish(evidence, dict(node='h100-5', main_pid=123, unit=own_unit, cgroup=own_group,
                unit_show='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (own_unit, own_group),
                resource_limits=dict(runtime_max_seconds=28800)))
            args = argparse.Namespace(base=str(base), reuse_local_relay_ready=str(path), reuse_local_relay_sha256=sha(path),
                reuse_local_relay_unit=owner_unit, model_gpus='2', tp=1, run_seconds=28500,
                allocation_evidence=str(evidence), upstream_ready=str(upstream),
                model_max_seqs=slot['seqs'], model_batch_tokens=slot['tokens'], model_kv_block_cap=slot['blocks'], model_reserve_gib=slot['reserve'])
            pool = SimpleNamespace(read=read, sha=sha, RUN_ID=run, try_lock=lock,
                nvml=lambda: {'GPU-2': dict(index=2, uuid='GPU-2', free_bytes=40 * 1024 ** 3)})
            states = {900: owner, 901: retriever, 123: own}
            real_stat, real_text = os.stat, Path.read_text
            def stat(path, *a, **k):
                return info if str(path) == '/proc/900/fd/94' else real_stat(path, *a, **k)
            def text(path, *a, **k):
                if str(path) == '/proc/900/fdinfo/94':
                    return 'lock:\t1: POSIX ADVISORY WRITE 900 %x:%x:%d 0 EOF\n' % (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
                return real_text(path, *a, **k)
            with patch.dict(os.environ, Q7B_SLOT_OWNER_READY=str(path), Q7B_SLOT_OWNER_READY_SHA256=sha(path), Q7B_SLOT_NAME='gpu2-tp1'), \
                 patch.object(SLOT.owner_module, 'process', side_effect=lambda pid: dict(states[pid])), \
                 patch('os.getpid', return_value=123), patch('socket.gethostname', return_value='h100-5'), \
                 patch('os.stat', side_effect=stat), patch.object(Path, 'read_text', text), patch.object(SLOT.time, 'time', return_value=1000), \
                 patch.object(SLOT.subprocess, 'check_output', return_value=json.dumps(dict(actual_cuda_runtime_uuids=['GPU-2']))) as probe:
                yield SimpleNamespace(base=base, args=args, pool=pool, states=states, master=master, probe=probe)

    def test_declared_slot_holds_only_its_own_lease(self):
        with self.fixture() as f:
            binding, locks = SLOT.admit(f.pool, f.args)
            try:
                self.assertEqual(binding['finite_slot'], 'gpu2-tp1')
                self.assertIn('/900-111/gpu2-tp1/', binding['slot_lease_paths']['GPU-2'])
                self.assertEqual(f.master.read_bytes(), b'original untouched primary')
                self.assertFalse(binding['shared_retriever_owned_by_consumer'])
                borrowed = SLOT.borrowed_relay(f.pool, f.args)
                f.states[900]['birth'] = 'different'
                self.assertEqual(borrowed.poll(), 1)
            finally:
                for fd in locks:os.close(fd)

    def test_undeclared_slot_fails_before_cuda_probe(self):
        with self.fixture() as f, patch.dict(os.environ, Q7B_SLOT_NAME='gpu4-tp1'):
            with self.assertRaisesRegex(ValueError, 'Undeclared'):
                SLOT.admit(f.pool, f.args)
            f.probe.assert_not_called()

    def test_slow_admission_cannot_extend_owner_or_evidenced_unit_deadline(self):
        for hard_end, expected in ((None, 29520), (29560, 29515)):
            with self.subTest(unit_hard_end=hard_end), self.fixture() as f:
                ready_path = Path(f.args.reuse_local_relay_ready)
                ready = read(ready_path)
                ready['consumer_deadline_epoch'] = 29520
                publish(ready_path, ready)
                f.args.reuse_local_relay_sha256 = sha(ready_path)
                evidence = read(f.args.allocation_evidence)
                evidence.update(captured_epoch=1000, resource_limits=dict(runtime_max_seconds=90000))
                if hard_end is not None:
                    evidence['unit_hard_end_epoch'] = hard_end
                publish(f.args.allocation_evidence, evidence)
                f.args.handoff_marker = None
                f.args.pool_worker = str(DIRECTORY / 'draining_pool_worker.py')
                f.args.relay_script = str(DIRECTORY / 'retrieval_relay.py')
                f.args.batch_reserve_seconds = 300
                f.pool.publish = publish
                f.pool.configuration = lambda _: {}
                f.pool.handoff = lambda *a: {}
                f.pool.URL = 'http://127.0.0.1:28831/retrieve'
                clock, deadlines, stopped = [1000], [], []

                def slow_probe(*a, **k):
                    clock[0] = 1040
                    return json.dumps(dict(actual_cuda_runtime_uuids=['GPU-2']))

                class Sampler:
                    pid = 700
                    returncode = 0
                    def poll(self):return self.returncode

                def spawn(command, **kwargs):
                    self.assertIn('--sampler-loop', command)
                    deadlines.append(float(command[command.index('--deadline') + 1]))
                    job = Path(command[command.index('--job-dir') + 1])
                    launch = Path(command[command.index('--launch-dir') + 1])
                    self.assertTrue(launch.is_dir())
                    publish(job / 'sampler-exit.json', dict(completed_own_batches=1,
                        completed_own_questions=128, completed_own_trajectories=640))
                    return Sampler()

                f.pool.stop_group = lambda p: stopped.append(p.pid)
                health = dict(ready=True, pid=901, run_id=f.pool.RUN_ID, gpu_shards=4,
                    logical_flat_shards=64, encoder_dtype='float16', index_dtype='float16',
                    original_global_ids=True, endpoint=f.pool.URL)
                with patch.dict(os.environ, Q7B_SLOT_OWNER_READY_SHA256=f.args.reuse_local_relay_sha256), \
                     patch.object(SLOT.consumer, 'admit', side_effect=SLOT.admit), \
                     patch.object(SLOT.consumer, 'borrowed_relay', side_effect=SLOT.borrowed_relay), \
                     patch.object(SLOT.time, 'time', side_effect=lambda: clock[0]), \
                     patch.object(SLOT.subprocess, 'check_output', side_effect=slow_probe), \
                     patch.object(SLOT.subprocess, 'Popen', side_effect=spawn), \
                     patch.object(SLOT.consumer.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(json.dumps(health).encode())), \
                     contextlib.redirect_stdout(io.StringIO()):
                    SLOT.consumer.controller(f.pool, f.args)
                job = next((f.base / 'pool/jobs').glob('job-source-slot-*/attempt-*'))
                self.assertEqual(deadlines, [expected])
                self.assertEqual(read(job / 'allocation.json')['source_model_deadline_epoch'], expected)
                self.assertEqual(read(job / 'complete.json')['completed_own_trajectories'], 640)
                self.assertEqual(stopped, [700], 'The borrowed owner and retriever must remain untouched')


if __name__ == '__main__':
    unittest.main()
