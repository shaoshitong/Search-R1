"""CPU-only real main/REMOTE/bootstrap/controller pipeline and ownership checks."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import relay_primary_control as control

DIRECTORY = Path(control.__file__).parent
with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):
    import relay_primary_worker as worker


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PrimaryTests(unittest.TestCase):
    def test_finite_topologies_cover_exactly_eight_primary_uuids_and_keep_full_recipe(self):
        self.assertEqual(control.TOPOLOGIES, worker.TOPOLOGIES)
        for topology, slots in control.TOPOLOGIES.items():
            self.assertEqual(sorted(i for config in slots.values() for i in config['indices']), list(range(8)))
            for slot, config in slots.items():
                identity, files, manifest = control.bundle(slot, topology)
                self.assertEqual(manifest['max_model_len'], 4096)
                self.assertEqual(manifest['model_configuration'], config)
                self.assertEqual(manifest['batch_reserve_seconds'], 1200)
                self.assertEqual(manifest['maximum_run_seconds'], 14400)
                self.assertTrue(manifest['primary_uuid_locks_owned'])
                self.assertFalse(manifest['local_relay_owned'])
                ast.parse(files['relay_primary_bootstrap.py'])
                for name in ('pool_worker.py', 'shared_vram_worker.py', 'shared_vram_vllm_adapter.py', 'draining_pool_worker.py'):
                    self.assertEqual(files[name], (DIRECTORY / name).read_bytes())
        with self.assertRaises(ValueError): control.bundle('tp2-0-4', 'tp4')
        self.assertEqual(control.TOPOLOGIES['tp4']['tp4-0-4-6-7']['blocks'], 512)

    def test_path_aliases_stay_inside_the_same_data_root(self):
        base = Path('/mnt/data/shaoshitong/test-run')
        self.assertEqual(worker.source_path(base, '/mnt/shared/workspace/training_data_node_10/shaoshitong/test-run/a.json'), base / 'a.json')
        with self.assertRaises(ValueError): worker.source_path(base, '/mnt/data/shaoshitong/test-run/../foreign/a')

    def test_v1_is_read_only_and_v2_has_separate_native_identity(self):
        for slot in control.TOPOLOGIES['tp4']:
            self.assertEqual(control.names(slot, 'v2'), worker.names(slot, 'v2'))
            self.assertNotEqual(control.names(slot, 'v1'), control.names(slot, 'v2'))
            old, _, old_manifest = control.bundle(slot, instance='v1')
            new, _, new_manifest = control.bundle(slot, instance='v2')
            self.assertNotEqual(old, new)
            self.assertNotEqual(old_manifest['directory'], new_manifest['directory'])
        with patch.object(sys, 'argv', ['primary', 'submit', '--slot', 'gpu1-tp1', '--instance', 'v1']), \
                patch.object(control.old, 'transport') as transport:
            with self.assertRaisesRegex(ValueError, 'read-only status'): control.main()
            transport.assert_not_called()

    def run_pipeline(self, fail=False):
        """No preamble globals injected: actual CLI builds and executes REMOTE."""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / 'nas'; local = Path(folder) / 'local'; (local / 'work').mkdir(parents=True)
            now = 10000.; slot = 'gpu1-tp1'; unit, directory = control.names(slot)
            cpu_unit = 'sst-q7b-g5-cpu-relay10-v1-' + control.RUN_ID[:12] + '.service'
            own_group, cpu_group = '/system.slice/' + unit, '0::/system.slice/' + cpu_unit + '\n'
            cpu_owner = dict(pid=700, birth='800', unit=cpu_unit, cgroup=cpu_group, netns_inode=1, mntns_inode=2)
            relay = dict(pid=701, birth='801', cgroup=cpu_group, netns_inode=1, mntns_inode=2)
            own = dict(pid=314, birth='500', cgroup='0::' + own_group + '\n', netns_inode=1, mntns_inode=3)
            source = dict(source_main_pid=1490214, source_main_birth='6807965', source_pid=1490465,
                source_birth='6808057', source_netns_inode=1, source_unit='sst-q7b-g5-sharedvram-v6-' + control.RUN_ID[:12] + '.service',
                source_sampling_deadline_epoch=30000)
            cpu_job = base / 'pool/jobs/cpu-owner/attempt-1'; cpu_ready_path = base / 'preparation/cpu-ready.json'
            source_path, upstream_path = base / 'proof/source.json', base / 'proof/upstream.json'
            allocation, transport_path = cpu_job / 'allocation.json', cpu_job / 'transport.json'
            write(source_path, source)
            write(upstream_path, dict(source, status='RELAY_LISTENING', role='upstream', process=dict(pid=1654561),
                code_sha256=sha(DIRECTORY / 'retrieval_relay.py')))
            write(transport_path, dict(source, status='RELAY_LISTENING', role='downstream', listen_url='http://127.0.0.1:28831',
                process=relay, upstream_proof=dict(sha256=sha(upstream_path)),
                code_sha256=sha(DIRECTORY / 'retrieval_relay_backpressure.py'),
                transport_core_sha256=sha(DIRECTORY / 'retrieval_relay.py')))
            write(allocation, dict(actual_cpu_owner=cpu_owner))
            ready = dict(status='LEGACY_SOURCE_RELAY_OWNER_READY', role='current-oldV6-source', run_id=control.RUN_ID,
                node='h100-10', owner=cpu_owner, relay=relay, slots=control.TOPOLOGIES['tp4'],
                gpu_primary_leases_owned=False, model_processes_owned=False,
                allocation_path=str(allocation), allocation_sha256=sha(allocation),
                source_context_path=str(source_path), source_context_sha256=sha(source_path),
                upstream_ready_path=str(upstream_path), upstream_ready_sha256=sha(upstream_path),
                transport_ready_path=str(transport_path), transport_ready_sha256=sha(transport_path),
                job_ready_path=str(cpu_job / 'owner-ready.json'), deadline_epoch=26000, consumer_deadline_epoch=25400)
            write(cpu_ready_path, ready); write(cpu_job / 'owner-ready.json', ready)
            write(base / 'preparation/pool-handoff.json', dict(run_id=control.RUN_ID))
            contexts, stops, starts, models, locks, assemblies = [], [], [], [], [], []
            inventory = {'GPU-%d' % i: dict(index=i, uuid='GPU-%d' % i, free_bytes=23 * 1024**3) for i in range(8)}
            def lock(path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600); locks.append((str(path), fd)); return fd
            pool = SimpleNamespace(read=read, sha=sha, publish=write, nvml=lambda: inventory,
                try_lock=lock, configuration=lambda _: {}, handoff=lambda _, marker: read(marker),
                RUN_ID=control.RUN_ID, URL='http://127.0.0.1:28831/retrieve', stop_group=lambda p: stops.append(p.pid))
            real_stat, real_read = os.stat, Path.read_text
            def stat(path, *args, **kwargs):
                if str(path) in ('/proc/self/ns/mnt', '/proc/1/ns/mnt', '/proc/self/ns/net', '/proc/1/ns/net'):
                    return SimpleNamespace(st_ino=3 if str(path) == '/proc/self/ns/mnt' else 1)
                return real_stat(path, *args, **kwargs)
            def read_text(path, *args, **kwargs):
                if str(path) == '/proc/self/cgroup': return own['cgroup']
                if str(path) == '/proc/self/stat': return '314 (python) S ' + '0 ' * 18 + '500 0'
                if str(path) == '/proc/sys/kernel/random/boot_id': return 'test-real-boot-id\n'
                return real_read(path, *args, **kwargs)
            def process(pid): return {700: cpu_owner, 701: relay, 314: own}[pid]
            def check_output(argv, **kwargs):
                if argv[0] == 'systemctl':
                    selected = argv[2]
                    if selected == cpu_unit:
                        return 'Id=%s\nMainPID=700\nActiveState=active\nControlGroup=/system.slice/%s\n' % (cpu_unit, cpu_unit)
                    return 'Id=%s\nMainPID=314\nActiveState=active\nControlGroup=%s\n' % (unit, own_group)
                self.assertIn('--cuda-probe', argv)
                return json.dumps(dict(actual_cuda_runtime_uuids=['GPU-1']))
            test = self
            class Model:
                def __init__(self, argv, env, **kwargs):
                    self.pid, self.returncode, self.polls = 900, None, 0; models.append(self)
                    self.job = Path(argv[argv.index('--job-dir') + 1])
                    self.launch = Path(argv[argv.index('--launch-dir') + 1])
                    self.env, self.argv = env, argv
                    # Exercise the real continuation-time device callback with
                    # the actual admission binding, before any fake batch result.
                    # Only CUDA's device-count boundary is a CPU substitute.
                    from reproduction.operations.qwen7b_resample import worker as scientific
                    s = scientific.s
                    with patch.dict(os.environ, env), patch.dict(sys.modules,
                            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))), \
                            patch.dict(s._CACHED, {}, clear=True), \
                            patch.object(s, '_configure_one_visible_gpu', s._configure_one_visible_gpu), \
                            patch.object(s, '_build_llm', s._build_llm):
                        binding = read(self.job / 'allocation.json')
                        installed = worker.shared.multi_sampler(pool, binding, SimpleNamespace(tp=1, launch_dir=str(self.launch)))
                        device = installed.s._configure_one_visible_gpu(0)
                        test.assertEqual(device['originally_assigned_device'], binding['original_cuda_visible_devices'])
                        test.assertEqual(device['gpu_uuids'], ['GPU-1'])
                        test.assertEqual(device['native_unit'], unit)
                    if not fail: write(self.launch / 'model-initialized.json', dict(max_model_len=4096))
                def poll(self):
                    self.polls += 1
                    if self.polls == 1: return None
                    self.returncode = 1 if fail else 0
                    if not fail:
                        result = dict(status='POOL_BATCH_BOUNDARY_TIME_LIMIT', run_id=control.RUN_ID, completed_own_batches=1,
                            questions=128, trajectories=640, global_goal_complete=False)
                        write(self.job / 'progress.json', result); write(self.job / 'sampler-exit.json', result)
                    return self.returncode
            def execute_model(_, argv, env):
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
                self.assertEqual(env['Q7B_SLOT_OWNER_DRAIN'], str(cpu_job / 'drain-owner.json'))
                evidence = read(Path(argv[argv.index('--allocation-evidence') + 1]))
                self.assertEqual(evidence['controller_birth'], 500)
                self.assertIs(type(evidence['controller_birth']), int)
                self.assertEqual(evidence['boot_id'], 'test-real-boot-id')
                with patch.dict(os.environ, env), patch.object(sys, 'argv', argv[2:]): worker.main()
            def command(argv, **kwargs):
                if argv[0] == 'systemctl':
                    return SimpleNamespace(returncode=0, stdout=('LoadState=loaded\nMainPID=314\nActiveState=active\n' if starts else
                        'LoadState=not-found\nMainPID=0\nActiveState=inactive\n'), stderr='')
                self.assertEqual(argv[0], 'systemd-run'); starts.append(argv)
                # Execute the actual generated bootstrap, replacing OS boundaries only.
                path = Path(argv[-1]); namespace = dict(__file__=str(path))
                try:
                    with patch('os.execvpe', side_effect=execute_model): exec(compile(path.read_bytes(), str(path), 'exec'), namespace)
                except ValueError:
                    if not fail: raise
                finally:
                    if namespace.get('log'): namespace['log'].close()
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            def transport(script, node):
                self.assertEqual(node, 'h100-10'); assemblies.append(script)
                output = io.StringIO()
                with contextlib.redirect_stdout(output): exec(compile(script, '<actual-main-REMOTE>', 'exec'), {})
                return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr='')
            def invoke(action):
                argv = ['relay_primary_control', action, '--slot', slot, '--cpu-ready', str(cpu_ready_path),
                    '--instance', 'v2',
                    '--cpu-ready-sha256', sha(cpu_ready_path), '--intent-id', 'cpu-once',
                    '--published-commit', 'f' * 40, '--published-ref', 'refs/heads/codex/cpu-only']
                with patch.object(sys, 'argv', argv): control.main()
            original_capture = worker.capture_context
            def capture(*args):
                with patch('time.time', return_value=now + 4 * len(contexts)):
                    value = original_capture(*args)
                contexts.append(value); return value
            original_file = worker.consumer.__file__
            try:
                patches = [patch.object(control, 'ROOT', local), patch.object(control, 'BASE', str(base)), patch.object(control, 'PHYSICAL', str(base)),
                    patch.object(control.old, 'transport', side_effect=transport),
                    patch.object(control.old, 'verify_publication', return_value=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED')),
                    patch('socket.gethostname', return_value='h100-10'), patch('os.geteuid', return_value=0),
                    patch('os.getpid', return_value=314), patch('os.stat', side_effect=stat), patch.object(Path, 'read_text', read_text),
                    patch.object(worker, 'process', side_effect=process), patch.object(worker, 'capture_context', side_effect=capture),
                    patch.object(worker.shared, 'load_pool', return_value=pool), patch.object(worker.shared, 'foreign_contexts', return_value=[]),
                    patch('subprocess.run', side_effect=command), patch('subprocess.check_output', side_effect=check_output),
                    patch('subprocess.Popen', side_effect=Model), patch('time.time', return_value=now), patch('time.sleep'),
                    patch('signal.signal'), patch('os.dup2'), patch('os.chdir'), patch.object(sys, 'platform', 'linux'),
                    patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()),
                    patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(dict(ready=True, pid=1490465,
                        run_id=control.RUN_ID, gpu_shards=4, logical_flat_shards=64, encoder_dtype='float16', index_dtype='float16',
                        original_global_ids=True, endpoint=pool.URL)).encode()))]
                with contextlib.ExitStack() as stack:
                    for item in patches: stack.enter_context(item)
                    invoke('stage')
                    root = base / 'preparation' / directory
                    write(root / 'slot-context.json', dict(run_id=control.RUN_ID, unit=unit, unsubmitted=True))
                    original = (root / 'slot-context.json').read_bytes()
                    invoke('submit'); intent = (root / 'native-submit-intent.json').read_bytes()
                    self.assertEqual(json.loads(intent)['instance'], 'v2')
                    self.assertEqual(read(root / 'slot-context.json')['instance'], 'v2')
                    invoke('submit')
                    self.assertEqual((root / 'native-submit-intent.json').read_bytes(), intent)
                    self.assertEqual((root / ('preintent-context-' + hashlib.sha256(original).hexdigest() + '.json')).read_bytes(), original)
            finally: worker.consumer.__file__ = original_file
            self.assertEqual((len(starts), len(contexts), len(models), len(assemblies)), (1, 2, 1, 3))
            self.assertEqual(set(stops), {900}); self.assertNotIn(700, stops); self.assertNotIn(701, stops)
            self.assertEqual(len(locks), 2)
            self.assertTrue(any('/pool/gpu-locks/GPU-1.lock' in path for path, _ in locks))
            self.assertTrue(any('/pool/relay-live-slots/' in path for path, _ in locks))
            for _, fd in locks:
                with self.assertRaises(OSError): os.fstat(fd)
            job = next((base / 'pool/jobs/job-relay10-primary-314').glob('attempt-*'))
            allocation = read(job / 'allocation.json')
            self.assertEqual((allocation['controller_birth'], allocation['boot_id']), (500, 'test-real-boot-id'))
            self.assertEqual(allocation['source_owner'], cpu_owner)
            self.assertEqual(allocation['source_owner_allocation_sha256'], sha(cpu_job / 'allocation.json'))
            self.assertEqual(allocation['instance'], 'v2')
            self.assertIn('original_cuda_visible_devices', allocation)
            self.assertEqual(models[0].argv[models[0].argv.index('--tp') + 1], '1')
            if fail:
                self.assertTrue((job / 'failure.json').exists()); self.assertFalse((job / 'complete.json').exists())
                self.assertNotIn('typed-initialization-oom-history.json', [p.name for p in job.iterdir()])
            else:
                self.assertEqual(read(job / 'complete.json')['trajectories'], 640)

    def test_real_main_remote_bootstrap_controller_seals_and_never_replays(self): self.run_pipeline()

    def test_failed_http_sampler_is_not_oom_retried_and_cleanup_only_owns_sampler(self): self.run_pipeline(fail=True)


if __name__ == '__main__': unittest.main()
