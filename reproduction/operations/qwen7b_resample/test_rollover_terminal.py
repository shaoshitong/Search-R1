"""CPU actual-file fixtures for dynamic, terminal-only rollover contracts."""
import ast
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import rollover_terminal as terminal
from . import rollover_terminal_control as control
from . import guarded_rollover_control as guarded
from . import legacy_relay_owner as cpu
from . import queued_health
from . import queued_generation_control as queued
from . import queued_relay_owner as queued_cpu
from . import queued_slot_consumer
from . import rollover_source_slots as slots
from . import rollover_retrieval_owner as rollover_owner
from . import queued_relay_slots as relay_slots
from . import rollover_paths


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RolloverTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, dual_view=False):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / 'alias' if dual_view else Path(folder)
            physical = Path(folder) / 'physical' if dual_view else base
            base.mkdir(exist_ok=True);run = terminal.RUN_ID
            write(base / 'config.json', dict(run_id=run))
            failed = base / 'preparation/old-four-failed/complete.json'
            write(failed, dict(status=terminal.FAILED, node='h100-10', run_id=run,
                predecessor_compatibility_status='OWN_PREDECESSOR_GENERATION_TERMINAL',
                termination_kind='OBSERVED_NATIVE_FAILURE_NOT_CONTROLLED_STOP', controlled_stop=False,
                release_observations=[dict(epoch=i, remaining_known_pids=[], own_gpu_contexts=[], loopback_28831_listeners=[], own_cgroups_empty=True) for i in (1, 4)]))
            def row(name, pid, node, borrows=(), kind='model-or-cpu-owner'):
                pattern = 'job-' + node + '-'
                if dual_view and node == 'h100-10':
                    pattern = 'job-relay10-primary-' if 'primary10-' in name else 'job-legacy-relay-owner-'
                root = base / 'pool/jobs' / (pattern + str(pid)) / 'attempt-fixture'
                allocation = root / 'allocation.json'
                write(allocation, dict(controller_pid=pid, native_unit=name, run_id=run, node=node))
                (root / 'sampler.log').write_text('genuine original terminal log\n')
                (root / 'bootstrap.py').write_text('# original retained entry\n')
                return dict(unit=name, pid=pid, birth=str(pid + 100), node=node, borrows_units=list(borrows),
                    allocation_path=str(allocation), allocation_sha256=digest(allocation), metadata_roots=[str(root)],
                    original_owned_births={str(pid): str(pid + 100)}, allocation_kind=kind)
            owner_unit = terminal.unit('cpu-relay10-v1')
            n10rows = [row(terminal.unit('primary10-' + slot + '-' + terminal.PRIMARY_INSTANCES[slot]), 7001 + i, 'h100-10', (owner_unit,))
                       for i, slot in enumerate(cpu.TOPOLOGIES['tp4'])]
            n10rows.append(row(owner_unit, 7000, 'h100-10'))
            ready = dict(status='LEGACY_SOURCE_RELAY_OWNER_READY', role='current-oldV6-source', run_id=run,
                slots=cpu.TOPOLOGIES['tp4'], owner=dict(unit=owner_unit, pid=7000, birth='7100'),
                allocation_path=n10rows[-1]['allocation_path'],
                allocation_sha256=n10rows[-1]['allocation_sha256'])
            ready_path = base / 'preparation/cpu-ready.json';write(ready_path, ready)
            common = dict(schema_version=2, run_id=run, original_partial_outputs_retained=True, collector_signals_no_processes=True)
            node10 = dict(common, node='h100-10', cohort_kind='legacy-cpu-and-primary-mixed',
                primary_instances=dict(terminal.PRIMARY_INSTANCES), ordered_units=n10rows,
                source={key: n10rows[-1][key] for key in ('unit', 'pid', 'birth')}, cpu_owner_ready=ready,
                cpu_owner_ready_path=str(ready_path), cpu_owner_ready_sha256=digest(ready_path),
                prior_failed_proof_path=str(failed), prior_failed_proof_sha256=digest(failed), prior_failed_proof_status=terminal.FAILED)
            n10path = base / 'preparation/rollover10/plan.json';write(n10path, node10)
            n5rows = [row(terminal.unit(name), 6001 + i, 'h100-5', (terminal.core.SOURCE['unit'],))
                      for i, name in enumerate(terminal.SOURCE_MODELS[:-1])]
            n5rows.append(row('sst-q7b-retrieval-relay5-v1-' + run[:12] + '.service', 1654561, 'h100-5',
                (terminal.core.SOURCE['unit'],), 'retained-upstream-ready'))
            upstream_path = Path(n5rows[-1]['allocation_path'])
            write(upstream_path, dict(status='RELAY_LISTENING', role='upstream', source_pid=1490465, source_birth='6808057',
                process=dict(pid=1654561, birth=n5rows[-1]['birth'], cgroup='0::/system.slice/' + n5rows[-1]['unit'] + '\n')))
            n5rows[-1]['allocation_sha256'] = digest(upstream_path)
            n5rows.append(row(terminal.core.SOURCE['unit'], 1490214, 'h100-5'))
            n5rows[-1]['birth'] = '6807965';n5rows[-1]['original_owned_births'] = {'1490214': '6807965', '1490465': '6808057'}
            source5 = dict(common, node='h100-5', cohort_kind='source5-v6-eight-models', ordered_units=n5rows,
                source=dict(unit=terminal.core.SOURCE['unit'], pid=1490214, birth='6807965'),
                source_mount_ns_inode=111, source_net_ns_inode=222, remote_node10_plan_path=str(n10path),
                remote_node10_plan_sha256=digest(n10path), remote_node10_proof_path=str(n10path.parent / 'complete.json'))
            n5path = base / 'preparation/rollover5/plan.json';write(n5path, source5)
            clock, queries, births, commands = [1000], [], [], []
            def query(name):
                queries.append(name)
                return dict(Id=name, MainPID='0', ActiveState='failed', ControlGroup='', Result='exit-code')
            def birth(pid):births.append(str(pid));return None
            def command(argv, **kwargs):
                commands.append(argv);assert argv[0] == 'nvidia-smi';return '9999\n'
            def sleep(seconds):clock[0] += seconds
            real_stat, real_text = os.stat, Path.read_text
            def stat(path, *a, **k):
                if str(path) == '/proc/self/ns/mnt':return SimpleNamespace(st_ino=111)
                if str(path) == '/proc/self/ns/net':return SimpleNamespace(st_ino=222)
                return real_stat(path, *a, **k)
            def text(path, *a, **k):return '0::/system.slice/independent-collector.service\n' if str(path) == '/proc/self/cgroup' else real_text(path, *a, **k)
            class Socket:
                def __enter__(self):return self
                def __exit__(self, *a):pass
                def bind(self, value):assert value == ('127.0.0.1', 28831)
            with patch.object(terminal, 'BASE', base), patch.object(terminal, 'PHYSICAL', physical), \
                 patch.object(terminal.core, 'BASE', base), patch.object(terminal.core, 'PHYSICAL', physical), \
                 patch('os.geteuid', return_value=0), patch.object(terminal.core, 'query', side_effect=query), \
                 patch.object(terminal.core, 'owned', return_value={}), patch.object(terminal.core, 'birth', side_effect=birth), \
                 patch.object(terminal.subprocess, 'check_output', side_effect=command), \
                 patch.object(terminal.time, 'time', side_effect=lambda: clock[0]), patch.object(terminal.time, 'sleep', side_effect=sleep), \
                 patch('os.stat', side_effect=stat), patch.object(Path, 'read_text', text), patch.object(terminal.socket, 'socket', return_value=Socket()):
                yield SimpleNamespace(base=base, physical=physical, node10=node10, source5=source5, n10path=n10path, n5path=n5path,
                    failed=failed, queries=queries, births=births, commands=commands)

    def test_dynamic_cpu_and_mixed_cohort_then_source5_seals_actual_terminal_inodes(self):
        with self.fixture() as f:
            failed_original = f.failed.read_bytes()
            with patch('socket.gethostname', return_value='h100-10'):
                n10 = terminal.collect(f.n10path, digest(f.n10path))
                n10proof = f.n10path.parent / 'complete.json'
                admitted = queued_cpu.verify_predecessor(None, SimpleNamespace(predecessor_proof=n10proof, predecessor_sha256=digest(n10proof)))
                self.assertTrue(admitted['actual_terminal_only_proof'])
            f.queries.clear();f.births.clear()
            originals = {path: path.stat().st_ino for path in terminal.metadata(f.source5['ordered_units'])}
            with patch('socket.gethostname', return_value='h100-5'):
                value = terminal.collect(f.n5path, digest(f.n5path))
                proof = f.n5path.parent / 'complete.json'
                accepted = terminal.verify_source_proof(proof, digest(proof), f.base, terminal.RUN_ID)
            self.assertEqual(n10['source']['pid'], 7000, 'Future owner PID comes from capture, never old V5')
            expected = {terminal.unit('primary10-' + name + '-' + instance) for name, instance in terminal.PRIMARY_INSTANCES.items()}
            self.assertEqual({row['unit'] for row in n10['ordered_units'][:-1]}, expected)
            self.assertEqual(accepted['status'], terminal.STATUS)
            self.assertFalse(value['controlled_stop'])
            self.assertEqual(value['termination_kind'], 'OBSERVED_TERMINAL_STATES_NO_SIGNALS_BY_COLLECTOR')
            self.assertEqual(value['release_observations'][1]['epoch'] - value['release_observations'][0]['epoch'], 3)
            self.assertEqual(len(value['ordered_units']), 9, 'Eight source models plus upstream must all be terminal')
            self.assertFalse(any('primary10-' in name or 'cpu-relay10-' in name for name in f.queries))
            self.assertTrue(set(f.births) <= set(value['original_owned_births']))
            self.assertEqual(f.failed.read_bytes(), failed_original)
            self.assertEqual({path: path.stat().st_ino for path in originals}, originals)
            self.assertTrue(all(cmd[0] == 'nvidia-smi' for cmd in f.commands))

    def test_old_v5_and_controlled_status_cannot_masquerade_as_replacement(self):
        with self.fixture() as f:
            fake = copy.deepcopy(f.node10)
            fake['source'] = dict(unit=terminal.core.CONSUMER['unit'], pid=2371322, birth='1958624331')
            with self.assertRaises(AssertionError):terminal.validate_plan(fake)
            wrong_instance = copy.deepcopy(f.node10)
            wrong_instance['primary_instances']['gpu2-tp1'] = 'v2'
            with self.assertRaises(AssertionError):terminal.validate_plan(wrong_instance)
            failed_instance = copy.deepcopy(f.node10)
            failed_instance['primary_instances']['gpu2-tp1'] = 'v1'
            with self.assertRaises(AssertionError):terminal.validate_plan(failed_instance)
            wrong_allocation = copy.deepcopy(f.node10['ordered_units'][0])
            wrong_allocation['unit'] = wrong_allocation['unit'].replace('-v2-', '-v3-')
            with self.assertRaises(AssertionError):terminal.verify_allocation(wrong_allocation)
            with patch('socket.gethostname', return_value='h100-10'):
                terminal.collect(f.n10path, digest(f.n10path))
            proof = f.n10path.parent / 'complete.json'
            wrong = terminal.core.read(proof);wrong['status'] = 'OWN_PREDECESSOR_GENERATION_TERMINAL';write(proof, wrong)
            with self.assertRaises(AssertionError):terminal.verify_proof(proof, digest(proof), 'h100-10')
            prior = terminal.core.read(f.failed);prior['controlled_stop'] = True;write(f.failed, prior)
            f.node10['prior_failed_proof_sha256'] = digest(f.failed)
            with self.assertRaises(AssertionError):terminal.verify_prior_failed(f.node10)

    def test_busy_original_model_prevents_proof_and_missing_retriever_rejects(self):
        with self.fixture() as f:
            fake = copy.deepcopy(f.source5);fake['ordered_units'][-1]['original_owned_births'].pop('1490465')
            with self.assertRaises(AssertionError):terminal.validate_plan(fake)
            with patch('socket.gethostname', return_value='h100-10'), patch.object(terminal.core, 'owned', return_value={'7001': '7101'}):
                with self.assertRaises(AssertionError):terminal.collect(f.n10path, digest(f.n10path))
            self.assertFalse((f.n10path.parent / 'complete.json').exists())

    def test_new_controls_consume_new_contract_without_mutating_old_remote(self):
        previous = guarded.original.REMOTE
        source = guarded.remote_source();ast.parse(source);ast.parse(control.REMOTE);ast.parse(queued.remote_source())
        self.assertIn('verify_source_proof', source)
        self.assertEqual(guarded.original.REMOTE, previous)
        identity, files, manifest = guarded.bundle()
        self.assertEqual(manifest['predecessor_contract'], terminal.STATUS)
        self.assertIn('rollover_terminal.py', manifest['source_files'])
        self.assertIn(b"str(stage/'rollover_retrieval_owner.py')", files['generation_bootstrap.py'])
        self.assertNotIn('systemctl\', \'stop', Path(terminal.__file__).read_text())

    def test_eight_source_slots_share_exact_gpu3_contract_only_in_new_entries(self):
        original_registry = guarded.original.SLOTS
        original_specs = copy.deepcopy(original_registry)
        owner_registry = rollover_owner.guard.owner.SLOTS
        owner_specs = copy.deepcopy(owner_registry)
        _, owner_files, owner_manifest = guarded.bundle()
        _, model_files, model_manifest = guarded.bundle('slot', slot='gpu3-tp1')
        self.assertEqual(set(owner_manifest['slots']), set(slots.DEFAULT_SLOTS))
        for name, spec in slots.TP1_SLOTS.items():self.assertEqual(json.loads(owner_files['slots.json'])[name], spec)
        for name in ('base-tp4', 'retrieval-tp4', 'gpu1-tp1', 'gpu2-tp1'):
            self.assertEqual(owner_manifest['slots'][name], original_specs[name])
        self.assertEqual(json.loads(model_files['slots.json']), {'gpu3-tp1': slots.GPU3_SLOT})
        self.assertIn(b"str(stage/'rollover_retrieval_owner.py')", owner_files['generation_bootstrap.py'])
        self.assertIn(b"'--slot-mode','source5'", model_files['generation_bootstrap.py'])
        self.assertIs(guarded.original.SLOTS, original_registry)
        self.assertEqual(original_registry, original_specs)
        calls = []
        try:
            with patch.object(rollover_owner.guard, 'main', side_effect=lambda: calls.append(copy.deepcopy(rollover_owner.guard.owner.SLOTS))):
                rollover_owner.main()
            fake_owner = SimpleNamespace(SLOTS=original_registry)
            fake_worker = SimpleNamespace(owner_module=fake_owner, main=lambda: calls.append(copy.deepcopy(fake_owner.SLOTS)))
            with patch.dict(sys.modules, {'source_slot_consumer_worker': fake_worker, 'rollover_source_slots': slots}), \
                 patch.object(sys, 'argv', ['queued_slot_consumer.py', '--slot-mode', 'source5']), \
                 patch.object(queued_health, 'install'):
                queued_slot_consumer.main()
            self.assertEqual([row['gpu3-tp1'] for row in calls], [slots.GPU3_SLOT, slots.GPU3_SLOT])
            self.assertEqual(owner_registry, owner_specs)
            self.assertEqual(original_registry, original_specs)
        finally:rollover_owner.guard.owner.SLOTS = owner_registry

    def test_future_health_wait_matches_queue_without_changing_retrieval_requests(self):
        calls = []
        def request(url, *args, **kwargs):calls.append((url, args, kwargs));return object()
        with patch.object(queued_health.urllib.request, 'urlopen', new=request):
            queued_health.install()
            queued_health.urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=3)
            queued_health.urllib.request.urlopen('http://127.0.0.1:28831/retrieve', data=b'original', timeout=3)
        self.assertEqual(calls[0][2]['timeout'], 60)
        self.assertEqual(calls[1][2], dict(data=b'original', timeout=3))
        _, files, manifest = guarded.bundle('slot', slot='gpu2-tp1')
        ast.parse(files['generation_bootstrap.py'])
        self.assertIn(b"'--slot-mode','source5'", files['generation_bootstrap.py'])
        for role in ('upstream', 'downstream', 'slot'):
            _, files, manifest = queued.bundle(role, generation='owner-v2', slot='tp4-0-3' if role == 'slot' else None)
            ast.parse(files['relay_generation_bootstrap.py'])
            self.assertIn(b"str(stage/'retrieval_relay_backpressure.py')", files['relay_generation_bootstrap.py'])
            self.assertIn(b"str(stage/'queued_relay_owner.py')", files['relay_generation_bootstrap.py'])
            self.assertIn(b"'--slot-mode','node10'", files['relay_generation_bootstrap.py'])
            self.assertEqual(manifest['readiness_timeout_seconds'], 60)

    def test_six_node10_slots_match_owner_entry_model_entry_and_actual_bootstrap(self):
        original_registry, original_defaults = queued.original.SLOTS, queued.original.DEFAULT_SLOTS
        original_specs = copy.deepcopy(original_registry)
        owner = queued_cpu.owner
        owner_registry, owner_defaults, predecessor = owner.SLOTS, owner.DEFAULT_SLOTS, owner.verify_predecessor
        owner_specs = copy.deepcopy(owner_registry)
        _, files, manifest = queued.bundle('downstream')
        declared = json.loads(files['slots.json'])
        self.assertEqual(set(declared), set(relay_slots.DEFAULT_SLOTS))
        self.assertEqual(manifest['finite_node10_slots'], declared)
        self.assertEqual(manifest['files']['queued_relay_slots.py']['sha256'],
                         hashlib.sha256(files['queued_relay_slots.py']).hexdigest())
        for name, spec in relay_slots.TP1_SLOTS.items():self.assertEqual(declared[name], spec)
        for name in ('tp4-0-3', 'tp4-4-7'):self.assertEqual(declared[name], original_specs[name])
        self.assertIs(queued.original.SLOTS, original_registry)
        self.assertEqual(queued.original.DEFAULT_SLOTS, original_defaults)
        calls = []
        try:
            with patch.object(owner, 'main', side_effect=lambda: calls.append(copy.deepcopy(owner.SLOTS))), \
                 patch.object(queued_health, 'install'):
                queued_cpu.main()
                self.assertEqual(owner.DEFAULT_SLOTS, relay_slots.DEFAULT_SLOTS)
            fake_owner = SimpleNamespace(SLOTS=owner_registry, DEFAULT_SLOTS=owner_defaults)
            fake_worker = SimpleNamespace(relay_owner=fake_owner,
                main=lambda: calls.append(copy.deepcopy(fake_owner.SLOTS)))
            with patch.dict(sys.modules, {'relay_slot_consumer_worker': fake_worker, 'queued_relay_slots': relay_slots}), \
                 patch.object(sys, 'argv', ['queued_slot_consumer.py', '--slot-mode', 'node10']), \
                 patch.object(queued_health, 'install'):
                queued_slot_consumer.main()
                self.assertEqual(fake_owner.DEFAULT_SLOTS, relay_slots.DEFAULT_SLOTS)
            self.assertEqual(calls, [declared, declared])
            self.assertEqual(owner_registry, owner_specs)
            self.assertEqual(original_registry, original_specs)
        finally:
            owner.SLOTS, owner.DEFAULT_SLOTS, owner.verify_predecessor = owner_registry, owner_defaults, predecessor
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            with patch.object(queued.original, 'BASE', str(base)):
                identity, files, manifest = queued.bundle('slot', slot='gpu2-tp1')
            stage = base / 'preparation' / manifest['directory'] / identity;stage.mkdir(parents=True)
            for name, data in files.items():(stage / name).write_bytes(data)
            write(stage / 'manifest.json', manifest)
            up = base / 'upstream-owner-ready.json';write(up, dict(transport_ready_path=str(base / 'upstream-transport.json')))
            ready = base / 'downstream-owner-ready.json'
            write(ready, dict(slots=declared, owner=dict(unit='future-cpu-owner.service'),
                consumer_deadline_epoch=20000, upstream_owner_ready=str(up), job_ready_path=str(base / 'downstream/relay-owner-ready.json')))
            write(stage.parent / 'submit-context.json', dict(downstream_owner_ready=str(ready), downstream_owner_ready_sha256=digest(ready)))
            group = '/system.slice/' + manifest['unit']
            detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=103079215104\nActiveEnterTimestampMonotonic=1000000\n' % (manifest['unit'], group)
            real_stat, real_text = os.stat, Path.read_text
            def stat(path, *a, **k):
                if str(path) in ('/proc/self/ns/mnt', '/proc/1/ns/mnt', '/proc/self/ns/net', '/proc/1/ns/net'):
                    return SimpleNamespace(st_ino=2 if str(path) == '/proc/self/ns/mnt' else 1)
                return real_stat(path, *a, **k)
            def text(path, *a, **k):return '0::' + group + '\n' if str(path) == '/proc/self/cgroup' else real_text(path, *a, **k)
            namespace = dict(__file__=str(stage / 'relay_generation_bootstrap.py'))
            try:
                with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                     patch('socket.gethostname', return_value='h100-10'), patch('os.stat', side_effect=stat), \
                     patch.object(Path, 'read_text', text), patch('time.time', return_value=1000), \
                     patch('time.monotonic', return_value=1), patch('subprocess.check_output', return_value=detail), \
                     patch('os.dup2'), patch('os.chdir'), patch('builtins.print'), patch('os.execvpe') as execute:
                    exec(compile(files['relay_generation_bootstrap.py'], namespace['__file__'], 'exec'), namespace)
            finally:
                if 'log' in namespace:namespace['log'].close()
            _, command, env = execute.call_args.args
            option = lambda key: command[command.index(key) + 1]
            self.assertIn(str((stage / 'queued_slot_consumer.py').resolve()), command)
            self.assertEqual(option('--slot-mode'), 'node10')
            self.assertEqual([option(key) for key in ('--model-gpus', '--tp', '--model-max-seqs', '--model-batch-tokens', '--model-kv-block-cap', '--model-reserve-gib')],
                             ['2', '1', '64', '8192', '4096', '1.0'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'], '1')
            self.assertEqual(json.loads(files['slots.json']), declared)

    def test_import_contract_and_handoff_commands_use_new_registry_without_global_leaks(self):
        old_owner_slots = queued_cpu.owner.SLOTS
        old_specs = copy.deepcopy(old_owner_slots)
        self.assertEqual(queued_cpu.SLOTS, relay_slots.specifications(old_specs))
        self.assertEqual(queued_cpu.DEFAULT_SLOTS, relay_slots.DEFAULT_SLOTS)
        self.assertIs(queued_cpu.owner.SLOTS, old_owner_slots)
        handoff = (control.ROOT / 'work/qwen7b-guarded-rollover-handoff-20261021.txt').read_text()
        variables = {key: value for key, value in (line.split('=', 1) for line in handoff.splitlines()
            if line.startswith(('SOURCE_SLOTS=', 'NODE10_SLOTS=', 'PRIMARY_INSTANCES=')))}
        before = [(module, module.bundle, module.REMOTE, module.SLOTS, module.DEFAULT_SLOTS)
                  for module in (queued.original, guarded.original)]
        checked = set()
        with patch.object(control.old, 'transport', side_effect=AssertionError('No remote action in CLI review')), \
             patch.object(control.old, 'verify_publication', side_effect=AssertionError('No Git action in CLI review')):
            for line in handoff.splitlines():
                if not line.startswith('python3 -m $M.'):continue
                tokens = shlex.split(line);name = tokens[2].removeprefix('$M.')
                if name not in ('queued_generation_control', 'guarded_rollover_control', 'guarded_rollover_slot_control', 'rollover_terminal_control'):continue
                args = [variables.get(token[1:], token) if token.startswith('$') else token for token in tokens[3:]]
                args[1 if name == 'queued_generation_control' else 0] = 'plan'
                entry = dict(queued_generation_control=queued.main, guarded_rollover_control=guarded.main,
                    guarded_rollover_slot_control=lambda: guarded.main('slot'), rollover_terminal_control=control.main)[name]
                with self.subTest(entry=name, arguments=args), patch.object(sys, 'argv', [name, *args]), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    entry()
                manifest = json.loads(output.getvalue())
                if name == 'queued_generation_control':self.assertEqual(set(manifest['finite_node10_slots']), set(relay_slots.DEFAULT_SLOTS))
                checked.add(name)
        self.assertEqual(len(checked), 4)
        for module, bundle, remote, specs, defaults in before:
            self.assertIs(module.bundle, bundle);self.assertEqual(module.REMOTE, remote)
            self.assertIs(module.SLOTS, specs);self.assertEqual(module.DEFAULT_SLOTS, defaults)
        self.assertEqual(old_owner_slots, old_specs)

    def test_manifest_hashes_and_staged_source_verifier_accept_actual_new_terminal_contract(self):
        bundles = [guarded.bundle('owner'), guarded.bundle('slot', slot='gpu3-tp1')]
        bundles += [queued.bundle(role, slot='gpu2-tp1' if role == 'slot' else None)
                    for role in ('upstream', 'downstream', 'slot')]
        for _, files, manifest in bundles:
            for section in ('files', 'source_files'):
                for name, row in manifest[section].items():
                    with self.subTest(role=manifest['role'], section=section, file=name):
                        self.assertEqual(row, dict(bytes=len(files[name]), sha256=hashlib.sha256(files[name]).hexdigest()))
        _, files, _ = bundles[0]
        with self.fixture() as f:
            with patch('socket.gethostname', return_value='h100-10'):terminal.collect(f.n10path, digest(f.n10path))
            with patch('socket.gethostname', return_value='h100-5'):
                terminal.collect(f.n5path, digest(f.n5path))
                stage = f.base / 'exact-published-future-stage';stage.mkdir()
                (stage / 'rollover_terminal.py').write_bytes(files['rollover_terminal.py'])
                spec = importlib.util.spec_from_file_location('staged_rollover_terminal_test', stage / 'rollover_terminal.py')
                runtime = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, {'predecessor_generation_terminal': terminal.core, 'rollover_paths': rollover_paths}):spec.loader.exec_module(runtime)
                runtime.BASE, runtime.PHYSICAL = f.base, f.base
                proof = f.n5path.parent / 'complete.json'
                accepted = runtime.verify_source_proof(proof, digest(proof), f.base, terminal.RUN_ID)
                self.assertEqual(accepted['status'], terminal.STATUS)
                wrong = runtime.core.read(proof);wrong['status'] = 'OWN_PREDECESSOR_GENERATION_TERMINAL';write(proof, wrong)
                with self.assertRaises(AssertionError):runtime.verify_source_proof(proof, digest(proof), f.base, terminal.RUN_ID)

    def test_private_alias_then_host_physical_view_capture_collect_and_owner_controls_keep_sha(self):
        with self.fixture(dual_view=True) as f:
            physical = terminal.core.physical
            ready_path = Path(f.node10['cpu_owner_ready_path'])
            ready = f.node10['cpu_owner_ready'];ready['allocation_path'] = physical(ready['allocation_path'])
            write(ready_path, ready)
            identity, files, manifest = control.bundle('h100-10')
            def stage_bundle(identity, files, manifest, status):
                root = terminal.readable(f.base) / 'preparation' / manifest['directory']
                stage = root / identity;stage.mkdir(parents=True)
                for name, data in files.items():(stage / name).write_bytes(data)
                write(stage / 'manifest.json', manifest);write(stage / 'stage-receipt.json', dict(status=status))
                return root, stage
            capture_root, _ = stage_bundle(identity, files, manifest, 'EXACT_TERMINAL_COLLECTOR_STAGE_SHA_PASS')
            capture_values = dict(BASE=str(f.base), PHYSICAL=str(f.physical), RUN_ID=terminal.RUN_ID,
                MANIFEST=manifest, IDENTITY=identity, ACTION='capture',
                PUBLICATION=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'),
                CPU_READY=physical(ready_path), CPU_SHA=digest(ready_path), PRIMARY_INSTANCES=terminal.PRIMARY_INSTANCES,
                FAILED_PROOF=physical(f.failed), FAILED_SHA=digest(f.failed))
            live = {row['unit']: row for row in f.node10['ordered_units']}
            def live_query(name):
                row = live[name]
                return dict(Id=name, MainPID=str(row['pid']), ActiveState='active', ControlGroup='/system.slice/' + name)
            def live_owned(group):return live[group.rsplit('/', 1)[1]]['original_owned_births']
            def live_birth(pid):return next(row['birth'] for row in live.values() if row['pid'] == pid)
            with patch('socket.gethostname', return_value='h100-10'), patch.object(terminal.core, 'query', side_effect=live_query), \
                 patch.object(terminal.core, 'owned', side_effect=live_owned), patch.object(terminal.core, 'birth', side_effect=live_birth), \
                 patch.dict(sys.modules, {'rollover_terminal': terminal}), patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                exec(control.REMOTE, capture_values)
            captured_path = capture_root / 'plan.json';captured_sha = digest(captured_path)
            captured = terminal.read(physical(captured_path))
            self.assertEqual(captured['prior_failed_proof_path'], physical(f.failed))
            self.assertFalse(Path(captured['prior_failed_proof_path']).exists(), 'Private view must really hide physical paths')
            f.source5.update(remote_node10_plan_path=physical(captured_path), remote_node10_plan_sha256=captured_sha,
                remote_node10_proof_path=physical(capture_root / 'complete.json'))
            write(f.n5path, f.source5);source_sha = digest(f.n5path)
            with patch('socket.gethostname', return_value='h100-10'):terminal.collect(physical(captured_path), captured_sha)
            with patch('socket.gethostname', return_value='h100-5'):terminal.collect(physical(f.n5path), source_sha)
            done10, done5 = physical(capture_root / 'complete.json'), physical(f.n5path.parent / 'complete.json')
            sha10, sha5 = terminal.sha(done10), terminal.sha(done5)
            proof5 = terminal.read(done5)
            self.assertEqual(proof5['source_plan'], physical(f.n5path))
            self.assertEqual(proof5['remote_node10_proof'], done10)
            self.assertTrue(all(row['path'].startswith(str(f.physical)) for row in proof5['original_files']))
            inode = captured_path.stat().st_ino

            def fake_run(command, **kwargs):
                if command[:2] == ['systemctl', 'show']:
                    return SimpleNamespace(returncode=0, stdout='Id=%s\nLoadState=not-found\nMainPID=0\nActiveState=inactive\n' % command[2], stderr='')
                self.assertEqual(command[0], 'systemd-run')
                return SimpleNamespace(returncode=0, stdout='fake CPU unit start; no process created\n', stderr='')

            def owner_controls(generation):
                gid, gfiles, gmanifest = guarded.bundle(generation=generation)
                groot, _ = stage_bundle(gid, gfiles, gmanifest, 'EXACT_OWNER_GENERATION_STAGE_SHA_PASS')
                common = dict(BASE=str(f.base), PHYSICAL=str(f.physical), RUN_ID=terminal.RUN_ID, ACTION='submit',
                    PUBLICATION=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'), INTENT='fixture-' + generation)
                env = dict(common, IDENTITY=gid, MANIFEST=gmanifest, PREDECESSOR=done5, PREDECESSOR_SHA=sha5,
                    FILES={name: base64.b64encode(data).decode() for name, data in gfiles.items()})
                with patch('socket.gethostname', return_value='h100-5'), patch.object(terminal.subprocess, 'run', side_effect=fake_run), \
                     patch.dict(sys.modules, {'rollover_terminal': terminal}), patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                    exec('import re\n' + guarded.remote_source(), env)
                self.assertEqual(terminal.read(groot / 'submit-context.json')['predecessor_sha256'], sha5)
                # Exercise the real queued control's proof reader too; systemd
                # and its future source/upstream are CPU fixtures, never jobs.
                owner = dict(pid=8100, birth='123')
                contract = dict(contract=queued_cpu.owner.guard.CONTRACT,
                    code_sha256=terminal.sha(queued_cpu.owner.guard.__file__),
                    dependency_path=str(queued_cpu.owner.guard.dependency_path(f.base, terminal.RUN_ID, owner)))
                allocation = f.base / 'preparation' / ('fixture-source-' + generation) / 'allocation.json'
                write(terminal.readable(f.base) / allocation.relative_to(f.base),
                    dict(owner=owner, run_id=terminal.RUN_ID, allocation_role='RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER', remote_relay_guard=contract))
                source = allocation.with_name('owner-ready.json')
                write(terminal.readable(f.base) / source.relative_to(f.base), dict(status='RETRIEVAL_ONLY_OWNER_READY', run_id=terminal.RUN_ID,
                    owner=owner, allocation_path=physical(allocation), allocation_sha256=terminal.sha(allocation),
                    remote_relay_guard=contract, consumer_deadline_epoch=100000))
                upstream = source.with_name('upstream-ready.json')
                write(terminal.readable(f.base) / upstream.relative_to(f.base), dict(role='upstream', run_id=terminal.RUN_ID, source_owner=owner,
                    job_ready_path=str(upstream.with_name('upstream-job-ready.json'))))
                qid, qfiles, qmanifest = queued.bundle('downstream', generation=generation)
                qroot, _ = stage_bundle(qid, qfiles, qmanifest, 'EXACT_RELAY_GENERATION_STAGE_SHA_PASS')
                env = dict(common, IDENTITY=qid, MANIFEST=qmanifest, PREDECESSOR=done10, PREDECESSOR_SHA=sha10,
                    SOURCE_READY=physical(source), SOURCE_SHA=terminal.sha(source), UP_READY=physical(upstream), UP_SHA=terminal.sha(upstream),
                    FILES={name: base64.b64encode(data).decode() for name, data in qfiles.items()})
                with patch('socket.gethostname', return_value='h100-10'), patch.object(terminal.subprocess, 'run', side_effect=fake_run), \
                     patch.dict(sys.modules, {'queued_relay_owner': queued_cpu}), patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                    exec(queued.remote_source(), env)
                self.assertEqual(terminal.read(qroot / 'submit-context.json')['predecessor_sha256'], sha10)

            owner_controls('private-v1')
            f.base.rename(f.physical)
            self.assertFalse(f.base.exists());self.assertTrue(f.physical.exists())
            with patch('socket.gethostname', return_value='h100-5'):
                self.assertEqual(terminal.verify_source_proof(done5, sha5, f.base, terminal.RUN_ID)['status'], terminal.STATUS)
            with patch('socket.gethostname', return_value='h100-10'):
                self.assertTrue(queued_cpu.verify_predecessor(None, SimpleNamespace(predecessor_proof=done10, predecessor_sha256=sha10))['actual_terminal_only_proof'])
                terminal.collect(physical(captured_path), captured_sha)
                with patch.dict(sys.modules, {'rollover_terminal': terminal}), patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                    exec(control.REMOTE, dict(capture_values))
            self.assertEqual(Path(physical(captured_path)).stat().st_ino, inode)
            self.assertEqual(terminal.sha(done5), sha5);self.assertEqual(terminal.sha(done10), sha10)
            owner_controls('host-v1')
            # If both spellings appear but the preferred alias has wrong bytes,
            # expected SHA must reject rather than silently try the other file.
            write(f.failed, dict(status='wrong alias payload'))
            with self.assertRaises(AssertionError):terminal.verify_prior_failed(captured)


if __name__ == '__main__':unittest.main()
