"""CPU-only predecessor proof coverage; every process/stop boundary is stubbed."""
import ast
import base64
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import predecessor_generation_terminal as terminal
from . import retrieval_only_control as control


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def cohort(node='h100-10', failed_stop=False, lingering=False):
    """Temporary own metadata; fake process tables are scoped to this node only."""
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / 'access'
        base.mkdir()
        write(base / 'config.json', dict(run_id=terminal.RUN_ID))
        physical = Path(directory) / 'physical'
        physical.symlink_to(base, target_is_directory=True)
        expected = terminal.SOURCE if node == 'h100-5' else terminal.CONSUMER
        borrower = 'sst-q7b-cpu-borrower-' + terminal.RUN_ID[:12] + '.service'
        rows = [dict(node=node, unit=borrower, pid=1001, birth='111', borrows_units=[expected['unit']]),
                dict(node=node, unit=expected['unit'], pid=expected['pid'], birth=expected['birth'])]
        own, active, calls, stops = {}, {}, [], []
        for ordinal, row in enumerate(rows):
            job = base / 'pool/jobs' / ('job-cpu-%d' % ordinal) / 'attempt-00'
            job.mkdir(parents=True)
            allocation = job / 'allocation.json'
            write(allocation, dict(node=node, run_id=terminal.RUN_ID, controller_pid=row['pid'],
                native_unit=row['unit'], native_cgroup='/system.slice/' + row['unit']))
            (job / 'sampler.log').write_text('original CPU fixture log\n')
            row.update(allocation_path=str(allocation), allocation_sha256=digest(allocation), metadata_roots=[str(job)])
            own[row['unit']] = {str(row['pid']): str(row['birth']), str(row['pid'] + 1): 'child-' + str(ordinal)}
            active[row['unit']] = True
        if node == 'h100-5':
            own[expected['unit']][str(terminal.SOURCE['retriever_pid'])] = terminal.SOURCE['retriever_birth']
        plan = dict(run_id=terminal.RUN_ID, node=node,
            source={key: expected[key] for key in ('unit', 'pid', 'birth')}, ordered_units=rows,
            original_partial_outputs_retained=True, old_loops_have_no_drain_hook=True)
        root = base / 'preparation/cpu-terminal'
        root.mkdir(parents=True)
        plan_path = root / 'plan.json'
        remote = root / 'remote-node10.json'
        if node == 'h100-5':
            remote_source = terminal.CONSUMER
            write(remote, dict(status='OWN_PREDECESSOR_GENERATION_TERMINAL', run_id=terminal.RUN_ID, node='h100-10',
                source_unit=remote_source['unit'], source_pid=remote_source['pid'], source_birth=remote_source['birth'],
                units=[dict(node='h100-10', unit=remote_source['unit'], pid=remote_source['pid'], birth=remote_source['birth'])],
                original_owned_births={str(remote_source['pid']): remote_source['birth']},
                release_observations=[dict(original_births_gone=True, own_cgroups_empty=True, own_gpu_contexts=[])] * 2))
            plan.update(remote_node10_proof=str(remote), remote_node10_proof_sha256=digest(remote))
        clock = [1000.0]

        def query(unit):
            calls.append(('query', unit))
            row = next(row for row in rows if row['unit'] == unit)
            return dict(Id=unit, MainPID=str(row['pid']) if active[unit] else '0',
                ActiveState='active' if active[unit] else 'inactive', Result='success', ExecMainStatus='0',
                ControlGroup='/system.slice/' + unit if active[unit] else '')

        def owned(group):
            unit = group.removeprefix('/system.slice/')
            if active[unit] or (lingering and unit == borrower):
                return dict(own[unit])
            return {}

        def birth(pid):
            calls.append(('birth', int(pid)))
            for unit, births in own.items():
                if str(pid) in births and (active[unit] or (lingering and unit == borrower)):
                    return births[str(pid)]
            return None

        def stop(argv, **kwargs):
            assert argv[:2] == ['systemctl', 'stop'] and argv[2] in active
            stops.append(argv[2])
            if not (failed_stop and argv[2] == borrower):
                active[argv[2]] = False
            return subprocess.CompletedProcess(argv, 1 if failed_stop else 0, '', 'stubbed stop')

        def check_output(argv, **kwargs):
            assert argv == ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
            return '999999\n'  # A foreign GPU PID must never be signalled or captured.

        original_text, original_stat = Path.read_text, os.stat

        def text(path, *args, **kwargs):
            return '0::/system.slice/cpu-terminal-producer.service\n' if str(path) == '/proc/self/cgroup' else original_text(path, *args, **kwargs)

        def stat(path, *args, **kwargs):
            if str(path) in ('/proc/self/ns/mnt', '/proc/1490214/ns/mnt'):
                return SimpleNamespace(st_ino=101)
            return original_stat(path, *args, **kwargs)

        def invoke(execute=False):
            write(plan_path, plan)
            argv = ['terminal-cpu', '--plan', str(plan_path), '--plan-sha256', digest(plan_path)]
            if execute:
                argv.append('--execute')
            output = io.StringIO()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
                terminal.main()
            return json.loads(output.getvalue())

        with patch.object(terminal, 'BASE', base), patch.object(terminal, 'PHYSICAL', physical), \
             patch.object(terminal, 'query', side_effect=query), patch.object(terminal, 'owned', side_effect=owned), \
             patch.object(terminal, 'birth', side_effect=birth), patch('os.geteuid', return_value=0), \
             patch('socket.gethostname', return_value=node), patch('os.stat', side_effect=stat), \
             patch.object(Path, 'read_text', text), patch.object(terminal.subprocess, 'run', side_effect=stop), \
             patch.object(terminal.subprocess, 'check_output', side_effect=check_output), \
             patch.object(terminal.time, 'time', side_effect=lambda: clock[0]), \
             patch.object(terminal.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            yield SimpleNamespace(base=base, physical=physical, plan=plan, rows=rows, root=root,
                plan_path=plan_path, remote=remote, invoke=invoke, stops=stops, calls=calls,
                own=own, active=active, borrower=borrower, source=expected)


class PlanAndOrderTests(unittest.TestCase):
    def test_plan_accepts_only_same_node_borrower_before_its_owner(self):
        for node in ('h100-5', 'h100-10'):
            with self.subTest(node=node), cohort(node) as fixture:
                self.assertEqual(terminal.validate_plan(fixture.plan)['node'], node)
                fixture.plan['ordered_units'][0]['borrows_units'] = [fixture.rows[0]['unit']]
                with self.assertRaisesRegex(AssertionError, 'borrowers before owners'):
                    terminal.validate_plan(fixture.plan)
        with cohort() as fixture:
            relay = dict(fixture.rows[0], unit='sst-q7b-cpu-relay-' + terminal.RUN_ID[:12] + '.service',
                         pid=9000, birth='222', borrows_units=[fixture.source['unit']])
            fixture.rows[0]['borrows_units'] = [relay['unit']]
            fixture.rows.insert(0, relay)
            with self.assertRaisesRegex(AssertionError, 'borrowers before owners'):
                terminal.validate_plan(fixture.plan)

    def test_unknown_dependency_foreign_node_foreign_unit_and_duplicate_pid_reject(self):
        mutations = [lambda p: p['ordered_units'][0].update(borrows_units=['unknown.service']),
            lambda p: p['ordered_units'][0].update(node='h100-5'),
            lambda p: p['ordered_units'][0].update(unit='foreign-slurm.service'),
            lambda p: p['ordered_units'][0].update(pid=p['ordered_units'][-1]['pid'])]
        for ordinal, mutate in enumerate(mutations):
            with self.subTest(case=ordinal), cohort() as fixture:
                mutate(fixture.plan)
                with self.assertRaises(AssertionError):
                    terminal.validate_plan(fixture.plan)
                self.assertEqual(fixture.stops, [])

    def test_fixed_source_birth_cannot_disagree_with_captured_last_row(self):
        for node in ('h100-5', 'h100-10'):
            with self.subTest(node=node), cohort(node) as fixture:
                fixture.rows[-1]['birth'] = '999999'
                with self.assertRaises(AssertionError):
                    terminal.validate_plan(fixture.plan)

    def test_metadata_scope_and_optional_allocation_binding_are_not_ignored(self):
        for kind in ('foreign-root', 'traversal', 'sampling', 'unpaired', 'foreign-allocation'):
            with self.subTest(kind=kind), cohort() as fixture:
                change = {'foreign-root': dict(metadata_roots=['/tmp/foreign']),
                    'traversal': dict(metadata_roots=[str(fixture.base / '../foreign')]),
                    'sampling': dict(metadata_roots=[str(fixture.base / 'sampling/raw')]),
                    'unpaired': dict(allocation_sha256=''),
                    'foreign-allocation': dict(allocation_path='/tmp/foreign/allocation.json')}[kind]
                fixture.rows[0].update(change)
                with self.assertRaises(AssertionError):
                    terminal.validate_plan(fixture.plan)
        with cohort() as fixture:
            fixture.rows[0]['allocation_sha256'] = '0' * 64
            with self.assertRaises(AssertionError):
                fixture.invoke(True)
            self.assertEqual(fixture.stops, [])
        for field, wrong in (('controller_pid', 999), ('native_unit', 'foreign.service'),
                             ('node', 'h100-5'), ('run_id', 'another-run')):
            with self.subTest(field=field), cohort() as fixture:
                path = Path(fixture.rows[0]['allocation_path'])
                value = read(path);value[field] = wrong;write(path, value)
                fixture.rows[0]['allocation_sha256'] = digest(path)
                with self.assertRaises(AssertionError):
                    fixture.invoke(True)
                self.assertEqual(fixture.stops, [])

    def test_readonly_plan_creates_no_stop_intent(self):
        with cohort() as fixture:
            value = fixture.invoke()
            self.assertEqual(value['status'], 'READ_ONLY_COHORT_PLAN_VERIFIED_NO_STOP_INTENT')
            self.assertFalse((fixture.root / 'stop-intent.json').exists())
            self.assertEqual(fixture.stops, [])

    def test_host_mismatch_never_queries_or_stops_same_numeric_remote_pid(self):
        with cohort('h100-5') as fixture, patch('socket.gethostname', return_value='h100-10'):
            with self.assertRaises(AssertionError):
                fixture.invoke(True)
            self.assertEqual(fixture.calls, [])
            self.assertEqual(fixture.stops, [])

    def test_reused_birth_after_capture_is_not_signalled(self):
        with cohort() as fixture:
            publish = terminal.publish
            def changed_after_intent(path, value):
                publish(path, value)
                if path.name == 'stop-intent.json':
                    fixture.own[fixture.borrower][str(fixture.rows[0]['pid'])] = 'reused-foreign-birth'
            with patch.object(terminal, 'publish', side_effect=changed_after_intent):
                with self.assertRaises(AssertionError):
                    fixture.invoke(True)
            self.assertEqual(fixture.stops, [])
            self.assertTrue(fixture.active[fixture.source['unit']])
            self.assertTrue((fixture.root / 'stop-intent.json').is_file())

    def test_failed_or_lingering_borrower_cannot_cut_owner_relay(self):
        for options in (dict(failed_stop=True), dict(lingering=True)):
            with self.subTest(options=options), cohort(**options) as fixture:
                with self.assertRaises(AssertionError):
                    fixture.invoke(True)
                self.assertEqual(fixture.stops, [fixture.borrower])
                self.assertTrue(fixture.active[fixture.source['unit']])
                self.assertTrue((fixture.root / 'stop-receipt-001.json').is_file())
                self.assertFalse((fixture.root / 'complete.json').exists())

    def test_node5_remote_proof_is_not_a_local_pid_inventory(self):
        with cohort('h100-5') as fixture:
            fixture.invoke(True)
            local_births = {pid for operation, pid in fixture.calls if operation == 'birth'}
            self.assertNotIn(terminal.CONSUMER['pid'], local_births)
            proof = read(fixture.root / 'complete.json')
            self.assertNotIn(str(terminal.CONSUMER['pid']), proof['original_owned_births'])
            self.assertEqual(proof['remote_node10_proof_sha256'], digest(fixture.remote))

    def test_node5_rejects_remote_proof_for_a_different_consumer(self):
        with cohort('h100-5') as fixture:
            wrong = read(fixture.remote);wrong['source_pid'] = 999;write(fixture.remote, wrong)
            fixture.plan['remote_node10_proof_sha256'] = digest(fixture.remote)
            with self.assertRaises(AssertionError):
                fixture.invoke(True)
            self.assertEqual(fixture.stops, [])


class OriginalHashTests(unittest.TestCase):
    def test_hash_pins_original_regular_inode_and_never_rewrites_it(self):
        with cohort() as fixture:
            path = Path(fixture.rows[0]['metadata_roots'][0]) / 'sampler.log'
            before = path.stat();value = terminal.terminal_hash(path);after = path.stat()
            self.assertEqual((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))
            self.assertEqual(value['sha256'], digest(path))
            self.assertEqual(value['path'], str(fixture.physical / path.relative_to(fixture.base)))
            self.assertEqual((value['device'], value['inode']), (before.st_dev, before.st_ino))

    def test_symlink_or_inode_replacement_during_read_is_rejected(self):
        with cohort() as fixture:
            path = Path(fixture.rows[0]['metadata_roots'][0]) / 'sampler.log'
            link = path.with_name('link.log');link.symlink_to(path)
            with self.assertRaises(OSError):
                terminal.terminal_hash(link)
            real_sha = hashlib.sha256
            class ReplacingDigest:
                def __init__(self):self.digest = real_sha();self.changed = False
                def update(self, block):
                    self.digest.update(block)
                    if not self.changed:
                        replacement = path.with_name('replacement.log');replacement.write_bytes(b'other original inode')
                        os.replace(replacement, path);self.changed = True
                def hexdigest(self):return self.digest.hexdigest()
            with patch.object(terminal.hashlib, 'sha256', side_effect=ReplacingDigest):
                with self.assertRaises(AssertionError):
                    terminal.terminal_hash(path)

    def test_metadata_inventory_excludes_raw_outputs_and_symlinks(self):
        with cohort() as fixture:
            root = Path(fixture.rows[0]['metadata_roots'][0])
            for directory in ('sampling', 'features', 'states', 'retrieval-cache'):
                write(root / directory / 'large.json', {'must_not_read': True})
            (root / 'foreign.log').symlink_to(root / 'sampler.log')
            paths = terminal.metadata_paths(fixture.rows)
            self.assertNotIn(root / 'foreign.log', paths)
            self.assertTrue(all(not {'sampling', 'features', 'states', 'retrieval-cache'} & set(p.relative_to(root).parts)
                                for p in paths if p.is_relative_to(root)))

    def test_immutable_publish_never_overwrites_final_or_leaves_partial_final(self):
        with cohort() as fixture:
            path = fixture.root / 'published.json'
            terminal.publish(path, {'original': True})
            before = path.stat();raw = path.read_bytes()
            with self.assertRaises(FileExistsError):
                terminal.publish(path, {'replace_original': True})
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(path.stat().st_ino, before.st_ino)
            self.assertEqual(list(fixture.root.glob('*.tmp-*')), [])
            interrupted = fixture.root / 'interrupted.json'
            with patch.object(terminal.os, 'link', side_effect=OSError('simulated link interruption')):
                with self.assertRaises(OSError):
                    terminal.publish(interrupted, {'not_published': True})
            self.assertFalse(interrupted.exists())
            self.assertEqual(list(fixture.root.glob('*.tmp-*')), [])


class ProofConsumerTests(unittest.TestCase):
    def test_capture_to_terminal_fields_match_both_real_new_owner_consumers(self):
        for node in ('h100-5', 'h100-10'):
            with self.subTest(node=node), cohort(node) as fixture:
                fixture.invoke(True)
                path = fixture.root / 'complete.json';proof = read(path)
                self.assertEqual(fixture.stops, [row['unit'] for row in fixture.rows])
                self.assertEqual(proof['source_birth'], proof['units'][-1]['birth'])
                self.assertEqual(proof['source_pid'], proof['units'][-1]['pid'])
                self.assertEqual(proof['source_unit'], proof['units'][-1]['unit'])
                self.assertEqual(proof['units'][-1]['observed']['MainPID'], str(fixture.source['pid']))
                self.assertEqual(set(proof['original_owned_births']), {pid for births in fixture.own.values() for pid in births})
                self.assertNotIn('999999', proof['original_owned_births'])
                self.assertEqual(len(proof['release_observations']), 2)
                self.assertGreaterEqual(proof['release_observations'][1]['epoch'] - proof['release_observations'][0]['epoch'], 3)
                self.assertTrue(all(row['observed']['MainPID'] == '0'
                    for observation in proof['release_observations'] for row in observation['units']))
                self.assertTrue(proof['original_partial_outputs_retained'])
                self.assertFalse(proof['foreign_signalled']);self.assertFalse(proof['global_goal_complete'])
                if node == 'h100-5':
                    tree = ast.parse(control.REMOTE)
                    branch = next(item for item in ast.walk(tree) if isinstance(item, ast.If)
                        and isinstance(item.test, ast.Compare) and isinstance(item.test.left, ast.Name)
                        and item.test.left.id == 'ROLE' and isinstance(item.test.comparators[0], ast.Constant)
                        and item.test.comparators[0].value == 'owner'
                        and any(isinstance(name, ast.Name) and name.id == 'PREDECESSOR' for name in ast.walk(item)))
                    main_tree = ast.parse(Path(control.__file__).read_text())
                    main = next(item for item in main_tree.body if isinstance(item, ast.FunctionDef) and item.name == 'main')
                    assembly = next(item for item in ast.walk(main) if isinstance(item, ast.Assign)
                        and any(isinstance(name, ast.Name) and name.id == 'script' for name in item.targets))
                    preamble = eval(compile(ast.Expression(assembly.value), '<actual-controller-preamble>', 'eval'),
                                    dict(values={}, REMOTE=''))
                    source = ast.fix_missing_locations(ast.Module(body=ast.parse(preamble).body + branch.body, type_ignores=[]))
                    namespace = dict(PREDECESSOR=str(path), PREDECESSOR_SHA=digest(path), RUN_ID=terminal.RUN_ID,
                        OLD_SOURCE_UNIT=terminal.SOURCE['unit'], read=read, sha=digest,
                        birth=lambda pid: None, query=lambda unit: dict(returncode=0, properties=fixture_query(unit)),
                        subprocess=SimpleNamespace(check_output=lambda *a, **k: '999999\n'),
                        base64=base64, FILES={'native_paths.sh': base64.b64encode(b'cpu-only').decode()}, stage=fixture.root)
                    exec(compile(source, '<actual-owner-predecessor-branch>', 'exec'), namespace)
                    self.assertEqual(namespace['context']['predecessor_sha256'], digest(path))
                else:
                    owner = importlib.import_module('.retrieval_relay_owner', __package__)
                    pool = SimpleNamespace(read=read, sha=digest, RUN_ID=terminal.RUN_ID)
                    args = SimpleNamespace(predecessor_proof=str(path), predecessor_sha256=digest(path))
                    with patch.object(owner.source, 'process', side_effect=FileNotFoundError), \
                         patch.object(owner.subprocess, 'check_output', side_effect=lambda argv, **kwargs:
                             '999999\n' if argv[0] == 'nvidia-smi' else
                             'Id=%s\nMainPID=0\nActiveState=inactive\n' % argv[2]):
                        value = owner.verify_predecessor(pool, args)
                    self.assertEqual(value['sha256'], digest(path))

    def test_existing_intent_or_complete_never_replays_a_stop(self):
        with cohort() as fixture:
            write(fixture.root / 'stop-intent.json', {'actual_original_intent': True})
            value = fixture.invoke(True)
            self.assertEqual(value['status'], 'EXISTING_INTENT_OBSERVE_ONLY_NO_STOP_REPLAY')
            self.assertEqual(fixture.stops, [])
        with cohort() as fixture:
            fixture.invoke(True);fixture.stops.clear()
            value = fixture.invoke(True)
            self.assertTrue(value['existing_proof_read_no_replay'])
            self.assertEqual(fixture.stops, [])


def fixture_query(unit):
    return dict(MainPID='0', ActiveState='inactive', Id=unit)


if __name__ == '__main__':
    unittest.main()
