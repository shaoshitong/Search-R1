"""CPU-only coverage for replacing a released TP4 sublease with TP1 models."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parent
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
PARENT_UNIT = 'sst-q7b-g5-sharedvram-v6-' + RUN_ID[:12] + '.service'
RETIRED_UNIT = 'sst-q7b-g5-source5-extra-model-v1-' + RUN_ID[:12] + '.service'
TERMINAL_STATUS = 'OWN_SOURCE5_MODEL_TP4_CONTROLLED_THROUGHPUT_STOP_FULL_SHA_RELEASE_PASS'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_worker():
    """Load only the admission module, with no model/retrieval import chain."""
    consumer = ModuleType('relay_consumer_worker')
    consumer.shared = SimpleNamespace(require=require)
    spec = importlib.util.spec_from_file_location('replacement_sublease_under_test',
                                                DIRECTORY / 'source_sublease_worker.py')
    worker = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, relay_consumer_worker=consumer):
        spec.loader.exec_module(worker)
    return worker


class FakePool:
    RUN_ID = RUN_ID

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@contextlib.contextmanager
def retired_fixture():
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        allocation_path = base / 'pool/jobs/job-source5-model-2021698/attempt-00/allocation.json'
        parent = dict(pid=1490214, birth='6807965', unit=PARENT_UNIT,
                      cgroup='0::/system.slice/' + PARENT_UNIT + '\n',
                      netns_inode=901, mntns_inode=902,
                      allocation_path=str(base / 'parent-allocation.json'),
                      allocation_sha256='a' * 64)
        model = [3, 4, 6, 7]
        lease_directory = base / 'pool/gpu-subleases' / RUN_ID / '1490214-6807965'
        leases = {}
        for index in model:
            path = lease_directory / ('GPU-%d.lock' % index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'retained retired child lease')
            leases['GPU-%d' % index] = str(path)
        retired_context_path = base / 'preparation/retired/sublease-context.json'
        retired_context = dict(run_id=RUN_ID, parent=copy.deepcopy(parent), model_indices=model,
                               model_cuda_uuids=['GPU-%d' % i for i in model])
        write(retired_context_path, retired_context)
        allocation = dict(run_id=RUN_ID, node='h100-5', controller_pid=2021698,
                          native_unit=RETIRED_UNIT, native_cgroup='/system.slice/' + RETIRED_UNIT,
                          tensor_parallel_size=4,
                          model_indices=model, model_cuda_uuids=['GPU-%d' % i for i in model],
                          source_parent=copy.deepcopy(parent), gpu_sublease_paths=leases,
                          source_context=str(retired_context_path),
                          source_context_sha256=FakePool.sha(retired_context_path))
        write(allocation_path, allocation)
        proof_path = base / 'preparation/retired/complete.json'
        proof = dict(status=TERMINAL_STATUS, run_id=RUN_ID,
                     retired=dict(unit=RETIRED_UNIT, pid=2021698, birth='7430669',
                                  allocation_path=str(allocation_path),
                                  allocation_sha256=FakePool.sha(allocation_path)),
                     original_files=[dict(path=str(allocation_path), sha256=FakePool.sha(allocation_path))],
                     release_observations=[dict(epoch=epoch, unit_main_pid=0,
                         original_births_gone=True, own_cgroup_gone=True,
                         own_gpu_contexts=[], foreign_signalled=False) for epoch in (100, 110)],
                     source_parent={key: parent[key] for key in ('pid', 'birth', 'unit')},
                     source_parent_and_retriever_preserved=True,
                     stopped_after_at_least_own_batches=1, original_partial_outputs_retained=True,
                     global_goal_complete=False, original_owned_births={'2021698': '7430669'})
        value = dict(schema_version=3, tensor_parallel_size=1,
                     mode='SOURCE5_SAME_RUN_GPU_SUBLEASE', run_id=RUN_ID, base=str(base),
                     model_limits=dict(seqs=32, tokens=8192, blocks=4096, reserve=1.0),
                     model_deadline_epoch=5000, source_sampling_deadline_epoch=9000,
                     model_indices=[4], model_cuda_uuids=['GPU-4'], parent=parent,
                     retriever=dict(pid=1490465, cgroup=parent['cgroup'], netns_inode=901),
                     retired_terminal=dict(path=str(proof_path), sha256=''))

        def seal():
            write(retired_context_path, retired_context)
            allocation['source_context_sha256'] = FakePool.sha(retired_context_path)
            write(allocation_path, allocation)
            digest = FakePool.sha(allocation_path)
            proof['retired']['allocation_sha256'] = digest
            if proof['original_files']:
                proof['original_files'][0]['sha256'] = digest
            write(proof_path, proof)
            value['retired_terminal']['sha256'] = FakePool.sha(proof_path)

        seal()
        yield SimpleNamespace(base=base, pool=FakePool(), parent=parent,
                              allocation=allocation, allocation_path=allocation_path,
                              retired_context=retired_context, retired_context_path=retired_context_path,
                              proof=proof, proof_path=proof_path, value=value, seal=seal)


class RetirementBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_worker()

    def validate(self, fixture, model=None, uuids=None):
        return self.worker.validate_retired_terminal(fixture.pool, fixture.value,
            fixture.parent, [4] if model is None else model, ['GPU-4'] if uuids is None else uuids)

    def test_accepts_each_selected_physical_gpu_with_same_original_parent(self):
        for index in (4, 6, 7):
            with self.subTest(index=index), retired_fixture() as fixture:
                binding = self.validate(fixture, [index], ['GPU-%d' % index])
                self.assertIsInstance(binding, dict)
                self.assertEqual(FakePool.read(fixture.allocation_path), fixture.allocation)

    def test_rejects_unsealed_or_changed_terminal_proof(self):
        with retired_fixture() as fixture:
            fixture.value['retired_terminal']['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                self.validate(fixture)

    def test_rejects_changed_retired_allocation(self):
        with retired_fixture() as fixture:
            fixture.allocation['controller_pid'] = 999
            write(fixture.allocation_path, fixture.allocation)
            with self.assertRaises(ValueError):
                self.validate(fixture)

    def test_legacy_allocation_without_top_run_is_bound_by_hashed_source_context(self):
        with retired_fixture() as fixture:
            del fixture.allocation['run_id']
            fixture.seal()
            self.assertIsInstance(self.validate(fixture), dict)
        with retired_fixture() as fixture:
            fixture.allocation['run_id'] = 'other-run'
            fixture.seal()
            with self.assertRaises(ValueError):
                self.validate(fixture)

    def test_rejects_changed_or_wrong_run_retired_source_context(self):
        with retired_fixture() as fixture:
            fixture.retired_context['run_id'] = 'other-run'
            write(fixture.retired_context_path, fixture.retired_context)
            with self.assertRaisesRegex(ValueError, 'context SHA'):
                self.validate(fixture)
        with retired_fixture() as fixture:
            fixture.retired_context['run_id'] = 'other-run'
            fixture.seal()
            with self.assertRaises(ValueError):
                self.validate(fixture)

    def test_rejects_wrong_retired_process_run_or_unit_even_when_resealed(self):
        changes = [('run_id', 'other-run'), ('status', 'STOP_REQUESTED_ONLY')]
        for key, wrong in changes:
            with self.subTest(key=key), retired_fixture() as fixture:
                fixture.proof[key] = wrong
                fixture.seal()
                with self.assertRaises(ValueError):
                    self.validate(fixture)
        for key, wrong in [('pid', 999), ('birth', 'new-birth'), ('unit', 'other.service')]:
            with self.subTest(retired_key=key), retired_fixture() as fixture:
                fixture.proof['retired'][key] = wrong
                fixture.seal()
                with self.assertRaises(ValueError):
                    self.validate(fixture)

    def test_rejects_incomplete_or_unsafe_release_observations(self):
        mutations = [lambda proof: proof.update(release_observations=proof['release_observations'][:1]),
                     lambda proof: proof['release_observations'][1].update(unit_main_pid=2021698),
                     lambda proof: proof['release_observations'][1].update(original_births_gone=False),
                     lambda proof: proof['release_observations'][1].update(own_cgroup_gone=False),
                     lambda proof: proof['release_observations'][1].update(own_gpu_contexts=[2021698]),
                     lambda proof: proof['release_observations'][1].update(foreign_signalled=True),
                     lambda proof: proof.update(source_parent_and_retriever_preserved=False),
                     lambda proof: proof.update(original_partial_outputs_retained=False),
                     lambda proof: proof.update(stopped_after_at_least_own_batches=0),
                     lambda proof: proof.update(global_goal_complete=True)]
        for ordinal, mutate in enumerate(mutations):
            with self.subTest(case=ordinal), retired_fixture() as fixture:
                mutate(fixture.proof)
                fixture.seal()
                with self.assertRaises(ValueError):
                    self.validate(fixture)

    def test_rejects_original_parent_or_selected_uuid_mismatch(self):
        with retired_fixture() as fixture:
            fixture.allocation['source_parent']['birth'] = 'different-birth'
            fixture.seal()
            with self.assertRaises(ValueError):
                self.validate(fixture)
        with retired_fixture() as fixture:
            with self.assertRaises(ValueError):
                self.validate(fixture, [4], ['GPU-6'])
        with retired_fixture() as fixture:
            with self.assertRaises(ValueError):
                self.validate(fixture, [2], ['GPU-2'])

    def test_rejects_missing_uuid_or_nested_lease_instead_of_retired_root_child(self):
        with retired_fixture() as fixture:
            with self.assertRaises(ValueError):
                self.validate(fixture, [4], [])
        with retired_fixture() as fixture:
            fixture.allocation['gpu_sublease_paths']['GPU-4'] = str(
                fixture.base / 'pool/gpu-subleases' / RUN_ID /
                '1490214-6807965/co-share-with-other/GPU-4.lock')
            fixture.seal()
            with self.assertRaises(ValueError):
                self.validate(fixture)

    def test_physical_nas_references_resolve_only_within_same_data_root(self):
        with retired_fixture() as fixture:
            physical = Path('/mnt/data/shaoshitong') / fixture.base.name
            original = fixture.proof_path
            fixture.value['retired_terminal']['path'] = str(physical / original.relative_to(fixture.base))
            binding = self.validate(fixture)
            self.assertEqual(binding['path'], str(original))
            fixture.value['retired_terminal']['path'] = str(fixture.base.parent / 'other-data/complete.json')
            with self.assertRaises(ValueError):
                self.validate(fixture)
            fixture.value['retired_terminal']['path'] = str(fixture.base / '../other-data/complete.json')
            with self.assertRaises(ValueError):
                self.validate(fixture)

    def test_rejects_missing_allocation_membership_or_source_in_owned_births(self):
        with retired_fixture() as fixture:
            fixture.proof['original_files'] = []
            fixture.seal()
            with self.assertRaises(ValueError):
                self.validate(fixture)
        for pid in (1490214, 1490465, 2013773):
            with self.subTest(pid=pid), retired_fixture() as fixture:
                fixture.proof['original_owned_births'][str(pid)] = 'retained-source'
                fixture.seal()
                with self.assertRaises(ValueError):
                    self.validate(fixture)

    def test_schema3_context_is_tp1_and_has_no_peer_dependency(self):
        with retired_fixture() as fixture:
            context_path = fixture.base / 'context.json'
            write(context_path, fixture.value)
            args = argparse.Namespace(reuse_local_relay_ready=str(context_path),
                                      reuse_local_relay_sha256=FakePool.sha(context_path),
                                      reuse_local_relay_unit=PARENT_UNIT, tp=1, base=str(fixture.base),
                                      model_max_seqs=32, model_batch_tokens=8192,
                                      model_kv_block_cap=4096, model_reserve_gib=1.0)
            with patch.dict(os.environ, Q7B_SOURCE_SUBLEASE_CONTEXT=str(context_path)):
                path, value = self.worker.context(fixture.pool, args)
            self.assertEqual(path, context_path)
            self.assertEqual(value['schema_version'], 3)
            self.assertNotIn('peer_sublease', value)
            args.tp = 4
            with patch.dict(os.environ, Q7B_SOURCE_SUBLEASE_CONTEXT=str(context_path)):
                with self.assertRaises(ValueError):
                    self.worker.context(fixture.pool, args)

    def test_schema3_rejects_peer_dependency_and_insufficient_source_cutoff(self):
        for change in (dict(peer_sublease={}), dict(model_deadline_epoch=6000)):
            with self.subTest(change=change), retired_fixture() as fixture:
                fixture.value.update(change)
                context_path = fixture.base / 'context.json'
                write(context_path, fixture.value)
                args = argparse.Namespace(reuse_local_relay_ready=str(context_path),
                    reuse_local_relay_sha256=FakePool.sha(context_path),
                    reuse_local_relay_unit=PARENT_UNIT, tp=1, base=str(fixture.base),
                    model_max_seqs=32, model_batch_tokens=8192,
                    model_kv_block_cap=4096, model_reserve_gib=1.0)
                with patch.dict(os.environ, Q7B_SOURCE_SUBLEASE_CONTEXT=str(context_path)):
                    with self.assertRaises(ValueError):
                        self.worker.context(fixture.pool, args)

    def test_schema3_rejects_wrong_capacity_or_data_root(self):
        for case in ('context-capacity', 'argv-capacity', 'data-root'):
            with self.subTest(case=case), retired_fixture() as fixture:
                if case == 'context-capacity':
                    fixture.value['model_limits']['seqs'] = 64
                context_path = fixture.base / 'context.json'
                write(context_path, fixture.value)
                args = argparse.Namespace(reuse_local_relay_ready=str(context_path),
                    reuse_local_relay_sha256=FakePool.sha(context_path),
                    reuse_local_relay_unit=PARENT_UNIT, tp=1, base=str(fixture.base),
                    model_max_seqs=32, model_batch_tokens=8192,
                    model_kv_block_cap=4096, model_reserve_gib=1.0)
                if case == 'argv-capacity':
                    args.model_kv_block_cap = 8192
                elif case == 'data-root':
                    args.base = str(fixture.base / 'another-root')
                with patch.dict(os.environ, Q7B_SOURCE_SUBLEASE_CONTEXT=str(context_path)):
                    with self.assertRaises(ValueError):
                        self.worker.context(fixture.pool, args)


class ReplacementBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = importlib.import_module('.source_tp1_replacement_control', __package__)

    @contextlib.contextmanager
    def bootstrap_fixture(self, group, model_deadline=5000, source_deadline=8600,
                          now=1000, maximum=14400):
        control = self.control
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            stage = base / 'preparation/replacement/code'
            stage.mkdir(parents=True)
            model = control.GROUPS[group]
            unit = 'sst-q7b-g5-source5-tp1-replacement-' + group + '-v1-' + RUN_ID[:12] + '.service'
            cgroup = '/system.slice/' + unit
            context_path = stage.parent / 'sublease-context.json'
            write(context_path, dict(schema_version=3, run_id=RUN_ID, model_indices=model,
                parent=dict(mntns_inode=20, netns_inode=30, unit=PARENT_UNIT),
                model_deadline_epoch=model_deadline, source_sampling_deadline_epoch=source_deadline))
            write(stage / 'manifest.json', dict(run_id=RUN_ID, files={}))
            detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (unit, cgroup)
            real_stat, real_read = os.stat, Path.read_text

            def stat(path, *args, **kwargs):
                if str(path) == '/proc/self/ns/mnt':
                    return SimpleNamespace(st_ino=20)
                if str(path) == '/proc/self/ns/net':
                    return SimpleNamespace(st_ino=30)
                return real_stat(path, *args, **kwargs)

            def read(path, *args, **kwargs):
                if str(path) == '/proc/self/cgroup':
                    return '0::' + cgroup + '\n'
                return real_read(path, *args, **kwargs)

            with patch.object(control, 'BASE', str(base)):
                source = control.bootstrap(unit, group, maximum)
            namespace = dict(__file__=str(stage / 'source_sublease_bootstrap.py'))
            try:
                with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                     patch('socket.gethostname', return_value='h100-5'), patch('os.stat', side_effect=stat), \
                     patch.object(Path, 'read_text', read), patch('time.time', return_value=now), \
                     patch('subprocess.check_output', return_value=detail) as query, patch('os.dup2'), \
                     patch('os.chdir'), patch('builtins.print'), patch('os.execvpe') as execute:
                    yield SimpleNamespace(source=source, namespace=namespace, execute=execute,
                        query=query, model=model, context_path=context_path, base=base)
            finally:
                if namespace.get('log'):
                    namespace['log'].close()

    def test_bundle_binds_distinct_single_gpu_units_and_immutable_retirement_proof(self):
        control = self.control
        self.assertEqual(control.GROUPS, {'gpu4': [4], 'gpu6': [6], 'gpu7': [7]})
        self.assertEqual(control.SOURCE_CUTOFF_SECONDS, 3600)
        terminal = control.TERMINAL_PROOF
        identities, units, directories = set(), set(), set()
        for group, model in control.GROUPS.items():
            with self.subTest(group=group):
                identity, files, manifest = control.bundle(group, terminal, 'f' * 64)
                identities.add(identity)
                units.add(manifest['unit'])
                directories.add(manifest['directory'])
                self.assertEqual(manifest['run_id'], RUN_ID)
                self.assertEqual(manifest['model_gpus'], model)
                self.assertEqual(manifest['tensor_parallel_size'], 1)
                self.assertIn(group, manifest['unit'])
                self.assertIn('-v1-', manifest['unit'])
                self.assertIn(group, manifest['directory'])
                self.assertIn('-v1', manifest['directory'])
                self.assertEqual(manifest['retired_terminal'], dict(path=terminal, sha256='f' * 64))
                self.assertEqual(manifest['context_schema_version'], 3)
                self.assertEqual(manifest['source_deadline_reserve_seconds'], 3600)
                self.assertEqual(manifest['maximum_run_seconds'], 14400)
                self.assertEqual(manifest['model_limits'], dict(seqs=32, tokens=8192, blocks=4096, reserve=1.0))
                self.assertEqual(set(files), set(manifest['files']))
                for name, data in files.items():
                    self.assertEqual(manifest['files'][name],
                                     dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                changed_identity, _, _ = control.bundle(group, terminal, 'e' * 64)
                self.assertNotEqual(identity, changed_identity)
        self.assertEqual(len(identities), 3)
        self.assertEqual(len(units), 3)
        self.assertEqual(len(directories), 3)

    def test_bundle_rejects_unapproved_gpu_proof_or_runtime(self):
        control = self.control
        for group, terminal, digest, seconds in (
                ('gpu3', control.TERMINAL_PROOF, 'f' * 64, 14400),
                ('gpu4', '/mnt/data/another-proof.json', 'f' * 64, 14400),
                ('gpu4', control.TERMINAL_PROOF, 'not-a-sha', 14400),
                ('gpu4', control.TERMINAL_PROOF, 'f' * 64, 14401),
                ('gpu4', control.TERMINAL_PROOF, 'f' * 64, 1260)):
            with self.subTest(group=group, terminal=terminal, seconds=seconds):
                with self.assertRaises(ValueError):
                    control.bundle(group, terminal, digest, seconds)

    def test_bootstrap_passes_tp1_limits_and_historical_import_profile(self):
        for group in self.control.GROUPS:
            with self.subTest(group=group), self.bootstrap_fixture(group) as fixture:
                exec(compile(fixture.source, fixture.namespace['__file__'], 'exec'), fixture.namespace)
                executable, argv, env = fixture.execute.call_args.args
                option = lambda flag: argv[argv.index(flag) + 1]
                self.assertTrue(executable.endswith('/env/bin/python'))
                self.assertEqual(option('--tp'), '1')
                self.assertEqual(option('--model-gpus'), ','.join(map(str, fixture.model)))
                self.assertEqual(option('--model-max-seqs'), '32')
                self.assertEqual(option('--model-batch-tokens'), '8192')
                self.assertEqual(option('--model-kv-block-cap'), '4096')
                self.assertEqual(float(option('--model-reserve-gib')), 1.0)
                self.assertEqual(option('--run-seconds'), '4000')
                self.assertEqual(option('--batch-reserve-seconds'), '1200')
                self.assertEqual(option('--reuse-local-relay-unit'), PARENT_UNIT)
                self.assertEqual(option('--reuse-local-relay-sha256'), FakePool.sha(fixture.context_path))
                self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'], '4')
                self.assertEqual(env['SEARCH_G1_B_PREFIX_TOKENS'], '20')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], option('--model-gpus'))
                self.assertEqual(Path(env['Q7B_SOURCE_SUBLEASE_CONTEXT']).resolve(), fixture.context_path.resolve())
                self.assertTrue(Path(env['HF_HOME']).is_relative_to(fixture.base))
                self.assertTrue(Path(env['TORCH_EXTENSIONS_DIR']).is_relative_to(fixture.base))

    def test_bootstrap_obeys_four_hour_maximum_and_rejects_missing_source_reserve(self):
        with self.bootstrap_fixture('gpu4', model_deadline=25000, source_deadline=28600) as fixture:
            exec(compile(fixture.source, fixture.namespace['__file__'], 'exec'), fixture.namespace)
            argv = fixture.execute.call_args.args[1]
            self.assertEqual(argv[argv.index('--run-seconds') + 1], '14400')
        for deadline, source_deadline in ((5000, 8500), (2200, 9000)):
            with self.subTest(deadline=deadline), self.bootstrap_fixture(
                    'gpu4', model_deadline=deadline, source_deadline=source_deadline) as fixture:
                with self.assertRaises(AssertionError):
                    exec(compile(fixture.source, fixture.namespace['__file__'], 'exec'), fixture.namespace)
                fixture.execute.assert_not_called()
                fixture.query.assert_not_called()

    def test_consumer_deadline_preserves_absolute_cap_after_delayed_admission(self):
        """Evaluate the real deadline expression without loading model modules."""
        tree = ast.parse((DIRECTORY / 'relay_consumer_worker.py').read_text())
        controller = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'controller')
        assignment = next(node for node in ast.walk(controller) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'deadline'
                                  for target in node.targets))
        expression = compile(ast.fix_missing_locations(ast.Expression(assignment.value)),
                             '<consumer-deadline>', 'eval')
        namespace = dict(time=SimpleNamespace(time=lambda: 5000),
                         args=SimpleNamespace(run_seconds=4000), end=10000,
                         binding=dict(source_model_deadline_epoch=6000))
        self.assertEqual(eval(expression, namespace), 6000)
        namespace['binding'] = {}
        self.assertEqual(eval(expression, namespace), 9000,
                         'Schemas1/2 retain their prior relative runtime behavior')
        namespace['end'] = 7000
        self.assertEqual(eval(expression, namespace), 6955)


if __name__ == '__main__':
    unittest.main()
