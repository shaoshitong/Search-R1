"""Metadata-only native Ray-driver routing fixtures; no Ray or CUDA execution."""
import ast
import copy
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from search_g1 import qprobe_driver_gpu as driver
from reproduction.common8.native_allocation import LOCAL_BINDING_RECEIPT, LOCAL_PLACEMENT_POLICY


class CpuRngBytes:
    """Explicit metadata-only stand-in for Torch's CPU RNG byte tensor."""
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return np.frombuffer(self.value, dtype=np.uint8)


def fixture():
    training = [1, 3, 0, 2]
    auxiliary = [5, 6]
    inventory = [[str(i), 'GPU-fixture-'+str(i), '81920', '0', '0'] for i in range(8)]
    ids = {str(i): 'GPU-fixture-'+str(i) for i in training+auxiliary}
    run = dict(run_id='a'*32, execution_id='b'*32, source_commit='c'*40,
        execution_backend='ssh_local', execution_host='test-a800', slurm_job_id=None, profile='common8',
        training_world_size=4, actor_micro_batch_per_rank=4, gpu_placement_policy=LOCAL_PLACEMENT_POLICY,
        gpu_placement_bound=True, gpu_placement_bound_unix=123., gpu_placement_binding_receipt=LOCAL_BINDING_RECEIPT,
        gpu_placement_binding_sha256='d'*64, training_physical_indices=training,
        auxiliary_physical_indices=auxiliary, retriever_physical_indices=auxiliary,
        probe_physical_index=6, excluded_physical_indices=[4, 7], physical_gpu_inventory=inventory,
        assigned_gpu_uuids=ids, training_gpu_uuids=[ids[str(i)] for i in training],
        pipeline_identity=dict(pid=123, start_ticks=456, uid=os.getuid(), sid=123),
        question_probe=dict(head_kind=driver.HEAD_KIND, head_device='cuda:0'))
    fields = ('run_id', 'execution_id', 'source_commit', 'training_physical_indices',
        'auxiliary_physical_indices', 'excluded_physical_indices', 'training_gpu_uuids',
        'assigned_gpu_uuids', 'physical_gpu_inventory', 'gpu_placement_bound_unix')
    placement = {key: copy.deepcopy(run[key]) for key in fields}
    placement['host'] = 'test-a800'
    allocation = dict(run_id=run['run_id'], execution_id=run['execution_id'], execution_backend='ssh_local',
        accepted=True, slurm_job_id=None, node='test-a800', training_world_size=4,
        training_physical_indices=training, auxiliary_physical_indices=auxiliary, excluded_physical_indices=[4, 7],
        training_visible_devices=list(map(str, training)), actual_cuda_runtime_uuids=run['training_gpu_uuids'],
        assigned_gpu_uuids=ids, gpu_placement_binding_sha256='d'*64, pipeline_identity=run['pipeline_identity'])
    pipeline = dict(allocation, allocation_sha256='e'*64)
    claim = dict(run_id=run['run_id'], execution_id=run['execution_id'], execution_backend='ssh_local', hostname='test-a800')
    return run, allocation, placement, pipeline, claim


class NativeDriverGpuTests(unittest.TestCase):
    def route(self, values=None, **kwargs):
        return driver.validate_driver_route(*(values or fixture()), head_device='cuda:0',
            hostname='test-a800', incoming_visible=kwargs.pop('incoming_visible', ''), **kwargs)

    def test_child_is_born_on_first_admitted_uuid_without_mutating_worker_environment_or_rng(self):
        route = self.route()
        self.assertEqual(route['physical_index'], 1)
        self.assertEqual(route['gpu_uuid'], 'GPU-fixture-1')
        self.assertEqual(route['ray_gpu_resources_reserved'], 0)
        self.assertTrue(route['shares_admitted_actor_training_gpu'])
        environment = {'CUDA_VISIBLE_DEVICES': ''}
        child = driver.child_environment(route, environment)
        self.assertEqual(environment, {'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(child, {'CUDA_VISIBLE_DEVICES': route['gpu_uuid']})
        before = random.getstate()
        calls = []
        def count():
            calls.append(('count', child['CUDA_VISIBLE_DEVICES'])); return 1
        def uuids(size):
            calls.append(('uuid', child['CUDA_VISIBLE_DEVICES'])); self.assertEqual(size, 1)
            return [route['gpu_uuid']]
        proof = driver.verify_driver_visibility(route, environ=child, birth_visible=route['gpu_uuid'],
            cuda_initialized=lambda: False, cuda_device_count=count, cuda_uuids=uuids)
        self.assertEqual(environment, {'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(calls, [('count', 'GPU-fixture-1'), ('uuid', 'GPU-fixture-1')])
        self.assertFalse(proof['rng_seed_performed'])
        self.assertFalse(proof['tensor_allocation_performed'])
        self.assertFalse(proof['visibility_changed_after_process_birth'])
        self.assertEqual(before, random.getstate())

    def test_auxiliary_visibility_wrong_uuid_order_or_unbound_launch_identity_reject(self):
        for fault in ('auxiliary', 'uuid_order', 'claim', 'placement', 'allocation', 'pipeline'):
            values = copy.deepcopy(fixture())
            incoming = ''
            if fault == 'auxiliary': incoming = '5,6'
            elif fault == 'uuid_order': values[1]['actual_cuda_runtime_uuids'] = values[1]['actual_cuda_runtime_uuids'][::-1]
            elif fault == 'claim': values[4]['execution_id'] = '9'*32
            elif fault == 'placement': values[2]['training_physical_indices'] = [0, 1, 2, 3]
            elif fault == 'allocation': values[1]['accepted'] = False
            elif fault == 'pipeline': values[3]['assigned_gpu_uuids'] = {}
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.route(values, incoming_visible=incoming)

    def test_preinitialized_cuda_or_ray_global_override_is_forbidden(self):
        route = self.route()
        for environment, initialized, born in (({'CUDA_VISIBLE_DEVICES': route['gpu_uuid']}, True, route['gpu_uuid']),
                ({'CUDA_VISIBLE_DEVICES': route['gpu_uuid'], 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1'}, False, route['gpu_uuid']),
                ({'CUDA_VISIBLE_DEVICES': route['gpu_uuid']}, False, '')):
            with self.subTest(initialized=initialized), self.assertRaises(ValueError):
                driver.verify_driver_visibility(route, environ=environment, birth_visible=born,
                    cuda_initialized=lambda: initialized,
                    cuda_device_count=lambda: self.fail('No identity query after forbidden environment'),
                    cuda_uuids=lambda _size: self.fail('No UUID query after forbidden environment'))
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], route['gpu_uuid'])

    def test_actual_cuda_mapping_must_be_one_exact_training_uuid(self):
        route = self.route()
        for count, uuids in ((0, []), (2, [route['gpu_uuid'], 'GPU-fixture-5']), (1, ['GPU-foreign'])):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'ACTUAL_CUDA_UUID_DIFFERS'):
                driver.verify_driver_visibility(route, environ={'CUDA_VISIBLE_DEVICES': route['gpu_uuid']},
                    birth_visible=route['gpu_uuid'], cuda_initialized=lambda: False,
                    cuda_device_count=lambda: count, cuda_uuids=lambda _size: uuids)

    def test_non_native_entry_has_no_file_or_cuda_side_effects(self):
        with patch.object(driver, '_json', side_effect=AssertionError('No manifest read for old heads')):
            self.assertFalse(driver.run_native_driver(dict(question_probe=dict(head_kind='count6_v1'))))

    def test_single_process_launch_uses_birth_environment_stdin_and_propagates_failure(self):
        payload = dict(schema_version=1, config={'unchanged': True}, binding={'run_id': 'a'*32},
            route=self.route(), cluster=driver._cluster_identity('127.0.0.1:1234', 'same-private-namespace'),
            parent_identity={'pid': 123, 'start_ticks': 456}, cpu_rng_sha256='f'*64)
        environment = driver.child_environment(payload['route'], {'CUDA_VISIBLE_DEVICES': '', 'RAY_ADDRESS': 'auto'})
        for code in (0, 3):
            child = Mock(pid=789, returncode=code)
            records = []
            with patch.object(driver.subprocess, 'Popen', return_value=child) as spawn, \
                    patch('reproduction.common8.native_allocation._write_local_once', side_effect=lambda p, v: records.append((p, v))):
                if code == 0:
                    driver._spawn_driver_process(Path('/data/fixture'), payload, environment)
                else:
                    with self.assertRaisesRegex(ValueError, 'FRESH_DRIVER_CHILD_EXIT_3'):
                        driver._spawn_driver_process(Path('/data/fixture'), payload, environment)
                spawn.assert_called_once()
                command, = spawn.call_args.args
                self.assertEqual(command[:4], [driver.sys.executable, '-m', 'search_g1.qprobe_driver_gpu', '--native-driver-child'])
                self.assertEqual(spawn.call_args.kwargs, dict(cwd=Path('/data/fixture/repo'), env=environment,
                    stdin=driver.subprocess.PIPE, start_new_session=False))
                raw, = child.communicate.call_args.args
                self.assertEqual(pickle.loads(raw), payload)
                self.assertEqual(records[0][0], Path('/data/fixture')/driver.PROCESS_CLAIM)
                self.assertEqual(records[1][1]['exit_code'], code)
                self.assertNotIn('RAY_ADDRESS', environment)
                self.assertEqual(records[0][1]['ray_gpu_resources_reserved'], 0)

    def test_cpu_rng_roundtrip_preserves_python_numpy_and_torch_sequences(self):
        # Test the exact handoff protocol without installing/loading Torch.
        # The mock has only CPU get/set APIs, so CUDA access fails immediately.
        cpu_backend = random.Random(123)
        backend = SimpleNamespace(
            get_rng_state=lambda: CpuRngBytes(pickle.dumps(cpu_backend.getstate(), protocol=4)),
            set_rng_state=lambda value: cpu_backend.setstate(pickle.loads(value.value)))
        with patch.dict('sys.modules', torch=backend):
            original = driver.capture_cpu_rng()
            try:
                state = pickle.loads(pickle.dumps(original, protocol=4))
                sha = driver.cpu_rng_sha256(state)
                expected = (random.random(), np.random.random(), cpu_backend.random())
                self.assertEqual(driver.restore_cpu_rng(state), sha)
                actual = (random.random(), np.random.random(), cpu_backend.random())
                self.assertEqual(actual, expected)
            finally:
                driver.restore_cpu_rng(original)

    def test_parent_death_guard_precedes_tensor_unpickling_and_checks_parent_birth(self):
        import hashlib
        raw = b'mocked owned stdin'
        parent = dict(pid=123, start_ticks=456, uid=os.getuid(), sid=123)
        events = []
        library = Mock()
        library.prctl.side_effect = lambda *args: events.append(('prctl', args)) or 0
        def unpack(_raw):
            events.append(('unpickle',)); return {'schema_version': 0}
        with patch.object(driver, '_birth', return_value=parent), patch('ctypes.CDLL', return_value=library), \
                patch.object(driver.pickle, 'loads', side_effect=unpack):
            with self.assertRaisesRegex(ValueError, 'WRAPPER_BIRTH_CHANGED'):
                driver._child(raw, hashlib.sha256(raw).hexdigest())
        self.assertEqual([item[0] for item in events], ['prctl', 'unpickle'])
        with patch.object(driver, '_birth', side_effect=[parent, dict(parent, start_ticks=999)]), \
                patch('ctypes.CDLL', return_value=library), patch.object(driver.pickle, 'loads') as unpack:
            with self.assertRaisesRegex(ValueError, 'WRAPPER_DIED_DURING_CHILD_BOOTSTRAP'):
                driver._child(raw, hashlib.sha256(raw).hexdigest())
            unpack.assert_not_called()

    def test_exact_body_relocation_keeps_original_bytes_ast_and_ray_bootstrap(self):
        # Load the actual narrow source guard without training dependencies.
        guard_path = Path(__file__).resolve().parents[1]/'reproduction/a800_common/adopt_initial_probes.py'
        guard = next(item for item in ast.parse(guard_path.read_bytes()).body
                     if isinstance(item, ast.FunctionDef) and item.name == '_outside_ray_bootstrap')
        def check(condition, message):
            if not condition: raise ValueError(message)
        context = dict(_check=check)
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(guard_path), 'exec'), context)
        _outside_ray_bootstrap = context['_outside_ray_bootstrap']
        path = Path(__file__).resolve().parents[1]/driver.ENTRY_PATH
        raw = path.read_bytes()
        tree = ast.parse(raw)
        function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'main_task')
        self.assertIsInstance(function.body[0], ast.ImportFrom)
        self.assertEqual(function.body[0].module, 'search_g1.qprobe_driver_gpu')
        self.assertEqual(function.body[1].test.func.id, 'run_native_driver')
        self.assertEqual(len(function.decorator_list), 1)
        self.assertEqual(ast.unparse(function.decorator_list[0]), 'ray.remote')
        exact = (b'@ray.remote\ndef main_task(config):\n'
                 b'    from search_g1.qprobe_driver_gpu import run_native_driver\n'
                 b'    if run_native_driver(config):\n'
                 b'        return\n'
                 b'    run_main_task(config)\n\n\n'
                 b'def run_main_task(config):\n')
        self.assertEqual(raw.count(exact), 1)
        previous = raw.replace(exact, b'@ray.remote\ndef main_task(config):\n', 1)
        self.assertEqual(_outside_ray_bootstrap(raw), _outside_ray_bootstrap(previous))
        previous_tree = ast.parse(previous)
        old = next(item for item in previous_tree.body if isinstance(item, ast.FunctionDef) and item.name == 'main_task')
        body = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'run_main_task')
        self.assertEqual([ast.dump(item) for item in body.body], [ast.dump(item) for item in old.body])
        changed = raw.replace(b'    trainer.fit()\n', b'    trainer.fit()\n    alter_reward()\n')
        self.assertNotEqual(_outside_ray_bootstrap(changed), _outside_ray_bootstrap(previous))
        changed = raw.replace(b'    run_main_task(config)\n', b'    run_main_task(config)\n    alter_reward()\n')
        self.assertNotEqual(_outside_ray_bootstrap(changed), _outside_ray_bootstrap(previous))
        helper = ast.parse(Path(driver.__file__).read_bytes())
        child = next(item for item in helper.body if isinstance(item, ast.FunctionDef) and item.name == '_child')
        init = next(item for item in ast.walk(child) if isinstance(item, ast.Call) and ast.unparse(item.func) == 'ray.init')
        self.assertEqual({item.arg for item in init.keywords}, {'address', 'namespace', 'runtime_env'})
        self.assertEqual(driver.RAY_RUNTIME_ENV, {'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        wrapper = next(item for item in helper.body if isinstance(item, ast.FunctionDef) and item.name == 'run_native_driver')
        self.assertFalse(any(isinstance(item, ast.Attribute) and item.attr == 'cuda' for item in ast.walk(wrapper)))


if __name__ == '__main__': unittest.main()
