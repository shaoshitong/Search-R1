"""Metadata-only native Ray-driver routing fixtures; no Ray or CUDA execution."""
import ast
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from search_g1 import qprobe_driver_gpu as driver
from reproduction.common8.native_allocation import LOCAL_BINDING_RECEIPT, LOCAL_PLACEMENT_POLICY


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

    def test_cpu_ray_task_routes_to_first_admitted_uuid_without_extra_gpu_or_rng_change(self):
        route = self.route()
        self.assertEqual(route['physical_index'], 1)
        self.assertEqual(route['gpu_uuid'], 'GPU-fixture-1')
        self.assertEqual(route['ray_gpu_resources_reserved'], 0)
        self.assertTrue(route['shares_admitted_actor_training_gpu'])
        environment = {'CUDA_VISIBLE_DEVICES': ''}
        before = torch.get_rng_state().clone()
        calls = []
        def count():
            calls.append(('count', environment['CUDA_VISIBLE_DEVICES'])); return 1
        def uuids(size):
            calls.append(('uuid', environment['CUDA_VISIBLE_DEVICES'])); self.assertEqual(size, 1)
            return [route['gpu_uuid']]
        proof = driver.install_driver_visibility(route, environ=environment, cuda_initialized=lambda: False,
            cuda_device_count=count, cuda_uuids=uuids)
        self.assertEqual(environment, {'CUDA_VISIBLE_DEVICES': 'GPU-fixture-1'})
        self.assertEqual(calls, [('count', 'GPU-fixture-1'), ('uuid', 'GPU-fixture-1')])
        self.assertFalse(proof['rng_seed_performed'])
        self.assertFalse(proof['tensor_allocation_performed'])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

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
        for environment, initialized in (({'CUDA_VISIBLE_DEVICES': ''}, True),
                ({'CUDA_VISIBLE_DEVICES': '', 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1'}, False)):
            with self.subTest(initialized=initialized), self.assertRaises(ValueError):
                driver.install_driver_visibility(route, environ=environment, cuda_initialized=lambda: initialized,
                    cuda_device_count=lambda: self.fail('No identity query after forbidden environment'),
                    cuda_uuids=lambda _size: self.fail('No UUID query after forbidden environment'))
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')

    def test_actual_cuda_mapping_must_be_one_exact_training_uuid(self):
        route = self.route()
        for count, uuids in ((0, []), (2, [route['gpu_uuid'], 'GPU-fixture-5']), (1, ['GPU-foreign'])):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'ACTUAL_CUDA_UUID_DIFFERS'):
                driver.install_driver_visibility(route, environ={}, cuda_initialized=lambda: False,
                    cuda_device_count=lambda: count, cuda_uuids=lambda _size: uuids)

    def test_non_native_entry_has_no_file_or_cuda_side_effects(self):
        with patch.object(driver, '_json', side_effect=AssertionError('No manifest read for old heads')):
            self.assertIsNone(driver.bind_native_driver_gpu(dict(question_probe=dict(head_kind='count6_v1'))))

    def test_exact_entry_hook_precedes_framework_work_and_source_guard_keeps_other_code_frozen(self):
        from reproduction.a800_common.adopt_initial_probes import _outside_ray_bootstrap
        path = Path(__file__).resolve().parents[1]/driver.ENTRY_PATH
        raw = path.read_bytes()
        tree = ast.parse(raw)
        function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'main_task')
        self.assertIsInstance(function.body[0], ast.ImportFrom)
        self.assertEqual(function.body[0].module, 'search_g1.qprobe_driver_gpu')
        self.assertEqual(function.body[1].value.func.id, 'bind_native_driver_gpu')
        self.assertEqual(len(function.decorator_list), 1)
        self.assertEqual(ast.unparse(function.decorator_list[0]), 'ray.remote')
        exact = (b'    from search_g1.qprobe_driver_gpu import bind_native_driver_gpu\n'
                 b'    bind_native_driver_gpu(config)\n')
        previous = raw.replace(exact, b'', 1)
        self.assertEqual(_outside_ray_bootstrap(raw), _outside_ray_bootstrap(previous))
        changed = raw.replace(b'    trainer.fit()\n', b'    trainer.fit()\n    alter_reward()\n')
        self.assertNotEqual(_outside_ray_bootstrap(changed), _outside_ray_bootstrap(previous))


if __name__ == '__main__': unittest.main()
