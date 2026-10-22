"""CPU execution of production phase methods; no vLLM/GPU imports or benchmarks."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as S
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def manager_fixture(enabled):
    path = ROOT / 'verl/workers/sharding_manager/fsdp_vllm.py'
    definition = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'FSDPVLLMShardingManager')
    methods = [copy.deepcopy(n) for n in definition.body if isinstance(n, ast.FunctionDef) and n.name != '__init__']
    events = []
    cuda = S(state=10)
    cuda.get_rng_state = lambda: cuda.state
    cuda.set_rng_state = lambda state: setattr(cuda, 'state', state)
    cuda.empty_cache = lambda: events.append('empty-cache')
    namespace = dict(torch=S(cuda=cuda), logger=None,
                     log_gpu_memory_usage=lambda *args, **kwargs: None, DataProto=object)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    cls = type('ProductionPhaseManager', (), {n.name: namespace[n.name] for n in methods})
    actor = S(weights=torch.tensor([1., 2.]), state_dict_calls=0)
    def state_dict():
        actor.state_dict_calls += 1
        return {'weight': actor.weights}
    actor.state_dict = state_dict
    actor.parameters = lambda: [actor.weights]
    actor.buffers = lambda: []
    actor.train = lambda: events.append('actor-train')
    engine = S(weights=None, syncs=0, offloads=0)
    def sync(params, load_format):
        self_test = params['weight']
        engine.weights = self_test.clone()  # normal loader owns the copy
        engine.syncs += 1
        events.append('sync')
    def offload():
        # Production offload swaps in uninitialized CPU storage; it is not a
        # weight cache and must never satisfy a future snapshot cache hit.
        engine.weights = None
        engine.offloads += 1
        events.append('offload')
    engine.sync_model_weights, engine.offload_model_weights = sync, offload
    sampling_model = S(parameters=lambda: [engine.weights], buffers=lambda: [])
    engine.llm_engine = S(model_executor=S(worker=S(model_runner=S(model=sampling_model))))
    manager = cls()
    manager.module, manager.inference_engine = actor, engine
    manager.model_config, manager.full_params, manager.device_mesh = None, True, object()
    manager._phase_snapshot = manager._phase_setup_snapshot = None
    manager._rng_swapped, manager.phase_weight_sync_count = False, 0
    manager.torch_random_states, manager.gen_random_states = 10, 100
    manager.reuse_snapshot_weights = enabled
    manager._resident_weight_snapshot, manager._retain_phase_weights = None, False
    return manager, actor, engine, cuda, events


def five_chunks(reuse):
    manager, actor, engine, cuda, events = manager_fixture(reuse)
    rows, boundary_rng, needs_sync = [], [], []
    for chunk in range(5):
        needs_sync.append(manager.phase_needs_weight_sync('policy-0', reuse=reuse))
        manager.begin_phase('policy-0', reuse=reuse, retain_weights=reuse and chunk < 4)
        # KV cache setup/end and request sequence remain five separate phases.
        events.append('kv-init')
        rows.append((chunk, engine.weights.tolist(), cuda.state))
        cuda.state += 7
        events.append('kv-free')
        manager.end_phase('policy-0')
        boundary_rng.append((cuda.state, manager.gen_random_states))
    return S(manager=manager, actor=actor, engine=engine, events=events,
             rows=rows, boundary_rng=boundary_rng, needs_sync=needs_sync)


class SnapshotWeightReuseTests(unittest.TestCase):
    def test_five_chunks_one_copy_same_order_rng_boundaries(self):
        legacy, cached = five_chunks(False), five_chunks(True)
        self.assertEqual(legacy.rows, cached.rows)
        self.assertEqual(legacy.boundary_rng, cached.boundary_rng)
        self.assertEqual(legacy.engine.syncs, 5)
        self.assertEqual(cached.engine.syncs, 1)
        self.assertEqual(cached.actor.state_dict_calls, 1)
        self.assertEqual(cached.needs_sync, [True, False, False, False, False])
        self.assertEqual(cached.events.count('kv-init'), 5)
        self.assertEqual(cached.events.count('kv-free'), 5)
        self.assertEqual(cached.manager.phase_weight_sync_count, 1)
        self.assertIsNone(cached.engine.weights)  # released at final chunk
        self.assertIsNone(cached.manager._resident_weight_snapshot)

    def test_new_snapshot_invalidates_before_actor_preload(self):
        manager, actor, engine, _, events = manager_fixture(True)
        manager.begin_phase('policy-0', reuse=True, retain_weights=True)
        manager.end_phase('policy-0')
        self.assertTrue(manager.phase_needs_weight_sync('policy-1', reuse=True))
        self.assertIsNone(engine.weights)
        actor.weights.add_(3)
        manager.begin_phase('policy-1', reuse=True)
        self.assertEqual(engine.weights.tolist(), [4., 5.])
        manager.end_phase('policy-1')
        self.assertEqual(engine.syncs, 2)
        self.assertGreaterEqual(events.count('offload'), 2)

    def test_vllm_cache_is_independent_of_actor_storage(self):
        manager, actor, engine, _, _ = manager_fixture(True)
        manager.begin_phase('policy', reuse=True, retain_weights=True)
        self.assertNotEqual(engine.weights.data_ptr(), actor.weights.data_ptr())
        manager.end_phase('policy')
        actor.weights.zero_()
        self.assertEqual(engine.weights.tolist(), [1., 2.])
        manager.release_snapshot_weights()
        self.assertTrue(manager.phase_needs_weight_sync('policy', reuse=True))

    def test_aliasing_loader_detaches_only_the_sampling_storage(self):
        manager, actor, engine, _, _ = manager_fixture(True)
        def aliasing_sync(params, load_format):
            engine.weights = params['weight'].detach()  # separate object, same backing storage
            engine.syncs += 1
        engine.sync_model_weights = aliasing_sync
        actor_pointer = actor.weights.data_ptr()
        manager.begin_phase('policy', reuse=True, retain_weights=True)
        self.assertEqual(actor.weights.data_ptr(), actor_pointer)
        self.assertNotEqual(engine.weights.data_ptr(), actor_pointer)
        self.assertEqual(engine.weights.tolist(), actor.weights.tolist())
        self.assertEqual(manager._last_weight_storage_proof['detached_actor_aliases'], 1)
        manager.end_phase('policy')
        actor.weights.zero_()
        self.assertEqual(engine.weights.tolist(), [1., 2.])

    def test_abort_never_retains_unfinished_snapshot(self):
        manager, _, engine, cuda, _ = manager_fixture(True)
        manager.begin_phase('policy', reuse=True, retain_weights=True)
        cuda.state = 123
        manager.end_phase('policy', abort=True)
        self.assertIsNone(manager._resident_weight_snapshot)
        self.assertIsNone(engine.weights)
        self.assertEqual(cuda.state, 10)
        self.assertEqual(manager.gen_random_states, 123)

    def test_disabled_profile_rejects_reuse(self):
        manager, *_ = manager_fixture(False)
        with self.assertRaisesRegex(ValueError, 'Undeclared'):
            manager.begin_phase('policy', reuse=True, retain_weights=True)
        self.assertIsNone(manager._phase_setup_snapshot)

    def test_legacy_context_invalidates_cache_and_syncs(self):
        manager, actor, engine, _, _ = manager_fixture(True)
        manager.begin_phase('policy', reuse=True, retain_weights=True)
        manager.end_phase('policy')
        actor.weights.add_(5)
        with manager:
            self.assertEqual(engine.weights.tolist(), [6., 7.])
        self.assertIsNone(manager._resident_weight_snapshot)
        self.assertEqual(engine.syncs, 2)

    def test_actor_work_releases_but_readonly_rng_boundary_does_not(self):
        path = ROOT / 'verl/workers/fsdp_workers.py'
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
        method = next(copy.deepcopy(n) for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == '_assert_generation_phase_inactive')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        manager, _, engine, _, _ = manager_fixture(True)
        worker = S(rollout_sharding_manager=manager)
        manager.begin_phase('policy', reuse=True, retain_weights=True)
        manager.end_phase('policy')
        assert_inactive = namespace['_assert_generation_phase_inactive']
        assert_inactive(worker, release_snapshot_weights=False)
        self.assertIsNotNone(engine.weights)
        assert_inactive(worker)
        self.assertIsNone(engine.weights)

    def test_actual_worker_skips_actor_preload_only_on_same_snapshot(self):
        import os
        from unittest.mock import patch
        from reproduction.common8.test_continuous_sampling import phase_worker_fixture
        worker, snapshot, events = phase_worker_fixture()
        manager, _, engine, _, _ = manager_fixture(True)
        worker.rollout_sharding_manager = manager
        worker.config.rollout.get = lambda key, default=None: True if key == 'reuse_snapshot_weights' else default
        with patch.dict(os.environ, SEARCH_G1_PROFILE='common8'):
            for chunk in range(5):
                current = copy.deepcopy(snapshot)
                current.update(phase_id='phase-' + str(chunk), snapshot_weight_reuse=True,
                               retain_snapshot_weights=chunk < 4)
                handle = worker.begin_generation_phase(current)
                # The worker fixture creates the same phase handle semantics
                # as the actual vLLM begin method.
                worker.rollout._generation_phase_setup_handle = handle
                worker.end_generation_phase(handle)
        self.assertEqual(events.count('load-actor'), 1)
        self.assertEqual(events.count('offload-actor'), 1)
        self.assertEqual(events.count('cache-init'), 5)
        self.assertEqual(engine.syncs, 1)
        self.assertEqual(manager.phase_weight_sync_count, 1)
        self.assertIsNone(engine.weights)


if __name__ == '__main__':
    unittest.main()
