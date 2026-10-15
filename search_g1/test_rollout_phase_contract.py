"""Execution contract regression for the four- and eight-engine deployments."""
import unittest
from types import SimpleNamespace
from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE, validate, hydra_arguments, configure_generation

class RolloutPhaseContractTests(unittest.TestCase):
    def test_eight_engine_binding_and_generation_keep_identical_limits(self):
        limits = dict(DEFAULT_ROLLOUT_PHASE)
        self.assertEqual(validate(limits, 8), limits)
        args = hydra_arguments(dict(training_world_size=8, infra_rollout=limits))
        self.assertIn('+actor_rollout_ref.rollout.async_phase_enabled=true', args)
        config = configure_generation(SimpleNamespace(num_gpus=8), dict(limits, async_phase_enabled=True))
        self.assertEqual(config.max_inflight_trajectories, 256)
        self.assertEqual(config.generation_max_num_seqs, 64)
        self.assertEqual(validate(limits, 4), limits)

    def test_engine_capacity_and_unsupported_world_remain_rejected(self):
        for world in (0, 1, 2, 16, True, 8.0):
            with self.assertRaises(ValueError): validate(dict(DEFAULT_ROLLOUT_PHASE), world)
        limits = dict(DEFAULT_ROLLOUT_PHASE, agent_workers=513, max_inflight_trajectories=513)
        with self.assertRaisesRegex(ValueError, 'actual engine sequence capacity'): validate(limits, 8)
        with self.assertRaisesRegex(ValueError, 'Unsupported'): validate(dict(DEFAULT_ROLLOUT_PHASE, mode='wrong'), 8)

if __name__ == '__main__': unittest.main()
