"""Focused CPU checks for strict LR-only resume; no Qwen/GPU model gates."""
import copy
import random
import unittest

from .train_epoch import (binding_digest, validate_learning_rate_transition,
                          validate_saved_binding, apply_restored_learning_rate)


def fixture():
    source = dict(schema_version=1, prefix_tokens=20, global_batch=256, world_size=8,
                  accumulation=1, learning_rate=.0003, seed=1729, epochs=1000,
                  test_manifest_sha256='b' * 64, loss='single_mean_BCEWithLogits')
    target = dict(source, learning_rate=.00003)
    transition = dict(schema_version=1, authorized_change_only='learning_rate', run_id='same-logical-run',
                      source_binding=source, target_binding=target,
                      source_binding_sha256=binding_digest(source), target_binding_sha256=binding_digest(target),
                      source_learning_rate=.0003, target_learning_rate=.00003,
                      source_complete_epoch=7, start_epoch=8, source_checkpoint_sha256='a' * 64)
    return source, target, transition


class LearningRateTransitionTests(unittest.TestCase):
    def test_explicit_lr_only_receipt_and_complete_source_boundary(self):
        source, target, transition = fixture()
        provenance = validate_learning_rate_transition(transition, target, 8, 'same-logical-run')
        self.assertEqual(provenance['source_complete_epoch'], 7)
        saved = dict(epoch=7, binding=source, binding_sha256=binding_digest(source))
        receipt = dict(epoch=7, binding_sha256=binding_digest(source), sha256='a' * 64)
        self.assertTrue(validate_saved_binding(saved, receipt, target, 8, transition))

    def test_any_other_scientific_field_change_is_rejected(self):
        for field, value in [('global_batch', 128), ('prefix_tokens', 40), ('test_manifest_sha256', 'c' * 64), ('seed', 99)]:
            _, target, transition = fixture()
            target[field] = value
            transition['target_binding'] = target
            transition['target_binding_sha256'] = binding_digest(target)
            with self.assertRaises(ValueError):
                validate_learning_rate_transition(transition, target, 8, 'same-logical-run')

    def test_old_binding_is_accepted_only_once_and_for_exact_source_bytes(self):
        source, target, transition = fixture()
        saved = dict(epoch=7, binding=source, binding_sha256=binding_digest(source))
        receipt = dict(epoch=7, binding_sha256=binding_digest(source), sha256='a' * 64)
        with self.assertRaises(ValueError):
            validate_saved_binding(saved, receipt, target, 8, None)
        receipt['sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            validate_saved_binding(saved, receipt, target, 8, transition)
        saved.update(epoch=8)
        receipt.update(epoch=8, sha256='a' * 64)
        with self.assertRaises(ValueError):
            validate_saved_binding(saved, receipt, target, 9, transition)
        saved.update(binding=target, binding_sha256=binding_digest(target))
        receipt.update(binding_sha256=binding_digest(target))
        self.assertFalse(validate_saved_binding(saved, receipt, target, 9, transition))

    def test_loaded_adam_lr_override_preserves_moments_steps_and_rng(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
        original = torch.optim.Adam([parameter], lr=.0003)
        parameter.square().sum().backward()
        original.step()
        state = copy.deepcopy(original.state_dict())
        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = torch.optim.Adam([resumed_parameter], lr=.00003)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.param_groups[0]['lr'], .0003)  # load overwrites construction LR.
        moment_before = {key: value.clone() for key, value in resumed.state[resumed_parameter].items()}
        options_before = {key: value for key, value in resumed.param_groups[0].items() if key not in ('lr', 'params')}
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        proof = apply_restored_learning_rate(resumed, .0003, .00003, True)
        self.assertEqual(proof['actual_optimizer_learning_rates'], [.00003])
        for key, value in moment_before.items():
            self.assertTrue(torch.equal(value, resumed.state[resumed_parameter][key]), key)
        self.assertEqual(options_before, {key: value for key, value in resumed.param_groups[0].items() if key not in ('lr', 'params')})
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_unreceipted_or_wrong_loaded_lr_is_rejected(self):
        class Optimizer:
            param_groups = [dict(lr=.0003)]
        with self.assertRaises(ValueError):
            apply_restored_learning_rate(Optimizer(), .0003, .00003, False)
        with self.assertRaises(ValueError):
            apply_restored_learning_rate(Optimizer(), .00003, .00003, True)
        _, target, transition = fixture()
        transition['source_learning_rate'] = .0001
        with self.assertRaises(ValueError):
            validate_learning_rate_transition(transition, target, 8, 'same-logical-run')


if __name__ == '__main__':
    unittest.main()
