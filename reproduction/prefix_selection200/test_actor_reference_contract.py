"""Small constructed CPU checks; no native model, GPU, evaluation or acceptance."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .actor import (ActorBudget, WORLD, rank_plan, reference_policy_objective,
                    virtual_minibatches, valid_actual_steps, _state_binding, restore_metadata)
from .actor_safety import (FrozenPolicyMonitor, POLICY_SAFEGUARDS, attempt_result,
                           frozen_monitor_ids, opportunity_result, reference_token_weights)
from search_g1.policy_safety import monitor


def actual_policy_loss():
    """Execute the real small functions without optional Ray/tensordict imports."""
    root = Path(__file__).resolve().parents[2]
    def load(path, name, namespace):
        tree = ast.parse((root / path).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[name]
    masked_mean = load('verl/utils/torch_functional.py', 'masked_mean', dict(torch=torch))
    return load('verl/trainer/ppo/core_algos.py', 'compute_policy_loss',
                dict(torch=torch, verl_F=SimpleNamespace(masked_mean=masked_mean)))


class ReferenceObjectiveTests(unittest.TestCase):
    def test_large_forward_and_oom1_match_original64_equal_micro2_values_and_gradients(self):
        loss_fn = actual_policy_loss()
        lengths = torch.tensor([1 + (i * 7) % 9 for i in range(128)])
        mask = (torch.arange(9)[None, :] < lengths[:, None]).double()
        mask[16:32] = 0.  # Empty pairs remain zero in the original equal mean.
        mask[52] = 0.  # OOM1 must not renormalize the surviving row independently.
        features = torch.arange(128 * 9, dtype=torch.float64).reshape(128, 9) / 113. - 4.
        advantages = torch.where(torch.arange(128)[:, None] % 3 == 0, -1., .7).expand(-1, 9).double()
        old = torch.zeros_like(features)
        parameter = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
        current = features * parameter
        original = parameter * 0.
        for start in range(0, 128, 2):
            if mask[start:start + 2].sum():
                original = original + loss_fn(old[start:start + 2], current[start:start + 2],
                    advantages[start:start + 2], mask[start:start + 2], cliprange=.2, dual_clip=3.)[0] / 64
        original.backward(); gradient = parameter.grad.clone()
        obsolete = parameter * 0.
        for start in range(0, 128, 16):
            if mask[start:start + 16].sum():
                obsolete = obsolete + loss_fn(old[start:start + 16], current[start:start + 16],
                    advantages[start:start + 16], mask[start:start + 16], cliprange=.2, dual_clip=3.)[0] / 8
        self.assertGreater(abs(float(obsolete.detach() - original.detach())), 1e-4)
        for requested in (4, 8, 16, 64, 128):
            parameter.grad = None; physical = []; budget = ActorBudget(requested)
            current = features * parameter
            for rank in range(WORLD):
                reduced = parameter * 0.
                for ids in virtual_minibatches(list(range(rank * 32, (rank + 1) * 32))):
                    masses = [int(mask[ids[pair:pair + 2]].sum()) for pair in range(0, 16, 2)]
                    for offset in range(0, 16, budget.local_micro):
                        subset = ids[offset:offset + budget.local_micro]
                        value, _ = reference_policy_objective(old[subset], current[subset], advantages[subset], mask[subset],
                            virtual_offset=offset, virtual_pair_token_masses=masses, loss_fn=loss_fn)
                        reduced = reduced + value
                physical.append(reduced)
            actual = sum(physical) / WORLD
            actual.backward()
            torch.testing.assert_close(actual.detach(), original.detach(), rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(parameter.grad, gradient, rtol=1e-13, atol=1e-13)

    def test_length_strata_are_original_virtual_rank_samples(self):
        plan = rank_plan(1729, 1)
        rows = [dict(response_info_mask=[1] * (1 + i % 103)) for i in range(2560)]
        observed = []
        for physical in plan:
            selected = frozen_monitor_ids(physical, rows)
            self.assertEqual(len(selected), 8)
            for virtual in range(2):
                assigned = [i for mini in physical for i in mini[virtual * 16:(virtual + 1) * 16]]
                ordered = sorted(assigned, key=lambda i: len(rows[i]['response_info_mask']))
                self.assertEqual(selected[virtual * 4:(virtual + 1) * 4], [ordered[i] for i in (0,106,212,319)])
            observed.extend(selected)
        self.assertEqual(len(set(observed)), 32)

    def test_refit_version_is_mutable_but_model_data_source_and_math_are_bound(self):
        config = dict(run_id='constructed', model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
            official_model_dir='/data/models/official', source_commit='a'*40,
            model_files_sha256={'config.json':'b'*64}, canonical_train='/data/train.parquet',
            canonical_train_sha256='c'*64, nonmixed_advantage_mode='zero_anchor_all_correct_only',
            grounding_probe_version='round0', grounding_manifest='/data/round0/manifest.json',
            grounding_manifest_sha256='d'*64)
        latest = dict(run_id='constructed', completed_outer=50, binding=_state_binding(config),
                      actual_adam_steps_per_rank=[751]*4, scheduler_completed_outer=50)
        next_config = dict(config, grounding_probe_version='round50',
            grounding_manifest='/data/round50/manifest.json', grounding_manifest_sha256='e'*64)
        restore_metadata(next_config, latest, 51)
        for field, value in [('model_revision','other'), ('official_model_dir','/data/models/other'),
                ('model_files_sha256',{'config.json':'f'*64}), ('source_commit','f'*40),
                ('canonical_train_sha256','f'*64)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                restore_metadata(dict(next_config, **{field:value}), latest, 51)
        for field, value in [('loss_reduction','obsolete_local16_mean'), ('dual_clip',4.), ('global_mini',256)]:
            mutated = copy.deepcopy(latest); mutated['binding'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                restore_metadata(next_config, mutated, 51)


class InlineGuardTests(unittest.TestCase):
    def report(self, delta, advantages=None, mask=None):
        delta = torch.as_tensor(delta, dtype=torch.float64)
        advantages = torch.full_like(delta, -1.) if advantages is None else torch.as_tensor(advantages).double()
        mask = torch.ones_like(delta) if mask is None else torch.as_tensor(mask).double()
        return monitor(delta, advantages, mask, reference_token_weights(mask),
            c=3., kl_log_bound=30., target_kl=.2, negative_tail_limit=.01)

    def test_actual_monitor_kl_tail_bound_and_pair_weights(self):
        mask = torch.tensor([[1,0,0,0], [1,0,0,0], [1,1,1,1], [1,1,1,1]])
        delta = torch.tensor([[1.,0,0,0], [1.,0,0,0], [0.,0,0,0], [0.,0,0,0]])
        report = self.report(delta, mask=mask)
        self.assertAlmostEqual(report['sampled_old_policy_kl'], (math.e - 2.) / 2.)
        self.assertIn('old_policy_kl', report['reasons'])
        self.assertTrue(self.report([[0.], [0.]])['allowed'])
        report = self.report([[math.log(3.) + 1e-6], [0.]])
        self.assertIn('negative_advantage_tail', report['reasons'])
        self.assertEqual(report['negative_tail_weight_fraction'], .5)
        report = self.report([[30.], [30.]], advantages=[[0.], [0.]])
        self.assertEqual(report['overflow_count'], 0)
        report = self.report([[30.0001], [0.]])
        self.assertIn('delta_above_kl_log_bound', report['reasons'])
        self.assertFalse(report['kl_estimate_complete']); self.assertIsNone(report['sampled_old_policy_kl'])
        report = self.report([[float('nan')], [0.]])
        self.assertIn('nonfinite_or_negative_effective_input', report['reasons'])

    def test_after_guard_retains_real_adam_then_processes_all20_opportunities(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        healthy = dict(allowed=True, reasons=[])
        guarded = dict(allowed=False, reasons=['negative_advantage_tail'])
        initial = parameter.detach().clone(); rows = []
        for opportunity in range(20):
            before = int(optimizer.state[parameter]['step']) if parameter in optimizer.state else 0
            if opportunity == 0:
                (parameter.square().sum()).backward(); optimizer.step(); optimizer.zero_grad()
            after = int(optimizer.state[parameter]['step'])
            rows.append(opportunity_result(opportunity=opportunity, before=before, after=after,
                pre=healthy if opportunity == 0 else guarded, post=guarded))
        self.assertFalse(torch.equal(parameter.detach(), initial))
        result = attempt_result(rows, writes_before=0)
        self.assertTrue(result['complete_outer']); self.assertFalse(result['full_budget_complete'])
        self.assertEqual(result['optimizer_writes_this_attempt'], 1)
        self.assertEqual(result['written_with_post_guard'], 1)
        self.assertEqual(result['skipped_update_opportunities'], 19)
        self.assertFalse(result['automatic_rollback_performed'])
        skipped = [opportunity_result(opportunity=i,before=0,after=0,pre=guarded,post=guarded) for i in range(20)]
        result = attempt_result(skipped, writes_before=0)
        self.assertEqual(result['status'], 'recoverable_guard'); self.assertEqual(result['scheduler_advances'], 0)
        with self.assertRaises(ValueError): attempt_result(rows[:-1], writes_before=0)
        self.assertTrue(valid_actual_steps([1]*4, 1)); self.assertTrue(valid_actual_steps([21]*4, 2))
        self.assertFalse(valid_actual_steps([0]*4, 1)); self.assertFalse(valid_actual_steps([21,20,21,21], 2))

    def test_adjacent_monitor_cache_binds_state_sample_and_thresholds(self):
        class ConstructedActor:
            actor_module = torch.nn.Linear(1, 1)
            calls = 0
            def _forward_micro_batch(self, batch, **kwargs):
                self.calls += 1
                return None, batch['old_log_probs'].clone()
        actor = ConstructedActor()
        batch = dict(responses=torch.zeros((8,2),dtype=torch.long), old_log_probs=torch.zeros((8,2)),
                     loss_mask=torch.ones((8,2)), advantages=-torch.ones((8,2)))
        inline = FrozenPolicyMonitor(actor, batch, device='cpu', local_micro=4,
                                     identity=dict(attempt_id='a'*32, actor_version='constructed'))
        before = inline.observe(stage='before_minibatch', opportunity=0, actual_adam_steps=0)
        self.assertTrue(before['allowed']); self.assertEqual(actor.calls, 2)
        inline.observe(stage='after_minibatch', opportunity=0, actual_adam_steps=1)
        before = inline.observe(stage='before_minibatch', opportunity=1, actual_adam_steps=1)
        self.assertTrue(before['reused_adjacent_post_observation']); self.assertEqual(actor.calls, 4)
        self.assertEqual(before['binding']['thresholds'], POLICY_SAFEGUARDS)
        inline.observe(stage='before_minibatch', opportunity=1, actual_adam_steps=2)
        self.assertEqual(actor.calls, 6)  # A new state cannot use the old after observation.
        inline.binding['thresholds']['target_old_policy_kl'] = .3
        with self.assertRaisesRegex(ValueError, 'bound monitor thresholds'):
            inline.observe(stage='before_minibatch', opportunity=1, actual_adam_steps=1)
        inline.binding['thresholds']['target_old_policy_kl'] = .2
        batch['old_log_probs'][0,0] = .1
        with self.assertRaisesRegex(ValueError, 'Frozen monitor inputs changed'):
            inline.observe(stage='after_minibatch', opportunity=1, actual_adam_steps=2)


if __name__ == '__main__': unittest.main()
