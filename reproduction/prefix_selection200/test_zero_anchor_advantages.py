"""CPU formula checks: virtual0 applies only to actual all-correct ownEM G5."""
import copy
import math
import unittest

from .actor import (ADVANTAGE_EPSILON, ZERO_ANCHOR_MODE, _state_binding,
                    advantage_contract, group_advantages, normalize_reward_group,
                    rank_plan, validate_actor_selection, virtual_minibatches)


def group(qid, correctness):
    return [dict(qid=qid, response_index=index, strict_em=correct)
            for index, correct in enumerate(correctness)]


def independent_normalize(values, anchor):
    statistics = [0.] + list(values) if anchor else list(values)
    mean = math.fsum(statistics) / len(statistics)
    sample_std = math.sqrt(math.fsum((value - mean) ** 2 for value in statistics) / (len(statistics) - 1))
    return [(value - mean) / (sample_std + ADVANTAGE_EPSILON) for value in values], mean, sample_std


class ZeroAnchorAdvantages(unittest.TestCase):
    def test_user_three_reward_example_uses_four_statistics_not_three(self):
        import torch
        values = [.97, 1.03, 1.05]
        actual, proof = normalize_reward_group(values, zero_anchor=True)
        expected, mean, std = independent_normalize(values, True)
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-6)
        self.assertEqual(len(actual), 3)
        self.assertEqual(proof['normalization_count'], 4)
        self.assertAlmostEqual(proof['normalization_mean'], mean, places=6)
        self.assertAlmostEqual(proof['normalization_sample_std'], std, places=6)
        self.assertTrue(proof['virtual_zero_in_statistics'])
        self.assertFalse(proof['virtual_zero_in_actor_batch'])

    def test_five_real_all_correct_rewards_keep_only_five_real_advantages(self):
        import torch
        values = [.97, 1.03, 1.05, .99, 1.01]
        rows = group('correct', [1] * 5)
        actual, proof = group_advantages(rows, values, mode=ZERO_ANCHOR_MODE, return_diagnostics=True)
        expected, _, _ = independent_normalize(values, True)
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-6)
        self.assertTrue(bool((actual > 0).all()))
        self.assertEqual(actual.numel(), 5)
        self.assertEqual(proof['groups'][0]['normalization_count'], 6)
        self.assertEqual(proof['summary']['actual_group_counts']['all_correct'], 1)
        self.assertEqual(proof['summary']['synthetic_actor_trajectories'], 0)

    def test_equal_positive_all_correct_values_gain_nonzero_positive_advantage(self):
        import torch
        rows = group('same', [1] * 5)
        values = torch.ones(5)
        original = group_advantages(rows, values)
        actual = group_advantages(rows, values, mode=ZERO_ANCHOR_MODE)
        self.assertEqual(original.tolist(), [0.] * 5)
        expected, _, _ = independent_normalize([1.] * 5, True)
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-6)
        self.assertTrue(bool((actual > 0).all()))

    def test_anchor_is_not_translation_invariant_but_original_formula_is(self):
        import torch
        values = torch.tensor([.97, 1.03, 1.05, .99, 1.01])
        raw, _ = normalize_reward_group(values)
        shifted_raw, _ = normalize_reward_group(values + 3)
        torch.testing.assert_close(raw, shifted_raw, rtol=1e-4, atol=1e-5)
        anchored, _ = normalize_reward_group(values, zero_anchor=True)
        shifted_anchor, _ = normalize_reward_group(values + 3, zero_anchor=True)
        self.assertGreater(float((anchored - shifted_anchor).abs().max()), .01)

    def test_mixed_and_all_wrong_preserve_exact_original_sample_std_formula(self):
        import torch
        values = torch.tensor([.1, .2, .4, .7, .9])
        expected = (values - values.mean()) / (values.std(correction=1) + 1e-6)
        for label in ([1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0] * 5):
            actual, proof = group_advantages(group('actual', label), values,
                mode=ZERO_ANCHOR_MODE, return_diagnostics=True)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(proof['summary']['virtual_zero_statistics_groups'], 0)
            self.assertEqual(proof['groups'][0]['normalization_count'], 5)

    def test_equal_reward_mixed_is_classified_by_em_and_remains_zero(self):
        rows = group('mixed', [1, 0, 0, 0, 0])
        actual, proof = group_advantages(rows, [1.] * 5, mode=ZERO_ANCHOR_MODE, return_diagnostics=True)
        self.assertEqual(actual.tolist(), [0.] * 5)
        self.assertEqual(proof['groups'][0]['group_kind'], 'mixed')
        self.assertEqual(proof['groups'][0]['actual_correct_count'], 1)

    def test_zero_only_rewards_finite_for_every_real_group_type(self):
        import torch
        for labels in ([1] * 5, [0] * 5, [1, 0, 0, 0, 0]):
            actual, proof = group_advantages(group('zero', labels), [0.] * 5,
                mode=ZERO_ANCHOR_MODE, return_diagnostics=True)
            self.assertEqual(actual.tolist(), [0.] * 5)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertEqual(proof['groups'][0]['normalization_sample_std'], 0.)

    def test_real_labels_response_identity_and_rewards_are_not_changed(self):
        import torch
        rows = group('correct', [1] * 5) + group('wrong', [0] * 5) + group('mixed', [1, 0, 1, 0, 1])
        original_rows = copy.deepcopy(rows)
        rewards = torch.tensor([.97, 1.03, 1.05, .99, 1.01] * 3)
        original_rewards = rewards.clone()
        result, proof = group_advantages(rows, rewards, mode=ZERO_ANCHOR_MODE, return_diagnostics=True)
        self.assertEqual(rows, original_rows)
        self.assertTrue(torch.equal(rewards, original_rewards))
        self.assertEqual(result.numel(), 15)
        self.assertEqual(proof['summary']['actual_group_counts'],
            dict(all_correct=1, all_wrong=1, mixed=1, original_g5=0))
        self.assertEqual(proof['summary']['actual_trajectories'], 15)
        self.assertFalse(proof['summary']['raw_rewards_modified'])
        self.assertEqual(proof['groups'][0]['raw_reward_mean'], proof['groups'][1]['raw_reward_mean'])
        self.assertNotEqual(proof['groups'][0]['normalization_mean'], proof['groups'][1]['normalization_mean'])

    def test_missing_inferred_or_invalid_em_and_incomplete_g5_fail_closed(self):
        for mutation in ('missing', 'boolean', 'outside_binary', 'response_duplicate', 'response_bool', 'short'):
            rows = group('bad', [1] * 5)
            if mutation == 'missing': rows[0].pop('strict_em')
            elif mutation == 'boolean': rows[0]['strict_em'] = True
            elif mutation == 'outside_binary': rows[0]['strict_em'] = 2
            elif mutation == 'response_duplicate': rows[0]['response_index'] = 1
            elif mutation == 'response_bool': rows[1]['response_index'] = True
            else: rows.pop()
            with self.assertRaises(ValueError):
                group_advantages(rows, [1.] * len(rows), mode=ZERO_ANCHOR_MODE)
        with self.assertRaises(ValueError):
            group_advantages(group('bad', [1] * 5), [1., 1., float('nan'), 1., 1.], mode=ZERO_ANCHOR_MODE)

    def test_production_binding_explicit_and_actor_budget_unchanged(self):
        config = dict(run_id='fresh', official_model_dir='/data/models/official', grounding_probe_version='fixed0',
                      nonmixed_advantage_mode=ZERO_ANCHOR_MODE)
        proof = advantage_contract(config)
        self.assertEqual(_state_binding(config)['advantage_normalization'], proof)
        self.assertFalse(proof['synthetic_actor_trajectory'])
        self.assertEqual(proof['anchored_group'], 'all_five_correct')
        for bad in (None, 'original_g5', 'zero_anchor_all_nonmixed'):
            changed = dict(config, nonmixed_advantage_mode=bad)
            with self.assertRaises(ValueError): advantage_contract(changed)
        plan = rank_plan(1729, 1)
        self.assertEqual(len(plan), 4)
        self.assertEqual({len(rank) for rank in plan}, {20})
        self.assertEqual({len(batch) for rank in plan for batch in rank}, {32})
        self.assertEqual({len(block) for rank in plan for batch in rank
                          for block in virtual_minibatches(batch)}, {16})
        self.assertEqual(sorted(i for rank in plan for batch in rank for i in batch), list(range(2560)))

    def test_actual_em_final_selection_is_bound_instead_of_predicted_selection(self):
        selected = [dict(qid='q%d' % i, candidate_ordinal=i,
                         trajectory_ids=['t%d-%d' % (i, j) for j in range(5)]) for i in range(512)]
        complete = dict(run_id='run', outer=10, actor_version='actor9', prefix_scan_sha256='c' * 64,
            actor_selection_sha256='a' * 64, projected_from_full_pool=True,
            full_pool_origin_selection_sha256='b' * 64, full_pool_complete_sha256='d' * 64,
            full_pool_own_em_sha256='e' * 64, predicted_selection_sha256='f' * 64)
        selection = dict(run_id='run', outer=10, actor_version='actor9', prefix_scan_sha256='c' * 64,
                         selected_questions=512, selected=selected)
        lineage = {key: complete[key] for key in ('full_pool_origin_selection_sha256',
            'full_pool_complete_sha256', 'full_pool_own_em_sha256', 'predicted_selection_sha256')}
        rows = [dict(qid=item['qid'], candidate_ordinal=item['candidate_ordinal'], trajectory_id=tid,
            selection_sha256='a' * 64, prefix_scan_sha256='c' * 64, **lineage)
            for item in selected for tid in item['trajectory_ids']]
        self.assertTrue(validate_actor_selection(complete, rows, selection))
        changed = copy.deepcopy(rows); changed[0]['selection_sha256'] = complete['predicted_selection_sha256']
        with self.assertRaises(ValueError): validate_actor_selection(complete, changed, selection)
        for key in lineage:
            changed = copy.deepcopy(rows); changed[0][key] = '0' * 64
            with self.assertRaises(ValueError): validate_actor_selection(complete, changed, selection)
        changed = copy.deepcopy(selection); changed['selected'][0]['trajectory_ids'][0] = 'different-sampling-event'
        with self.assertRaises(ValueError): validate_actor_selection(complete, rows, changed)


if __name__ == '__main__': unittest.main()
