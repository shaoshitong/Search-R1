"""Pure-CPU scope and recovery-phase checks; no model or production data."""
import copy
import unittest

from . import full169028_protocol as p


class FullDataProtocolTests(unittest.TestCase):
    def config(self):
        return dict(base=str(p.BASE), mode='main', fit_questions=169028,
                    fit_real_trajectories=1690280, steps_per_epoch=5283,
                    epochs=4, update_budget=21132,
                    matched_evaluation_updates=[3882, 7764, 11646, 15528],
                    save_every=128, current_world_size=1, lr=5e-6, ema_decay=.99,
                    loss_kwargs=dict(trajectory_weight=1., binary_mixed_weight=1.),
                    source_checkpoint=dict(steps=14418, sha256='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'))

    def test_real_full_budget_and_original_matched_nodes(self):
        self.assertEqual((p.STEPS_PER_EPOCH, p.UPDATE_BUDGET), (5283, 21132))
        self.assertEqual(p.evaluation_node(5283)['epoch'], 1)
        self.assertEqual(p.evaluation_node(15528), dict(
            kind='matched_124194_update_budget', epoch=None, cursor=15528,
            matched_reference_epoch=4, folder='matched-update-015528'))
        self.assertIsNone(p.evaluation_node(15529))
        self.assertTrue(p.pinned(15528))
        self.assertFalse(p.pinned(15529))

    def test_resume_does_not_lose_fractional_epoch_comparison(self):
        at_node = p.required_past_nodes(15528)
        after_node = p.required_past_nodes(15529)
        self.assertEqual([x['cursor'] for x in at_node], [0, 3882, 5283, 7764, 10566, 11646])
        self.assertEqual([x['cursor'] for x in after_node], [0, 3882, 5283, 7764, 10566, 11646, 15528])
        self.assertEqual(p.evaluation_node(21132)['epoch'], 4)
        self.assertEqual(len(p.required_past_nodes(21132)), 8)

    def test_scope_or_optimizer_clock_drift_rejected(self):
        c = self.config()
        p.validate_config(p.BASE, c)
        for key, bad in [('fit_questions', 124194), ('update_budget', 15528),
                         ('steps_per_epoch', 3882), ('lr', 5e-5), ('current_world_size', 4),
                         ('matched_evaluation_updates', [15528])]:
            value = copy.deepcopy(c)
            value[key] = bad
            with self.assertRaises(ValueError):
                p.validate_config(p.BASE, value)
        value = copy.deepcopy(c)
        value['source_checkpoint']['steps'] = 0
        with self.assertRaises(ValueError):
            p.validate_config(p.BASE, value)
        with self.assertRaises(ValueError):
            p.validate_config(str(p.BASE).replace('169028', '124194'), c)

    def test_invalid_cursor_cannot_look_like_valid_evaluation(self):
        for value in (-1, 21133, True, 5283.0):
            with self.assertRaises(ValueError):
                p.evaluation_node(value)


if __name__ == '__main__':
    unittest.main()
