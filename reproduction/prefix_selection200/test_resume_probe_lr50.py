"""CPU checks for a trained Adam200 LR-only rollback, not Adam0 recovery."""
import copy
import unittest

from .checkpoint_cpu_once import digest
from .resume_probe_lr50 import rebind_trained_probe, validate_change


class TrainedLR50Tests(unittest.TestCase):
    def saved(self):
        value = dict(completed_outer=40, actual_adam_steps_per_rank=[200] * 8,
            binding=dict(learning_rate=1e-6, loss='unchanged'), learning_rate=1e-6,
            model={'weights': object()}, optimizer=dict(state={0: {'step': 200, 'moment': object()}},
                param_groups=[dict(lr=1e-6, params=[0], betas=(.9, .999), weight_decay=0)]),
            rank_rng=[dict(rank=r, actual_adam_steps=200, saved_rng=object()) for r in range(8)])
        value['binding_sha256'] = digest(value['binding'])
        return value

    def test_lr_change_preserves_trained_moments_model_and_all_rng(self):
        source = self.saved(); result = rebind_trained_probe(source)
        self.assertIs(result['model'], source['model'])
        self.assertIs(result['optimizer']['state'], source['optimizer']['state'])
        self.assertIs(result['rank_rng'], source['rank_rng'])
        self.assertEqual(result['optimizer']['param_groups'][0]['lr'], 1e-5)
        self.assertEqual(source['optimizer']['param_groups'][0]['lr'], 1e-6)
        self.assertEqual(result['optimizer']['state'][0]['step'], 200)
        self.assertEqual(result['binding_sha256'], digest(result['binding']))

    def test_rejects_reset_clock_and_wrong_rank_rng(self):
        for key, changed in [('completed_outer', 0), ('actual_adam_steps_per_rank', [0] * 8),
                             ('optimizer', None)]:
            value = self.saved(); value[key] = changed
            with self.assertRaises(ValueError): rebind_trained_probe(value)
        value = self.saved(); value['rank_rng'][7]['actual_adam_steps'] = 199
        with self.assertRaises(ValueError): rebind_trained_probe(value)

    def test_rejects_accidental_actor_or_budget_change(self):
        original = dict(base='/mnt/data/shaoshitong/source', source_commit='old',
                        probe_learning_rate=1e-6, actor_learning_rate=1e-6,
                        actor_global_mini_batch=128, probe_update_period=20)
        changed = dict(original, base='/mnt/data/shaoshitong/new', source_commit='new',
            probe_learning_rate=1e-5, resume_joint=dict(completed_outer=50,
                human_authorization=dict(resume_from50=True, probe_lr_only=True,
                                        raw_human_message='需要变成1e-5跑，探针部分')))
        validate_change(changed, original)
        for key, value in [('actor_learning_rate', 1e-5), ('actor_global_mini_batch', 64),
                           ('probe_update_period', 1), ('probe_learning_rate', 3e-5)]:
            with self.assertRaises(ValueError): validate_change(dict(changed, **{key: value}), original)
        value = copy.deepcopy(changed); value['resume_joint']['completed_outer'] = 5
        with self.assertRaises(ValueError): validate_change(value, original)


if __name__ == '__main__':
    unittest.main()
