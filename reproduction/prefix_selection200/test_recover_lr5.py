"""The approved rollback must preserve actor recipe and pre20 probe clock."""
import copy
import unittest

from .recover_lr5 import SCIENCE, rebind_untrained_probe, validate_change
from .checkpoint_cpu_once import digest


class LR5RecoveryTests(unittest.TestCase):
    def test_only_probe_lr_may_change(self):
        original = {key: key for key in SCIENCE}
        original['probe_learning_rate'] = 1e-6
        changed = dict(original, probe_learning_rate=1e-5,
                       resume_joint=dict(completed_outer=5, human_authorization=dict(resume_from5=True)))
        validate_change(changed, original)
        for key in ('actor_global_mini_batch', 'probe_updates_per_period', 'hidden_capture_contract'):
            with self.assertRaises(ValueError):
                validate_change(dict(changed, **{key: 'modified'}), original)
        with self.assertRaises(ValueError):
            validate_change(dict(changed, probe_learning_rate=3e-5), original)

    def test_rebind_does_not_import_an_updated20_probe_or_reset_adam(self):
        model = {'unchanged': object()}
        source = dict(completed_outer=0, actual_adam_steps_per_rank=[0]*8,
                      optimizer=None, rank_rng=None, binding=dict(learning_rate=1e-6, loss='joint'),
                      model=model, learning_rate=1e-6)
        changed = rebind_untrained_probe(source, 1e-5)
        self.assertIs(changed['model'], model)
        self.assertEqual(changed['binding_sha256'], digest(changed['binding']))
        self.assertEqual(changed['learning_rate'], 1e-5)
        self.assertEqual(source['binding']['learning_rate'], 1e-6)
        updated = copy.deepcopy(source)
        updated.update(completed_outer=20, actual_adam_steps_per_rank=[100]*8, optimizer={})
        with self.assertRaises(ValueError):
            rebind_untrained_probe(updated, 1e-5)


if __name__ == '__main__':
    unittest.main()
