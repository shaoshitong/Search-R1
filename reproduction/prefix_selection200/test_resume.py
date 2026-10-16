"""CPU tests for the complete actor/probe/pool recovery boundary."""
import copy
import unittest
from .actor import _state_binding
from .resume import boundary


class ResumeTests(unittest.TestCase):
    def test_resume5_requires_both_real_adam_clocks_and_next_candidate_cursor(self):
        c = dict(run_id='run', official_model_dir='/dataset/model', grounding_probe_version='fixed0',
                 resume_joint=dict(completed_outer=5))
        a = dict(run_id='run', completed_outer=5, binding=_state_binding(c),
                 actual_adam_steps_per_rank=[100]*8, scheduler_completed_outer=5)
        p = dict(run_id='run', completed_outer=5, actual_adam_steps_per_rank=[50]*8)
        pool = dict(next_outer=6, consumed_candidates=25600, cycle=0, cursor=25600, order=list(range(168510)))
        joint = dict(run_id='run', outer=5, actor=a, probe=p)
        self.assertEqual(boundary(c, joint, a, p, pool), 5)
        changed = copy.deepcopy(p); changed['actual_adam_steps_per_rank'][7] = 49
        with self.assertRaises(ValueError): boundary(c, joint, a, changed, pool)
        with self.assertRaises(ValueError): boundary(c, joint, a, p, dict(pool, cursor=0))
        changed = copy.deepcopy(a); changed['scheduler_completed_outer'] = 4
        with self.assertRaises(ValueError): boundary(c, dict(joint, actor=changed), changed, p, pool)


if __name__ == '__main__': unittest.main()
