import unittest
from search_g1.easy7b_infra import flags


class InfraFlagsTest(unittest.TestCase):
    def test_old_and_current_manifests_remain_disabled(self):
        self.assertEqual(flags({}),dict(selected_policy_logits=False,exact_retrieval_cache=False))
        self.assertEqual(flags({'infra_optimizations':dict(selected_policy_logits=False,exact_retrieval_cache=False)}),
                         dict(selected_policy_logits=False,exact_retrieval_cache=False))

    def test_enabled_is_scoped_to_explicit_easy7b(self):
        from reproduction.easy2560_joint4.test_common_contracts import easy_run
        run=easy_run(8)
        run.update(model_revision='a09a35458c702b33eeacc393d103063234e8bc28',
            budget=dict(model_id='Qwen/Qwen2.5-7B-Instruct',model_revision='a09a35458c702b33eeacc393d103063234e8bc28'),
            infra_optimizations=dict(selected_policy_logits=True,exact_retrieval_cache=True))
        self.assertTrue(all(flags(run).values()))
        with self.assertRaises(AssertionError):flags(dict(run,easy2560_selection=False))

    def test_reject_unknown_or_untyped_flags(self):
        for values in [dict(selected_policy_logits=True),dict(selected_policy_logits=1,exact_retrieval_cache=True)]:
            with self.assertRaises(ValueError):flags(dict(infra_optimizations=values))
