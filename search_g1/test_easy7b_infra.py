import unittest
from search_g1.easy7b_infra import flags, profile_flags


class InfraFlagsTest(unittest.TestCase):
    def test_profile_launch_keeps_geometry_and_recipe(self):
        import tempfile
        from reproduction.easy2560_joint4.control import BASE, launch_command
        from reproduction.easy2560_joint4.prepare_training import selection_settings
        original = launch_command(BASE, 'a'*40, 8)
        for profile in ('sparse-exact-v1', 'exact-cache-v1'):
            candidate = launch_command(BASE, 'a'*40, 8, infra_profile=profile)
            self.assertEqual(candidate[:-1], original)
            self.assertEqual(candidate[-1], profile)
            with tempfile.TemporaryDirectory() as td:
                baseline = selection_settings(td, 8)
                optimized = selection_settings(td, 8, infra_profile=profile)
            self.assertEqual(optimized.pop('infra_optimizations'), profile_flags(profile))
            self.assertEqual(optimized, baseline)
        self.assertEqual(profile_flags('exact-cache-v1'),
                         dict(selected_policy_logits=False, exact_retrieval_cache=True))
        with self.assertRaises(ValueError):profile_flags('unknown')

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
        run['infra_optimizations'] = profile_flags('exact-cache-v1')
        self.assertEqual(flags(run), dict(selected_policy_logits=False, exact_retrieval_cache=True))
        with self.assertRaises(AssertionError):flags(dict(run,easy2560_selection=False))

    def test_reject_unknown_or_untyped_flags(self):
        for values in [dict(selected_policy_logits=True),dict(selected_policy_logits=1,exact_retrieval_cache=True)]:
            with self.assertRaises(ValueError):flags(dict(infra_optimizations=values))
