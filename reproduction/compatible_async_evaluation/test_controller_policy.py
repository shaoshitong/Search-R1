"""Pure policy/loader contract tests; no torch, model, or GPU execution."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from reproduction.compatible_async_evaluation.async_controller import (
    content_policy, request_policy, check_response_append)
from reproduction.compatible_async_evaluation.worker_runtime import (
    fp32_loader_options, phase_profile, PROFILE)


class CompatibleContentPolicyTest(unittest.TestCase):
    def test_no_context_early_final_and_extra_final_generation(self):
        self.assertFalse(request_policy(4096, 3))
        self.assertTrue(request_policy(4096, 4))
        with self.assertRaisesRegex(ValueError, 'rolling_window'):
            request_policy(4097, 3)

    def test_response_cap_excludes_prompt_and_never_silently_crops(self):
        check_response_append(3500, 500, 96)
        with self.assertRaisesRegex(ValueError, 'response_prefix_budget'):
            check_response_append(3500, 500, 97)
        # Old final append permits this; a subsequent total request would fail.
        check_response_append(2500, 500, 0)
        with self.assertRaises(ValueError):
            request_policy(2048+3000, 3)

    def test_content_dimensions_and_no_answer_reserve(self):
        recipe = SimpleNamespace(max_start_length=2048,max_prompt_length=4096,
            max_response_length=500,max_obs_length=500,max_turns=4)
        value = content_policy(recipe)
        self.assertIsNone(value['answer_reserve_tokens'])
        self.assertEqual(value['role'], 'old-context-fp32-async')
        recipe.max_obs_length = 499
        with self.assertRaises(ValueError):
            content_policy(recipe)

    def test_loader_explicit_fp32_without_optimizer(self):
        old = {'model_dtype':'bf16','mixed_precision':{'param_dtype':'bf16'}}
        options = fp32_loader_options(old)
        self.assertEqual(old['model_dtype'], 'bf16')
        self.assertEqual(options['fsdp_config']['model_dtype'], 'fp32')
        self.assertTrue(all(v == 'fp32' for v in options['fsdp_config']['mixed_precision'].values()))
        self.assertIsNone(options['optim_config'])
        self.assertFalse(options['use_remove_padding'])

    def test_phase_profile_restores_after_failure(self):
        original = os.environ.get('SEARCH_G1_PROFILE')
        try:
            os.environ['SEARCH_G1_PROFILE'] = PROFILE
            with self.assertRaises(RuntimeError):
                with phase_profile():
                    self.assertEqual(os.environ['SEARCH_G1_PROFILE'], 'common8')
                    raise RuntimeError('actual phase failure')
            self.assertEqual(os.environ['SEARCH_G1_PROFILE'], PROFILE)
        finally:
            if original is None:os.environ.pop('SEARCH_G1_PROFILE',None)
            else:os.environ['SEARCH_G1_PROFILE']=original

    def test_isolated_controller_keeps_async_and_old_observation_api(self):
        path=Path(__file__).with_name('async_controller.py')
        source=path.read_text();ast.parse(source)
        self.assertIn('service.begin(snapshot)',source)
        self.assertIn('service.poll(timeout=0.01)',source)
        self.assertIn('_observation_ids(manager, [text])',source)
        self.assertNotIn('observation_room(',source)
        self.assertNotIn('final_stage(',source)
        self.assertNotIn('observation_tokens(',source)


if __name__ == '__main__':unittest.main()
