"""Legacy context/observation invariants without GPU or evaluation reruns."""
import unittest
from .eval import CONTRACT, build_evaluation_engine, legacy_final_turn, legacy_observation_tokens


class LegacyEvaluationTests(unittest.TestCase):
    def test_raw_observation_clip_does_not_repair_tags(self):
        class Tokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {'input_ids': list(range(600))}
        self.assertEqual(legacy_observation_tokens(Tokenizer(), 'tool'), list(range(500)))

    def test_final_turn_only_after_four_tool_turns(self):
        self.assertEqual([legacy_final_turn(i) for i in range(5)], [False] * 4 + [True])
        self.assertEqual(CONTRACT['compute_dtype'], 'float32')
        self.assertEqual(CONTRACT['answer_reserve_tokens'], 0)
        self.assertFalse(CONTRACT['early_final'])
        self.assertTrue(CONTRACT['asynchronous_generation'])

    def test_constructor_oom_alone_releases_and_lowers_cache_without_changing_fp32_recipe(self):
        import torch
        calls, cleanups = [], []
        sentinel = object()
        def factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1: raise torch.cuda.OutOfMemoryError('actual mocked constructor CUDA OOM')
            return sentinel
        llm, util, limits, failed = build_evaluation_engine({}, '/mnt/model', 1, 79 * 2**30, 80 * 2**30,
            factory=factory, cleanup=lambda: cleanups.append(True))
        self.assertIs(llm, sentinel); self.assertEqual(cleanups, [True])
        self.assertEqual(calls[0]['gpu_memory_utilization'], .95)
        self.assertEqual(calls[0]['max_num_seqs'], 1024)
        self.assertEqual(calls[1]['max_num_seqs'], 512)
        self.assertEqual(util, .95 * .85)
        self.assertEqual(limits['max_num_batched_tokens'], 16384)
        self.assertTrue(failed[0]['actual_cuda_oom'])
        self.assertTrue(all(call['dtype'] == 'float32' and call['max_model_len'] == 4596 for call in calls))
        self.assertTrue(all(call['enable_chunked_prefill'] is False and call['enable_prefix_caching'] is False for call in calls))
        def unknown(**kwargs): raise RuntimeError('unknown CUDA collective failure')
        with self.assertRaisesRegex(RuntimeError, 'unknown CUDA'):
            build_evaluation_engine({}, '/mnt/model', 1, 79 * 2**30, 80 * 2**30,
                factory=unknown, cleanup=lambda: self.fail('Unknown errors must not retry as OOM'))


if __name__ == '__main__': unittest.main()
