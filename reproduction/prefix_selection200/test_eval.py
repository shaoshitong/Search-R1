"""Shared context/observation and constructor tests without GPU reruns."""
import unittest
from types import SimpleNamespace
from .eval import CONTRACT, build_evaluation_engine, evaluation_stream_snapshot, first_action_cut, stream_action_present


class UnifiedEvaluationTests(unittest.TestCase):
    def test_eval_and_rollout_share_the_interaction_contract(self):
        from .interaction_contract import INTERACTION_CONTRACT, BUDGET
        from . import rollout
        self.assertIs(rollout.first_action_cut, first_action_cut)
        self.assertIs(rollout.stream_action_present, stream_action_present)
        self.assertEqual(CONTRACT['compute_dtype'], 'bfloat16')
        self.assertEqual(CONTRACT['answer_reserve_tokens'], BUDGET['answer_reserve_tokens'])
        self.assertEqual(CONTRACT['max_model_tokens'], 4096)
        self.assertTrue(CONTRACT['early_final'])
        self.assertTrue(CONTRACT['asynchronous_generation'])
        for key, value in INTERACTION_CONTRACT.items(): self.assertEqual(CONTRACT[key], value)

    def test_stream_boundary_snapshots_metadata_before_abort(self):
        from .test_rollout import decode
        native = SimpleNamespace(token_ids=[ord(c) for c in '<answer>yes</answer>'],
            finish_reason=None, stop_reason=None)
        result = SimpleNamespace(request_id='r', outputs=[native], prompt_token_ids=[65], finished=False)
        aborted = []
        def abort(rid):
            aborted.append(rid); native.token_ids.clear(); native.finish_reason = 'abort'
        actual = evaluation_stream_snapshot(result, [65], [0], decode, abort)
        self.assertEqual(aborted, ['r'])
        self.assertIsNone(actual['finish_reason'])
        self.assertEqual(decode(actual['raw_generated_token_ids']), '<answer>yes</answer>')
        self.assertEqual(actual['prefix_cut']['action'], 'answer')
        self.assertTrue(actual['native_action_boundary_abort'])

    def test_partial_stream_waits_and_native_eos_finishes_without_tool(self):
        from .test_rollout import decode
        native = SimpleNamespace(token_ids=[ord(c) for c in '<search>incomplete'],
            finish_reason=None, stop_reason=None)
        result = SimpleNamespace(request_id='r', outputs=[native], prompt_token_ids=[65], finished=False)
        self.assertIsNone(evaluation_stream_snapshot(result, [65], [0], decode,
            lambda _: self.fail('A partial action must continue generation')))
        native.token_ids += [0]; native.finish_reason = 'stop'; result.finished = True
        actual = evaluation_stream_snapshot(result, [65], [0], decode,
            lambda _: self.fail('Native-finished request needs no abort'))
        self.assertTrue(actual['prefix_cut']['native_terminal'])
        self.assertIsNone(actual['prefix_cut']['action'])
        self.assertNotIn(0, actual['prefix_cut']['kept_token_ids'])

    def test_constructor_oom_alone_releases_and_lowers_cache_without_changing_bf16_recipe(self):
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
        self.assertTrue(all(call['dtype'] == 'bfloat16' and call['max_model_len'] == 4096 for call in calls))
        self.assertTrue(all(call['enable_chunked_prefill'] and call['enable_prefix_caching'] and
                            call['disable_async_output_proc'] for call in calls))
        def unknown(**kwargs): raise RuntimeError('unknown CUDA collective failure')
        with self.assertRaisesRegex(RuntimeError, 'unknown CUDA'):
            build_evaluation_engine({}, '/mnt/model', 1, 79 * 2**30, 80 * 2**30,
                factory=unknown, cleanup=lambda: self.fail('Unknown errors must not retry as OOM'))


if __name__ == '__main__': unittest.main()
