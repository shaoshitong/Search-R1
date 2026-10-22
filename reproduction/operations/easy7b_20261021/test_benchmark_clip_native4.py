"""Small CPU fixtures for the independent native4 clipping wrapper."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from reproduction.operations.easy7b_20261021 import benchmark_clip_native4 as benchmark
from search_g1 import gradient_clipping


class NativeClipComparisonTests(unittest.TestCase):
    @staticmethod
    def args(output):
        return argparse.Namespace(mode='leancompact', steps=20, selection_step=51,
            source_base='/remote/source', checkpoint='/remote/source/global_step_50', output=str(output))

    @staticmethod
    def complete(output):
        benchmark.native.write(output/'rank-0.json', dict(mode='leancompact', rank=0,
            actual_optimizer_writes=20, actual_optimizer_calls=20,
            source_binding={'selection_step': 51}))
        benchmark.native.write(output/'result.json', dict(mode='leancompact', actual_world_size=4,
            status='NATIVE4_REAL_ADAM_BENCHMARK_COMPLETED', steps=20))

    def test_cpu_actual_backward_twenty_adam_inputs_equal_and_wrapper_restored(self):
        original = gradient_clipping.clip_global_l2
        fixture = {}

        def native_run(args):
            actor = torch.nn.Linear(3, 2, bias=False)
            reference = torch.nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                actor.weight.copy_(torch.tensor([[.1, -.2, .3], [-.4, .5, -.6]]))
                reference.weight.copy_(actor.weight)
            optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
            inputs = torch.tensor([[2., -3., .5], [-1., .5, 4.]])
            for index in range(20):
                optimizer.zero_grad(); reference_optimizer.zero_grad()
                actor(inputs).square().sum().backward()
                reference(inputs).square().sum().backward()
                expected_norm = original(reference, 1., diagnostic=False,
                    check_finite=False, measure_post_norm=False)
                actual_norm = gradient_clipping.clip_global_l2(actor, 1., diagnostic=False,
                    check_finite=False, measure_post_norm=False, context={'minibatch_index': index})
                self.assertTrue(torch.equal(actual_norm, expected_norm))
                self.assertTrue(torch.equal(actor.weight.grad, reference.weight.grad))
                self.assertIsNone(actor._last_clipped_global_l2)
                optimizer.step(); reference_optimizer.step()
                self.assertTrue(torch.equal(actor.weight, reference.weight))
                for key, value in optimizer.state[actor.weight].items():
                    self.assertTrue(torch.equal(value, reference_optimizer.state[reference.weight][key]))
            fixture['steps'] = int(optimizer.state[actor.weight]['step'])
            self.complete(Path(args.output))

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.dict(benchmark.os.environ, {'RANK': '0'}), \
                 patch.object(benchmark.native, 'run', side_effect=native_run), \
                 patch.object(torch.cuda, 'synchronize') as sync, \
                 contextlib.redirect_stdout(io.StringIO()):
                result = benchmark.run(self.args(output))
            self.assertIs(gradient_clipping.clip_global_l2, original)
            self.assertEqual(fixture['steps'], 20)
            self.assertEqual(sync.call_count, 80)
            self.assertEqual(result['status'], 'NATIVE4_CLIP_COMPARISON_COMPLETE')
            self.assertTrue(result['all_twenty_calls_equal'])
            self.assertTrue(result['original_clipping_function_restored'])
            self.assertEqual(result['actual_clipping_calls'], 20)
            self.assertTrue(all(row['gradient_placement'] == {
                'cpu': {'parameters': 1, 'elements': 6, 'bytes': 24}} for row in result['calls']))
            self.assertTrue(all(row['new_clipped_gradients_left_for_adam'] for row in result['calls']))
            saved = json.loads((output/'clip-comparison-rank-0.json').read_text())
            self.assertEqual(saved['status'], result['status'])
            self.assertFalse(saved['tensor_files_saved'])
            self.assertFalse(list(output.glob('*.pt')))

    def test_presync_is_recorded_separately_from_body(self):
        events = []
        comparison = benchmark.ClipComparison(lambda *a, **kw: events.append(kw) or 'norm',
                                             lambda: events.append('sync'))
        with patch.object(benchmark.time, 'perf_counter', side_effect=[1., 3., 4., 9.]):
            result, timing = comparison.timed(object(), 1., {'context': {'outer': 51}}, fast=True)
        self.assertEqual(result, 'norm')
        self.assertEqual(timing, {'pre_sync_seconds': 2., 'body_wall_seconds': 5.})
        self.assertEqual(events, ['sync', dict(context={'outer': 51}, diagnostic=False,
            check_finite=False, measure_post_norm=False), 'sync'])

    def test_native_failure_restores_function_and_writes_partial_failure(self):
        original = gradient_clipping.clip_global_l2

        def failed(args):
            module = torch.nn.Linear(2, 1, bias=False)
            module.weight.grad = torch.tensor([[3., 4.]])
            gradient_clipping.clip_global_l2(module, 1.)
            raise RuntimeError('fixture native failure after first clip')

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.dict(benchmark.os.environ, {'RANK': '0'}), \
                 patch.object(benchmark.native, 'run', side_effect=failed), \
                 patch.object(torch.cuda, 'synchronize'), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(RuntimeError, 'fixture native failure'):
                benchmark.run(self.args(output))
            self.assertIs(gradient_clipping.clip_global_l2, original)
            saved = json.loads((output/'clip-comparison-rank-0.json').read_text())
            self.assertEqual(saved['status'], 'FAILED')
            self.assertEqual(saved['actual_clipping_calls'], 1)
            self.assertFalse(saved['all_twenty_calls_equal'])
            self.assertIsNone(saved['native_benchmark_completion'])
            self.assertTrue(saved['original_clipping_function_restored'])

    def test_native_completion_without_twenty_wrapper_calls_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.dict(benchmark.os.environ, {'RANK': '0'}), \
                 patch.object(benchmark.native, 'run', side_effect=lambda args: self.complete(output)), \
                 self.assertRaisesRegex(ValueError, 'twenty complete'):
                benchmark.run(self.args(output))
            saved = json.loads((output/'clip-comparison-rank-0.json').read_text())
            self.assertEqual(saved['status'], 'FAILED')
            self.assertEqual(saved['actual_clipping_calls'], 0)

    def test_only_leancompact_selection51_twenty_steps(self):
        for field, value in [('mode', 'legacy'), ('mode', 'leanexpanded'),
                             ('steps', 1), ('selection_step', 50)]:
            args = self.args(Path('/unused'))
            setattr(args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                benchmark.validate_args(args)


if __name__ == '__main__':
    unittest.main()
