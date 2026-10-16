"""CPU tests for bounded actual-entrypoint observations; no model admission."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from reproduction.common8 import model_smoke as smoke


def data_fixture():
    tokens = torch.arange(12, dtype=torch.int64).reshape(2, 6)
    return SimpleNamespace(batch={key: tokens.clone() for key in smoke.LOGPROB_INPUTS},
        meta_info=dict(micro_batch_size=8, temperature=1., use_dynamic_bsz=False,
                       max_token_len=16384, do_sample=False, common8_model_smoke=True))


class FakeActor:
    def __init__(self):
        self.actor_module = torch.nn.Linear(2, 2)
        self.calls = 0
        self.error = None

    def compute_log_prob(self, data):
        self.calls += 1
        self.actor_module.eval()
        if self.error:
            raise self.error
        return data.batch['responses'].float() * 0


class LogprobObservationTests(unittest.TestCase):
    def test_two_original_calls_only_preserve_inputs_rng_and_restore_method(self):
        actor, data, rows = FakeActor(), data_fixture(), []
        original = actor.compute_log_prob
        before = {k: v.clone() for k, v in data.batch.items()}
        rng = torch.get_rng_state().clone()
        meta = copy.deepcopy(data.meta_info)
        with smoke.observe_logprob_entrypoints(actor, rows):
            first = actor.compute_log_prob(data)
            second = actor.compute_log_prob(data=data)
        self.assertEqual(actor.calls, 2)
        self.assertEqual(actor.compute_log_prob, original)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(meta, data.meta_info)
        for key in before:
            self.assertTrue(torch.equal(before[key], data.batch[key]))
            a, b = rows[0]['entry']['inputs'][key], rows[1]['entry']['inputs'][key]
            self.assertEqual(a, b)
            self.assertEqual(a['sha256'], hashlib.sha256(before[key].numpy().tobytes()).hexdigest())
            self.assertEqual(set(a), {'sha256', 'shape', 'stride', 'dtype', 'device', 'storage_offset'})
        self.assertTrue(rows[0]['entry']['module_training'])
        self.assertFalse(rows[1]['entry']['module_training'])
        self.assertTrue(all(r['forward_completed'] and not r['module_training_after_forward'] for r in rows))

    def test_noncontiguous_layout_is_observed_without_modification(self):
        data, actor = data_fixture(), FakeActor()
        data.batch['responses'] = data.batch['responses'].t()
        observed = smoke.logprob_entry_snapshot(data, actor.actor_module)['inputs']['responses']
        self.assertEqual(observed['shape'], [6, 2])
        self.assertEqual(observed['stride'], list(data.batch['responses'].stride()))
        self.assertFalse(data.batch['responses'].is_contiguous())

    def test_forward_exception_and_observation_failure_do_not_repeat_forward(self):
        actor, data, rows = FakeActor(), data_fixture(), []
        original = actor.compute_log_prob
        actor.error = RuntimeError('original forward failed')
        with self.assertRaisesRegex(RuntimeError, 'original forward failed'):
            with smoke.observe_logprob_entrypoints(actor, rows):
                actor.compute_log_prob(data)
        self.assertEqual(actor.calls, 1)
        self.assertEqual(actor.compute_log_prob, original)
        self.assertEqual(rows[0]['forward_error'], 'RuntimeError')
        actor.error = None
        rows = []
        with patch.object(smoke, 'logprob_entry_snapshot', side_effect=OSError('observation failed')):
            with smoke.observe_logprob_entrypoints(actor, rows):
                actor.compute_log_prob(data)
        self.assertEqual(actor.calls, 2)
        self.assertIn('observation_error', rows[0])
        proof = smoke.internal_logprob_diagnostic(torch.zeros(2, 6), torch.zeros(2, 6),
            torch.ones(2, 6, dtype=torch.bool), ['q0', 'q1'], rows)
        self.assertTrue(proof['local_accepted'])
        self.assertFalse(proof['observation_complete'])


class InternalLogprobDiagnosticTests(unittest.TestCase):
    def proof(self, first, second, mask=None):
        actor, data, rows = FakeActor(), data_fixture(), []
        with smoke.observe_logprob_entrypoints(actor, rows):
            actor.compute_log_prob(data)
            actor.compute_log_prob(data)
        if mask is None:
            mask = torch.ones_like(first, dtype=torch.bool)
        return smoke.internal_logprob_diagnostic(first, second, mask,
            ['question-' + str(i) for i in range(first.shape[0])], rows)

    def test_finite_gate_is_unchanged_and_top_eight_coordinates_have_question_ids(self):
        first = torch.zeros(2, 6)
        second = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 100
        proof = self.proof(first, second)
        self.assertFalse(proof['local_accepted'])
        self.assertEqual(proof['reason'], 'EXCEEDS_LOGPROB_ATOL')
        self.assertEqual(proof['tolerance'], .05)
        self.assertEqual(len(proof['top_errors']), 8)
        top = proof['top_errors'][0]
        self.assertEqual((top['row'], top['response_token_offset'], top['question_id']), (1, 5, 'question-1'))
        self.assertAlmostEqual(proof['max_abs'], .11)
        self.assertAlmostEqual(proof['mean_abs'], .055)
        self.assertTrue(proof['observation_complete'])
        self.assertTrue(proof['input_values_equal'])
        self.assertLess(len(json.dumps(proof, allow_nan=False)), 64 * 1024)
        for delta in (0., .049, .051):
            result = self.proof(first, first + delta)
            self.assertEqual(result['local_accepted'], delta <= .05)

    def test_empty_nonfinite_and_masked_padding_are_distinguished(self):
        first = torch.zeros(2, 6)
        second = first.clone()
        empty = self.proof(first, second, torch.zeros_like(first, dtype=torch.bool))
        self.assertFalse(empty['local_accepted'])
        self.assertEqual(empty['reason'], 'EMPTY_VALID_TOKENS')
        self.assertEqual(empty['valid_tokens'], 0)
        self.assertIsNone(empty['max_abs'])
        for invalid in (float('nan'), float('inf')):
            second[1, 2] = invalid
            result = self.proof(first, second)
            self.assertFalse(result['local_accepted'])
            self.assertEqual(result['reason'], 'NONFINITE_VALID_TOKEN_DIFFERENCE')
            self.assertEqual(result['nonfinite_differences'], 1)
            self.assertIsNone(result['max_abs'])
            self.assertIsNone(result['top_errors'][0]['abs_error'])
            json.dumps(result, allow_nan=False)
            mask = torch.ones_like(first, dtype=torch.bool)
            mask[1, 2] = False
            self.assertTrue(self.proof(first, second, mask)['local_accepted'])

    def test_immutable_fsync_receipt_and_publish_failure_return_false_with_console_evidence(self):
        value = self.proof(torch.zeros(2, 6), torch.zeros(2, 6))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rank-0.json'
            with patch.object(smoke.os, 'fsync', wraps=smoke.os.fsync) as fsync:
                self.assertTrue(smoke.publish_logprob_diagnostic(path, value))
                self.assertEqual(fsync.call_count, 2)
            before = path.read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertFalse(smoke.publish_logprob_diagnostic(path, dict(value, reason='changed')))
            self.assertEqual(path.read_bytes(), before)
            self.assertIn('COMMON8_MODEL_INTERNAL_LOGPROB_DIAGNOSTIC_PUBLISH_FAILED=', output.getvalue())
            failing = Path(directory) / 'rank-1.json'
            with redirect_stdout(output), patch.object(smoke.os, 'fsync', side_effect=OSError('cannot sync')):
                self.assertFalse(smoke.publish_logprob_diagnostic(failing, value))
            self.assertIn('cannot sync', output.getvalue())


if __name__ == '__main__':
    unittest.main()
