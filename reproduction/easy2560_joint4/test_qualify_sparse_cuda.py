"""CPU-only loader/guard regression; no HF, model load, CUDA, or remote I/O."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from reproduction.easy2560_joint4.qualify_sparse_cuda import block_updates, prepare_micro_batch
from reproduction.easy2560_joint4 import qualify_sparse_cuda as entry


def fixture():
    labels = [[0] * 5, [1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1] * 5,
              [1, 1, 1, 0, 0], [1, 0, 1, 1, 1], [0, 0, 0, 0, 1],
              [1, 0, 0, 1, 0], [0, 1, 1, 1, 0], [1, 0, 1, 1, 1]]
    rows, width, response_width = len(labels) * 5, 13, 9
    ids = torch.arange(rows * width, dtype=torch.long).reshape(rows, width)
    attention = torch.ones_like(ids)
    attention[::2, :2] = 0
    attention[::3, -1] = 0
    info = attention.clone()
    info[:, -response_width:] = 0
    for column in (0, 3, 5, 7):
        info[:, -response_width + column] = attention[:, -response_width + column]
    batch = dict(input_ids=ids, responses=ids[:, -response_width:].clone(),
        attention_mask=attention, position_ids=(attention.cumsum(-1) - 1).clamp(min=0),
        info_mask=info,
        sampling_log_probs=-torch.arange(rows * response_width, dtype=torch.float32).reshape(rows, response_width)/100)
    return dict(output=SimpleNamespace(batch=batch), strict_correctness=labels,
        ordinals=list(range(100, 100 + len(labels))),
        identity=dict(run_id='fixture', logical_update=1, policy_snapshot_id='original-pre-update-policy'))


class RawPoolTests(unittest.TestCase):
    def test_actual_context_mask_lp_and_group_normalization_on_all_four_ranks(self):
        raw = fixture()
        original = {name: value.clone() for name, value in raw['output'].batch.items()}
        labels = torch.tensor(raw['strict_correctness'], dtype=torch.float32)
        expected = ((labels - labels.mean(1, keepdim=True)) /
                    (labels.std(1, correction=1, keepdim=True) + 1e-6)).flatten()
        selected = [row for group, values in enumerate(raw['strict_correctness'])
                    if 0 < sum(values) < 5 for row in range(group * 5, group * 5 + 5)][:32]
        rng = torch.get_rng_state().clone()
        recovered_rows = []
        for rank in range(4):
            micro, proof = prepare_micro_batch(raw, rank)
            indices = selected[rank * 8:(rank + 1) * 8]
            recovered_rows.extend(proof['selected_pool_rows'])
            self.assertEqual(proof['global_selected_pool_rows'], selected)
            self.assertEqual(proof['raw_pool_identity']['policy_snapshot_id'], 'original-pre-update-policy')
            self.assertIn('NOT_production_grounding_rewards', proof['advantages']['role'])
            for name in ('input_ids', 'responses', 'attention_mask', 'position_ids'):
                torch.testing.assert_close(micro[name], original[name][indices], rtol=0, atol=0)
            torch.testing.assert_close(micro['old_log_probs'], original['sampling_log_probs'][indices], rtol=0, atol=0)
            mask = original['info_mask'][indices, -9:]
            torch.testing.assert_close(micro['loss_mask'], mask, rtol=0, atol=0)
            self.assertTrue(bool((mask.sum(1) == 4).all()))
            self.assertTrue(bool((mask[:, 1:3] == 0).all()))  # actual intervening tool positions
            torch.testing.assert_close(micro['advantages'], expected[indices, None].expand(-1, 9) * mask,
                                       rtol=0, atol=0)
            for row, record in enumerate(proof['rows']):
                self.assertEqual(record['question_ordinal'], 100 + indices[row] // 5)
                self.assertNotIn(record['correct_count'], (0, 5))
            micro['input_ids'][0, 0] = -99
            torch.testing.assert_close(raw['output'].batch['input_ids'], original['input_ids'], rtol=0, atol=0)
        self.assertEqual(recovered_rows, selected)
        self.assertEqual(len(set(recovered_rows)), 32)
        self.assertNotEqual(len(recovered_rows) % 5, 0)  # last group deliberately partial
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_partial_group_uses_five_labels_not_two_selected_members(self):
        raw = fixture()
        micro, proof = prepare_micro_batch(raw, 3)
        self.assertEqual(proof['selected_pool_rows'][-2:], [40, 41])
        # Last selected group's labels [0,1,1,1,0]: selected [0,1] alone
        # would have a different mean/std and therefore different advantages.
        original_five = torch.tensor([0., 1., 1., 1., 0.])
        expected = (original_five - original_five.mean()) / (original_five.std() + 1e-6)
        got = micro['advantages'][-2:, 0]
        torch.testing.assert_close(got, expected[:2], rtol=0, atol=0)
        subset = original_five[:2]
        wrong = (subset - subset.mean()) / (subset.std() + 1e-6)
        self.assertFalse(torch.allclose(got, wrong))

    def test_rejects_missing_binary_labels_mask_padding_and_wrong_geometry(self):
        raw = fixture(); raw['strict_correctness'][1][0] = 0.2
        with self.assertRaisesRegex(ValueError, 'FIVE_BINARY_EM'):
            prepare_micro_batch(raw, 0)
        raw = fixture(); raw['strict_correctness'] = [[0] * 5] * 10
        with self.assertRaisesRegex(ValueError, 'AT_LEAST_32'):
            prepare_micro_batch(raw, 0)
        raw = fixture(); raw['output'].batch['info_mask'][6, -1] = 1
        with self.assertRaisesRegex(ValueError, 'INCLUDES_PADDING'):
            prepare_micro_batch(raw, 0)
        raw = fixture(); raw['output'].batch['sampling_log_probs'] = torch.zeros(50, 8)
        with self.assertRaisesRegex(ValueError, 'CONTEXT_SHAPES'):
            prepare_micro_batch(raw, 0)
        with self.assertRaisesRegex(ValueError, 'FOUR_BY_EIGHT'):
            prepare_micro_batch(fixture(), 0, world_size=8)


class UpdateGuardTests(unittest.TestCase):
    def test_zero_updates_and_both_forbidden_calls_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        saved = copy.deepcopy(scheduler.state_dict())
        worker = SimpleNamespace(actor_optimizer=optimizer, actor_lr_scheduler=scheduler)
        counters = block_updates(worker)
        self.assertEqual(counters, dict(optimizer_step=0, scheduler_step=0))
        self.assertEqual(scheduler.state_dict(), saved)
        with self.assertRaisesRegex(RuntimeError, 'optimizer_step'):
            optimizer.step()
        with self.assertRaisesRegex(RuntimeError, 'scheduler_step'):
            scheduler.step()
        self.assertEqual(counters, dict(optimizer_step=1, scheduler_step=1))
        self.assertEqual(scheduler.state_dict(), saved)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(parameter.item(), 2.)

    def test_independent_process_failure_is_recorded_and_propagates_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'diagnostics/result.json'
            with mock.patch.object(entry, 'run', side_effect=RuntimeError('injected CUDA OOM')) as operation:
                with mock.patch.dict('os.environ', {'RANK': '2'}):
                    with self.assertRaisesRegex(RuntimeError, 'injected CUDA OOM'):
                        entry.main(['--base', folder, '--raw-pool-file', '/unused/pool.pt',
                                    '--output', str(output)])
            self.assertEqual(operation.call_count, 1)
            self.assertFalse(output.exists())
            failure = json.loads(output.with_name('result-rank-2-failure.json').read_text())
            self.assertFalse(failure['accepted'])
            self.assertFalse(failure['worker_reuse_permitted'])
            self.assertFalse(failure['formal_actual_restore_receipt_written'])
            self.assertEqual(failure['error'], 'injected CUDA OOM')


if __name__ == '__main__':
    unittest.main()
