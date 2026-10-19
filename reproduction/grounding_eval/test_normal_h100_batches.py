"""CPU-only sealed-boundary recovery and actual-token accounting checks."""
import json
from pathlib import Path
import tempfile
import unittest
from reproduction.grounding_eval.normal_h100 import (
    sealed_intervals, validate_saved_rows, batch_token_totals, checkpoint_inventory)


class Batches(unittest.TestCase):
    def test_model_identity_excludes_mutable_transfer_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x':'model-01.safetensors'}}))
            for name in ('model-01.safetensors', 'complete.json', 'config.json', 'asset-status.json', 'asset-migration-complete.json'):
                (root / name).write_text('{}')
            inventory = checkpoint_inventory(root)
            self.assertIn('model-01.safetensors', inventory)
            self.assertIn('complete.json', inventory)
            self.assertNotIn('asset-status.json', inventory)
            self.assertNotIn('asset-migration-complete.json', inventory)

    def test_recover_previous_larger_batch_after_lower_execution_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for directory in ('raw', 'normalized'):
                (out / directory).mkdir()
                for start, end in ((0, 64), (64, 192), (192, 448)):
                    (out / directory / ('batch-%06d-%06d.json' % (start, end))).write_text('{}')
            self.assertEqual(sealed_intervals(out, 3610), {0: 64, 64: 192, 192: 448})
            # A recovery with live cap 128 must reuse the sealed 256-row batch.
            self.assertEqual(sealed_intervals(out, 3610).get(192, 192 + 128), 448)

    def test_raw_only_final_batch_is_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw'; path.mkdir()
            (path / 'batch-000000-000064.json').write_text('{}')
            self.assertEqual(sealed_intervals(folder, 128), {0: 64})

    def test_overlap_hole_normalized_only_and_out_of_budget_reject(self):
        cases = [('raw', ((0, 64), (32, 96))), ('raw', ((64, 128),)),
                 ('normalized', ((0, 64),)), ('raw', ((0, 129),))]
        for directory, intervals in cases:
            with self.subTest(directory=directory, intervals=intervals), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / directory; path.mkdir()
                for start, end in intervals:
                    (path / ('batch-%06d-%06d.json' % (start, end))).write_text('{}')
                with self.assertRaises(ValueError):
                    sealed_intervals(folder, 128)

    def test_binding_and_exact_occurrence_order_are_required(self):
        saved = dict(binding_sha256='bound', rows=[dict(evaluation_ordinal=i) for i in (0, 1)])
        validate_saved_rows(saved, 'bound', [0, 1])
        for binding, order in [('wrong', [0, 1]), ('bound', [1, 0]), ('bound', [0])]:
            with self.assertRaises(ValueError):
                validate_saved_rows(saved, binding, order)

    def test_batch_ledger_counts_raw_output_separately_from_retained(self):
        row = dict(raw_generated_tokens=50, model_generated_tokens=20, observation_tokens=30,
                   initial_prompt_tokens=10, trajectory_total_tokens=60, cumulative_model_input_tokens=100)
        totals = batch_token_totals([row, row])
        self.assertEqual(totals['raw_generated_tokens'], 100)
        self.assertEqual(totals['model_generated_tokens'], 40)
        self.assertEqual(totals['cumulative_model_input_tokens'], 200)


if __name__ == '__main__':
    unittest.main()
