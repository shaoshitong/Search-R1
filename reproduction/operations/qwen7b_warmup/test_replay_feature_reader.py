"""Small synthetic packing/reader and label-free downstream-drift regressions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .compare_replay_probe_outputs import comparison
from .replay_feature_reader import ReplayFeatureReader
from .replay_probe_layers import (CAPTURE_KIND, QMAX, REVISION, WIDTH, ReplayLayerWriter,
                                   canonical, fingerprints)


class ReplayReaderTests(unittest.TestCase):
    def fixture(self, root, layer=26):
        root = root.resolve()
        rows, values = [], {}
        for i, (qid, plen) in enumerate((('q0', 3), ('q0', 0), ('q1', 2))):
            row = dict(trajectory_id='t%d'%i, qid=qid, question_length=3,
                policy_length=plen, initial_prompt_token_ids=[11, 12, 13],
                policy_token_ids=list(range(21, 21+plen)), policy_position_ids=[3, 4, 8][:plen],
                source_prefix_sha256=hashlib.sha256(('source%d'%i).encode()).hexdigest())
            feature = torch.zeros(WIDTH, 2, dtype=torch.bfloat16)
            feature[:3] = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])+i*.125
            feature[QMAX:QMAX+plen] = torch.arange(plen*2).reshape(plen, 2)+10+i
            mask = torch.zeros(WIDTH, dtype=torch.bool)
            mask[:3] = True
            mask[QMAX:QMAX+plen] = True
            positions = torch.zeros(WIDTH, dtype=torch.long)
            positions[:3] = torch.arange(3)
            positions[QMAX:QMAX+plen] = torch.tensor(row['policy_position_ids'], dtype=torch.long)
            data = dict(features=feature, attention_mask=mask, position_ids=positions,
                question_length=3, policy_length=plen, capture_layer=layer, capture_kind=CAPTURE_KIND)
            rows.append(row)
            values[row['trajectory_id']] = data
        writer = ReplayLayerWriter(root/('layer%d'%layer), layer, rows, dimension=2)
        for row in rows:
            writer.write(row, values[row['trajectory_id']])
        return writer.finish(), values

    def test_lossless_get_dtype_masks_positions_metadata_and_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor, expected = self.fixture(Path(folder))
            reader = ReplayFeatureReader([descriptor], 26, synthetic_fixture=True, fixture_dimension=2)
            own = reader.get(['t2', 't0', 't1', 't0'], 'cpu')
            self.assertEqual(own['features'].dtype, torch.bfloat16)
            self.assertFalse(own['features'].requires_grad)
            self.assertEqual(own['capture_layer'], 26)
            self.assertEqual(own['capture_kind'], CAPTURE_KIND)
            for i, tid in enumerate(['t2', 't0', 't1', 't0']):
                for key in ('features', 'attention_mask', 'position_ids'):
                    self.assertTrue(torch.equal(own[key][i], expected[tid][key]))
            reader.close()
            with self.assertRaises(ValueError):
                reader.get(['t0'], 'cpu')

    def test_rejects_changed_files_wrong_layer_and_unbound_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor, _ = self.fixture(Path(folder))
            with self.assertRaises(ValueError):
                ReplayFeatureReader([descriptor], 23, synthetic_fixture=True, fixture_dimension=2)
            with self.assertRaises(ValueError):
                ReplayFeatureReader([descriptor['path']], 26, synthetic_fixture=True, fixture_dimension=2)
            path = Path(descriptor['path']).parent/'policy.bin'
            data = path.read_bytes()
            path.write_bytes(data[:-1]+bytes([data[-1]^1]))
            with self.assertRaises(ValueError):
                ReplayFeatureReader([descriptor], 26, synthetic_fixture=True, fixture_dimension=2)

    def test_other_qualified_layers_are_read_with_their_explicit_capture_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            for layer in (20, 23):
                descriptor, expected = self.fixture(Path(folder), layer=layer)
                reader = ReplayFeatureReader([descriptor], layer, synthetic_fixture=True, fixture_dimension=2)
                try:
                    own = reader.get(['t0'], 'cpu')
                    self.assertEqual(own['capture_layer'], layer)
                    self.assertTrue(torch.equal(own['features'][0], expected['t0']['features']))
                finally:
                    reader.close()

    def test_wrong_capture_kind_or_declared_dtype_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor, _ = self.fixture(Path(folder))
            path = Path(descriptor['path'])
            original = json.loads(path.read_text())
            for key, value in (('capture_kind', 'pre_block'), ('feature_dtype', 'float16_bits_uint16')):
                changed = dict(original)
                changed[key] = value
                path.write_bytes(canonical(changed)+b'\n')
                with self.assertRaises(ValueError):
                    ReplayFeatureReader([fingerprints(path)], 26, synthetic_fixture=True, fixture_dimension=2)

    def test_rejects_bad_index_positions_even_with_resealed_file_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor, _ = self.fixture(Path(folder))
            path = Path(descriptor['path'])
            manifest = json.loads(path.read_text())
            index = path.parent/'index.jsonl'
            rows = [json.loads(line) for line in index.read_text().splitlines()]
            rows[0]['policy_positions'][0] = 2
            index.write_bytes(b''.join(canonical(row)+b'\n' for row in rows))
            manifest['files']['index.jsonl'] = fingerprints(index)
            path.write_bytes(canonical(manifest)+b'\n')
            with self.assertRaises(ValueError):
                ReplayFeatureReader([fingerprints(path)], 26, synthetic_fixture=True, fixture_dimension=2)

    def test_retained_feature_sha_is_rechecked_on_every_get(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor, _ = self.fixture(Path(folder))
            reader = ReplayFeatureReader([descriptor], 26, synthetic_fixture=True, fixture_dimension=2)
            path = Path(descriptor['path']).parent/'policy.bin'
            data = path.read_bytes()
            path.write_bytes(bytes([data[0]^1])+data[1:])
            try:
                with self.assertRaises(ValueError):
                    reader.get(['t0'], 'cpu')
            finally:
                reader.close()


class DownstreamDriftMathTests(unittest.TestCase):
    def ordered(self):
        return [dict(trajectory_id='q%d-r%d'%(q, i), qid='q%d'%q) for q in range(3) for i in range(5)]

    def test_identical_backend_has_zero_drift_and_no_rank_or_threshold_changes(self):
        ordered = self.ordered()
        values = {row['trajectory_id']:[.01, .2, .5][i//5] for i,row in enumerate(ordered)}
        result = comparison(values, dict(values), ordered)
        self.assertEqual(result['trajectory_probabilities']['max_absolute_difference'], 0.)
        self.assertEqual(result['mixed_probabilities']['threshold_changes'], 0)
        self.assertEqual(result['ranking']['strict_pair_order_inversions'], 0)
        self.assertEqual(result['ranking']['questions_changing_deterministic_rank'], 0)
        self.assertAlmostEqual(result['ranking']['spearman_with_average_ties'], 1.)
        self.assertFalse(result['labels_read'])

    def test_known_feature_drift_reports_fixed_threshold_flips_and_rank_inversion(self):
        ordered = self.ordered()
        before = {row['trajectory_id']:[.01, .2, .5][i//5] for i,row in enumerate(ordered)}
        after = {row['trajectory_id']:[.5, .01, .6][i//5] for i,row in enumerate(ordered)}
        result = comparison(before, after, ordered)
        self.assertEqual(result['mixed_probabilities']['threshold_changes'], 2)
        self.assertEqual(result['trajectory_probabilities']['threshold_changes'], 5)
        self.assertGreater(result['ranking']['strict_pair_order_inversions'], 0)
        self.assertFalse(result['threshold_fitted'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
