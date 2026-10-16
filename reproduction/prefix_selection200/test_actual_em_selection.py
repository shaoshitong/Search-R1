"""CPU checks for periodic actual-mixed selection and its independent lineage."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .contract import CANDIDATE_QUESTIONS, G, SELECTED_QUESTIONS, WORLD
from .periodic_data import actual_em_selection, project_top512, select_actual_mixed_groups
from reproduction.prefix20_online.pool import canonical, digest
from .rollout import file_item, iter_rows


def candidates(mixed_questions):
    groups, labels, prefixes = [], [], {}
    provenance = dict(actor_version='actor0', prefix_scan_sha256='scan', pool_selection_sha256='pool')
    for i in range(CANDIDATE_QUESTIONS):
        tids = ['q%d-g%d' % (i, j) for j in range(G)]
        groups.append(dict(qid='q%d' % i, question_hash='%064x' % i, candidate_ordinal=i,
            trajectory_ids=tids, q=i / CANDIDATE_QUESTIONS))
        for j, tid in enumerate(tids):
            prefixes[tid] = '%064x' % (CANDIDATE_QUESTIONS + i * G + j)
            labels.append(dict(trajectory_id=tid, qid='q%d' % i,
                strict_em=int(i < mixed_questions and j == 0), actor_version='actor0',
                prefix_scan_sha256='scan', selection_sha256='pool', prefix_sha256=prefixes[tid]))
    return groups, labels, prefixes, provenance


def select(groups, labels, prefixes, provenance, mode='probe_score_fill'):
    return select_actual_mixed_groups(groups, labels, fallback_mode=mode,
        provenance=provenance, expected_prefixes=prefixes)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n')


class ActualMixedSelectionTest(unittest.TestCase):
    def test_higher_scored_nonmixed_cannot_replace_actual_mixed(self):
        groups, labels, prefixes, provenance = candidates(600)
        selected, stats = select(groups, labels, prefixes, provenance)
        self.assertEqual([r['candidate_ordinal'] for r in selected], list(range(599, 87, -1)))
        self.assertEqual(stats['actual_mixed_candidate_count'], 600)
        self.assertEqual(stats['chosen_mixed_count'], SELECTED_QUESTIONS)
        self.assertEqual(stats['fallback_count'], 0)
        self.assertTrue(stats['all_selected_are_actual_mixed'])
        self.assertEqual(stats['actual_candidate_correct_count_histogram'][1], 600)

    def test_shortfall_retains_all_mixed_then_explicit_predicted_fill(self):
        groups, labels, prefixes, provenance = candidates(511)
        selected, stats = select(groups, labels, prefixes, provenance)
        self.assertEqual([r['candidate_ordinal'] for r in selected[:-1]], list(range(510, -1, -1)))
        self.assertEqual(selected[-1]['candidate_ordinal'], CANDIDATE_QUESTIONS - 1)
        self.assertEqual(stats['chosen_mixed_count'], 511)
        self.assertEqual(stats['fallback_count'], 1)
        self.assertFalse(stats['all_selected_are_actual_mixed'])
        self.assertEqual(stats['selection_mode'], 'actual_mixed_then_probe_score_fill')
        with self.assertRaisesRegex(ValueError, 'no implicit fill'):
            select(groups, labels, prefixes, provenance, mode='require_512_mixed')
        with self.assertRaisesRegex(ValueError, 'explicit'):
            select(groups, labels, prefixes, provenance, mode=None)

    def test_ties_use_original_ordinal_and_k5_is_not_mixed(self):
        groups, labels, prefixes, provenance = candidates(513)
        for row in groups:
            row['q'] = .7
        # One all-correct G5 must be excluded despite having successful answers.
        for row in labels[:G]:
            row['strict_em'] = 1
        selected, stats = select(list(reversed(groups)), labels, prefixes, provenance)
        self.assertEqual([r['candidate_ordinal'] for r in selected], list(range(1, 513)))
        self.assertEqual(stats['actual_candidate_correct_count_histogram'][5], 1)
        self.assertEqual(stats['actual_mixed_candidate_count'], 512)

    def test_labels_cannot_be_replaced_duplicated_or_partially_missing(self):
        groups, labels, prefixes, provenance = candidates(600)
        for field, wrong in (('actor_version', 'other'), ('prefix_scan_sha256', 'other'),
                             ('selection_sha256', 'other'), ('prefix_sha256', 'other'),
                             ('qid', 'other'), ('strict_em', True)):
            altered = list(labels); altered[0] = dict(labels[0], **{field: wrong})
            with self.assertRaisesRegex(ValueError, 'Actual EM label'):
                select(groups, altered, prefixes, provenance)
        with self.assertRaisesRegex(ValueError, 'Actual EM label'):
            select(groups, labels + [labels[0]], prefixes, provenance)
        with self.assertRaisesRegex(ValueError, 'all12800'):
            select(groups, labels[:-1], prefixes, provenance)

    def test_ordinary_and_final200_never_read_own_em_for_selection(self):
        with patch('reproduction.prefix_selection200.periodic_data.read',
                   side_effect=AssertionError('Ordinary actor selection must not open EM metadata')):
            for outer in (1, 9, 11, 199, 200):
                with self.assertRaisesRegex(ValueError, 'Only periodic'):
                    actual_em_selection('/not-an-actual-config', outer)

    def test_projection_binds_final_actual_selection_preserving_predicted_manifest(self):
        groups, labels, prefixes, provenance = candidates(600)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); root = base / 'outers/outer-000010'
            prefix = dict(run_id='run', outer=10, actor_version='actor0', scan_sha256='scan')
            write_json(root / 'prefix/complete.json', prefix)
            common = dict(run_id='run', outer=10, actor_version='actor0', prefix_scan_sha256='scan')
            pool_path = root / 'probe-score/pool-selection.json'
            predicted_path = root / 'probe-score/selection.json'
            write_json(pool_path, dict(common, selected=groups, selected_questions=CANDIDATE_QUESTIONS))
            write_json(predicted_path, dict(common, selected=list(reversed(groups))[:SELECTED_QUESTIONS],
                                          selected_questions=SELECTED_QUESTIONS, outcome_labels_used=False))
            for row in labels:
                row['selection_sha256'] = digest(pool_path)
            index_path = root / 'probe-score/index.json'
            write_json(index_path, [dict(trajectory_id=tid, prefix_sha256=value) for tid, value in prefixes.items()])
            manifest_path = root / 'probe-score/manifest.json'
            write_json(manifest_path, dict(index_sha256=digest(index_path), selection_sha256=digest(predicted_path),
                                          prefix_scan_sha256='scan'))
            before_predicted, before_manifest = predicted_path.read_bytes(), manifest_path.read_bytes()
            full_dir = root / 'full-pool'; full_dir.mkdir()
            own = full_dir / 'own-em.jsonl'
            own.write_text(''.join(json.dumps(row) + '\n' for row in labels))
            shards = []
            for rank in range(WORLD):
                path = full_dir / ('rank-%d-full.jsonl.gz' % rank)
                with gzip.open(path, 'wt') as output:
                    for row in labels:
                        ordinal = int(row['qid'][1:])
                        if ordinal % WORLD == rank:
                            output.write(json.dumps(dict(row, candidate_ordinal=ordinal)) + '\n')
                shards.append(file_item(path, CANDIDATE_QUESTIONS * G // WORLD))
            write_json(full_dir / 'complete.json', dict(common, questions=CANDIDATE_QUESTIONS,
                trajectories=CANDIDATE_QUESTIONS * G, selection_sha256=digest(pool_path),
                own_em=file_item(own, len(labels)), shards=shards))
            config = base / 'config.json'
            write_json(config, dict(base=str(base), run_id='run', actor_version='actor0', actor_model_dir='/official',
                                    actual_em_selection_fallback='probe_score_fill'))
            final_path = actual_em_selection(config, 10)
            complete = project_top512(config, 10, final_path)
            self.assertEqual(final_path.name, 'actual-em-selection.json')
            self.assertEqual(predicted_path.read_bytes(), before_predicted)
            self.assertEqual(manifest_path.read_bytes(), before_manifest)
            self.assertEqual(complete['actor_selection_path'], str(final_path.resolve()))
            self.assertEqual(complete['actor_selection_sha256'], digest(final_path))
            self.assertNotEqual(complete['selection_sha256'], digest(predicted_path))
            self.assertEqual(complete['actual_selected_mixed_questions'], 512)
            self.assertEqual(complete['fallback_count'], 0)
            for shard in complete['shards']:
                for row in iter_rows(shard['path']):
                    self.assertEqual(row['selection_sha256'], digest(final_path))
                    for key in ('full_pool_origin_selection_sha256', 'full_pool_complete_sha256',
                                'full_pool_own_em_sha256', 'predicted_selection_sha256'):
                        self.assertEqual(row[key], complete[key])
            for label in iter_rows(complete['own_em']['path']):
                self.assertEqual(label['selection_sha256'], digest(final_path))
                self.assertEqual(label['full_pool_origin_selection_sha256'], digest(pool_path))


if __name__ == '__main__':
    unittest.main()
