"""Small CPU fixtures: closed coverage, actual IDs, policy denominator and reuse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.recursive_bran_a.ablation150_aggregate_watch_v1.test_watch import Fixture, write
from release.recursive_bran_a.ablation150_tree_summary_v1 import collect


def tree(path, receipt, outer=5, question='question-fixture', root_mask=True):
    common = {key: receipt[key] for key in ('run_id', 'config_sha256', 'source_commit')}
    common.update(outer=outer, question_id=question)
    rows = [dict(event='question_start', root_id='tree-root', initial_reward=0, target=5, budget=10),
            dict(event='search_start', eligible=[dict(id='root', generation=0)], parent_id='root')]
    for attempt in range(1, 5):
        rows.extend([dict(event='branch_start', attempt=attempt, parent_id='root', action_index=attempt-1),
            dict(event='branch_end', attempt=attempt, status='complete', id='child'+str(attempt),
                reward=1 if attempt == 1 else 0, parent_reward=0, generation=1,
                source='contrast_accept' if attempt == 1 else 'same_reward',
                cost=dict(new_generated_tokens=3, model_calls=1, retrieval_calls=0,
                          branch_wall_seconds=2., full_training_model_tokens=6))])
    rows.append(dict(event='question_end', final_ids=['root']+['child'+str(i) for i in range(1, 5)],
        attempts=4, accepted_branches=1, same_reward_candidates=3, fill_count=3,
        complete_group=True, stop_reason='budget_exhausted'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(dict(row, event_seq=index, **common))+'\n'
                            for index, row in enumerate(rows, 1)))
    if root_mask:
        write(path.parent/'root.json', dict(state=dict(trajectory='root', tree='tree-root',
            response=[7, 8, 9, 10, 11], model_mask=[1, 1, 0, 1, 1])))
    return rows


class TreeSummaryTests(unittest.TestCase):
    def test_incomplete_coverage_never_becomes_tree512(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder)/'study')
            path = fixture.output/'recursive-events/batch/0/events.jsonl'
            tree(path, fixture.receipt)
            summary, sources = collect.tree_summary([fixture.output], fixture.receipt, 5, folder)
            self.assertEqual(summary['status'], 'INCOMPLETE_TREE_COVERAGE')
            self.assertEqual(summary['closed_questions'], 1)
            self.assertEqual(len(sources), 1)

    def test_full_closed_coverage_preserves_native_tree_and_exact_policy_fraction(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder)/'study')
            for question in range(512):
                tree(fixture.output/'recursive-events/batch'/str(question)/'events.jsonl',
                     fixture.receipt, question='private-question-'+str(question))
            summary, _ = collect.tree_summary([fixture.output], fixture.receipt, 5, folder)
            self.assertEqual(summary['questions'], 512)
            self.assertEqual(summary['selected'], 2560)
            self.assertEqual(summary['selected_policy_token_occurrences'], 512*28)
            self.assertEqual(summary['selected_generations'], {0:512, 1:2048})
            self.assertEqual(summary['mixed'], 512)
            self.assertEqual(summary['reward_0_to_1'], 512)
            self.assertEqual(summary['recorded_all_candidate_branch_cost_sum']['branch_wall_seconds'], 4096.)
            native = dict(fixture.log_lines and {'timing_s/step':10.},
                          **{'rollout/raw_inherited_leaf_tokens':512*8})
            line = 'step:5 - '+' - '.join(f'{key}:{value}' for key,value in native.items())+'\n'
            costs = collect.cost_summary({5:line}, 5, summary, recovered=True)
            fraction = costs['inherited_policy_occurrence_fraction']
            self.assertEqual(fraction['numerator'], 4096)
            self.assertEqual(fraction['denominator'], 14336)
            self.assertAlmostEqual(fraction['value'], 8/28)
            self.assertFalse(costs['complete_end_to_end_cost'])
            self.assertTrue(costs['recovered_original_sampling_cost_missing'])
            self.assertNotIn('private-question', json.dumps(summary))

    def test_observation_lengths_cannot_substitute_for_missing_policy_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder)/'study')
            for question in range(512):
                tree(fixture.output/'recursive-events/batch'/str(question)/'events.jsonl',
                     fixture.receipt, question=str(question), root_mask=question != 0)
            summary, _ = collect.tree_summary([fixture.output], fixture.receipt, 5, folder)
            self.assertIsNone(summary['selected_policy_token_occurrences'])
            self.assertEqual(summary['policy_denominator_missing_questions'], 1)
            costs = collect.cost_summary({5:'step:5 - timing_s/step:10 - response_length/mean:100 - '
                'rollout/raw_inherited_leaf_tokens:4096\n'}, 5, summary, recovered=False)
            self.assertIsNone(costs['inherited_policy_occurrence_fraction']['value'])

    def test_wrong_source_duplicate_question_and_truncated_record_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder)/'study')
            path = fixture.output/'recursive-events/batch/0/events.jsonl'
            tree(path, fixture.receipt)
            raw = path.read_text()
            path.write_text(raw.replace(fixture.receipt['source_commit'], 'wrong-source'))
            with self.assertRaisesRegex(ValueError, 'source/run/config'):
                collect.tree_rows(path, fixture.receipt, 5)
            path.write_text(raw.rstrip('\n'))
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                collect.tree_rows(path, fixture.receipt, 5)
            path.write_text(raw)
            tree(fixture.output/'recursive-events/batch/1/events.jsonl', fixture.receipt)
            with self.assertRaisesRegex(ValueError, 'Duplicate closed question'):
                collect.tree_summary([fixture.output], fixture.receipt, 5, folder)

    def test_only_formal_closed_boundary_reports_once_and_never_marks_goal_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder)/'study')
            for question in range(512):
                tree(fixture.output/'recursive-events/batch'/str(question)/'events.jsonl',
                     fixture.receipt, question=str(question))
            tree(fixture.output/'recursive-events/live/0/events.jsonl', fixture.receipt, outer=6, question='live-6')
            output = Path(folder)/'analysis'
            result = collect.collect(fixture.base, output)
            self.assertEqual(result['status'], 'ACTIVE')
            self.assertFalse(result['goal_complete'])
            self.assertEqual(result['collected_unique_reports'], 1)
            summary = json.loads((output/'C1-root-only/outer-005/summary.json').read_text())
            self.assertEqual(summary['dev512']['all']['hits'], 92)
            self.assertTrue(summary['costs']['recovered_original_sampling_cost_missing'])
            self.assertEqual(result['matched_reuse']['generation-1']['tree_summary_same_actual_condition'],
                             result['reports']['C1-root-only'])
            with patch.object(collect, 'tree_summary', side_effect=AssertionError('Accepted trees were reparsed')):
                repeated = collect.collect(fixture.base, output)
            self.assertEqual(repeated['reports'], result['reports'])
            # The scalar/dev log alone cannot authorize another sparse milestone.
            fixture.closed.pop()
            fixture.write_closed()
            new = collect.collect(fixture.base, Path(folder)/'analysis-other')
            self.assertEqual(new['collected_unique_reports'], 0)


if __name__ == '__main__':
    unittest.main()
