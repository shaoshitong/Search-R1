"""Pure synthetic tests, without any old trajectory EM/probe outputs or remote IO."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from .confirmation_materialize import digest
from .validation_g10_materialize import (ROLE, final_exclusion_from_rows, question_rows_only,
    render_validation_script, validate_isolation, validation_plan_events)


def row(i):
    content = 'validation question%d' % i
    return dict(qid=json.dumps(['nq', 'nq:dev:'+str(i)]),
        question_hash=hashlib.sha256(content.encode()).hexdigest(), data_source='nq',
        prompt=[dict(role='user', content=content)], ability='qa',
        reward_model=dict(style='rule', ground_truth=dict(target=['source-answer'])),
        extra_info=dict(index='nq:dev:'+str(i), split='old-viewed-validation'),
        source_question_id='nq:dev:'+str(i), canonical_question_sha256='normalized%d' % i,
        initial_prompt_token_ids=[1, 2, 3], candidate_ordinal=i)


class ValidationG10Tests(unittest.TestCase):
    def test_questions_only_ignores_old_em_and_probe_fields(self):
        original = [row(i) for i in range(4)]
        clean = question_rows_only(original, expected=4)
        changed = copy.deepcopy(original)
        for own in changed:
            own.update(labels=[1]*10, strict_em=1, p_j=[.5]*5, predicted_mixed=1., probe_score=999,
                       trajectory_ids=['old-trajectory']*5)
        self.assertEqual(clean, question_rows_only(changed, expected=4))
        self.assertNotIn('labels', clean[0])
        self.assertNotIn('p_j', clean[0])
        self.assertEqual(clean[0]['split'], 'recipe_selection_validation')

    def test_qid_prompt_or_duplicate_corruption_rejected(self):
        rows = [row(i) for i in range(4)]
        rows[0]['prompt'][0]['content'] = 'changed'
        with self.assertRaises(ValueError):
            question_rows_only(rows, expected=4)
        rows = [row(i) for i in range(4)]
        rows[3] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):
            question_rows_only(rows, expected=4)

    def test_two_new_events_keep_validation_role_and_no_label_selection(self):
        rows = question_rows_only([row(i) for i in range(4)], expected=4)
        planned = validation_plan_events(rows, ['newA', 'newB'], ['attemptA', 'attemptB'], 19)
        self.assertEqual(planned['role'], ROLE)
        self.assertFalse(planned['untouched_final'])
        self.assertTrue(planned['recipe_selection_allowed'])
        self.assertFalse(planned['fitting_on_validation_allowed'])
        self.assertEqual(planned['fixed_G5']['summary']['role'], 'validation')
        self.assertFalse(planned['fixed_G5']['summary']['labels_consulted'])
        for own in planned['planned_G10']:
            self.assertEqual(len(set(own['trajectory_ids'])), 10)
            self.assertTrue(all(t.startswith(('newA-', 'newB-')) for t in own['trajectory_ids']))
        changed = copy.deepcopy(rows)
        for own in changed:
            own['labels'] = [1]*10
        self.assertEqual(planned, validation_plan_events(changed, ['newA', 'newB'], ['attemptA', 'attemptB'], 19))

    def test_final_exclusion_has_expected_training_preparer_contract(self):
        finals = {'confirmation2048':[row(10), row(11)], 'stability2048':[row(12), row(13)]}
        value = final_exclusion_from_rows(finals, [dict(path='sha-bound-final-inputs', sha256='a'*64)])
        self.assertEqual(value['role'], 'final')
        self.assertTrue(value['frozen'])
        self.assertEqual(value['ordered_qid_sha256'], digest([r['qid'] for r in value['questions']]))
        self.assertFalse(value['labels_read'])
        self.assertTrue(all('qid' in r and 'question_hash' in r for r in value['questions']))
        with self.assertRaises(ValueError):
            final_exclusion_from_rows({'a':[row(10)], 'b':[row(10)]}, [])

    def test_validation_isolation_rejects_any_train_diag_final_overlap(self):
        rows = [row(0), row(1)]
        final_plan = final_exclusion_from_rows({'confirmation2048':[row(10)]}, [])
        report = validate_isolation(rows, [row(20), row(21)], final_plan)
        self.assertTrue(report['disjoint_train_and512diagnostic'])
        self.assertTrue(report['disjoint_confirmation_and_stability'])
        with self.assertRaises(ValueError):
            validate_isolation(rows, [row(0)], final_plan)
        with self.assertRaises(ValueError):
            validate_isolation(rows, [], final_exclusion_from_rows({'a':[row(1)]}, []))

    def test_rendered_preflight_is_explicit_validation_and_shell_valid(self):
        script = render_validation_script(Path('/data/user/test/validation/event-A'), Path('/data/user/test/code'))
        self.assertIn('validation_g10_materialize --check-event-config', script)
        self.assertNotIn('confirmation_materialize --check-event-config', script)
        self.assertIn('export QWEN7B_CODE_ROOT=/data/user/test/code', script)
        self.assertIn('#SBATCH --job-name=q7b-valid-G5', script)
        result = subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
