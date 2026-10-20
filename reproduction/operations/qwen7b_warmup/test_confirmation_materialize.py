"""Pure synthetic confirmation-plan tests: no real dataset/model/GPU/remote IO."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from .confirmation_materialize import SOURCE_SHA, build_rows, normalize, plan_events, render_event_script


def frozen_pool(count=4):
    questions = []
    for i in range(count):
        question = 'untouched question%d?' % i
        questions.append(dict(source_question_id='nq:dev:'+str(i),
            canonical_question_sha256=hashlib.sha256(normalize(question).encode()).hexdigest(),
            source_row=dict(id=i, question=question, golden_answers=['answer%d' % i])))
    return dict(role='UNTOUCHED_FINAL_CONFIRMATION_NOT_FOR_TUNING', source_sha256=SOURCE_SHA,
        generation_not_started=True, selection_before_any_rollout_or_probe_score=True, questions=questions)


def template():
    return dict(prompt=[dict(role='user', content='Use search to answer: TEMPLATE?')], ability='qa',
                reward_model=dict(style='rule', ground_truth=dict(target=['template-answer'])))


def identity(row):
    content = '\n'.join(m['content'] for m in row['prompt'])
    return dict(qid=json.dumps([row['data_source'], row['extra_info']['index']]),
                question_hash=hashlib.sha256(content.encode()).hexdigest())


def build(pool=None, tokenize=None):
    return build_rows(pool or frozen_pool(), template(), 'TEMPLATE?', tokenize or (lambda prompt: [1, 2, 3]), identity, expected=4)


class ConfirmationMaterializeTests(unittest.TestCase):
    def test_materializes_exact_frozen_rows_and_source_answers(self):
        rows = build()
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['source_question_id'] for r in rows], ['nq:dev:'+str(i) for i in range(4)])
        for i, row in enumerate(rows):
            self.assertEqual(row['reward_model']['ground_truth']['target'], ['answer%d' % i])
            self.assertEqual(row['initial_prompt_token_ids'], [1, 2, 3])
            self.assertEqual(row['candidate_ordinal'], i)
            self.assertEqual(row['split'], 'confirmation')

    def test_no_frozen_question_replacement_or_silent_truncation(self):
        with self.assertRaises(ValueError):
            build(tokenize=lambda prompt: list(range(274)))
        changed = frozen_pool()
        changed['questions'][0]['source_row']['question'] = 'changed'
        with self.assertRaises(ValueError):
            build(changed)
        changed = frozen_pool()
        changed['questions'][1] = copy.deepcopy(changed['questions'][0])
        with self.assertRaises(ValueError):
            build(changed)

    def test_two_fresh_g5_events_fixed_view_without_labels(self):
        rows = build()
        plan = plan_events(rows, ['freshA', 'freshB'], ['attemptA', 'attemptB'], 17)
        self.assertEqual(plan['actual_responses_generated'], 0)
        self.assertFalse(plan['labels_observed'])
        for group in plan['planned_G10']:
            self.assertEqual(len(set(group['trajectory_ids'])), 10)
            self.assertTrue(all(t.startswith('freshA-') for t in group['trajectory_ids'][:5]))
            self.assertTrue(all(t.startswith('freshB-') for t in group['trajectory_ids'][5:]))
        changed = copy.deepcopy(rows)
        for row in changed:
            row['labels'] = [1]*10
            row['probe_score'] = 999
        same = plan_events(changed, ['freshA', 'freshB'], ['attemptA', 'attemptB'], 17)
        self.assertEqual(plan, same)
        self.assertTrue(plan['fixed_G5']['summary']['labels_consulted'] is False)
        with self.assertRaises(ValueError):
            plan_events(rows, ['same', 'same'], ['a', 'b'])

    def test_sbatch_script_preserves_environment_variable_and_is_valid_shell(self):
        script = render_event_script(Path('/data/user/test/event-A'), Path('/data/user/test/code'))
        self.assertIn('export QWEN7B_CODE_ROOT=/data/user/test/code', script)
        self.assertIn('--partition=acd_u', script)
        self.assertIn('--gres=gpu:2', script)
        self.assertNotIn('__CODE_ROOT__', script)
        checked = subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
