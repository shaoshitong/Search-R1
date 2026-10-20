"""Synthetic state/plan checks; no model, tokens from real data, or GPU are used."""
import copy
import subprocess
from pathlib import Path
import unittest

from .extra_g5_prepare import ROLE, REVISION, choose_questions, extra_initial_state, validate_config, render_extra_sbatch


def config(offset=5):
    return dict(role=ROLE, model_revision=REVISION, base='/data/user/test/extra',
        original_base='/data/user/test/original', run_id='extra-new', original_run_id='old',
        original_sampling_attempt_id=None, sampling_attempt_id='new-distinct-attempt',
        extra_response_offset=offset, extra_response_indices=list(range(offset, offset+5)),
        warmup_only=True, actor_updates_authorized=False, actor_optimizer_updates=0,
        capture_prefix_tokens=20, feature_layer=26, question_max_tokens=273,
        feature_dimension=3584, extra_shards=4, gpus_per_sampling_job=2,
        actor_version='official7b-'+REVISION)


def fake_state_factory(c, outer, group, response_index, tokenizer):
    return dict(run_id=c['run_id'], actor_version=c['actor_version'], response_index=response_index,
                trajectory_id='%s-outer%d-%s-g%d' % (c['run_id'], outer, group['qid'], response_index),
                sampling_attempt_id=c['sampling_attempt_id'])


def group(i):
    return dict(qid='q%d' % i, question_hash='h%d' % i, split='train',
                trajectory_ids=['original-q%d-g%d' % (i, j) for j in range(5)], labels=[0, 1, 0, 1, 0])


class ExtraG5Tests(unittest.TestCase):
    def test_rendered_export_identifier_survives_path_substitution(self):
        script=render_extra_sbatch(Path('/remote/run'),Path('/remote/repo'))
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        exports='\n'.join(x for x in script.splitlines() if x.startswith('export '))
        result=subprocess.run(['bash','-euc',exports+'\nprintf %s "$QWEN7B_CODE_ROOT"'],text=True,capture_output=True,check=True)
        self.assertEqual(result.stdout,'/remote/repo')

    def test_offsets_are_real_separate_five_slot_events(self):
        own = group(0)
        own['original_trajectory_ids'] = own['trajectory_ids']
        all_ids = set()
        for offset in (5, 10, 15):
            c = config(offset)
            rows = [extra_initial_state(c, 1, own, i, None, factory=fake_state_factory) for i in range(5)]
            self.assertEqual([r['response_index'] for r in rows], list(range(offset, offset+5)))
            ids = {r['trajectory_id'] for r in rows}
            self.assertEqual(len(ids), 5)
            self.assertFalse(ids.intersection(own['trajectory_ids']))
            self.assertFalse(ids.intersection(all_ids))
            all_ids.update(ids)

    def test_no_actor_update_or_old_run_or_validation_state(self):
        for patch in ({'actor_updates_authorized':True}, {'run_id':'old'},
                      {'sampling_attempt_id':None}, {'prefix_diversity':{'mode':'anything'}}):
            c = config()
            c.update(patch)
            with self.assertRaises(ValueError):
                validate_config(c)
        c = config()
        own = group(0)
        own['split'] = 'diagnostic'
        with self.assertRaises(ValueError):
            extra_initial_state(c, 1, own, 0, None, factory=fake_state_factory)
        with self.assertRaises(ValueError):
            extra_initial_state(c, 1, group(0), 5, None, factory=fake_state_factory)

    def test_question_plan_is_label_blind_and_excludes_ids_and_hashes(self):
        rows = [group(i) for i in range(10)]
        diagnostic = [dict(qid='q0', question_hash='h0')]
        heldout = [dict(qid='heldout-other-id', question_hash='h1')]
        final = [dict(qid='q2', question_hash='h2')]
        first = choose_questions(rows, diagnostic, heldout, final, 4, 17)
        changed = copy.deepcopy(rows)
        for row in changed:
            row['labels'] = [1-y for y in row['labels']]
        second = choose_questions(changed, diagnostic, heldout, final, 4, 17)
        self.assertEqual([r['qid'] for r in first], [r['qid'] for r in second])
        self.assertFalse({r['qid'] for r in first}.intersection({'q0', 'q1', 'q2'}))
        self.assertEqual(len(first), 4)
        with self.assertRaises(ValueError):
            choose_questions(rows, diagnostic, heldout, final, 10, 17)

    def test_original_id_collision_fails_closed(self):
        def colliding(c, outer, group, response_index, tokenizer):
            value = fake_state_factory(c, outer, group, response_index, tokenizer)
            value['trajectory_id'] = group['original_trajectory_ids'][0]
            return value
        own = group(0)
        own['original_trajectory_ids'] = own['trajectory_ids']
        with self.assertRaises(ValueError):
            extra_initial_state(config(), 1, own, 0, None, factory=colliding)

    def test_matched_cohort_preserves_exact_order_and_exclusions(self):
        rows=[group(i) for i in range(10)]
        ids=['q9','q4','q7','q3']
        selected=choose_questions(rows,[],[],[],4,17,ordered_qids=ids)
        self.assertEqual([r['qid'] for r in selected],ids)
        changed=copy.deepcopy(rows)
        for row in changed:row['labels']=[1-y for y in row['labels']]
        self.assertEqual([r['qid'] for r in choose_questions(changed,[],[],[],4,98,ordered_qids=ids)],ids)
        for bad in (['q9','q4','q7','q9'],['q9','q4','q7','unknown'],ids[:3]):
            with self.assertRaises(ValueError):
                choose_questions(rows,[],[],[],4,17,ordered_qids=bad)
        with self.assertRaises(ValueError):
            choose_questions(rows,[],[],[dict(qid='other-id',question_hash='h9')],4,17,ordered_qids=ids)


if __name__ == '__main__':
    unittest.main()
