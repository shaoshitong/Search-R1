"""CPU tests for the complete actor/probe/pool recovery boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from .actor import _state_binding
from .resume import boundary, pending_reservation
from .pool import initial_state, reserve_indices, state_sha
from .actor import sha


class ResumeTests(unittest.TestCase):
    def reservation_fixture(self, directory):
        source = Path(directory)
        (_, _, committed) = reserve_indices(initial_state(6000), 1)
        indices, cycles, after = reserve_indices(committed, 2)
        following = source / 'outers/outer-000002'
        following.mkdir(parents=True)
        (source / 'question-pool').mkdir()
        def put(path, value): path.write_text(json.dumps(value))
        put(source / 'question-pool/state.json', after)
        put(source / 'question-pool/binding.json', dict(sha256='pool'))
        put(following / 'pool-reservation.json', dict(outer=2, pool_sha256='pool',
            before_sha256=state_sha(committed), after_sha256=state_sha(after),
            after=after, indices=indices, cycles=cycles))
        rows = [dict(qid=str(index), pool_index=index, pool_cycle=cycle,
                     candidate_ordinal=i) for i, (index, cycle) in enumerate(zip(indices, cycles))]
        put(following / 'selection.json', dict(run_id='run', outer=2, questions=5120, rows=rows))
        put(following / 'questions.json', dict(run_id='run', outer=2, questions=5120,
            selection_sha256=sha(following / 'selection.json'),
            rows=[dict(row, prompt='original prompt', gold='answer') for row in rows]))
        return source, committed, following

    def test_failed_next_reservation_keeps_exact_questions_across_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            source, committed, following = self.reservation_fixture(directory)
            value = pending_reservation(source, committed, 1, 'run')
            self.assertEqual(value['outer'], 2)
            self.assertEqual(value['files']['questions.json'], sha(following / 'questions.json'))
            self.assertTrue(value['no_second_candidate_consumption'])
            self.assertEqual(committed['next_outer'], 2)
            self.assertEqual(json.loads((source / 'question-pool/state.json').read_text())['next_outer'], 3)

    def test_changed_reserved_question_or_newer_complete_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, committed, following = self.reservation_fixture(directory)
            path = following / 'questions.json'
            value = json.loads(path.read_text()); value['rows'][0]['qid'] = 'wrong'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): pending_reservation(source, committed, 1, 'run')
            (following / 'complete.json').write_text('{}')
            with self.assertRaises(ValueError): pending_reservation(source, committed, 1, 'run')

    def test_no_pending_claim_does_not_require_next_outer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory); (source / 'question-pool').mkdir()
            committed = initial_state(6000)
            (source / 'question-pool/state.json').write_text(json.dumps(committed))
            self.assertIsNone(pending_reservation(source, committed, 0, 'run'))

    def test_resume5_requires_both_real_adam_clocks_and_next_candidate_cursor(self):
        c = dict(run_id='run', official_model_dir='/dataset/model', grounding_probe_version='fixed0',
                 resume_joint=dict(completed_outer=5))
        a = dict(run_id='run', completed_outer=5, binding=_state_binding(c),
                 actual_adam_steps_per_rank=[100]*8, scheduler_completed_outer=5)
        p = dict(run_id='run', completed_outer=5, actual_adam_steps_per_rank=[50]*8)
        pool = dict(next_outer=6, consumed_candidates=25600, cycle=0, cursor=25600, order=list(range(168510)))
        joint = dict(run_id='run', outer=5, actor=a, probe=p)
        self.assertEqual(boundary(c, joint, a, p, pool), 5)
        changed = copy.deepcopy(p); changed['actual_adam_steps_per_rank'][7] = 49
        with self.assertRaises(ValueError): boundary(c, joint, a, changed, pool)
        with self.assertRaises(ValueError): boundary(c, joint, a, p, dict(pool, cursor=0))
        changed = copy.deepcopy(a); changed['scheduler_completed_outer'] = 4
        with self.assertRaises(ValueError): boundary(c, dict(joint, actor=changed), changed, p, pool)


if __name__ == '__main__': unittest.main()
