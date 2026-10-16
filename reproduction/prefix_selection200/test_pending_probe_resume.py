"""CPU-only asymmetric recovery guards; no model construction or GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .actor import _state_binding
from .launch import commit_outer
from .pending_probe_resume import validate_boundary
from reproduction.prefix20_online.launch import sha


def fixture():
    source = dict(run_id='r'*32, source_commit='a'*40, execution_id='b'*32,
        official_model_dir='/data/official-qwen', seed=1729,
        grounding_probe_version='qualified-grounding',
        nonmixed_advantage_mode='zero_anchor_all_correct_only')
    config = dict(source, source_commit='c'*40, execution_id='d'*32,
        resume_pending_probe=dict(source_commit=source['source_commit'],
            source_execution_id=source['execution_id'], completed_joint_outer=9, pending_outer=10))
    actor = dict(run_id=config['run_id'], completed_outer=10,
        binding=_state_binding(config), actual_adam_steps_per_rank=[200]*4,
        new_adam_steps_per_rank=[20]*4, scheduler_completed_outer=10)
    probe = dict(run_id=config['run_id'], completed_outer=0, actual_adam_steps_per_rank=[0]*4)
    joint = dict(run_id=config['run_id'], outer=9, source_commit=source['source_commit'],
        actor=dict(completed_outer=9, actual_adam_steps_per_rank=[180]*4,
                   scheduler_completed_outer=9), probe=probe)
    pool = dict(next_outer=11, consumed_candidates=25600, cycle=0,
        order=list(range(30000)), cursor=25600)
    reservation = dict(outer=10, after=copy.deepcopy(pool))
    return config, source, joint, actor, probe, pool, reservation


class PendingProbeRecovery(unittest.TestCase):
    def test_completed_actor_is_preserved_above_last_joint_without_faking_joint10(self):
        args = fixture()
        self.assertEqual(validate_boundary(*args), 10)
        self.assertEqual(args[2]['outer'], 9)
        self.assertEqual(args[3]['actual_adam_steps_per_rank'], [200]*4)
        self.assertEqual(args[4]['actual_adam_steps_per_rank'], [0]*4)

    def test_reject_actor_rewind_partial_or_scheduler_and_probe_disagreement(self):
        for changes, index in ((dict(completed_outer=9), 3),
                (dict(actual_adam_steps_per_rank=[199,200,200,200]), 3),
                (dict(scheduler_completed_outer=9), 3),
                (dict(actual_adam_steps_per_rank=[1]*4), 4),
                (dict(consumed_candidates=23040), 5)):
            args = list(copy.deepcopy(fixture())); args[index].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_boundary(*args)

    def test_reject_new_run_same_execution_or_science_change(self):
        for changes in (dict(run_id='x'*32), dict(execution_id='b'*32),
                dict(source_commit='a'*40), dict(seed=999)):
            args = list(copy.deepcopy(fixture())); args[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_boundary(*args)

    def _commit_fixture(self, base, *, probe_steps=50, complete=True):
        root = base/'outers/outer-000010'
        for name in ('actor', 'probe', 'question-pool', 'outers/outer-000010/probe-update',
                     'outers/outer-000010/probe-score', 'outers/outer-000010/full'):
            (base/name).mkdir(parents=True, exist_ok=True)
        def write(path, value):
            path.write_text(json.dumps(value)); return value
        config = dict(run_id='r', execution_id='new', source_commit='new-source',
            resume_pending_probe=dict(source_commit='old-source',source_execution_id='old',
                source_base='/data/old',completed_joint_outer=9,pending_outer=10,
                qualification_proof_sha256='f'*64))
        actor = write(base/'actor/latest.json', dict(run_id='r', completed_outer=10,
            actual_adam_steps_per_rank=[200]*4,scheduler_completed_outer=10))
        probe = write(base/'probe/latest.json', dict(run_id='r',completed_outer=10,
            actual_adam_steps_per_rank=[probe_steps]*4))
        (base/'probe/latest.pt').write_bytes(b'fake sealed probe bytes, not a tensor')
        write(base/'question-pool/state.json', dict(next_outer=11,consumed_candidates=25600,
            cycle=0,order=list(range(30000)),cursor=25600))
        write(root/'probe-score/selection.json',dict(predicted=True))
        selection = root/'probe-score/actual-em-selection.json'
        write(selection,dict(outcome_labels_used=True))
        write(root/'full/complete.json',dict(trajectories=2560,source='/data/old/immutable-rollout'))
        metrics = root/'probe-update/metrics.json'
        write(metrics,dict(run_id='r',outer=10,new_adam_steps_per_rank=50))
        if complete:
            write(root/'probe-update/complete.json',dict(
                status='UNSELECTED2560G5_PERIODIC_PROBE_50_ADAM_COMPLETE',run_id='r',outer=10,
                new_adam_steps_per_rank=50,metrics_sha256=sha(metrics),checkpoint=probe))
        return config, root, selection, actor, probe

    def test_joint10_waits_for_real_probe_complete_and_keeps_parent_source_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); c, root, selection, actor, probe = self._commit_fixture(base)
            receipt = dict(status='IMPORTED',actor_adam_per_rank=[200]*4,pending_outer=10)
            result = commit_outer(base,c,10,root,selection,{},recovery=receipt)
            self.assertEqual(result['actor'], actor)
            self.assertEqual(result['probe'], probe)
            self.assertEqual(result['source_commit'],'new-source')
            self.assertEqual(result['source_lineage']['parent_source_commit'],'old-source')
            self.assertEqual(result['pending_probe_recovery']['imported_actor_updates_reexecuted'],0)
            self.assertEqual(json.loads((base/'latest-joint.json').read_text())['outer'],10)
            self.assertEqual((base/'probe/latest.pt').stat().st_ino,
                (root/'probe-state-at-completion.pt').stat().st_ino)

    def test_no_joint10_on_partial_probe_or_missing_complete(self):
        for partial in (True,False):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                c,root,selection,_,_ = self._commit_fixture(base,
                    probe_steps=49 if partial else 50,complete=partial)
                with self.assertRaises((RuntimeError,FileNotFoundError)):
                    commit_outer(base,c,10,root,selection,{})
                self.assertFalse((base/'latest-joint.json').exists())
                self.assertFalse((root/'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
