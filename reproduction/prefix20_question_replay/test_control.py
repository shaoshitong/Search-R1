"""CPU contract checks for continuing the exact already-completed replay state."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import launch


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return launch.sha(path)


def frozen_replay(root):
    source, target = root/'source-original', root/'source-resumed'
    for base in (source, target):
        (base/'preparation').mkdir(parents=True)
    (source/'train').mkdir()
    test = source/'fixed-test/features'
    test.mkdir(parents=True)
    manifest_sha = write(test/'manifest.json', dict(fixed_identity='same512G5'))
    checkpoint = source/'train/latest.pt'
    checkpoint.write_bytes(b'complete-learned-head-adam-eight-rng-not-reset')
    checkpoint_sha = launch.sha(checkpoint)
    binding = dict(test_manifest_sha256=manifest_sha, question_max_tokens=273, head_max_tokens=293,
                   source_epoch=71, source_checkpoint_sha256='a'*64,
                   source_adam_steps_per_rank=[7100]*8, learning_rate=3e-6)
    receipt_sha = write(source/'train/latest.json', dict(replay_iteration=3, sha256=checkpoint_sha,
        actual_adam_steps_per_rank=[7400]*8, actual_learning_rate=3e-6, binding_sha256='b'*64))
    claim_sha = write(source/'train/run-claim.json', dict(binding=binding, binding_sha256='b'*64))
    qmax_sha = write(source/'preparation/question-max.json', dict(question_max_tokens=273))
    write(source/'preparation/replay-sources.json', dict(sources=list(range(1,71))))
    proof_path = source/'preparation/frozen-replay-proof.json'
    proof = dict(source_base=str(source), source_run_id='logical-run', native_job_id=3410,
        native_state='CANCELLED', owned_births_present=0, native_cgroup_exists=False,
        completed_iteration=3, checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint.stat().st_size,
        latest_receipt_path=str(source/'train/latest.json'), latest_receipt_sha256=receipt_sha,
        run_claim_path=str(source/'train/run-claim.json'), run_claim_sha256=claim_sha,
        question_max_proof_path=str(source/'preparation/question-max.json'), question_max_proof_sha256=qmax_sha,
        question_max_tokens=273, fixed_test_features_path=str(test), fixed_test_manifest_sha256=manifest_sha,
        actual_adam_steps_per_rank=[7400]*8, actual_learning_rate=3e-6)
    proof_sha = write(proof_path, proof)
    c = dict(run_id='logical-run', start_iteration=4, source_epoch=71,
        source_checkpoint_sha256='a'*64, source_adam_steps_per_rank=[7100]*8, learning_rate=3e-6,
        resume_replay=dict(source_base=str(source), completed_iteration=3,
            checkpoint_sha256=checkpoint_sha, fixed_test_features_path=str(test),
            fixed_test_manifest_sha256=manifest_sha, proof_path=str(proof_path), proof_sha256=proof_sha))
    return source, target, c


class ResumeControlTest(unittest.TestCase):
    def test_nonzero_replay_copies_exact_state_and_keeps_fixed_test_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, target, c = frozen_replay(root)
            original = (source/'train/latest.pt').read_bytes()
            with patch.object(launch, 'REPLAY_NAMESPACE_PREFIX', str(root/'source-')):
                result = launch.prepare_resume_state(target, c)
            self.assertEqual((target/'train/latest.pt').read_bytes(), original)
            self.assertEqual((source/'train/latest.pt').read_bytes(), original)
            for name in ('latest.json', 'run-claim.json'):
                self.assertEqual((target/'train'/name).read_bytes(), (source/'train'/name).read_bytes())
            self.assertEqual((target/'preparation/replay-sources.json').read_bytes(),
                             (source/'preparation/replay-sources.json').read_bytes())
            self.assertEqual(result['actual_adam_steps_per_rank'], [7400]*8)
            self.assertEqual(result['fixed_test_features_path'], c['resume_replay']['fixed_test_features_path'])
            self.assertEqual(list(range(launch.replay_start_iteration(c),71))[0], 4)
            self.assertFalse((target/'fixed-test').exists())

    def test_resume_cannot_replay_previously_completed_iterations(self):
        with self.assertRaises(ValueError):
            launch.replay_start_iteration(dict(start_iteration=1,
                resume_replay=dict(completed_iteration=3)))

    def test_changed_fixed_test_is_rejected_before_checkpoint_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, target, c = frozen_replay(root)
            write(source/'fixed-test/features/manifest.json', dict(fixed_identity='different'))
            with patch.object(launch, 'REPLAY_NAMESPACE_PREFIX', str(root/'source-')):
                with self.assertRaises(ValueError):
                    launch.prepare_resume_state(target,c)
            self.assertFalse((target/'train/latest.pt').exists())


if __name__=='__main__':
    unittest.main()
