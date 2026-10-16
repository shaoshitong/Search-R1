import json
from pathlib import Path
import tempfile
import unittest

from .actor import sha
from .checkpoint_native_once import actual_selection_binding


class SelectionBindingTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory); score = base / 'outers/outer-000043/probe-score'
        score.mkdir(parents=True)
        previous = base / 'outers/outer-000040/probe-update'; previous.mkdir(parents=True)
        expected = dict(completed_outer=40, actual_adam_steps_per_rank=[200] * 8, sha256='actual40')
        (previous / 'metrics.json').write_text(json.dumps(dict(checkpoint=expected)))
        common = dict(run_id='run', outer=43, actor_version='actor42')
        selection = dict(common, probe_before_checkpoint_sha256='actual40', outcome_labels_used=False)
        (score / 'selection.json').write_text(json.dumps(selection))
        selection_sha = sha(score / 'selection.json')
        manifest = dict(common, probe_before_checkpoint_sha256='actual40', selection_sha256=selection_sha)
        (score / 'manifest.json').write_text(json.dumps(manifest))
        complete = dict(common, selection_sha256=selection_sha, manifest_sha256=sha(score / 'manifest.json'))
        (score / 'complete.json').write_text(json.dumps(complete))
        joint = dict(run_id='run', actor=dict(sampling_actor_version='actor42'),
                     selection_sha256=selection_sha, rollout=dict(selection_sha256=selection_sha))
        return base, joint, expected

    def test_actual_probe_is_bound_even_when_legacy_driver_has_old_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base, joint, expected = self.fixture(directory)
            driver = dict(prefix_probe_receipt=dict(path='/old/probe/latest.json', value=dict(
                completed_outer=20, actual_adam_steps_per_rank=[100] * 8, sha256='old20')))
            proof = actual_selection_binding(base, 43, joint, driver)
            self.assertEqual(proof['actual_probe'], expected)
            self.assertFalse(proof['driver_auxiliary_probe_matches_selection'])
            self.assertFalse(proof['original_driver_modified'])

    def test_wrong_selection_version_cannot_be_excused_by_driver_auxiliary(self):
        with tempfile.TemporaryDirectory() as directory:
            base, joint, expected = self.fixture(directory)
            joint['actor']['sampling_actor_version'] = 'wrong-actor'
            with self.assertRaises(ValueError):
                actual_selection_binding(base, 43, joint, dict(prefix_probe_receipt=dict(value=expected)))

    def test_correct_current_driver_reference_is_recorded_as_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            base, joint, expected = self.fixture(directory)
            proof = actual_selection_binding(base, 43, joint, dict(prefix_probe_receipt=dict(value=expected)))
            self.assertTrue(proof['driver_auxiliary_probe_matches_selection'])


if __name__ == '__main__':
    unittest.main()
