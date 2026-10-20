"""Small CPU boundary tests; no remote, model, audit replay or real closure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reproduction.operations.qwen7b_resample.operational_reads import node10_mixed_productive_closure as closure


class MixedClosureTest(unittest.TestCase):
    def test_missing_or_changed_independent_cache_refuses_acceptance(self):
        with self.assertRaisesRegex(AssertionError, 'VERIFIED'):
            closure.valid_cache(None, dict(epoch=10), 'input', 'config', {}, 128)
        cache = dict(status='VERIFIED', verified_epoch=9, binding=dict(run_id=closure.RUN, seed=20261021,
            auditor_sha256=closure.AUDITOR, root_config_sha256='config', input_sha256='input', files={'full': {'inode': 1}}),
            proof=dict(questions=128, trajectories=640, labels=640))
        with mock.patch.object(closure, 'stat', return_value={'inode': 1}):
            closure.valid_cache(cache, dict(epoch=10), 'input', 'config', {'full': Path('/metadata-only')}, 128)
        with mock.patch.object(closure, 'stat', return_value={'inode': 2}):
            with self.assertRaisesRegex(AssertionError, 'binding changed'):
                closure.valid_cache(cache, dict(epoch=10), 'input', 'config', {'full': Path('/metadata-only')}, 128)
        with self.assertRaises(AssertionError):closure.valid_cache(cache, dict(epoch=8), 'input', 'config', {}, 128)

    def test_new_full_unit_must_match_current_mixed_instance(self):
        indices, unit = (0, 4, 6, 7), 'actual-v3-tp4.service'
        allocation = dict(model_cuda_uuids=['a', 'b', 'c', 'd'])
        full = dict(run_id=closure.RUN, phase='continue', actor_version='official7b-' + closure.REVISION,
            device_binding=dict(native_unit=unit, tensor_parallel_size=4, physical_gpu_indices=list(indices),
                gpu_uuids=allocation['model_cuda_uuids'], normalized_cuda_visible_devices='0,4,6,7'))
        closure.valid_full(full, allocation, unit, indices, 4)
        full['device_binding']['native_unit'] = 'old-prefix-producer.service'
        with self.assertRaisesRegex(AssertionError, 'another unit'):closure.valid_full(full, allocation, unit, indices, 4)

    def test_each_replacement_requires_two_own_events(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory);log = job / 'sampler.log';job_id = 'relay10-primary-123'
            first = dict(status='POOL_OWN_SEALED_BATCH_ADVANCED', run_id=closure.RUN, job_id=job_id,
                completed_own_batches=1, latest_input='first.json')
            second = dict(first, completed_own_batches=2, latest_input='second.json')
            log.write_text(json.dumps(first) + '\n')
            with self.assertRaisesRegex(ValueError, 'fewer than two'):closure.first_two(job, job_id)
            log.write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
            self.assertEqual(len(closure.first_two(job, job_id)), 2)
            with self.assertRaises(AssertionError):closure.first_two(job, 'foreign-job')

    def test_controlled_readiness_stop_label_and_original_failure_are_preserved(self):
        old = dict(pid=2905668, unit='old-v2.service', original_attempt_reason='AUTHORIZED_READINESS_REPAIR_CONTROLLED_SYSTEMEXIT143_NOT_FAILURE_OR_OOM',
            original_attempt_controlled_stop=True, failure=dict(sha256='original-failure143'), terminal=dict(path='/terminal.json', sha256='real-terminal'))
        acceptance = dict(path='/acceptance.json', sha256='acceptance', value=dict(epoch=10,
            replacement_new_complete_trajectories=6400, replacements=[dict(controller_pid=999, directory='/new-attempt')]))
        value = closure.resolution(old, acceptance)
        self.assertEqual(value['status'], closure.RESOLVED)
        self.assertTrue(value['original_attempt_controlled_stop'])
        self.assertIn('CONTROLLED_READINESS_REPAIR', value['resolution_kind'])
        self.assertEqual(value['original_failure_sha256'], 'original-failure143')
        self.assertEqual(value['terminal_proof_sha256'], 'real-terminal')
        self.assertFalse(value['whole_goal_complete'])
        self.assertFalse(value['strict_em_recomputed'])
        self.assertFalse(value['states_shards_independently_audited'])

    def test_metadata_hash_entry_cannot_open_large_trajectory(self):
        with self.assertRaisesRegex(AssertionError, 'Only small JSON'):
            closure.record(Path('/must-never-open/rank-0-full.jsonl.gz'))


if __name__ == '__main__':unittest.main()
