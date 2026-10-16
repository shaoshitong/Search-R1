import json
from pathlib import Path
import tempfile
import unittest

from .actor_hidden import ActorHiddenCapture
from .rollout import phase_worker_environment


class FailureContextTests(unittest.TestCase):
    def test_sync_diagnostic_applies_only_to_the_failed_worker_and_outer(self):
        config = dict(cuda_failure_diagnostic=dict(outer=44, phase='prefix', rank=4, launch_blocking=True))
        original = dict(OTHER='unchanged')
        self.assertEqual(phase_worker_environment(config, 44, 'prefix', 4, original),
                         dict(OTHER='unchanged', CUDA_LAUNCH_BLOCKING='1'))
        for outer, phase, rank in [(45, 'prefix', 4), (44, 'continue', 4), (44, 'prefix', 0)]:
            self.assertEqual(phase_worker_environment(config, outer, phase, rank, original), original)
        self.assertEqual(original, dict(OTHER='unchanged'))

    def test_no_diagnostic_keeps_original_cuda_setting(self):
        original = dict(CUDA_LAUNCH_BLOCKING='0')
        self.assertEqual(phase_worker_environment({}, 44, 'prefix', 4, original), original)

    def test_failure_preserves_only_existing_cpu_tokens_and_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = ActorHiddenCapture.__new__(ActorHiddenCapture)
            capture.failure_context_path = Path(directory) / 'failed.json'
            capture.actor_version = 'actor43'; capture.calls = 17
            capture.query_rows = [dict(request_id='request', tokens=[10, 20], positions=[0, 1])]
            capture.requests = dict(request=dict(trajectory_id='trajectory', request_tokens=[10, 20]))
            capture.preserve_failed_input(RuntimeError('actual CUDA failure'))
            value = json.loads(capture.failure_context_path.read_text())
            self.assertEqual(value['query_rows'], capture.query_rows)
            self.assertEqual(value['failed_native_call'], 18)
            self.assertFalse(value['gpu_tensor_read_attempted'])
            first = capture.failure_context_path.read_bytes()
            capture.preserve_failed_input(RuntimeError('later error'))
            self.assertEqual(first, capture.failure_context_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
