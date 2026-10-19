"""Offline admission checks; no GPU lease, NVML, CUDA, or model is accessed."""
import unittest
from reproduction.grounding_eval.normal_h100 import validate_gpu_snapshot


class Admission(unittest.TestCase):
    def test_only_empty_matching_uuid_is_accepted(self):
        validate_gpu_snapshot(2, 'GPU-a', [['2', 'GPU-a', '4', '80000', '0']], [], 48)

    def test_numeric_cuda_mapping_must_match_frozen_uuid(self):
        with self.assertRaises(RuntimeError):
            validate_gpu_snapshot(1, 'GPU-a', [['2', 'GPU-a', '4', '80000', '0']], [], 48)

    def test_compute_even_with_zero_utilization_is_not_idle(self):
        with self.assertRaises(RuntimeError):
            validate_gpu_snapshot(2, 'GPU-a', [['2', 'GPU-a', '4', '80000', '0']], [['GPU-a', '123']], 48)

    def test_insufficient_free_or_pending_activity_is_rejected(self):
        for row in [['2', 'GPU-a', '4', '40000', '0'],
                    ['2', 'GPU-a', '4', '80000', '1'],
                    ['2', 'GPU-a', '1024', '80000', '0']]:
            with self.assertRaises(RuntimeError):
                validate_gpu_snapshot(2, 'GPU-a', [row], [], 48)


if __name__ == '__main__':
    unittest.main()
