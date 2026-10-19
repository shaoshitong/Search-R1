"""Offline admission checks; no GPU lease, NVML, CUDA, or model is accessed."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from reproduction.grounding_eval.normal_h100 import validate_gpu_snapshot, controller_reservations


class Admission(unittest.TestCase):
    def fake_controller(self, root, uuids, locked=True):
        directory = root / 'leases'; directory.mkdir()
        process = root / '123'; (process/'fd').mkdir(parents=True); (process/'fdinfo').mkdir()
        for index, uuid in enumerate(uuids):
            path = directory / (uuid + '.lock'); path.touch()
            (process/'fd'/str(index)).symlink_to(path)
            text = 'lock:\t1: OFDLCK ADVISORY WRITE -1 00:01:123 0 EOF\n' if locked else 'pos:\t0\n'
            (process/'fdinfo'/str(index)).write_text(text)
        return process, directory

    def test_proven_other_gpu_reservations_can_admit_unreserved_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            process, directory = self.fake_controller(Path(folder), ['GPU-a','GPU-b','GPU-c','GPU-d','GPU-e'])
            result = controller_reservations(process, directory, 'GPU-f')
            self.assertEqual(result['reservation_uuids'], ['GPU-a','GPU-b','GPU-c','GPU-d','GPU-e'])
            self.assertTrue(result['selected_uuid_excluded'])

    def test_open_files_without_held_ofd_reservation_are_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            process, directory = self.fake_controller(Path(folder), ['GPU-a'], locked=False)
            with self.assertRaises(RuntimeError):
                controller_reservations(process, directory, 'GPU-f')

    def test_controller_holding_selected_gpu_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            process, directory = self.fake_controller(Path(folder), ['GPU-f'])
            with self.assertRaises(RuntimeError):
                controller_reservations(process, directory, 'GPU-f')

    def test_unreadable_fdinfo_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            process, directory = self.fake_controller(Path(folder), ['GPU-a'])
            with patch.object(Path, 'read_text', side_effect=PermissionError('denied')):
                with self.assertRaises(RuntimeError):
                    controller_reservations(process, directory, 'GPU-f')

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
