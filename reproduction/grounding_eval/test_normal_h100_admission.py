"""Offline admission checks; no GPU lease, NVML, CUDA, or model is accessed."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from reproduction.grounding_eval.normal_h100 import (validate_gpu_snapshot, controller_reservations,
    python_module,prefix_process_reservations,NoHeldGPUReservations)


class Admission(unittest.TestCase):
    def test_only_actual_python_module_option_matches(self):
        self.assertEqual(python_module([b'python',b'-u',b'-m',b'reproduction.prefix_selection200.rollout'],'/usr/bin/python3.10'),'reproduction.prefix_selection200.rollout')
        for command in ([b'python',b'script.py',b'-m',b'reproduction.prefix_selection200.rollout'],
                        [b'python',b'-c',b'reproduction.prefix_selection200.rollout'],
                        [b'bash',b'-m',b'reproduction.prefix_selection200.rollout']):
            self.assertIsNone(python_module(command,'/bin/bash' if command[0]==b'bash' else '/usr/bin/python3'))

    def child_scope_case(self, selected='GPU-f', parent_reservations=None, changed=False, module='reproduction.prefix_selection200.rollout'):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);child=root/'123';parent=root/'100';parent.mkdir();child.mkdir()
        (parent/'cmdline').write_bytes(b'python\0-m\0reproduction.actor50_grpo.worker\0')
        (parent/'exe').symlink_to('/usr/bin/python3.10')
        before=dict(pid=123,ppid=100,birth_ticks=10);after=dict(before,ppid=101)if changed else before
        owner=dict(pid=100,ppid=1,birth_ticks=5)
        proof=dict(pid=100,reservation_uuids=parent_reservations if parent_reservations is not None else ['GPU-a','GPU-b','GPU-c','GPU-d','GPU-e'])
        rows=[[str(i),'GPU-'+name]for i,name in enumerate('abcdef')]
        with patch('reproduction.grounding_eval.normal_h100.controller_reservations',side_effect=[NoHeldGPUReservations('child has no held lock'),proof]),patch('reproduction.grounding_eval.normal_h100.process_identity',side_effect=[before,owner,after,owner]),patch('reproduction.grounding_eval.normal_h100.cuda_visible_devices',return_value='0,1,2,3'):
            return prefix_process_reservations(child,root/'locks',selected,rows,module)

    def test_known_rollout_child_with_stable_reserved_parent_is_scoped(self):
        self.assertEqual(self.child_scope_case()['parent_reservation_proof']['reservation_uuids'],['GPU-a','GPU-b','GPU-c','GPU-d','GPU-e'])

    def test_rollout_changed_parent_link_or_parent_claiming_selected_rejects(self):
        with self.assertRaises(RuntimeError):self.child_scope_case(changed=True)
        with self.assertRaises(RuntimeError):self.child_scope_case(selected='GPU-a')
        with self.assertRaises(RuntimeError):self.child_scope_case(parent_reservations=[])

    def test_unknown_child_module_cannot_borrow_parent_scope(self):
        with self.assertRaises(RuntimeError):self.child_scope_case(module='reproduction.prefix_selection200.bootstrap')

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
