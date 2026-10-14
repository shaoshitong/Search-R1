"""CPU-only protocol tests; no remote operations, child launches or model imports."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import storage_observer719225 as observer


class StorageObservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = Path(self.temp.name) / 'once'
        self.control.mkdir()
        self.claim = dict(measure=str(observer.MEASURE), uid=os.getuid(), started_unix=0,
                          source=dict(path='/pinned/observer.py', commit='a'*40, sha256='b'*64))
        self.process = dict(pid=123, start_ticks=456, sid=123, uid=os.getuid())
        observer.publish(self.control / 'claim.json', self.claim)

    def tearDown(self):
        self.temp.cleanup()

    def contexts(self):
        return (patch.object(observer, 'UID', os.getuid()),
                patch.object(observer, 'CONTROL', self.control),
                patch.object(observer, 'validate', return_value=self.claim['source']))

    def inspect_with(self, actual):
        a, b, c = self.contexts()
        with a, b, c, patch.object(observer, 'birth', return_value=actual):
            return observer.inspect()

    def test_only_exact_single_line_completed0_is_a_usage_measurement(self):
        line = '1234\t' + str(observer.MEASURE) + '\n'
        self.assertEqual(observer.allocated_usage(line, '', 0), 1234)
        for out, err, code in ((line, '', 1), (line, '\n', 0), (line + line, '', 0),
                               ('1234\t/some/other/root\n', '', 0), ('1.2\t'+str(observer.MEASURE), '', 0)):
            with self.subTest(output=out, stderr=err, exit_code=code):
                self.assertIsNone(observer.allocated_usage(out, err, code))

    def test_authorized_shared_mou_parent_is_accepted_with_actual_group_access(self):
        metadata = SimpleNamespace(st_uid=observer.SHARED_UID, st_gid=observer.SHARED_GID,
                                   st_mode=stat.S_IFDIR | 0o770)
        with patch.object(Path, 'lstat', return_value=metadata), \
                patch.object(os, 'getgroups', return_value=[observer.SHARED_GID]), \
                patch.object(os, 'access', return_value=True):
            self.assertEqual(observer.shared_directory(observer.ROOT, write=True), metadata)

    def test_foreign_or_symlink_or_inaccessible_shared_parent_is_rejected(self):
        for owner, group, mode, member, access in (
                (99999, observer.SHARED_GID, stat.S_IFDIR | 0o770, True, True),
                (observer.SHARED_UID, 99999, stat.S_IFDIR | 0o770, True, True),
                (observer.SHARED_UID, observer.SHARED_GID, stat.S_IFLNK | 0o770, True, True),
                (observer.SHARED_UID, observer.SHARED_GID, stat.S_IFDIR | 0o770, False, True),
                (observer.SHARED_UID, observer.SHARED_GID, stat.S_IFDIR | 0o770, True, False)):
            metadata = SimpleNamespace(st_uid=owner, st_gid=group, st_mode=mode)
            with self.subTest(owner=owner, group=group, mode=mode, member=member, access=access), \
                    patch.object(Path, 'lstat', return_value=metadata), \
                    patch.object(os, 'getgroups', return_value=[observer.SHARED_GID] if member else []), \
                    patch.object(os, 'access', return_value=access), self.assertRaises(ValueError):
                observer.shared_directory(observer.ROOT, write=True)

    def test_existing_claim_never_relaunches_even_without_handle(self):
        a, b, c = self.contexts()
        with a, b, c, patch.object(observer.subprocess, 'Popen') as spawn:
            value = observer.start()
        spawn.assert_not_called()
        self.assertEqual(value['status'], 'CLAIM_WITHOUT_LIVE_HANDLE_OR_RESULT_NO_REPLAY')

    def test_reused_pid_is_not_the_original_process(self):
        observer.publish(self.control / 'process.json', self.process)
        actual = dict(self.process, start_ticks=457)
        value = self.inspect_with(actual)
        self.assertFalse(value['actual_live_birth'])
        self.assertEqual(value['status'], 'CLAIM_WITHOUT_LIVE_HANDLE_OR_RESULT_NO_REPLAY')

    def test_missing_process_cannot_be_called_a_success(self):
        observer.publish(self.control / 'process.json', self.process)
        self.assertEqual(self.inspect_with(None)['status'],
                         'CLAIM_WITHOUT_LIVE_HANDLE_OR_RESULT_NO_REPLAY')

    def test_terminal_result_survives_missing_process(self):
        observer.publish(self.control / 'process.json', self.process)
        child = dict(self.process, pid=124, start_ticks=457)
        observer.publish(self.control / 'du-process.json', child)
        stdout = '1234\t' + self.claim['measure'] + '\n'
        result = dict(source=self.claim['source'], measure=self.claim['measure'],
                      status='ACTUAL_DU_COMPLETED0_USAGE_MEASURED', actual_du_exit_code=0,
                      allocated_bytes=1234, process=self.process, du_process=child,
                      uid=os.getuid(), actual_du_child_still_live=False,
                      claim_sha256=hashlib.sha256((self.control/'claim.json').read_bytes()).hexdigest(),
                      hard_quota_verified=False, training_control_actions=0,
                      started_unix=1, completed_unix=2, stdout=stdout, stderr='',
                      stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                      stderr_sha256=hashlib.sha256(b'').hexdigest())
        observer.bound_outputs(result)
        observer.publish(self.control / 'result.json', result)
        value = self.inspect_with(None)
        self.assertEqual(value['result']['allocated_bytes'], 1234)
        self.assertFalse(value['actual_live_birth'])

    def test_claimed_success_without_actual_child_and_wait_binding_is_rejected(self):
        observer.publish(self.control / 'result.json', dict(source=self.claim['source'],
                         measure=self.claim['measure'], allocated_bytes=1234,
                         status='ACTUAL_DU_COMPLETED0_USAGE_MEASURED'))
        with self.assertRaises(ValueError):
            self.inspect_with(None)

    def test_large_failure_output_is_bounded_without_losing_its_full_sha(self):
        stderr = 'Permission denied\n' * 200000
        value = observer.bound_outputs(dict(stdout='', stderr=stderr))
        self.assertEqual(value['stderr_sha256'], hashlib.sha256(stderr.encode()).hexdigest())
        self.assertEqual(value['stderr_bytes'], len(stderr.encode()))
        self.assertTrue(value['stderr_truncated'])
        self.assertLessEqual(len(value['stderr'].encode()), 16384)
        self.assertLess(len(json.dumps(value)), 2**21)
        self.assertIsNone(observer.allocated_usage('123\t'+str(observer.MEASURE)+'\n',
                                                  value['stderr'], 0))

    def test_wrong_result_scope_is_rejected(self):
        observer.publish(self.control / 'result.json', dict(source=self.claim['source'],
                         measure='/other', status='ACTUAL_DU_COMPLETED0_USAGE_MEASURED'))
        with self.assertRaises(ValueError):
            self.inspect_with(None)

    def test_metadata_symlink_is_rejected(self):
        link = self.control / 'linked.json'
        link.symlink_to(self.control / 'claim.json')
        with patch.object(observer, 'UID', os.getuid()), self.assertRaises(ValueError):
            observer.read_json(link)

    def test_publish_preserves_original_on_second_attempt(self):
        p = self.control / 'immutable.json'
        observer.publish(p, {'original': True})
        with self.assertRaises(FileExistsError):
            observer.publish(p, {'original': False})
        self.assertEqual(json.loads(p.read_text()), {'original': True})


if __name__ == '__main__':
    unittest.main()
