"""Constructed CPU filesystem races; no Slurm, GPU, model or training evidence."""
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8.ledger_snapshot import read_owned_ledger_snapshot


class _ReadHook:
    def __init__(self, stream, hook):
        self.stream, self.hook, self.reads = stream, hook, 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def fileno(self):
        return self.stream.fileno()

    def read(self, size):
        self.reads += 1
        self.hook(self.reads)
        return self.stream.read(size)


class LedgerSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'process-ledger.json'
        self.document = dict(run_id='constructed', jobs={'trainer': {'pid': 12}},
                             padding='x' * (9 * 1024 * 1024))
        self.raw = json.dumps(self.document).encode()
        self.path.write_bytes(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_large_snapshot_hash_and_identity(self):
        document, proof = read_owned_ledger_snapshot(self.path)
        self.assertEqual(document, self.document)
        self.assertEqual(proof['sha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(proof['bytes'], len(self.raw))
        self.assertEqual(proof['uid'], os.getuid())
        self.assertEqual(proof['inode'], self.path.stat().st_ino)
        self.assertEqual(proof['read_attempts'], 1)

    def test_missing_open_retries_only_the_read(self):
        real_open = os.open
        calls = []

        def transient(path, flags):
            calls.append(flags)
            if len(calls) == 1:
                raise FileNotFoundError(errno.ENOENT, 'constructed atomic replacement gap')
            return real_open(path, flags)

        with patch('reproduction.common8.ledger_snapshot.os.open', side_effect=transient), \
                patch('reproduction.common8.ledger_snapshot.time.sleep') as sleep:
            document, proof = read_owned_ledger_snapshot(self.path)
        self.assertEqual(document, self.document)
        self.assertEqual(proof['read_attempts'], 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertTrue(all(flags & os.O_NOFOLLOW for flags in calls))

    def test_failure_after_first_8mib_reopens_without_mixing_old_prefix(self):
        real_fdopen, real_open = os.fdopen, os.open
        replacement = json.dumps(dict(run_id='new-snapshot', jobs={})).encode()
        for code in (errno.ENOENT, errno.ESTALE):
            with self.subTest(errno=code):
                self.path.write_bytes(self.raw)
                descriptors, opened = [], []

                def tracked_open(path, flags):
                    fd = real_open(path, flags)
                    descriptors.append(fd)
                    return fd

                def wrapped(fd, mode):
                    stream = real_fdopen(fd, mode)
                    opened.append(fd)
                    if len(opened) != 1:
                        return stream

                    def hook(reads):
                        if reads == 2:
                            temp = self.path.with_suffix('.replacement')
                            temp.write_bytes(replacement)
                            os.replace(temp, self.path)
                            raise OSError(code, 'constructed late NFS read failure')

                    return _ReadHook(stream, hook)

                with patch('reproduction.common8.ledger_snapshot.os.open', side_effect=tracked_open), \
                        patch('reproduction.common8.ledger_snapshot.os.fdopen', side_effect=wrapped), \
                        patch('reproduction.common8.ledger_snapshot.time.sleep'):
                    document, proof = read_owned_ledger_snapshot(self.path)
                self.assertEqual(document['run_id'], 'new-snapshot')
                self.assertEqual(proof['sha256'], hashlib.sha256(replacement).hexdigest())
                self.assertEqual(proof['bytes'], len(replacement))
                self.assertEqual(proof['read_attempts'], 2)
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    def test_atomic_replacement_keeps_one_complete_old_fd_snapshot(self):
        real_fdopen = os.fdopen
        replacement = b'{"run_id":"later","jobs":{}}'

        def wrapped(fd, mode):
            def hook(reads):
                if reads == 2:
                    temp = self.path.with_suffix('.replacement')
                    temp.write_bytes(replacement)
                    os.replace(temp, self.path)
            return _ReadHook(real_fdopen(fd, mode), hook)

        with patch('reproduction.common8.ledger_snapshot.os.fdopen', side_effect=wrapped):
            document, proof = read_owned_ledger_snapshot(self.path)
        self.assertEqual(self.path.read_bytes(), replacement)
        self.assertEqual(document, self.document)
        self.assertEqual(proof['sha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(proof['read_attempts'], 1)

    def test_persistent_missing_or_stale_read_still_fails_bounded(self):
        for code in (errno.ENOENT, errno.ESTALE):
            with self.subTest(errno=code), \
                    patch('reproduction.common8.ledger_snapshot.os.open',
                          side_effect=OSError(code, 'persistent constructed failure')) as opened, \
                    patch('reproduction.common8.ledger_snapshot.time.sleep') as sleep:
                with self.assertRaises(OSError) as raised:
                    read_owned_ledger_snapshot(self.path, attempts=3)
                self.assertEqual(raised.exception.errno, code)
                self.assertEqual(opened.call_count, 3)
                self.assertEqual(sleep.call_count, 2)

    def test_corrupt_json_wrong_owner_and_symlink_are_not_retried(self):
        with patch('reproduction.common8.ledger_snapshot.time.sleep') as sleep:
            with self.assertRaisesRegex(AssertionError, 'owner'):
                read_owned_ledger_snapshot(self.path, expected_uid=os.getuid() + 1)
            self.path.write_bytes(b'{broken-json')
            with self.assertRaises(json.JSONDecodeError):
                read_owned_ledger_snapshot(self.path)
            link = self.path.with_suffix('.link')
            link.symlink_to(self.path)
            with self.assertRaises(OSError):
                read_owned_ledger_snapshot(link)
            self.assertEqual(sleep.call_count, 0)

    def test_fifo_and_other_io_failures_are_fatal_without_wait(self):
        fifo = self.path.with_suffix('.fifo')
        os.mkfifo(fifo)
        with self.assertRaisesRegex(AssertionError, 'regular file'):
            read_owned_ledger_snapshot(fifo)
        for code in (errno.EACCES, errno.EIO):
            with self.subTest(errno=code), \
                    patch('reproduction.common8.ledger_snapshot.os.open',
                          side_effect=OSError(code, 'real constructed storage failure')) as opened, \
                    patch('reproduction.common8.ledger_snapshot.time.sleep') as sleep:
                with self.assertRaises(OSError):
                    read_owned_ledger_snapshot(self.path)
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(sleep.call_count, 0)

    def test_in_place_writer_breaks_atomic_publication_contract(self):
        real_fdopen = os.fdopen

        def wrapped(fd, mode):
            def hook(reads):
                if reads == 2:
                    self.path.write_bytes(b'{}')
            return _ReadHook(real_fdopen(fd, mode), hook)

        with patch('reproduction.common8.ledger_snapshot.os.fdopen', side_effect=wrapped), \
                patch('reproduction.common8.ledger_snapshot.time.sleep') as sleep:
            with self.assertRaisesRegex(AssertionError, 'changed in place'):
                read_owned_ledger_snapshot(self.path)
            self.assertEqual(sleep.call_count, 0)


if __name__ == '__main__':
    unittest.main()
