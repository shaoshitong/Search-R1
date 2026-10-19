"""Offline safety tests; no SSH, HF requests, model staging, or GPU allocation."""
import sys
import unittest

from reproduction.grounding_eval.upload_h100_hf_stream import (
    MAX_READ, RemoteFile, metadata_match,
)


class SourceStreamSafety(unittest.TestCase):
    def setUp(self):
        self.payload = b'0123456789' * 1000
        self.record = dict(path='/unused', bytes=len(self.payload), mtime_ns=0, inode=0)

    def factory(self, offset):
        code = 'import sys;sys.stdout.buffer.write(%r[%d:])' % (self.payload, offset)
        return [sys.executable, '-c', code]

    def test_reopened_seek_eof_and_source_bounds(self):
        with RemoteFile({}, self.record, 2, command_factory=self.factory) as stream:
            self.assertEqual(stream.read(100), self.payload[:100])
            self.assertEqual(stream.seek(-10, 1), 90)
            self.assertEqual(stream.read(21), self.payload[90:111])
            stream.seek(-10, 2)
            self.assertEqual(stream.read(), self.payload[-10:])
            self.assertEqual(stream.read(), b'')
            with self.assertRaises(ValueError):
                stream.seek(self.record['bytes'] + 1)

    def test_no_unbounded_model_materialization(self):
        with RemoteFile({}, dict(self.record, bytes=MAX_READ + 1), 2,
                        command_factory=self.factory) as stream:
            with self.assertRaises(ValueError):
                stream.read()
            self.assertIsNone(stream.process)

    def test_read_deadline_stops_only_own_child(self):
        factory = lambda offset: [sys.executable, '-c', 'import time;time.sleep(5)']
        with RemoteFile({}, self.record, .1, command_factory=factory) as stream:
            with self.assertRaises(TimeoutError):
                stream.read(1)
            self.assertIsNone(stream.process)

    def test_short_source_does_not_advance_or_get_verified(self):
        factory = lambda offset: [sys.executable, '-c', 'print("x",end="")']
        with RemoteFile({}, self.record, 2, command_factory=factory) as stream:
            with self.assertRaises(IOError):
                stream.read(10)
            self.assertEqual(stream.tell(), 0)

    def test_git_and_lfs_identities_are_distinct(self):
        record = dict(bytes=10, sha256='a' * 64)
        self.assertTrue(metadata_match(dict(size=10, lfs=dict(sha256='a' * 64)), record))
        self.assertFalse(metadata_match(dict(size=10, blob_id='a' * 64), record))
        self.assertTrue(metadata_match(dict(size=10, blob_id='b' * 40), record, 'b' * 40))
        self.assertFalse(metadata_match(dict(size=11, lfs=dict(sha256='a' * 64)), record))


if __name__ == '__main__':
    unittest.main()
