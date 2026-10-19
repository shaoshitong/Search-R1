"""Offline safety tests; no SSH, HF requests, model staging, or GPU allocation."""
import sys
import io
import json
import hashlib
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import unittest

from reproduction.grounding_eval.upload_h100_hf_stream import (
    MAX_READ, RemoteFile, metadata_match, upload_selected,
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


class ParallelCommitSafety(unittest.TestCase):
    def fixture(self):
        payloads={('file-%02d.json'%i):('file %02d payload'%i).encode() for i in range(13)}
        records=[dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in payloads.items()]
        fresh=[dict(path=r['name'],bytes=r['bytes'],sha256=r['sha256'],mtime_ns=0,inode=0) for r in records]
        plan=dict(repo='Sutoonq/TMP_MODEL',prefix='grounding-eval-20261019/test')
        streams=[]
        class Stream(io.BytesIO):
            def __init__(self,cfg,bound,timeout,progress):
                super().__init__(payloads[bound['path']]);self.progress=progress;self.busy=threading.Lock();streams.append(self)
            def read(self,n=-1):
                if not self.busy.acquire(False):raise AssertionError('Concurrent seek/read of the same file')
                try:
                    b=super().read(n);self.progress(self.tell());return b
                finally:self.busy.release()
        return payloads,records,fresh,plan,streams,Stream

    def test_single_commit_independent_file_threads_and_verified_accounting(self):
        payloads,records,fresh,plan,streams,Stream=self.fixture();test=self
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);status=dict(receipts=[],verified_bytes=0)
            class API:
                uploaded=False;calls=0
                def model_info(self,*a,**kw):return SimpleNamespace(sha='a'*40)
                def get_paths_info(self,repo,paths,**kw):
                    if not self.uploaded:return []
                    return [SimpleNamespace(path=p,size=len(payloads[p.rsplit('/',1)[-1]]),lfs={'sha256':hashlib.sha256(payloads[p.rsplit('/',1)[-1]]).hexdigest()})for p in paths]
                def create_commit(self,repo,operations,num_threads,**kw):
                    self.calls+=1;test.assertEqual(len(operations),13);test.assertEqual(num_threads,3)
                    test.assertEqual(len({id(o.path_or_fileobj)for o in operations}),13)
                    saved=json.loads((root/'status.json').read_text());test.assertEqual(saved['verified_bytes'],0)
                    barrier=threading.Barrier(3)
                    def upload(pair):
                        i,op=pair
                        if i<3:barrier.wait(timeout=3)
                        b=bytearray()
                        while True:
                            x=op.path_or_fileobj.read(3)
                            if not x:break
                            b.extend(x)
                        test.assertEqual(hashlib.sha256(b).digest(),op.upload_info.sha256)
                    with ThreadPoolExecutor(max_workers=num_threads)as pool:list(pool.map(upload,enumerate(operations)))
                    self.uploaded=True;return SimpleNamespace(oid='b'*40)
            api=API();revision=upload_selected(api,lambda **kw:SimpleNamespace(**kw),lambda **kw:SimpleNamespace(**kw),plan,records,fresh,{},root,status,1,3,Stream)
            self.assertEqual(revision,'b'*40);self.assertEqual(api.calls,1);self.assertEqual(len(status['receipts']),13)
            self.assertEqual(status['verified_bytes'],sum(len(p)for p in payloads.values()))
            self.assertEqual(set(status['file_progress']),set(payloads));self.assertTrue(all(s.closed for s in streams))

    def test_commit_error_closes_all_streams_without_verified_claim(self):
        payloads,records,fresh,plan,streams,Stream=self.fixture()
        class API:
            def model_info(self,*a,**kw):return SimpleNamespace(sha='a'*40)
            def get_paths_info(self,*a,**kw):return []
            def create_commit(self,*a,**kw):raise OSError('Synthetic upload failure')
        with tempfile.TemporaryDirectory()as directory:
            status=dict(receipts=[],verified_bytes=0)
            with self.assertRaises(OSError):upload_selected(API(),lambda **kw:SimpleNamespace(**kw),lambda **kw:SimpleNamespace(**kw),plan,records,fresh,{},Path(directory),status,1,3,Stream)
            self.assertEqual(status['verified_bytes'],0);self.assertEqual(status['receipts'],[]);self.assertEqual(len(streams),13)
            self.assertTrue(all(s.closed for s in streams))


if __name__ == '__main__':
    unittest.main()
