"""Tiny CPU file-copy checks; no model, GPU, deployment or old-copy replay."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from .checkpoint_backup import exported, full_read, collect_hashes


class BackupTests(unittest.TestCase):
    def test_rejects_local_or_traversal_sources(self):
        self.assertEqual(str(exported('/mnt/data/shaoshitong/checkpoint')), '/mnt/shared/workspace/training_data_node_8/shaoshitong/checkpoint')
        for value in('/Users/local/weights','/mnt/data/../secret','relative'):
            with self.assertRaises(ValueError):exported(value)

    def test_actual_copy_and_independent_destination_read(self):
        with tempfile.TemporaryDirectory()as tmp:
            source=Path(tmp)/'source';target=Path(tmp)/'target';payload=b'actual-checkpoint-bytes'*170
            source.write_bytes(payload)
            with target.open('xb')as output:copied=full_read(source,output,rate=10**9)
            independent=full_read(target,rate=10**9)
            self.assertEqual(copied,independent);self.assertEqual(copied,(len(payload),hashlib.sha256(payload).hexdigest()))

    def test_source_proofs_cannot_conflict(self):
        rows={};collect_hashes([dict(path='/mnt/data/file',bytes=2,sha256='a')]*2,rows)
        with self.assertRaises(ValueError):collect_hashes(dict(path='/mnt/data/file',bytes=2,sha256='b'),rows)


if __name__=='__main__':unittest.main()
