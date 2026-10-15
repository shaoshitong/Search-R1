import hashlib
from pathlib import Path
import tempfile
import unittest
from reproduction.h1008_qprobe.stage_official_backup import copy_file

class StageOfficialTests(unittest.TestCase):
    def test_complete_source_and_staged_readback_match_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            s=Path(d)/'source';t=Path(d)/'staged';s.write_bytes(b'small fixture')
            sha=hashlib.sha256(s.read_bytes()).hexdigest();r=copy_file(s,t,sha,rate=None)
            self.assertEqual(r['source_full_read_sha256'],r['staged_full_read_sha256'])
            with self.assertRaises(FileExistsError):copy_file(s,t,sha,rate=None)
    def test_corrupt_source_preserves_original_and_partial_without_pass(self):
        with tempfile.TemporaryDirectory() as d:
            s=Path(d)/'source';t=Path(d)/'staged';s.write_bytes(b'small fixture')
            with self.assertRaisesRegex(ValueError,'Official source full SHA'):copy_file(s,t,'0'*64,rate=None)
            self.assertTrue(s.exists());self.assertTrue(t.exists())

if __name__=='__main__':unittest.main()
