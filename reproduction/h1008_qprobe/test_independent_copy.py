import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from reproduction.h1008_qprobe.independent_copy import copy_and_verify, digest

class IndependentCopyTests(unittest.TestCase):
    def test_actual_source_and_independent_target_full_reads_before_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.write_bytes(bytes(range(256))*1024)
            target=root/'target';sha=hashlib.sha256(source.read_bytes()).hexdigest()
            phases=[]
            result=copy_and_verify(source,target,sha,source.stat().st_size,rate=0,reserve=0,progress=lambda count,phase='source_copy':phases.append(phase))
            self.assertIn('independent_target_full_read',phases)
            self.assertEqual(result['source_full_read_sha256'],sha)
            self.assertEqual(result['target_full_read_sha256'],sha)
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertFalse(target.with_name('target.partial').exists())
    def test_wrong_expected_sha_cannot_publish_and_original_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.write_bytes(b'preserved original')
            target=root/'target'
            with self.assertRaisesRegex(ValueError,'SOURCE_FULL_SHA_DIFFERS'):
                copy_and_verify(source,target,'0'*64,source.stat().st_size,rate=0,reserve=0)
            self.assertFalse(target.exists());self.assertTrue(target.with_name('target.partial').exists())
            self.assertEqual(source.read_bytes(),b'preserved original')

    def test_independent_target_corruption_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.write_bytes(b'complete original')
            expected=digest(source);target=root/'target'
            def corrupt_then_read(path):
                Path(path).write_bytes(b'target corrupted after write')
                return digest(path)
            with mock.patch('reproduction.h1008_qprobe.independent_copy.digest',side_effect=corrupt_then_read):
                with self.assertRaisesRegex(ValueError,'INDEPENDENT_TARGET_FULL_SHA_DIFFERS'):
                    copy_and_verify(source,target,expected,source.stat().st_size,rate=0,reserve=0)
            self.assertFalse(target.exists());self.assertEqual(source.read_bytes(),b'complete original')

if __name__=='__main__':unittest.main()
