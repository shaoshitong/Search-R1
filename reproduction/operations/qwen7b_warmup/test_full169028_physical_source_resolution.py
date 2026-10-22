"""Asset-free check that same protocol tid does not imply physical identity."""
import unittest
from .resolve_full169028_physical_sources import payload,core
class PhysicalPayloadTest(unittest.TestCase):
 def test_archival_relocation_keeps_payload_identity(self):
  self.assertEqual(payload(dict(path='old',bytes=3,sha256='a')),payload(dict(path='new',bytes=3,sha256='a')))
 def test_same_tid_different_execution_payload_is_not_equal(self):
  self.assertNotEqual(payload(dict(path='donor',bytes=3,sha256='a')),payload(dict(path='hpc3',bytes=3,sha256='b')))
 def test_full_ref_binds_path_size_and_hash(self):
  self.assertEqual(core(dict(path='p',bytes=2,sha256='s',count=5)),dict(path='p',bytes=2,sha256='s'))
if __name__=='__main__':unittest.main()
