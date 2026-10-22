"""Asset-free check that same protocol tid does not imply physical identity."""
import unittest
from .resolve_full169028_physical_sources import payload,core
class PhysicalPayloadTest(unittest.TestCase):
 def test_archival_relocation_keeps_payload_identity(self):
  self.assertEqual(payload(dict(path='old',bytes=3,sha256='a')),payload(dict(path='new',bytes=3,sha256='a')))
 def test_same_tid_different_execution_payload_is_not_equal(self):
  self.assertNotEqual(payload(dict(path='donor',bytes=3,sha256='a')),payload(dict(path='hpc3',bytes=3,sha256='b')))
 def test_shared_protocol_tid_and_prefix20_even_same_em_do_not_choose_payload(self):
  common=dict(qid='q',trajectory_id='same-protocol-tid',actor='official7b',policy20=list(range(20)),strict_em=1)
  donor=dict(common,policy80=list(range(80)),full=dict(path='donor',bytes=100,sha256='donor-sha'))
  chosen=dict(common,policy80=list(range(20))+list(range(100,160)),full=dict(path='chosen-hpc3',bytes=100,sha256='chosen-sha'))
  sealed=core(chosen['full']);self.assertEqual(donor['policy20'],chosen['policy20']);self.assertEqual(donor['strict_em'],chosen['strict_em']);self.assertNotEqual(donor['policy80'],chosen['policy80'])
  candidates=[donor,chosen];accepted=[r for r in candidates if core(r['full'])==sealed];self.assertEqual(accepted,[chosen]);self.assertNotEqual(payload(donor['full']),payload(sealed))
 def test_full_ref_binds_path_size_and_hash(self):
  self.assertEqual(core(dict(path='p',bytes=2,sha256='s',count=5)),dict(path='p',bytes=2,sha256='s'))
if __name__=='__main__':unittest.main()
