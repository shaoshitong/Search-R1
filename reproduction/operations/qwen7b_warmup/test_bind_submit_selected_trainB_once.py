"""Tiny controller reconciliation tests, no actual metadata, SSH or Slurm calls."""
import tempfile,unittest
from pathlib import Path
from . import bind_submit_selected_trainB_once as m
class BindSubmitTest(unittest.TestCase):
 def test_terminal_exact_job_and_zero_exit(self):
  self.assertTrue(m.terminal_rows('749935|COMPLETED|0:0\n','749935'));self.assertFalse(m.terminal_rows('749935|RUNNING|0:0','749935'));self.assertFalse(m.terminal_rows('749935|COMPLETED|1:0','749935'));self.assertFalse(m.terminal_rows('749936|COMPLETED|0:0','749935'))
 def test_same_name_owner_exact_and_multiple_rejected(self):
  self.assertEqual(m.matching_queue_ids('101|name|mou176\n102|other|mou176\n103|name|other','name'),['101'])
  with self.assertRaises(RuntimeError):m.matching_queue_ids('101|name|mou176\n102|name|mou176','name')
 def test_job_comment_entry_identity(self):
  good='JobId=101 JobName=name UserId=mou176(1) Command=/remote/e.sh /remote/cfg sha Comment=tag'
  self.assertEqual(m.verify_job(good,'101','/remote/e.sh','name','tag'),'101')
  with self.assertRaises(RuntimeError):m.verify_job(good,'101','/remote/wrong.sh','name','tag')
  with self.assertRaises(RuntimeError):m.verify_job(good,'101','/remote/e.sh','name','wrong')
 def test_unknown_intent_never_recommends_retry(self):
  result=m.unresolved(dict(config='same'),'101|name|COMPLETED|0:0');self.assertFalse(result['Slurm_submitted']);self.assertTrue(result['requires_root_reconciliation']);self.assertIn('NO_RESUBMISSION',result['status'])
 def test_durable_once_reuse_and_conflict(self):
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'intent.json';m.durable_once(p,dict(config='one'));m.durable_once(p,dict(config='one'))
   with self.assertRaises(AssertionError):m.durable_once(p,dict(config='two'))
if __name__=='__main__':unittest.main()
