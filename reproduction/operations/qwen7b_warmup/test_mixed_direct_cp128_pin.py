import json,hashlib,os,tempfile,unittest
from pathlib import Path
from .mixed_direct_cp128_pin import pin,KEYS,small
class PinTests(unittest.TestCase):
 def setup_source(self,folder,cursor=128,published=True):
  m=Path(folder)/'main';s=Path(folder)/'snapshot';(m/'checkpoints').mkdir(parents=True);cfg=dict(base=str(m),initialization_source='official_initial',fit_questions=169028,update_budget=21132,loss={'binary_mixed_weight':1.});(m/'config.json').write_text(json.dumps(cfg));ref=small(m/'config.json')[1];p=m/'checkpoints'/('.rolling-accepted-cursor-%06d-fixture.pt'%cursor);p.write_bytes(b'synthetic metadata-only hardlink fixture, not a model');r=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cursor=cursor,native_steps=cursor,head_steps=cursor,config_sha256=ref['sha256'],allocation_job_id='fake',full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed=True,accepted_hashes={k:'a'*64 for k in KEYS});(m/'latest.json').write_text(json.dumps(r));(m/'mixed-checkpoint-transaction.json').write_text(json.dumps(dict(phase='published'if published else'verified_new_body_before_publish',new_checkpoint=r)));return m,s,ref,p
 def test_hardlink_survives_source_reclamation_and_config_bytes_exact(self):
  with tempfile.TemporaryDirectory()as d:
   m,s,ref,p=self.setup_source(d);r=pin(m,s,ref,'fake');q=Path(r['checkpoint']['path']);self.assertTrue(os.path.samefile(p,q));self.assertEqual((m/'config.json').read_bytes(),(s/'source-config.json').read_bytes());p.unlink();self.assertTrue(q.exists());self.assertEqual(pin(m,s,ref,'fake')['status'],'CP128_IMMUTABLE_HARDLINK_ALREADY_ACCEPTED')
 def test_wait_for_full_cpu_verified_publication_and_reject_wrong_clocks(self):
  with tempfile.TemporaryDirectory()as d:
   m,s,ref,p=self.setup_source(d,published=False);self.assertEqual(pin(m,s,ref,'fake')['status'],'WAITING_FOR_CP128_ATOMIC_PUBLICATION');self.assertFalse((s/'inputs/checkpoint-cp128.pt').exists())
   x=json.loads((m/'latest.json').read_text());x['native_steps']=127;(m/'latest.json').write_text(json.dumps(x))
   with self.assertRaises(AssertionError):pin(m,s,ref,'fake')
 def test_wait_before128_and_reject_missed_window_without_substitution(self):
  with tempfile.TemporaryDirectory()as d:
   m,s,ref,p=self.setup_source(d,7);self.assertEqual(pin(m,s,ref,'fake')['status'],'WAITING_FOR_ACCEPTED_MAIN_CP128')
  with tempfile.TemporaryDirectory()as d:
   m,s,ref,p=self.setup_source(d,256)
   with self.assertRaises(ValueError):pin(m,s,ref,'fake')
if __name__=='__main__':unittest.main()
