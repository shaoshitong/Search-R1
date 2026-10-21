import copy
import importlib.util
from pathlib import Path
import unittest
P=Path(__file__).with_name('hpc3_tail_g5.py');s=importlib.util.spec_from_file_location('tail',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TailChecks(unittest.TestCase):
 def test_original_two_shard_to_eight_shard_ordinals(self):
  o=m.source_ordinals(dict(source_shard=1,source_batch=390,questions=128))
  self.assertEqual(o[0],99841);self.assertEqual(o[-1],100095)
  self.assertEqual(len(o),len(set(o)))
  for n in o:self.assertEqual((n%8)+8*(128*((n//8)//128)+(n//8)%128),n)
 def test_sampling_and_execution_identities_cannot_be_confused(self):
  c=dict(role=m.ROLE,base=str(m.BASE),run_id=m.DONOR_RUN,sampling_protocol_source_run_id=m.DONOR_RUN,sampling_attempt_id=m.ATTEMPT,seed=20261021,model_revision=m.REVISION,execution_host_cluster='HPC3',execution_run_id='a'*32,potentialOverlap=True,shared_donor_atomic_claim=False,actor_optimizer_updates=0,actor_updates_authorized=False,warmup_only=True,actor_hidden_native=False,capture_prefix_tokens=20,question_max_tokens=273,workers=2)
  self.assertEqual(m.validate(c)['run_id'],m.DONOR_RUN)
  for key,value in [('execution_run_id',m.DONOR_RUN),('potentialOverlap',False),('shared_donor_atomic_claim',True),('seed',1729),('actor_optimizer_updates',1)]:
   bad=copy.deepcopy(c);bad[key]=value
   with self.assertRaises(ValueError):m.validate(bad)
 def test_two_normal_jobs_no_dependency_or_submit_action(self):
  for w in(0,1):
   text=m.render_sbatch(m.BASE,Path('/data/code'),Path('/data/qualified'),w)
   for expected in ['--gres=gpu:2','--qos=normal','--partition=acd_u','--time=08:00:00','--worker '+str(w)]:self.assertIn(expected,text)
   self.assertNotIn('dependency',text);self.assertNotIn('sbatch ',text)
  with self.assertRaises(ValueError):m.render_sbatch(m.BASE,Path('/data/code'),Path('/data/qualified'),2)
if __name__=='__main__':unittest.main()
