"""Small CPU checks of terminal gate and the actual seeded sampler, no model."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('z_seed2_acceptor',P/'accept_full169028_Z_seed2_checkpoint.py');accept=importlib.util.module_from_spec(spec);spec.loader.exec_module(accept)
class MetadataAcceptanceTests(unittest.TestCase):
 def test_incomplete_never_opens_checkpoint(self):
  with tempfile.TemporaryDirectory()as root:
   base=Path(root)/'base';out=Path(root)/'out';base.mkdir();out.mkdir()
   with patch.object(accept,'BASE',base),patch.object(accept,'OUT',out),patch.object(accept.subprocess,'check_output',side_effect=AssertionError('No accounting before complete')):
    self.assertEqual(accept.validate(10)['status'],'WAITING_FOR_PUBLISHED_COMPLETE_NOT_OPENING_CHECKPOINT')
 def test_running_even_with_complete_never_opens_checkpoint(self):
  with tempfile.TemporaryDirectory()as root:
   base=Path(root)/'base';out=Path(root)/'out';base.mkdir();out.mkdir();(base/'root-submission.json').write_text(json.dumps(dict(job_id=749935)));(base/'complete.json').write_text('{}')
   with patch.object(accept,'BASE',base),patch.object(accept,'OUT',out),patch.object(accept.subprocess,'check_output',return_value='749935|RUNNING|0:0\n'):
    self.assertEqual(accept.validate(10)['status'],'WAITING_FOR_ACTUAL_COMPLETED_0_0_NOT_OPENING_CHECKPOINT')
 def test_actual_seeded_module_uses_explicit_plan_seed(self):
  from reproduction.operations.qwen7b_warmup.online_packed_variance_seeded import training_plan
  groups={str(q):dict(qid=str(q),trajectory_ids=['%s-%s'%(q,t)for t in range(10)],labels=[t%2 for t in range(10)])for q in range(20)}
  a=training_plan(groups,1,2026102302);b=training_plan(groups,1,2026102227);c=training_plan(groups,2,2026102302)
  self.assertNotEqual(a,b);self.assertNotEqual(a,c);self.assertEqual(a,training_plan(groups,1,2026102302));self.assertEqual(len({r['qid']for r in a}),20)
  flipped={q:dict(g,labels=[1-y for y in g['labels']])for q,g in groups.items()};f=training_plan(flipped,1,2026102302)
  self.assertEqual([(r['qid'],r['M'],r['trajectory_ids'])for r in a],[(r['qid'],r['M'],r['trajectory_ids'])for r in f])
if __name__=='__main__':unittest.main()
