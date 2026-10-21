import copy,tempfile,unittest,zipfile
from pathlib import Path
from . import direct_head_family_inputs as f
from . import direct_head_family_row_audit as rows

class InputTests(unittest.TestCase):
 def test_missing_actual_assets_never_make_refs(self):
  with tempfile.TemporaryDirectory()as tmp:
   base=Path(tmp);paths,missing=f.wait_for_assets(base);self.assertEqual(len(missing),3)
   paths['cache_complete'].parent.mkdir();paths['cache_complete'].write_text('{}')
   _,missing=f.wait_for_assets(base);self.assertEqual(set(missing),{'baseline_CP0','baseline_epoch0_evaluation'})
 def test_small_zip_scope_without_tensor_load(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp)/'toy.pt'
   with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED)as z:z.writestr('toy/data.pkl',b'metadata');z.writestr('toy/data/0',b'1234')
   self.assertFalse(f.cp0_zip(p)['torch_tensor_loaded'])
   with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED)as z:z.writestr('toy/data.pkl',b'metadata');z.writestr('toy/data/0',b'1234')
   with self.assertRaises(ValueError):f.cp0_zip(p)
 def test_coarse_six_latents_have_no_count_correctness_claim(self):
  own=dict(trajectory_ids=['t%d'%i for i in range(5)],labels=[0,0,1,0,1],domain='nq')
  sem='unadjusted eventsoftmax Mixed; tau-reference balanced decision score, not calibrated original-distribution Mixed probability'
  row=dict(qid='q',trajectory_ids=own['trajectory_ids'],strict_em=own['labels'],K=2,domain='NQ',q=.6,unadjusted_six_count_probabilities=[.1,.1,.2,.1,.2,.3],unadjusted_event_probabilities=[.4,.6],p_j=[.5]*5,coarse_event_NLL=.7)
  rows.validate_row(row,own,'binary',sem)
  row['unadjusted_event_probabilities']=[.1,.6,.3];rows.validate_row(row,own,'three_event',sem)
  bad=copy.deepcopy(row);bad['strict_em'][0]=1
  with self.assertRaises(ValueError):rows.validate_row(bad,own,'three_event',sem)
  bad=copy.deepcopy(row);bad['q']=.9
  with self.assertRaises(ValueError):rows.validate_row(bad,own,'three_event',sem)
  with self.assertRaises(ValueError):rows.validate_row(row,own,'three_event','calibrated six-count probability')

if __name__=='__main__':unittest.main()
