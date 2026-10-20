import copy,gzip,inspect,json,math,subprocess,tempfile,unittest
from pathlib import Path
from . import score_peer_g10_calibration as s
from . import deploy_peer_g10_calibration as d
class Tests(unittest.TestCase):
 def group(self,n=0):return dict(qid='q'+str(n),question_hash='%064x'%n,trajectory_ids=['q%d-t%d'%(n,i) for i in range(10)],labels=[i%2 for i in range(10)])
 def test_score_has_all_ten_own_ids_and_stable_extremes(self):
  g=self.group();z=[-1000.,1000.,0.,1.,-1.,2.,-2.,3.,-3.,4.];p=[1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v)) for v in z];row=s.score_row(g,z,p)
  self.assertEqual(row['trajectory_ids'],g['trajectory_ids']);self.assertEqual(row['strict_em'],g['labels']);self.assertEqual(len(row['logits']),10);self.assertTrue(0<=row['predicted_mixed_G10']<=1)
  changed=copy.deepcopy(g);changed['labels']=[1]*10;other=s.score_row(changed,z,p);self.assertEqual(other['p_j'],row['p_j']);self.assertEqual(other['predicted_mixed_G10'],row['predicted_mixed_G10'])
 def test_probability_mismatch_and_nonfinite_rejected(self):
  for z,p in [([float('nan')]*10,[.5]*10),([0.]*10,[1.]*10),([0.]*9,[.5]*10)]:
   with self.assertRaises(ValueError):s.score_row(self.group(),z,p)
 def test_exact8192_unique_and_order_binding(self):
  groups=[self.group(i) for i in range(8192)];r=s.identities(groups);self.assertEqual(r['trajectories'],81920)
  self.assertNotEqual(s.identities(list(reversed(groups)))['ordered_qids_sha256'],r['ordered_qids_sha256']);groups[-1]['question_hash']=groups[0]['question_hash']
  with self.assertRaises(ValueError):s.identities(groups)
 def test_complete_prediction_own_em_join(self):
  groups=[self.group(i) for i in range(3)]
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'scores.gz'
   def write(rows):
    with gzip.open(p,'wt') as f:
     for row in rows:f.write(json.dumps(row)+'\n')
   rows=[s.score_row(g,[0.]*10,[.5]*10) for g in groups];write(rows);self.assertEqual(s.verify_predictions(p,groups)['questions'],3)
   write(rows[:-1])
   with self.assertRaises(ValueError):s.verify_predictions(p,groups)
   rows[0]['strict_em']=[1]*10;write(rows)
   with self.assertRaises(ValueError):s.verify_predictions(p,groups)
 def test_render_actual_export_and_bash_syntax(self):
  text=s.render_sbatch(s.TARGET/'code-test','a'*64);self.assertNotIn('@@',text);self.assertIn('--qos=normal',text);self.assertNotIn('afterok',text);self.assertIn('--time=01:00:00',text)
  subprocess.run(['bash','-n'],input=text,text=True,check=True)
  exports='\n'.join(x for x in text.splitlines() if x.startswith('export '));out=subprocess.check_output(['bash','-c',exports+'\nprintf "%s" "$QWEN7B_CODE_ROOT"'],text=True);self.assertEqual(out,str(s.TRAIN/('code-'+s.PIN[:12])))
 def test_remote_control_boundaries_and_unknown_submit_guard(self):
  values=dict(base=str(s.TARGET),target=str(s.TARGET),train=str(s.TRAIN),pin=s.PIN)
  for action in ('init','chunk','extract','prepare','submit','status'):
   code=d.remote_source(values,action);compile(code,action,'exec');self.assertLess(len(code.encode()),16000)
  self.assertIn('submit-intent.json',d.SUBMIT);self.assertIn("['sbatch','--hold','--parsable'",d.SUBMIT);self.assertIn('EXISTING_CALIBRATION_JOB_NO_DUPLICATE',d.SUBMIT)
 def test_no_optimizer_or_local_tensor_loading(self):
  source=inspect.getsource(s);self.assertNotIn('torch.optim.',source);self.assertNotIn('.backward(',source);self.assertIn('torch.inference_mode()',source);self.assertIn('parameters_unchanged=True',source)
if __name__=='__main__':unittest.main()
