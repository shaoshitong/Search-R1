import inspect,subprocess,unittest
from . import score_e2_samehf_baseline as e,deploy_e2_samehf_baseline as d
class Tests(unittest.TestCase):
 def test_actual_export_and_sbatch(self):
  script=e.render(e.TARGET/'code-test','a'*64);subprocess.run(['bash','-n'],input=script,text=True,check=True);self.assertNotIn('@@',script);self.assertIn('--time=00:20:00',script);self.assertIn('--qos=normal',script);self.assertNotIn('afterok',script)
  exports='\n'.join(x for x in script.splitlines() if x.startswith('export '));out=subprocess.check_output(['bash','-c',exports+'\nprintf "%s" "$QWEN7B_CODE_ROOT"'],text=True);self.assertEqual(out,str(e.s.TRAIN/('code-'+e.s.PIN[:12])))
 def test_readonly_same_HF_and_qualified_E2_source(self):
  text=inspect.getsource(e);self.assertIn('ReplayFeatureReader(manifests,26)',text);self.assertIn('e.validate_checkpoint_metadata',text);self.assertIn("native=state['ema_buffers']",text);self.assertNotIn('torch.optim.',text);self.assertNotIn('.backward(',text);self.assertIn('calibration_grid_member=False',text)
 def test_controller_syntax_unique_job_and_no_unknown_retry(self):
  v=dict(base=str(e.TARGET),target=str(e.TARGET),train=str(e.s.TRAIN),pin=e.s.PIN)
  for action in ('init','chunk','extract','prepare','submit','status'):compile(d.remote_source(v,action),action,'exec')
  self.assertIn("name='"+e.JOB+"'",d.SUBMIT);self.assertIn('Unknown previous submission outcome',d.SUBMIT);self.assertIn('score_e2_samehf_baseline.py',d.PREPARE)
if __name__=='__main__':unittest.main()
