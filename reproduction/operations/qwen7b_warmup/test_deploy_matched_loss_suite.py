"""Submission environment must survive code-root expansion verbatim."""
import subprocess, unittest
from .deploy_matched_loss_suite import render_matched_sbatch

class SubmissionTemplateTest(unittest.TestCase):
 def test_environment_and_command(self):
  script=render_matched_sbatch('trajectory','/remote/run','/remote/repo-test')
  self.assertIn('export QWEN7B_CODE_ROOT=/remote/repo-test\n',script)
  self.assertIn('bash /remote/repo-test/reproduction/operations/qwen7b_warmup/runtime.sh',script)
  self.assertNotIn('__CODE_ROOT__',script)
  subprocess.run(['bash','-n'],input=script,text=True,check=True)
  exports='\n'.join(line for line in script.splitlines() if line.startswith('export '))
  p=subprocess.run(['bash','-euc',exports+'\nprintf %s "$QWEN7B_CODE_ROOT"'],text=True,capture_output=True,check=True)
  self.assertEqual(p.stdout,'/remote/repo-test')
if __name__=='__main__':unittest.main()
