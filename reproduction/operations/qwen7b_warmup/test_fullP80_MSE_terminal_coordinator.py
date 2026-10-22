"""Small metadata-only completion gates, no model or CP files."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import fullP80_MSE_terminal_CPU_coordinator as c
def completed(job):
 init,plan=c.JOBS[job];return dict(status='FULL169028_P80_OWNG5_MSE_TRAINING_COMPLETE',epochs=10,cursor=13210,batch_attempts=13210,actual_Adam_updates=13210,EMA_updates=13210,margin_satisfied_batches=0,training_questions=169028,final4096_scored=False,binding=dict(init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228,training_questions=169028,training_scope='full169028_fresh_official_actor_not_pilot_resume',fixed_m=5,layer=15,prefix=80,objective='mse'))
class TerminalGate(unittest.TestCase):
 def test_both_exact_job_seed_completions_pass(self):
  for job in c.JOBS:c.validate_complete(completed(job),job)
 def test_wrong_seed_or_incomplete_budget_rejected(self):
  with self.assertRaises(ValueError):c.validate_complete(completed('750189'),'750190')
  bad=completed('750190');bad['epochs']=1
  with self.assertRaises(ValueError):c.validate_complete(bad,'750190')
 def test_running_waits_failed_records_success_requires_complete(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d);(base/'root-submission.json').write_text(json.dumps(dict(job_id=750190)))
   with patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='750190|RUNNING|0:0\n',stderr='')):self.assertFalse(c.inspect_terminal(base,'750190')[0])
   with patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='750190|FAILED|1:0\n',stderr='')):
    with self.assertRaises(RuntimeError):c.inspect_terminal(base,'750190')
   (base/'complete.json').write_text(json.dumps(completed('750190')))
   with patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='750190|COMPLETED|0:0\n',stderr='')):self.assertTrue(c.inspect_terminal(base,'750190')[0])
 def test_one_FileNotFound_then_recovery_continues_same_job(self):
  with tempfile.TemporaryDirectory()as d:
   out=Path(d);observation=dict(job_id='750190',accounting='750190|COMPLETED|0:0',actual_complete=completed('750190'),rank_progress={})
   with patch.object(c,'inspect_terminal',side_effect=[FileNotFoundError('transient rank-progress rename'),(True,observation)])as query,patch.object(c.time,'sleep')as sleep:
    self.assertEqual(c.wait_for_terminal(out,'750190',out),observation);self.assertEqual(query.call_count,2);sleep.assert_called_once_with(30)
   row=json.loads((out/'observation-errors.jsonl').read_text());self.assertEqual(row['error_type'],'FileNotFoundError');self.assertFalse((out/'failure.json').exists());self.assertEqual(json.loads((out/'progress.json').read_text())['observation_errors'],1)
 def test_JSON_OS_and_sacct_transient_errors_are_recorded_then_recovered(self):
  errors=[OSError('ESTALE'),json.JSONDecodeError('temporary incomplete JSON','',0),c.subprocess.TimeoutExpired('sacct',15),c.ObservationUnavailable('sacct temporarily unavailable')]
  with tempfile.TemporaryDirectory()as d:
   out=Path(d);observed=dict(job_id='750190',accounting='750190|COMPLETED|0:0',actual_complete=completed('750190'),rank_progress={})
   with patch.object(c,'inspect_terminal',side_effect=errors+[(True,observed)]),patch.object(c.time,'sleep'):
    self.assertEqual(c.wait_for_terminal(out,'750190',out),observed)
   self.assertEqual(len((out/'observation-errors.jsonl').read_text().splitlines()),4)
 def test_real_training_failed_or_timeout_is_not_swallowed_after_retry(self):
  for state in('FAILED','TIMEOUT'):
   with tempfile.TemporaryDirectory()as d:
    base=Path(d);(base/'root-submission.json').write_text(json.dumps(dict(job_id=750190)));failure=SimpleNamespace(returncode=0,stdout='750190|'+state+'|1:0\n',stderr='')
    with patch.object(c.subprocess,'run',side_effect=[FileNotFoundError('transient observation'),failure]),patch.object(c.time,'sleep')as sleep:
     with self.assertRaises(c.TrainingTerminalFailure):c.wait_for_terminal(base,'750190',base)
     sleep.assert_called_once_with(30)
if __name__=='__main__':unittest.main()
