"""Bounded CPU metadata tests; no real CP, final data, network or Slurm."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import control_selected_fullP80_once as m
class ControlTest(unittest.TestCase):
 def candidate(self):
  choices=[dict(epoch=e,state=s,Top512_Mixed=300 if e in(3,4)else 200,evaluation={'path':str(e)})for e in range(1,11)for s in('raw','EMA')]
  selected=next(r for r in choices if(r['epoch'],r['state'])==(3,'EMA'))
  return dict(status='ACTUAL_FULL169028_P80_FAMILY_SELECTED_AND_VIEWED_PAIRED_ANALYSIS_COMPLETE',selection_rule='maxseed1count_then_earliest_epoch_then_EMA',seed2_used_for_selection=False,same_epoch_state_on_seed2=True,final4096_read=False,all20_seed1_choices=choices,actual_selected=selected,selected_epoch=3,selected_state='EMA')
 def test_actual_seed1_rule_all20_ties_and_wrong_epoch(self):
  v=self.candidate();self.assertEqual(m.selection(v),(3,'EMA'))
  for change in(dict(selected_epoch=4),dict(seed2_used_for_selection=True),dict(all20_seed1_choices=v['all20_seed1_choices'][:-1])):
   with self.assertRaises(ValueError):m.selection(dict(v,**change))
 def test_selected_acceptance_epoch_source_clock_exact(self):
  binding={'same':'actual'};cp=dict(path=str(m.PARENTS['seed1']/'checkpoint-epoch-03.pt'),bytes=3773200000,sha256='a'*64)
  v=dict(status=m.ACCEPT_STATUS,job_id='750189',epoch=3,batch_attempts=3963,actual_Adam_updates=3963,EMA_updates=3963,epoch_complete=True,next_epoch=4,within_next_epoch_question_position=0,execution_phase='epoch_complete_pre_evaluation',CPU_only=True,final4096_read=False,binding=binding,checkpoint=cp,epoch_plan_SHAs={'3':'b'*64},models={'raw':{},'EMA':{}},RNG_by_rank=[{},{}],Adam_CPU_restore=True)
  self.assertEqual(m.accepted_metadata(v,'seed1',3,dict(binding=binding)),cp)
  for field,value in(('epoch',4),('binding',{'wrong':'source'}),('actual_Adam_updates',3964)):
   with self.assertRaises(ValueError):m.accepted_metadata(dict(v,**{field:value}),'seed1',3,dict(binding=binding))
 def test_not_ready_never_freezes_or_writes(self):
  with patch.object(m,'prerequisite_gate',side_effect=m.Waiting('training')),patch.object(m,'durable_once')as write,patch.object(m,'final_binding')as freeze:
   result=m.execute('freeze-final');self.assertIn('WAITING',result['status']);write.assert_not_called();freeze.assert_not_called()
 def test_unknown_submission_intent_never_calls_sbatch(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d);gpu=base/'gpu';gpu.mkdir();entry=gpu/'audit.sbatch';entry.write_text('#SBATCH --job-name=owned\n');cfg=base/'config.json';cfg.write_text('{}');eref=m.proof(entry);pub=dict(path=str(gpu/'publication'),bytes=2,sha256='a'*64)
   m.durable_once(base/'audit-submit-intent.json',dict(time=100,name='owned',entry=eref,config=m.proof(cfg)))
   commands=[]
   def fake(argv):
    commands.append(argv);return ''if argv[0]=='squeue'else'42|owned|COMPLETED|0:0'
   with patch.object(m,'BASE',base),patch.object(m,'target',return_value=(dict(base=gpu),pub,eref)),patch.object(m,'command',side_effect=fake):r=m.submit_once('audit',cfg)
   self.assertIn('NO_RESUBMISSION',r['status']);self.assertFalse(any(c[0]=='sbatch'for c in commands))
 def test_owned_queue_reconciliation_and_hash_boundary(self):
  self.assertEqual(m.matching_queue_ids('11|owned|mou176\n12|other|mou176','owned'),['11'])
  with self.assertRaises(RuntimeError):m.matching_queue_ids('11|owned|mou176\n12|owned|mou176','owned')
  s='JobId=11 JobName=owned UserId=mou176(1) Command=/remote/audit.sbatch cfg sha Comment=exact'
  self.assertEqual(m.verify_job(s,'11','/remote/audit.sbatch','owned','exact'),'11')
  with self.assertRaises(RuntimeError):m.verify_job(s,'11','/remote/audit.sbatch','owned','wrong')
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'checkpoint.pt';p.write_bytes(b'not tensor')
   with self.assertRaises(ValueError):m.proof(p)
if __name__=='__main__':unittest.main()
