"""Narrow CPU fixtures for canonical whole-G5 choice; no model/data assets."""
import copy,importlib.util,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('intake',Path(__file__).with_name('hpc3_tail_sealed_intake.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def candidate(q='q',run='execution-a',priority=10):
 return dict(qid=q,question_hash='a'*64,trajectory_ids=[q+'-t%d'%i for i in range(5)],labels=[0,1,0,1,0],metadata_verified=True,execution_run_id=run,execution_host_cluster='HPC3',source_priority=priority,seal={'path':'/'+run+'/seal.json'})
class WholeG5(unittest.TestCase):
 def test_label_blind_whole_source_choice(self):
  a=candidate();b=candidate(run='execution-b',priority=20);b['trajectory_ids']=['b-t%d'%i for i in range(5)]
  selected,dropped=m.choose_whole_g5([b,a]);self.assertIs(selected[0],a);self.assertEqual(len(dropped),1)
  changed=copy.deepcopy([b,a]);changed[0]['labels']=[1]*5;changed[1]['labels']=[0]*5
  chosen,_=m.choose_whole_g5(changed);self.assertEqual(chosen[0]['execution_run_id'],a['execution_run_id']);self.assertEqual(chosen[0]['trajectory_ids'],a['trajectory_ids'])
 def test_canonical_distinct_qids_same_text_retained(self):
  a=candidate('q1');b=candidate('q2');selected,_=m.choose_whole_g5([b,a]);self.assertEqual(len(selected),2)
 def test_partial_or_conflicting_or_duplicate_ownids_rejected(self):
  a=candidate();a['trajectory_ids']=a['trajectory_ids'][:4]
  with self.assertRaises(ValueError):m.choose_whole_g5([a])
  a=candidate();b=candidate(run='b');b['question_hash']='b'*64
  with self.assertRaises(ValueError):m.choose_whole_g5([a,b])
  a=candidate('q1');b=candidate('q2');b['trajectory_ids']=a['trajectory_ids']
  with self.assertRaises(ValueError):m.choose_whole_g5([a,b])
if __name__=='__main__':unittest.main()
