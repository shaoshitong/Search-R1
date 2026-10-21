"""Small identity/whole-event tests; no model, tensor, data or network."""
import importlib.util,pathlib,unittest,copy
s=importlib.util.spec_from_file_location('donor_prepare',pathlib.Path(__file__).with_name('prepare_donor14080_l26.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def group(q,hash_value=None):
 ids=['old-'+q+'-'+str(i)for i in range(5)]+['new-'+q+'-'+str(i)for i in range(5)]
 return dict(qid=q,question_hash=hash_value or q,trajectory_ids=ids,labels=[0]*10,real_G5_events=[dict(event='original',trajectory_ids=ids[:5]),dict(event='newseed20261021',trajectory_ids=ids[5:])])
class Tests(unittest.TestCase):
 def test_canonical_qid_excludes_entire_different_new_payload(self):
  a=[group('a'),group('b'),group('c')];tail=[group('b'),group('x')];tail[0]['labels']=[1]*10
  chosen,proof=m.select_groups(a,tail,(3,2,1,2));self.assertEqual([g['qid']for g in chosen],['a','c']);self.assertFalse(proof['selected_labels_or_scores_consulted']);self.assertEqual([len(g['trajectory_ids'])for g in chosen],[10,10])
 def test_training_text_hash_repeat_retained_without_qid_dedup(self):
  chosen,_=m.select_groups([group('a','same'),group('b','same')],[group('x')],(2,1,0,2));self.assertEqual(len(chosen),2)
 def test_duplicate_ids_or_mixed_event_order_rejected(self):
  bad=[group('a'),group('a')]
  with self.assertRaises(ValueError):m.select_groups(bad,[group('x')],(2,1,0,2))
  bad=[group('a')];bad[0]['real_G5_events'][1]['trajectory_ids'][0]='foreign'
  with self.assertRaises(ValueError):m.select_groups(bad,[group('x')],(1,1,0,1))
 def test_physical_execution_and_no_future_payload(self):
  row=dict(qid='a',trajectory_id='new-a-0',prefix_sha256='f'*64);source=dict(event='newseed20261021',prefix={'path':str(m.DONOR_RAW/'source/prefix.jsonl.gz')},seal={'path':str(m.DONOR_RAW/'source/complete.json')});expected=dict(row,event='newseed20261021');seal=dict(run_id=m.RUN,model_revision=m.REVISION,actor_optimizer_updates=0)
  binding=m.physical_binding(row,source,'newseed20261021',expected,seal);self.assertEqual(binding['canonical_prefix_row_sha256'],m.digest(row))
  with self.assertRaises(ValueError):m.physical_binding(dict(row,strict_em=1),source,'newseed20261021',expected,seal)
  with self.assertRaises(ValueError):m.physical_binding(row,source,'newseed20261021',expected,dict(seal,execution_host_cluster='HPC3'))
if __name__=='__main__':unittest.main()
