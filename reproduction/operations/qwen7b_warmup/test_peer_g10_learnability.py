"""Metadata-only deterministic grouping/input equivalence tests."""
import copy,math,tempfile,unittest
from pathlib import Path
from . import peer_g10_learnability as m
from .test_peer_newseed_ingest import trajectory

def group(k,q='q'):
 return dict(qid=q,trajectory_ids=[q+'-t%d'%i for i in range(10)],labels=[1]*k+[0]*(10-k))
def seal(v):
 v['prefix_sha256']=m.digest({k:x for k,x in v.items() if k!='prefix_sha256' and not k.startswith('feature_')});return v

class LearnabilityTests(unittest.TestCase):
 def test_hypergeometric_expected_mixed_and_class_mass(self):
  gs=[group(k,'q%d'%k) for k in (0,1,5,9,10)];v,occ=m.view_summary(gs,[[g['qid'] for g in gs]])
  probabilities={x['K']:x['uniform252_G5_Mixed_probability'] for x in v['per_source_G10_K']}
  for k,value in ((0,0),(1,.5),(5,250/252),(9,.5),(10,0)):self.assertAlmostEqual(probabilities[k],value)
  self.assertEqual(v['views'],20);self.assertEqual(sum(v['realized_G5_K_histogram']),20)
  self.assertEqual(v['realized_weighted_Mixed_fraction'],v['realized_Mixed_fraction'])
  self.assertTrue(v['per_question_total_weight_verified']);self.assertEqual(sum(v['trajectory_occurrence_count_histogram']),50)
 def test_views_are_deterministic_and_labels_never_select_IDs(self):
  g=group(3);v1=m.grouping.build_training_views([g],mode='random',views_per_question=4,seed=1729,epoch=0)
  g2=copy.deepcopy(g);g2['labels']=[0]*10;v2=m.grouping.build_training_views([g2],mode='random',views_per_question=4,seed=1729,epoch=0)
  self.assertEqual([v['trajectory_ids'] for v in v1['groups']],[v['trajectory_ids'] for v in v2['groups']])
  self.assertEqual(len({tuple(v['source_indices']) for v in v1['groups']}),4)
  self.assertEqual(m.view_summary([g],[['q']])[0],m.view_summary([g],[['q']])[0])
 def test_causal_equivalence_ignores_namespace_but_keeps_tool_context(self):
  _,row,_=trajectory(dict(run_id='synthetic',seed=1729));first=m.causal_signature(row)
  second=copy.deepcopy(row);second.update(trajectory_id='other-real-tid',run_id='other-event',seed=100,feature_file='/DO_NOT_READ_TENSOR')
  self.assertEqual(first,m.causal_signature(seal(second)))
  third=copy.deepcopy(row);seg=third['prefix_segments'][1];seg['context_tokens'][len(third['prefix_segments'][0]['context_tokens'])]+=1
  a,b=m.causal_signature(seal(third));self.assertNotEqual(a,first[0]);self.assertEqual(b,first[1])
  invalid=copy.deepcopy(row);invalid['prefix_segments'][1]['context_tokens'].append(99)
  with self.assertRaises(ValueError):m.causal_signature(seal(invalid))
 def test_exact_input_label_conflict_entropy_bound(self):
  g=group(5);s={t:('same-causal','same-token') for t in g['trajectory_ids']};v=m.collision_summary([g],s)
  self.assertEqual(v['identical_input_pairs'],45);self.assertEqual(v['identical_input_opposite_EM_pairs'],25)
  self.assertEqual(v['true_G10_Mixed_input_conflict_fraction'],1);self.assertAlmostEqual(v['finite_sample_trajectory_BCE_minimum_nats'],math.log(2))
  s={t:(t,'same-token') for t in g['trajectory_ids']};self.assertEqual(m.collision_summary([g],s)['finite_sample_trajectory_BCE_minimum_nats'],0)
 def test_actual_B32x256_plan_is_loaded_without_importing_gpu_runner(self):
  path=Path(__file__).with_name('peer_newseed_train.py');groups=[dict(qid='synthetic%d'%i) for i in range(8192)]
  plan=m.pure_training_plan(path,groups);self.assertEqual(len(plan['batches']),256);self.assertTrue(all(len(b)==32 for b in plan['batches']))
  self.assertEqual(len({q for b in plan['batches'] for q in b}),8192);self.assertEqual(set(plan['batch_epochs']),{0})
  self.assertEqual(m.validate_factory(Path(__file__).with_name('peer_newseed_fit_operation.py')),dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'))

if __name__=='__main__':unittest.main()
