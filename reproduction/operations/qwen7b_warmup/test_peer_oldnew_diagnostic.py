import copy,math,unittest
from . import peer_oldnew_diagnostic as d
class PairedCPU(unittest.TestCase):
 def row(self,q,y,ps):
  return dict(qid=q,trajectory_ids=[q+'-'+str(i) for i in range(5)],y=y,p=ps,z=[math.log(p/(1-p)) for p in ps],k=sum(y),q=d.qscore(ps))
 def test_identity_strata_label_blind_and_disjoint(self):
  qids=['q%d'%i for i in range(8)];a=d.plan_ids(qids,stratum_size=2);b=d.plan_ids(list(reversed(qids)),stratum_size=2)
  self.assertEqual(a,b);self.assertEqual(len(set(q for g in a['groups'] for q in g['qids'])),8)
  self.assertFalse(a['labels_consulted']);self.assertFalse(a['probabilities_consulted'])
 def test_known_group_and_trajectory_metrics(self):
  rows=[self.row('a',[0]*5,[.1]*5),self.row('b',[1,0,1,0,0],[.9,.1,.8,.2,.3]),self.row('c',[1]*5,[.9]*5)]
  m=d.group_metrics(rows);self.assertEqual(m['mixed_balanced_accuracy'],1);self.assertEqual(m['mixed_roc_auc'],1)
  t=d.trajectory_metrics(rows);self.assertEqual(t['trajectory_roc_auc'],1);self.assertEqual(t['within_true_Mixed']['question_mean_pairwise_trajectory_AUC'],1)
  self.assertGreater(t['within_true_Mixed']['question_mean_correct_minus_wrong_probability'],0)
  self.assertAlmostEqual(d.bce(1000,1),0);self.assertAlmostEqual(d.bce(-1000,0),0)
 def test_own_em_id_and_sigmoid_validation(self):
  g=dict(qid='q',question_hash='h',trajectory_ids=['t%d'%i for i in range(10)],labels=[0]*5+[1]*5);p=[.1]*5+[.9]*5
  r=dict(qid='q',question_hash='h',trajectory_ids=g['trajectory_ids'],strict_em=g['labels'],p_j=p,logits=[math.log(x/(1-x)) for x in p],predicted_mixed_G10=d.qscore(p))
  a,b=d.validate_row(g,r);self.assertEqual((a['k'],b['k']),(0,5))
  for field in ('strict_em','trajectory_ids','p_j'):
   bad=copy.deepcopy(r);bad[field][0]=1 if field=='strict_em' else 'wrong' if field=='trajectory_ids' else .8
   with self.assertRaises((ValueError,TypeError)):d.validate_row(g,bad)
 def test_same_question_transition_and_error(self):
  a=[self.row('q',[0]*5,[.1]*5)];b=[self.row('q',[1,0,0,0,0],[.5]*5)]
  p=d.pairing(a,b);self.assertEqual(p['actual_Mixed_transition'],{'0->1':1});self.assertEqual(p['classification_error_transition'],{'0->0':1});self.assertFalse(p['trajectory_index_pairing_claim'])
if __name__=='__main__':unittest.main()
