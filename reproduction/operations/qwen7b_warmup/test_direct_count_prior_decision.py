import unittest
from . import direct_count_prior_decision as d
class DecisionTests(unittest.TestCase):
 def fixture(self):return [dict(qid=str(i),score=x,strict_em=y) for i,(x,y) in enumerate([(.4,[1,0,0,0,0]),(.35,[0,0,0,0,0]),(.3,[1,1,0,0,0]),(.2,[1,1,1,1,1])])]
 def test_decision_only_auc_and_ranking_fixed(self):
  r=self.fixture();a=d.metrics(r,.5);b=d.metrics(r,.33);self.assertEqual(a['Mixed_TPR'],0);self.assertEqual(a['nonMixed_TNR'],1);self.assertEqual(b['Mixed_TPR'],.5);self.assertEqual(b['nonMixed_TNR'],.5);self.assertEqual(a['AUC'],b['AUC']);self.assertEqual(d.rankings(r)['AUC'],a['AUC'])
 def test_ties_half_credit_and_same_original_scores(self):
  r=self.fixture();r[0]['score']=r[1]['score'];positives=[x for x in r if 0<sum(x['strict_em'])<5];neg=[x for x in r if sum(x['strict_em'])in(0,5)];expected=sum((x['score']>y['score'])+.5*(x['score']==y['score']) for x in positives for y in neg)/(len(positives)*len(neg));self.assertEqual(d.auc(r),expected)
if __name__=='__main__':unittest.main()
