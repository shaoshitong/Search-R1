import unittest
from .domain_existing_predictions import domain_summary,join_domains,metrics,own_predictions

class DomainTests(unittest.TestCase):
    def test_global_selection_never_reselected_per_domain(self):
        rs=[dict(qid=str(i),strict_em=[i%2,0,0,0,0],score=1-i/600) for i in range(600)]
        domains={str(i):('nq' if i<520 else 'hotpotqa') for i in range(600)}
        m=domain_summary(rs,domains)
        self.assertEqual(m['global_Top512']['questions'],512)
        self.assertEqual(m['domains']['hotpotqa']['pool']['questions'],80)
        self.assertEqual(m['domains']['hotpotqa']['original_global_Top512_members']['questions'],0)
        self.assertIsNone(m['domains']['triviaqa']['pool']['BA'])
    def test_source_binding_and_zero_denominators(self):
        g=dict(qid='a',question_hash='x',trajectory_ids=['1','2','3','4','5'],labels=[0,0,0,0,0])
        self.assertEqual(join_domains([g],[dict(qid='a',question_hash='x',data_source='nq')]),{'a':'nq'})
        with self.assertRaises(ValueError):join_domains([g],[dict(qid='a',question_hash='y',data_source='nq')])
        rs,err=own_predictions([dict(qid='a',trajectory_ids=g['trajectory_ids'],strict_em=g['labels'],p_j=[0]*5,predicted_mixed=0)],{'a':g})
        self.assertEqual(err,0);self.assertIsNone(metrics(rs)['BA']);self.assertEqual(metrics(rs)['TNR'],1)
        self.assertIsNone(metrics(rs)['TPR']);self.assertIsNone(metrics(rs)['allcorrect_FPR'])

if __name__=='__main__':unittest.main()
