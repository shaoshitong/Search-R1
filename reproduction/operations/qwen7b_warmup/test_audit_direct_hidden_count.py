import copy,unittest
from .audit_direct_hidden_count import accept_rows,compare
from . import probe_ablation_train as u


class AuditTests(unittest.TestCase):
    def fixture(self):
        rows=[];frozen=[]
        for i,k in enumerate([0,1,2,3,4,5]):
            own=[int(j<k) for j in range(5)];ids=['q%d:t%d'%(i,j) for j in range(5)]
            counts=[.4,.1,.1,.1,.1,.2] if i%2 else [.1,.2,.2,.2,.2,.1]
            rows.append(dict(qid='q%d'%i,trajectory_ids=ids,strict_em=own,p_j=[.9]*5,count_probabilities=counts,
                predicted_mixed=sum(counts[1:5]),score_kind='hidden_direct_count6'))
            frozen.append(dict(qid='q%d'%i,trajectory_ids=ids,labels=own))
        return rows,frozen
    def test_independent_metrics_match_and_scores_are_not_old_product(self):
        rows,frozen=self.fixture();actual=accept_rows(rows,frozen)
        compare(actual,dict(pool=u.prediction_metrics(rows),selected512=u.prediction_metrics(sorted(rows,key=lambda x:(-x['predicted_mixed'],x['qid'])))))
        self.assertEqual(actual['pool']['k_histogram'],[1]*6)
    def test_own_truth_and_count_q_corruption_rejected(self):
        rows,frozen=self.fixture();bad=copy.deepcopy(rows);bad[0]['strict_em'][0]=1
        with self.assertRaises(ValueError):accept_rows(bad,frozen)
        bad=copy.deepcopy(rows);bad[0]['predicted_mixed']=.01
        with self.assertRaises(ValueError):accept_rows(bad,frozen)

if __name__=='__main__':unittest.main()
