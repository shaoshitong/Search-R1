"""Remote CPU tests for direct binary selection, private exploration and ties."""
import copy
import itertools
import unittest
import numpy as np
from search_g1.qprobe_selection import PredictQuestion,QuestionPool,binary_predictions,prequential_metrics,select_questions,validate_budget


def questions(n):return [PredictQuestion(i,str(i),'question '+str(i)) for i in range(n)]
def prediction(p,v=None):
    return dict(mixed_probability=np.asarray(p,dtype=np.float64),expected_variance=np.zeros(len(p)) if v is None else np.asarray(v,dtype=np.float64))


class QuestionProbeSelectionTests(unittest.TestCase):
    def test_zero_exploration_uses_direct_mixed_score(self):
        q=questions(600);p=prediction([0]*87+[1]*513)
        a=select_questions(q,p,exploration_questions=0,rng=np.random.default_rng(1),tie_seed='top512')
        b=select_questions(q,p,exploration_questions=0,rng=np.random.default_rng(999),tie_seed='top512')
        self.assertEqual(a,b);self.assertEqual(len(a),512)
        self.assertTrue(all(r.ordinal>=87 and r.selection_source=='predicted_mixed' for r in a))
    def test_aux_variance_never_overrides_direct_mixed_probability(self):
        p=prediction([.1,.9],[.3,0])
        result=select_questions(questions(2),p,selected_questions=1,exploration_questions=0,rng=np.random.default_rng(42))
        self.assertEqual([r.ordinal for r in result],[1])
    def test_all_strict_five_outcomes_binary_target_and_unbiased_variance(self):
        for values in itertools.product((0,1),repeat=5):
            k=sum(values);v=k*(5-k)/20;y=int(1<=k<=4)
            self.assertAlmostEqual(v,np.var(values,ddof=1))
            m=prequential_metrics(prediction([y],[v]),[values],['predicted_mixed'])
            self.assertEqual(m['all']['mixed_brier'],0)
            self.assertEqual(m['all']['variance_normalized_mse'],0)
    def test_uniform_first_nested_exploration_budgets_and_order(self):
        q=questions(5120);p=prediction([0]*4096+[1]*1024);sets=[]
        for n in (0,64,128,256):
            result=select_questions(q,p,exploration_questions=n,rng=np.random.default_rng(20261013),tie_seed=5)
            random={r.ordinal for r in result if r.selection_source=='random_exploration'}
            focus=[r for r in result if r.selection_source=='predicted_mixed']
            self.assertEqual(len(result),512);self.assertEqual(len(random),n);self.assertEqual(len(focus),512-n)
            self.assertTrue(all(r.ordinal>=4096 for r in focus))
            self.assertEqual([r.ordinal for r in result],sorted(r.ordinal for r in result));sets.append(random)
        self.assertTrue(sets[0]<=sets[1]<=sets[2]<=sets[3])
    def test_ties_hash_identity_not_presented_order(self):
        q=questions(64);p=prediction([.5]*64)
        a=select_questions(q,p,selected_questions=16,exploration_questions=0,rng=np.random.default_rng(55),tie_seed='same')
        b=select_questions(q[::-1],p,selected_questions=16,exploration_questions=0,rng=np.random.default_rng(55),tie_seed='same')
        self.assertEqual(a,b);self.assertNotEqual([r.ordinal for r in a],list(range(16)))
    def test_duplicate_content_reservation_and_selection_rejected(self):
        pool=QuestionPool(3);self.assertEqual(pool.reserve('a','same   question'),0)
        self.assertIsNone(pool.reserve('b','same question'));self.assertIsNone(pool.reserve('a','other'))
        with self.assertRaises(ValueError):select_questions([PredictQuestion(0,'a','same'),PredictQuestion(1,'b','same')],prediction([.5,.5]),selected_questions=1,exploration_questions=0,rng=np.random.default_rng(42))
    def test_invalid_count_outputs_and_budgets_rejected(self):
        for args in ((5120,512,-1),(5120,512,513),(128,512,0),(5120,True,0),(5120,512,True)):
            with self.assertRaises(ValueError):validate_budget(*args)
        with self.assertRaises(ValueError):binary_predictions({'probabilities':np.ones((1,6))/6})
    def test_absent_random_measurement_not_fabricated_zero(self):
        m=prequential_metrics(prediction([1.],[.3]),[[1,1,0,0,0]],['predicted_mixed'])
        self.assertEqual(m['random_exploration'],{'questions':0})
        self.assertEqual(m['predicted_mixed']['variance_mse'],0)
        with self.assertRaises(ValueError):prequential_metrics(prediction([1.],[.3]),[[1,1,0,0]],['predicted_mixed'])

if __name__=='__main__':unittest.main()
