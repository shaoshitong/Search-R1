"""Focused CPU math and identity checks for posthoc mixed diagnostics."""
import math
import unittest
from .mixed_diagnostic import mixed_probability, aggregate, group_predictions


class MixedDiagnosticTests(unittest.TestCase):
    def test_formula_and_endpoints(self):
        q, _ = mixed_probability([math.log(.2/.8)]*5)
        self.assertAlmostEqual(q,.672)
        self.assertEqual(mixed_probability([1000.]*5)[0],0.)
        self.assertEqual(mixed_probability([-1000.]*5)[0],0.)
        self.assertEqual(mixed_probability([1000.,-1000.,1000.,1000.,1000.])[0],1.)

    def test_group_by_identity_not_prediction_order(self):
        index={};pred=[]
        for q in range(512):
            for j in range(5):
                tid=f'{q}-{j}';index[tid]=dict(qid=str(q),question_hash=str(q),strict_em=int(j==0))
                pred.append((tid,0.,int(j==0)))
        rows=group_predictions(pred[::-1],index)
        self.assertEqual(len(rows),512)
        self.assertTrue(all(row['correct_count']==1 for row in rows))
        summary=aggregate(rows)
        self.assertEqual(summary['mixed_count'],512)
        self.assertEqual(summary['selected_q_ge_0_5_count'],512)
        self.assertEqual(summary['majority_accuracy'],1.)
        self.assertIsNone(summary['balanced_accuracy'])
        with self.assertRaises(ValueError):group_predictions(pred[:-1]+[pred[0]],index)
        bad=list(pred);bad[0]=(pred[0][0],0.,0)
        with self.assertRaises(ValueError):group_predictions(bad,index)

    def test_exact_endpoint_scoring(self):
        good = aggregate([dict(mixed_label=0, mixed_probability=0.),
                          dict(mixed_label=1, mixed_probability=1.)])
        self.assertEqual(good['bce'], 0.)
        self.assertFalse(good['bce_is_positive_infinity'])
        bad = aggregate([dict(mixed_label=1, mixed_probability=0.),
                         dict(mixed_label=0, mixed_probability=1.)])
        self.assertIsNone(bad['bce'])
        self.assertTrue(bad['bce_is_positive_infinity'])
        self.assertEqual(bad['wrong_exact_endpoint_count'], 2)

    def test_auc_ties_and_empty_selection(self):
        rows=[dict(mixed_label=y,mixed_probability=.2) for y in [0,1]]
        summary=aggregate(rows)
        self.assertEqual(summary['auroc'],.5)
        self.assertEqual(summary['selected_q_ge_0_5_count'],0)
        self.assertIsNone(summary['actual_mixed_among_selected'])


if __name__=='__main__':unittest.main()
