import unittest
import numpy as np
import torch
from reproduction.hpc3_qprobe.analyze_sigma3_fit_readonly import metrics
from reproduction.hpc3_qprobe.reconstruct_sigma3_head_metrics import exact_tree, NotExact


class MetricDefinitions(unittest.TestCase):
    def test_perfect(self):
        m = metrics(np.eye(6), np.arange(6))
        for k in ('count_accuracy','binary_probability_accuracy','binary_argmax6_collapsed_accuracy'):
            self.assertEqual(m[k],1)
        for k in ('sigma_mse','variance_mse','sigma_mae','variance_mae'):
            self.assertEqual(m[k],0)
        self.assertEqual(m['sigma_r2'],1); self.assertEqual(m['variance_r2'],1)

    def test_probability_binary_and_collapsed_argmax_differ(self):
        p = np.array([[.4,.15,.15,.15,.15,0]])
        m = metrics(p,[0])
        self.assertEqual(m['count_accuracy'],1)
        self.assertEqual(m['binary_argmax6_collapsed_accuracy'],1)
        self.assertEqual(m['binary_probability_accuracy'],0)

    def test_expected_variance_not_square_of_expected_sigma(self):
        m = metrics([[.5,0,.5,0,0,0]],[0])
        self.assertAlmostEqual(m['variance_predicted_mean'],.15)
        self.assertAlmostEqual(m['sigma_predicted_mean']**2,.075)
        self.assertAlmostEqual(m['variance_mse'],.0225)

    def test_age_weights(self):
        p = np.eye(6)[[0,0]]
        m = metrics(p,[0,5],[1,3])
        self.assertEqual(m['count_accuracy'],.25)
        self.assertEqual(m['binary_probability_accuracy'],1)

    def test_exact_tree_checks_optimizer_rng_dtype_and_all_labels(self):
        state = dict(model=torch.tensor([1.]), optimizer={'exp_avg':torch.tensor([2.])},
            rng=torch.tensor([3],dtype=torch.uint8), labels=[{'K':2}])
        exact_tree(state,state)
        for key in ('model','optimizer','rng','labels'):
            import copy
            different=copy.deepcopy(state)
            if key=='optimizer': different[key]['exp_avg'][0]+=1e-5
            elif key=='labels':different[key][0]['K']=3
            elif key=='rng': different[key]=different[key].long()
            else:different[key][0]+=1e-6
            with self.assertRaises(NotExact):exact_tree(different,state)


if __name__ == '__main__': unittest.main()
