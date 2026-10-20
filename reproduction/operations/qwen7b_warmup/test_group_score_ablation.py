"""Small synthetic checks; no real prediction/label data are used."""
import unittest

import numpy as np

from .group_score_ablation import auc_ap, choose_threshold, feature_matrix, fit_logistic, sigmoid


class GroupScoreAblationTest(unittest.TestCase):
    def test_features_are_symmetric_and_label_free(self):
        p = np.array([[.1, .2, .3, .4, .5], [.7, .8, .9, .4, .6]])
        x, names = feature_matrix(p)
        swapped, other_names = feature_matrix(p[:, ::-1])
        self.assertEqual(names, other_names)
        np.testing.assert_array_equal(x, swapped)
        self.assertEqual(x.shape, (2, 30))
        self.assertTrue(np.isfinite(x).all())

    def test_auc_and_ap_use_tie_blocks(self):
        labels = np.array([0, 1, 0, 1])
        self.assertEqual(auc_ap(labels, np.array([.1, .8, .2, .9])), dict(roc_auc=1., average_precision=1.))
        self.assertEqual(auc_ap(labels, np.zeros(4)), dict(roc_auc=.5, average_precision=.5))
        self.assertEqual(auc_ap(labels, np.array([.1, .2, .1, .2])), dict(roc_auc=1., average_precision=1.))

    def test_newton_and_threshold_recover_separable_toy(self):
        labels = np.array([0., 0., 1., 1.])
        x = np.array([[0.], [1.], [2.], [3.]])
        coefficient, fit = fit_logistic(x, labels, .01)
        predicted = sigmoid(coefficient[0]+x.dot(coefficient[1:]))
        threshold, balanced = choose_threshold(labels, predicted)
        self.assertEqual(balanced, 1.)
        np.testing.assert_array_equal(predicted >= threshold, labels)
        self.assertLess(fit['maximum_absolute_gradient'], 1e-7)


if __name__ == '__main__':
    unittest.main()
