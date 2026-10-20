"""Synthetic CPU qualification only; no train/diagnostic/2048 data are read.

The corrected three-event loss fixes individual homogeneous-label gradient
directions. It does not make the product formula identify a joint event from
marginal correctness probabilities when a shared latent outcome remains.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from reproduction.operations.qwen7b_warmup.event_candidate_loss import event_candidate_loss


def counterexample():
    # One observed feature configuration: probability 1/2 allwrong, 1/2
    # allcorrect; there are no Mixed outcomes. Sharing z across these two
    # possible labels enforces that unobserved outcomes cannot be leaked in.
    y = torch.tensor([[0.] * 5, [1.] * 5], dtype=torch.float64)
    pairs = (y[:, :, None] == 1) & (y[:, None, :] == 0)

    def objective(z):
        return event_candidate_loss(z[None].expand(2, -1), y, pairs,
                                    mixed_fraction=.5)['loss']

    z = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    result = event_candidate_loss(z[None].expand(2, -1), y, pairs,
                                  mixed_fraction=.5)
    gradient = torch.autograd.grad(result['loss'], z)[0]
    hessian = torch.autograd.functional.hessian(objective, z)
    return dict(
        scenario='Identical observed inputs; allwrong/allcorrect equiprobable',
        true_event_probabilities=dict(allwrong=.5, mixed=0., allcorrect=.5),
        optimal_trajectory_marginals=z.sigmoid().detach().tolist(),
        implied_event_probabilities=dict(allwrong=1/32, mixed=30/32,
                                         allcorrect=1/32),
        predicted_q=result['mixed_probability'].detach().tolist(),
        gradient=gradient.tolist(),
        hessian_eigenvalues=torch.linalg.eigvalsh(hessian).tolist(),
        expected_objective=float(result['loss'].detach()),
        analytic_expected_objective='(1+1/5)*sum_j BCE(p_j, 0.5)',
        analytic_unique_minimum='p_j=0.5 for every j; Hessian diagonal 0.3',
        conclusion='Negative-gradient repair alone cannot identify residual '
                   'within-question outcome dependence from five marginals',
        evidence_limit='Synthetic existence proof only; does not establish '
                       'the magnitude or cause of dependence in actual data',
        live_training_modified=False, actual_dataset_read=False)


class SharedInputLimitTest(unittest.TestCase):
    def test_candidate_can_predict_mixed_for_zero_mixed_population(self):
        r = counterexample()
        self.assertEqual(r['true_event_probabilities']['mixed'], 0.)
        for q in r['predicted_q']:
            self.assertAlmostEqual(q, .9375, places=12)
        self.assertTrue(all(abs(g) < 1e-12 for g in r['gradient']))
        for e in r['hessian_eigenvalues']:
            self.assertAlmostEqual(e, .3, places=12)
        self.assertAlmostEqual(r['expected_objective'], 6*math.log(2), places=12)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SharedInputLimitTest)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    result = counterexample()
    result['test_status'] = 'PASS_SYNTHETIC_CPU_ONLY'
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
