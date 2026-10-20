"""Necessary finite objective/gradient and toy fullstate tests; no GPU claim."""
import copy
import unittest
import torch
from .exact_uniform_g5_loss import exact_uniform_g5_loss
from .exact_uniform_g5_dp_loss import exact_uniform_g5_dp_loss
from .probe_exact_uniform_g5_dp import bounded_exact_dp_update
from .test_exact_uniform_g5_loss import groups, brute
from .test_probe_g10_train import setup, ToyReader
from . import probe_g10_train as original
from . import probe_ablation_train as u


class ExactDPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9137)

    def test_G10_all252_double_value_and_gradient_old32_and_independent(self):
        gs = groups([0, 1, 5, 9, 10])
        z = (torch.randn(5, 10, dtype=torch.float64) * 2).requires_grad_(True)
        kw = dict(trajectory_weight=.7, three_event_weight=1.3, combination_chunk=19)
        new = exact_uniform_g5_dp_loss(z, gs, **kw)
        old = exact_uniform_g5_loss(z, gs, **kw)
        independent = brute(z, gs, .7, 1.3)
        grad = torch.autograd.grad(new['loss'], z)[0]
        for expected in (old['loss'], independent):
            torch.testing.assert_close(new['loss'], expected, rtol=2e-13, atol=2e-13)
            torch.testing.assert_close(grad, torch.autograd.grad(expected, z)[0], rtol=2e-12, atol=2e-13)
        self.assertEqual(new['confusion_integer_all_views'], old['confusion_integer_all_views'])
        self.assertEqual(new['maximum_resident_pattern_rows'], 0)

    def test_G20_all15504_extreme_endpoints_and_old32_agree(self):
        gs = groups([0, 10, 20], 20)
        z = torch.stack((torch.full((20,), -1000.),
            torch.tensor([-1000., 1000.] * 10), torch.full((20,), 1000.))).double().requires_grad_(True)
        new = exact_uniform_g5_dp_loss(z, gs, combination_chunk=257)
        old = exact_uniform_g5_loss(z, gs, combination_chunk=127)
        grad = torch.autograd.grad(new['loss'], z)[0]
        self.assertTrue(torch.isfinite(new['loss']) and torch.isfinite(grad).all())
        torch.testing.assert_close(new['loss'], old['loss'], rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(grad, torch.autograd.grad(old['loss'], z)[0], rtol=2e-12, atol=2e-13)
        self.assertEqual(new['combinations_per_question'], 15504)
        self.assertEqual(new['denominators']['event_view'], 3 * 15504)
        self.assertEqual(new['retained_loss_gradient_elements'], 60)
        self.assertLessEqual(new['maximum_resident_G5_rows'], 3 * 257)

    def test_G20_permutation_and_chunk_invariance(self):
        gs = groups([3, 13], 20)
        z = torch.randn(2, 20, dtype=torch.float64).requires_grad_(True)
        a = exact_uniform_g5_dp_loss(z, gs, combination_chunk=61)
        ga = torch.autograd.grad(a['loss'], z)[0]
        perm = torch.randperm(20)
        pg = [dict(g, trajectory_ids=[g['trajectory_ids'][i] for i in perm],
            labels=[g['labels'][i] for i in perm]) for g in gs]
        zz = z.detach()[:, perm].requires_grad_(True)
        b = exact_uniform_g5_dp_loss(zz, pg, combination_chunk=1024)
        gb = torch.autograd.grad(b['loss'], zz)[0]
        torch.testing.assert_close(a['loss'], b['loss'], rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(ga[:, perm], gb, rtol=2e-12, atol=2e-13)
        self.assertEqual(a['confusion_integer_all_views'], b['confusion_integer_all_views'])

    def test_source_shape_and_unsupported_components_fail_closed(self):
        gs = groups([2])
        with self.assertRaises(TypeError):
            exact_uniform_g5_dp_loss(torch.zeros(1, 10), gs, binary_mixed_weight=1.)
        for chunk in (0, 4097, True):
            with self.assertRaises(ValueError):
                exact_uniform_g5_dp_loss(torch.zeros(1, 10), gs, combination_chunk=chunk)
        for z in (torch.zeros(1, 20), torch.full((1, 10), float('nan'))):
            with self.assertRaises(ValueError):
                exact_uniform_g5_dp_loss(z, gs)
        bad = groups([2]); bad[0]['trajectory_ids'][1] = bad[0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):
            exact_uniform_g5_dp_loss(torch.zeros(1, 10), bad)

    def test_bounded_DP_toy_Adam_EMA_RNG_replay_and_next_update_recovery_G20(self):
        model, ema, opt, gs, _, source = setup(dropout=True)
        gs = [dict(g, trajectory_ids=['%s:t%d' % (g['qid'], i) for i in range(20)],
            labels=g['labels'] + [1 - y for y in g['labels']]) for g in gs]
        d = torch.device('cpu')
        kw = dict(loss_kwargs=dict(trajectory_weight=1., three_event_weight=1.),
            micro_trajectories=3, combination_chunk=256)
        got = bounded_exact_dp_update(model, ema, opt, ToyReader(), gs, d, **kw)
        expected = u.state_hashes(model, ema, opt, d)
        u.restore_blob(source, model, ema, opt, d)
        bounded_exact_dp_update(model, ema, opt, ToyReader(), gs, d, **kw)
        self.assertEqual(expected, u.state_hashes(model, ema, opt, d))
        saved = copy.deepcopy(original.blob(model, ema, opt, d, 1,
            dict(config_sha256='toy', matched_binding_sha256='toy'), source['source_all_rank_states']))
        bounded_exact_dp_update(model, ema, opt, ToyReader(), gs, d, **kw)
        expected = u.state_hashes(model, ema, opt, d)
        u.restore_blob(saved, model, ema, opt, d)
        bounded_exact_dp_update(model, ema, opt, ToyReader(), gs, d, **kw)
        self.assertEqual(expected, u.state_hashes(model, ema, opt, d))
        self.assertEqual(u.steps(opt), 10466)
        self.assertEqual(saved['cursor'], 1)
        self.assertFalse(saved['optimizer_reset'])
        self.assertEqual(got['actual_forward_trajectory_exposures'], 120)
        self.assertEqual(got['maximum_resident_pattern_rows'], 0)
        self.assertEqual(got['grouping']['combinations_per_question'], 15504)

    def test_typed_G20_preparation_preserves_plan_E1_and_never_authorizes_GPU(self):
        from .prepare_exact_uniform_g20_dp import make_config, REQUIRED
        qids = ['synthetic-%d' % i for i in range(6528)]
        plan = dict(cohort_qids=qids, batches=[qids[i:i + 48] for i in range(0, 6528, 48)], batch_epochs=[0] * 136)
        source = dict(source_checkpoint=dict(sha256=u.SOURCE_SHA), global_questions=48,
            update_budget=136, fit_questions=6528, fit_real_trajectories=130560,
            grouping=dict(mode='random', seed=1729), view_weight_denominator=4,
            recipe=dict(loss_kwargs=dict(trajectory_weight=1., three_event_weight=1.)),
            plan=dict(path='/synthetic/preserved-plan', sha256='a' * 64),
            evaluations=[dict(role='validation', name='fixedG5')])
        hashes = {'reproduction/operations/qwen7b_warmup/' + name: 'b' * 64 for name in REQUIRED}
        result = make_config(source, plan, hashes=hashes, commit='c' * 40)
        self.assertEqual(result['plan'], source['plan'])
        self.assertEqual(result['source_checkpoint'], source['source_checkpoint'])
        self.assertEqual(result['evaluations'], source['evaluations'])
        self.assertEqual(result['combinations_per_question'], 15504)
        self.assertEqual(result['view_weight_denominator'], 15504)
        self.assertFalse(result['explicit_launch_authorized'])
        self.assertIsNone(result['training_entrypoint'])
        self.assertEqual(result['GPU_requested'], 0)


if __name__ == '__main__':
    unittest.main()
