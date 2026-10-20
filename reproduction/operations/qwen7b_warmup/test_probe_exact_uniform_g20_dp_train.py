"""Small explicit callback/real-G20 plan adapter chain tests; no GPU claims."""
import ast
import copy
import inspect
import unittest
import torch
from . import probe_g20_train as native
from . import probe_ablation_train as u
from . import probe_exact_uniform_g20_dp_train as entry
from .prepare_exact_uniform_g20_dp import REQUIRED as CPU_REQUIRED
from .g20_grouping import digest, validate
from .probe_exact_uniform_g5_dp import bounded_exact_dp_update
from .test_probe_g10_train import setup, ToyReader


def fixture():
    groups = [dict(qid='q%d' % q, trajectory_ids=['q%d:t%d' % (q, t) for t in range(20)],
        labels=[int(t < 10) for t in range(20)]) for q in range(6528)]
    qids = [g['qid'] for g in groups]
    plan = dict(cohort_qids=qids, batches=[qids[i:i + 48] for i in range(0, 6528, 48)],
        batch_epochs=[0] * 136, source_group_size=20, source_groups_sha256=digest(validate(groups)))
    reference = dict(source_checkpoint=dict(sha256=u.SOURCE_SHA),
        global_questions=48, update_budget=136, fit_questions=6528, fit_real_trajectories=130560,
        grouping=dict(mode='random', seed=1729), view_weight_denominator=4,
        recipe=dict(loss_kwargs=dict(trajectory_weight=1., three_event_weight=1.)),
        plan=dict(path='/synthetic/original-plan', sha256=entry.PLAN_SHA),
        evaluations=[dict(role='validation', name='fixedG5')], micro_trajectories=5,
        smaller_question_budget_acknowledged=True,
        g20_complete=dict(sha256=entry.TRAIN_SHA),
        validation_g10_complete=dict(sha256=entry.VALIDATION_SHA))
    hashes = {'reproduction/operations/qwen7b_warmup/' + name: 'b' * 64
        for name in set(CPU_REQUIRED) | set(entry.REQUIRED)}
    config = entry.make_config(reference, plan, hashes=hashes, commit='c' * 40,
        cpu_preparation=dict(path='/synthetic/CPU-preparation', sha256='d' * 64))
    return groups, plan, reference, config


class ExactG20EntryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_defaults_all_three_update_calls_and_honest_exact_plan_adapter(self):
        signature = inspect.signature(native.run)
        self.assertIs(signature.parameters['update_callback'].default, native.run_update)
        self.assertIs(signature.parameters['plan_validator'].default, native.validate_plan)
        tree = ast.parse(inspect.getsource(native.run))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertEqual(sum(n.func.id == 'update_callback' for n in calls), 3)
        self.assertEqual(sum(n.func.id == 'run_update' for n in calls), 0)
        groups, plan, reference, config = fixture()
        self.assertEqual(len(entry.plan_validator(plan, groups, config)), 6528)
        self.assertEqual(config['plan'], reference['plan'])
        self.assertEqual(config['source_checkpoint'], reference['source_checkpoint'])
        self.assertEqual(config['reference_grouping'], reference['grouping'])
        self.assertEqual(config['actual_grouping']['mode'], 'exact_uniform_G20_all15504')
        self.assertNotEqual(config['grouping']['mode'], 'random')
        self.assertEqual(config['global_virtual_G5_views'], 744192)
        self.assertEqual(config['maximum_working_G5_chunk_rows'], 12288)
        self.assertNotIn('max_virtual_views', config)
        bad = copy.deepcopy(config); bad['recipe']['loss_kwargs']['within_rank_weight'] = 1.
        with self.assertRaises(ValueError):
            entry.plan_validator(plan, groups, bad)

    def test_callback_actual_G20_loss_and_Adam_match_explicit_bounded_update(self):
        groups, plan, reference, config = fixture()
        lookup = {g['qid']: g for g in groups}
        batch = plan['batches'][0]
        model, ema, opt, _, _, source = setup(dropout=True)
        device = torch.device('cpu')
        got = entry.update_callback(model, ema, opt, ToyReader(), lookup, batch, config, device, 0)
        wanted = u.state_hashes(model, ema, opt, device)
        u.restore_blob(source, model, ema, opt, device)
        bounded_exact_dp_update(model, ema, opt, ToyReader(), [lookup[q] for q in batch], device,
            loss_kwargs=dict(trajectory_weight=1., three_event_weight=1.), combination_chunk=256,
            micro_trajectories=5)
        self.assertEqual(u.state_hashes(model, ema, opt, device), wanted)
        self.assertEqual(got['virtual_views'], 744192)
        self.assertEqual(got['unique_real_trajectories'], 960)
        self.assertEqual(got['maximum_resident_pattern_rows'], 0)
        self.assertEqual(got['retained_loss_gradient_elements'], 960)
        self.assertEqual(u.steps(opt), 10465)


if __name__ == '__main__':
    unittest.main()
