"""CPU execution of the real actor methods; no CUDA/framework stubs on the host.

AST extraction avoids importing vLLM/FlashAttention for a small numerical
fixture. The forward is a differentiable toy; the loss, logical weighting,
clipping and Adam are the production implementations.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from search_g1.actor_performance import lean_actor, validate_lean_binding
from search_g1.gradient_clipping import clip_global_l2
from search_g1.numerics import require_finite

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, namespace, *, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name) if class_name else tree.body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class ActorPerformanceTests(unittest.TestCase):
    def run_outer(self, profile):
        functional = {'torch': torch}
        masked_mean = function('verl/utils/torch_functional.py', 'masked_mean', functional)
        core = {'torch': torch, 'verl_F': SimpleNamespace(masked_mean=masked_mean)}
        policy_loss = function('verl/trainer/ppo/core_algos.py', 'compute_policy_loss', core)
        finite_calls = []; snapshots = []; monitors = []; forwards = []; write_rows = []

        def finite(*args, **kwargs):
            finite_calls.append(kwargs['stage'])
            return require_finite(*args, **kwargs)

        def append(metrics, values):
            for key, value in values.items():
                metrics.setdefault(key, []).append(value)

        namespace = dict(torch=torch, os=os, json=json, FSDP=FSDP,
            require_finite=finite, core_algos=SimpleNamespace(compute_policy_loss=policy_loss), append_to_dict=append)
        actor = SimpleNamespace()
        for name in ('_optimizer_step', '_run_minibatch'):
            actor_method = function('verl/workers/actor/dp_actor.py', name, namespace, class_name='DataParallelPPOActor')
            setattr(actor, name, MethodType(actor_method, actor))
        module = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[.1, -.2]]))
        optimizer = torch.optim.Adam(module.parameters(), lr=.001)
        actor.actor_module = module
        actor.actor_optimizer = optimizer
        actor.config = Config(performance_profile=profile, grad_clip=1.,
            use_dynamic_bsz=False, ppo_micro_batch_size=8, ppo_logical_micro_batch_size=4,
            state_masking=True, entropy_coeff=0., use_kl_loss=False, clip_ratio=.2)
        actor.gradient_accumulation = 4
        actor._recoverable_guard_mode = True
        actor._optimizer_writes = actor._optimizer_calls = 0
        actor._invalidate_monitor_forward_cache = lambda **kwargs: None
        actor._monitor_old_policy = lambda *args: monitors.append(args[-1])

        def forward(*, micro_batch, **kwargs):
            forwards.append(True)
            return None, module(micro_batch['features']).squeeze(-1)
        actor._forward_micro_batch = forward

        class Micro(dict):
            def cuda(self, **kwargs):
                return self
        generator = torch.Generator().manual_seed(294)
        micros = []
        for i in range(4):
            mask = torch.ones(8, 6)
            mask[::2, -2:] = 0
            micros.append(Micro(responses=torch.ones(8, 6, dtype=torch.long), loss_mask=mask,
                old_log_probs=torch.randn(8, 6, generator=generator) * .01,
                advantages=torch.randn(8, 6, generator=generator),
                features=torch.randn(8, 6, 2, generator=generator)))
        mini = SimpleNamespace(split=lambda size: micros)
        safety = Config(dual_clip=3., extra_policy_monitor_forwards=True)
        # Production binding rejects a lean run with this flag true. Keeping it
        # true here also verifies the hot path itself cannot execute monitors.
        metrics = {}
        with patch('torch.cuda.current_device', return_value=0), \
             patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'common8'}), \
             patch('search_g1.optimizer_transaction.LocalOptimizerTransaction',
                   side_effect=lambda *args: snapshots.append(True)), \
             patch('search_g1.gradient_clipping.require_finite', side_effect=finite), \
             contextlib.redirect_stdout(io.StringIO()):
            for index in range(20):
                actor._gradient_diagnostic_context = dict(logical_update=1, minibatch_index=index)
                actor._pending_optimizer_transaction = None
                actor._run_minibatch(mini, temperature=1., safety_enabled=True, safety=safety,
                    frozen=None, tail_limit=.1, metrics=metrics)
                write_rows.append(dict(actor._pending_write_receipt))
        return dict(parameter=module.weight.detach().clone(),
            state={key: value.clone() if torch.is_tensor(value) else value
                   for key, value in optimizer.state[module.weight].items()},
            grad=module.weight.grad.clone(), finite_calls=finite_calls, snapshots=snapshots,
            monitors=monitors, forwards=forwards, writes=actor._optimizer_writes,
            calls=actor._optimizer_calls, rows=write_rows, metrics=metrics)

    def test_lean_twenty_real_adam_writes_identical_math_without_scans_copies_or_monitor_forwards(self):
        legacy = self.run_outer(None)
        lean = self.run_outer('lean_v1')
        self.assertEqual(lean['finite_calls'], [])
        self.assertEqual(lean['snapshots'], [])
        self.assertEqual(lean['monitors'], [])
        self.assertEqual(len(legacy['snapshots']), 20)
        self.assertTrue(legacy['finite_calls'])
        self.assertEqual(len(legacy['monitors']), 40)
        self.assertEqual(len(lean['forwards']), 80)
        self.assertEqual((lean['writes'], lean['calls']), (20, 20))
        self.assertTrue(torch.equal(lean['parameter'], legacy['parameter']))
        self.assertTrue(torch.equal(lean['grad'], legacy['grad']))
        for key in legacy['state']:
            self.assertTrue(torch.equal(lean['state'][key], legacy['state'][key]))
        for lean_row, legacy_row in zip(lean['rows'], legacy['rows']):
            self.assertIsNone(lean_row['gradient_norm_after'])
            self.assertIs(lean_row['gradient_norm_after_measured'], False)
            self.assertEqual(lean_row['gradient_norm_after_source'], 'not_measured')
            self.assertEqual(lean_row['gradient_clip_threshold'], 1.)
            self.assertEqual(lean_row['gradient_clip_coefficient'],
                min(1., 1./(legacy_row['gradient_norm_before']+1e-6)))
            self.assertEqual({key: value for key, value in lean_row.items()
                              if key in legacy_row and key != 'gradient_norm_after'},
                             {key: value for key, value in legacy_row.items()
                              if key != 'gradient_norm_after'})
        self.assertEqual(lean['metrics'], legacy['metrics'])

    def test_lean_clipping_preserves_fp64_extreme_and_bf16_rounding(self):
        for dtype, magnitude in ((torch.float32, 1e20), (torch.bfloat16, 1e38),
                                 (torch.float32, .01)):
            legacy = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
            lean = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
            legacy.weight.grad = torch.full_like(legacy.weight, magnitude)
            lean.weight.grad = legacy.weight.grad.clone()
            a = clip_global_l2(legacy, 1., chunk_size=1, diagnostic=False)
            with patch('search_g1.gradient_clipping.require_finite', side_effect=AssertionError('numerical scan')):
                b = clip_global_l2(lean, 1., chunk_size=1, diagnostic=False, check_finite=False)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(legacy.weight.grad, lean.weight.grad))
            self.assertTrue(torch.equal(legacy._last_clipped_global_l2, lean._last_clipped_global_l2))

    def test_only_explicit_native4_easy_lean_binding_is_accepted(self):
        self.assertFalse(lean_actor({}))
        self.assertFalse(lean_actor({'performance_profile': 'legacy'}))
        with self.assertRaises(ValueError):
            lean_actor({'performance_profile': 'lean_typo'})
        config = dict(performance_profile='lean_v1', policy_safety={'extra_policy_monitor_forwards': False})
        run = dict(performance_profile='lean_v1', easy2560_selection=True, profile='common8',
            execution_site='hpc3', training_world_size=4, data_selection_mode='mixed_first',
            actor_logical_micro_batch_per_rank=4, extra_policy_monitor_forwards=False)
        self.assertTrue(validate_lean_binding(config, run))
        for key, value in [('training_world_size', 8), ('extra_policy_monitor_forwards', True),
                           ('performance_profile', None), ('question_probe', {'enabled': True})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_lean_binding(config, dict(run, **{key: value}))
        with self.assertRaises(ValueError):
            validate_lean_binding({}, run)

    def test_dualclip_default_guard_stays_enabled_and_lean_only_skips_scan(self):
        from search_g1.policy_safety import dual_clip_terms
        delta = torch.tensor([[.1, -.2, .4]], requires_grad=True)
        advantages = torch.tensor([[1., -1., .5]])
        mask = torch.ones_like(delta)
        legacy = dual_clip_terms(delta, advantages, mask, .2, 3.)
        with patch('torch.isfinite', side_effect=AssertionError('finite scan')):
            lean = dual_clip_terms(delta, advantages, mask, .2, 3., check_finite=False)
        self.assertTrue(torch.equal(legacy, lean))
        self.assertTrue(torch.equal(torch.autograd.grad(legacy.sum(), delta)[0],
                                    torch.autograd.grad(lean.sum(), delta)[0]))
        with self.assertRaises(ValueError):
            dual_clip_terms(torch.full_like(delta, float('nan')), advantages, mask, .2, 3.)

    def test_actual_actor_result_reports_rollback_capability_honestly(self):
        namespace = dict(torch=torch, DataProto=object, FSDP=FSDP)
        method = function('verl/workers/actor/dp_actor.py', 'update_policy', namespace,
                          class_name='DataParallelPPOActor')
        for profile in (None, 'lean_v1'):
            actor = SimpleNamespace(actor_module=object(),
                config=Config(performance_profile=profile,
                    policy_safety={'extra_policy_monitor_forwards': profile != 'lean_v1'}))
            def update(data):
                actor._optimizer_writes = actor._optimizer_calls = 20
                actor._last_opportunities = [dict(status='written', guard_stage=None, guard_reasons=[])
                                             for _ in range(20)]
                return {'actual_update': True}
            actor._update_policy_attempt = update
            data = SimpleNamespace(meta_info=dict(attempt_id='a' * 32, attempt_index=1, logical_update=1))
            with patch('search_g1.policy_safety.bind_actual_config'), \
                 patch('search_g1.policy_safety.recoverable_mode', return_value=True), \
                 patch('search_g1.policy_safety._phase_agreement'), \
                 patch('search_g1.policy_safety.agree_attempt_result'):
                self.assertEqual(method(actor, data), {'actual_update': True})
            result = actor._last_attempt_result
            self.assertIs(result['current_step_rollback'], profile != 'lean_v1')
            self.assertIs(result['no_automatic_rollback'], profile == 'lean_v1')
            self.assertEqual(result['optimizer_writes_this_attempt'], 20)

    def test_actual_scheduler_aborted_receipt_reports_rollback_capability_honestly(self):
        namespace = dict(WRITES_PER_OUTER=20, _learning_rates=lambda optimizer: [1e-6],
                         _need=lambda value, message: None if value else self.fail(message))
        method = function('reproduction/common8/scheduler.py', 'abort_attempt', namespace,
                          class_name='OuterSchedulerEvidence')
        for lean in (False, True):
            rows = []
            scheduler = SimpleNamespace(recoverable=True, pending={'writes_before': 40},
                prepared=None, rejected=False, optimizer=object(), lean_performance_profile=lean,
                _check_state=lambda: None, _append=rows.append)
            result = method(scheduler, optimizer_writes=40, opportunities_processed=1,
                            guard_stage='test_only', guard_reasons=['test_only'])
            self.assertIs(result['current_step_rollback'], not lean)
            self.assertIs(result['no_automatic_rollback'], lean)
            self.assertEqual(result['scheduler_advances'], 0)
            self.assertEqual(rows, [result])

    def test_actual_transaction_auditor_accepts_bound_lean_zero_monitors_and_rejects_false_rollback(self):
        import copy
        from reproduction.common8.formal_attempt_audit import _audit_transaction_rows, consumed_lr
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        run = dict(performance_profile='lean_v1', easy2560_selection=True, profile='common8',
            execution_site='hpc3', training_world_size=4, data_selection_mode='mixed_first',
            actor_logical_micro_batch_per_rank=4, extra_policy_monitor_forwards=False)
        binding = dict(run_id='a' * 32, calibration_sha256='b' * 64)
        attempt = dict(logical_update=1, attempt_id='c' * 32, attempt_index=1,
            behavior_probability_gate_sha256='d' * 64, opportunities_processed=20,
            optimizer_writes_before=0, optimizer_writes_after=20,
            opportunity_statuses=['written'] * 20, complete_outer=True,
            actual_monitor_forward_count=0, reused_monitor_forward_count=0)
        rows_by_rank = {}
        for rank in range(4):
            rows = []
            header = dict(rank=rank, run_id=binding['run_id'])
            for mini in range(20):
                context = dict(logical_update=1, minibatch_index=mini, prior_optimizer_writes=mini,
                    attempt_id=attempt['attempt_id'], attempt_index=1,
                    behavior_probability_gate_sha256=attempt['behavior_probability_gate_sha256'])
                rows.append(dict(header, kind='OPTIMIZER_WRITE', context=context, write_count=mini + 1,
                    learning_rates=[consumed_lr(1)], gradient_norm_before=2., gradient_norm_after=.5))
                rows.append(dict(header, kind='OPPORTUNITY', opportunity_index=mini, context=context,
                    processed=True, status='written', optimizer_write_succeeded=True,
                    optimizer_write_rolled_back=False, optimizer_call_attempted=True,
                    optimizer_writes_before=mini, optimizer_writes_after=mini + 1,
                    guard_stage=None, guard_reasons=[]))
            rows.append(dict(header, kind='OUTER_BUDGET', opportunities_processed=20,
                optimizer_writes=20, cumulative_writes=20, extra_policy_monitor_forwards=False,
                actual_monitor_forward_count=0, reused_monitor_forward_count=0))
            rows_by_rank[rank] = rows
        result = _audit_transaction_rows(rows_by_rank, binding, attempt, 0, run=run)
        self.assertEqual(result['global_monitor_count'], 0)
        self.assertEqual(result['actual_monitor_forward_count'], 0)
        forged = copy.deepcopy(rows_by_rank)
        forged[0][1].update(status='rolled_back_guard', optimizer_write_rolled_back=True)
        with self.assertRaisesRegex(EvidenceError, 'no per-write rollback'):
            _audit_transaction_rows(forged, binding, attempt, 0, run=run)
        with self.assertRaises(EvidenceError):
            _audit_transaction_rows(rows_by_rank, binding, attempt, 0,
                                    run=dict(run, performance_profile=None))


if __name__ == '__main__':
    unittest.main()
