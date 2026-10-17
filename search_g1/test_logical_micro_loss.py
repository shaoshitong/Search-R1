"""CPU objective/gradient tests; physical GPU memory fit is tested by real runs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import torch

from search_g1.logical_micro_loss import (
    logical_masked_mean, logical_policy_loss, logical_token_weights,
)
from search_g1.numerics import require_finite
from reproduction.common8.topology import formal_monitor_geometry, validate_base
from reproduction.common8.batch_contract import check_global, check_local


ROOT = Path(__file__).resolve().parents[1]


def actual_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(ROOT / path), 'exec'), namespace)
    return namespace[name]


MASKED_MEAN = actual_function('verl/utils/torch_functional.py', 'masked_mean', {'torch': torch})
PG = actual_function('verl/trainer/ppo/core_algos.py', 'compute_policy_loss',
                     {'torch': torch, 'verl_F': NS(masked_mean=MASKED_MEAN)})
KL = actual_function('verl/trainer/ppo/core_algos.py', 'kl_penalty', {'torch': torch})


class LogicalMicroLossTests(unittest.TestCase):
    def inputs(self):
        generator = torch.Generator().manual_seed(413)
        shape = (32, 17)
        # Deliberately unequal token counts per logical micro. Global token
        # averaging must fail this fixture rather than accidentally matching.
        lengths = torch.tensor([1]*4 + [17]*4 + [3]*4 + [15]*4 +
                               [5]*4 + [13]*4 + [7]*4 + [11]*4)
        mask = (torch.arange(17).unsqueeze(0) < lengths.unsqueeze(1)).double()
        old = torch.randn(shape, generator=generator, dtype=torch.float64) / 3
        delta = torch.randn(shape, generator=generator, dtype=torch.float64) * .8
        advantages = torch.randn(shape, generator=generator, dtype=torch.float64)
        return old, delta, advantages, mask

    def test_policy_value_and_each_logprob_gradient_independent_of_physical_partition(self):
        old, delta, advantages, mask = self.inputs()
        for dual_clip in (None, 3.):
            expected = None
            for physical in (4, 8, 16, 32):
                new = (old + delta).clone().requires_grad_(True)
                losses = [logical_policy_loss(PG, logical_micro_batch_size=4,
                    old_log_prob=o, log_prob=n, advantages=a, eos_mask=m,
                    cliprange=.2, dual_clip=dual_clip)
                    for o, n, a, m in zip(old.split(physical), new.split(physical),
                                         advantages.split(physical), mask.split(physical))]
                result = tuple(torch.stack(values).mean() for values in zip(*losses))
                result[0].backward()
                if expected is None:
                    expected = tuple(v.detach() for v in result), new.grad.clone()
                else:
                    for actual, target in zip(result, expected[0]):
                        torch.testing.assert_close(actual, target, atol=1e-14, rtol=1e-14)
                    torch.testing.assert_close(new.grad, expected[1], atol=1e-14, rtol=1e-14)
            naive = PG(old, old+delta, advantages, mask, .2, dual_clip=dual_clip)[0]
            self.assertGreater(abs(float(naive - expected[0][0])), .01)

    def test_monitor_mass_is_two_logical_groups_per_rank_even_with_physical32(self):
        _, _, _, mask = self.inputs()
        mask = mask[:8]
        weights = logical_token_weights(mask, 4)
        self.assertEqual(float(weights.sum()), 2.)
        self.assertEqual(float(weights[:4].sum()), 1.)
        self.assertEqual(float(weights[4:].sum()), 1.)
        split = torch.cat([logical_token_weights(part, 4) for part in mask.split(4)])
        torch.testing.assert_close(weights, split, atol=0, rtol=0)
        for physical in (4, 8, 16, 32):
            run = dict(profile='common8', training_world_size=4,
                actor_micro_batch_per_rank=physical, actor_logical_micro_batch_per_rank=4)
            proof = formal_monitor_geometry(run)
            self.assertEqual(proof['actor_micro_batch_per_rank'], physical)
            self.assertEqual(proof['expected_complete_effective_weight_mass'], 8.)

    def test_actual_monitor_forward32_on_eight_rows_keeps_two_logical_masses(self):
        old, delta, advantages, mask = [part[:8] for part in self.inputs()]
        method = actual_function('verl/workers/actor/dp_actor.py', '_monitor_old_policy',
            dict(torch=torch, copy=copy, json=json, FSDP=type('NoFSDP', (), {}),
                 require_finite=require_finite))
        class Batch(dict):
            def split(self, count):
                return [Batch({key: value[i:i+count] for key, value in self.items()})
                        for i in range(0, self['old_log_probs'].shape[0], count)]
            def cuda(self, **kwargs): return self
        class Precision:
            def validate_worker(self, **kwargs): pass
        class Safety(dict):
            __getattr__ = dict.__getitem__
        frozen = Batch(old_log_probs=old, advantages=advantages, loss_mask=mask)
        identity = dict(tensor_sha256={name: hashlib.sha256(value.detach().cpu().contiguous()
            .view(torch.uint8).numpy().tobytes()).hexdigest() for name, value in frozen.items()})
        forwards = []; reports = []
        def forward(batch, **kwargs):
            forwards.append(batch['old_log_probs'].shape[0])
            return None, batch['old_log_probs'] + .01 * delta[:batch['old_log_probs'].shape[0]]
        actor = NS(actor_module=torch.nn.Linear(1, 1), compute_precision=Precision(),
            use_remove_padding=False,
            config=NS(ppo_micro_batch_size=32, ppo_logical_micro_batch_size=4,
                      state_masking=True, use_remove_padding=False),
            _gradient_diagnostic_context=dict(minibatch_index=0, prior_optimizer_writes=0),
            _monitor_identity=identity, _monitor_temperature=1., _monitor_forward_cache=None,
            _safety_calibration_sha='test-only', _policy_execution_role='production',
            _actual_monitor_forward_count=0, _reused_monitor_forward_count=0,
            _forward_micro_batch=forward, _monitor_cache_key=lambda sha: sha,
            _invalidate_monitor_forward_cache=lambda: None)
        safety = Safety(dual_clip=3., kl_log_bound=30., target_old_policy_kl=100.)
        with patch('reproduction.common8.worker_evidence.append',
                   side_effect=lambda kind, report, **kwargs: reports.append(report)), \
                patch('torch.cuda.current_device', return_value=0):
            report = method(actor, frozen, safety, 1., 'before_minibatch')
        self.assertEqual(forwards, [8])
        self.assertEqual(float(report['effective_weight_mass']), 2.)
        self.assertEqual(len(reports), 1)
        self.assertTrue(report['allowed'])
        self.assertEqual(report['objective_weight_definition'],
            'equal four-row logical micro masked_mean; weighted global numerator/denominator')

    def test_masked_entropy_and_kl_keep_logical_means_and_ignore_padding_nan(self):
        _, delta, _, mask = self.inputs()
        values = torch.where(mask.bool(), delta.square(), torch.full_like(delta, float('nan')))
        expected = torch.stack([MASKED_MEAN(v, m) for v, m in zip(values.split(4), mask.split(4))]).mean()
        torch.testing.assert_close(logical_masked_mean(values, mask, 4), expected)
        self.assertFalse(torch.isclose(MASKED_MEAN(values, mask), expected))

    def test_native4_global_then_local_contract_accepts128_physical_and_local4_logical(self):
        run = dict(profile='common8', training_world_size=4,
            actor_micro_batch_per_rank=32, actor_logical_micro_batch_per_rank=4)
        config = NS(actor=NS(ppo_mini_batch_size=128, ppo_micro_batch_size=128,
            ppo_logical_micro_batch_size=4, ppo_epochs=1,
            optim=NS(lr=1e-6, lr_warmup_steps_ratio=0., lr_warmup_steps=57, warmup_style='constant')),
            rollout=NS(n=1, n_agent=5, log_prob_micro_batch_size=32),
            ref=NS(log_prob_micro_batch_size=32))
        check_global(config, 4, 1, run)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base/'logs').mkdir()
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            with patch.dict(os.environ, SEARCH_G1_BASE=folder):
                config.actor.ppo_mini_batch_size //= 4
                config.actor.ppo_micro_batch_size //= 4
                config.rollout.log_prob_micro_batch_size //= 4
                config.ref.log_prob_micro_batch_size //= 4
                check_local(config, actor=True, rollout=True, reference=True)
                config.actor.ppo_logical_micro_batch_size = 32
                with self.assertRaises(AssertionError):
                    check_local(config, actor=True, rollout=False, reference=False)

    def test_new_namespace_is_native4_easy_only(self):
        base = '/data/user/jxu957/experiments/easy2560-corollout-unit'
        run = dict(profile='common8', execution_site='hpc3', training_world_size=4,
                   execution_backend='slurm', easy2560_selection=True)
        self.assertEqual(str(validate_base(base, run)), base)
        for updates in (dict(easy2560_selection=False), dict(training_world_size=8),
                        dict(execution_backend='other')):
            with self.subTest(updates=updates), self.assertRaises(AssertionError):
                validate_base(base, dict(run, **updates))

    def run_actor_minibatch(self, physical):
        old, delta, advantages, mask = self.inputs()
        parameter = torch.nn.Parameter(torch.tensor(.23, dtype=torch.float64))
        optimizer = torch.optim.SGD([parameter], lr=.01)
        forwards = []; backwards = []; writes = []
        parameter.register_hook(lambda grad: backwards.append(grad.clone()))
        def append(metrics, values):
            for key, value in values.items(): metrics.setdefault(key, []).append(value)
        method = actual_function('verl/workers/actor/dp_actor.py', '_run_minibatch',
            dict(torch=torch, FSDP=type('NoFSDP', (), {}),
                 core_algos=NS(compute_policy_loss=PG, kl_penalty=KL),
                 require_finite=require_finite, append_to_dict=append))
        class Batch(dict):
            def split(self, count):
                return [Batch({key: value[i:i+count] for key, value in self.items()})
                        for i in range(0, self['responses'].shape[0], count)]
            def cuda(self, **kwargs): return self
        batch = Batch(responses=torch.ones_like(old, dtype=torch.long), loss_mask=mask,
            old_log_probs=old, advantages=advantages, feature=delta, ref_log_prob=old-.1)
        def forward(*, micro_batch, **kwargs):
            forwards.append(micro_batch['responses'].shape[0])
            new = micro_batch['old_log_probs'] + parameter * micro_batch['feature']
            return parameter.square().expand_as(new), new
        def step():
            writes.append(parameter.grad.clone()); optimizer.step()
            return parameter.grad.abs()
        actor = NS(actor_module=object(), actor_optimizer=optimizer,
            config=NS(use_dynamic_bsz=False, ppo_micro_batch_size=physical,
                ppo_logical_micro_batch_size=4, state_masking=True, entropy_coeff=.01,
                use_kl_loss=True, kl_loss_coef=.1, kl_loss_type='low_var_kl', clip_ratio=.2),
            gradient_accumulation=32//physical, _forward_micro_batch=forward, _optimizer_step=step)
        with patch('torch.cuda.current_device', return_value=0):
            method(actor, batch, temperature=1., safety_enabled=False, safety=None,
                   frozen=None, tail_limit=None, metrics={})
        return parameter.detach(), writes, forwards, backwards

    def test_actual_actor_method_one_forward_and_backward_at32_matches_eight_at4(self):
        expected = None
        for physical in (4, 8, 16, 32):
            value, writes, forwards, backwards = self.run_actor_minibatch(physical)
            self.assertEqual(forwards, [physical] * (32//physical))
            self.assertEqual(len(backwards), 32//physical)
            self.assertEqual(len(writes), 1)
            if expected is None: expected = value, writes[0]
            else:
                torch.testing.assert_close(value, expected[0], atol=1e-14, rtol=1e-14)
                torch.testing.assert_close(writes[0], expected[1], atol=1e-14, rtol=1e-14)


if __name__ == '__main__': unittest.main()
