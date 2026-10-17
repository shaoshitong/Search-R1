import ast
import datetime
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
import unittest
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from search_g1.compute_precision import ComputePrecision


@contextmanager
def preserve_tf32():
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class Config(dict):
    def __init__(self, **values):
        super().__init__(entropy_coeff=0.0, **values)

    def __getattr__(self, name):
        return self[name]


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 4)
        self.head = torch.nn.Linear(4, 11)
        self.forward_calls = 0
        self.return_bfloat16 = False

    def forward(self, input_ids, **kwargs):
        self.forward_calls += 1
        logits = self.head(self.embedding(input_ids))
        return SimpleNamespace(logits=logits.bfloat16() if self.return_bfloat16 else logits)


def collective_precision_worker(rank, store, result_prefix):
    dist.init_process_group('gloo', init_method='file://' + store, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=20))
    group = dist.new_group([0, 1], backend='gloo')
    results = []
    try:
        cls = actual_methods()
        batch = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                     position_ids=torch.tensor([[0, 1, 2]]), responses=torch.tensor([[3]]))
        for fault in ('module_dtype', 'worker_tf32', 'logits_dtype'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model = TinyPolicy()
            model.process_group = group
            if fault == 'module_dtype' and rank == 1:
                model.bfloat16()
            initial = [value.detach().clone() for value in model.parameters()]
            simulated_writes = 0
            error = None
            try:
                with patch('torch.compile', side_effect=lambda fn, **kwargs: fn):
                    policy = cls(Config(compute_dtype='float32', allow_tf32=False,
                                        ulysses_sequence_parallel_size=1), model)
                if fault == 'worker_tf32' and rank == 1:
                    torch.backends.cuda.matmul.allow_tf32 = True
                if fault == 'logits_dtype' and rank == 1:
                    model.return_bfloat16 = True
                policy._forward_micro_batch(batch, 1.0)
                simulated_writes += 1
            except (RuntimeError, ValueError) as failure:
                error = str(failure)
            results.append(dict(rank=rank, fault=fault, error=error, simulated_writes=simulated_writes,
                                forward_calls=model.forward_calls,
                                parameters_unchanged=all(torch.equal(before, after)
                                                         for before, after in zip(initial, model.parameters()))))
        Path(result_prefix + str(rank)).write_text(json.dumps(results))
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


def actual_methods():
    source = Path(__file__).resolve().parents[1] / 'verl/workers/actor/dp_actor.py'
    tree = ast.parse(source.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'DataParallelPPOActor')
    methods = [node for node in actor.body if isinstance(node, ast.FunctionDef)
               and node.name in ('__init__', '_forward_micro_batch')]

    class Base:
        def __init__(self, config):
            self.config = config

    def logprobs(logits, labels):
        return torch.log_softmax(logits.float(), -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def entropy(logits):
        logs = torch.log_softmax(logits.float(), -1)
        return -(logs.exp() * logs).sum(-1)

    namespace = dict(torch=torch, nn=torch.nn, os=os, json=json, Tuple=Tuple, Optional=Optional, BasePPOActor=Base,
                     ComputePrecision=ComputePrecision, logprobs_from_logits=logprobs,
                     logprobs_of_labels_v2=logprobs,
                     verl_F=SimpleNamespace(entropy_from_logits=entropy))
    cls = ast.ClassDef(name='DataParallelPPOActor', bases=[ast.Name(id='BasePPOActor', ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['DataParallelPPOActor']


class ComputePrecisionTests(unittest.TestCase):
    def test_default_context_is_original_bfloat16(self):
        precision = ComputePrecision.from_config({})
        with patch('torch.autocast') as context:
            precision.autocast()
        context.assert_called_once_with(device_type='cuda', dtype=torch.bfloat16, enabled=True)

    def test_explicit_float32_disables_scoped_cuda_autocast(self):
        precision = ComputePrecision.from_config(dict(compute_dtype='float32'))
        with patch('torch.autocast') as context:
            precision.autocast()
        context.assert_called_once_with(device_type='cuda', dtype=torch.bfloat16, enabled=False)

    def test_invalid_configuration_rejected(self):
        for dtype in ('fp32', 'bf16', 'float16', None, 32):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                ComputePrecision(dtype)
        for flag in ('false', 0, 1, [], {}):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ComputePrecision(allow_tf32=flag)

    def test_tf32_null_preserves_flags_and_explicit_setting_changes_both(self):
        with preserve_tf32():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = False
            ComputePrecision().configure_worker()
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            for value in (False, True):
                ComputePrecision(allow_tf32=value).configure_worker()
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, value)
                self.assertEqual(torch.backends.cudnn.allow_tf32, value)

    def test_float32_requires_parameters_buffers_and_all_fsdp_casts(self):
        precision = ComputePrecision('float32')
        model = TinyPolicy()
        precision.validate_module(model)
        with self.assertRaises(ValueError):
            precision.validate_module(model.bfloat16())
        model.float().register_buffer('frequency', torch.ones(2, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            precision.validate_module(model)
        model.float()
        for field in ('param_dtype', 'reduce_dtype', 'buffer_dtype'):
            model.head.mixed_precision = SimpleNamespace(**{field: torch.bfloat16})
            with self.subTest(field=field), self.assertRaises(ValueError):
                precision.validate_module(model)
        model.head.mixed_precision = SimpleNamespace(param_dtype=torch.float32,
                                                     reduce_dtype=torch.float32, buffer_dtype=None)
        precision.validate_module(model)
        with self.assertRaises(ValueError):
            precision.validate_module(torch.nn.Identity())
        # The default path remains compatible with the old BF16 policy.
        ComputePrecision().validate_module(model.bfloat16())

    def test_actual_constructor_rejects_bad_module_before_flag_mutation(self):
        cls = actual_methods()
        config = Config(compute_dtype='float32', allow_tf32=True, ulysses_sequence_parallel_size=1)
        with preserve_tf32():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with self.assertRaises(ValueError):
                cls(config, TinyPolicy().bfloat16())
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_conflicting_worker_setting_rejected_before_forward(self):
        with preserve_tf32():
            first = ComputePrecision('float32', allow_tf32=False)
            second = ComputePrecision('float32', allow_tf32=True)
            first.configure_worker()
            first.validate_worker()
            second.configure_worker()
            with patch('torch.autocast') as context, self.assertRaises(RuntimeError):
                first.autocast()
            context.assert_not_called()
            with self.assertRaises(RuntimeError):
                first.observe_logits(torch.zeros(2, dtype=torch.float32))
            # Legacy policies impose no extra requirement on current flags.
            ComputePrecision().observe_logits(torch.zeros(2, dtype=torch.bfloat16))

    def test_actual_actor_and_reference_forward_preserve_slices_and_context(self):
        cls = actual_methods()
        torch.manual_seed(13)
        model = TinyPolicy()
        batch = dict(input_ids=torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
                     attention_mask=torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]),
                     position_ids=torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]]),
                     responses=torch.tensor([[2, 3], [6, 7]]))
        temperature = 0.7
        full = model(**batch).logits / temperature
        expected = torch.log_softmax(full[:, -3:-1].float(), -1).gather(
            -1, batch['responses'].unsqueeze(-1)).squeeze(-1)
        original = torch.autocast

        def cpu_context(**kwargs):
            # Execute the actual method using real CPU autocast; native CUDA
            # context argument selection is checked separately above.
            kwargs['device_type'] = 'cpu'
            return original(**kwargs)

        with patch('torch.compile', side_effect=lambda fn, **kwargs: fn):
            policies = [cls(Config(compute_dtype='float32', ulysses_sequence_parallel_size=1), model,
                            actor_optimizer=object() if role == 'actor' else None)
                        for role in ('actor', 'reference')]
            default = cls(Config(ulysses_sequence_parallel_size=1), model)
        with patch('torch.autocast', side_effect=cpu_context):
            with original(device_type='cpu', dtype=torch.bfloat16):
                self.assertTrue(torch.is_autocast_cpu_enabled())
                for policy in policies:
                    entropy, logs = policy._forward_micro_batch(batch, temperature)
                    self.assertEqual(entropy.shape, (2, 2))
                    self.assertTrue(torch.equal(logs, expected))
                    self.assertEqual(policy.last_forward_precision['logits_dtype'], 'torch.float32')
                    self.assertTrue(torch.is_autocast_cpu_enabled())
                default._forward_micro_batch(batch, temperature)
                self.assertEqual(default.last_forward_precision['logits_dtype'], 'torch.bfloat16')
        self.assertIs(torch.autocast, original)

    def test_actual_logits_validation_and_observation(self):
        precision = ComputePrecision('float32')
        with self.assertRaises(RuntimeError):
            precision.observe_logits(torch.zeros(2, 3, dtype=torch.bfloat16))
        logits = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)
        observation = precision.observe_logits(logits)
        self.assertEqual(observation['logits_shape'], [2, 3])
        self.assertEqual(observation['logits_device'], 'cpu')
        self.assertEqual(observation['allow_tf32_matmul'], torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(observation['allow_tf32_cudnn'], torch.backends.cudnn.allow_tf32)
        self.assertEqual(observation['autocast_cuda_enabled'], torch.is_autocast_enabled())
        self.assertTrue(logits.requires_grad)

    def test_legacy_default_does_not_add_collectives(self):
        precision = ComputePrecision()
        with patch('search_g1.compute_precision._reject_collectively') as collective, patch('torch.autocast'):
            precision.validate_module(TinyPolicy().bfloat16())
            precision.autocast()
            precision.observe_logits(torch.zeros(2, dtype=torch.bfloat16))
        collective.assert_not_called()

    def test_actual_two_rank_precision_faults_reject_both_without_writes(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(collective_precision_worker, args=(root + '/store', root + '/rank'), nprocs=2, join=True)
            reports = [json.loads(Path(root + '/rank' + str(rank)).read_text()) for rank in range(2)]
        for rank, rows in enumerate(reports):
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertIsNotNone(row['error'])
                self.assertIn('collectively', row['error'])
                self.assertEqual(row['simulated_writes'], 0)
                self.assertTrue(row['parameters_unchanged'])
                self.assertEqual(row['forward_calls'], 1 if row['fault'] == 'logits_dtype' else 0)
                if rank == 0:
                    self.assertIn('another policy rank', row['error'])
                elif row['fault'] == 'worker_tf32':
                    self.assertIn('Worker TF32', row['error'])
                else:
                    self.assertIn('torch.bfloat16', row['error'])


if __name__ == '__main__':
    unittest.main()
