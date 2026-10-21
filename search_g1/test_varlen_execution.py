"""Model structure follows the loaded configuration, including native 7B restore."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from search_g1.varlen_execution import verify_execution_configuration


class Qwen2FlashAttention2(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError('configuration verification must never run a forward')


class LoadedModel(torch.nn.Module):
    def __init__(self, configured, actual, *, master=torch.float32, compute=torch.bfloat16):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation='flash_attention_2', num_hidden_layers=configured)
        self.layers = torch.nn.ModuleList(Qwen2FlashAttention2() for _ in range(actual))
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=master))
        self.mixed_precision = SimpleNamespace(param_dtype=compute)


class VarlenExecutionTest(unittest.TestCase):
    def check(self, model):
        def publish(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SEARCH_G1_BASE=tmp), \
             patch.dict(sys.modules, {'search_g1.full_state_restore': SimpleNamespace(publish=publish)}):
            verify_execution_configuration(model, 2)
            return json.loads((Path(tmp)/'logs/actual-varlen-execution-rank-2.json').read_text())

    def test_loaded_7b28_and_original_3b36_structures_both_validate_without_forward(self):
        for count in (28, 36):
            with self.subTest(layers=count):
                proof = self.check(LoadedModel(count, count))
                self.assertEqual(proof['actual_qwen_flash_attention_layers'], count)
                self.assertEqual(proof['configured_transformer_layers'], count)
                self.assertFalse(proof['actual_loaded_model_forward'])
                self.assertNotIn('optimizer_writes', proof)

    def test_incomplete_attention_conversion_still_rejected(self):
        with self.assertRaises(AssertionError):
            self.check(LoadedModel(28, 27))

    def test_precision_mismatch_still_rejected(self):
        for master, compute in ((torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)):
            with self.subTest(master=master, compute=compute), self.assertRaises(AssertionError):
                self.check(LoadedModel(28, 28, master=master, compute=compute))


if __name__ == '__main__':
    unittest.main()
