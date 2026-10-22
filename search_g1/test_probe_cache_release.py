"""CPU score/RNG mechanics with CUDA accounting mocked, never a GPU fit claim."""
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch


def load_server():
    class App:
        def post(self, *args, **kwargs): return lambda function:function
        get = post
    fake = {}
    for name in ('fastapi', 'pydantic', 'transformers', 'uvicorn', 'joblib'):
        fake[name] = ModuleType(name)
    fake['fastapi'].FastAPI = App
    fake['pydantic'].BaseModel = object
    fake['transformers'].AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(
        side_effect=AssertionError('No model load in CPU cache test')))
    fake['transformers'].AutoTokenizer = SimpleNamespace(from_pretrained=Mock(
        side_effect=AssertionError('No tokenizer load in CPU cache test')))
    path = Path(__file__).resolve().parents[1]/'probe_server.py'
    spec = importlib.util.spec_from_file_location('_probe_cache_fixture', path)
    module = importlib.util.module_from_spec(spec)
    # Server import's proxy cleanup must not change the test runner's environment.
    with patch.dict(sys.modules, fake), patch.dict(os.environ):
        spec.loader.exec_module(module)
    return module


def rng():
    return pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()


class Classifier:
    def __init__(self, scale): self.scale = scale
    def predict_proba(self, features):
        values = 1/(1+np.exp(-features[:, 0]*self.scale))
        return np.stack((1-values, values), axis=1)


class ProbeCacheTests(unittest.TestCase):
    def test_scores_forward_calls_weights_and_rng_match_without_cache_release(self):
        server = load_server()
        model = torch.nn.Linear(3, 2, bias=False).eval()
        with torch.no_grad(): model.weight.copy_(torch.tensor([[.1,.2,.3],[.2,-.1,.4]]))
        weights = model.weight.detach().clone()
        token = object(); events = []; held = []
        @contextmanager
        def exclusion():
            held.append(True)
            try: yield
            finally: held.pop()
        def feature(texts, layer, given_model, given_token):
            self.assertIs(given_model, model); self.assertIs(given_token, token)
            self.assertTrue(server.LOCK.locked()); self.assertTrue(held)
            events.append((tuple(texts), layer))
            values = torch.tensor([[len(text), sum(map(ord,text)) % 17, layer] for text in texts],
                                  dtype=torch.float32)
            with torch.no_grad(): return model(values).numpy()
        server.STATE.update(model=model, tok=token, version='saved:150:exact', round=150,
            commit={'position':'Xc', 'layer':1, 'clf':Classifier(.2)},
            boundary={'layer':2, 'clf':Classifier(.3)})
        items = [SimpleNamespace(id='a', prompt='Question one?', response='<answer>yes</answer>',
                    model_ranges=[[0,len('<answer>yes</answer>')]]),
                 SimpleNamespace(id='b', prompt='Question two?', response='<answer>no</answer>',
                    model_ranges=[[0,len('<answer>no</answer>')]])]
        request = SimpleNamespace(items=items)
        with patch.object(server, '_last_hidden', side_effect=feature), \
             patch.object(server, 'evaluator_gpu_lock', exclusion):
            before = rng()
            with patch.object(server, '_release_judge_cuda_cache') as bypass:
                original = server.judge_probe(request)
            self.assertEqual(bypass.call_count, 1)
            original_calls = list(events); events.clear()
            after_original = rng()
            self.assertEqual(before[:2], after_original[:2]); self.assertTrue(torch.equal(before[2],after_original[2]))
            def cuda_action():
                self.assertTrue(server.LOCK.locked()); self.assertTrue(held)
            stdout = io.StringIO()
            with patch.object(torch.cuda, 'synchronize', side_effect=cuda_action) as sync, \
                 patch.object(torch.cuda, 'empty_cache', side_effect=cuda_action) as empty, \
                 patch.object(torch.cuda, 'memory_allocated', return_value=64), \
                 patch.object(torch.cuda, 'memory_reserved', side_effect=[1024,128]), \
                 patch.object(torch.cuda, 'mem_get_info', side_effect=[(2000,8000),(2896,8000)]), \
                 patch.object(torch.cuda, 'current_device', return_value=0), redirect_stdout(stdout):
                actual = server.judge_probe(request)
            self.assertEqual(actual, original)
            self.assertEqual(events, original_calls); self.assertEqual(len(events), 2)
            self.assertTrue(torch.equal(model.weight, weights)); self.assertIs(server.STATE['model'], model)
            after = rng(); self.assertEqual(before[:2], after[:2]); self.assertTrue(torch.equal(before[2],after[2]))
            sync.assert_called_once_with(); empty.assert_called_once_with()
            record = json.loads(stdout.getvalue().split('=',1)[1])
            self.assertEqual(record['pid'],os.getpid()); self.assertEqual(record['version'],'saved:150:exact')
            self.assertEqual(record['released_reserved_bytes'],896)
            self.assertEqual(record['before']['allocated_bytes'],record['after']['allocated_bytes'])

    def test_empty_request_keeps_original_no_forward_no_cuda_behavior(self):
        server = load_server(); server.STATE.update(model=object(), version='saved')
        @contextmanager
        def exclusion(): yield
        with patch.object(server, 'evaluator_gpu_lock', exclusion), \
             patch.object(server, '_last_hidden') as forward, \
             patch.object(server, '_release_judge_cuda_cache') as release:
            result = server.judge_probe(SimpleNamespace(items=[]))
        self.assertEqual(result, {'results':[], 'version':'saved'})
        forward.assert_not_called(); release.assert_not_called()


if __name__ == '__main__': unittest.main()
