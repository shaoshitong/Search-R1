"""Actual adapter transitions with synthetic native token IDs; no GPU/model."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .calibration_interaction import rollouts, settings
from .context_budget import observation_tokens


class Tokenizer:
    eos_token_id, pad_token_id, all_special_ids = 99, 0, [0, 99]

    def encode(self, text, **kwargs):
        return [ord(c)+1000 for c in text]

    def decode(self, ids, *, skip_special_tokens=False, **kwargs):
        return ''.join(chr(i-1000) for i in ids if i not in self.all_special_ids)


class Native:
    def __init__(self, tokenizer, script):
        self.tokenizer, self.script, self.requests = tokenizer, script, []

    def generate(self, requests, params, **kwargs):
        ordinal = len(self.requests); self.requests.append(requests)
        return [SimpleNamespace(prompt_token_ids=r['prompt_token_ids'], outputs=[SimpleNamespace(
            token_ids=self.tokenizer.encode(self.script(ordinal))+[99])]) for r in requests]


class CalibrationInteraction(unittest.TestCase):
    def execute(self, script, *, prompt='Question?', evidence='x'*1000):
        token = Tokenizer(); native = Native(token, script); calls = []
        def search(_retriever, query):
            calls.append(query); return evidence
        with TemporaryDirectory() as root:
            Path(root, 'generation_config.json').write_text(json.dumps({'eos_token_id':[99]}))
            with patch('search_g1.async_rollout_runtime.BoundedRetriever.search', search):
                result = rollouts(native, [prompt], token, True, settings(root, 'http://fixture/retrieve'))
        return token, native, result, calls

    def test_wrapper_budget_and_exact_original_prefix_next_request(self):
        token, native, result, calls = self.execute(
            lambda n:'<search>query</search>' if n==0 else '<answer>A</answer>')
        observation='\n\n<information>'+'x'*1000+'</information>\n\n'
        expected,_=observation_tokens(token,observation,500)
        self.assertEqual(native.requests[1][0]['prompt_token_ids'],
            token.encode('Question?')+token.encode('<search>query</search>')+expected)
        self.assertEqual(calls,['query']); self.assertEqual(result[2],[1])
        text=result[0][0];a,b=result[3][0][0]
        self.assertTrue(text[a:b].startswith('\n\n<information>'))
        self.assertTrue(text[a:b].endswith('</information>\n\n'))
        self.assertEqual(len(token.encode(text[a:b])),500)
        self.assertEqual(result[1][0][1][0],b)

    def test_early_final_uses_shared_budget_and_forbids_extra_retrieval(self):
        _, native, result, calls = self.execute(
            lambda n:'x'*470+'<search>q</search>',prompt='Q'*2048)
        self.assertLess(len(native.requests),5)
        self.assertEqual(result[2],[len(native.requests)-1])
        self.assertEqual(len(calls),len(native.requests)-1)
        for request in native.requests:
            self.assertLessEqual(len(request[0]['prompt_token_ids'])+500,4096)

    def test_malformed_correction_provenance_matches_training_executor(self):
        _, _, result, calls = self.execute(lambda n:'bad' if n==0 else '<answer>A</answer>')
        self.assertEqual(calls,[]);self.assertEqual(result[2],[0])
        a,b=result[4][0][0]
        self.assertIn('My previous action is invalid.',result[0][0][a:b])
        self.assertEqual(result[3],[[]])

    def test_intervention_budget_rejects_overflow_instead_of_cropping(self):
        source=Path(__file__).resolve().parents[2]/'ts_probe_dump.py'
        tree=ast.parse(source.read_text());fn=next(n for n in tree.body
            if isinstance(n,ast.FunctionDef)and n.name=='gen_batch')
        namespace={'SamplingParams':lambda **kw:SimpleNamespace(**kw)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        token=Tokenizer();llm=SimpleNamespace(_g1_calibration_total_limit=4096,
            _g1_calibration_prompt_limit=4096,_g1_calibration_tokenizer=token,
            _g1_calibration_context_stats=[],llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)))
        with self.assertRaisesRegex(ValueError,'never silently crop'):
            namespace['gen_batch'](llm,['q'*4065],32,False,['</answer>'])


if __name__=='__main__':unittest.main()
