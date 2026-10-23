"""CPU protocol checks of actual production capture helpers, without GPU work.

AST extraction executes the real pure helper bodies, not a copied implementation.
An optional installed-vLLM case exercises its CPU request-group parent method.
Neither case authenticates native sampling or proves model probability parity.
"""
import ast
import copy
import math
import os
import hashlib
import json
import tempfile
import importlib.util
from pathlib import Path
from types import SimpleNamespace as S
from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import unittest
from unittest.mock import patch
from reproduction.common8 import sampling_capture as stop_capture

ROOT = Path(__file__).resolve().parents[2]


def helper(relative, name):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(math=math, copy=copy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]


capture = helper('verl/third_party/vllm/vllm_v_0_6_3/llm.py', 'capture_request_behavior')
sampler = helper('verl/workers/rollout/vllm_rollout/vllm_rollout.py', 'actual_sampling_metadata')


def actual_output_method():
    relative = 'verl/third_party/vllm/vllm_v_0_6_3/llm.py'
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLM')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_post_process_outputs')
    namespace = dict(torch=torch, pad_sequence=pad_sequence, capture_request_behavior=capture,
                     List=List, Tuple=Tuple, RequestOutput=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace['_post_process_outputs']


def request(tokens=(5, 7), values=(-.2, -.3), name='actual-1'):
    return S(request_id=name, prompt_token_ids=[7, 8], finished=True,
             outputs=[S(token_ids=list(tokens), logprobs=[{token: S(logprob=value)}
                          for token, value in zip(tokens, values)], finish_reason='stop', stop_reason=7)])


def parameters():
    return S(n=1, best_of=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.,
             presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
             logprobs=1, min_tokens=0, max_tokens=500, logits_processors=None,
             truncate_prompt_tokens=None, ignore_eos=False, stop=['</answer>'], stop_token_ids=[7])


def stop_provenance_fixture(record, *, run_id='cpu-fixture-run', slurm_job_id='12345',
                            source_sha256=None, effective_stop_token_ids=None):
    """Constructed CPU schema fixture, never native/file/GPU proof.

    The caller supplies the same complete sampling map as actual_sampling_metadata.
    For smaller historical fixtures use complete_sampling_fixture first and bind
    that returned map consistently in the record, trace and expected sampler.
    """
    sampling = copy.deepcopy(record['sampling'])
    if set(sampling) != stop_capture.SAMPLING_KEYS:
        raise ValueError('CPU fixture sampling must be the complete capture map')
    eos, pad = record['eos_token_id'], record['pad_token_id']
    stops = ([eos] if effective_stop_token_ids is None else list(effective_stop_token_ids))
    stops = sorted(set(stops))
    caller_stop = sorted(set(sampling['stop_token_ids'] or []))
    generation_eos = stops
    effective = copy.deepcopy(sampling)
    effective['stop_token_ids'] = sorted(set(caller_stop) | {value for value in generation_eos if value != eos})
    generation = dict(schema_version=1, role=stop_capture.GENERATION_ROLE,
        run_id=run_id, slurm_job_id=slurm_job_id, source_commit='0' * 40,
        source_map_sha256=stop_capture.digest(source_sha256 or {'constructed_cpu_fixture.py': 'b' * 64}),
        model_path='/constructed-cpu-fixture/model',
        generation_config_path='/constructed-cpu-fixture/model/generation_config.json',
        generation_config_sha256='c' * 64, generation_config_eos_token_ids=generation_eos,
        loader_source=dict(path='/constructed-cpu-fixture/loader.py', sha256='d' * 64,
                           callable='constructed_cpu_fixture.loader'),
        actual_loaded_fields_sha256=stop_capture.digest({'eos_token_id': generation_eos}))
    proof = dict(schema_version=1, role=stop_capture.ROLE,
        rollout_call_id=record['rollout_call_id'], row_id=record['row_id'],
        request_id=record['request_id'], seq_id=0,
        actual_request_tokens_sha256=stop_capture.digest(record['actual_request_token_ids']),
        caller_sampling_sha256=stop_capture.digest(sampling),
        effective_sampling_sha256=stop_capture.digest(effective), caller_is_actual_group_input=True,
        seq_eos_token_id=eos, tokenizer_eos_token_id=eos, tokenizer_pad_token_id=pad,
        effective_stop_token_ids=effective['stop_token_ids'],
        effective_all_stop_token_ids=sorted(set(caller_stop) | set(generation_eos) | {eos}),
        effective_sampling=effective, generation_config=generation,
        update_source=dict(path='/constructed-cpu-fixture/sampling_params.py', sha256='e' * 64,
                           callable='constructed_cpu_fixture.update_from_generation_config'))
    stop_capture.validate_effective_stop_provenance(proof, request_id=record['request_id'],
        actual_request_token_ids=record['actual_request_token_ids'], sampling=sampling,
        eos_token_id=eos, pad_token_id=pad)
    return proof


def complete_sampling_fixture(values):
    """Explicit CPU-only defaults for real captured optional metadata keys."""
    result = {key: None for key in stop_capture.SAMPLING_KEYS}
    result.update(values)
    return result


class FixtureParams(S):
    """A CPU metadata clone fixture, not vLLM sampling or stop execution."""
    def update_from_generation_config(self, fields, eos):
        ids = set(fields.get('eos_token_id', []))
        self._all_stop_token_ids = set(self.stop_token_ids) | ids | ({eos} if eos is not None else set())
        if not self.ignore_eos:
            self.stop_token_ids = sorted(set(self.stop_token_ids) | (ids - {eos}))

    @property
    def all_stop_token_ids(self):
        return self._all_stop_token_ids


class FixtureBase:
    def _create_sequence_group_with_sampling(self, request_id, seq, params, *args, **kwargs):
        self.actual_base_calls.append((request_id, seq, params))
        effective = copy.deepcopy(params)
        effective.update_from_generation_config(self.generation_config_fields, seq.eos_token_id)
        return S(request_id=request_id, get_seqs=lambda: [seq],
                 prompt_token_ids=list(seq.prompt_token_ids), sampling_params=effective)


def actual_engine_hook_class():
    relative = 'verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py'
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMEngine')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and
                  n.name == '_create_sequence_group_with_sampling')
    extracted = ast.ClassDef(name='ActualHook', bases=[ast.Name(id='FixtureBase', ctx=ast.Load())],
                            keywords=[], body=[method], decorator_list=[])
    namespace = dict(FixtureBase=FixtureBase)
    tree = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    exec(compile(tree, relative, 'exec'), namespace)
    return namespace['ActualHook']


def fixture_loader(config):
    return json.loads((Path(config.model) / 'generation_config.json').read_text())


def fixture_transformers_loader(config):
    from transformers import GenerationConfig
    return GenerationConfig.from_pretrained(config.model).to_diff_dict()


class SamplingCaptureTests(unittest.TestCase):
    def test_actual_fsdp_guard_rejects_hidden_recompute_and_sampled_smoke(self):
        relative = 'verl/workers/fsdp_workers.py'
        tree = ast.parse((ROOT / relative).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_sequences')
        guard = method.body[0]
        self.assertIsInstance(guard, ast.If)
        namespace = dict(os=os)
        code = compile(ast.Module(body=[guard], type_ignores=[]), relative, 'exec')
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'common8'}):
            for meta in ({}, {'recompute_log_prob': True}, {'common8_model_smoke': True, 'do_sample': True}):
                namespace['prompts'] = S(meta_info=meta)
                with self.assertRaises(ValueError): exec(code, namespace)
            for meta in ({'recompute_log_prob': False},
                         {'common8_model_smoke': True, 'do_sample': False, 'recompute_log_prob': True}):
                namespace['prompts'] = S(meta_info=meta); exec(code, namespace)
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'a800'}):
            namespace['prompts'] = S(meta_info={}); exec(code, namespace)

    def test_real_postprocess_legacy_tuple_and_common8_prepad_records(self):
        method = actual_output_method()
        worker = S(llm_engine=S(tokenizer=S(pad_token_id=7, eos_token_id=7)))
        requests = [request(), request(tokens=(7,), values=(-1.25,), name='actual-2')]
        legacy = method(worker, requests)
        self.assertEqual(len(legacy), 2)
        worker.capture_behavior_trace = True
        tokens, probabilities, records = method(worker, requests)
        self.assertEqual(tokens.tolist(), [[5, 7], [7, 7]])
        self.assertEqual(records[1]['raw_generated_token_count'], 1)
        self.assertEqual(records[1]['selected_logprobs'], [-1.25])
        self.assertEqual(probabilities[1, 0].item(), -1.25)
        self.assertTrue(torch.equal(tokens, legacy[0]))
        self.assertTrue(torch.equal(probabilities, legacy[1]))

    def test_raw_eos_equal_padding_is_retained_by_actual_length(self):
        rows = capture([request(), request(tokens=(7,), values=(-1.25,), name='actual-2')])
        self.assertEqual([r['raw_generated_token_count'] for r in rows], [2, 1])
        self.assertEqual(rows[1]['raw_generated_token_ids'], [7])
        self.assertEqual(rows[1]['selected_logprobs'], [-1.25])
        self.assertEqual(rows[0]['actual_request_token_ids'], [7, 8])
        self.assertEqual(rows[0]['finish_reason'], 'stop')
        self.assertEqual(rows[0]['stop_reason'], 7)

    def test_missing_truncated_nonfinite_logprob_reject(self):
        for modify in (lambda r: setattr(r.outputs[0], 'logprobs', None),
                       lambda r: r.outputs[0].logprobs.pop(),
                       lambda r: r.outputs[0].logprobs.__setitem__(0, {}),
                       lambda r: setattr(r.outputs[0].logprobs[0][5], 'logprob', float('nan'))):
            r = request(); modify(r)
            with self.assertRaises(ValueError): capture([r])

    def test_incomplete_n2_invalid_tokens_duplicate_request_reject(self):
        for modify in (lambda r: setattr(r, 'finished', False),
                       lambda r: r.outputs.append(r.outputs[0]),
                       lambda r: setattr(r, 'prompt_token_ids', [True])):
            r = request(); modify(r)
            with self.assertRaises(ValueError): capture([r])
        with self.assertRaises(ValueError): capture([request(), request()])

    def test_capture_does_not_alias_mutable_request(self):
        r = request(); row = capture([r])[0]
        r.outputs[0].token_ids[0] = 99; r.prompt_token_ids[0] = 99
        self.assertEqual(row['raw_generated_token_ids'], [5, 7]); self.assertEqual(row['actual_request_token_ids'], [7, 8])

    def test_effective_neutral_train_and_greedy_metadata(self):
        p = parameters(); result = sampler(p, True)
        self.assertEqual(result['temperature'], 1.0)
        self.assertEqual(result['logprob_semantics'], 'selected_token_logprob_at_actual_sampling_call')
        p.stop.append('changed'); self.assertEqual(result['stop'], ['</answer>'])
        p.temperature = 0.; self.assertEqual(sampler(p, False)['temperature'], 0.)

    def test_actual_transformed_policy_reject(self):
        for key, value in [('top_p', .95), ('temperature', .9), ('n', 2),
                           ('logits_processors', [object()]), ('repetition_penalty', 1.01),
                           ('min_tokens', 1), ('ignore_eos', True), ('truncate_prompt_tokens', 20),
                           ('logit_bias', {5: 1.}), ('allowed_token_ids', [5])]:
            p = parameters(); setattr(p, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): sampler(p, True)


class ActualGroupCaptureTests(unittest.TestCase):
    """Actual production hook with CPU objects; no native sampling/health proof."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name).resolve() / 'model'
        self.model.mkdir()
        self.path = self.model / 'generation_config.json'
        self.document = {'eos_token_id': [7, 9]}
        raw = json.dumps(self.document).encode()
        self.path.write_bytes(raw)
        self.run = dict(profile='common8', run_id='cpu-fixture-run',
            initial_model=str(self.model), source_commit='0' * 40,
            source_sha256={'constructed_cpu_fixture.py': 'b' * 64},
            initial_model_sha256={'generation_config.json': hashlib.sha256(raw).hexdigest()})
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345'}):
            document, source = stop_capture.generation_source(S(model=str(self.model)), fixture_loader, run=self.run)
        self.generation = stop_capture.loaded_generation(source, document, document)
        self.engine = actual_engine_hook_class()()
        self.engine.generation_config_fields = copy.deepcopy(document)
        self.engine._common8_generation_provenance = self.generation
        self.engine.tokenizer = S(eos_token_id=7, pad_token_id=0)
        self.engine.actual_base_calls = []
        self.params = FixtureParams(**vars(parameters()))
        self.params.stop = []; self.params.stop_token_ids = []
        self.metadata = sampler(self.params, True)
        self.seq = S(seq_id=12, eos_token_id=7, prompt_token_ids=[7, 8])

    def scope(self, rows=('cpu-row',), requests=((7, 8),)):
        return stop_capture.capture_scope(self.engine, self.params, self.metadata,
                                          'cpu-call', rows, requests)

    def completion(self, name='actual-1'):
        return capture([request(name=name)])[0]

    def record(self):
        with self.scope() as actual:
            group = self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params,
                                                                     arrival_time=0)
            rows = [self.completion()]
            actual.attach(rows)
        return group, rows[0]

    def validate(self, proof, **changes):
        kwargs = dict(request_id='actual-1', actual_request_token_ids=[7, 8],
                      sampling=self.metadata, eos_token_id=7, pad_token_id=0)
        kwargs.update(changes)
        return stop_capture.validate_effective_stop_provenance(proof, **kwargs)

    def test_actual_hook_calls_base_once_keeps_caller_and_output_and_attaches_effective_eos(self):
        prior = copy.deepcopy(vars(self.params))
        original = self.completion()
        group, row = self.record()
        self.assertEqual(vars(self.params), prior)
        self.assertEqual(len(self.engine.actual_base_calls), 1)
        self.assertIs(self.engine.actual_base_calls[0][2], self.params)
        self.assertEqual({key: row[key] for key in original}, original)
        self.assertEqual(group.sampling_params.stop_token_ids, [9])
        proof = row['effective_stop_provenance']
        self.assertEqual(set(proof), stop_capture.PROVENANCE_FIELDS)
        self.assertEqual(set(proof['generation_config']), stop_capture.GENERATION_FIELDS)
        self.assertEqual(proof['seq_id'], 12)
        self.assertEqual(proof['effective_stop_token_ids'], [9])
        self.assertEqual(self.validate(proof), (frozenset((7, 9)), 7))
        self.assertIsNone(self.engine._common8_stop_capture)

    @unittest.skipUnless(importlib.util.find_spec('vllm'),
                         'Local CPU fixture environment lacks vLLM; target must execute this case')
    def test_installed_vllm_parent_group_cpu_clone_stop_and_capture_binding(self):
        """Real installed parent/params/sequence; constructed CPU input, no engine init."""
        from vllm.engine.llm_engine import LLMEngine as InstalledEngine
        from vllm.sampling_params import SamplingParams
        from vllm.sequence import Sequence

        self.assertFalse(torch.cuda.is_initialized())
        relative = 'verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py'
        source = ast.parse((ROOT / relative).read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'LLMEngine')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and
                      node.name == '_create_sequence_group_with_sampling')
        extracted = ast.ClassDef(name='InstalledParentHook',
            bases=[ast.Name(id='InstalledEngine', ctx=ast.Load())],
            keywords=[], body=[method], decorator_list=[])
        namespace = dict(InstalledEngine=InstalledEngine)
        code = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
        exec(compile(code, relative, 'exec'), namespace)
        hook = namespace['InstalledParentHook']

        # Both neutral representations must survive the real parent's
        # _build_logits_processors without None/[] normalization or mutation.
        for processors in (None, []):
            with self.subTest(logits_processors=processors):
                engine = hook.__new__(hook)
                engine.model_config = S(max_logprobs=1)
                engine.tokenizer = S(eos_token_id=7, pad_token_id=0)
                engine.generation_config_fields = copy.deepcopy(self.document)
                engine._common8_generation_provenance = copy.deepcopy(self.generation)
                params = SamplingParams(n=1, best_of=1, temperature=1., top_p=1., top_k=-1,
                    min_p=0., presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
                    logprobs=1, max_tokens=500, min_tokens=0, ignore_eos=False,
                    stop_token_ids=[6], logits_processors=copy.deepcopy(processors))
                metadata = sampler(params, True)
                original_stops = set(params.all_stop_token_ids)
                seq = Sequence(seq_id=12, inputs={'prompt_token_ids': [7, 8]},
                               block_size=16, eos_token_id=7)
                row = self.completion()
                original_row = copy.deepcopy(row)
                with stop_capture.capture_scope(engine, params, metadata, 'cpu-call',
                                                ['cpu-row'], [[7, 8]]) as actual:
                    group = engine._create_sequence_group_with_sampling('actual-1', seq, params,
                        arrival_time=0., lora_request=None, trace_headers={'scope': 'constructed-cpu'},
                        prompt_adapter_request=None, encoder_seq=None, priority=2)
                    self.assertIs(group.get_seqs()[0], seq)
                    self.assertIs(group.prompt_token_ids, seq.prompt_token_ids)
                    self.assertIsInstance(group.sampling_params, SamplingParams)
                    self.assertIsNot(group.sampling_params, params)
                    self.assertEqual(set(group.sampling_params.stop_token_ids), {6, 9})
                    self.assertEqual(set(group.sampling_params.all_stop_token_ids), {6, 7, 9})
                    self.assertEqual(group.sampling_params.logits_processors, processors)
                    self.assertEqual(stop_capture.canonical(sampler(params, True)),
                                     stop_capture.canonical(metadata))
                    self.assertEqual(set(params.all_stop_token_ids), original_stops)
                    changed = copy.deepcopy(row)
                    changed['actual_request_token_ids'][0] = 8
                    with self.assertRaisesRegex(ValueError, 'completion differs'):
                        actual.attach([changed])
                    actual.attach([row])
                self.assertEqual({key: row[key] for key in original_row}, original_row)
                proof = row['effective_stop_provenance']
                self.assertEqual(proof['seq_id'], seq.seq_id)
                self.assertEqual(proof['request_id'], group.request_id)
                self.assertEqual(proof['actual_request_tokens_sha256'],
                                 stop_capture.digest(seq.prompt_token_ids))
                self.assertEqual(stop_capture.validate_effective_stop_provenance(proof,
                    request_id='actual-1', actual_request_token_ids=[7, 8], sampling=metadata,
                    eos_token_id=7, pad_token_id=0), (frozenset((6, 7, 9)), 7))
                self.assertIsNone(engine._common8_stop_capture)
        self.assertFalse(torch.cuda.is_initialized())

    def test_scope_restores_on_error_and_ordinary_hook_has_no_capture(self):
        with self.assertRaisesRegex(RuntimeError, 'cpu failure'):
            with self.scope():
                raise RuntimeError('cpu failure')
        self.assertIsNone(self.engine._common8_stop_capture)
        group = self.engine._create_sequence_group_with_sampling('ordinary', self.seq, self.params)
        self.assertEqual(group.request_id, 'ordinary')
        self.assertEqual(len(self.engine.actual_base_calls), 1)
        with self.scope():
            with self.assertRaisesRegex(ValueError, 'nested'):
                with self.scope(): pass

    def test_different_caller_same_values_rejects_before_base(self):
        with self.scope():
            with self.assertRaisesRegex(ValueError, 'actual caller'):
                self.engine._create_sequence_group_with_sampling('actual-1', self.seq, copy.deepcopy(self.params))
        self.assertEqual(self.engine.actual_base_calls, [])

    def test_mutated_caller_and_memory_generation_reject(self):
        with self.scope():
            self.params.top_p = .95
            with self.assertRaisesRegex(ValueError, 'actual caller'):
                self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params)
        self.params.top_p = 1.
        self.engine.generation_config_fields['eos_token_id'] = [7, 10]
        with self.assertRaisesRegex(ValueError, 'generation config drifted'):
            with self.scope(): pass

    def test_group_caller_request_and_completion_crosswiring_reject(self):
        with self.scope() as actual:
            wrong = S(seq_id=13, eos_token_id=7, prompt_token_ids=[8, 7])
            with self.assertRaisesRegex(ValueError, 'request tokens differ'):
                self.engine._create_sequence_group_with_sampling('wrong', wrong, self.params)
        with self.scope() as actual:
            self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params)
            for rows in ([], [self.completion('different')], [self.completion(), self.completion()]):
                with self.subTest(rows=len(rows)), self.assertRaises(ValueError): actual.attach(rows)
            row = self.completion(); row['actual_request_token_ids'][0] = 8
            with self.assertRaisesRegex(ValueError, 'completion differs'): actual.attach([row])

    def test_duplicate_group_request_rejects_before_second_base(self):
        with self.scope():
            self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params)
            with self.assertRaises(ValueError):
                self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params)
        self.assertEqual(len(self.engine.actual_base_calls), 1)

    def test_additional_eos_equal_pad_is_stop_and_primary_eos_can_differ_from_tokenizer(self):
        self.engine.tokenizer.pad_token_id = 9
        self.seq.eos_token_id = 9
        _, row = self.record()
        proof = row['effective_stop_provenance']
        self.assertEqual(proof['effective_stop_token_ids'], [7])
        self.assertEqual(self.validate(proof, pad_token_id=9), (frozenset((7, 9)), 9))
        self.assertEqual(proof['tokenizer_eos_token_id'], 7)

    def test_closed_schema_hash_stop_and_sampling_drift_reject(self):
        _, row = self.record(); proof = row['effective_stop_provenance']
        changes = (lambda p: p.update(unknown=True),
                   lambda p: p.update(caller_is_actual_group_input=False),
                   lambda p: p.update(seq_id=True),
                   lambda p: p.update(effective_all_stop_token_ids=[7, 9, 98]),
                   lambda p: p['generation_config'].update(unknown=True),
                   lambda p: p['generation_config'].update(generation_config_eos_token_ids=[True, 9]),
                   lambda p: p['effective_sampling'].update(top_p=.95),
                   lambda p: p.update(caller_sampling_sha256='f' * 64))
        for change in changes:
            damaged = copy.deepcopy(proof); change(damaged)
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(damaged)
        for kwargs in (dict(request_id='wrong'), dict(actual_request_token_ids=[8, 7]),
                       dict(eos_token_id=9), dict(pad_token_id=7)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.validate(proof, **kwargs)

    def test_original_generation_file_loader_and_metadata_bindings_reject_drift(self):
        wrong_run = copy.deepcopy(self.run); wrong_run['initial_model_sha256']['generation_config.json'] = 'f' * 64
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345'}):
            with self.assertRaisesRegex(ValueError, 'original generation file'):
                stop_capture.generation_source(S(model=str(self.model)), fixture_loader, run=wrong_run)
            with self.assertRaisesRegex(ValueError, 'different model'):
                stop_capture.generation_source(S(model=str(self.model.parent)), fixture_loader, run=self.run)
        self.path.write_text('{"eos_token_id":[7,10]}')
        with self.assertRaisesRegex(ValueError, 'changed across actual loader'):
            stop_capture.loaded_generation(self.generation, self.document, self.document)
        with self.assertRaisesRegex(ValueError, 'actual loader fields'):
            stop_capture.loaded_generation(self.generation, {'eos_token_id': 7}, self.document)

    def test_restored_loader_accepts_only_verified_checkpoint_and_original_metadata(self):
        checkpoint = self.model.parent / 'checkpoint5'; checkpoint.mkdir()
        saved_document = dict(self.document, transformers_version='4.46.3')
        (checkpoint / 'generation_config.json').write_text(json.dumps(saved_document))
        run = copy.deepcopy(self.run)
        run['full_state_restore'] = dict(source_receipt='/cpu-fixture/receipt.json',
            source_receipt_sha256='e' * 64, source_checkpoint=str(checkpoint))
        bound = (Path('/cpu-fixture/receipt.json'), {'files': [dict(relative_path='generation_config.json', full_read_sha256=hashlib.sha256((checkpoint / 'generation_config.json').read_bytes()).hexdigest())]}, checkpoint, self.model.parent, self.run)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345'}), \
             patch('search_g1.full_state_restore.source', return_value=bound), \
             patch('search_g1.full_state_restore.digest', return_value='e' * 64):
            document, record = stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=run)
            self.assertEqual(record['model_path'], str(checkpoint))
            self.assertEqual(document, saved_document)
            with self.assertRaisesRegex(ValueError, 'different model'):
                stop_capture.generation_source(S(model=str(self.model)), fixture_loader, run=run)
            changed = copy.deepcopy(run); changed['full_state_restore']['source_receipt_sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=changed)
            (checkpoint / 'generation_config.json').write_text('{"eos_token_id":[7,10]}')
            with self.assertRaisesRegex(ValueError, 'saved generation file changed'):
                stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=run)
            bound[1]['files'][0]['full_read_sha256'] = hashlib.sha256((checkpoint / 'generation_config.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'sampling fields changed'):
                stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=run)

    def test_native_dynrpo_generation_metadata_matches_official_without_legacy_file_list(self):
        checkpoint = self.model.parent / 'checkpoint3'; checkpoint.mkdir()
        path = checkpoint / 'generation_config.json'
        saved = dict(self.document, transformers_version='4.46.3')
        path.write_text(json.dumps(saved))
        run = copy.deepcopy(self.run)
        run['full_state_restore'] = dict(source_receipt='/cpu-fixture/native.json',
            source_receipt_sha256='e'*64, source_checkpoint=str(checkpoint))
        verified = dict(kind='dynrpo7b_native4', actor_ref=dict(files_sha256={}))
        bound = (Path('/cpu-fixture/native.json'), verified, checkpoint, self.model.parent, self.run)
        with patch.dict(os.environ, {'SLURM_JOB_ID':'12345'}), \
             patch('search_g1.full_state_restore.source', return_value=bound), \
             patch('search_g1.full_state_restore.digest', return_value='e'*64):
            document, record = stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=run)
            self.assertEqual(document, saved)
            self.assertEqual(record['generation_config_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text(json.dumps(dict(saved, eos_token_id=[7,10])))
            with self.assertRaisesRegex(ValueError, 'sampling fields changed'):
                stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=run)
            path.write_text(json.dumps(saved))
            verified['actor_ref']['files_sha256']['generation_config.json'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'saved generation file changed'):
                stop_capture.generation_source(S(model=str(checkpoint)), fixture_loader, run=run)

    def test_generation_metadata_symlink_and_oversized_files_reject(self):
        target = self.model / 'other.json'; target.write_text('{}')
        self.path.unlink(); self.path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'nonregular'): stop_capture._read(self.path)
        self.path.unlink(); self.path.write_bytes(b'x' * (stop_capture.LIMIT + 1))
        with self.assertRaisesRegex(ValueError, 'oversized'): stop_capture._read(self.path)

    @unittest.skipUnless(importlib.util.find_spec('transformers'),
                         'Local CPU fixture environment lacks transformers; target must execute this case')
    def test_actual_constructor_loader_branch_binds_transformers_file_fields(self):
        # Execute the exact production constructor branch. The CPU loader reads
        # the local GenerationConfig file; it is not a native vLLM loader claim.
        relative = 'verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py'
        tree = ast.parse((ROOT / relative).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMEngine')
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        branch = next(n for n in constructor.body if isinstance(n, ast.If) and
                      'SEARCH_G1_PROFILE' in ast.unparse(n.test))
        base = self.model.parent / 'base'; (base / 'logs').mkdir(parents=True)
        (base / 'logs/run-manifest.json').write_text(json.dumps(self.run))
        namespace = dict(os=os, self=S(), model_config=S(model=str(self.model)),
                         _load_generation_config_dict=fixture_transformers_loader)
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'common8', 'SEARCH_G1_BASE': str(base),
                                     'SLURM_JOB_ID': '12345'}):
            exec(compile(ast.Module(body=[branch], type_ignores=[]), relative, 'exec'), namespace)
        engine = namespace['self']
        self.assertEqual(engine._common8_generation_provenance['generation_config_sha256'],
                         self.run['initial_model_sha256']['generation_config.json'])
        self.assertEqual(engine._common8_generation_provenance['actual_loaded_fields_sha256'],
                         stop_capture.digest(engine.generation_config_fields))
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'a800'}):
            namespace['self'] = S()
            exec(compile(ast.Module(body=[branch], type_ignores=[]), relative, 'exec'), namespace)
        self.assertFalse(hasattr(namespace['self'], '_common8_generation_provenance'))

    def test_boolean_integer_alias_and_duplicate_sequence_reject_before_base(self):
        with self.scope():
            self.params.n = True
            with self.assertRaisesRegex(ValueError, 'actual caller'):
                self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params)
        self.params.n = 1
        with self.scope(rows=('cpu-row0', 'cpu-row1'), requests=((7, 8), (7, 8))):
            self.engine._create_sequence_group_with_sampling('actual-1', self.seq, self.params)
            with self.assertRaisesRegex(ValueError, 'duplicate actual sequence'):
                self.engine._create_sequence_group_with_sampling('actual-2', self.seq, self.params)
        self.assertEqual(len(self.engine.actual_base_calls), 1)

    def test_stop_provenance_does_not_alias_live_group_or_completion(self):
        group, row = self.record()
        proof = copy.deepcopy(row['effective_stop_provenance'])
        group.sampling_params.stop_token_ids.append(98)
        self.engine.generation_config_fields['eos_token_id'].append(98)
        self.assertEqual(row['effective_stop_provenance'], proof)
        row['selected_logprobs'][0] = -9.
        self.assertEqual(row['effective_stop_provenance'], proof)

    def test_cpu_factory_is_closed_nonphysical_and_does_not_mutate_record(self):
        row = self.completion()
        row.update(rollout_call_id='cpu-call', row_id='cpu-row', sampling=self.metadata,
                   eos_token_id=7, pad_token_id=0)
        prior = copy.deepcopy(row)
        proof = stop_provenance_fixture(row, effective_stop_token_ids=[7, 9])
        self.assertEqual(row, prior)
        self.assertEqual(self.validate(proof), (frozenset((7, 9)), 7))
        self.assertIn('constructed-cpu-fixture', proof['generation_config']['model_path'])

    def test_actual_rollout_try_scope_attaches_records_and_restores_flags(self):
        relative = 'verl/workers/rollout/vllm_rollout/vllm_rollout.py'
        tree = ast.parse((ROOT / relative).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'vLLMRollout')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_sequences')
        block = next(n for n in ast.walk(method) if isinstance(n, ast.Try) and
                     any(isinstance(v, ast.If) for v in n.body))
        parent = self
        class Inference:
            capture_behavior_trace = False
            llm_engine = parent.engine
            def generate(self, **kwargs):
                parent.assertIs(kwargs['sampling_params'], parent.params)
                parent.engine._create_sequence_group_with_sampling('actual-1', parent.seq, parent.params)
                return torch.tensor([[5, 7]]), torch.tensor([[-.2, -.3]]), [parent.completion()]
        inference = Inference()
        namespace = dict(self=S(inference_engine=inference, sampling_params=self.params),
            common8=True, sampler_actual=self.metadata, call_id='cpu-call', row_ids=['cpu-row'],
            idx_list=[[7, 8]], prior_capture=False)
        exec(compile(ast.Module(body=[block], type_ignores=[]), relative, 'exec'), namespace)
        output = namespace['output']
        self.assertEqual(output[0].tolist(), [[5, 7]])
        self.assertEqual(output[2][0]['selected_logprobs'], [-.2, -.3])
        self.assertIn('effective_stop_provenance', output[2][0])
        self.assertIs(inference.capture_behavior_trace, False)
        self.assertIsNone(self.engine._common8_stop_capture)


if __name__ == '__main__':
    unittest.main()
