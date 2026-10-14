"""Tiny CPU engine tests; constructed objects are never GPU sampling proof."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import os

from reproduction.common8.sampling_capture import (
    ContinuousRequestService, RequestCapture, generation_source, loaded_generation,
)

ROOT = Path(__file__).resolve().parents[2]


def actual_helper(relative, name):
    node = next(n for n in ast.parse((ROOT/relative).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(copy=copy, math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]


read_behavior = actual_helper('verl/third_party/vllm/vllm_v_0_6_3/llm.py', 'capture_request_behavior')
sampling_metadata = actual_helper('verl/workers/rollout/vllm_rollout/vllm_rollout.py', 'actual_sampling_metadata')


class Params(S):
    def update_from_generation_config(self, fields, eos):
        self._all_stop_token_ids = set(self.stop_token_ids) | set(fields['eos_token_id']) | {eos}
        self.stop_token_ids = sorted(set(self.stop_token_ids) | (set(fields['eos_token_id'])-{eos}))

    @property
    def all_stop_token_ids(self):
        return self._all_stop_token_ids


def params():
    return Params(n=1, best_of=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
        presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        logprobs=1, prompt_logprobs=None, max_tokens=500, min_tokens=0, stop=[], stop_token_ids=[],
        ignore_eos=False, include_stop_str_in_output=False, skip_special_tokens=True,
        spaces_between_special_tokens=True, truncate_prompt_tokens=None, seed=None, detokenize=False,
        logits_processors=None, logit_bias=None, allowed_token_ids=None, guided_decoding=None)


def loader(config):
    return json.loads((Path(config.model)/'generation_config.json').read_text())


class Engine:
    def __init__(self, document, provenance):
        self.generation_config_fields = document
        self._common8_generation_provenance = provenance
        self.tokenizer = S(eos_token_id=7, pad_token_id=0)
        self.pending = {}
        self.sequence = 0
        self.aborted = []
        self.calls = []

    def add_request(self, request_id, prompt, caller):
        tokens = prompt['prompt_token_ids']
        seq = S(seq_id=self.sequence, eos_token_id=7, prompt_token_ids=tokens)
        self.sequence += 1
        self._common8_stop_capture.before(request_id, seq, caller)
        effective = copy.deepcopy(caller)
        effective.update_from_generation_config(self.generation_config_fields, seq.eos_token_id)
        group = S(request_id=request_id, prompt_token_ids=tokens, sampling_params=effective,
                  get_seqs=lambda: [seq])
        self._common8_stop_capture.after(request_id, seq, caller, group)
        self.pending[request_id] = dict(tokens=list(tokens), delay=3 if request_id == 'slow' else 1)
        self.calls.append(('add', request_id))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        self.calls.append(('step', tuple(self.pending)))
        outputs = []
        for request_id, value in list(self.pending.items()):
            value['delay'] -= 1
            finished = value['delay'] == 0
            outputs.append(S(request_id=request_id, finished=finished, prompt_token_ids=value['tokens'],
                outputs=[S(token_ids=[5, 7], logprobs=[{5: S(logprob=-0.2)}, {7: S(logprob=-0.3)}],
                           finish_reason='stop', stop_reason=7)]))
            if finished:
                del self.pending[request_id]
        return outputs

    def abort_request(self, request_id):
        self.pending.pop(request_id, None)
        self.aborted.append(request_id)


def phase_worker_fixture(fail_setup=False, fail_preload=False):
    events = []
    distributed = S(get_world_size=lambda: 2,
        all_gather_object=lambda target, value: target.__setitem__(slice(None), [copy.deepcopy(value), copy.deepcopy(value)]))
    fake_torch = S(distributed=distributed, cuda=S(current_device=lambda: 0, empty_cache=lambda: None))
    def preload(**kw):
        events.append('load-actor')
        if fail_preload:raise ValueError('constructed CPU preload failure')
    namespace = dict(torch=fake_torch, os=os,
        load_fsdp_param_and_grad=preload,
        offload_fsdp_param_and_grad=lambda **kw: events.append('offload-actor'))
    worker_path = 'verl/workers/fsdp_workers.py'
    cls = next(n for n in ast.parse((ROOT/worker_path).read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
    names = ('_assert_generation_phase_inactive', '_generation_boundary_agreement', '_generation_setup_agreement',
             'begin_generation_phase', 'end_generation_phase')
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in methods:node.decorator_list = []
    exec(compile(ast.Module(body=methods, type_ignores=[]), worker_path, 'exec'), namespace)
    worker_class = type('ActualPhaseWorker', (), {name: namespace[name] for name in names})
    manager_path = 'verl/workers/sharding_manager/fsdp_vllm.py'
    cls = next(n for n in ast.parse((ROOT/manager_path).read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'FSDPVLLMShardingManager')
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('begin_phase', 'end_phase')]
    manager_namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), manager_path, 'exec'), manager_namespace)
    manager_class = type('ActualPhaseManager', (), dict(begin_phase=manager_namespace['begin_phase'],
        end_phase=manager_namespace['end_phase'], __enter__=lambda self: events.append('sync-once'),
        __exit__=lambda self, *args: events.append('close-rng-offload')))
    manager = manager_class();manager._phase_snapshot=None;manager._phase_setup_snapshot=None;manager.phase_weight_sync_count=0
    identity = dict(policy_snapshot_id='snapshot', run_id='run', source_sha256={'fixture.py': 'a'*64}, slurm_job_id=None)
    from reproduction.common8.sampling_capture import digest
    rollout = S(_generation_phase=None, _generation_phase_setup_handle=None,
        inference_engine=S(llm_engine=S(_common8_generation_provenance=dict(
        run_id='run', slurm_job_id=None, source_map_sha256=digest(identity['source_sha256'])))))
    def begin(snapshot):
        events.append('cache-init')
        if fail_setup:raise ValueError('constructed CPU setup failure')
        rollout._generation_phase = dict(service=S(pending={}))
    def end(handle, abort=False):
        rollout._generation_phase=None
        return dict(completed_requests=0, engine_steps=0)
    rollout.begin_generation_phase=begin;rollout.end_generation_phase=end
    worker=worker_class();worker._is_rollout=True;worker._is_offload_param=True;worker._is_offload_grad=False
    worker.actor_module_fsdp=object();worker.rollout=rollout;worker.rollout_sharding_manager=manager
    worker.config=S(rollout=S(name='vllm',tensor_model_parallel_size=1))
    snapshot=dict(phase_id='phase',policy_snapshot_id='snapshot',behavior_identity=identity)
    return worker,snapshot,events


class ContinuousSamplingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name).resolve()
        raw = json.dumps({'eos_token_id': [7, 9]}).encode()
        (model/'generation_config.json').write_bytes(raw)
        run = dict(profile='common8', run_id='constructed-cpu-run', initial_model=str(model),
            execution_backend='ssh_local', execution_id='0'*32, slurm_job_id=None,
            source_commit='0'*40, source_sha256={'cpu-fixture.py': 'a'*64},
            initial_model_sha256={'generation_config.json': hashlib.sha256(raw).hexdigest()})
        document, source = generation_source(S(model=str(model)), loader, run=run)
        self.engine = Engine(document, loaded_generation(source, document, document))
        capture = RequestCapture(self.engine)
        self.engine._common8_stop_capture = capture
        self.service = ContinuousRequestService(self.engine, capture, read_behavior)

    def submit(self, request_id, row_id, tokens):
        caller = params()
        self.service.submit(request_id, tokens, caller, sampling_metadata(caller, True),
            dict(rollout_call_id='call-'+request_id, row_id=row_id))

    def test_wait_any_new_enqueue_retains_real_request_row_and_stop_binding(self):
        self.submit('slow', 'row-a', [11, 12])
        self.submit('fast', 'row-b', [31])
        first = self.service.advance(8)
        self.assertEqual([r['request_id'] for r in first], ['fast'])
        self.assertEqual(self.service.engine_steps, 1)
        self.submit('later', 'row-c', [41, 42, 43])
        second = self.service.advance(8)
        third = self.service.advance(8)
        results = first+second+third
        self.assertEqual([r['request_id'] for r in results], ['fast', 'later', 'slow'])
        for result, row_id, tokens in zip(results, ['row-b', 'row-c', 'row-a'], [[31], [41, 42, 43], [11, 12]]):
            behavior = result['behavior']
            self.assertEqual(behavior['actual_request_token_ids'], tokens)
            self.assertEqual(behavior['effective_stop_provenance']['row_id'], row_id)
            self.assertEqual(behavior['effective_stop_provenance']['rollout_call_id'], 'call-'+result['request_id'])
            self.assertEqual(behavior['raw_generated_token_ids'], [5, 7])
            self.assertEqual(behavior['selected_logprobs'], [-0.2, -0.3])
        self.assertEqual(self.service.close(), dict(completed_requests=3, engine_steps=3))

    def test_request_id_reuse_and_close_with_pending_refuse_then_abort_owns_only_pending(self):
        self.submit('slow', 'row-a', [11])
        with self.assertRaises(ValueError): self.service.close()
        with self.assertRaises(ValueError): self.submit('slow', 'row-b', [12])
        self.assertEqual(self.service.close(abort=True), dict(completed_requests=0, engine_steps=0))
        self.assertEqual(self.engine.aborted, ['slow'])
        self.assertFalse(self.service.pending)
        self.assertFalse(self.engine._common8_stop_capture.requests)

    def test_worker_advance_executes_service_without_torch_or_batch_generator(self):
        relative = 'verl/workers/fsdp_workers.py'
        cls = next(n for n in ast.parse((ROOT/relative).read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'advance_generation_phase'))
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), relative, 'exec'), namespace)
        handle = dict(phase_id='constructed-phase', policy_snapshot_id='constructed-snapshot')
        calls = []
        worker = S(_resident_generation_phase=handle,
            rollout=S(advance_generation_phase=lambda h, r, max_steps: calls.append((h, r, max_steps)) or 'raw-result'))
        self.assertEqual(namespace['advance_generation_phase'](worker, handle, [], max_steps=8), 'raw-result')
        self.assertEqual(calls, [(handle, [], 8)])
        with self.assertRaises(ValueError): namespace['advance_generation_phase'](worker, dict(handle, phase_id='wrong'), [])

    def test_actual_phase_sync_counter_once_and_actor_offload_precedes_cache(self):
        worker,snapshot,events=phase_worker_fixture()
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE':'common8'}):
            handle=worker.begin_generation_phase(snapshot)
            result=worker.end_generation_phase(handle)
        self.assertEqual(result['weight_syncs'],1)
        self.assertTrue(result['actor_params_offloaded_during_phase'])
        self.assertEqual(events,['load-actor','sync-once','offload-actor','cache-init','close-rng-offload'])

    def test_partial_begin_abort_is_scoped_and_does_not_invent_weight_sync(self):
        for fail_setup in (False,True):
            worker,snapshot,events=phase_worker_fixture(fail_setup)
            if not fail_setup:snapshot['behavior_identity']['run_id']='wrong-run'
            with patch.dict(os.environ, {'SEARCH_G1_PROFILE':'common8'}):
                with self.assertRaises(RuntimeError):worker.begin_generation_phase(snapshot)
                if fail_setup:
                    with self.assertRaises(RuntimeError):worker._assert_generation_phase_inactive()
                handle={key:snapshot[key] for key in ('phase_id','policy_snapshot_id')}
                result=worker.end_generation_phase(handle,abort=True)
                with self.assertRaises(RuntimeError):worker.end_generation_phase(dict(handle,phase_id='foreign'),abort=True)
            self.assertEqual(result['weight_syncs'],int(fail_setup))
            self.assertEqual(events.count('sync-once'),int(fail_setup))
            self.assertEqual(events.count('offload-actor'),int(fail_setup))

    def test_preload_failure_agrees_before_sync_and_uniform_abort_releases_partial_actor(self):
        worker,snapshot,events=phase_worker_fixture(fail_preload=True)
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE':'common8'}):
            with self.assertRaisesRegex(RuntimeError,'actor_preload'):worker.begin_generation_phase(snapshot)
            with self.assertRaises(RuntimeError):worker._assert_generation_phase_inactive()
            handle={key:snapshot[key] for key in ('phase_id','policy_snapshot_id')}
            result=worker.end_generation_phase(handle,abort=True)
            worker._assert_generation_phase_inactive()
        self.assertEqual(result['weight_syncs'],0)
        self.assertEqual(events,['load-actor','offload-actor'])

    def test_partial_cache_init_is_freed_by_exact_abort_handle(self):
        relative='verl/workers/rollout/vllm_rollout/vllm_rollout.py'
        cls=next(n for n in ast.parse((ROOT/relative).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='vLLMRollout')
        method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='end_generation_phase'))
        namespace=dict(copy=copy)
        exec(compile(ast.Module(body=[method],type_ignores=[]),relative,'exec'),namespace)
        handle=dict(phase_id='phase',policy_snapshot_id='snapshot');events=[]
        rollout=S(_generation_phase=None,_generation_phase_setup_handle=handle,_generation_cache_setup_started=True,
            generation_scheduler_limits=dict(max_num_seqs=64,max_num_batched_tokens=8192),
            inference_engine=S(llm_engine=S(_common8_stop_capture=None),free_cache_engine=lambda:events.append('free-cache')))
        with self.assertRaises(ValueError):namespace['end_generation_phase'](rollout,dict(handle,phase_id='wrong'),abort=True)
        result=namespace['end_generation_phase'](rollout,handle,abort=True)
        self.assertTrue(result['partial_setup_aborted']);self.assertEqual(events,['free-cache'])
        self.assertFalse(rollout._generation_cache_setup_started)


if __name__ == '__main__':
    unittest.main()
