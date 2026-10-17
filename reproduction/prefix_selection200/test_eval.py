"""Small CPU checks of the shared new QA contract; no model/evaluation run."""
from pathlib import Path
import gzip
import json
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from . import eval as evaluation
from .interaction_protocol import CONTRACT, PROTOCOL_ID
from .test_interaction_protocol import TOKENIZER


class SharedEvaluationTests(unittest.TestCase):
    def test_qa_contract_freezes_new_bounds_and_only_greedy_sampling_differs(self):
        self.assertEqual(evaluation.INTERACTION_CONTRACT, CONTRACT)
        self.assertEqual(evaluation.CONTRACT['protocol_id'], PROTOCOL_ID)
        self.assertEqual(evaluation.CONTRACT['compute_dtype'], 'bfloat16')
        self.assertEqual(evaluation.CONTRACT['max_model_tokens'], 4096)
        self.assertEqual(evaluation.CONTRACT['max_request_tokens'], 3596)
        self.assertEqual(evaluation.CONTRACT['answer_reserve_tokens'], 500)
        self.assertTrue(evaluation.CONTRACT['early_final'])
        self.assertEqual(evaluation.CONTRACT['temperature'], 0.)
        self.assertEqual(evaluation.CONTRACT, dict(CONTRACT, temperature=0., top_p=1., top_k=-1, n=1))

    def test_constructor_oom_lowers_only_physical_capacity_and_retains_bf16_protocol(self):
        import torch
        calls, cleanups = [], []
        sentinel = object()
        def factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1: raise torch.cuda.OutOfMemoryError('actual mocked constructor CUDA OOM')
            return sentinel
        llm, util, limits, failed = evaluation.build_evaluation_engine({}, '/mnt/model', 1, 79 * 2**30, 80 * 2**30,
            factory=factory, cleanup=lambda: cleanups.append(True))
        self.assertIs(llm, sentinel); self.assertEqual(cleanups, [True])
        self.assertEqual(calls[0]['gpu_memory_utilization'], .95)
        self.assertEqual(calls[0]['max_num_seqs'], 1024)
        self.assertEqual(calls[1]['max_num_seqs'], 512)
        self.assertEqual(util, .95 * .85)
        self.assertEqual(limits['max_num_batched_tokens'], 16384)
        self.assertTrue(failed[0]['actual_cuda_oom'])
        self.assertTrue(all(call['dtype'] == 'bfloat16' and call['max_model_len'] == 4096 for call in calls))
        self.assertTrue(all(call['enable_chunked_prefill'] and call['enable_prefix_caching'] and
                            call['disable_async_output_proc'] for call in calls))
        def unknown(**kwargs): raise RuntimeError('unknown CUDA collective failure')
        with self.assertRaisesRegex(RuntimeError, 'unknown CUDA'):
            evaluation.build_evaluation_engine({}, '/mnt/model', 1, 79 * 2**30, 80 * 2**30,
                factory=unknown, cleanup=lambda: self.fail('Unknown errors must not retry as OOM'))

    def test_canned_native_stream_aborts_before_finished_and_keeps_closed_observation(self):
        # Exercise the actual QA event loop with two canned native streams.
        # No model, dataset, GPU, retrieval service or inference is invoked.
        import torch
        class Params(SimpleNamespace):
            @property
            def all_stop_token_ids(self): return set(self.stop_token_ids)
        class Engine:
            def __init__(self):
                self.model_config = SimpleNamespace(dtype=torch.bfloat16)
                self.scheduler = [SimpleNamespace(waiting=[], running=[], swapped=[])]
                self.current = None; self.added = []; self.aborted = []; self.steps = 0
            def add_request(self, rid, prompt, params):
                self.assert_idle = self.current is None
                self.current = dict(rid=rid, ids=prompt['prompt_token_ids'], params=params, step=0,
                                    plan=[1, 2, 3] if not self.added else [6, 7, 8])
                self.added.append(self.current)
                self.scheduler[0].waiting.append(SimpleNamespace(request_id=rid,
                    prompt_token_ids=prompt['prompt_token_ids'], sampling_params=params))
            def has_unfinished_requests(self): return self.current is not None
            def step(self):
                current = self.current; current['step'] += 1; self.steps += 1
                if current['step'] > 2:
                    raise AssertionError('QA waited past the first complete action for native finished')
                ids = current['plan'][:1] if current['step'] == 1 else current['plan']
                native = SimpleNamespace(token_ids=ids,
                    logprobs=[{token: SimpleNamespace(logprob=-.125)} for token in ids],
                    finish_reason=None, stop_reason=None)
                return [SimpleNamespace(request_id=current['rid'], prompt_token_ids=current['ids'],
                                        outputs=[native], finished=False)]
            def abort_request(self, rid):
                self.aborted.append(rid); self.current = None; self.scheduler[0].waiting.clear()
        engine = Engine()
        def environment(manager, text, do_search, retriever):
            if text == '<search>x</search>':
                self.assertTrue(do_search)
                return '<information>' + 'evidence ' * 100 + '</information>', False, 1, 1
            self.assertEqual(text, '<answer>ok</answer>')
            return '', True, 1, 0
        fake_modules = {
            'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: TOKENIZER)),
            'vllm': SimpleNamespace(SamplingParams=Params),
            'search_g1.async_rollout_runtime': SimpleNamespace(AsyncRolloutConfig=SimpleNamespace,
                BoundedRetriever=lambda manager, config: SimpleNamespace(config=config), execute_one_action=environment),
            'search_r1.llm_agent.generation': SimpleNamespace(GenerationConfig=SimpleNamespace,
                LLMGenerationManager=lambda *a, **k: object()),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model = root / 'model'; model.mkdir()
            (model / 'generation_config.json').write_text(json.dumps(dict(eos_token_id=0)))
            config = root / 'config.json'; config.write_text(json.dumps(dict(run_id='run', actor_version='actor', search_url='unused')))
            questions = root / 'questions.json'; questions.write_text('[]')
            row = dict(qid='q', data_source='nq', evaluation_ordinal=0, prompt=[],
                       reward_model=dict(ground_truth=dict(target=['ok'])))
            with patch.dict(sys.modules, fake_modules), \
                 patch.object(evaluation, 'remote', side_effect=lambda value: Path(value)), \
                 patch('reproduction.prefix_selection200.rollout._configure_one_visible_gpu', return_value={'fake': True}), \
                 patch.object(torch.cuda, 'mem_get_info', return_value=(79 * 2**30, 80 * 2**30)), \
                 patch.object(evaluation, 'load_questions', return_value=[row]), \
                 patch.object(evaluation, 'build_evaluation_engine', return_value=(SimpleNamespace(llm_engine=engine), .95,
                    dict(max_num_seqs=4, max_num_batched_tokens=4096), [])), \
                 patch.object(evaluation, 'score_tokens', return_value=dict(em=1, f1=1., answer_present=True)):
                receipt = evaluation.worker(config, model, root / 'out', questions, 512, 0)
            with gzip.open(receipt['full_output']['path'], 'rt') as stream:
                completed = json.loads(stream.readline())
            self.assertEqual(engine.steps, 4)
            self.assertEqual(len(engine.aborted), 2)
            self.assertTrue(all(request['params'].max_tokens == 500 for request in engine.added))
            self.assertTrue(all(request['params'].temperature == 0. for request in engine.added))
            self.assertEqual(receipt['actual_compute_dtype'], 'torch.bfloat16')
            self.assertEqual(receipt['interaction_protocol'], CONTRACT)
            self.assertEqual(completed['actual_retrieval_calls'], 1)
            self.assertEqual(completed['behavior_turns'][0]['prefix_cut']['kept_token_ids'], [1, 2, 3])
            observation = completed['behavior_turns'][0]['observation_token_ids']
            self.assertEqual(len(observation), 500)
            self.assertTrue(TOKENIZER.decode(observation).endswith('</information>'))
            self.assertEqual(engine.added[1]['ids'], completed['initial_prompt_token_ids'] + [1, 2, 3] + observation)
            self.assertTrue(all(turn['actual_sampling_chunks'][0]['native_action_boundary_abort']
                                for turn in completed['behavior_turns']))

    def test_legacy_complete_receipt_cannot_satisfy_new_qa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'out').mkdir()
            config = dict(run_id='run', actor_version='actor')
            old = dict(config, qa_count=512, question_source_sha256='source',
                       evaluation_contract=dict(kind='legacy_pre_d00_async_fp32'))
            documents = {root / 'config': config, root / 'out' / 'complete.json': old}
            (root / 'out' / 'complete.json').write_text('{}')
            with patch.object(evaluation, 'remote', side_effect=lambda value: Path(value)), \
                 patch.object(evaluation, 'read', side_effect=lambda value: documents[Path(value)]), \
                 patch.object(evaluation, 'sha', return_value='source'), \
                 patch.object(evaluation.subprocess, 'Popen') as child:
                with self.assertRaisesRegex(ValueError, 'receipt differs'):
                    evaluation.launch(root / 'config', root / 'model', root / 'out', root / 'questions')
                child.assert_not_called()


if __name__ == '__main__': unittest.main()
