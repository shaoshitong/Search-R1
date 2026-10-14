"""Tiny CPU state-machine tests; fake native records, no model/GPU execution."""
import copy
import threading
import re
import unittest
from unittest.mock import patch

from verl import DataProto
from reproduction.common8.generation_behavior import _left_pad
from reproduction.common8.test_generation_behavior import (
    ManagerFixture, TokenizerFixture, WorkerFixture, generation_fixture,
)
from reproduction.common8.test_sampling_capture import stop_provenance_fixture
from reproduction.common8.behavior_trace import BehaviorTraceError
from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, run


class EnvironmentTokenizer(TokenizerFixture):
    fragments = dict(TokenizerFixture.fragments, **{})
    fragments.update({40: '<search>slow', 41: '<search>fast'})

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert text in ('evidence', 'correction'), 'Only environment text may be encoded'
        self.encode_calls += 1
        return [30 if text == 'evidence' else 31]


class FakePhaseService:
    def __init__(self, worker=None, failure=None):
        self.worker = worker or WorkerFixture()
        self.failure = failure
        self.requests, self.queued = [], []
        self.begin_count = 0
        self.end_calls = []

    def begin(self, snapshot):
        self.begin_count += 1
        self.snapshot = snapshot

    def submit(self, request, rank):
        self.requests.append(copy.deepcopy(request))
        self.queued.append((request, rank))

    def poll(self, timeout=0.01):
        # Complete in the opposite order to submission, deliberately.
        if not self.queued:
            return []
        request, _ = self.queued.pop()
        call = DataProto.from_dict(_left_pad([request['prompt_token_ids']], 0),
            non_tensors={'behavior_row_id': [request['row_id']]}, meta_info=request['meta_info'])
        output = self.worker.generate_sequences(call)
        record = output.non_tensor_batch['sampling_behavior'][0]
        record['request_id'] = request['request_id']
        record['effective_stop_provenance'] = stop_provenance_fixture(record,
            run_id='f' * 32, slurm_job_id='12345', source_sha256={'cpu_fixture.py': 'a' * 64})
        if self.failure == 'request':
            record['request_id'] = 'wrong-native-request'
        if self.failure == 'lp':
            output.batch['sampling_log_probs'][0, 0] = -999
        return [dict(request_id=request['request_id'], output=output)]

    def end(self, abort=False):
        self.end_calls.append(abort)
        return [dict(weight_sync_count=1)]


def tiny_fixture(count=2, *, training=False, prompt=None):
    batch, initial = generation_fixture([prompt or [10, 11]] * count, validation=not training)
    repeats = 5 if training else 1
    batch.meta_info['behavior_identity']['parent_group_ids'] = ['group-' + str(index) for index in range(count // repeats)]
    return batch, initial


class AsyncSchedulerTests(unittest.TestCase):
    def manager(self):
        manager = ManagerFixture()
        manager.config.num_gpus = 2
        manager.tokenizer = EnvironmentTokenizer()
        def parse(texts):
            matches = [re.search(r'<(search|answer)>(.*?)</\1>', text, re.DOTALL) for text in texts]
            return ([match.group(1) if match else None for match in matches],
                    [match.group(2).strip() if match else '' for match in matches])
        manager.postprocess_predictions = parse
        return manager

    def call(self, manager, batch, initial, service, tool):
        with patch('reproduction.common8.behavior_runtime.failed_generation_evidence') as failure:
            result = run(manager, batch, initial, service=service, tool_runner=tool,
                config=AsyncRolloutConfig(agent_workers=10, max_inflight_trajectories=10,
                    retrieval_concurrency=2, tool_timeout=0.1, tool_retries=1))
            failure.assert_not_called()
        return result

    def test_fast_trajectory_resumes_before_other_tool_and_output_keeps_original_order(self):
        release = threading.Event()
        events = []
        def script(ordinal, row):
            return ([40 if row.endswith('0') else 41, 21, 99] if ordinal == 0 else [23, 24, 99])
        def tool(text, do_search):
            if '<search>slow' in text:
                events.append('slow_wait')
                self.assertTrue(release.wait(timeout=2), 'Fast trajectory was blocked behind slow retrieval')
                return 'evidence', False, 1, 1
            if '<search>fast' in text:
                return 'evidence', False, 1, 1
            events.append('answer'); release.set()
            return '', True, 1, 0
        manager = self.manager(); batch, initial = tiny_fixture()
        service = FakePhaseService(WorkerFixture(script))
        result = self.call(manager, batch, initial, service, tool)
        self.assertEqual(service.begin_count, 1)
        self.assertEqual(service.end_calls, [False])
        self.assertEqual(result.non_tensor_batch['behavior_trajectory_id'].tolist(), batch.non_tensor_batch['behavior_row_id'].tolist())
        self.assertEqual(result.meta_info['actual_retrieval_stats'], [1, 1])
        self.assertEqual(result.batch['responses'].tolist(), [[40, 21, 30, 23, 24], [41, 21, 30, 23, 24]])
        self.assertEqual(result.batch['sampling_mask'].tolist(), [[True, True, False, True, True]] * 2)
        self.assertEqual(result.batch['sampling_log_probs'].tolist(), [[-.25, -.375, 0., -.25, -.375]] * 2)
        self.assertEqual(result.meta_info['async_rollout']['completed_trajectories'], 2)
        self.assertEqual(len({request['request_id'] for request in service.requests}), 4)
        for request in service.requests:
            self.assertEqual(request['policy_snapshot_id'], 'cpu-policy0')
            if request['rollout_ordinal'] == 1:
                self.assertIn(30, request['prompt_token_ids'])

    def test_five_response_groups_survive_out_of_order_engine_completion(self):
        manager = self.manager(); batch, initial = tiny_fixture(count=10, training=True)
        service = FakePhaseService(WorkerFixture(lambda ordinal, row: [23, 24, 99]))
        result = self.call(manager, batch, initial, service, lambda text, do_search: ('', True, 1, 0))
        self.assertEqual(result.non_tensor_batch['behavior_trajectory_id'].tolist(), batch.non_tensor_batch['behavior_row_id'].tolist())
        self.assertEqual([(request['question_id'], request['response_index']) for request in service.requests],
                         [(f'group-{index // 5}', index % 5) for index in range(10)])
        self.assertEqual(len(result), 10)

    def test_exhausted_tool_rejects_outer_and_closes_phase_with_private_raw_evidence(self):
        manager = self.manager(); batch, initial = tiny_fixture()
        service = FakePhaseService()
        def broken_tool(text, do_search):
            raise BehaviorTraceError('async_retrieval_attempts_exhausted:Timeout')
        with patch('reproduction.common8.behavior_runtime.failed_generation_evidence') as failure:
            with self.assertRaisesRegex(BehaviorTraceError, 'attempts_exhausted'):
                run(manager, batch, initial, service=service, tool_runner=broken_tool)
            self.assertTrue(any('raw_generated_token_ids' in row for row in failure.call_args.args[1]))
        self.assertEqual(service.end_calls, [True])

    def test_wrong_actual_request_or_probability_rejects_before_environment(self):
        for reason, expected in (('request', 'request_row_identity'), ('lp', 'selected_probability_transport')):
            manager = self.manager(); batch, initial = tiny_fixture()
            service = FakePhaseService(failure=reason)
            with patch('reproduction.common8.behavior_runtime.failed_generation_evidence'), patch('search_g1.async_rollout_runtime.execute_one_action') as tool:
                with self.assertRaisesRegex(BehaviorTraceError, expected):
                    run(manager, batch, initial, service=service)
                tool.assert_not_called()
            self.assertEqual(service.end_calls, [True])

    def test_retriever_retry_count_and_timeout_are_finite(self):
        import requests
        manager = self.manager()
        manager.config.search_url = 'http://fixture/retrieve'
        manager.config.topk = 3
        options = AsyncRolloutConfig(tool_timeout=0.1, tool_retries=2)
        with patch('requests.Session') as session:
            client = session.return_value.__enter__.return_value
            client.post.side_effect = requests.Timeout('fixture')
            with self.assertRaisesRegex(BehaviorTraceError, 'attempts_exhausted:Timeout'):
                BoundedRetriever(manager, options).search('query')
            self.assertEqual(client.post.call_count, 3)
            self.assertEqual(client.post.call_args.kwargs['timeout'], (0.1, 0.1))

    def test_answer_reserve_forces_only_its_own_final_turn_without_tool(self):
        manager = self.manager(); manager.config.max_start_length = 3500
        batch, initial = tiny_fixture(count=1, prompt=[10] * 3200)
        service = FakePhaseService(WorkerFixture(lambda ordinal, row: [20, 21, 99]))
        searches = []
        result = self.call(manager, batch, initial, service,
            lambda text, do_search: (searches.append(do_search) or ('', False, 1, 1)))
        self.assertEqual(searches, [False])
        self.assertEqual(result.meta_info['actual_retrieval_stats'], [0])
        self.assertEqual(result.meta_info['early_final_turn_stats'], [True])


if __name__ == '__main__':
    unittest.main()
