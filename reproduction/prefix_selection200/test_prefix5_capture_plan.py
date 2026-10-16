"""Focused CPU checks for four-rank prefix5 continuity and natural G5 plans."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .actor_hidden import ActorHiddenCapture
from .contract import DIM, G, PREFIX_TOKENS, QMAX, WIDTH, WORLD
from .periodic import complete_pool_plan, is_probe_update, probe_boundary
from .pool import fixed_test_identities
from .rollout import (
    _configure_one_visible_gpu, _states, append_chunk, binding,
    discard_hidden_lookahead, file_item, next_request, prefix_record,
    start_action, validate_rank_receipt,
)
from reproduction.prefix20_online.pool import canonical, digest


def trajectory_state():
    return dict(run_id='run', outer=1, actor_version='actor0', trajectory_id='t0',
        candidate_ordinal=0, response_index=0, group=dict(qid='q0', question_hash='hash0'),
        prompt=[65, 66], response=[], info_mask=[], log_probs=[], turns=[],
        ordinal=0, chunks_started=0, action=None, retrieval_calls=0,
        terminal=False, terminal_reason=None)


class Prefix5CapturePlanTest(unittest.TestCase):
    def test_complete_pool_visits_all_g5_once_with_four_equal_rank_totals(self):
        plan = complete_pool_plan(1729, 20)
        self.assertEqual((WORLD, PREFIX_TOKENS, QMAX, WIDTH, DIM, G), (4, 5, 273, 278, 2048, 5))
        self.assertEqual(len(plan), 100)
        self.assertEqual([row['global_questions'] for row in plan].count(51), 80)
        self.assertEqual([row['global_questions'] for row in plan].count(52), 20)
        assigned = [[] for _ in range(WORLD)]
        for row in plan:
            self.assertEqual(len(row['rank_questions']), WORLD)
            self.assertEqual(sum(row['rank_questions']), row['global_questions'])
            self.assertTrue(set(row['rank_questions']) <= {12, 13})
            offset = 0
            for rank, count in enumerate(row['rank_questions']):
                assigned[rank].extend(row['question_indices'][offset:offset+count])
                offset += count
        self.assertEqual([len(ids) for ids in assigned], [1280] * WORLD)
        self.assertEqual(sorted(index for ids in assigned for index in ids), list(range(5120)))
        self.assertNotEqual(plan, complete_pool_plan(1729, 40))
        self.assertFalse(is_probe_update(19))
        self.assertTrue(is_probe_update(20))
        self.assertEqual(probe_boundary(19), dict(last_update_outer=0, actual_adam_per_rank=0))
        self.assertEqual(probe_boundary(200), dict(last_update_outer=200, actual_adam_per_rank=1000))

    def test_sixth_native_token_and_lp_do_not_enter_prefix_or_continuation(self):
        state = trajectory_state()
        start_action(state, dict(total_tokens=4096, action_tokens=500, answer_reserve_tokens=500))
        request = next_request(state, 'prefix', {})
        self.assertEqual(request['max_tokens'], 5)
        retained = dict(raw_generated_token_ids=[101, 102, 103, 104, 105],
            selected_logprobs=[-.1] * 5, finish_reason=None, stop_reason=None)
        actual = dict(raw_generated_token_ids=retained['raw_generated_token_ids'] + [999],
            selected_logprobs=retained['selected_logprobs'] + [-.9],
            finish_reason='length', stop_reason=None)
        snapshot, evidence = discard_hidden_lookahead(retained, actual)
        self.assertEqual(evidence['discarded_sampled_lookahead_tokens'], 1)
        self.assertTrue(evidence['lookahead_token_not_passed_to_ppo_or_continuation'])
        cut = append_chunk(state, request, snapshot['raw_generated_token_ids'], snapshot['selected_logprobs'],
            dict(actor_hidden_flush=evidence), finish_reason=snapshot['finish_reason'], stop_reason=None,
            terminal_ids=[0], decode=lambda ids: ''.join(chr(token) for token in ids))
        self.assertIsNone(cut)
        self.assertEqual(prefix_record(state)['policy_token_ids'], retained['raw_generated_token_ids'])
        continued = next_request(copy.deepcopy(state), 'continue', {})
        self.assertEqual(continued['max_tokens'], 495)
        self.assertEqual(continued['actual_request_token_ids'], [65, 66, 101, 102, 103, 104, 105])
        self.assertNotIn(999, continued['actual_request_token_ids'])
        self.assertEqual(state['action']['selected_logprobs'], [-.1] * 5)
        wrong = dict(actual, raw_generated_token_ids=[777] + actual['raw_generated_token_ids'][1:])
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            discard_hidden_lookahead(retained, wrong)

    def test_native_layer_capture_has_question_and_five_slots_only(self):
        capture = ActorHiddenCapture.__new__(ActorHiddenCapture)
        capture.torch, capture.actor_version = torch, 'actor0'
        capture.requests, capture.trajectories = {}, {}
        capture.calls, capture.captured_rows = 0, 0
        state = trajectory_state()
        start_action(state, dict(total_tokens=4096, action_tokens=500, answer_reserve_tokens=500))
        request = next_request(state, 'prefix', {})
        capture.register(state, request)
        tokens = [65, 66, 101, 102, 103, 104, 105, 999]
        capture.query_rows = [dict(request_id=request['request_id'], start=0, end=len(tokens),
            tokens=tokens, positions=list(range(len(tokens))))]
        capture.native_hidden = torch.arange(len(tokens), dtype=torch.float32)[:, None].expand(-1, DIM)
        capture._consume_native()
        data = capture.finish(dict(trajectory_id='t0', actor_version='actor0', retained_policy_length=5,
            initial_prompt_token_ids=[65, 66], policy_token_ids=[101, 102, 103, 104, 105],
            policy_position_ids=[2, 3, 4, 5, 6]))
        self.assertEqual(tuple(data['features'].shape), (278, 2048))
        self.assertEqual(int(data['attention_mask'].sum()), 7)
        self.assertEqual(data['features'][QMAX:, 0].tolist(), [2., 3., 4., 5., 6.])
        self.assertTrue(bool((data['features'][2:QMAX] == 0).all()))
        self.assertEqual(capture.captured_rows, 7)

    def test_visible_uuid_binding_uses_exact_four_and_rejects_alias_duplicates(self):
        inventory = '\n'.join('%d, GPU-%d' % (j, j) for j in range(8))
        with patch.dict(os.environ, dict(CUDA_VISIBLE_DEVICES='GPU-0,GPU-2,GPU-3,GPU-4', RANK='7'), clear=True), \
             patch('subprocess.check_output', return_value=inventory):
            evidence = _configure_one_visible_gpu(2)
            self.assertEqual(evidence['physical_gpu_index'], 3)
            self.assertEqual(evidence['gpu_uuid'], 'GPU-3')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3')
            self.assertNotIn('RANK', os.environ)
        with patch.dict(os.environ, dict(CUDA_VISIBLE_DEVICES='0,GPU-0,3,4'), clear=True), \
             patch('subprocess.check_output', return_value=inventory):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                _configure_one_visible_gpu(0)

    def test_diagnostic_prefix_partitions_complete_g5_over_four_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'outers/outer-000001'
            root.mkdir(parents=True)
            rows = [dict(qid='q%d' % j, question_hash='h%d' % j, candidate_ordinal=j) for j in range(512)]
            (root / 'questions.json').write_text(json.dumps(dict(outer=1, rows=rows)))
            config = dict(base=temporary, rollout_candidate_questions=512)
            with patch('reproduction.prefix_selection200.rollout.initial_state',
                       side_effect=lambda config, outer, group, response, tokenizer:
                       (group['candidate_ordinal'], response)):
                ranks = [_states(config, 1, rank, 'prefix', None, None) for rank in range(WORLD)]
            self.assertEqual([len(rows) for rows in ranks], [640] * WORLD)
            self.assertEqual(len({pair for rows in ranks for pair in rows}), 2560)
            for rank, part in enumerate(ranks):
                self.assertTrue(all(question % WORLD == rank for question, _ in part))

    def test_continuation_receipt_requires640_trajectories_per_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(run_id='run', actor_version='actor0')
            conf = root / 'config.json'; conf.write_bytes(canonical(config))
            full = root / 'full.gz'; full.write_bytes(b'cpu-only producer fixture')
            own = root / 'own.jsonl'; own.write_bytes(b'cpu-only own-label fixture')
            saved = dict(run_id='run', outer=1, phase='continue', rank=3, actor_version='actor0',
                trajectories=640, config_sha256=digest(conf), full=file_item(full, 640), own_em=file_item(own, 640))
            receipt = root / 'complete.json'; receipt.write_bytes(canonical(saved))
            validated, reused = validate_rank_receipt(conf, config, 1, 'continue', 3, receipt)
            self.assertEqual(validated['trajectories'], 640)
            self.assertIsNone(reused)
            saved['trajectories'] = 320; receipt.write_bytes(canonical(saved))
            with self.assertRaisesRegex(ValueError, 'different actual'):
                validate_rank_receipt(conf, config, 1, 'continue', 3, receipt)

    def test_bound_recipe_rejects_old_world_and_prefix(self):
        config = dict(base='/not-an-actual-runtime', actor_model_dir='/official', actor_version='actor0')
        for invalid in (dict(world_size=8), dict(prefix_policy_tokens=20)):
            with self.assertRaisesRegex(ValueError, 'four-rank prefix5'):
                binding(dict(config, **invalid), 1)

    def test_explicit_fixed512_identity_sha_and_unique_normalized_questions(self):
        records = [dict(qid='q%d' % j, question_hash='%064x' % j,
                        normalized_question_hash='%064x' % (j+512)) for j in range(512)]
        config = dict(fixed_test_identity_path='/data/accepted/fixed512-identities.json',
                      fixed_test_identity_sha256='a' * 64)
        manifest = dict(schema_version=1, questions=512, trajectories=2560, records=records)
        with patch('reproduction.prefix_selection200.pool.digest', return_value='a' * 64), \
             patch('reproduction.prefix_selection200.pool.read', return_value=manifest):
            self.assertEqual(fixed_test_identities(config), records)
            bad = copy.deepcopy(manifest)
            bad['records'][1]['normalized_question_hash'] = bad['records'][0]['normalized_question_hash']
            with patch('reproduction.prefix_selection200.pool.read', return_value=bad):
                with self.assertRaisesRegex(ValueError, 'duplicated'):
                    fixed_test_identities(config)
        with patch('reproduction.prefix_selection200.pool.digest', return_value='b' * 64):
            with self.assertRaisesRegex(ValueError, 'SHA'):
                fixed_test_identities(config)


if __name__ == '__main__':
    unittest.main()
