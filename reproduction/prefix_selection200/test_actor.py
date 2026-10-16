"""Small CPU checks of authentic PPO inputs, full budgets and retry identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .actor import (ActorBudget, SAMPLING_CONTRACT, _state_binding, collect_oom_retry, group_advantages,
                    learning_rate, masked_policy_objective, oom_retry_plan, rank_plan,
                    micro_loss_weight, restore_metadata, tensor_batch, validate_record)


def fixture():
    sampling = dict(temperature=1., top_p=1., top_k=-1, max_tokens=20, n=1)
    first = dict(request_id='prefix', actual_request_token_ids=[11, 12],
                 raw_generated_token_ids=[21, 22], selected_logprobs=[-.1, -.2], sampling=sampling)
    second = dict(request_id='continuation', actual_request_token_ids=[11, 12, 21, 22],
                  raw_generated_token_ids=[23, 24], selected_logprobs=[-.3, -.4],
                  sampling=dict(sampling, max_tokens=498))
    turn0 = dict(turn_index=0, actual_request_token_ids=[11, 12], actual_sampling_chunks=[first, second],
                 raw_generated_token_ids=[21, 22, 23, 24], selected_logprobs=[-.1, -.2, -.3, -.4],
                 kept_token_count=3, prefix_cut=dict(kept_token_ids=[21, 22, 23]),
                 observation_token_ids=[99, 98], correction_token_ids=[])
    last = dict(request_id='next', actual_request_token_ids=[11, 12, 21, 22, 23, 99, 98],
                raw_generated_token_ids=[31, 32], selected_logprobs=[-.5, -.6], sampling=dict(sampling, max_tokens=500))
    turn1 = dict(turn_index=1, actual_request_token_ids=last['actual_request_token_ids'],
                 actual_sampling_chunks=[last], raw_generated_token_ids=[31, 32], selected_logprobs=[-.5, -.6],
                 kept_token_count=2, prefix_cut=dict(kept_token_ids=[31, 32]),
                 observation_token_ids=[], correction_token_ids=[])
    return dict(actor_version='v0', policy_snapshot_id='v0', sampling_contract=SAMPLING_CONTRACT,
        trajectory_id='t0', qid='q0', candidate_ordinal=3, response_index=0,
        initial_prompt_token_ids=[11, 12], response_token_ids=[21, 22, 23, 99, 98, 31, 32],
        response_info_mask=[1, 1, 1, 0, 0, 1, 1], response_log_probs=[-.1, -.2, -.3, 0., 0., -.5, -.6],
        strict_em=1, behavior_turns=[turn0, turn1], sampling_journal=[first, second, last],
        reward_model=dict(ground_truth=dict(target=['gold'])), data_source='nq')


class ActorTests(unittest.TestCase):
    def test_real_first_eos_is_retained_g5_outcome_not_a_fake_policy_token(self):
        import torch
        row = fixture(); row.update(response_token_ids=[], response_info_mask=[],
            response_log_probs=[], strict_em=0, terminal_reason='natural_eos')
        chunk = dict(request_id='eos', actual_request_token_ids=[11, 12], raw_generated_token_ids=[0],
            selected_logprobs=[-.01], sampling=dict(temperature=1., top_p=1., top_k=-1, max_tokens=20, n=1))
        row['behavior_turns'] = [dict(turn_index=0, actual_request_token_ids=[11, 12],
            actual_sampling_chunks=[chunk], raw_generated_token_ids=[0], selected_logprobs=[-.01],
            kept_token_count=0, prefix_cut=dict(kept_token_ids=[]),
            observation_token_ids=[], correction_token_ids=[])]
        row['sampling_journal'] = [chunk]
        validate_record(row, 'v0')
        batch = tensor_batch([row], 0)
        self.assertEqual(batch['responses'].shape, (1, 1))
        self.assertEqual(int(batch['loss_mask'].sum()), 0)
        self.assertEqual(int(batch['attention_mask'].sum()), 2)
        current = torch.tensor([[7.]], requires_grad=True)
        def forbidden(*args, **kwargs): raise AssertionError('Zero-mass loss must not divide by zero')
        loss, clip, kl = masked_policy_objective(current.detach(), current, current.detach(),
                                                batch['loss_mask'], loss_fn=forbidden)
        loss.backward()
        self.assertEqual(current.grad.tolist(), [[0.]])
        self.assertEqual((float(loss.detach()), float(clip), float(kl)), (0., 0., 0.))

    def test_micro_fallback_preserves_unequal_length_token_mean_and_gradient(self):
        import torch
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]])
        values = torch.tensor([[1., 2., 3.], [4., 99., 99.], [99., 99., 99.], [5., 6., 99.]], requires_grad=True)
        full = (values * mask).sum() / mask.sum()
        full.backward(); expected_gradient = values.grad.clone(); values.grad = None
        micro_losses = []
        for start in (0, 2):
            micro_mask = mask[start:start+2]; micro_value = values[start:start+2]
            mass = int(micro_mask.sum())
            micro_losses.append((micro_value * micro_mask).sum() / mass * micro_loss_weight(mass, int(mask.sum())))
        reduced = sum(micro_losses); reduced.backward()
        self.assertEqual(float(full.detach()), 3.5)
        self.assertTrue(torch.equal(expected_gradient, values.grad))
        self.assertEqual(float(reduced.detach()), float(full.detach()))
        self.assertEqual(micro_loss_weight(0, 0), 0.)

    def test_native_prefix_context_and_probability_identity(self):
        row = fixture(); self.assertIs(validate_record(row, 'v0'), row)
        changed = copy.deepcopy(row)
        changed['behavior_turns'][0]['actual_sampling_chunks'][1]['actual_request_token_ids'][2] = 100
        with self.assertRaisesRegex(ValueError, 'same actual prefix'):
            validate_record(changed, 'v0')
        changed = copy.deepcopy(row); changed['response_log_probs'][0] = -.11
        with self.assertRaisesRegex(ValueError, 'oldLP'):
            validate_record(changed, 'v0')

    def test_policy_prefix_and_observations_masked_exactly(self):
        row = fixture(); shorter = copy.deepcopy(row); shorter['initial_prompt_token_ids'] = [12]
        batch = tensor_batch([row, shorter], 0)
        self.assertEqual(batch['prompts'].tolist(), [[11, 12], [0, 12]])
        self.assertEqual(batch['loss_mask'][0].tolist(), row['response_info_mask'])
        self.assertEqual(batch['old_log_probs'][0, 3:5].tolist(), [0., 0.])
        self.assertEqual(batch['position_ids'][1].tolist(), [0, 0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(batch['sampling_mask'][0, :3].all())

    def test_natural_plan_covers_every_trajectory_once(self):
        plan = rank_plan(1729, 1)
        ids = [i for rank in plan for batch in rank for i in batch]
        self.assertEqual(sorted(ids), list(range(2560)))
        self.assertEqual({len(rank) for rank in plan}, {20})
        self.assertEqual({len(batch) for rank in plan for batch in rank}, {16})
        self.assertEqual(ActorBudget().local_micro, 16)
        self.assertEqual(ActorBudget().accumulation, 1)
        self.assertEqual(ActorBudget(16).accumulation, 8)
        with self.assertRaises(ValueError): ActorBudget(24)

    def test_oom_does_not_stitch_abandoned_partial_writes(self):
        retry = oom_retry_plan(outer=4, failed_micro=128, latest_complete_outer=3,
            rollout_actor_version='v3', latest_actor_version='v3', partial_writes=[7] * 8)
        self.assertEqual(retry['actor_global_micro_batch_size'], 64)
        self.assertEqual(retry['gradient_accumulation'], 2)
        self.assertFalse(retry['stitch_partial_writes'])
        self.assertEqual(retry['restore_completed_outer'], 3)
        with self.assertRaises(ValueError):
            oom_retry_plan(outer=4, failed_micro=128, latest_complete_outer=2,
                rollout_actor_version='v3', latest_actor_version='v2', partial_writes=[7] * 8)

    def test_g5_grpo_uses_own_reward_and_original_sample_std(self):
        rows = [dict(qid='one')] * 5 + [dict(qid='two')] * 5
        result = group_advantages(rows, [0., 0., 0., 0., 1.] + [3.] * 5)
        self.assertAlmostEqual(float(result[:5].mean()), 0., places=6)
        self.assertAlmostEqual(float(result[:5].std()), 1., places=5)
        self.assertEqual(result[5:].tolist(), [0.] * 5)
        self.assertEqual(learning_rate(0), 0.)
        self.assertEqual(learning_rate(57), 1e-6)
        self.assertEqual(learning_rate(200), 1e-6)

    def test_resume_requires_latest_exact_boundary(self):
        config = dict(run_id='new', official_model_dir='/mnt/model', grounding_probe_version='fixed0')
        receipt = dict(run_id='new', completed_outer=2, binding=_state_binding(config),
                       actual_adam_steps_per_rank=[40] * 8, scheduler_completed_outer=2)
        restore_metadata(config, receipt, 3)
        with self.assertRaisesRegex(ValueError, 'immediately previous'):
            restore_metadata(config, receipt, 4)
        changed = copy.deepcopy(receipt); changed['actual_adam_steps_per_rank'][7] = 39
        with self.assertRaises(ValueError): restore_metadata(config, changed, 3)

    def test_parent_oom_collector_ignores_old_attempt_and_preserves_observed_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); attempt = 'a' * 32
            config = dict(run_id='run', actor_attempt_id=attempt, official_model_dir='/mnt/model',
                          grounding_probe_version='fixed0', actor_global_micro_batch_size=128)
            (root / 'config.json').write_text(json.dumps(config))
            rollout = root / 'rollout.json'
            rollout.write_text(json.dumps(dict(run_id='run', outer=1, actor_version='run-official0')))
            output = root / 'actor'; directory = output / 'outers' / ('outer-0001-attempt-' + attempt)
            directory.mkdir(parents=True)
            for rank in range(8):
                rows = [dict(event='ACTUAL_RESTORE', rank=rank, outer=1)]
                rows += [dict(event='ACTUAL_ADAM_WRITE', rank=rank, outer=1, opportunity=i,
                    writes_this_attempt=i + 1, actual_adam_steps=i + 1) for i in range(3)]
                (directory / ('rank-%d-updates.jsonl' % rank)).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (directory / 'rank-4-failure.json').write_text(json.dumps(dict(rank=4, outer=1,
                run_id='run', attempt_id=attempt, cached_rollout_path=str(rollout), actual_cuda_oom=True)))
            # An unrelated stale non-OOM failure is neither selected nor overwritten.
            (output / 'native-failure-outer-0001-attempt-old-rank-0.json').write_text('{}')
            result = collect_oom_retry(output, root / 'config.json', 1, rollout, attempt, phase_exit_code=1)
            self.assertTrue(result['retry'])
            self.assertEqual(result['observed_committed_partial_writes'], [3] * 8)
            self.assertEqual(result['actor_global_micro_batch_size'], 64)
            self.assertEqual(len(result['failures']), 1)
            self.assertTrue((output / 'native-failure-outer-0001-attempt-old-rank-0.json').exists())
            self.assertFalse((output / 'latest.json').exists())
            (directory / 'rank-4-failure.json').unlink()
            result = collect_oom_retry(output, root / 'config.json', 1, rollout, attempt, phase_exit_code=1)
            self.assertEqual(result['status'], 'NON_OOM_FAILURE')
            self.assertFalse(result['retry'])


if __name__ == '__main__': unittest.main()
