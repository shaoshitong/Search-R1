import copy
import unittest

import torch

from reproduction.prefix20_group_joint100.loss import joint_loss
from reproduction.prefix20_group_joint100.train_epoch import ddp_mean_scale, rank_question_slice
from reproduction.prefix_selection200.probe import (
    DIM, LR, QMAX, SOURCE_RUN, TEST_SHA, WIDTH, candidate_groups,
    initial_online_state, join_own_em, mixed_probabilities, online_plan,
    prefix_row, stable_top512, validate_online_layout,
)


class ProbeContracts(unittest.TestCase):
    def test_complete_g5_average256_without_accumulation(self):
        plan = online_plan(1729, 1)
        self.assertEqual([x['global_questions'] for x in plan].count(51), 8)
        self.assertEqual([x['global_questions'] for x in plan].count(52), 2)
        all_ids = []
        for rank in range(8):
            ids = [j for step in plan for j in rank_question_slice(step, rank)]
            self.assertEqual(len(ids), 64)
            self.assertEqual(len(set(ids)), 64)
            all_ids.extend(ids)
        self.assertEqual(sorted(all_ids), list(range(512)))
        self.assertEqual(sum(x['global_questions']*5 for x in plan)/10, 256)
        self.assertNotEqual(plan, online_plan(1729, 2))

    def test_ddp_unequal_rank_means_equal_global_joint_gradient(self):
        z = torch.randn(51, 5, dtype=torch.float64, requires_grad=True)
        labels = (torch.arange(255).reshape(51, 5) % 3 == 0).double()
        expected = torch.autograd.grad(joint_loss(z, labels)['loss'], z)[0]
        counts = online_plan(1729, 1)[0]['rank_questions']
        offset, losses = 0, []
        for n in counts:
            losses.append(joint_loss(z[offset:offset+n], labels[offset:offset+n])['loss']
                          * ddp_mean_scale(n, 51) / 8)
            offset += n
        actual = torch.autograd.grad(sum(losses), z)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_mixed_formula_and_differentiable_joint_k1(self):
        p = torch.tensor([[.9, .9, .1, .1, .5], [.5]*5], dtype=torch.float64)
        z = torch.logit(p).requires_grad_()
        q = mixed_probabilities(z)
        torch.testing.assert_close(q, 1-p.prod(1)-(1-p).prod(1))
        loss = joint_loss(z, torch.tensor([[1., 0, 0, 0, 0], [0., 0, 0, 0, 0]], dtype=torch.float64))
        self.assertEqual(loss['mixed_label'].tolist(), [1., 0.])
        torch.testing.assert_close(q, loss['mixed_probability'])
        loss['loss'].backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue((z.grad[0] != 0).all())

    def test_identity_grouping_not_contiguous_fives_and_stable_ties(self):
        index = {}
        for response in range(5):
            for qid, ordinal in [('z', 9), ('a', 3), ('m', 8)]:
                tid = qid+str(response)
                index[tid] = dict(qid=qid, question_hash=qid, candidate_ordinal=ordinal, response_index=response)
        groups = candidate_groups(index, 3)
        self.assertEqual([x['qid'] for x in groups], ['a', 'm', 'z'])
        selected, _ = stable_top512(groups, {tid: 0. for tid in index}, k=2)
        self.assertEqual([x['qid'] for x in selected], ['a', 'm'])
        broken = dict(index)
        broken.pop('z4')
        with self.assertRaises(ValueError):
            candidate_groups(broken, 3)

    def test_own_outcome_join_rejects_replaced_prefix_or_actor(self):
        index = {'t'+str(j): dict(qid='q', prefix_sha256='own-prefix-'+str(j)) for j in range(5)}
        records = [dict(trajectory_id=tid, qid='q', strict_em=int(j==0), actor_version='actor7',
                        prefix_scan_sha256='scan7', prefix_sha256='own-prefix-'+str(j)) for j, tid in enumerate(index)]
        joined = join_own_em(index, records, 'actor7', 'scan7', 'selection7')
        self.assertEqual([r['strict_em'] for r in joined.values()], [1, 0, 0, 0, 0])
        wrong = copy.deepcopy(records); wrong[0]['actor_version'] = 'actor8'
        with self.assertRaises(ValueError):
            join_own_em(index, wrong, 'actor7', 'scan7', 'selection7')
        wrong = copy.deepcopy(records); wrong[0]['prefix_sha256'] = 'newly-resampled-prefix'
        with self.assertRaises(ValueError):
            join_own_em(index, wrong, 'actor7', 'scan7', 'selection7')
        wrong = copy.deepcopy(records); wrong[0]['prefix_scan_sha256'] = 'other-prefix'
        with self.assertRaises(ValueError):
            join_own_em(index, wrong, 'actor7', 'scan7', 'selection7')

    def test_model_only_start_preserves_learned_weights_discards_old_clock(self):
        model = {'output.weight': torch.ones(1, WIDTH*DIM)}
        model.update({str(j): torch.tensor([float(j)]) for j in range(13)})
        source = dict(run_id=SOURCE_RUN, epoch=100,
                      binding=dict(question_max_tokens=QMAX, head_max_tokens=WIDTH),
                      test_manifest_sha256=TEST_SHA, model=model, optimizer={'old': 'moments'},
                      rank_rng=['old']*8, actual_adam_steps_per_rank=[10000]*8)
        initial = initial_online_state(source, dict(run_id='new-formal-run'))
        self.assertIs(initial['model'], model)
        self.assertIsNone(initial['optimizer']); self.assertIsNone(initial['rank_rng'])
        self.assertEqual(initial['actual_adam_steps_per_rank'], [0]*8)
        self.assertEqual(initial['learning_rate'], LR)
        self.assertEqual(source['actual_adam_steps_per_rank'], [10000]*8)
        wrong = dict(source, epoch=99)
        with self.assertRaises(ValueError):
            initial_online_state(wrong, dict(run_id='new'))

    def test_initial_eos_keeps_real_question_and_zero_policy_mask(self):
        row = prefix_row(dict(trajectory_id='eos', length=0, segments=[], segment_bounds=[],
                              initial_context=dict(context_offset=7, request_length=3)))
        self.assertEqual(row['question_length'], 3)
        self.assertEqual(row['policy_length'], 0)
        question = torch.zeros(1, WIDTH, dtype=torch.bool); question[:, :3] = True
        policy = torch.zeros_like(question); positions = torch.zeros(1, WIDTH, dtype=torch.long)
        positions[:, :3] = torch.arange(3)
        batch = dict(features=torch.zeros(1, WIDTH, DIM, dtype=torch.bfloat16),
                     attention_mask=question, question_mask=question, policy_mask=policy,
                     position_ids=positions, question_lengths=torch.tensor([3]), policy_segments=[[]])
        validate_online_layout(batch, 32768)
        bad = dict(batch, policy_segments=[[dict(turn_index=0, policy_start=0, policy_length=1, position_start=3)]])
        with self.assertRaises(ValueError):
            validate_online_layout(bad, 32768)


if __name__ == '__main__':
    unittest.main()
