"""CPU checks of the representation transition, not a GPU execution gate."""
import unittest

import torch
from torch import nn

from .model import (HIDDEN_SIZE, POLICY_TOKENS, masked_flatten, migrate_checkpoint,
                    validate_question_policy_layout)


class SmallHead(nn.Module):
    """Retain real flattened output dimensions without loading the official block."""

    def __init__(self, question_tokens=0):
        super().__init__()
        self.block = nn.Linear(2, 2)
        self.policy_tokens = POLICY_TOKENS
        self.hidden_size = HIDDEN_SIZE
        self.output = nn.Linear((question_tokens + POLICY_TOKENS) * HIDDEN_SIZE, 1)


def checkpoint():
    torch.manual_seed(135)
    model = SmallHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    for _ in range(3):
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, (index + 1) / 10)
        optimizer.step()
    saved = dict(epoch=70, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 actual_adam_steps_per_rank=[3] * 8,
                 rank_rng=[dict(rank=rank, saved='unchanged') for rank in range(8)],
                 binding=dict(question_hidden_passed_to_head=False),
                 pool_bundle=dict(cursor=21380, cycle=2))
    return model, saved


class MigrationTest(unittest.TestCase):
    def test_learned_output_and_adam_history_survive_new_question_columns(self):
        original, saved = checkpoint()
        target = SmallHead(question_tokens=3)
        migrated, proof = migrate_checkpoint(saved, target)
        prefix_width = 3 * HIDDEN_SIZE
        weight = migrated['model']['output.weight']
        self.assertTrue(torch.equal(weight[:, prefix_width:], saved['model']['output.weight']))
        self.assertEqual(torch.count_nonzero(weight[:, :prefix_width]).item(), 0)
        weight_id = saved['optimizer']['param_groups'][0]['params'][-2]
        for moment in ('exp_avg', 'exp_avg_sq'):
            old = saved['optimizer']['state'][weight_id][moment]
            new = migrated['optimizer']['state'][weight_id][moment]
            self.assertTrue(torch.equal(new[:, prefix_width:], old))
            self.assertEqual(torch.count_nonzero(new[:, :prefix_width]).item(), 0)
        target.load_state_dict(migrated['model'])
        resumed = torch.optim.Adam(target.parameters(), lr=3e-6)
        resumed.load_state_dict(migrated['optimizer'])
        self.assertEqual(resumed.state[target.output.weight]['step'].item(), 3)
        self.assertEqual(resumed.param_groups[0]['lr'], 3e-6)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], 3e-5)
        self.assertEqual(proof['source_learning_rate'], 3e-5)
        self.assertEqual(proof['target_learning_rate'], 3e-6)
        # A real next Adam update confirms the expanded moments load and step
        # onward normally rather than reinitializing the optimizer.
        for parameter in target.parameters():
            parameter.grad = torch.ones_like(parameter)
        resumed.step()
        self.assertEqual(resumed.state[target.output.weight]['step'].item(), 4)
        self.assertEqual(proof['source_actual_adam_steps_per_rank'], [3] * 8)
        self.assertFalse(proof['identical_logits_claimed'])

    def test_block_bias_rng_pool_and_source_objects_are_preserved(self):
        _, saved = checkpoint()
        source_weight = saved['model']['output.weight'].clone()
        migrated, _ = migrate_checkpoint(saved, SmallHead(question_tokens=2))
        self.assertTrue(torch.equal(saved['model']['output.weight'], source_weight))
        self.assertIs(migrated['model']['block.weight'], saved['model']['block.weight'])
        self.assertIs(migrated['model']['output.bias'], saved['model']['output.bias'])
        self.assertIs(migrated['rank_rng'], saved['rank_rng'])
        self.assertIs(migrated['pool_bundle'], saved['pool_bundle'])
        block_id = saved['optimizer']['param_groups'][0]['params'][0]
        self.assertIs(migrated['optimizer']['state'][block_id], saved['optimizer']['state'][block_id])
        self.assertEqual(saved['model']['output.weight'].shape, (1, POLICY_TOKENS * HIDDEN_SIZE))

    def test_layout_preserves_question_gap_and_original_policy_positions(self):
        qmax = 3
        width = qmax + POLICY_TOKENS
        mask = torch.zeros((1, width), dtype=torch.bool)
        mask[0, :2] = True
        mask[0, qmax:qmax + 2] = True
        question_mask = torch.zeros_like(mask)
        question_mask[0, :2] = True
        positions = torch.zeros((1, width), dtype=torch.long)
        positions[0, :2] = torch.tensor([0, 1])
        positions[0, qmax:qmax + 2] = torch.tensor([17, 18])
        batch = dict(features=torch.zeros((1, width, HIDDEN_SIZE)), attention_mask=mask,
                     position_ids=positions, question_lengths=torch.tensor([2]),
                     question_mask=question_mask, policy_mask=mask & ~question_mask,
                     policy_segments=[[dict(policy_start=0, policy_length=2,
                                            position_start=17, turn_index=0)]])
        validate_question_policy_layout(batch, qmax, 32768)
        shifted = dict(batch, position_ids=positions.clone())
        shifted['position_ids'][0, qmax] = 2
        with self.assertRaises(ValueError):
            validate_question_policy_layout(shifted, qmax, 32768)

    def test_masked_flatten_eliminates_question_and_policy_padding(self):
        hidden = torch.tensor([[[1., 2.], [float('nan'), float('nan')],
                                [3., 4.], [19., 23.]]])
        mask = torch.tensor([[True, False, True, False]])
        self.assertTrue(torch.equal(masked_flatten(hidden, mask),
                                    torch.tensor([[1., 2., 0., 0., 3., 4., 0., 0.]])))


if __name__ == '__main__':
    unittest.main()
