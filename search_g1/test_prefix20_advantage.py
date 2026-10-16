import unittest
import torch
from search_g1.prefix20_advantage import anchored_grpo


class VirtualZeroTest(unittest.TestCase):
    def test_only_all_correct_gets_virtual_zero_despite_positive_soft_rewards(self):
        real = torch.tensor([.97, 1.03, 1.05, .99, 1.01])
        rewards = real.repeat(3).reshape(-1, 1)
        mask = torch.tensor([[1., 0.]]).repeat(15, 1)
        reward_tokens = torch.cat((rewards, torch.zeros_like(rewards)), dim=1)
        labels = [1]*5 + [0]*5 + [0, 1, 0, 1, 0]
        actual, returns = anchored_grpo(reward_tokens, mask, ['a']*5+['b']*5+['c']*5, labels)
        baseline = torch.cat((torch.zeros(1), real))
        expected_anchor = (real-baseline.mean()) / (baseline.std()+1e-6)
        expected_original = (real-real.mean()) / (real.std()+1e-6)
        torch.testing.assert_close(actual[:5, 0], expected_anchor)
        torch.testing.assert_close(actual[5:10, 0], expected_original)
        torch.testing.assert_close(actual[10:, 0], expected_original)
        self.assertTrue(bool((actual[:, 1] == 0).all()))
        self.assertEqual(actual.shape, (15, 2))
        torch.testing.assert_close(actual, returns)

    def test_group_balancing_permutation_preserves_each_trajectory_advantage(self):
        rewards = torch.arange(10, dtype=torch.float32).reshape(10, 1) / 10
        groups = ['a']*5+['b']*5; labels = [1]*5+[0, 1, 0, 0, 1]
        mask = torch.ones_like(rewards)
        before, _ = anchored_grpo(rewards, mask, groups, labels)
        order = [9, 0, 7, 2, 5, 4, 8, 1, 6, 3]
        after, _ = anchored_grpo(rewards[order], mask[order], [groups[i] for i in order], [labels[i] for i in order])
        torch.testing.assert_close(after, before[order])
        torch.testing.assert_close(rewards, torch.arange(10).reshape(10, 1)/10)

    def test_fail_closed_on_missing_or_incomplete_labels(self):
        with self.assertRaisesRegex(ValueError, 'ROW_ALIGNED'):
            anchored_grpo(torch.ones(5, 1), torch.ones(5, 1), ['q']*5, [])
        with self.assertRaisesRegex(ValueError, 'COMPLETE_G5'):
            anchored_grpo(torch.ones(4, 1), torch.ones(4, 1), ['q']*4, [1]*4)


if __name__ == '__main__': unittest.main()
