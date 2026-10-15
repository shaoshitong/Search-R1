"""Small CPU fixtures for two-rank plans and prefix-only absolute positions."""
import random
import types
import unittest

import torch

from .model import validate_prefix_positions
from .train_ddp import batch_plan, validate_cache, EPOCHS, STEPS_PER_EPOCH, WORLD, LOCAL_BATCH


class PrefixOnlyContractTests(unittest.TestCase):
    def test_same_balanced_plans_across_four_arms_and_two_ranks(self):
        labels = [1] * 5627 + [0] * (25600 - 5627)
        plans = [batch_plan(labels, 'balanced', random.Random(1729)) for _ in (10, 20, 40, 80)]
        self.assertTrue(all(plan == plans[0] for plan in plans))
        self.assertEqual((WORLD, LOCAL_BATCH, EPOCHS * STEPS_PER_EPOCH), (2, 128, 2000))
        for batch in plans[0]:
            self.assertEqual(len(batch), 256)
            self.assertEqual(sum(labels[i] for i in batch), 128)
            self.assertEqual(batch[:LOCAL_BATCH] + batch[LOCAL_BATCH:], batch)

    def test_natural_optional_mode_has_exact_epoch_coverage(self):
        labels = [1] * 5000 + [0] * 20600
        batches = batch_plan(labels, 'natural', random.Random(1729))
        self.assertEqual(len(batches), 100)
        self.assertEqual(sorted(sum(batches, [])), list(range(25600)))

    def test_absolute_prefix_positions_and_short_eos_padding(self):
        mask = torch.tensor([[True, True, True], [True, False, False]])
        positions = torch.tensor([[242, 243, 244], [111, 0, 0]], dtype=torch.long)
        starts = torch.tensor([242, 111], dtype=torch.long)
        validate_prefix_positions(mask, positions, starts)

    def test_zero_reset_future_positions_and_holes_are_rejected(self):
        starts = torch.tensor([242], dtype=torch.long)
        mask = torch.tensor([[True, True, False]])
        for positions in (torch.tensor([[0, 1, 0]]), torch.tensor([[242, 244, 0]])):
            with self.assertRaises(ValueError):
                validate_prefix_positions(mask, positions, starts)
        with self.assertRaises(ValueError):
            validate_prefix_positions(torch.tensor([[True, False, True]]), torch.tensor([[242, 0, 244]]), starts)

    def test_cache_only_prefix_slots_and_same_capacity80(self):
        train = ['train-%d' % i for i in range(25600)]
        test = ['test-%d' % i for i in range(2560)]
        index = {}
        for split, ids in [('train', train), ('test', test)]:
            for i, tid in enumerate(ids):
                index[tid] = dict(qid='%s-q%d' % (split, i // 5), question_hash='%s-h%d' % (split, i // 5),
                                  strict_em=i % 2, prompt_length=242, prefix_length=10, length=10)
        cache = types.SimpleNamespace(ids_by_split=dict(train=train, test=test), index=index,
                                      max_tokens=80, feature_dim=2048,
                                      manifest=dict(representation='prefix_only_first_request_layer27_v1'),
                                      spec=dict(prefix_length=10))
        self.assertEqual(validate_cache(cache, 10), (train, test))
        cache.max_tokens = 262
        with self.assertRaises(ValueError):
            validate_cache(cache, 10)
        cache.max_tokens = 80
        cache.index[train[0]]['length'] = 252  # Prompt positions must never enter the head.
        with self.assertRaises(ValueError):
            validate_cache(cache, 10)


if __name__ == '__main__':
    unittest.main()
