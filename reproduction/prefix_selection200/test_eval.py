"""Legacy context/observation invariants without GPU or evaluation reruns."""
import unittest
from .eval import CONTRACT, legacy_final_turn, legacy_observation_tokens


class LegacyEvaluationTests(unittest.TestCase):
    def test_raw_observation_clip_does_not_repair_tags(self):
        class Tokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {'input_ids': list(range(600))}
        self.assertEqual(legacy_observation_tokens(Tokenizer(), 'tool'), list(range(500)))

    def test_final_turn_only_after_four_tool_turns(self):
        self.assertEqual([legacy_final_turn(i) for i in range(5)], [False] * 4 + [True])
        self.assertEqual(CONTRACT['compute_dtype'], 'float32')
        self.assertEqual(CONTRACT['answer_reserve_tokens'], 0)
        self.assertFalse(CONTRACT['early_final'])
        self.assertTrue(CONTRACT['asynchronous_generation'])


if __name__ == '__main__': unittest.main()
