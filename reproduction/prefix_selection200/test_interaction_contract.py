"""Actual token fixtures for one train/QA context and termination contract."""
import unittest

from .interaction_contract import (BUDGET, final_turn, first_action_cut,
    interaction_outcome, stream_action_present)
from .test_rollout import CharacterTokenizer, decode, state, record
from .rollout import start_action, next_request, finish_action, prefix_record


class SharedInteractionTests(unittest.TestCase):
    def test_same_budget_early_final_and_observation_wrapper(self):
        self.assertFalse(final_turn(3096, 0))
        self.assertTrue(final_turn(3097, 0))
        self.assertTrue(final_turn(10, 4))
        with self.assertRaisesRegex(ValueError, 'accommodate'):
            final_turn(3597, 0)
        s = state(); start_action(s, BUDGET)
        request = next_request(s, 'continue', {})
        cut = record(s, request, [ord(c) for c in '<search>x</search>'], None, True)
        environment = ('<information>' + 'x' * 700 + '</information>', False, 1, 1)
        transition = interaction_outcome(CharacterTokenizer(), len(s['prompt']),
            cut['kept_token_count'], s['action']['final_turn'], cut, environment)
        finish_action(s, cut, environment, CharacterTokenizer(), BUDGET)
        observation = s['turns'][0]['observation_token_ids']
        self.assertEqual(observation, transition['observation_token_ids'])
        self.assertEqual(len(observation), 500)
        self.assertTrue(decode(observation).startswith('<information>'))
        self.assertTrue(decode(observation).endswith('</information>'))
        self.assertTrue(transition['observation_clipped'])

    def test_eos_is_not_policy_or_tool_observation(self):
        raw = [ord(c) for c in 'unfinished'] + [0]
        cut = first_action_cut(raw, decode, terminal_ids=[0], native_stopped=True)
        self.assertEqual(decode(cut['kept_token_ids']), 'unfinished')
        transition = interaction_outcome(CharacterTokenizer(), 10, cut['kept_token_count'],
            False, cut, ('<information>never</information>', False, 1, 1))
        self.assertTrue(transition['terminal'])
        self.assertEqual(transition['observation_token_ids'], [])
        self.assertEqual(transition['retrieval_calls'], 0)

    def test_actual_stream_first_action_trims_future_action(self):
        text = '<answer>yes</answer><search>future</search>'
        raw = [ord(c) for c in text]
        self.assertTrue(stream_action_present(raw, decode))
        cut = first_action_cut(raw, decode)
        self.assertEqual(cut['action'], 'answer')
        self.assertEqual(decode(cut['kept_token_ids']), '<answer>yes</answer>')

    def test_twenty_token_prefix_resumes_identical_context_without_lookahead(self):
        s = state(); start_action(s, BUDGET)
        first = next_request(s, 'prefix', {})
        tokens = [ord(c) for c in 'a' * 20]
        self.assertIsNone(record(s, first, tokens))
        scored = prefix_record(s)
        next_chunk = next_request(s, 'continue', {})
        self.assertEqual(next_chunk['actual_request_token_ids'], s['prompt'] + tokens)
        self.assertEqual(next_chunk['max_tokens'], 480)
        self.assertEqual(scored['policy_token_ids'], tokens)
        self.assertEqual(scored['policy_position_ids'], list(range(2, 22)))
        self.assertFalse(scored['future_information_included'])


if __name__ == '__main__': unittest.main()
