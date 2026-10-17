"""CPU-only behavioral checks for train/QA parity and original prefix5 evidence."""
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from . import eval as evaluation
from . import interaction_protocol as protocol
from . import rollout


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    vocabulary = {1: '<search>', 2: 'x', 3: '</search>', 4: 'Reason', 5: 'ing ',
                  6: '<answer>', 7: 'ok', 8: '</answer>', 9: '</answer>suffix'}
    def encode(self, text, **kwargs):
        return [ord(char) + 1000 for char in text]
    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text))
    def apply_chat_template(self, messages, **kwargs):
        return 'official prompt'
    def decode(self, tokens, **kwargs):
        return ''.join(self.vocabulary[token] if token in self.vocabulary else chr(token - 1000)
                       for token in tokens if token)


TOKENIZER = Tokenizer()


def state():
    return rollout.initial_state(dict(run_id='run', actor_version='actor'), 1,
        dict(candidate_ordinal=0, qid='q', question_hash='hash', prompt=[]), 0, TOKENIZER)


def record(current, phase, tokens, *, stopped=False):
    protocol.start_action(current, protocol.BUDGET)
    request = rollout.next_request(current, phase, {})
    cut = protocol.append_chunk(current, request, tokens, [-.125] * len(tokens),
        dict(max_tokens=request['max_tokens'], logprobs=1, temperature=0. if phase == 'evaluate' else 1.),
        finish_reason='stop' if stopped else ('length' if len(tokens) == request['max_tokens'] else None),
        stop_reason=None, terminal_ids=[0], decode=TOKENIZER.decode,
        action_boundary_abort=protocol.stream_action_present(tokens, TOKENIZER.decode))
    return request, cut


class InteractionProtocolTests(unittest.TestCase):
    def test_guard_resampling_identity_changes_native_seeds_without_changing_initial_seed(self):
        import hashlib
        original = ['run', 1, 'trajectory', 0, 0, 1729]
        expected = int.from_bytes(hashlib.sha256(rollout.canonical(original)).digest()[:8], 'big') % (2 ** 31 - 1)
        self.assertEqual(rollout.seed_for('run', 1, 'trajectory', 0, 0), expected)
        retry = rollout.seed_for('run', 1, 'trajectory', 0, 0, sampling_attempt_id='retry1')
        self.assertNotEqual(retry, expected)
        self.assertEqual(retry, rollout.seed_for('run', 1, 'trajectory', 0, 0, sampling_attempt_id='retry1'))
        current = state(); protocol.start_action(current, protocol.BUDGET)
        request = rollout.next_request(current, 'prefix', {'sampling_attempt_id': 'retry1'})
        self.assertEqual(request['sampling_attempt_id'], 'retry1')
        self.assertNotEqual(request['seed'], rollout.seed_for(current['run_id'], current['outer'],
            current['trajectory_id'], 0, 0))

    def test_train_and_qa_use_same_action_implementation(self):
        for name in ('start_action', 'append_chunk', 'finish_action', 'stream_action_present',
                     'native_snapshot', 'registered_sampling', 'terminal_token_ids'):
            self.assertIs(getattr(evaluation, name), getattr(protocol, name))
            self.assertIs(getattr(rollout, name), getattr(protocol, name))
        self.assertEqual(rollout.INTERACTION_CONTRACT, evaluation.INTERACTION_CONTRACT)

    def test_prefix5_pause_keeps_original_ids_lp_masks_positions_and_matches_qa(self):
        train, qa = state(), state()
        environment = ('\n\n<information>' + 'evidence ' * 100 + '</information>\n\n', False, 1, 1)
        for current, phase in ((train, 'prefix'), (qa, 'evaluate')):
            _, cut = record(current, phase, [1, 2, 3])
            protocol.finish_action(current, cut, environment, TOKENIZER, protocol.BUDGET)
        observation = train['turns'][0]['observation_token_ids']
        self.assertEqual(len(observation), 500)
        self.assertTrue(TOKENIZER.decode(observation).strip().startswith('<information>'))
        self.assertTrue(TOKENIZER.decode(observation).strip().endswith('</information>'))
        request, cut = record(train, 'prefix', [4, 5])
        self.assertEqual(request['max_tokens'], 2)
        self.assertIsNone(cut)
        before = rollout.prefix_record(train)
        prefix_lp = train['log_probs'] + train['action']['selected_logprobs']
        prefix_mask = train['info_mask'] + [1, 1]
        train = json.loads(json.dumps(train))  # The actual selected-state transport boundary.
        request, cut = record(train, 'continue', [6, 7, 8])
        self.assertEqual(request['max_tokens'], 498)
        self.assertEqual(request['actual_request_token_ids'], before['prefix_segments'][-1]['context_tokens'])
        protocol.finish_action(train, cut, ('', True, 1, 0), TOKENIZER, protocol.BUDGET)
        _, cut = record(qa, 'evaluate', [4, 5, 6, 7, 8])
        protocol.finish_action(qa, cut, ('', True, 1, 0), TOKENIZER, protocol.BUDGET)
        after = rollout.prefix_record(train)
        self.assertEqual(before['policy_token_ids'], [1, 2, 3, 4, 5])
        for key in ('policy_token_ids', 'policy_position_ids', 'prefix_segments'):
            self.assertEqual(before[key], after[key])
        self.assertEqual(train['response'][:len(prefix_mask)], qa['response'][:len(prefix_mask)])
        self.assertEqual(train['info_mask'][:len(prefix_mask)], prefix_mask)
        self.assertEqual(train['log_probs'][:len(prefix_lp)], prefix_lp)
        for key in ('prompt', 'response', 'info_mask', 'log_probs', 'ordinal', 'terminal', 'terminal_reason'):
            self.assertEqual(train[key], qa[key])
        self.assertEqual(sum(train['info_mask']), 8)
        self.assertTrue(all(lp == 0. for lp, mask in zip(train['log_probs'], train['info_mask']) if not mask))

    def test_stream_boundary_clips_at_first_action_and_preserves_original_boundary_token(self):
        self.assertFalse(protocol.stream_action_present([6, 7], TOKENIZER.decode))
        self.assertTrue(protocol.stream_action_present([6, 7, 8], TOKENIZER.decode))
        cut = protocol.first_action_cut([6, 7, 8, 1, 2, 3], TOKENIZER.decode)
        self.assertEqual(cut['kept_token_ids'], [6, 7, 8])
        self.assertEqual(cut['execution_text'], '<answer>ok</answer>')
        straddled = protocol.first_action_cut([6, 7, 9, 1], TOKENIZER.decode)
        self.assertEqual(straddled['kept_token_ids'], [6, 7, 9])
        self.assertEqual(straddled['execution_text'], '<answer>ok</answer>')

    def test_early_final_reserve_and_no_context_cropping(self):
        current = state(); current['response'] = [7] * (3097 - len(current['prompt']))
        original = current['prompt'] + current['response']
        protocol.start_action(current, protocol.BUDGET)
        self.assertEqual(current['action']['actual_request_token_ids'], original)
        self.assertTrue(current['action']['final_turn'])
        _, cut = record(current, 'evaluate', [1, 2, 3])
        protocol.finish_action(current, cut, ('<information>ignored</information>', False, 1, 1),
                               TOKENIZER, protocol.BUDGET)
        self.assertTrue(current['terminal'])
        self.assertEqual(current['retrieval_calls'], 0)
        self.assertEqual(current['turns'][0]['observation_token_ids'], [])
        self.assertEqual(current['response'][:-3], original[len(current['prompt']):])
        too_long = state(); too_long['prompt'] = [7] * 3597
        with self.assertRaisesRegex(ValueError, 'cannot accommodate'):
            protocol.start_action(too_long, protocol.BUDGET)
        self.assertEqual(len(too_long['prompt']), 3597)
        no_room, clipped = protocol.observation_tokens(TOKENIZER, '<information>body</information>', 5)
        self.assertEqual(no_room, []); self.assertTrue(clipped)
        final = state(); final['ordinal'] = 4
        protocol.start_action(final, protocol.BUDGET)
        self.assertTrue(final['action']['final_turn'])

    def test_natural_eos_is_terminal_without_added_environment_tokens(self):
        current = state(); _, cut = record(current, 'prefix', [7, 0], stopped=True)
        protocol.finish_action(current, cut, ('<information>ignored</information>', False, 0, 1),
                               TOKENIZER, protocol.BUDGET)
        self.assertEqual(current['response'], [7])
        self.assertEqual(current['log_probs'], [-.125])
        self.assertEqual(current['info_mask'], [1])
        self.assertTrue(current['terminal'])
        self.assertEqual(current['terminal_reason'], 'natural_eos')
        stop, terminal = protocol.terminal_token_ids(dict(eos_token_id=[0, 99]), TOKENIZER)
        self.assertEqual(stop, [0, 99]); self.assertEqual(terminal, [0, 99])

    def test_official_strict_scorer_still_rejects_environment_tags_and_multiple_actions(self):
        # Load the unmodified scorer with its real normalization/extraction code.
        # Bypass only verl.__init__'s unrelated GPU-training dependency imports.
        root = Path(__file__).resolve().parents[2]
        modules = {}
        for name, path in (('verl', root / 'verl'), ('verl.utils', root / 'verl/utils'),
                           ('verl.utils.reward_score', root / 'verl/utils/reward_score')):
            module = ModuleType(name); module.__path__ = [str(path)]; modules[name] = module
        scorer = ModuleType('verl.utils.reward_score.qa_em')
        scorer.__dict__.update(runpy.run_path(str(root / 'verl/utils/reward_score/qa_em.py')))
        modules['verl.utils.reward_score.qa_em'] = scorer
        modules['verl.utils.reward_score'].qa_em = scorer
        with patch.dict(sys.modules, modules):
            for text, expected in (('<answer>ok</answer>', 1),
                                   ('</information><answer>ok</answer>', 0),
                                   ('<answer>ok</answer>suffix', 0),
                                   ('<answer>ok</answer><search>x</search>', 0),
                                   ('<answer><information>ok</information></answer>', 0)):
                ids = TOKENIZER.encode(text)
                result = evaluation.score_tokens(TOKENIZER, [], ids, [1] * len(ids), ['ok'])
                self.assertEqual(result['em'], expected)
                self.assertEqual(rollout.strict_label(TOKENIZER, ids, [1] * len(ids), dict(target=['ok'])), expected)
            ids = TOKENIZER.encode('<answer>ok</answer>')
            self.assertEqual(evaluation.score_tokens(TOKENIZER, [], ids, [0] * len(ids), ['ok'])['em'], 0)


if __name__ == '__main__': unittest.main()
