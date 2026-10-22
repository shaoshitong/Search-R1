"""Small CPU tests for the opt-in boundary, with no vLLM/GPU dependency."""
from types import SimpleNamespace
import unittest

from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag
from search_g1.search_action_stop import SearchActionStopChecker, SEARCH_STOP_REASON


class Tokenizer:
    eos_token_id, pad_token_id, all_special_ids = 0, 99, [0, 90, 99]
    pieces = {0: '<EOS>', 1: '<search>x', 2: '</sea', 3: 'rch>TAIL',
        4: '</answer>', 5: 'answer then ', 6: 'more', 7: '</search>',
        8: '</search>\ufffd', 90: '<SPECIAL>', 99: '<PAD>'}

    def __init__(self):
        self.calls = []

    def decode(self, ids, skip_special_tokens):
        self.calls.append((list(ids), skip_special_tokens))
        return ''.join(self.pieces[token] for token in ids
            if not skip_special_tokens or token not in self.all_special_ids)


class Sequence:
    def __init__(self, identity=1):
        self.seq_id, self.eos_token_id = identity, 0
        self.tokens, self.logprobs = [], []
        self.status, self.stop_reason = 'RUNNING', None
        self.full_token_reads = 0

    def append(self, token):
        self.tokens.append(token); self.logprobs.append(-len(self.tokens) / 10)

    def get_output_len(self): return len(self.tokens)
    def get_last_token_id(self): return self.tokens[-1]
    def get_output_token_ids(self):
        self.full_token_reads += 1
        return self.tokens
    def is_finished(self): return self.status != 'RUNNING'


class OriginalChecker:
    """Tiny mock of the relevant original EOS/configured-stop/length priority."""
    def maybe_stop_sequence(self, seq, new_char_count, params, lora_req=None):
        if seq.get_output_len() < params.min_tokens:
            return
        if not params.ignore_eos and seq.get_last_token_id() == seq.eos_token_id:
            seq.status = 'NATIVE_STOP'; return
        if seq.get_last_token_id() in params.stop_token_ids:
            seq.status = 'NATIVE_STOP'; seq.stop_reason = seq.get_last_token_id(); return
        if seq.get_output_len() == params.max_tokens:
            seq.status = 'NATIVE_LENGTH'


class SearchBoundaryTests(unittest.TestCase):
    def fixture(self, **overrides):
        tokenizer = Tokenizer()
        params = SimpleNamespace(min_tokens=0, max_tokens=500, ignore_eos=False,
            stop_token_ids=[], all_stop_token_ids={0})
        for key, value in overrides.items(): setattr(params, key, value)
        decoders = (lambda ids: tokenizer.decode(ids, False), lambda ids: tokenizer.decode(ids, True))
        checker = SearchActionStopChecker(OriginalChecker(), tokenizer, 'SEARCH_STOP', decoders=decoders)
        return tokenizer, params, checker, Sequence()

    def feed(self, seq, checker, params, tokens):
        for token in tokens:
            seq.append(token)
            checker.maybe_stop_sequence(seq, 0, params)
            if seq.is_finished(): break

    def test_cross_token_search_preserves_complete_boundary_token_and_logprobs(self):
        tokenizer, params, checker, seq = self.fixture()
        self.feed(seq, checker, params, [1, 2, 3, 6, 0])
        self.assertEqual(seq.tokens, [1, 2, 3])
        self.assertEqual(seq.logprobs, [-.1, -.2, -.3])
        self.assertEqual(seq.stop_reason, SEARCH_STOP_REASON)
        self.assertEqual(checker.last_cut['retained_token_boundary_suffix'], 'TAIL')
        old = raw_prefix_to_first_closing_tag([1, 2, 3, 6, 0], lambda ids: tokenizer.decode(ids, False),
            eos_token_id=0, pad_token_id=99, special_token_ids=[0, 90, 99],
            executor_decode=lambda ids: tokenizer.decode(ids, True), effective_stop_token_ids=[0])
        for key in ('kept_token_ids', 'execution_text', 'retained_token_boundary_suffix'):
            self.assertEqual(checker.last_cut[key], old[key])

    def test_answer_before_search_never_stops_at_answer(self):
        _, params, checker, seq = self.fixture()
        self.feed(seq, checker, params, [4, 5, 6])
        self.assertFalse(seq.is_finished())
        self.assertEqual(checker.report()['prefix_confirmations'], 0)
        self.feed(seq, checker, params, [1, 2, 3])
        self.assertEqual(seq.stop_reason, SEARCH_STOP_REASON)
        self.assertEqual(checker.last_cut['closing_tag'], '</search>')
        self.assertTrue(checker.last_cut['execution_text'].startswith('</answer>'))

    def test_skipped_special_between_tag_fragments_remains_in_original_tokens(self):
        _, params, checker, seq = self.fixture()
        self.feed(seq, checker, params, [1, 2, 90, 3])
        self.assertEqual(seq.tokens, [1, 2, 90, 3])
        self.assertEqual(len(seq.logprobs), 4)
        self.assertEqual(checker.last_cut['included_nonterminal_special_positions'], [2])
        self.assertTrue(checker.last_cut['execution_text'].endswith('</search>'))

    def test_native_eos_takes_priority_without_decoding(self):
        tokenizer, params, checker, seq = self.fixture()
        self.feed(seq, checker, params, [0])
        self.assertEqual(seq.status, 'NATIVE_STOP')
        self.assertIsNone(seq.stop_reason)
        self.assertEqual(tokenizer.calls, [])

    def test_native_length_takes_priority_on_search_boundary(self):
        _, params, checker, seq = self.fixture(max_tokens=3)
        self.feed(seq, checker, params, [1, 2, 3])
        self.assertEqual(seq.status, 'NATIVE_LENGTH')
        self.assertIsNone(seq.stop_reason)
        self.assertIsNone(checker.last_cut)

    def test_original_configured_stop_takes_priority(self):
        tokenizer, params, checker, seq = self.fixture(stop_token_ids=[7], all_stop_token_ids={0, 7})
        self.feed(seq, checker, params, [7])
        self.assertEqual(seq.status, 'NATIVE_STOP')
        self.assertEqual(seq.stop_reason, 7)
        self.assertEqual(tokenizer.calls, [])

    def test_ordinary_tokens_use_single_piece_cache_not_growing_decode(self):
        tokenizer, params, checker, seq = self.fixture()
        self.feed(seq, checker, params, [6] * 100)
        self.assertFalse(seq.is_finished())
        self.assertEqual(tokenizer.calls, [([6], True)])
        self.assertEqual(seq.full_token_reads, 0)
        self.assertEqual(checker.report()['prefix_confirmations'], 0)

    def test_fragment_trigger_requires_full_prefix_confirmation(self):
        tokenizer, params, checker, seq = self.fixture()
        def executor(ids):
            if len(ids) > 1: return 'not a closing tag'
            return tokenizer.decode(ids, True)
        checker.executor_decode = executor
        self.feed(seq, checker, params, [1, 7])
        self.assertFalse(seq.is_finished())
        self.assertEqual(checker.report()['confirmation_rejections'], 1)

    def test_unresolved_utf8_suffix_conservatively_keeps_original_generation(self):
        _, params, checker, seq = self.fixture()
        self.feed(seq, checker, params, [1, 8])
        self.assertFalse(seq.is_finished())
        self.assertEqual(checker.report()['confirmation_rejections'], 1)

    def test_unresolved_visible_utf8_suffix_also_keeps_original_generation(self):
        tokenizer, params, checker, seq = self.fixture()
        checker.visible_decode = lambda ids: tokenizer.decode(ids, False) + ('\ufffd' if ids and ids[-1] == 3 else '')
        self.feed(seq, checker, params, [1, 2, 3])
        self.assertFalse(seq.is_finished())
        self.assertEqual(checker.report()['confirmation_rejections'], 1)


if __name__ == '__main__':
    unittest.main()
