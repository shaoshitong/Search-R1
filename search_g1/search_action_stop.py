"""Opt-in search-only action stopping for a vLLM0.6.3 engine instance.

Disabled by default. The original StopChecker runs first. This
wrapper may stop only after a complete sampled token closes </search>; it never
stops on </answer>, retokenizes output, edits token IDs/logprobs, or invents EOS.
Single-token decoded fragments are only a cheap trigger. The original dual
decoder/prefix-cut implementation confirms the actual complete token boundary.
"""
from reproduction.common8.behavior_trace import BehaviorTraceError, raw_prefix_to_first_closing_tag

SEARCH_CLOSING = '</search>'
SEARCH_STOP_REASON = 'search_action_boundary:</search>'


class SearchActionStopChecker:
    def __init__(self, original_checker, tokenizer, finished_status, *, decoders=None):
        if decoders is None:
            from reproduction.common8.generation_behavior import prefix_text_decoders
            decoders = prefix_text_decoders(tokenizer)
        self.original, self.tokenizer, self.finished_status = original_checker, tokenizer, finished_status
        self.visible_decode, self.executor_decode = decoders
        self.fragments, self.sequences = {}, {}
        self.last_cut, self.last_sequence_id = None, None
        self.counts = dict(token_checks=0, single_token_decodes=0, filter_triggers=0,
            prefix_confirmations=0, confirmed_search_stops=0, confirmation_rejections=0,
            ambiguous_prefixes=0, native_finished_first=0)

    def _fragment(self, token):
        if token not in self.fragments:
            self.fragments[token] = self.executor_decode([token])
            self.counts['single_token_decodes'] += 1
        return self.fragments[token]

    def boundary(self, seq, sampling_params):
        """Return an original prefix-cut dict or None after native stop checks.

        A CPU replay can call this with a tiny fake Sequence. get_output_token_ids
        is read only on a filter trigger or a previously unseen multi-token row.
        last_cut retains only the most recent confirmed dict, not a growing log.
        """
        count = seq.get_output_len()
        if count < sampling_params.min_tokens or not count:
            return None
        seen, suffix = self.sequences.get(seq.seq_id, (0, ''))
        if count == seen:
            return None
        if count == seen + 1:
            tokens = (seq.get_last_token_id(),)
        else:
            # Handles CPU replay attaching at a later prefix, without decoding
            # the entire growing prefix on every ordinary single-token tick.
            tokens = seq.get_output_token_ids()[seen:] if count > seen else seq.get_output_token_ids()
            if count < seen:
                suffix = ''
        trigger = False
        for token in tokens:
            text = suffix + self._fragment(token)
            trigger |= SEARCH_CLOSING in text
            suffix = text[-(len(SEARCH_CLOSING) - 1):]
        self.sequences[seq.seq_id] = (count, suffix)
        self.counts['token_checks'] += len(tokens)
        if not trigger:
            return None
        self.counts['filter_triggers'] += 1
        raw = list(seq.get_output_token_ids())
        stops = set(sampling_params.all_stop_token_ids)
        stops.add(self.tokenizer.eos_token_id)
        if seq.eos_token_id is not None:
            stops.add(seq.eos_token_id)
        self.counts['prefix_confirmations'] += 1
        try:
            cut = raw_prefix_to_first_closing_tag(raw, self.visible_decode,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                special_token_ids=sorted(set(self.tokenizer.all_special_ids)),
                executor_decode=self.executor_decode, effective_stop_token_ids=sorted(stops))
        except BehaviorTraceError:
            # A lossy/incomplete decode is not evidence of a safe boundary.
            # Continue the unchanged original sampler rather than trim/repair it.
            self.counts['ambiguous_prefixes'] += 1
            return None
        if (cut['closing_tag'] != SEARCH_CLOSING or cut['kept_token_count'] != count or
                cut['kept_token_ids'] != raw or
                cut['decoded_retained_prefix'].endswith('\ufffd') or
                cut['decoded_executor_retained_prefix'].endswith('\ufffd')):
            self.counts['confirmation_rejections'] += 1
            return None
        self.last_cut, self.last_sequence_id = cut, seq.seq_id
        return cut

    def maybe_stop_sequence(self, seq, new_char_count, sampling_params, lora_req=None):
        self.original.maybe_stop_sequence(seq, new_char_count, sampling_params, lora_req=lora_req)
        if seq.is_finished():
            self.sequences.pop(seq.seq_id, None)
            self.counts['native_finished_first'] += 1
            return
        if self.boundary(seq, sampling_params) is None:
            return
        # The actual vLLM output processor has already appended this complete
        # token and its selected logprob. Keep both and let its normal scheduler
        # free the finished sequence. This is a genuine custom string reason.
        seq.status = self.finished_status
        seq.stop_reason = SEARCH_STOP_REASON
        self.sequences.pop(seq.seq_id, None)
        self.counts['confirmed_search_stops'] += 1

    def report(self):
        return dict(enabled=True, search_only=True, answer_stop_enabled=False,
            stop_reason=SEARCH_STOP_REASON, native_stop_checker_runs_first=True,
            raw_tokens_or_logprobs_mutated=False, piece_cache_entries=len(self.fragments),
            live_sequence_filters=len(self.sequences), **self.counts)


def install_search_action_stop(engine, tokenizer):
    """Install on this engine instance only, after its normal construction."""
    from vllm.sequence import SequenceStatus
    checker = SearchActionStopChecker(engine.output_processor.stop_checker, tokenizer,
        SequenceStatus.FINISHED_STOPPED)
    engine.output_processor.stop_checker = checker
    return checker
