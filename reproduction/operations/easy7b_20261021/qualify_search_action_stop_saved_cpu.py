"""Remote CPU replay of saved actual tokens/LP; no model or new generation.

Compares only the retained action, not the intentionally different raw-tail or
EOS metadata. This is a fixed-token replay, not a live RNG-equivalence claim.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace


RETAINED = ('kept_token_count', 'kept_token_ids', 'closing_tag', 'execution_text',
    'decoded_retained_prefix', 'decoded_executor_retained_prefix',
    'decoded_executor_previous_prefix', 'retained_token_boundary_suffix',
    'token_boundary_suffix_classification', 'retained_trailing_whitespace',
    'token_boundary_whitespace_retained', 'included_nonterminal_special_positions')


class ReplaySequence:
    def __init__(self, seq_id, eos):
        self.seq_id, self.eos_token_id = seq_id, eos
        self.tokens, self.logprobs, self.status, self.stop_reason = [], [], 'running', None

    def get_output_len(self): return len(self.tokens)
    def get_last_token_id(self): return self.tokens[-1]
    def get_output_token_ids(self): return self.tokens
    def is_finished(self): return self.status != 'running'


class SavedTokenStopChecker:
    """Original EOS -> configured token-stop -> length order for this recipe.

    The saved caller is detokenize=False, has no string stops and max_tokens500;
    there is no string-decoder or model-length termination to invent here.
    """
    def maybe_stop_sequence(self, seq, new_char_count, params, lora_req=None):
        if seq.get_output_len() < params.min_tokens:
            return
        token = seq.get_last_token_id()
        if not params.ignore_eos and token == seq.eos_token_id:
            seq.status = 'stop'
        elif token in params.stop_token_ids:
            seq.status, seq.stop_reason = 'stop', token
        elif seq.get_output_len() >= params.max_tokens:
            seq.status = 'length'


def lp_bytes(values):
    return struct.pack('<' + 'd' * len(values), *values)


def replay_turn(turn, checker, ordinal):
    raw, lp, expected = (turn[key] for key in
        ('raw_generated_token_ids', 'selected_logprobs', 'prefix_cut'))
    proof = turn['effective_stop_provenance']
    sampling = dict(proof['effective_sampling'])
    if (sampling['detokenize'] is not False or sampling.get('stop') not in (None, []) or
            sampling['max_tokens'] != 500 or len(raw) != len(lp) or not raw or
            not all(math.isfinite(value) for value in lp)):
        raise ValueError('unsupported_or_incomplete_saved_token_recipe')
    sampling.update(min_tokens=sampling.get('min_tokens') or 0,
        stop_token_ids=list(proof['effective_stop_token_ids']),
        all_stop_token_ids=list(proof['effective_all_stop_token_ids']))
    params = SimpleNamespace(**sampling)
    original = ReplaySequence(ordinal, proof['seq_eos_token_id'])
    candidate = ReplaySequence(ordinal, proof['seq_eos_token_id'])
    errors, cut = Counter(), None
    for token, value in zip(raw, lp):
        if original.is_finished():
            errors['original_finished_before_saved_raw_end'] += 1
            break
        original.tokens.append(token); original.logprobs.append(value)
        checker.original.maybe_stop_sequence(original, 0, params)
        if not candidate.is_finished():
            candidate.tokens.append(token); candidate.logprobs.append(value)
            checker.maybe_stop_sequence(candidate, 0, params)
            if candidate.stop_reason == 'search_action_boundary:</search>':
                cut = checker.last_cut
                errors['wrong_sequence_cut'] += int(checker.last_sequence_id != ordinal)
    errors['original_finish_metadata'] += int(original.status != turn['finish_reason'] or
        original.stop_reason != turn['stop_reason'])
    kept = expected['kept_token_count']
    errors['saved_kept_ids'] += int(expected['kept_token_ids'] != raw[:kept] or
                                    turn['kept_token_count'] != kept)
    if cut is not None:
        for key in RETAINED:
            errors[key] += int(cut[key] != expected[key])
        errors['retained_logprobs'] += int(lp_bytes(candidate.logprobs) != lp_bytes(lp[:kept]))
        errors['retained_tokens'] += int(candidate.tokens != expected['kept_token_ids'])
    else:
        errors['unrequested_early_stop'] += int(len(candidate.tokens) != len(raw) or
            candidate.status != original.status or candidate.stop_reason != original.stop_reason)
        errors['unmodified_full_logprobs'] += int(lp_bytes(candidate.logprobs) != lp_bytes(lp))
    eligible = (expected['closing_tag'] == '</search>' and 0 < kept < len(raw) and
                kept < params.max_tokens and kept >= params.min_tokens and
                raw[kept - 1] not in params.all_stop_token_ids)
    counts = Counter(actions=1, original_raw_tokens=len(raw), replay_candidate_tokens=len(candidate.tokens),
        search_cuts=int(expected['closing_tag'] == '</search>'),
        answer_cuts=int(expected['closing_tag'] == '</answer>'),
        no_action_tag=int(expected['closing_tag'] is None), native_eos_or_stop=int(original.status == 'stop'),
        native_length=int(original.status == 'length'), confirmed_search_stops=int(cut is not None),
        eligible_search_stops=int(eligible), missed_eligible_search_stops=int(eligible and cut is None),
        tokens_avoided=len(raw) - len(candidate.tokens), retained_logprobs_compared=kept if cut else 0)
    digest = hashlib.sha256(json.dumps({key: expected[key] for key in RETAINED},
        sort_keys=True, separators=(',', ':')).encode() + lp_bytes(lp[:kept])).digest()
    return counts, +errors, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-file', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('Remote CPU only; never copy raw candidate tensors to Mac.')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        RAYON_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    from transformers import AutoTokenizer
    from search_g1.search_action_stop import SearchActionStopChecker
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    report = dict(scope='SAVED_ACTUAL_TOKEN_PREFIX_REPLAY_ONLY_NOT_LIVE_RNG_OR_GPU_QUALIFICATION',
        files=[], totals={}, model_loaded=False, generation_performed=False, cpu_threads=1,
        production_modified=False, token_or_text_output=False, raw_eos_metadata_compared_as_whole=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        temporary = args.output.with_suffix(args.output.suffix + '.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    totals, all_errors = Counter(), Counter()
    for path in args.candidate_file:
        began = time.monotonic()
        row = dict(path=str(path), source_bytes=path.stat().st_size, status='LOADING', actions=0)
        report['files'].append(row); publish()
        record = torch.load(path, map_location='cpu', weights_only=False)
        row.update(load_seconds=time.monotonic() - began, status='REPLAYING')
        checker = SearchActionStopChecker(SavedTokenStopChecker(), tokenizer, 'stop')
        counts, errors, digest = Counter(), Counter(), hashlib.sha256()
        for trajectory in record['output'].non_tensor_batch['behavior_turns']:
            for turn in trajectory:
                values, mismatch, action_hash = replay_turn(turn, checker, counts['actions'])
                counts.update(values); errors.update(mismatch); digest.update(action_hash)
                if counts['actions'] % 256 == 0:
                    row.update(counts, errors=dict(errors), elapsed_seconds=time.monotonic() - began)
                    publish()
        row.update(counts, errors=dict(errors), retained_content_sha256=digest.hexdigest(),
            checker=checker.report(), source_bytes_after=path.stat().st_size,
            elapsed_seconds=time.monotonic() - began, status='COMPLETE')
        row['replay_seconds'] = row['elapsed_seconds'] - row['load_seconds']
        row['tokens_avoided_fraction'] = counts['tokens_avoided'] / max(1, counts['original_raw_tokens'])
        totals.update(counts); all_errors.update(errors)
        report.update(totals=dict(totals), errors=dict(all_errors), cuda_initialized=torch.cuda.is_initialized())
        publish(); print(json.dumps(row), flush=True)
        del record, checker, trajectory, turn
        gc.collect()
    report['all_compared_retained_content_exact'] = totals['actions'] > 0 and not all_errors
    report['all_eligible_search_stops_found'] = totals['missed_eligible_search_stops'] == 0
    publish(); print(json.dumps(report), flush=True)
    # Conservative no-stop fallbacks are reported separately, not called an
    # incorrect retained action or turned into a new production admission gate.
    if not report['all_compared_retained_content_exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
