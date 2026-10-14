"""V2 bounded-stream CPU-only historical TRAIN strict-C reconstruction.

This never replays the shaped reward or computes a correlation.  It authenticates
committed update -> native gate -> complete raw trace, joins the original hashed
extra_info.index to a unique original ground_truth, and calls only the frozen
tokenizer decoder / strict scorer.  The complete trace is first SHA-streamed;
only its canonical first actor_rows array is decoded, one bounded row at a time.
The much larger token/sampling journal is not materialized.  Production writes require an explicit fresh
Linux diagnostic directory outside the live experiment.  No raw tokens, text,
questions, golds or per-row records are printed to stdout.  Do not execute before
the separate metadata gold-join audit and source publication have passed.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import sys


class Hold(Exception):
    """Controlled reason codes must contain no private row contents."""


def require(value, reason):
    if not value:
        raise Hold(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def document_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read_bound(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path,
            'missing_or_symlink_input')
    require(path.stat().st_size <= 32 * 1024 * 1024, 'bound_metadata_JSON_exceeds_limit')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'fresh_input_sha_mismatch')
    return json.loads(raw)


def gold_mapping(rows):
    """The hash is exactly behavior_trace.document_sha256(RLDataset.index)."""
    candidates = defaultdict(dict)
    counts = Counter()
    for row in rows:
        extra = row.get('extra_info') or {}
        require(isinstance(extra, dict), 'invalid_extra_info')
        index_hash = document_sha(extra.get('index', 0))
        model = row.get('reward_model')
        require(isinstance(model, dict) and isinstance(model.get('ground_truth'), dict),
                'missing_original_ground_truth')
        gold = model['ground_truth']
        require('target' in gold, 'missing_gold_target')
        targets = gold['target']
        require(isinstance(targets, str) or
                (isinstance(targets, list) and all(isinstance(x, str) for x in targets)),
                'unsupported_gold_target_type')
        candidates[index_hash][document_sha(gold)] = gold
        counts[index_hash] += 1
    return candidates, counts


def unique_gold(mapping, index_hash):
    require(index_hash in mapping, 'missing_index_gold_join')
    require(len(mapping[index_hash]) == 1, 'ambiguous_index_gold_join')
    return next(iter(mapping[index_hash].values()))


def actor_parts(row):
    ids, attn, info, positions = [row[k] for k in
                                ('input_ids', 'attention_mask', 'info_mask', 'position_ids')]
    width = row['prompt_width']
    require(all(isinstance(x, list) for x in (ids, attn, info, positions)), 'invalid_actor_arrays')
    require(len(ids) == len(attn) == len(info) == len(positions) and
            type(width) is int and 0 < width < len(ids), 'invalid_actor_shape')
    require(all(type(x) is int and x >= 0 for x in ids), 'invalid_token_ids')
    require(all(type(x) in (int, bool) and x in (0, 1) for x in attn + info), 'invalid_actor_mask')
    require(attn[:width] == sorted(attn[:width]) and
            attn[width:] == sorted(attn[width:], reverse=True), 'actor_attention_hole')
    require(all(x <= y for x, y in zip(info, attn)), 'info_outside_attention')
    require(info[:width] == attn[:width], 'prompt_info_mask_changed')
    require([p for p, a in zip(positions, attn) if a] == list(range(sum(attn))),
            'actor_active_position_mismatch')
    plen, rlen = sum(attn[:width]), sum(attn[width:])
    require(plen > 0 and rlen > 0, 'empty_prompt_or_response')
    # This reproduces the original last-valid-prompt / first-valid-response slices.
    return ids[width - plen:width], ids[width:width + rlen], info[width:width + rlen]


def verify_native_gate(gate, update, run):
    step, attempt = update['step'], update['attempt_id']
    expected_snapshot = run['run_id'] + ':' + str(step) + ':policy-' + attempt
    require(gate.get('accepted') is True and gate.get('trajectory_count') == 2560,
            'unaccepted_or_incomplete_native_gate')
    require(gate['run_id'] == run['run_id'] and gate['logical_update'] == step,
            'gate_run_or_step_mismatch')
    require(gate['source_sha256'] == run['source_sha256'], 'gate_source_mismatch')
    require(gate['policy_snapshot_id'] == expected_snapshot, 'gate_snapshot_mismatch')
    require(gate.get('attempt_id') == attempt and gate.get('attempt_index') == update['attempt_index'],
            'native_gate_attempt_mismatch')
    require(gate.get('private_trace_sha256') == update['behavior_trace_sha256'] and
            gate.get('trace_sha256') == update['behavior_trace_sha256'], 'trace_sha_chain_mismatch')
    ids = gate['trajectory_ids']
    require(len(ids) == len(set(ids)) == 2560, 'incomplete_or_duplicate_gate_trajectory_ids')
    require(set(gate['upstream_group_sha256']) == set(ids) and
            set(gate['row_tensor_sha256']) == set(ids) and len(gate['actor_token_counts']) == 2560,
            'incomplete_gate_row_bindings')
    hashes = []
    for i, identity in enumerate(ids):
        group = gate['upstream_group_sha256'][identity]
        require(re.fullmatch('[0-9a-f]{64}', group) is not None, 'invalid_parent_index_sha')
        if i % 5:
            require(group == hashes[-1], 'five_sibling_upstream_hash_mismatch')
        parent = str(i // 5) + '-' + group
        expected_id = (run['run_id'] + '-' + str(step) + '-attempt-' + attempt +
                       '-' + parent + '-sample-' + str(i % 5))
        require(identity == expected_id, 'sample_occurrence_mismatch')
        require(type(gate['actor_token_counts'][i]) is int and gate['actor_token_counts'][i] > 0,
                'invalid_native_actor_token_count')
        hashes.append(group)
    return hashes


def stream_actor_rows(path, *, chunk_chars=65536, max_row_json_chars=8 * 1024 * 1024):
    """Decode only the canonical leading actor_rows array with bounded buffering.

    Caller must fresh-hash the entire immutable original trace against its native
    ledger before using this iterator.  This parser does not authenticate the
    remaining trace metadata or token journal by parsing them; the whole-file SHA
    and native gate do that.  Every streamed row is separately canonical-hashed
    into the native actor_rows SHA by the caller.
    """
    require(type(chunk_chars) is int and 1 <= chunk_chars <= max_row_json_chars,
            'invalid_stream_buffer_limit')
    decoder = json.JSONDecoder()
    prefix = '{"actor_rows":['
    with Path(path).open('r', encoding='utf-8') as stream:
        require(stream.read(len(prefix)) == prefix, 'noncanonical_trace_actor_rows_prefix')
        buffer = ''

        def refill():
            nonlocal buffer
            part = stream.read(chunk_chars)
            require(bool(part), 'truncated_actor_rows_array')
            buffer += part
            require(len(buffer) <= max_row_json_chars, 'actor_row_exceeds_stream_buffer_limit')

        refill()
        if buffer.startswith(']'):
            return
        while True:
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    refill()
            require(isinstance(row, dict), 'actor_row_not_object')
            buffer = buffer[end:]
            yield row
            if not buffer:
                refill()
            if buffer[0] == ',':
                buffer = buffer[1:]
                if not buffer:
                    refill()
                require(not buffer.startswith(']'), 'trailing_actor_row_comma')
                continue
            require(buffer[0] == ']', 'invalid_actor_array_separator')
            buffer = buffer[1:]
            if not buffer:
                refill()
            require(buffer.startswith(','), 'missing_remaining_native_trace_metadata')
            return


def stream_trace_sha(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path,
            'missing_or_symlink_trace')
    before = path.stat()
    actual = file_sha(path)
    after = path.stat()
    require(actual == expected, 'fresh_complete_trace_sha_mismatch')
    require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) ==
            (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev),
            'native_trace_changed_during_SHA_stream')
    return dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns, inode=after.st_ino,
                device=after.st_dev, fresh_full_sha256=actual)


def verify_streamed_actor(row, index, gate):
    require(index < 2560, 'too_many_streamed_actor_rows')
    require(set(row) == {'trajectory_id', 'input_ids', 'attention_mask', 'info_mask',
                         'position_ids', 'prompt_width'}, 'native_actor_row_schema_mismatch')
    require(row['trajectory_id'] == gate['trajectory_ids'][index], 'streamed_actor_ID_order_mismatch')
    prompt, response, mask = actor_parts(row)
    require(sum(mask) == gate['actor_token_counts'][index], 'native_actor_token_count_mismatch')


def verify_actor_array_digest(digest, count, gate):
    require(count == 2560, 'incomplete_streamed_actor_rows')
    require(digest == gate['actor_rows_sha256'], 'streamed_actor_array_sha_mismatch')


def load_pure_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require('torch' not in sys.modules, 'torch_import_forbidden')
    return module


def reconstruct_row(row, gold, tokenizer, signals, qa_em):
    prompt_ids, response_ids, model_mask = actor_parts(row)
    prompt = tokenizer.decode(prompt_ids)
    response, model_ranges = signals.decode_segments(tokenizer, response_ids, model_mask)
    context = signals.contexts(prompt, response, model_ranges)
    answer = context['answer'] if context is not None else None
    correct = int(answer is not None and bool(qa_em.em_check(answer, gold['target'])))
    return dict(trajectory_id=row['trajectory_id'], C_raw=correct,
                final_action_valid=answer is not None,
                model_authored_search_count=signals.model_search_count(response, model_ranges),
                decoded_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                decoded_response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                model_ranges_sha256=document_sha(model_ranges),
                original_online_reward=False, reconstructed_strict_C=True)


def native_em_check(hits, total, printed):
    require(type(hits) is int and type(total) is int and 0 <= hits <= total and total > 0,
            'invalid_C_count')
    require(isinstance(printed, (int, float)) and math.isfinite(printed) and 0 <= printed <= 1,
            'invalid_native_EM')
    exact = hits / total
    difference = abs(exact - printed)
    require(difference <= 0.0005001, 'strict_C_native_EM_mismatch')
    return dict(C_hits_exact=hits, trajectory_count=total, C_mean_exact=exact,
                native_train_EM_printed=printed, absolute_difference=difference,
                native_EM_rounding_tolerance=0.0005001, native_EM_check_pass=True)


def write_json(path, document):
    with Path(path).open('xb') as stream:
        stream.write(canonical(document)); stream.flush(); os.fsync(stream.fileno())


def run(args):
    require(sys.platform.startswith('linux') and socket.gethostname() == args.expected_node,
            'execution_is_not_bound_remote_linux_node')
    require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'CPU_only_environment_not_established')
    require(1 <= args.start_step <= args.end_step <= 150, 'scope_outside_TRAIN_1_to_150')
    require(file_sha(__file__) == args.helper_sha256, 'unpublished_helper_bytes')
    base = Path(args.base).absolute()
    require(base.resolve() == base, 'symlink_experiment_base')
    run_record = read_bound(base / 'logs/run-manifest.json', args.run_manifest_sha256)
    require(run_record['run_id'] == args.expected_run_id and
            run_record.get('policy_safety_mode') == 'inline_first_five', 'unsupported_original_run')
    require((base / 'logs/run-manifest.json').stat().st_uid == os.getuid(), 'foreign_experiment_owner')
    cfgproof = read_bound(base / 'logs/resolved-training-config.json', args.resolved_config_sha256)
    config = cfgproof['resolved_config']
    # The original configuration wrapper canonicalization uses ensure_ascii=True.
    resolved_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(cfgproof['run_id'] == run_record['run_id'] and resolved_sha == cfgproof['resolved_config_sha256'],
            'resolved_configuration_binding_mismatch')
    require(config['data']['train_batch_size'] == 512 and
            config['actor_rollout_ref']['rollout']['n_agent'] == 5 and
            config['actor_rollout_ref']['rollout']['n'] == 1, 'TRAIN_512x5_contract_mismatch')
    audit = read_bound(args.gold_audit, args.gold_audit_sha256)
    require(audit.get('readonly') is True and audit['run_id'] == run_record['run_id'] and
            audit['training_gold_identity'].get('gold_mapping_from_persisted_hashed_index_unique') is True,
            'metadata_gold_join_not_yet_accepted')
    require(audit['source_commit'] == run_record['source_commit'] and
            audit.get('committed_trace_shape_claim_verified_by_fresh_native_gates') is True,
            'metadata_audit_source_or_complete_gate_claim_mismatch')
    require(audit['resolved_configuration_identity']['fresh_file_sha256'] == args.resolved_config_sha256,
            'metadata_audit_configuration_mismatch')

    repo = base / 'repo'
    required_sources = ['search_g1/signals.py', 'search_r1/qa_protocol.py',
                        'verl/utils/reward_score/qa_em.py', 'verl/utils/tokenizer.py',
                        'verl/utils/dataset/rl_dataset.py', 'verl/trainer/main_ppo_eco.py',
                        'reproduction/common8/behavior_runtime.py', 'reproduction/common8/behavior_trace.py',
                        'verl/trainer/ppo/ray_trainer.py']
    sources = {}
    for relative in required_sources:
        expected = run_record['source_sha256'].get(relative)
        require(expected and file_sha(repo / relative) == expected, 'frozen_scoring_source_sha_mismatch')
        sources[relative] = expected
    sys.path.insert(0, str(repo))
    qa_em = load_pure_module('g1_history_strict_em', repo / 'verl/utils/reward_score/qa_em.py')
    signals = load_pure_module('g1_history_signals', repo / 'search_g1/signals.py')
    hf_utils = load_pure_module('g1_history_tokenizer', repo / 'verl/utils/tokenizer.py')
    import pyarrow.parquet as pq
    train_paths = config['data']['train_files']
    train_paths = [train_paths] if isinstance(train_paths, str) else train_paths
    require(train_paths and len({Path(x).name for x in train_paths}) == len(train_paths), 'ambiguous_train_paths')
    mapping, multiplicity, data_proofs = {}, Counter(), []
    for name in train_paths:
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'invalid_train_file_path')
        expected = run_record['input_sha256'].get(path.name)
        before = file_sha(path)
        require(expected and before == expected, 'original_TRAIN_data_sha_mismatch')
        row_count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,
                                                     columns=['extra_info', 'reward_model']):
            rows = batch.to_pylist()
            row_count += len(rows)
            candidates, counts = gold_mapping(rows)
            for index, values in candidates.items():
                mapping.setdefault(index, {}).update(values)
            multiplicity.update(counts)
            del rows, candidates, counts
        require(file_sha(path) == before, 'TRAIN_data_changed_during_read')
        data_proofs.append(dict(path=str(path), rows=row_count, fresh_sha256=before,
                                original_run_sha256=expected, exact_original_data_verified=True))

    tokenizer_path = Path(run_record['initial_model'])
    require(config['actor_rollout_ref']['model']['path'] == str(tokenizer_path), 'initial_tokenizer_model_path_mismatch')
    tokenizer_proofs = {}
    for name, expected in run_record['initial_model_sha256'].items():
        if name.endswith(('.bin', '.pt', '.safetensors')):
            continue
        require(file_sha(tokenizer_path / name) == expected, 'original_tokenizer_asset_sha_mismatch')
        tokenizer_proofs[name] = expected
    require({'tokenizer.json', 'tokenizer_config.json'}.issubset(tokenizer_proofs), 'missing_tokenizer_manifest_files')
    # Tokenizer-only Transformers code is explicitly told not to import a DL backend.
    os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    tokenizer = hf_utils.hf_tokenizer(str(tokenizer_path), local_files_only=True)
    require('torch' not in sys.modules and tokenizer.is_fast, 'non_CPU_or_unexpected_tokenizer')
    import transformers
    import tokenizers
    decoder_identity = dict(tokenizer_class=type(tokenizer).__name__,
                            transformers_version=transformers.__version__,
                            tokenizers_version=tokenizers.__version__,
                            clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                            decode_kwargs='original defaults; no skip_special_tokens override')

    ledger_path = base / 'logs/actual-updates.jsonl'
    ledger_raw = ledger_path.read_bytes()
    require(ledger_raw.endswith(b'\n'), 'ledger_concurrent_partial_last_record')
    ledger = [json.loads(line) for line in ledger_raw.splitlines()]
    require(all(row['run_id'] == run_record['run_id'] for row in ledger), 'foreign_update_ledger_row')
    require([row['step'] for row in ledger] == list(range(1, len(ledger) + 1)), 'noncontiguous_committed_update_ledger')
    updates = [row for row in ledger if args.start_step <= row['step'] <= args.end_step]
    require([row['step'] for row in updates] == list(range(args.start_step, args.end_step + 1)),
            'requested_TRAIN_updates_not_committed')
    audited_updates = {row['step']: row for row in audit['trajectory_coverage']}
    require(len(audited_updates) == len(audit['trajectory_coverage']), 'duplicate_metadata_audit_step')
    for update in updates:
        row = audited_updates.get(update['step'])
        require(row is not None and row['attempt_id'] == update['attempt_id'] and
                row['probe_version'] == update['probe_version'] and
                row['native_gate_fresh_sha256'] == update['behavior_probability_gate_sha256'] and
                row['trace_sha256_bound_native_record'] == update['behavior_trace_sha256'],
                'selected_update_not_bound_to_accepted_metadata_audit')
    log_path = Path(run_record['train_log'])
    native, log_sha = {}, hashlib.sha256()
    with log_path.open('rb') as stream:
        for raw in stream:
            log_sha.update(raw)
            text = raw.decode(errors='replace')
            match = re.search(r'\bstep:(\d+)\s*-', text)
            if match and args.start_step <= int(match[1]) <= args.end_step:
                em = re.search(r'\btrain/perf/em:(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', text)
                if em:
                    require(int(match[1]) not in native, 'duplicate_native_TRAIN_scalar_step')
                    native[int(match[1])] = dict(em=float(em[1]), line_sha256=hashlib.sha256(raw).hexdigest())
    require(set(native) == {row['step'] for row in updates}, 'native_TRAIN_EM_rows_missing')

    out = Path(args.diagnostic_base).absolute()
    require(out.resolve() == out and out != base and not out.is_relative_to(base) and
            not base.is_relative_to(out) and out.parent.is_dir() and not out.exists(),
            'diagnostic_path_is_not_fresh_independent_remote_directory')
    out.mkdir(mode=0o700)
    write_json(out / 'inputs.json', dict(run_id=run_record['run_id'], helper_sha256=args.helper_sha256,
        sources=sources, original_train_data=data_proofs, initial_tokenizer_path=str(tokenizer_path),
        tokenizer_asset_fresh_sha256=tokenizer_proofs, decoder_identity=decoder_identity,
        model_weight_hashes_not_read=True, bounded_stream_version=2,
        original_gate_scope='sampling/context/LP proof only, not original reward proof',
        metadata_gold_audit_sha256=args.gold_audit_sha256,
        committed_ledger_read_sha256=hashlib.sha256(ledger_raw).hexdigest(),
        native_log_stream_read_sha256=log_sha.hexdigest()))
    summaries = []
    for update in updates:
        aid = update['attempt_id']
        require(re.fullmatch('[0-9a-f]{32}', aid) is not None, 'invalid_committed_attempt_id')
        gate_path = base / 'logs' / ('behavior-probability-attempt-' + aid + '.json')
        trace_path = base / 'artifacts/behavior-traces' / ('attempt-' + aid + '.json')
        gate = read_bound(gate_path, update['behavior_probability_gate_sha256'])
        hashes = verify_native_gate(gate, update, run_record)
        trace_binding = stream_trace_sha(trace_path, update['behavior_trace_sha256'])
        target = out / ('TRAIN-' + str(update['step']) + '-raw-C.jsonl')
        hits = invalid = searched = 0
        digest = hashlib.sha256()
        actor_digest = hashlib.sha256(b'[')
        row_count = 0
        with target.open('xb') as stream:
            for index, row in enumerate(stream_actor_rows(trace_path)):
                verify_streamed_actor(row, index, gate)
                if index:
                    actor_digest.update(b',')
                actor_digest.update(canonical(row))
                group = hashes[index]
                value = reconstruct_row(row, unique_gold(mapping, group), tokenizer, signals, qa_em)
                value.update(run_id=run_record['run_id'], step=update['step'], attempt_id=aid,
                    policy_snapshot_id=gate['policy_snapshot_id'], upstream_group_sha256=group,
                    probe_version_for_future_R_replay=update['probe_version'])
                raw = canonical(value) + b'\n'; stream.write(raw); digest.update(raw)
                hits += value['C_raw']; invalid += not value['final_action_valid']
                searched += value['model_authored_search_count'] > 0
                row_count += 1
            stream.flush(); os.fsync(stream.fileno())
        actor_digest.update(b']')
        verify_actor_array_digest(actor_digest.hexdigest(), row_count, gate)
        after = trace_path.stat()
        require((after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) ==
                (trace_binding['bytes'], trace_binding['mtime_ns'], trace_binding['inode'], trace_binding['device']),
                'native_trace_changed_during_C_stream')
        summary = native_em_check(hits, 2560, native[update['step']]['em'])
        summary.update(step=update['step'], attempt_id=aid, native_scalar_line_sha256=native[update['step']]['line_sha256'],
            probe_version_for_future_R_replay=update['probe_version'], invalid_final_action_count=invalid,
            searched_trajectory_count=searched, raw_C_remote_path=str(target), raw_C_fresh_sha256=digest.hexdigest(),
            native_gate_fresh_sha256=update['behavior_probability_gate_sha256'],
            complete_trace_fresh_sha256=update['behavior_trace_sha256'],
            actor_rows_streamed_canonical_sha256=actor_digest.hexdigest(), bounded_stream_version=2,
            remaining_trace_token_journal_not_materialized=True,
            independent_C_reconstruction=True, shaped_R_replayed=False, Pearson_computed=False)
        summaries.append(summary)
        write_json(out / ('TRAIN-' + str(update['step']) + '-summary.json'), summary)
        del gate, hashes
    for relative, expected in sources.items():
        require(file_sha(repo / relative) == expected, 'frozen_source_changed_during_preflight')
    for name, expected in tokenizer_proofs.items():
        require(file_sha(tokenizer_path / name) == expected, 'tokenizer_assets_changed_during_preflight')
    require('torch' not in sys.modules, 'torch_import_forbidden')
    result = dict(status='ACCEPTED_STRICT_C_PREFLIGHT_ONLY', CPU_only=True, torch_imported=False,
        run_id=run_record['run_id'], observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        committed_iterations=len(summaries), total_exact_C_rows=len(summaries) * 2560,
        C_hits_total=sum(row['C_hits_exact'] for row in summaries), iterations=summaries,
        diagnostic_remote_path=str(out), shaped_R_replayed=False, Pearson_computed=False,
        no_new_generation=True, no_search_or_service_calls=True, original_training_unmodified=True)
    write_json(out / 'accepted-summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'diagnostic-base', 'expected-run-id', 'run-manifest-sha256',
                 'resolved-config-sha256', 'gold-audit', 'gold-audit-sha256',
                 'helper-sha256', 'expected-node'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--start-step', type=int, default=1)
    parser.add_argument('--end-step', type=int, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        result = dict(status='HOLD', reason=str(error) if isinstance(error, Hold) else type(error).__name__,
                      raw_private_exception_text_not_exported=True, Pearson_computed=False,
                      diagnostic_partial_must_be_preserved=True)
        print(json.dumps(result, ensure_ascii=False)); return 2
    print(json.dumps(result, ensure_ascii=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
