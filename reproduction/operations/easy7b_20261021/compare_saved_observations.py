"""Compare saved actual GPU4 actor-visible observations with single-query GPU2 HTTP.

Only native raw tensors are read on CPU.  The original official tokenizer is
loaded locally, never an actor model.  Each new HTTP request contains exactly
one saved actual query.  Source GPU4 performs no new work.  This sampled test
compares the exact retained observation token IDs, not unavailable full old
HTTP documents/scores or all future queries.  Query/token text is never output.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True


class InsufficientEvidence(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise InsufficientEvidence(reason)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while part := stream.read(1024 * 1024):
            value.update(part)
    return value.hexdigest()


def raw_metadata(path):
    value = path.stat()
    return dict(bytes=value.st_size, device=value.st_dev, inode=value.st_ino,
                mtime_ns=value.st_mtime_ns)


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def format_original_passages(documents):
    # Exact LLMGenerationManager._passages2string implementation.  The caller
    # applies the original result.strip() before wrapping <information>.
    value = ''
    for index, item in enumerate(documents):
        content = item['document']['contents']
        title = content.split('\n')[0]
        text = '\n'.join(content.split('\n')[1:])
        value += f'Doc {index+1}(Title: {title}) {text}\n'
    return value


def choose_cases(output, maximum):
    ids = output.non_tensor_batch['behavior_trajectory_id'].tolist()
    trajectories = output.non_tensor_batch['behavior_turns'].tolist()
    actual_counts = output.meta_info['actual_retrieval_stats']
    need(len(ids) == len(trajectories) == len(actual_counts) == 2560,
         'expected_one_complete_original_512_by5_chunk')
    cases, seen = [], set()
    skipped_empty = 0
    for row, (tid, turns, actual_count) in enumerate(zip(ids, trajectories, actual_counts)):
        need(type(turns) is list and 1 <= len(turns) <= 5, 'native_turns_incomplete')
        actual_searches = 0
        candidates = []
        for index, turn in enumerate(turns):
            need(turn.get('trajectory_id') == tid and turn.get('turn_index') == index and
                 type(turn.get('final_turn')) is bool, 'native_turn_identity_differs')
            cut = turn.get('prefix_cut')
            need(type(cut) is dict and type(cut.get('execution_text')) is str,
                 'native_executor_text_missing')
            match = re.search(r'<(search|answer)>(.*?)</\1>', cut['execution_text'], re.DOTALL)
            if match is None or match.group(1) != 'search' or turn['final_turn']:
                continue
            actual_searches += 1
            tokens = turn.get('observation_token_ids')
            need(type(tokens) is list and all(type(token) is int for token in tokens),
                 'saved_actual_observation_tokens_missing')
            if not tokens:
                skipped_empty += 1
                continue
            request = turn.get('actual_request_token_ids')
            kept = turn.get('kept_token_count')
            need(type(request) is list and request and type(kept) is int and
                 cut.get('kept_token_count') == kept and len(cut.get('kept_token_ids', [])) == kept,
                 'saved_native_request_or_kept_token_length_missing')
            candidates.append(dict(query=match.group(2).strip(), old_tokens=tokens,
                request_length=len(request), kept_length=kept,
                trajectory_id_sha256=sha(tid), turn_index=index, row_index=row))
        need(type(actual_count) is int and actual_searches == actual_count,
             'saved_row_actual_retrieval_count_differs')
        # One case per distinct trajectory, and no repeated query in the sample.
        for candidate in candidates:
            if candidate['query'] not in seen:
                seen.add(candidate['query']); cases.append(candidate)
                break
        if len(cases) == maximum:
            break
    need(bool(cases), 'no_nonempty_saved_actual_observation_cases')
    return cases, dict(scanned_trajectories=row + 1, skipped_empty_observation_events=skipped_empty,
                      distinct_query_count=len(cases), distinct_trajectory_count=len(cases))


def health(session, endpoint, ready):
    response = session.get(endpoint.rsplit('/', 1)[0] + '/health', timeout=(10, 30))
    response.raise_for_status()
    value = response.json()
    need(value.get('ok') is True and
         all(value.get(key) == ready.get(key) for key in
             ('service_job_id', 'service_run_id', 'source_commit', 'index_sha256',
              'corpus_sha256', 'encoder_manifest_sha256')) and
         value.get('physical_gpu_count') == 2 and value.get('logical_shards') == 4 and
         value.get('exact_query_cache_enabled') is False,
         'live_service_identity_or_original_four_logical_shards_differ')
    return dict(service_job_id=value['service_job_id'], observed_unix=value.get('observed_unix'),
                statistics=value.get('statistics'))


def run(args):
    raw = args.raw_pool_file.resolve(strict=True)
    output_path = args.output.resolve()
    need(raw.name in ('0.pt', '512.pt', '1024.pt', '1536.pt', '2048.pt') and
         raw.parents[1].name == 'mixed-candidate-pools' and raw.parents[2].name == 'artifacts',
         'explicit_original_raw_pool_chunk_path_required')
    training_base = raw.parents[3]
    need(not output_path.is_relative_to(training_base), 'audit_output_must_be_outside_source_training_directory')
    ready = json.loads(args.service_ready.read_text())
    need(ready.get('ready') is True and ready.get('physical_gpu_count') == 2 and
         ready.get('logical_shards') == 4 and ready.get('logical_to_physical_gpu') == [0, 1, 0, 1] and
         ready.get('exact_query_cache_enabled') is False, 'two_gpu_original_shard_service_not_ready')
    result = dict(schema_version=1, role='actual_saved_gpu4_observation_vs_gpu2_single_query',
        status='LOADING_CPU_EVIDENCE', checked_unix=time.time(), completed_cases=0,
        requested_max_cases=args.max_cases, raw_pool_file=str(raw), raw_metadata=raw_metadata(raw),
        service_ready_path=str(args.service_ready), service_ready_sha256=file_sha(args.service_ready),
        service_job_id=ready['service_job_id'], service_run_id=ready['service_run_id'],
        service_source_commit=ready['source_commit'], endpoint=ready['endpoint'],
        request_shape='queries=[one_original_actual_query],topk=3,return_scores=true',
        selection='first distinct actual nonfinal search per distinct original trajectory; unique queries',
        old_gpu_requests=0, actor_models_loaded=0, generation_performed=False,
        raw_source_modified=False, query_or_token_text_written=False, tokenizer_only=True,
        scope='SAMPLED_EXACT_ACTOR_VISIBLE_OBSERVATION_TOKENS_ONLY_NOT_FULL_OLD_HTTP_DOCUMENTS_OR_FUTURE_ALL_QUERIES',
        old_full_document_ids_or_scores_available=False, cases=[])
    publish(output_path, result)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import requests
    from transformers import AutoTokenizer
    from reproduction.common8.context_budget import observation_room, observation_tokens
    need(not torch.cuda.is_initialized(), 'cuda_initialized_in_cpu_audit')
    before = raw_metadata(raw)
    with raw.open('rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    need(before == raw_metadata(raw), 'raw_source_changed_during_cpu_load')
    identity, output = saved['identity'], saved['output']
    source_run = json.loads((training_base / 'logs/run-manifest.json').read_text())
    official = json.loads((REPO / 'reproduction/easy2560_joint4/official7b.json').read_text())
    need(identity['run_id'] == source_run['run_id'] and source_run.get('model_revision') == official['revision'],
         'source_is_not_the_bound_official_7b_candidate_generation')
    budget = output.meta_info['behavior_context_policy']
    need(budget.get('total_tokens') == 4096 and budget.get('answer_reserve_tokens') == 500 and
         budget.get('action_tokens') == 500, 'original_context_budget_differs')
    cases, selection = choose_cases(output, args.max_cases)
    tokenizer_proofs = []
    for entry in official['files']:
        if entry['file'] in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
            path = args.tokenizer / entry['file']
            need(path.stat().st_size == entry['bytes'] and file_sha(path) == entry['sha256'],
                 'original_official_7b_tokenizer_files_differ')
            tokenizer_proofs.append(dict(file=entry['file'], sha256=entry['sha256']))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True,
                                             trust_remote_code=False, padding_side='left')
    result.update(status='COMPARING_SINGLE_QUERY_HTTP', checked_unix=time.time(),
        actual_source_run_id=identity['run_id'], outer=identity['logical_update'],
        attempt_id=identity.get('attempt_id'), source_sampling_slurm_job_id=identity['slurm_job_id'],
        source_chunk_complete_trajectories=2560, source_chunk_question_ordinals=saved['ordinals'],
        saved_context_budget=budget, max_observation_tokens=500, sample_selection=selection,
        sample_count=len(cases), tokenizer_files=tokenizer_proofs, cuda_initialized=False)
    del saved, output
    gc.collect()
    publish(output_path, result)
    with requests.Session() as session:
        session.trust_env = False
        result['service_health_before'] = health(session, ready['endpoint'], ready)
        for number, case in enumerate(cases, 1):
            started = time.monotonic()
            response = session.post(ready['endpoint'], json=dict(queries=[case['query']], topk=3,
                return_scores=True), timeout=(10, 120))
            response.raise_for_status()
            payload = response.json()
            need(type(payload.get('result')) is list and len(payload['result']) == 1 and
                 type(payload['result'][0]) is list and len(payload['result'][0]) == 3,
                 'single_query_original_http_shape_differs')
            documents = payload['result'][0]
            need(all(type(item.get('document')) is dict and
                     type(item['document'].get('contents')) is str and
                     type(item.get('score')) in (float, int) and math.isfinite(item['score'])
                     for item in documents), 'service_document_or_score_schema_differs')
            text = '\n\n<information>' + format_original_passages(documents).strip() + '</information>\n\n'
            room = observation_room(case['request_length'], case['kept_length'], budget)
            new, clipped = observation_tokens(tokenizer, text, min(500, room))
            old = case['old_tokens']
            differences = sum(a != b for a, b in zip(old, new)) + abs(len(old) - len(new))
            result['cases'].append(dict(case=number, trajectory_id_sha256=case['trajectory_id_sha256'],
                row_index=case['row_index'], turn_index=case['turn_index'],
                actual_request_tokens=case['request_length'], kept_policy_tokens=case['kept_length'],
                available_observation_room=room, effective_observation_limit=min(500, room),
                old_visible_tokens=len(old), new_visible_tokens=len(new), new_observation_clipped=clipped,
                exact_token_list_equal=(old == new), differing_token_positions_plus_length_gap=differences,
                old_visible_observation_sha256=sha(old), new_visible_observation_sha256=sha(new),
                new_returned_documents_sha256=[sha(item['document']) for item in documents],
                http_and_projection_seconds=time.monotonic() - started))
            result.update(completed_cases=number, exact_match_cases=sum(row['exact_token_list_equal'] for row in result['cases']),
                maximum_differing_token_positions=max(row['differing_token_positions_plus_length_gap'] for row in result['cases']),
                checked_unix=time.time())
            if number % 8 == 0 or number == len(cases):
                result['service_health_latest'] = health(session, ready['endpoint'], ready)
                publish(output_path, result)
                print(json.dumps(dict(status=result['status'], completed_cases=number,
                    exact_match_cases=result['exact_match_cases'], sample_count=len(cases))), flush=True)
    need(raw_metadata(raw) == before, 'raw_source_changed_during_observation_audit')
    need(not torch.cuda.is_initialized(), 'cuda_initialized_during_cpu_audit')
    result.update(status='COMPLETE', checked_unix=time.time(),
        all_sampled_actor_visible_observations_exact=(result['exact_match_cases'] == len(cases)),
        requested_sample_count_reached=(len(cases) == args.max_cases),
        raw_source_metadata_unchanged=True, cuda_initialized=False,
        http_and_projection_total_seconds=sum(row['http_and_projection_seconds'] for row in result['cases']))
    publish(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-pool-file', required=True, type=Path)
    parser.add_argument('--service-ready', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-cases', type=int, default=64)
    args = parser.parse_args()
    need(all(path.is_absolute() for path in (args.raw_pool_file, args.service_ready, args.tokenizer, args.output))
         and 1 <= args.max_cases <= 2560, 'absolute_paths_and_valid_max_cases_required')
    need(len(args.raw_pool_file.parents) >= 4 and
         not args.output.resolve().is_relative_to(args.raw_pool_file.parents[3].resolve()),
         'audit_output_must_be_outside_source_training_directory')
    try:
        result = run(args)
    except Exception as error:
        value = dict(status='INSUFFICIENT_EVIDENCE_OR_REQUEST_FAILURE', checked_unix=time.time(),
            reason=str(error) if isinstance(error, InsufficientEvidence) else type(error).__name__,
            query_or_token_text_written=False, old_gpu_requests=0)
        if args.output.exists():
            value['partial_progress'] = json.loads(args.output.read_text())
        publish(args.output, value)
        print(json.dumps(dict(status=value['status'], reason=value['reason'])), flush=True)
        return 2
    print(json.dumps(dict(status=result['status'], completed_cases=result['completed_cases'],
        exact_match_cases=result['exact_match_cases'], maximum_differing_token_positions=result['maximum_differing_token_positions'])), flush=True)
    return 0 if result['all_sampled_actor_visible_observations_exact'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
