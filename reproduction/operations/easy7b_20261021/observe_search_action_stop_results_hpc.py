"""Read existing search-only stop A/B JSON; stdin is {"root": ".../gpu-pair-r1"}.

No tensor loading, HTTP, subprocesses, GPU calls or writes. Timings retain the
producer's candidate-chunk scope and do not describe an entire training outer.
"""
import json
from pathlib import Path
import sys
import time


RESULT_KEYS = (
    'status', 'actual_world_size', 'completed_chunks', 'completed_questions',
    'completed_trajectories', 'generation_phase_wall_seconds',
    'pipeline_seconds_including_input_io_score_and_output_save',
    'actual_raw_generated_tokens', 'actual_generation_requests',
    'actual_prompt_tokens', 'actual_retrieval_calls',
    'actual_fa3_prefill_and_decode', 'input_fingerprints_sha256',
)
CHUNK_KEYS = (
    'chunk', 'questions', 'trajectories', 'generation_seconds',
    'load_parent_seconds', 'strict_score_seconds', 'actual_raw_generated_tokens',
    'retained_model_tokens', 'actual_generation_requests', 'actual_prompt_tokens',
    'actual_retrieval_calls', 'mixed_groups', 'correct_trajectories',
    'strict_correctness_sha256',
)


def read(path, errors):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        errors.append(dict(path=str(path), error_type=type(error).__name__, error=str(error)))
        return None


def selection(value):
    return ({key: item for key, item in value.items() if key != 'selected_ordinals'}
            if isinstance(value, dict) else value)


def sum_complete(rows, key):
    values = [row.get(key) for row in rows]
    return sum(values) if values and all(type(v) in (int, float) for v in values) else None


def variant(directory, errors):
    candidate = directory / 'candidate'
    result = read(candidate / 'result.json', errors)
    execution = read(directory / 'execution.json', errors)
    failure = read(candidate / 'failure.json', errors)
    chunks, ranks = [], {}
    paths = sorted(candidate.glob('chunk-*.json'),
                   key=lambda path: int(path.stem.split('-')[-1]))
    for path in paths:
        row = read(path, errors)
        if not isinstance(row, dict):
            continue
        chunks.append({key: row.get(key) for key in CHUNK_KEYS})
        for record in (row.get('phase') or {}).get('phase_metrics', []):
            rank = record.get('rank')
            if type(rank) is not int:
                errors.append(dict(path=str(path), error='Phase metric has no integer rank'))
                continue
            calls = (record.get('flash3_execution') or {}).get('calls', {})
            stop = record.get('search_action_stop')
            ranks[str(rank)] = dict(
                rank=rank, latest_completed_chunk=row.get('chunk'),
                flash3_calls=calls,
                actual_fa3_prefill_and_decode=(calls.get('prefill', 0) > 0 and
                                               calls.get('decode', 0) > 0),
                actual_work_current_phase=record.get('actual_work'),
                search_action_stop=stop,
                confirmed_search_stops=(stop.get('confirmed_search_stops')
                                        if isinstance(stop, dict) else None),
            )
    result = result if isinstance(result, dict) else None
    execution = execution if isinstance(execution, dict) else {}
    item = dict(
        candidate_result_present=result is not None,
        candidate_exit_code=execution.get('candidate_exit_code'),
        candidate_process_wall_seconds=execution.get('candidate_process_wall_seconds'),
        progress=read(candidate / 'progress.json', errors),
        failure=failure,
        execution_error={key: execution[key] for key in ('error_type', 'error') if key in execution},
        result=({key: result.get(key) for key in RESULT_KEYS} if result else None),
        chunks=chunks, per_rank=ranks,
        completed_chunk_questions=sum_complete(chunks, 'questions'),
        completed_chunk_trajectories=sum_complete(chunks, 'trajectories'),
        completed_chunk_raw_generated_tokens=sum_complete(chunks, 'actual_raw_generated_tokens'),
        completed_chunk_retained_model_tokens=sum_complete(chunks, 'retained_model_tokens'),
        rank_counter_scope='Latest completed chunk per rank; FA3/stop counters are cumulative '
            'engine counters and are not summed across chunks. actual_work is that phase only.',
        all4_actual_fa3_prefill_and_decode=(
            set(ranks) == {'0', '1', '2', '3'} and
            all(row['actual_fa3_prefill_and_decode'] for row in ranks.values())),
    )
    if result:
        item['result']['selection'] = selection(result.get('selection'))
    item['actual_512_by_5_complete'] = bool(result and
        result.get('status') == 'REAL_CANDIDATE_PIPELINE_COMPLETED' and
        result.get('actual_world_size') == 4 and result.get('completed_chunks') == 1 and
        result.get('completed_questions') == 512 and result.get('completed_trajectories') == 2560 and
        item['completed_chunk_questions'] == 512 and item['completed_chunk_trajectories'] == 2560)
    item['candidate_completed_and_exit_zero'] = bool(
        item['actual_512_by_5_complete'] and item['candidate_exit_code'] == 0)
    return item


def observe(root):
    root = Path(root)
    runs, errors = root / 'runs', []
    result = read(runs / 'result.json', errors)
    failure = read(runs / 'failure.json', errors)
    comparison = dict(result.get('comparison', {})) if isinstance(result, dict) else None
    if comparison is not None:
        for key in ('baseline_selection', 'early_stop_selection'):
            if key in comparison:
                comparison[key] = selection(comparison[key])
    variants = {name: variant(runs / name, errors) for name in ('baseline', 'early-stop')}
    return dict(
        schema='search_action_stop_read_only_results_v1', epoch=time.time(), root=str(root),
        read_only=True, tensor_files_read=False, http_called=False, subprocesses_created=False,
        production_modified=False,
        status=(result.get('status') if isinstance(result, dict) else
                'FAILED' if failure else 'NO_FINAL_PAIR_RESULT_YET'),
        slurm_job_id=result.get('slurm_job_id') if isinstance(result, dict) else None,
        runner=read(runs / 'progress.json', errors), runner_failure=failure,
        comparison=comparison, variants=variants,
        both_actual_512_by_5_completed_and_exit_zero=all(
            row['candidate_completed_and_exit_zero'] for row in variants.values()),
        both_all4_actual_fa3_prefill_and_decode=all(
            row['all4_actual_fa3_prefill_and_decode'] for row in variants.values()),
        observation_errors=errors,
        timing_scope='Producer candidate generation/pipeline/process timings retained unchanged. '
            'One512-question x5-trajectory chunk per arm; not a full training outer. '
            'Search-only stopping can change raw work and asynchronous RNG consumption; '
            'no final-quality equivalence or full-outer causal speedup is established.',
        covers_entire_training_outer=False, causal_full_outer_speedup_proven=False,
    )


if __name__ == '__main__':
    print(json.dumps(observe(json.load(sys.stdin)['root']), sort_keys=True), flush=True)
