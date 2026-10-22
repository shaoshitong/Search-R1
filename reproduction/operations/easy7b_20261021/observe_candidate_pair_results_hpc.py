"""Read completed candidate A/B results and actual private service counters."""
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def observe(root):
    runs = Path(root) / 'runs'
    result = dict(epoch=time.time(), root=str(root), variants={}, runner=read(runs / 'progress.json'))
    keys = ('status', 'completed_questions', 'completed_trajectories', 'generation_phase_wall_seconds',
        'pipeline_seconds_including_input_io_score_and_output_save', 'actual_raw_generated_tokens',
        'actual_generation_requests', 'actual_prompt_tokens', 'actual_retrieval_calls',
        'actual_fa3_prefill_and_decode', 'input_fingerprints_sha256', 'selection')
    for name in ('startup-legacy', 'startup-async', 'full-legacy', 'full-async'):
        directory = runs / name
        data = read(directory / 'candidate/result.json')
        execution = read(directory / 'execution.json')
        item = dict(candidate_complete=data is not None, progress=read(directory / 'candidate/progress.json'))
        if data is not None:
            item['result'] = {key: data.get(key) for key in keys}
            if item['result'].get('selection'):
                item['result']['selection'].pop('selected_ordinals', None)
        if execution is not None:
            delta = execution.get('service_numeric_statistics_delta', {})
            attempts = delta.get('batch_attempts', 0)
            item.update(candidate_exit_code=execution.get('candidate_exit_code'),
                service_counter_delta=delta,
                actual_mean_faiss_batch=delta.get('batch_input_queries', 0) / attempts if attempts else None,
                proxy=execution.get('proxy_health_after', {}).get('response'),
                service_statistics_after=execution.get('service_health_after', {}).get('response', {}).get('statistics'))
        result['variants'][name] = item
    return result


if __name__ == '__main__':
    print(json.dumps(observe(json.load(sys.stdin)['root']), sort_keys=True), flush=True)
