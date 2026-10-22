"""Read a saved local observer JSON; summarize real candidate A/B measurements.

No network, GPU, model, tensor or training operations. Full here means the
2560-parent candidate pool, never an entire training outer or final QA quality.
"""
import argparse
import json
from pathlib import Path

WORK = ('actual_raw_generated_tokens', 'actual_prompt_tokens',
        'actual_generation_requests', 'actual_retrieval_calls')
TIMES = ('generation_phase_wall_seconds',
         'pipeline_seconds_including_input_io_score_and_output_save')


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def summarize(document):
    transport = []
    while 'variants' not in document and isinstance(document.get('stdout'), str):
        transport.append({key: document.get(key) for key in ('returncode', 'stderr')})
        document = json.loads(document['stdout'])
    variants = {}
    for name, item in document.get('variants', {}).items():
        result, proxy = item.get('result') or {}, item.get('proxy') or {}
        cache, delta = proxy.get('cache') or {}, item.get('service_counter_delta') or {}
        variants[name] = dict(candidate_complete=item.get('candidate_complete', False),
            candidate_exit_code=item.get('candidate_exit_code'),
            **{key: result.get(key) for key in ('status', 'completed_questions',
                'completed_trajectories', 'actual_fa3_prefill_and_decode',
                'input_fingerprints_sha256', 'selection', *TIMES, *WORK)},
            actual_mean_faiss_batch=item.get('actual_mean_faiss_batch'),
            faiss_batch_attempts=delta.get('batch_attempts'),
            faiss_batch_input_queries=delta.get('batch_input_queries'),
            proxy=dict(logical_requests=proxy.get('logical_requests'),
                backend_requests=proxy.get('backend_requests'), cache_hits=proxy.get('cache_hits'),
                cache_hit_fraction=cache.get('served_hit_rate'),
                failed_requests=proxy.get('failed_requests'),
                backend_exceptions=cache.get('backend_exceptions')))
    pairs = {}
    for scope in ('startup', 'full'):
        a, b = (variants.get(scope + '-' + mode, {}) for mode in ('legacy', 'async'))
        complete = all(row.get('candidate_complete') is True and row.get('candidate_exit_code') == 0
            and row.get('status') == 'REAL_CANDIDATE_PIPELINE_COMPLETED' for row in (a, b))
        pair = dict(both_complete=complete,
            scope='512-parent startup candidates' if scope == 'startup' else '2560-parent full candidate pool',
            covers_entire_training_outer=False)
        if complete:
            pair.update(same_input_fingerprints=a['input_fingerprints_sha256'] == b['input_fingerprints_sha256'],
                timings={key: dict(legacy_seconds=a[key], async_seconds=b[key],
                    observed_legacy_over_async_ratio=ratio(a[key], b[key]),
                    observed_wall_reduction_percent=(100 * (1 - b[key] / a[key]) if a[key] else None))
                    for key in TIMES},
                async_over_legacy_work_ratios={key: ratio(b[key], a[key]) for key in WORK},
                work_counts_equal=all(a[key] == b[key] for key in WORK),
                causal_speedup_proven=False,
                interpretation='Observed candidate-stage timing. Same parents/policy can produce different '
                    'stochastic asynchronous trajectories; assess work ratios alongside wall reduction.')
        pairs[scope] = pair
    return dict(source_root=document.get('root'), observed_epoch=document.get('epoch'),
        transport_exits=transport, runner=document.get('runner'), variants=variants, pairs=pairs,
        scope='Independent candidate generation, retrieval and selection only; no full training-outer '
            'speedup or final-quality equivalence proven. Missing batch counters remain null.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Saved outer JSON from easy7b_diag_remote.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = dict(input=str(args.input.resolve()), **summarize(json.loads(args.input.read_text())))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
