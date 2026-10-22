"""Independent four-GPU baseline/search-only-stop candidate comparison.

Both fresh512-parent runs use the same chosen proxy/retrieval mode (default
legacy /retrieve-original), cold cache, checkpoint and input parents. Only the second candidate child
enables the search-only StopChecker. No retrieval batching change is mixed in.
This measures a candidate chunk, not an entire training outer or final quality.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from reproduction.operations.easy7b_20261021 import run_candidate_retrieval_pair as pair

WORK = ('actual_raw_generated_tokens', 'actual_prompt_tokens',
        'actual_generation_requests', 'actual_retrieval_calls')
TIMES = ('generation_phase_wall_seconds',
         'pipeline_seconds_including_input_io_score_and_output_save')


def compare(baseline, early_stop):
    a, b = baseline['candidate_result_summary'], early_stop['candidate_result_summary']
    return dict(baseline_result=baseline['candidate_result'], early_stop_result=early_stop['candidate_result'],
        same_input_fingerprints=a['input_fingerprints_sha256'] == b['input_fingerprints_sha256'],
        timings={key: dict(baseline_seconds=a[key], early_stop_seconds=b[key],
            observed_baseline_over_early_stop_ratio=a[key] / b[key] if b[key] else None,
            observed_wall_reduction_percent=100 * (1 - b[key] / a[key]) if a[key] else None)
            for key in TIMES},
        baseline_actual_work={key: a[key] for key in WORK},
        early_stop_actual_work={key: b[key] for key in WORK},
        early_stop_over_baseline_work_ratios={key: b[key] / a[key] if a[key] else None for key in WORK},
        baseline_selection=a['selection'], early_stop_selection=b['selection'],
        causal_full_outer_speedup_proven=False, covers_entire_training_outer=False,
        scope='One512-parent candidate chunk per arm. Search-only stopping changes raw completion '
            'lengths and asynchronous RNG consumption; interpret observed wall alongside actual work. '
            'No final-quality equivalence or full training-outer gain is proven.')


def run(args):
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=False)
    job = os.environ.get('SLURM_JOB_ID')
    pair.publish(root / 'launch.json', dict(role='independent_search_action_stop_pair',
        arguments=vars(args), slurm_job_id=job, started_unix=time.time(),
        order=['baseline', 'early-stop'], both_proxy_mode=args.retrieval_mode,
        both_service_route='/retrieve-original' if args.retrieval_mode == 'legacy' else '/retrieve',
        production_modified=False))
    if not (job and job.isdecimal()):
        raise ValueError('Use this runner only in its independent four-GPU Slurm allocation')
    ready_path, ready, urls = pair.wait_service(args, root)
    baseline = pair.one(args, root, ready_path, urls, mode=args.retrieval_mode, chunks=1,
        name='baseline', search_only_early_stop=False)
    early_stop = pair.one(args, root, ready_path, urls, mode=args.retrieval_mode, chunks=1,
        name='early-stop', search_only_early_stop=True)
    result = dict(status='SEARCH_ACTION_STOP_STARTUP_PAIR_COMPLETED', slurm_job_id=job,
        checkpoint=args.checkpoint, candidate_files=args.candidate_files,
        baseline=baseline['candidate_result_summary'], early_stop=early_stop['candidate_result_summary'],
        comparison=compare(baseline, early_stop), both_proxy_mode=args.retrieval_mode,
        service_endpoint=urls[args.retrieval_mode], service_ready=str(ready_path),
        retrieval_asset_versions={key: ready[key] for key in
            ('encoder_manifest_sha256', 'index_sha256', 'corpus_sha256')},
        optimizer_loaded=False, optimizer_writes=0, production_modified=False,
        finished_unix=time.time())
    pair.publish(root / 'result.json', result)
    pair.event(root, result['status'], completed_children=2, completed_questions_per_arm=512)


def main():
    parser = pair.parser(); parser.description = __doc__
    parser.add_argument('--retrieval-mode', choices=('legacy', 'async'), default='legacy',
        help='Use this identical proxy/service mode in both baseline and early-stop arms')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root.is_relative_to('/Users') or root.is_relative_to('/private'):
        parser.error('Raw candidate output must remain remote')
    try:
        run(args)
    except BaseException as error:
        if root.is_dir():
            pair.publish(root / 'failure.json', dict(status='FAILED', error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), training_modified=False))
        raise


if __name__ == '__main__':
    main()
