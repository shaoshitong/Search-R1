"""Final-only, read-only CPU audit of exactly 20 retained writes per native4 outer.

The existing authenticated restore-chain and no201 readers prove the complete
per-attempt/per-rank ledger chain. Their limit is 20 retained writes per completed
outer; 200 completed outers and 4000 retained writes therefore force 20 in every
outer on every rank. This does not prove zero additional Adam calls or rollbacks.
No model, Torch, GPU, training, checkpoint mutation or report-file output is used.
"""
import argparse
import json
from pathlib import Path
import re
import time

from reproduction.easy2560_joint4 import checkpoint_evidence
from reproduction.easy2560_joint4 import final_cpu_acceptance


STATUS = 'EASY2560_STRICT_FINAL_RETAINED20_NATIVE4_PASS'
REJECTED = 'EASY2560_STRICT_FINAL_RETAINED20_NATIVE4_REJECTED'


def need(condition, message):
    if not condition:
        raise ValueError('EASY2560_STRICT_FINAL_WRITES_' + message)


def exact_int(value, expected):
    return type(value) is int and value == expected


def verify_strict_final_writes(base, *, training_commit):
    """Consume existing final evidence only; a missing outer200 or invalid chain fails.

    ``training_commit`` is mandatory and names the original training source,
    not the commit containing this independent operator. Restore ancestors keep
    their own source identities through authenticated_restore_chain().
    """
    base = Path(base).resolve()
    run, run_record = final_cpu_acceptance.small(base/'logs/run-manifest.json')
    need(isinstance(run, dict) and isinstance(training_commit, str) and
         re.fullmatch('[0-9a-f]{40}', training_commit) is not None and
         run.get('source_commit') == training_commit, 'EXPLICIT_TRAINING_COMMIT_REQUIRED')
    need(run.get('easy2560_selection') is True and
         exact_int(run.get('training_world_size'), 4), 'NATIVE4_EASY_RUN_REQUIRED')
    source_binding = final_cpu_acceptance.published_source_binding(base, run, training_commit)
    # Do not replace these authoritative readers with copied counter logic.
    chain = checkpoint_evidence.authenticated_restore_chain(base, run)
    progress = final_cpu_acceptance.no201(base, run, chain=chain)
    need(isinstance(progress, dict) and exact_int(progress.get('actual_completed_updates'), 200) and
         exact_int(progress.get('rank_count'), 4) and progress.get('no_update201') is True and
         progress.get('no_post200_driver_scheduler_worker_attempts') is True,
         'PROVEN_COMPLETE200_NATIVE4_NO201_REQUIRED')
    need(exact_int(progress.get('cumulative_actual_optimizer_writes_per_rank'), 4000),
         'EXACT4000_RETAINED_WRITES_REQUIRED')
    attempt_count = progress.get('actual_attempt_count')
    need(type(attempt_count) is int and attempt_count >= 200, 'PROVEN_ATTEMPT_COUNT_REQUIRED')
    ranks = progress.get('ranks')
    need(isinstance(ranks, dict) and set(ranks) == {'0', '1', '2', '3'}, 'ALL4_RANK_PROOFS_REQUIRED')
    for rank in range(4):
        roles = ranks[str(rank)]
        need(isinstance(roles, dict) and set(roles) == {'scheduler', 'optimizer'}, 'RANK_ROLES_REQUIRED')
        for role in ('scheduler', 'optimizer'):
            proof = roles[role]
            need(isinstance(proof, dict) and exact_int(proof.get('final_completed_outer'), 200) and
                 exact_int(proof.get('cumulative_actual_writes'), 4000) and
                 exact_int(proof.get('terminal_attempt_index'), attempt_count),
                 'RANK_FINAL_COUNTERS_DIFFER')
    return dict(schema_version=1, status=STATUS, strict_retained_write_audit_accepted=True,
                base=str(base), run_id=run['run_id'], source_commit=training_commit,
                run_manifest=run_record, source_publication_binding=source_binding,
                actual_completed_updates=200, native_rank_count=4,
                retained_optimizer_writes_per_completed_outer_per_rank=20,
                retained_optimizer_writes_per_rank=4000, actual_attempt_count=attempt_count,
                proof=progress,
                inference='200 completed outers, each with at most 20 retained writes, sum to 4000; '
                          'authenticated per-rank write ledgers therefore retain 20 in every outer',
                zero_extra_historical_adam_calls_proven=False,
                zero_historical_rollbacks_proven=False,
                read_only=True, new_training_updates=0, model_reloads=0, gpu_allocations=0,
                actual_restore_performed=False, actual_cuda_restore_accepted=False,
                overall_goal_complete=False, checked_unix=time.time(),
                scope='strict retained-write budget across the authenticated original/restored execution chain; '
                      'extra optimizer calls, rollbacks, evaluation, archives and actual CUDA restore are separate')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--training-commit', required=True,
                        help='explicit 40-character original training commit; never inferred from operator HEAD')
    args = parser.parse_args(argv)
    try:
        result = verify_strict_final_writes(args.base, training_commit=args.training_commit)
        exit_code = 0
    except Exception as failure:
        result = dict(schema_version=1, status=REJECTED,
                      strict_retained_write_audit_accepted=False, read_only=True,
                      overall_goal_complete=False, error_type=type(failure).__name__, error=str(failure))
        exit_code = 1
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
