"""Fresh-rollout continuation accounting, distinct from completed-outer credit."""
import copy
import os
from pathlib import Path

PROTOCOL = 'tree_guard_continue_fresh_rollout_v1'
RESULT_PROTOCOL = 'tree_fresh_actor_result_v1'


def enabled(actor):
    return actor.get('guard_continue_protocol') == PROTOCOL


def execution_profile_bound(environ=None):
    """Recognize the authorized fresh H100 entry as well as HKUST resume."""
    import os
    environ = os.environ if environ is None else environ
    return bool(environ.get('RECURSIVE_A_CONTINUATION_BINDING')) or (
        environ.get('SEARCH_R1_EXECUTION_BACKEND') in ('slurm_h100', 'h100_shared')
        and environ.get('RECURSIVE_A_H100_FRESH_PROFILE') == '1'
        and bool(environ.get('SEARCH_R1_BASELINE_RUN_ID'))
        and len(environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256', '')) == 64)


def round_progress_enabled(actor, environ=None):
    """Explicit new contract: consumed sampling rounds and Adam writes are separate."""
    import os
    environ = os.environ if environ is None else environ
    return (enabled(actor) and environ.get('RECURSIVE_A_GUARD_ROUND_PROGRESS') == '1'
            and execution_profile_bound(environ))


def normal_guard_summary(refusal, writes, ledger):
    """Small normal-threshold receipt; programming faults never become a skip."""
    if (getattr(refusal, 'matched_all_ranks', None) is not True or
            getattr(refusal, 'safety_guard', None) is not True):
        raise ValueError('Only an explicit matched normal safety guard may consume a round')
    validate_ledger(ledger, writes, allow_zero=True)
    return dict(status='NORMAL_SAFETY_GUARD_ROUND_CONSUMED',
        optimizer_writes=writes, opportunities_processed=20, opportunities_skipped=20-writes,
        zero_write_round=(writes == 0), threshold_reason=str(refusal),
        automatic_rollback=False, model_optimizer_snapshot_saved=False,
        successful_update_claim=(writes > 0))


def begin_attempt(trainer):
    if trainer.global_steps != trainer.actual_outer_updates + 1:
        raise ValueError('Fresh attempt must target the next uncredited completed outer')
    trainer.training_attempts = getattr(trainer, 'training_attempts', 0) + 1
    return dict(baseline_outer_step=trainer.global_steps,
                baseline_attempt_id=trainer.training_attempts,
                baseline_completed_outer=trainer.actual_outer_updates)


def evidence_suffix(meta):
    attempt = meta.get('baseline_attempt_id')
    if attempt is None:
        return ''
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Evidence attempt identity must be a positive integer')
    return f'-attempt-{attempt:06d}'


def accounting(trainer):
    partial = copy.deepcopy(getattr(trainer, 'partial_optimizer_calls_per_rank', {}))
    completed = trainer.actual_outer_updates
    complete_calls = copy.deepcopy(getattr(trainer, 'completed_optimizer_calls_per_rank', {}))
    result = dict(protocol=PROTOCOL, training_attempts=getattr(trainer, 'training_attempts', completed),
                refused_attempts=getattr(trainer, 'refused_attempts', 0),
                guard_refused_attempts=getattr(trainer, 'guard_refused_attempts', 0),
                completed_outer_updates=completed, partial_optimizer_calls_per_rank=partial,
                completed_optimizer_calls_per_rank=complete_calls,
                total_optimizer_calls_per_rank={str(rank): complete_calls.get(str(rank), 0) + partial.get(str(rank), 0)
                    for rank in range(trainer.actor_rollout_wg.world_size)},
                partial_evidence=copy.deepcopy(getattr(trainer, 'partial_attempt_evidence', [])))
    if (round_progress_enabled(trainer.config.actor_rollout_ref.actor) and
            getattr(trainer, '_guard_round_contract_active', True)):
        zeros = getattr(trainer, 'zero_write_sampling_rounds', 0)
        result.update(round_progress_contract='consumed_sampling_rounds_separate_adam_v1',
            consumed_sampling_rounds=completed, zero_write_sampling_rounds=zeros,
            optimizer_write_rounds=completed-zeros,
            normal_guard_rounds=getattr(trainer, 'normal_guard_rounds', 0))
    return result


def validate_ledger(ledger, writes, allow_zero=True):
    if (not isinstance(ledger, list) or len(ledger) != 20 or type(writes) is not int
            or not (0 if allow_zero else 1) <= writes <= 20):
        raise ValueError('Incomplete twenty-opportunity ledger or invalid actual writes')
    seen = []
    for index, entry in enumerate(ledger, 1):
        if (type(entry.get('index')) is not int or entry.get('index') != index
                or type(entry.get('actual_writes')) is not int or entry.get('actual_writes') not in (0, 1)
                or type(entry.get('rows')) is not int or entry['rows'] != 16
                or entry.get('local_row_indices') != list(range(len(seen), len(seen)+entry['rows']))
                or entry.get('status') not in ('SUCCESS_ADAM_WRITE', 'SUCCESS_ADAM_WRITE_BRAKE_AFTER_WRITE',
                    'SKIPPED_EMPTY', 'SKIPPED_SAFETY_GUARD_BEFORE_WRITE', 'SKIPPED_BY_PRIOR_ROLLOUT_BRAKE')
                or entry['status'].startswith('SUCCESS') != bool(entry['actual_writes'])):
            raise ValueError('Opportunity ledger index/row/status/write differs from the real mini enumeration')
        seen.extend(entry['local_row_indices'])
    if sum(entry['actual_writes'] for entry in ledger) != writes:
        raise ValueError('Opportunity ledger does not sum to successful optimizer returns')
    if seen != list(range(320)):
        raise ValueError('Complete fresh opportunity ledger must enumerate every actual local row exactly once')


def validate_completed_result(trainer, output):
    """DataProto.concat uses rank zero metadata; this result was gathered first."""
    result = output.meta_info.get('tree_fresh_actor_result')
    expected = dict(protocol=RESULT_PROTOCOL, decision='COMPLETED_OUTER',
                    attempt_id=trainer.training_attempts, accepted_outer=trainer.global_steps,
                    completed_outer_before=trainer.actual_outer_updates)
    from .tree_partial_state import partial_context
    from .training_state import file_sha
    import json
    import os
    if not isinstance(result, dict) or result.get('identity') != partial_context(trainer)['identity']:
        raise ValueError('Completed result source/run/context identity differs')
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('Missing or mismatched explicit all-rank completed result')
    ranks = result.get('ranks')
    if not isinstance(ranks, list) or len(ranks) != trainer.actor_rollout_wg.world_size:
        raise ValueError('Incomplete explicit completed rank result')
    for rank, value in enumerate(ranks):
        validate_ledger(value.get('opportunity_ledger'), value.get('optimizer_calls'),
                        allow_zero=round_progress_enabled(trainer.config.actor_rollout_ref.actor))
        if any(type(value.get(key)) is not int for key in ('rank','attempt_id','accepted_outer',
                'completed_outer_before','optimizer_calls','total_optimizer_calls',
                'opportunities_processed','opportunities_skipped','scheduler_before','scheduler_after')):
            raise ValueError('Completed rank counters must be real integers')
        if (value.get('rank') != rank or value.get('opportunities_processed') != 20
                or value.get('opportunities_skipped') != 20-value['optimizer_calls']
                or value.get('scheduler_before') != trainer.actual_outer_updates
                or value.get('scheduler_after') != trainer.global_steps
                or any(value.get(key) != expected[key] for key in ('attempt_id', 'accepted_outer', 'completed_outer_before'))):
            raise ValueError('Completed actor rank count/identity/scheduler differs')
        direct = value.get('direct_rank_record', {})
        expected_path = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'fresh-state'/\
            f'outer-{trainer.global_steps:06d}-attempt-{trainer.training_attempts:06d}-rank{rank:02d}.json'
        path = Path(direct.get('path', ''))
        if (path != expected_path or path.is_symlink() or not path.is_file()
                or file_sha(path) != direct.get('sha256')):
            raise ValueError('Completed direct rank physical bytes/attempt path differ')
        physical = json.loads(path.read_text())
        physical_expected = dict(
                rank=rank,world_size=trainer.actor_rollout_wg.world_size,attempt_id=trainer.training_attempts,
                observed_outer=trainer.global_steps,actual_optimizer_calls=value['optimizer_calls'],
                opportunities_processed=20,opportunities_skipped=value['opportunities_skipped'],
                optimizer_opportunity_ledger=value['opportunity_ledger'],
                scheduler_step_before=trainer.actual_outer_updates,scheduler_step_after=trainer.global_steps)
        physical_expected.update({key:result['identity'][key] for key in
            ('job_id','run_id','code_commit','config_sha256','data_identity','model_identity')})
        if any(physical.get(key) != expected_value for key, expected_value in physical_expected.items()):
            raise ValueError('Completed direct rank file disagrees with gathered actual result')
    if len({r['optimizer_calls'] for r in ranks}) != 1:
        raise ValueError('Matched opportunities produced different per-rank successful optimizer counts')
    totals = getattr(trainer, 'completed_optimizer_calls_per_rank', {})
    candidate = {str(r['rank']): totals.get(str(r['rank']), 0) + r['optimizer_calls'] for r in ranks}
    partial = getattr(trainer, 'partial_optimizer_calls_per_rank', {})
    if any(r['total_optimizer_calls'] != candidate[str(r['rank'])] + partial.get(str(r['rank']), 0) for r in ranks):
        raise ValueError('Actual accumulated optimizer calls omit completed or partial history')
    return candidate


def record_completed_result(trainer, output):
    from .training_state import atomic_save
    candidate = validate_completed_result(trainer, output)
    # Attempt numbers restart at the restored checkpoint across independent runs.
    # Keep immutable receipts under this run's already identity-bound output.
    directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'attempt-results'
    writes = output.meta_info['tree_fresh_actor_result']['ranks'][0]['optimizer_calls']
    value = dict(protocol=RESULT_PROTOCOL, status=('ALL_OPPORTUNITIES_SKIPPED_SAMPLING_ROUND_CONSUMED'
            if writes == 0 else 'COMPLETED_OPPORTUNITIES_OUTER_CREDITED'),
        successful_optimizer_update=(writes > 0), actual_writes_this_round=writes,
        attempt_id=trainer.training_attempts, accepted_outer=trainer.global_steps,
        all_rank_result=output.meta_info['tree_fresh_actor_result'],
        normal_guard=copy.deepcopy(output.meta_info.get('tree_normal_guard')),
        actual_completed_optimizer_calls_per_rank=candidate, automatic_rollback=False)
    atomic_save(value, directory/f'attempt-{trainer.training_attempts:06d}.json', json_format=True)
    trainer.completed_optimizer_calls_per_rank = candidate
    if round_progress_enabled(trainer.config.actor_rollout_ref.actor):
        trainer.zero_write_sampling_rounds = getattr(trainer, 'zero_write_sampling_rounds', 0) + int(writes == 0)
        trainer.normal_guard_rounds = getattr(trainer, 'normal_guard_rounds', 0) + int('tree_normal_guard' in output.meta_info)
