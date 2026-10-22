"""Pure structural verification of common8 calibration worker records.

No files, workers, thresholds, or training state are changed. The result is not
production admission: Qwen identity, fresh initialization, probability parity,
parameter/Adam finite proofs and health rules require separate bound evidence.
"""

import copy
import math
import re
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
CALIBRATION_ROLES = frozenset((
    'policy_safety_calibration_discovery',
    'policy_safety_calibration_confirmation',
))


def _enforcement(role):
    return dict(negative_tail='observe_only' if role == 'policy_safety_calibration_discovery' else 'enforced',
                old_policy_kl='enforced', numerical_finite='enforced', dual_clip='enforced')
OBJECTIVE_WEIGHT_DEFINITION = 'equal two-row micro masked_mean; weighted global numerator/denominator'
LOGICAL4_OBJECTIVE_WEIGHT_DEFINITION = 'equal four-row logical micro masked_mean; weighted global numerator/denominator'


def objective_weight_definition(run=None):
    """Keep historical receipt text; the new Easy scope names its logical loss."""
    from reproduction.common8.precision_contract import easy2560_native4
    from reproduction.common8.topology import _bound_run
    run = _bound_run() if run is None else run
    if run is not None and run.get('easy2560_selection') is True:
        assert easy2560_native4(run), 'Logical4 monitor requires the explicit Easy scope'
        return LOGICAL4_OBJECTIVE_WEIGHT_DEFINITION
    return OBJECTIVE_WEIGHT_DEFINITION


MONITORING_SCOPE = 'fixed four trajectories per rank across length strata; not full rollout'
QUANTILE_METHOD = 'exact global nearest-order statistic'
# IEEE-754 binary32 has 24 significant bits: epsilon is 2**-23 and
# round-to-nearest unit roundoff is epsilon/2. These are numerical tolerances,
# independent of policy KL/tail thresholds or any future health criterion.
FLOAT32_EPSILON = 2. ** -23
GRADIENT_CLIP_LIMIT = 1.
GRADIENT_CLIP_FLOAT32_TOLERANCE = FLOAT32_EPSILON
GLOBAL_FIELDS = (
    'allowed', 'reasons', 'sampled_old_policy_kl', 'kl_estimate_complete',
    'effective_weight_mass', 'weighted_k3_numerator',
    'negative_effective_weight_mass', 'tail_effective_weight_mass',
    'effective_token_count', 'negative_token_count', 'negative_tail_count',
    'negative_tail_count_fraction', 'negative_tail_weight_fraction',
    'dual_clip_saturated_objective_weight_fraction', 'invalid_count',
    'overflow_count', 'quantile_method', 'negative_log_ratio_p95',
    'negative_log_ratio_p99', 'negative_log_ratio_max',
)
ADJACENT_FLOAT_FIELDS = frozenset((
    'weighted_k3_numerator', 'sampled_old_policy_kl',
    'negative_tail_count_fraction', 'negative_tail_weight_fraction',
    'dual_clip_saturated_objective_weight_fraction',
    'negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max',
))
MONITOR_TENSORS = frozenset((
    'responses', 'input_ids', 'attention_mask', 'position_ids',
    'old_log_probs', 'advantages', 'loss_mask', 'ref_log_prob',
))
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


class EvidenceError(ValueError):
    """Input evidence cannot prove a complete common8 calibration segment."""


def _require(ok, message):
    if not ok:
        raise EvidenceError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, 'Invalid integer: ' + name)
    return value


def _finite(value, name, nonnegative=True):
    _require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite number: ' + name)
    _require(not nonnegative or value >= 0, 'Negative number: ' + name)
    return value


def _same_arithmetic(actual, expected, name):
    # This tolerance only checks FP64 report arithmetic, never health thresholds.
    _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
             'Inconsistent report arithmetic: ' + name)


def optimizer_gradient_evidence(row, *, run=None,
        clip_message='Actual post-clip norm exceeds clip1 numerical bound'):
    """Validate measured legacy norms or explicitly unmeasured native4 lean logs.

    A reported coefficient is checked only as CPU arithmetic. It does not
    establish a measured post-clip norm, gradient finite scan, or GPU clipping
    correctness. Historical records retain their finite measured-norm checks.
    """
    before = _finite(row['gradient_norm_before'], 'gradient_norm_before')
    unmeasured = (row.get('gradient_norm_after_measured') is False or
                  row.get('gradient_norm_after') is None)
    if unmeasured:
        from search_g1.actor_performance import lean_run
        _require(run is not None and lean_run(run) is True,
                 'Unmeasured post-clip norm requires explicit native4 lean_run binding')
        _require('gradient_norm_after' in row and row['gradient_norm_after'] is None and
                 row.get('gradient_norm_after_measured') is False and
                 row.get('gradient_norm_after_source') == 'not_measured',
                 'Unmeasured post-clip norm must be explicitly null and not_measured')
        threshold = _finite(row['gradient_clip_threshold'], 'gradient_clip_threshold')
        coefficient = _finite(row['gradient_clip_coefficient'], 'gradient_clip_coefficient')
        _require(threshold == GRADIENT_CLIP_LIMIT and coefficient <= 1.,
                 'Lean clipping threshold or coefficient differs')
        _same_arithmetic(coefficient, min(1., threshold / (before + 1e-6)),
                         'reported gradient clip coefficient')
        return dict(gradient_norm_before=before, gradient_norm_after=None,
                    gradient_norm_after_measured=False, gradient_norm_after_source='not_measured',
                    gradient_clip_threshold=threshold, gradient_clip_coefficient=coefficient,
                    coefficient_arithmetic_consistent=True, measured_post_clip_bound_proven=False)
    after = _finite(row['gradient_norm_after'], 'gradient_norm_after')
    _require(after <= GRADIENT_CLIP_LIMIT + GRADIENT_CLIP_FLOAT32_TOLERANCE, clip_message)
    return dict(gradient_norm_before=before, gradient_norm_after=after,
                gradient_norm_after_measured=True, measured_post_clip_bound_proven=True)


def _hash(value, name):
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             'Invalid SHA256: ' + name)


def _adjacent(previous, current):
    # Identical weights/inputs between post(j) and pre(j+1) mean the same
    # observation. Counts, decisions and all masses must remain exact. Only
    # forward-derived floats permit one binary32 epsilon (relative and absolute
    # at unit scale) for FP32 recomputation; this is not a model-error bound.
    for name in GLOBAL_FIELDS:
        left, right = previous[name], current[name]
        if name in ADJACENT_FLOAT_FIELDS and left is not None and right is not None:
            _require(math.isclose(left, right, rel_tol=FLOAT32_EPSILON, abs_tol=FLOAT32_EPSILON),
                     'Adjacent unchanged-policy float differs: ' + name)
        else:
            _require(left == right, 'Adjacent unchanged-policy exact field differs: ' + name)


def _identity(value, monitor_count=4, *, run=None):
    _require(type(monitor_count) is int and monitor_count in (4, 8), 'Unsupported local monitor count')
    _require(isinstance(value, Mapping), 'Missing local monitor identity')
    _require(set(value) == {'selection_indices', 'actor_token_lengths', 'tensor_sha256'},
             'Unexpected monitor identity schema')
    indices = value['selection_indices']
    lengths = value['actor_token_lengths']
    _require(isinstance(indices, list) and len(indices) == monitor_count, 'Incomplete local monitor indices')
    for index in indices:
        _integer(index, 'selection_index')
        _require(index < 80 * monitor_count, 'Monitor index outside local rollout')
    _require(len(set(indices)) == monitor_count, 'Duplicate monitor index')
    _require(isinstance(lengths, list) and len(lengths) == monitor_count, 'Incomplete local monitor lengths')
    for length in lengths:
        _integer(length, 'actor_token_length', minimum=1)
    _require(lengths == sorted(lengths), 'Monitor lengths are not stratified')
    hashes = value['tensor_sha256']
    # Production actor batches omit reference probabilities when ref-KL is
    # disabled. The sampled-old-policy monitor still authenticates all seven
    # actual input tensors; calibration batches may additionally carry ref.
    # The Easy producer freezes every actual selected batch tensor, including
    # the mixed/fallback diagnostic label. It is not a substitute for an input
    # hash and must remain frozen across the same pre/post monitor sequence.
    # Historical calibration/common8 scopes keep their original exact schema.
    from reproduction.common8.precision_contract import easy2560_native4
    selection = {'selection_fallback'} if run is not None and easy2560_native4(run) else set()
    _require(isinstance(hashes, Mapping) and
             (MONITOR_TENSORS - {'ref_log_prob'}) | selection <= set(hashes),
             'Missing actual common8 monitor tensor hashes')
    _require(set(hashes) <= MONITOR_TENSORS | selection,
             'Unexpected actual common8 monitor tensor hashes')
    for name, digest in hashes.items():
        _hash(digest, name)


def _statistics(row):
    _require(row['allowed'] is True and row['reasons'] == [], 'Rejected monitor')
    _require(row['kl_estimate_complete'] is True, 'Incomplete KL is diagnostic only')
    for name in ('effective_token_count', 'negative_token_count', 'negative_tail_count',
                 'invalid_count', 'overflow_count'):
        _integer(row[name], name)
    _require(row['invalid_count'] == row['overflow_count'] == 0, 'Invalid or over-bound monitor')
    count, negative, tail = (row[name] for name in
                            ('effective_token_count', 'negative_token_count', 'negative_tail_count'))
    _require(count > 0 and 0 <= tail <= negative <= count, 'Inconsistent token counts')
    mass, numerator, negative_mass, tail_mass = (
        _finite(row[name], name) for name in
        ('effective_weight_mass', 'weighted_k3_numerator',
         'negative_effective_weight_mass', 'tail_effective_weight_mass'))
    _require(mass > 0, 'Empty effective weight mass')
    _require(negative_mass <= mass or math.isclose(negative_mass, mass, rel_tol=1e-12, abs_tol=1e-12),
             'Negative mass exceeds total mass')
    _require(tail_mass <= negative_mass or math.isclose(tail_mass, negative_mass, rel_tol=1e-12, abs_tol=1e-12),
             'Tail mass exceeds negative mass')
    _require((negative == 0) == (negative_mass == 0), 'Negative count/mass disagree')
    _require((tail == 0) == (tail_mass == 0), 'Tail count/mass disagree')
    expected = {
        'sampled_old_policy_kl': numerator / mass,
        'negative_tail_count_fraction': tail / negative if negative else 0.,
        'negative_tail_weight_fraction': tail_mass / negative_mass if negative_mass else 0.,
        'dual_clip_saturated_objective_weight_fraction': tail_mass / mass,
    }
    for name, ratio in expected.items():
        _same_arithmetic(_finite(row[name], name), ratio, name)
    _require(row['quantile_method'] == QUANTILE_METHOD, 'Unexpected quantile method')
    quantiles = [row[name] for name in
                 ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max')]
    if negative:
        for value in quantiles:
            _finite(value, 'negative log-ratio quantile', nonnegative=False)
        _require(quantiles == sorted(quantiles), 'Inconsistent negative quantiles')
    else:
        _require(quantiles == [None, None, None], 'Quantiles without negative tokens')
    return {name: copy.deepcopy(row[name]) for name in GLOBAL_FIELDS}


def _row(row, rank, binding, kind):
    _require(isinstance(row, Mapping), 'Malformed worker record')
    _require(row['kind'] == kind and row['run_id'] == binding['run_id'], 'Wrong record kind or run')
    _require(_integer(row['rank'], 'rank') == rank, 'Wrong rank in worker log')


def _context(row, outer, mini, prior):
    context = row['context']
    expected = dict(logical_update=outer, minibatch_index=mini, prior_optimizer_writes=prior)
    gate_key = 'behavior_probability_gate_sha256'
    _require(isinstance(context, Mapping) and set(context) in
             (set(expected), set(expected) | {gate_key}), 'Unexpected context schema')
    if gate_key in context:
        _hash(context[gate_key], 'actual behavior gate context')
    for name, value in expected.items():
        _require(_integer(context[name], name) == value, 'Wrong mini/write context: ' + name)


def verify_segment(rows_by_rank, binding):
    """Validate complete fresh-counter logs and coalesce rank replicas once.

    ``binding`` carries schema_version=1, a calibration discovery/confirmation
    role, run_id, calibration_sha256, outer_count and initial_optimizer_writes=0.
    ``rows_by_rank`` contains the actual ordered worker records for ranks 0..7.
    Explicit synthetic/component roles are rejected. This pure check cannot
    authenticate physical execution from JSON; it never grants production or
    healthy admission, even when all supplied records are structurally valid.
    """
    try:
        return _verify_segment(rows_by_rank, binding)
    except EvidenceError:
        raise
    except (KeyError, TypeError, OverflowError) as failure:
        raise EvidenceError('Malformed calibration evidence: ' + str(failure)) from failure


def verify_formal_segment(rows_by_rank, binding):
    """Audit20 processed opportunities and actual retained Adam writes separately.

    This pure check does not authenticate physical execution or accuracy health.
    Rejected monitors must correspond to explicit guarded skip/postwrite records.
    """
    from reproduction.common8.formal_contract import FORMAL_SAFEGUARDS, SAFETY_MODE, consumed_lr
    size = binding.get('training_world_size', 8)
    _require(type(size) is int and size in (4, 8), 'Unsupported formal evidence topology')
    from reproduction.common8.topology import formal_monitor_geometry
    run=dict(profile='common8',training_world_size=size)
    for name in ('actor_micro_batch_per_rank', 'actor_logical_micro_batch_per_rank',
                 'easy2560_selection', 'execution_site', 'execution_backend', 'data_selection_mode', 'question_probe',
                 'performance_profile', 'extra_policy_monitor_forwards', 'policy_calibration_role'):
        if name in binding:
            run[name] = binding[name]
    try:geometry=formal_monitor_geometry(run)
    except (AssertionError,KeyError,TypeError) as failure:
        raise EvidenceError('Invalid bound formal monitor geometry: '+str(failure)) from failure
    monitor_count=geometry['monitor_trajectories_per_rank'];micro=geometry['actor_micro_batch_per_rank']
    scope = ('fixed four trajectories per rank across length strata; not full rollout' if size == 8 else
             'fixed eight trajectories per rank across length strata; not full rollout')
    _require(binding['schema_version'] == 1 and binding['role'] == 'production' and
             binding['policy_safety_mode'] == SAFETY_MODE, 'Not explicit inline formal evidence')
    _require(binding['candidate_thresholds'] == FORMAL_SAFEGUARDS, 'Frozen formal safeguards differ')
    outers = _integer(binding['outer_count'], 'outer_count', minimum=1)
    _require(outers <= 200 and binding['initial_optimizer_writes'] == 0, 'Invalid formal counters')
    _hash(binding['calibration_sha256'], 'formal plan binding')
    attempts = binding['complete_attempts']
    _require([r['step'] for r in attempts] == list(range(1, outers + 1)), 'Incomplete complete-outer selection')
    _require(len({r['attempt_id'] for r in attempts}) == outers, 'Duplicate completed attempt identity')
    _require(set(rows_by_rank) == set(range(size)), 'Need all actual formal rank logs')
    observations = {}; identities = {}; dispositions = {}; selected_attempts = {}
    previous_attempt = 0; previous_writes = 0
    for record in attempts:
        outer = record['step']; index = _integer(record['attempt_index'], 'attempt_index', minimum=1)
        start = _integer(record['optimizer_writes_before'], 'optimizer_writes_before')
        end = _integer(record['optimizer_writes_after'], 'optimizer_writes_after')
        _require(index > previous_attempt and start >= previous_writes and 1 <= end - start <= 20,
                 'Completed opportunity/actual-write counters differ')
        _require(record['opportunities_processed'] == 20, 'Incomplete opportunities cannot count as outer')
        selected_attempts[outer] = record
        previous_attempt, previous_writes = index, end
    for rank in range(size):
        rows = rows_by_rank[rank]; cursor = 0
        for outer in range(1, outers + 1):
            attempt = selected_attempts[outer]; counter = attempt['optimizer_writes_before']
            identity = None; frozen = None
            for mini in range(20):
                _require(cursor < len(rows), 'Missing formal opportunity evidence')
                before = rows[cursor]; cursor += 1
                write = None
                if cursor < len(rows) and rows[cursor]['kind'] == 'OPTIMIZER_WRITE':
                    write = rows[cursor]; cursor += 1
                _require(cursor + 1 < len(rows), 'Missing post-monitor or opportunity decision')
                after, decision = rows[cursor:cursor + 2]; cursor += 2
                prior = counter; succeeded = write is not None
                expected_context = dict(logical_update=outer, minibatch_index=mini, prior_optimizer_writes=prior,
                    attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'])
                for event in [before, after, decision] + ([] if write is None else [write]):
                    context = event['context']
                    _require(set(context) == set(expected_context) | {'behavior_probability_gate_sha256'},
                             'Unbound formal opportunity context')
                    _require(all(context[key] == value for key, value in expected_context.items()),
                             'Formal opportunity context differs')
                    _hash(context['behavior_probability_gate_sha256'], 'formal actual behavior gate')
                    _require(context['behavior_probability_gate_sha256'] == attempt['behavior_probability_gate_sha256'],
                             'Completed attempt gate differs')
                _row(decision, rank, binding, 'OPPORTUNITY')
                _require(decision['processed'] is True and decision['opportunity_index'] == mini,
                         'Skipped opportunities were not actually processed in order')
                _require(type(decision['optimizer_write_succeeded']) is bool and
                         decision['optimizer_write_succeeded'] == succeeded, 'Claimed Adam write differs from actual record')
                _require(decision['optimizer_writes_before'] == prior and
                         decision['optimizer_writes_after'] == prior + int(succeeded), 'Opportunity actual-write counters differ')
                guards = decision['guard_events']
                _require(isinstance(guards, list), 'Missing explicit opportunity guard events')
                for event in guards:
                    _require(isinstance(event['stage'], str) and event['stage'] and isinstance(event['reasons'], list)
                             and event['reasons'] and all(isinstance(x, str) and x for x in event['reasons']),
                             'Malformed real guard event')
                expected_status = ('written_with_post_guard' if guards else 'written') if succeeded else 'skipped_guard'
                _require(decision['status'] == expected_status and (succeeded or bool(guards)),
                         'Opportunity skip/write disposition differs')
                local_global = {name: copy.deepcopy(value) for name, value in decision.items() if name != 'rank'}
                key = (outer, mini)
                if key in dispositions:
                    _require(dispositions[key] == local_global, 'Ranks disagree on actual opportunity outcome')
                else:
                    dispositions[key] = local_global
                for row, stage, writes in ((before, 'before_minibatch', prior),
                                          (after, 'after_minibatch', prior + int(succeeded))):
                    _row(row, rank, binding, 'POLICY_MONITOR')
                    _require(row['stage'] == stage and row['optimizer_writes'] == writes, 'Formal monitor/actual-write order differs')
                    _require(row['calibration_sha256'] == binding['calibration_sha256'], 'Formal frozen plan differs')
                    _require(row['execution_role'] == 'production' and row['enforcement'] == _enforcement('production'),
                             'Formal brakes are not enforced')
                    _require(row['objective_weight_definition'] == objective_weight_definition(run) and
                             row['monitoring_scope'] == scope, 'Formal monitor weighting/scope differs')
                    _integer(row['rejection_count'], 'actual cumulative rejection_count')
                    _identity(row['monitor_identity'], monitor_count=monitor_count, run=run)
                    if identity is None:
                        identity = copy.deepcopy(row['monitor_identity']); identities[(outer, rank)] = identity
                    _require(identity == row['monitor_identity'], 'Frozen formal monitor inputs changed')
                    _require(type(row['allowed']) is bool, 'Invalid guard decision')
                    if row['kl_estimate_complete'] is True:
                        numerical = dict(row, allowed=True, reasons=[])
                        statistics = _statistics(numerical)
                        reasons = []
                        if statistics['sampled_old_policy_kl'] > FORMAL_SAFEGUARDS['target_old_policy_kl']:
                            reasons.append('old_policy_kl')
                        if statistics['negative_tail_weight_fraction'] > FORMAL_SAFEGUARDS['negative_tail_limit']:
                            reasons.append('negative_advantage_tail')
                        _require(row['reasons'] == reasons and row['allowed'] == (not reasons),
                                 'Actual complete monitor does not match frozen brake decision')
                        statistics['allowed'] = row['allowed']
                        statistics['reasons'] = copy.deepcopy(row['reasons'])
                        fixed = tuple(statistics[name] for name in ('effective_weight_mass', 'negative_effective_weight_mass',
                            'effective_token_count', 'negative_token_count'))
                        if frozen is None: frozen = fixed
                        _require(fixed == frozen, 'Frozen formal sample weights changed')
                    else:
                        _require(row['allowed'] is False and row['sampled_old_policy_kl'] is None and row['reasons'],
                                 'Incomplete numerical estimate cannot be admitted as true KL')
                        _require(any(reason in row['reasons'] for reason in (
                            'nonfinite_global_totals', 'nonfinite_or_negative_effective_input',
                            'delta_above_kl_log_bound', 'empty_effective_samples')), 'Unexplained incomplete estimate')
                        statistics = {name: copy.deepcopy(row[name]) for name in GLOBAL_FIELDS}
                    if not row['allowed']:
                        _require(any(event['stage'] == stage and event['reasons'] == row['reasons'] for event in guards),
                                 'Rejected policy monitor lacks its actual guard disposition')
                        if stage == 'before_minibatch':
                            _require(not succeeded, 'Rejected pre-monitor cannot enter Adam write')
                    key = (outer, mini, stage)
                    if key not in observations:
                        observations[key] = dict(run_id=binding['run_id'], context=copy.deepcopy(row['context']),
                            stage=stage, optimizer_writes=writes, calibration_sha256=binding['calibration_sha256'],
                            execution_role='production', enforcement=_enforcement('production'), **statistics)
                    else:
                        _require(all(observations[key][name] == statistics[name] for name in GLOBAL_FIELDS),
                                 'Actual rank global formal observations disagree')
                if write is not None:
                    _row(write, rank, binding, 'OPTIMIZER_WRITE')
                    _require(write['write_count'] == prior + 1 and write['learning_rates'] and
                             all(type(rate) in (int, float) and rate == consumed_lr(outer) for rate in write['learning_rates']),
                             'Actual formal Adam write or consumed LR differs')
                    optimizer_gradient_evidence(write, run=run, clip_message='Actual formal L2 clipping differs')
                    counter += 1
            _require(cursor < len(rows), 'Missing completed opportunity budget')
            budget = rows[cursor]; cursor += 1; _row(budget, rank, binding, 'OUTER_BUDGET')
            for name, value in dict(logical_update=outer, attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'],
                local_trajectories=2560 // size, local_mini=128 // size, local_micro=micro, opportunities_processed=20,
                optimizer_writes=attempt['optimizer_writes_after'] - attempt['optimizer_writes_before'],
                optimizer_writes_before=attempt['optimizer_writes_before'], optimizer_writes_after=counter,
                cumulative_writes=counter).items():
                _require(budget[name] == value, 'Completed formal budget differs: ' + name)
            _require(counter == attempt['optimizer_writes_after'], 'Complete outer actual success count differs')
        _require(cursor == len(rows), 'Extra completed opportunity evidence')
    for (outer, mini, stage), observation in observations.items():
        count = sum(sum(identities[(outer, rank)]['actor_token_lengths']) for rank in range(size))
        _require(observation['effective_token_count'] == count, 'Formal global token count differs')
        if observation['kl_estimate_complete']:
            _same_arithmetic(observation['effective_weight_mass'],geometry['expected_complete_effective_weight_mass'],
                             'formal global masked-mean weight mass')
        if mini and stage == 'before_minibatch':
            _adjacent(observations[(outer, mini - 1, 'after_minibatch')], observation)
    return dict(role='formal_completed_opportunity_monitor_audit', run_id=binding['run_id'],
        complete_outer_count=outers, processed_opportunities_per_rank=outers * 20,
        completed_outer_actual_writes_per_rank=sum(r['optimizer_writes_after'] - r['optimizer_writes_before'] for r in attempts),
        actual_cumulative_optimizer_writes_per_rank=attempts[-1]['optimizer_writes_after'],
        guard_triggered_monitor_count=sum(not r['allowed'] for r in observations.values()),
        global_monitors=list(observations.values()), opportunity_dispositions=list(dispositions.values()),
        monitor_weight_binding=geometry,
        production_admission=False, scope='pure structural validation; physical execution separately authenticated')


def _verify_segment(rows_by_rank, binding):
    _require(isinstance(binding, Mapping), 'Missing evidence binding')
    _require(type(binding['schema_version']) is int and binding['schema_version'] == SCHEMA_VERSION,
             'Unsupported evidence schema')
    _require(binding['role'] in CALIBRATION_ROLES, 'Not an actual-policy calibration role')
    _require(isinstance(binding['run_id'], str) and bool(binding['run_id']), 'Missing run identity')
    _hash(binding['calibration_sha256'], 'calibration binding')
    outers = _integer(binding['outer_count'], 'outer_count', minimum=1)
    _require(_integer(binding['initial_optimizer_writes'], 'initial_optimizer_writes') == 0,
             'Calibration counters must start at zero')
    _require(isinstance(rows_by_rank, Mapping) and
             all(type(rank) is int for rank in rows_by_rank) and set(rows_by_rank) == set(range(8)),
             'Need exactly eight distinct rank logs')
    observations = {}
    behavior_gates = {}
    identities = {outer: {} for outer in range(1, outers + 1)}
    for rank in range(8):
        rows = rows_by_rank[rank]
        _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and len(rows) == outers * 61,
                 'Incomplete or duplicate write/check/budget schedule on rank ' + str(rank))
        cursor = 0
        for outer in range(1, outers + 1):
            identity = None
            frozen_statistics = None
            for mini in range(20):
                prior = (outer - 1) * 20 + mini
                before, write, after = rows[cursor:cursor + 3]
                cursor += 3
                for row, stage, writes in ((before, 'before_minibatch', prior),
                                          (after, 'after_minibatch', prior + 1)):
                    _row(row, rank, binding, 'POLICY_MONITOR')
                    _context(row, outer, mini, prior)
                    _require(row['stage'] == stage, 'Wrong pre/post monitor order')
                    _require(_integer(row['optimizer_writes'], 'optimizer_writes') == writes,
                             'Wrong monitor write count')
                    _require(row['calibration_sha256'] == binding['calibration_sha256'], 'Changed calibration binding')
                    _require(row['execution_role'] == binding['role'] and
                             row['enforcement'] == _enforcement(binding['role']),
                             'Calibration role or enforced brakes differ')
                    _require(row['objective_weight_definition'] == OBJECTIVE_WEIGHT_DEFINITION and
                             row['monitoring_scope'] == MONITORING_SCOPE, 'Unexpected monitoring definition')
                    _require(_integer(row['rejection_count'], 'rejection_count') == 0, 'Prior policy rejection')
                    _identity(row['monitor_identity'])
                    if identity is None:
                        identity = copy.deepcopy(row['monitor_identity'])
                        identities[outer][rank] = identity
                    _require(row['monitor_identity'] == identity, 'Local monitor identity changed within outer')
                    statistics = _statistics(row)
                    fixed = tuple(statistics[name] for name in
                                  ('effective_weight_mass', 'negative_effective_weight_mass',
                                   'effective_token_count', 'negative_token_count'))
                    if frozen_statistics is None:
                        frozen_statistics = fixed
                    _require(fixed == frozen_statistics, 'Frozen mask/advantage statistics changed within outer')
                    key = (binding['run_id'], outer, mini, stage)
                    if key not in observations:
                        observations[key] = dict(run_id=binding['run_id'], context=copy.deepcopy(row['context']),
                                                 stage=stage, optimizer_writes=writes,
                                                 calibration_sha256=binding['calibration_sha256'],
                                                 execution_role=binding['role'],
                                                 enforcement=copy.deepcopy(row['enforcement']), **statistics)
                    else:
                        _require(all(observations[key][name] == statistics[name] for name in GLOBAL_FIELDS),
                                 'Global monitor fields disagree across ranks')
                _row(write, rank, binding, 'OPTIMIZER_WRITE')
                _context(write, outer, mini, prior)
                # Old immutable logs have no behavior receipt. New logs bind
                # every check/write, across all ranks, to one gate per outer.
                for event in (before, write, after):
                    gate = event['context'].get('behavior_probability_gate_sha256')
                    if outer not in behavior_gates:
                        behavior_gates[outer] = gate
                    _require(gate == behavior_gates[outer],
                             'Behavior probability gate changed within outer or across ranks')
                _require(_integer(write['write_count'], 'write_count') == prior + 1, 'Wrong actual write count')
                rates = write['learning_rates']
                _require(isinstance(rates, list) and bool(rates), 'Missing actual learning rates')
                for rate in rates:
                    _require(_finite(rate, 'learning_rate') == 1e-6, 'Actual learning rate differs')
                # FP32 coordinate rounding has unit roundoff epsilon/2; one
                # epsilon also leaves room for FP64 norm arithmetic/serialization.
                optimizer_gradient_evidence(write)
            budget = rows[cursor]
            cursor += 1
            _row(budget, rank, binding, 'OUTER_BUDGET')
            for name, expected in dict(logical_update=outer, local_trajectories=320,
                                       local_mini=16, local_micro=2, optimizer_writes=20,
                                       cumulative_writes=outer * 20).items():
                _require(_integer(budget[name], name) == expected, 'Wrong actual outer budget: ' + name)
    for observation in observations.values():
        outer = observation['context']['logical_update']
        count = sum(sum(identity['actor_token_lengths']) for identity in identities[outer].values())
        _require(observation['effective_token_count'] == count, 'Global token count differs from frozen monitors')
        _same_arithmetic(observation['effective_weight_mass'], 16., 'eight-rank two-row micro weight mass')
    for outer in range(1, outers + 1):
        for mini in range(1, 20):
            previous = observations[(binding['run_id'], outer, mini - 1, 'after_minibatch')]
            current = observations[(binding['run_id'], outer, mini, 'before_minibatch')]
            _require(previous['optimizer_writes'] == current['optimizer_writes'],
                     'Unexpected write between adjacent monitors')
            _adjacent(previous, current)
    return dict(schema_version=SCHEMA_VERSION, role='policy_safety_calibration_structural_verification',
                execution_role=binding['role'], run_id=binding['run_id'],
                calibration_sha256=binding['calibration_sha256'], outer_count=outers,
                optimizer_writes_per_rank=outers * 20, global_monitors=list(observations.values()),
                monitor_identities=identities,
                enforcement=_enforcement(binding['role']), production_admission=False)
