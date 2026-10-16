"""Actual CPU optimizer checks and pure provenance rejection tests; not eight GPUs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from reproduction.common8.formal_contract import FORMAL_COUNTING, FORMAL_SAFEGUARDS, SAFETY_MODE, consumed_lr
from reproduction.common8.formal_inline_safety import validate_optimizer_state, verify_method
from reproduction.common8.policy_calibration_plan import document_sha256
from search_g1.policy_safety import monitor


def guarded_opportunity_fixture(world_size=8, actor_micro=2):
    """Actual CPU masked-mean statistics; rank records are synthetic provenance."""
    import hashlib
    from reproduction.common8.safety_calibration_evidence import MONITOR_TENSORS, OBJECTIVE_WEIGHT_DEFINITION
    from reproduction.common8.topology import monitor_scope
    def sha(value): return hashlib.sha256(value.encode()).hexdigest()
    attempt_id = 'a' * 32; gate = sha('gate'); plan = sha('plan')
    mask = torch.zeros((32, 5), dtype=torch.float64); weights = torch.zeros_like(mask)
    advantages = torch.ones_like(mask)
    monitor_count = 32 // world_size
    local_lengths = sorted([2, 3, 4, 5] * (monitor_count // 4))
    for row in range(32):
        length = local_lengths[row % monitor_count]; mask[row, :length] = 1
        if row % 4 in (0, 2): advantages[row] = -1
    # Match the worker's per-micro denominator, independently of the reader's
    # expected mass. Do not turn a desired8/16 into synthetic input statistics.
    for rank in range(world_size):
        local = mask[rank * monitor_count:(rank + 1) * monitor_count]
        for offset in range(0, monitor_count, actor_micro):
            micro = local[offset:offset + actor_micro]
            begin = rank * monitor_count + offset
            weights[begin:begin + len(micro)] = micro / micro.sum().clamp_min(1)
    reports = [monitor(torch.full_like(mask, delta), advantages, mask, weights,
        c=3., kl_log_bound=30., target_kl=.2, negative_tail_limit=.01) for delta in (0., 1.)]
    actual = dict(step=1, run_id='component', attempt_id=attempt_id, attempt_index=1,
        optimizer_writes_before=0, optimizer_writes_after=1, opportunities_processed=20,
        behavior_probability_gate_sha256=gate)
    binding = dict(schema_version=1, role='production', policy_safety_mode=SAFETY_MODE,
        run_id='component', calibration_sha256=plan, outer_count=1, initial_optimizer_writes=0,
        candidate_thresholds=FORMAL_SAFEGUARDS, complete_attempts=[actual])
    geometry = dict(profile='common8', training_world_size=world_size,
                    actor_micro_batch_per_rank=actor_micro)
    if world_size != 8 or actor_micro != 2:
        binding.update(training_world_size=world_size, actor_micro_batch_per_rank=actor_micro)
    ranks = {}
    for rank in range(world_size):
        rows = []; count = 0
        identity = dict(selection_indices=list(range(monitor_count)),
            actor_token_lengths=local_lengths,
            tensor_sha256={name: sha(name + str(rank)) for name in MONITOR_TENSORS})
        for mini in range(20):
            prior = count; written = mini == 0
            context = dict(logical_update=1, attempt_id=attempt_id, attempt_index=1,
                minibatch_index=mini, prior_optimizer_writes=prior, behavior_probability_gate_sha256=gate)
            before, after = reports[int(mini > 0)], reports[1]
            events = []
            for stage, stats, writes in (('before_minibatch', before, prior),
                                         ('after_minibatch', after, prior + int(written))):
                if stage == 'after_minibatch' and written:
                    rows.append(dict(kind='OPTIMIZER_WRITE', run_id='component', rank=rank, context=copy.deepcopy(context),
                        write_count=1, gradient_norm_before=2., gradient_norm_after=1., learning_rates=[0.]))
                rows.append(dict(kind='POLICY_MONITOR', run_id='component', rank=rank, context=copy.deepcopy(context),
                    stage=stage, optimizer_writes=writes, calibration_sha256=plan,
                    monitor_identity=copy.deepcopy(identity), rejection_count=mini + int(not stats['allowed']),
                    objective_weight_definition=OBJECTIVE_WEIGHT_DEFINITION, monitoring_scope=monitor_scope(geometry),
                    **copy.deepcopy(stats)))
                if not stats['allowed']: events.append(dict(stage=stage, reasons=stats['reasons']))
            count += int(written)
            rows.append(dict(kind='OPPORTUNITY', run_id='component', rank=rank, context=copy.deepcopy(context),
                opportunity_index=mini, processed=True,
                status='written_with_post_guard' if written else 'skipped_guard',
                optimizer_write_succeeded=written, optimizer_writes_before=prior, optimizer_writes_after=count,
                guard_stage=events[0]['stage'], guard_reasons=events[0]['reasons'], guard_events=copy.deepcopy(events)))
        rows.append(dict(kind='OUTER_BUDGET', run_id='component', rank=rank, logical_update=1,
            attempt_id=attempt_id, attempt_index=1, local_trajectories=2560 // world_size,
            local_mini=128 // world_size, local_micro=actor_micro,
            opportunities_processed=20, optimizer_writes=1, optimizer_writes_before=0, optimizer_writes_after=1,
            cumulative_writes=1))
        ranks[rank] = rows
    return ranks, binding


class InlineActualOptimizerTests(unittest.TestCase):
    def test_skipped_opportunities_leave_actual_adam_counter_and_scheduler_lr_distinct(self):
        # At magnitude1 the second consumed LR is below half a float32 ULP.
        # Keep real float32 Adam, using a representable production-like scale.
        parameter = torch.nn.Parameter(torch.tensor([.01], dtype=torch.float32))
        initial_value=parameter.item()
        optimizer = torch.optim.Adam([parameter], lr=consumed_lr(1))
        validate_optimizer_state(optimizer, 1, 0, complete=False)
        opportunities = 20; actual_writes = 0
        for opportunity in range(opportunities):
            # Actual skip decisions exercise counts; this is no physical training claim.
            if opportunity in (1, 4, 9):
                optimizer.zero_grad(); continue
            parameter.grad = torch.ones_like(parameter); optimizer.step(); actual_writes += 1
        self.assertEqual(actual_writes, 17)
        self.assertEqual(parameter.item(), initial_value)  # LR0 still creates real Adam steps
        self.assertEqual(optimizer.state[parameter]['step'].item(), 17)
        validate_optimizer_state(optimizer, 1, actual_writes, complete=True)
        optimizer.param_groups[0]['lr'] = consumed_lr(2)
        validate_optimizer_state(optimizer, 2, 17, complete=False)
        parameter.grad = torch.ones_like(parameter); optimizer.step()
        self.assertEqual(optimizer.state[parameter]['step'].item(), 18)
        self.assertLess(parameter.item(), initial_value)
        validate_optimizer_state(optimizer, 2, 18, complete=True)

    def test_state_steps_and_wrong_lr_reject_without_mutation(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.Adam([parameter], lr=consumed_lr(1))
        parameter.grad = torch.ones_like(parameter); optimizer.step()
        before = parameter.detach().clone()
        with self.assertRaises(AssertionError):
            validate_optimizer_state(optimizer, 1, 0, complete=False)
        optimizer.param_groups[0]['lr'] = 1e-6
        with self.assertRaises(AssertionError):
            validate_optimizer_state(optimizer, 1, 1, complete=True)
        self.assertTrue(torch.equal(parameter, before))

    def test_sampling_kl_point2_brake_keeps_reference_regularization_independent(self):
        kwargs = dict(c=3., kl_log_bound=30., target_kl=.2, negative_tail_limit=.01,
                      execution_role='production')
        advantage = torch.ones((2, 3), dtype=torch.float32)
        mask = torch.ones_like(advantage)
        weights = torch.ones_like(advantage)
        admitted = monitor(torch.full_like(advantage, .4), advantage, mask, weights, **kwargs)
        rejected = monitor(torch.full_like(advantage, 1.), advantage, mask, weights, **kwargs)
        self.assertTrue(admitted['allowed'])
        self.assertGreater(admitted['sampled_old_policy_kl'], .01)
        self.assertFalse(rejected['allowed'])
        self.assertEqual(rejected['reasons'], ['old_policy_kl'])
        self.assertTrue(torch.equal(advantage, torch.ones_like(advantage)))


class InlineProspectiveMethodTests(unittest.TestCase):
    def fixture(self):
        run = dict(source_commit='a' * 40, source_sha256={'one.py': 'b' * 64}, started_unix=20.)
        method = dict(schema_version=1, role='formal_inline_training_method',
            source_commit=run['source_commit'], source_map_sha256=document_sha256(run['source_sha256']),
            created_unix=10., policy_safety_mode=SAFETY_MODE, target_updates=200, warmup_steps=57,
            same_process_after_five=True, fresh_states=True, optimizer_counting=FORMAL_COUNTING,
            safeguards=copy.deepcopy(FORMAL_SAFEGUARDS),
            negative_tail_semantics='prospective operational candidate; not calibrated health')
        return run, method

    def test_exact_prospective_point2_and_opportunity_counting(self):
        run, method = self.fixture(); verify_method(method, run)
        self.assertEqual(method['safeguards']['target_old_policy_kl'], .2)
        self.assertEqual(method['optimizer_counting'], 'all20_opportunities_processed_and_at_least_one_write')

    def test_old_threshold_retroactive_method_state_resume_and_wrong_source_reject(self):
        for key, value in (('created_unix', 21.), ('source_commit', 'c' * 40),
                           ('fresh_states', False), ('optimizer_counting', 'twenty_writes')):
            run, method = self.fixture(); method[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError): verify_method(method, run)
        run, method = self.fixture(); method['safeguards']['target_old_policy_kl'] = .01
        with self.assertRaises(AssertionError): verify_method(method, run)


class FormalOpportunityEvidenceTests(unittest.TestCase):
    def test_actual_cpu_monitor_brake_one_write_and_nineteen_skips_count_one_outer(self):
        from reproduction.common8.safety_calibration_evidence import verify_formal_segment
        rows, binding = guarded_opportunity_fixture()
        proof = verify_formal_segment(rows, binding)
        self.assertEqual(proof['complete_outer_count'], 1)
        self.assertEqual(proof['processed_opportunities_per_rank'], 20)
        self.assertEqual(proof['completed_outer_actual_writes_per_rank'], 1)
        self.assertEqual(len(proof['global_monitors']), 40)
        self.assertEqual(proof['guard_triggered_monitor_count'], 39)
        self.assertFalse(proof['production_admission'])

    def test_missing_opportunity_fake_write_rejected_prewrite_or_divergent_rank_reject(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError, verify_formal_segment
        for mutation in ('missing', 'fake_write', 'divergent', 'pre_write'):
            rows, binding = guarded_opportunity_fixture()
            if mutation == 'missing': rows[7].pop(3)
            elif mutation == 'fake_write': rows[7][3]['optimizer_write_succeeded'] = False
            elif mutation == 'divergent': rows[7][3]['guard_events'] = []
            else:
                rows[7][0].update(allowed=False, reasons=['old_policy_kl'])
            with self.subTest(mutation=mutation), self.assertRaises((EvidenceError, KeyError)):
                verify_formal_segment(rows, binding)


def attempt_history_fixture(base, last=5, skipped_before=(1, 3), world_size=8, actor_micro=2):
    """CPU monitor statistics and fabricated owned receipts for the pure reader.

    The full2560 native sampling gate is isolated here, not replaced with a
    fabricated physical acceptance claim. Its independent real-byte tests stay
    required. No receipt in this fixture is native training evidence.
    """
    from reproduction.common8.policy_calibration_plan import document_sha256
    templates, binding = guarded_opportunity_fixture(world_size, actor_micro)
    run = dict(profile='common8', training_world_size=world_size,
               run_id=binding['run_id'], policy_safety_mode=SAFETY_MODE)
    if world_size != 8 or actor_micro != 2:
        run['actor_micro_batch_per_rank'] = actor_micro
    plan = binding['calibration_sha256']; (base / 'logs').mkdir(exist_ok=True)
    attempts = []; records = []; physical = 0; skipped = 0
    for outer in range(1, last + 1):
        for complete in ([False, True] if outer in skipped_before else [True]):
            ordinal = len(attempts) + 1; count = 20 if complete else 0
            skipped += int(not complete)
            attempt = dict(run_id=run['run_id'], slurm_job_id='123', logical_update=outer,
                attempt_id=f'{ordinal:032x}', attempt_index=ordinal,
                optimizer_writes_before=physical, optimizer_writes_after=physical + count,
                optimizer_writes_this_attempt=count, opportunities_processed=20,
                opportunity_statuses=['written' if complete else 'skipped_guard'] * 20,
                complete_outer=complete, status='complete' if complete else 'recoverable_guard',
                all_ranks_agreed=True, real_rank_count=world_size, scheduler_advances=int(complete),
                completed_outer=outer if complete else outer - 1,
                guard_stage=None if complete else 'all_update_opportunities_skipped',
                guard_reasons=[] if complete else ['no_successful_adam_write'],
                retained_partial_adam_writes=not complete, no_automatic_rollback=True,
                cumulative_aborted_attempts=skipped)
            attempts.append(attempt)
            if complete:
                records.append(dict(step=outer, run_id=run['run_id'], attempt_id=attempt['attempt_id'],
                    attempt_index=ordinal, optimizer_writes_before=physical, optimizer_writes_after=physical + count,
                    opportunities_processed=20, actual_optimizer_writes=count,
                    metrics={'actor/optimizer_steps_this_update':count,
                             'actor/update_opportunities_this_attempt':20,
                             'actor/skipped_update_opportunities':20-count},
                    behavior_probability_gate_sha256=document_sha256(ordinal)))
            physical += count
    def write(path, rows):
        path.write_text(''.join(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n' for row in rows))
    write(base / 'logs/actual-attempts.jsonl', attempts)
    scheduler_receipts = {}
    for rank in range(world_size):
        workers = []; schedulers = []
        for attempt in attempts:
            outer = attempt['logical_update']; ordinal = attempt['attempt_index']; complete = attempt['complete_outer']
            before = attempt['optimizer_writes_before']; gate = document_sha256(ordinal)
            for mini in range(20):
                prior = before + (mini if complete else 0)
                context = dict(logical_update=outer, attempt_id=attempt['attempt_id'], attempt_index=ordinal,
                    minibatch_index=mini, prior_optimizer_writes=prior, behavior_probability_gate_sha256=gate)
                # The all-zero case is an actual supported numerical skip:
                # unchanged policy monitors stay allowed, before_backward guard
                # retains parameters, and no Adam receipt is invented.
                for stage, writes in (('before_minibatch', prior), ('after_minibatch', prior + int(complete))):
                    if complete and stage == 'after_minibatch':
                        workers.append(dict(kind='OPTIMIZER_WRITE', run_id=run['run_id'], rank=rank,
                            context=copy.deepcopy(context), write_count=prior + 1,
                            learning_rates=[consumed_lr(outer)], gradient_norm_before=2., gradient_norm_after=1.))
                    monitor_row = copy.deepcopy(templates[rank][0])
                    monitor_row.update(context=copy.deepcopy(context), stage=stage, optimizer_writes=writes,
                                       rejection_count=0)
                    workers.append(monitor_row)
                guard_events = [] if complete else [dict(stage='before_backward',
                    reasons=['nonfinite_or_empty_effective_numerical_input'])]
                workers.append(dict(kind='OPPORTUNITY', run_id=run['run_id'], rank=rank,
                    context=copy.deepcopy(context), opportunity_index=mini, processed=True,
                    status='written' if complete else 'skipped_guard', optimizer_write_succeeded=complete,
                    optimizer_writes_before=prior, optimizer_writes_after=prior + int(complete),
                    guard_stage=None if complete else 'before_backward',
                    guard_reasons=[] if complete else ['nonfinite_or_empty_effective_numerical_input'],
                    guard_events=guard_events))
            terminal = dict(kind='OUTER_BUDGET' if complete else 'ABORTED_ATTEMPT',
                run_id=run['run_id'], rank=rank, logical_update=outer,
                attempt_id=attempt['attempt_id'], attempt_index=ordinal, opportunities_processed=20,
                optimizer_writes=attempt['optimizer_writes_this_attempt'], cumulative_writes=attempt['optimizer_writes_after'])
            if complete:
                terminal.update(local_trajectories=2560 // world_size, local_mini=128 // world_size, local_micro=actor_micro,
                    optimizer_writes_before=before, optimizer_writes_after=attempt['optimizer_writes_after'])
            else:
                terminal.update(context=copy.deepcopy(context), writes_before=before,
                    behavior_probability_gate_sha256=gate, guard_stage=attempt['guard_stage'],
                    guard_reasons=attempt['guard_reasons'], recoverable_guard=True, complete_outer=False,
                    scheduler_advances=0, retained_partial_adam_writes=True, no_automatic_rollback=True)
            workers.append(terminal)
            for kind in (['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'] if complete else ['BEGIN', 'ABORTED_ATTEMPT']):
                end = kind != 'BEGIN'; advanced = kind == 'COMPLETE'
                schedulers.append(dict(kind=kind, run_id=run['run_id'], rank=rank, slurm_job_id='123',
                    logical_update=outer, attempt_id=attempt['attempt_id'], attempt_index=ordinal,
                    batch_gate_sha256=gate, writes_before=before,
                    writes_after=attempt['optimizer_writes_after'] if end else before,
                    actual_optimizer_writes=attempt['optimizer_writes_this_attempt'] if end else 0,
                    opportunities_processed=20 if end else 0, optimizer_opportunities=20,
                    completed_outer_before=outer - 1, completed_outer_after=outer if advanced else outer - 1,
                    scheduler_advances=int(advanced)))
        write(base / 'logs' / f'optimizer-rank-{rank}.jsonl', workers)
        write(base / 'logs' / f'scheduler-rank-{rank}.jsonl', schedulers)
        scheduler_receipts[str(rank)] = dict(selected_rows=schedulers)
    return run, plan, records, scheduler_receipts


class AllAttemptMonitorAuditTests(unittest.TestCase):
    def test_real_cpu_post_write_policy_guard_retains_one_write_and_exact_rejections(self):
        from reproduction.common8.formal_attempt_audit import _audit_monitor_rows
        rows, binding = guarded_opportunity_fixture()
        for rank in range(8):
            rejected = 0
            for row in rows[rank]:
                if row['kind'] == 'POLICY_MONITOR':
                    rejected += int(not row['allowed']); row['rejection_count'] = rejected
        actual = binding['complete_attempts'][0]
        attempt = dict(logical_update=1, attempt_id=actual['attempt_id'], attempt_index=1,
            optimizer_writes_before=0, optimizer_writes_after=1, optimizer_writes_this_attempt=1,
            complete_outer=True, behavior_probability_gate_sha256=actual['behavior_probability_gate_sha256'],
            opportunity_statuses=[row['status'] for row in rows[0] if row['kind'] == 'OPPORTUNITY'])
        run = dict(profile='common8', training_world_size=8, run_id=binding['run_id'])
        proof = _audit_monitor_rows(rows, binding, attempt, 0, run=run)
        self.assertEqual(proof['global_monitor_count'], 40)
        self.assertEqual(proof['guard_triggered_monitor_count'], 39)
        self.assertEqual(proof['rejection_count'], 39)
        rows[7][2]['rejection_count'] = 0
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        with self.assertRaises(EvidenceError): _audit_monitor_rows(rows, binding, attempt, 0, run=run)

    def test_real_cpu_incomplete_monitor_diagnostics_keep_none_kl_and_unavailable_totals(self):
        from reproduction.common8.formal_attempt_audit import _monitor_statistics
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        mask = torch.ones((32, 4), dtype=torch.float64); advantages = -torch.ones_like(mask)
        for delta, weight, expected_reason in ((31., 1., 'delta_above_kl_log_bound'),
                                               (0., 1e308, 'nonfinite_global_totals')):
            with self.subTest(reason=expected_reason):
                report = monitor(torch.full_like(mask, delta), advantages, mask, torch.full_like(mask, weight),
                    c=3., kl_log_bound=30., target_kl=.2, negative_tail_limit=.01)
                self.assertFalse(report['allowed']); self.assertFalse(report['kl_estimate_complete'])
                self.assertIn(expected_reason, report['reasons'])
                checked = _monitor_statistics(report)
                self.assertIsNone(checked['sampled_old_policy_kl'])
                if expected_reason == 'nonfinite_global_totals': self.assertIsNone(checked['effective_weight_mass'])
                bad = copy.deepcopy(report); bad['sampled_old_policy_kl'] = 0.
                with self.assertRaises(EvidenceError): _monitor_statistics(bad)
                bad = copy.deepcopy(report); bad['reasons'].append('old_policy_kl')
                with self.assertRaises(EvidenceError): _monitor_statistics(bad)

    def verify(self, base, run, plan, records, schedulers, last=None):
        from reproduction.common8.formal_attempt_audit import verify_attempt_history
        last = len(records) if last is None else last
        selected = records[:last]; cutoff = selected[-1]['attempt_index']
        receipts = {rank: dict(selected_rows=[row for row in receipt['selected_rows'] if row['attempt_index'] <= cutoff])
                    for rank, receipt in schedulers.items()}
        def gate(_base, _run, step, update_record=None, attempt_id=None, attempt_index=None):
            ordinal = update_record['attempt_index'] if update_record is not None else attempt_index
            return dict(proof_sha256=document_sha256(ordinal))
        with patch('reproduction.common8.behavior_runtime.read_gate_receipt', side_effect=gate):
            return verify_attempt_history(base, run, selected, receipts, last=last, plan_sha256=plan)

    def change_rows(self, path, mutate):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))

    def test_all_zero_attempts_have_real40_monitors20_opportunities_and_no_write(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root); fixture = attempt_history_fixture(base)
            proof = self.verify(base, *fixture)
            self.assertEqual(proof['completed_outer_count'], 5)
            self.assertEqual(proof['actual_attempt_count'], 7)
            self.assertEqual(proof['fully_skipped_attempt_count'], 2)
            self.assertEqual(proof['completed_outer_global_monitor_count'], 200)
            self.assertEqual(proof['retry_global_monitor_count'], 80)
            self.assertEqual(proof['unique_global_monitor_count'], 280)
            self.assertEqual(proof['processed_opportunities_per_rank'], 140)
            self.assertEqual(proof['actual_optimizer_writes_per_rank'], 100)
            self.assertFalse(proof['production_admission'])

    def test_paired_restore_indexes_actual_suffix_relative_to_authenticated_parent(self):
        # The pure reader receives only this execution's actual suffix. The
        # restored prefix itself is supplied by the independent receipt reader,
        # isolated here so fabricated CPU rows cannot claim CUDA acceptance.
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            run, plan, records, schedulers = attempt_history_fixture(base, last=6)
            prefix = dict(completed_outer=3, writes=60, attempt_index=5, rejections=0, aborted=2)
            run['full_state_restore'] = {'kind': 'question_probe_pair'}
            for path in (base / 'logs').glob('*.jsonl'):
                self.change_rows(path, lambda rows: rows.__setitem__(slice(None), [row for row in rows
                    if (row.get('context') or row)['attempt_index'] > prefix['attempt_index']]))
            suffix = records[3:]
            suffix_schedulers = {rank: dict(selected_rows=[row for row in receipt['selected_rows']
                if row['attempt_index'] > prefix['attempt_index']]) for rank, receipt in schedulers.items()}
            with patch('search_g1.full_state_restore.verified_runtime_prefix', return_value=prefix):
                proof = self.verify(base, run, plan, suffix, suffix_schedulers, last=6)
            self.assertEqual(proof['source_completed_outer'], 3)
            self.assertEqual(proof['source_actual_optimizer_writes_per_rank'], 60)
            self.assertEqual(proof['completed_outer_count'], 6)
            self.assertEqual(proof['actual_attempt_count'], 3)
            self.assertEqual(proof['actual_optimizer_writes_per_rank'], 120)
            self.assertEqual([row['logical_update'] for row in proof['attempts']], [4, 5, 6])
            self.assertEqual(proof['completed_outer_global_monitor_count'], 120)
            self.assertFalse(proof['production_admission'])

    def test_delete_or_change_all_zero_monitor_guard_identity_or_rank_statistics_rejects(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        for mutation in ('one_missing', 'all_missing', 'arithmetic', 'decision', 'identity', 'guard', 'rank_statistics'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                base = Path(root); fixture = attempt_history_fixture(base)
                path = base / 'logs/optimizer-rank-3.jsonl'
                def mutate(rows):
                    if mutation == 'one_missing': rows.pop(0)
                    elif mutation == 'all_missing':
                        rows[:] = [row for row in rows if not (row['kind'] == 'POLICY_MONITOR' and row['context']['attempt_index'] == 1)]
                    elif mutation == 'arithmetic': rows[0]['weighted_k3_numerator'] = .1
                    elif mutation == 'decision': rows[0].update(allowed=False, reasons=['old_policy_kl'])
                    elif mutation == 'identity': rows[1]['monitor_identity']['tensor_sha256']['advantages'] = 'f' * 64
                    elif mutation == 'guard': rows[2]['guard_events'] = []
                    else:
                        # Locally consistent arithmetic, but different global
                        # rank replicas and unchanged-policy pre/post reports.
                        rows[0].update(weighted_k3_numerator=1.6, sampled_old_policy_kl=.1)
                self.change_rows(path, mutate)
                with self.assertRaises(EvidenceError): self.verify(base, *fixture)

    def test_driver_attempt_omission_counter_status_or_opportunity_forgery_rejects(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        for mutation in ('omission', 'duplicate', 'counter', 'status', 'opportunity', 'retry_count'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                base = Path(root); fixture = attempt_history_fixture(base)
                def mutate(rows):
                    if mutation == 'omission': rows.pop(0)
                    elif mutation == 'duplicate': rows.insert(1, copy.deepcopy(rows[0]))
                    elif mutation == 'counter': rows[0]['optimizer_writes_after'] = 1
                    elif mutation == 'status': rows[0]['complete_outer'] = True
                    elif mutation == 'opportunity': rows[0]['opportunity_statuses'][0] = 'written'
                    else: rows[0]['cumulative_aborted_attempts'] = 0
                self.change_rows(base / 'logs/actual-attempts.jsonl', mutate)
                with self.assertRaises(EvidenceError): self.verify(base, *fixture)

    def test_later_sixth_outer_monitor_is_audited_and_bad_clip_or_gate_rejects(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        for mutation in ('threshold', 'clip', 'gate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                base = Path(root); fixture = attempt_history_fixture(base, last=6)
                self.assertEqual(self.verify(base, *fixture)['completed_outer_global_monitor_count'], 240)
                def mutate(rows):
                    row = next(row for row in rows if (row.get('context') or row)['logical_update'] == 6
                               and row['kind'] == ('OPTIMIZER_WRITE' if mutation == 'clip' else 'POLICY_MONITOR'))
                    if mutation == 'threshold': row.update(weighted_k3_numerator=16., sampled_old_policy_kl=1.)
                    elif mutation == 'clip': row['gradient_norm_after'] = 1.01
                    else: row['context']['behavior_probability_gate_sha256'] = 'f' * 64
                self.change_rows(base / 'logs/optimizer-rank-7.jsonl', mutate)
                with self.assertRaises(EvidenceError): self.verify(base, *fixture)

    def test_completed_scalar_write_ledger_cannot_disagree_with_actual_attempt(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        for mutation in ('actual_writes', 'metric_writes', 'metric_skips', 'metric_opportunities'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                base = Path(root); run, plan, records, schedulers = attempt_history_fixture(base)
                if mutation == 'actual_writes': records[0]['actual_optimizer_writes'] = 19
                else:
                    key = {'metric_writes':'actor/optimizer_steps_this_update',
                           'metric_skips':'actor/skipped_update_opportunities',
                           'metric_opportunities':'actor/update_opportunities_this_attempt'}[mutation]
                    records[0]['metrics'][key] += 1
                with self.assertRaises(EvidenceError): self.verify(base, run, plan, records, schedulers)

    def test_full200_history_requires_the_two_hundredth_outer_monitor(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        with tempfile.TemporaryDirectory() as root:
            base = Path(root); fixture = attempt_history_fixture(base, last=200, skipped_before=(1,))
            proof = self.verify(base, *fixture)
            self.assertEqual(proof['completed_outer_count'], 200)
            self.assertEqual(proof['completed_outer_global_monitor_count'], 8000)
            self.assertEqual(proof['retry_global_monitor_count'], 40)
            self.assertEqual(proof['actual_optimizer_writes_per_rank'], 4000)
            driver = base / 'logs/actual-attempts.jsonl'; original_driver = driver.read_bytes()
            def append_after_final(rows):
                row = copy.deepcopy(rows[-1]); row.update(attempt_index=202, attempt_id=f'{202:032x}', logical_update=201)
                rows.append(row)
            self.change_rows(driver, append_after_final)
            with self.assertRaises(EvidenceError): self.verify(base, *fixture)
            driver.write_bytes(original_driver)
            def mutate(rows):
                index = next(index for index, row in enumerate(rows) if row['kind'] == 'POLICY_MONITOR'
                             and row['context']['logical_update'] == 200)
                rows.pop(index)
            self.change_rows(base / 'logs/optimizer-rank-7.jsonl', mutate)
            with self.assertRaises(EvidenceError): self.verify(base, *fixture)

    def test_first_five_sealed_prefix_survives_future_append_and_binds_prior_retry(self):
        from reproduction.common8.safety_calibration_evidence import EvidenceError
        with tempfile.TemporaryDirectory() as root:
            base = Path(root); fixture = attempt_history_fixture(base)
            before = self.verify(base, *fixture)
            previous = {path.name: path.read_bytes() for path in (base / 'logs').iterdir()}
            extended = attempt_history_fixture(base, last=6)
            for name, prefix in previous.items():
                self.assertTrue((base / 'logs' / name).read_bytes().startswith(prefix))
            after = self.verify(base, *extended, last=5)
            self.assertEqual(before['selected_evidence_sha256'], after['selected_evidence_sha256'])
            self.assertEqual(before, after)
            # Deleting an earlier zero attempt's monitors invalidates that sealed
            # first-five proof even though all five successful outers remain.
            self.change_rows(base / 'logs/optimizer-rank-0.jsonl',
                lambda rows: rows.__setitem__(slice(None), [row for row in rows if not
                    (row['kind'] == 'POLICY_MONITOR' and row['context']['attempt_index'] == 1)]))
            with self.assertRaises(EvidenceError): self.verify(base, *extended, last=5)


if __name__ == '__main__':
    unittest.main()
