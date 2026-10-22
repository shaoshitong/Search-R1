"""Fabricated log inputs test structural checks; no native execution is claimed."""

import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8.safety_calibration_evidence import (
    EvidenceError, MONITOR_TENSORS, OBJECTIVE_WEIGHT_DEFINITION,
    FLOAT32_EPSILON, GRADIENT_CLIP_FLOAT32_TOLERANCE,
    MONITORING_SCOPE, QUANTILE_METHOD, verify_segment,
)


class ActualMonitorTensorTests(unittest.TestCase):
    def test_reference_free_actual_actor_monitor_retains_required_inputs(self):
        from reproduction.common8.safety_calibration_evidence import _identity
        identity=dict(selection_indices=[0,1,2,3],actor_token_lengths=[1,2,3,4],
                      tensor_sha256={key:'a'*64 for key in MONITOR_TENSORS})
        _identity(identity)
        identity['tensor_sha256'].pop('ref_log_prob');_identity(identity)
        identity['tensor_sha256'].pop('old_log_probs')
        with self.assertRaises(EvidenceError):_identity(identity)
        identity['tensor_sha256']['old_log_probs']='a'*64
        identity['tensor_sha256']['unknown_tensor']='a'*64
        with self.assertRaises(EvidenceError):_identity(identity)


class OptimizerGradientEvidenceTests(unittest.TestCase):
    def run_binding(self):
        return dict(performance_profile='lean_v1', easy2560_selection=True, profile='common8',
            execution_site='hpc3', training_world_size=4, data_selection_mode='mixed_first',
            actor_logical_micro_batch_per_rank=4, extra_policy_monitor_forwards=False)

    def unmeasured(self, before=2.):
        return dict(gradient_norm_before=before, gradient_norm_after=None,
            gradient_norm_after_measured=False, gradient_norm_after_source='not_measured',
            gradient_clip_coefficient=min(1.,1./(before+1e-6)), gradient_clip_threshold=1.)

    def test_explicit_bound_lean_unmeasured_coefficient_is_not_a_measured_clip_proof(self):
        from reproduction.common8.safety_calibration_evidence import optimizer_gradient_evidence
        for before in (0.,.2,2.,1e120):
            result=optimizer_gradient_evidence(self.unmeasured(before),run=self.run_binding())
            self.assertIsNone(result['gradient_norm_after'])
            self.assertIs(result['gradient_norm_after_measured'],False)
            self.assertIs(result['measured_post_clip_bound_proven'],False)
            self.assertIs(result['coefficient_arithmetic_consistent'],True)

    def test_legacy_missing_or_nonfinite_post_norm_remains_rejected(self):
        from reproduction.common8.safety_calibration_evidence import optimizer_gradient_evidence
        for run in (None,{},dict(self.run_binding(),performance_profile=None)):
            with self.assertRaises(ValueError):optimizer_gradient_evidence(self.unmeasured(),run=run)
        for after in (None,float('inf'),float('nan'),1.+2.*GRADIENT_CLIP_FLOAT32_TOLERANCE):
            with self.assertRaises(EvidenceError):
                optimizer_gradient_evidence(dict(gradient_norm_before=2.,gradient_norm_after=after))
        old=optimizer_gradient_evidence(dict(gradient_norm_before=2.,gradient_norm_after=1.))
        self.assertIs(old['measured_post_clip_bound_proven'],True)

    def test_unmeasured_schema_and_coefficient_are_explicit_and_arithmetically_consistent(self):
        from reproduction.common8.safety_calibration_evidence import optimizer_gradient_evidence
        for mutation in (dict(gradient_norm_before=float('inf')),dict(gradient_norm_after=.9),
                dict(gradient_norm_after_measured=True),dict(gradient_norm_after_source='derived'),
                dict(gradient_clip_threshold=2.),dict(gradient_clip_coefficient=.8),
                dict(gradient_clip_coefficient=float('nan'))):
            with self.subTest(mutation=mutation),self.assertRaises(EvidenceError):
                optimizer_gradient_evidence(dict(self.unmeasured(),**mutation),run=self.run_binding())
        missing=self.unmeasured();missing.pop('gradient_norm_after_measured')
        with self.assertRaises(EvidenceError):optimizer_gradient_evidence(missing,run=self.run_binding())

    def test_wrong_native_scope_cannot_claim_lean_unmeasured_acceptance(self):
        from reproduction.common8.safety_calibration_evidence import optimizer_gradient_evidence
        for mutation in (dict(training_world_size=8),dict(extra_policy_monitor_forwards=True),
                         dict(easy2560_selection=False)):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                optimizer_gradient_evidence(self.unmeasured(),run=dict(self.run_binding(),**mutation))

    def test_transaction_audit_reports_unmeasured_norms_without_measured_clip_claim(self):
        from reproduction.common8.formal_attempt_audit import _audit_transaction_rows, consumed_lr
        run=self.run_binding();binding=dict(run_id='a'*32,calibration_sha256='b'*64)
        attempt=dict(logical_update=1,attempt_id='c'*32,attempt_index=1,
            behavior_probability_gate_sha256='d'*64,opportunities_processed=20,
            optimizer_writes_before=0,optimizer_writes_after=20,opportunity_statuses=['written']*20,
            complete_outer=True,actual_monitor_forward_count=0,reused_monitor_forward_count=0)
        ranks={}
        for rank in range(4):
            rows=[];header=dict(rank=rank,run_id=binding['run_id'])
            for mini in range(20):
                context=dict(logical_update=1,minibatch_index=mini,prior_optimizer_writes=mini,
                    attempt_id=attempt['attempt_id'],attempt_index=1,
                    behavior_probability_gate_sha256=attempt['behavior_probability_gate_sha256'])
                rows.append(dict(header,kind='OPTIMIZER_WRITE',context=context,write_count=mini+1,
                    learning_rates=[consumed_lr(1)],**self.unmeasured()))
                rows.append(dict(header,kind='OPPORTUNITY',context=context,opportunity_index=mini,
                    processed=True,status='written',optimizer_write_succeeded=True,
                    optimizer_write_rolled_back=False,optimizer_call_attempted=True,
                    optimizer_writes_before=mini,optimizer_writes_after=mini+1,
                    guard_stage=None,guard_reasons=[]))
            rows.append(dict(header,kind='OUTER_BUDGET',opportunities_processed=20,optimizer_writes=20,
                cumulative_writes=20,extra_policy_monitor_forwards=False,
                actual_monitor_forward_count=0,reused_monitor_forward_count=0))
            ranks[rank]=rows
        report=_audit_transaction_rows(ranks,binding,attempt,0,run=run)['gradient_norm_measurement']
        self.assertEqual(report['actual_write_receipts_checked'],80)
        self.assertEqual(report['measured_post_clip_receipts'],0)
        self.assertEqual(report['explicitly_unmeasured_post_clip_receipts'],80)
        self.assertIs(report['all_post_clip_bounds_measured'],False)
        ranks[0][0]['gradient_clip_coefficient']=.9
        with self.assertRaises(EvidenceError):_audit_transaction_rows(ranks,binding,attempt,0,run=run)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def global_statistics(step):
    # Independently construct eight ranks of four trajectories. Each pair of
    # rows uses its own masked-mean denominator, matching two-row micros.
    lengths = [2, 3, 4, 5]
    deltas, weights, negative = [], [], []
    for _ in range(8):
        for row, length in enumerate(lengths):
            denominator = sum(lengths[(row // 2) * 2:(row // 2) * 2 + 2])
            for token in range(length):
                deltas.append(2. + step * .001 if row == 2 and token == 0 else .1 + step * .001)
                weights.append(1. / denominator)
                negative.append(row in (0, 2))
    mass = math.fsum(weights)
    numerator = math.fsum(w * (math.expm1(d) - d) for d, w in zip(deltas, weights))
    negative_mass = math.fsum(w for w, n in zip(weights, negative) if n)
    tail = [n and d > math.log(3.) for d, n in zip(deltas, negative)]
    tail_mass = math.fsum(w for w, t in zip(weights, tail) if t)
    negatives = sorted(d for d, n in zip(deltas, negative) if n)
    return dict(
        allowed=True, reasons=[], kl_estimate_complete=True,
        sampled_old_policy_kl=numerator / mass,
        effective_weight_mass=mass, weighted_k3_numerator=numerator,
        negative_effective_weight_mass=negative_mass, tail_effective_weight_mass=tail_mass,
        effective_token_count=len(deltas), negative_token_count=sum(negative),
        negative_tail_count=sum(tail), negative_tail_count_fraction=sum(tail) / sum(negative),
        negative_tail_weight_fraction=tail_mass / negative_mass,
        dual_clip_saturated_objective_weight_fraction=tail_mass / mass,
        invalid_count=0, overflow_count=0, quantile_method=QUANTILE_METHOD,
        negative_log_ratio_p95=negatives[round(.95 * (len(negatives) - 1))],
        negative_log_ratio_p99=negatives[round(.99 * (len(negatives) - 1))],
        negative_log_ratio_max=negatives[-1],
    )


def fabricated_segment(outer_count=1, role='policy_safety_calibration_confirmation'):
    binding = dict(schema_version=1, role=role,
                   run_id='fabricated-structural-test', calibration_sha256=digest('test plan'),
                   outer_count=outer_count, initial_optimizer_writes=0)
    logs = {}
    for rank in range(8):
        rows = []
        for outer in range(1, outer_count + 1):
            identity = dict(selection_indices=[rank, 106 + rank, 213 + rank, 312 + rank],
                            actor_token_lengths=[2, 3, 4, 5],
                            tensor_sha256={name: digest(f'{outer}:{rank}:{name}') for name in MONITOR_TENSORS})
            for mini in range(20):
                prior = (outer - 1) * 20 + mini
                context = dict(logical_update=outer, minibatch_index=mini, prior_optimizer_writes=prior)
                for post in range(2):
                    if post:
                        rows.append(dict(kind='OPTIMIZER_WRITE', run_id=binding['run_id'], rank=rank,
                                         context=copy.deepcopy(context), write_count=prior + 1,
                                         gradient_norm_before=2., gradient_norm_after=.9999995,
                                         learning_rates=[1e-6]))
                    rows.append(dict(kind='POLICY_MONITOR', run_id=binding['run_id'], rank=rank,
                                     execution_role=role,
                                     enforcement=dict(negative_tail='observe_only' if role.endswith('discovery') else 'enforced',
                                                      old_policy_kl='enforced', numerical_finite='enforced', dual_clip='enforced'),
                                     context=copy.deepcopy(context),
                                     stage='after_minibatch' if post else 'before_minibatch',
                                     optimizer_writes=prior + post, calibration_sha256=binding['calibration_sha256'],
                                     monitor_identity=copy.deepcopy(identity), rejection_count=0,
                                     objective_weight_definition=OBJECTIVE_WEIGHT_DEFINITION,
                                     monitoring_scope=MONITORING_SCOPE, **global_statistics(mini + post)))
            rows.append(dict(kind='OUTER_BUDGET', run_id=binding['run_id'], rank=rank,
                             logical_update=outer, local_trajectories=320, local_mini=16,
                             local_micro=2, optimizer_writes=20, cumulative_writes=outer * 20))
        logs[rank] = rows
    return logs, binding


class CalibrationEvidenceTests(unittest.TestCase):
    def test_legacy_three_field_and_new_gate_contexts_coalesce_once(self):
        logs, binding = fabricated_segment(2)
        legacy = verify_segment(logs, binding)
        gates = {outer: digest('fabricated CPU receipt outer ' + str(outer)) for outer in (1, 2)}
        for rows in logs.values():
            for row in rows:
                if row['kind'] in ('POLICY_MONITOR', 'OPTIMIZER_WRITE'):
                    row['context']['behavior_probability_gate_sha256'] = gates[row['context']['logical_update']]
        original = copy.deepcopy(logs)
        result = verify_segment(logs, binding)
        self.assertEqual(len(result['global_monitors']), 80)
        self.assertEqual(result['optimizer_writes_per_rank'], 40)
        self.assertIs(result['production_admission'], False)
        self.assertNotIn('accepted', result)
        for previous, current in zip(legacy['global_monitors'], result['global_monitors']):
            self.assertEqual(current['weighted_k3_numerator'], previous['weighted_k3_numerator'])
            self.assertEqual(current['effective_weight_mass'], previous['effective_weight_mass'])
            self.assertEqual(current['context']['behavior_probability_gate_sha256'],
                             gates[current['context']['logical_update']])
        self.assertEqual(logs, original)

    def test_gate_context_bad_hash_and_unknown_field_rejected(self):
        for value in ('not-a-sha', 'a' * 63, 'g' * 64, None, True):
            logs, binding = fabricated_segment()
            logs[7][0]['context']['behavior_probability_gate_sha256'] = value
            with self.subTest(hash=value), self.assertRaises(EvidenceError):
                verify_segment(logs, binding)
        for include_gate in (False, True):
            logs, binding = fabricated_segment()
            context = logs[7][0]['context']
            if include_gate:
                context['behavior_probability_gate_sha256'] = digest('fabricated CPU receipt')
            context['unknown_context_field'] = 'unbound'
            with self.subTest(include_gate=include_gate), self.assertRaisesRegex(EvidenceError, 'context schema'):
                verify_segment(logs, binding)

    def test_gate_cannot_change_between_checks_writes_or_minibatches(self):
        for index in (0, 1, 2, 3, 31, 59):
            logs, binding = fabricated_segment()
            for rows in logs.values():
                for row in rows:
                    if row['kind'] in ('POLICY_MONITOR', 'OPTIMIZER_WRITE'):
                        row['context']['behavior_probability_gate_sha256'] = digest('fabricated CPU original gate')
            logs[0][index]['context']['behavior_probability_gate_sha256'] = digest('fabricated CPU different gate')
            with self.subTest(event=index), self.assertRaisesRegex(EvidenceError, 'gate changed'):
                verify_segment(logs, binding)

    def test_all_rank_gate_presence_and_hash_must_agree_within_outer(self):
        for rank_seven_gate in (None, digest('fabricated CPU rank seven different gate')):
            logs, binding = fabricated_segment()
            for rank, rows in logs.items():
                for row in rows:
                    if row['kind'] in ('POLICY_MONITOR', 'OPTIMIZER_WRITE'):
                        if rank != 7:
                            row['context']['behavior_probability_gate_sha256'] = digest('fabricated CPU common gate')
                        elif rank_seven_gate is not None:
                            row['context']['behavior_probability_gate_sha256'] = rank_seven_gate
            with self.subTest(rank_seven_gate=rank_seven_gate), self.assertRaisesRegex(EvidenceError, 'across ranks'):
                verify_segment(logs, binding)

    def test_complete_segment_coalesces_global_reports_once(self):
        logs, binding = fabricated_segment()
        result = verify_segment(logs, binding)
        self.assertEqual(result['role'], 'policy_safety_calibration_structural_verification')
        self.assertIs(result['production_admission'], False)
        self.assertNotIn('accepted', result)
        self.assertEqual(result['optimizer_writes_per_rank'], 20)
        self.assertEqual(len(result['global_monitors']), 40)
        first = result['global_monitors'][0]
        self.assertEqual(first['effective_token_count'], 112)
        self.assertEqual(first['negative_token_count'], 48)
        self.assertEqual(first['negative_tail_count'], 8)
        self.assertAlmostEqual(first['effective_weight_mass'], 16.)
        self.assertAlmostEqual(first['weighted_k3_numerator'], global_statistics(0)['weighted_k3_numerator'])
        self.assertNotEqual(result['monitor_identities'][1][0], result['monitor_identities'][1][7])

    def test_outer_refresh_is_allowed_and_counters_continue(self):
        logs, binding = fabricated_segment(2)
        result = verify_segment(logs, binding)
        self.assertEqual(len(result['global_monitors']), 80)
        self.assertEqual(result['optimizer_writes_per_rank'], 40)
        self.assertNotEqual(result['monitor_identities'][1][0], result['monitor_identities'][2][0])
        self.assertEqual(result['global_monitors'][40]['optimizer_writes'], 20)

    def test_explicit_schema_role_and_zero_start_required(self):
        for key, value in [('schema_version', 2), ('schema_version', True),
                           ('role', 'synthetic_native_component_fixture_not_production'),
                           ('role', 'healthy_pretraining_policy_calibration'),
                           ('initial_optimizer_writes', 1), ('outer_count', 0),
                           ('calibration_sha256', 'fixture')]:
            with self.subTest(key=key, value=value):
                logs, binding = fabricated_segment()
                binding[key] = value
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)
        logs, binding = fabricated_segment(role='policy_safety_calibration_discovery')
        result=verify_segment(logs, binding)
        self.assertIs(result['production_admission'], False)
        self.assertEqual(result['enforcement']['negative_tail'],'observe_only')
        logs[7][0]['enforcement']['negative_tail']='enforced'
        with self.assertRaises(EvidenceError): verify_segment(logs,binding)

    def test_missing_extra_and_wrong_rank_logs_rejected(self):
        for variant in ('missing', 'extra', 'duplicate_payload'):
            with self.subTest(variant=variant):
                logs, binding = fabricated_segment()
                if variant == 'missing':
                    del logs[7]
                elif variant == 'extra':
                    logs[8] = copy.deepcopy(logs[0])
                else:
                    logs[7] = copy.deepcopy(logs[0])
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)

    def test_missing_duplicate_and_reordered_records_rejected(self):
        for variant in ('missing_write', 'missing_post', 'missing_budget', 'duplicate', 'reordered'):
            with self.subTest(variant=variant):
                logs, binding = fabricated_segment()
                if variant.startswith('missing'):
                    logs[7].pop({'missing_write': 1, 'missing_post': 2, 'missing_budget': -1}[variant])
                elif variant == 'duplicate':
                    logs[7].insert(1, copy.deepcopy(logs[7][1]))
                else:
                    logs[7][1], logs[7][2] = logs[7][2], logs[7][1]
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)

    def test_write_context_counters_and_outer_budget_required(self):
        for index, key, value in [(0, 'optimizer_writes', 1), (1, 'write_count', 2),
                                   (2, 'optimizer_writes', 0), (60, 'local_micro', 16),
                                   (60, 'local_mini', 128), (60, 'local_trajectories', 128),
                                   (60, 'optimizer_writes', 19), (60, 'cumulative_writes', 19)]:
            with self.subTest(index=index, key=key):
                logs, binding = fabricated_segment()
                logs[7][index][key] = value
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)
        logs, binding = fabricated_segment()
        logs[7][1]['context']['prior_optimizer_writes'] = 1
        with self.assertRaises(EvidenceError):
            verify_segment(logs, binding)

    def test_every_optimizer_learning_rate_and_norm_checked(self):
        for rates in ([], [1e-6, 2e-6], [float('nan')], [0.], ['1e-6']):
            with self.subTest(rates=rates):
                logs, binding = fabricated_segment()
                logs[7][58]['learning_rates'] = rates
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)
        for name in ('gradient_norm_before', 'gradient_norm_after'):
            logs, binding = fabricated_segment()
            logs[7][58][name] = float('inf')
            with self.assertRaises(EvidenceError):
                verify_segment(logs, binding)

    def test_post_clip_bound_uses_float32_rounding_tolerance(self):
        logs, binding = fabricated_segment()
        logs[7][1]['gradient_norm_after'] = 1. + GRADIENT_CLIP_FLOAT32_TOLERANCE
        self.assertIs(verify_segment(logs, binding)['production_admission'], False)
        for value in (1. + 2. * GRADIENT_CLIP_FLOAT32_TOLERANCE, 2.):
            with self.subTest(value=value):
                logs, binding = fabricated_segment()
                logs[7][58]['gradient_norm_after'] = value
                with self.assertRaisesRegex(EvidenceError, 'post-clip norm'):
                    verify_segment(logs, binding)

    def test_all_ranks_coherent_float_jump_without_write_rejected(self):
        for field in ('weighted_k3_numerator', 'quantiles'):
            with self.subTest(field=field):
                logs, binding = fabricated_segment()
                # Row2 is post(mini0), row3 is pre(mini1): same actual write count.
                for rows in logs.values():
                    row = rows[3]
                    if field == 'weighted_k3_numerator':
                        row[field] += .01
                        row['sampled_old_policy_kl'] = row[field] / row['effective_weight_mass']
                    else:
                        for name in ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max'):
                            row[name] += .01
                with self.assertRaisesRegex(EvidenceError, 'Adjacent unchanged-policy float'):
                    verify_segment(logs, binding)

    def test_adjacent_float32_rounding_allowed_with_exact_masses_and_counts(self):
        logs, binding = fabricated_segment()
        for rows in logs.values():
            row = rows[3]
            row['weighted_k3_numerator'] += FLOAT32_EPSILON / 4.
            row['sampled_old_policy_kl'] = row['weighted_k3_numerator'] / row['effective_weight_mass']
        self.assertIs(verify_segment(logs, binding)['production_admission'], False)

    def test_all_ranks_exact_tail_count_or_mass_jump_without_write_rejected(self):
        for field in ('negative_tail_count', 'tail_effective_weight_mass'):
            with self.subTest(field=field):
                logs, binding = fabricated_segment()
                for rows in logs.values():
                    row = rows[3]
                    row[field] += 1 if field == 'negative_tail_count' else FLOAT32_EPSILON / 4.
                    row['negative_tail_count_fraction'] = row['negative_tail_count'] / row['negative_token_count']
                    row['negative_tail_weight_fraction'] = row['tail_effective_weight_mass'] / row['negative_effective_weight_mass']
                    row['dual_clip_saturated_objective_weight_fraction'] = row['tail_effective_weight_mass'] / row['effective_weight_mass']
                with self.assertRaisesRegex(EvidenceError, 'Adjacent unchanged-policy exact'):
                    verify_segment(logs, binding)

    def test_real_cpu_monitor_and_actual_json_writer_readback_compatibility(self):
        # Real CPU monitor()/append()/read_console_form(); the schedule is still
        # fabricated test data, with no GPU, Qwen or optimizer execution claim.
        import torch
        from search_g1.policy_safety import monitor
        from reproduction.common8.worker_evidence import append, read_console_form

        self.assertEqual(FLOAT32_EPSILON, torch.finfo(torch.float32).eps)
        logs, binding = fabricated_segment()
        local_inputs = {}
        for rank in range(8):
            mask = (torch.arange(5)[None, :] < torch.tensor([2, 3, 4, 5])[:, None]).long()
            ids = torch.arange(20).reshape(4, 5) + rank * 20
            local_inputs[rank] = dict(responses=ids, input_ids=ids.clone(), attention_mask=mask,
                                      position_ids=torch.arange(5).repeat(4, 1),
                                      old_log_probs=torch.zeros(4, 5),
                                      advantages=torch.tensor([-1., 1., -1., 1.])[:, None].repeat(1, 5),
                                      loss_mask=mask.clone(), ref_log_prob=torch.zeros(4, 5))
        masks = torch.cat([value['loss_mask'] for value in local_inputs.values()])
        advantages = torch.cat([value['advantages'] for value in local_inputs.values()])
        weights = torch.cat([micro.double() / micro.sum().double() for micro in masks.split(2)])
        reports = {}
        for step in range(21):
            new = torch.full((32, 5), .1 + step * .001, dtype=torch.float32)
            new[2::4, 0] = 2. + step * .001
            old = torch.zeros_like(new)
            delta = torch.where(masks.bool(), new, 0.).double() - torch.where(masks.bool(), old, 0.).double()
            reports[step] = monitor(delta, advantages, masks, weights,
                                    c=3., kl_log_bound=30., target_kl=10., negative_tail_limit=.5,
                                    execution_role=binding['role'])
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, SEARCH_G1_BASE=root):
            base = Path(root)
            (base / 'logs').mkdir()
            (base / 'logs/run-manifest.json').write_text(json.dumps(dict(profile='common8', run_id=binding['run_id'],
                                                                        policy_safety_required=True)))
            for rank, rows in logs.items():
                hashes = {name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                          for name, value in local_inputs[rank].items()}
                for row in rows:
                    payload = {name: copy.deepcopy(value) for name, value in row.items() if name not in ('kind', 'run_id')}
                    if row['kind'] == 'POLICY_MONITOR':
                        payload.update(reports[row['optimizer_writes']])
                        payload['monitor_identity']['tensor_sha256'] = hashes
                    append(row['kind'], payload)
            readback = {rank: [json.loads(line) for line in
                               (base / 'logs' / f'optimizer-rank-{rank}.jsonl').read_text().splitlines()]
                        for rank in range(8)}
            result = verify_segment(readback, binding)
            console = read_console_form(base, binding['run_id'])
        self.assertEqual(len(result['global_monitors']), 40)
        self.assertEqual(console.count('SEARCH_G1_OPTIMIZER_WRITE='), 160)
        self.assertEqual(console.count('SEARCH_G1_OUTER_BUDGET='), 8)
        self.assertAlmostEqual(result['global_monitors'][0]['weighted_k3_numerator'], reports[0]['weighted_k3_numerator'])
        self.assertEqual(result['global_monitors'][0]['effective_token_count'], 112)
        self.assertIs(result['production_admission'], False)

    def test_incomplete_kl_cannot_be_rebuilt_from_finite_zero_numerator(self):
        logs, binding = fabricated_segment()
        for rank in range(8):
            logs[rank][0].update(kl_estimate_complete=False, sampled_old_policy_kl=None,
                                 weighted_k3_numerator=0.)
        with self.assertRaisesRegex(EvidenceError, 'Incomplete KL'):
            verify_segment(logs, binding)

    def test_rejection_invalid_overflow_and_nonfinite_values_rejected(self):
        for key, value in [('allowed', False), ('allowed', 1), ('reasons', ['old_policy_kl']),
                           ('invalid_count', 1), ('overflow_count', 1),
                           ('weighted_k3_numerator', float('nan')),
                           ('effective_weight_mass', float('inf')),
                           ('negative_effective_weight_mass', None),
                           ('negative_log_ratio_p99', float('nan')), ('rejection_count', 1)]:
            with self.subTest(key=key):
                logs, binding = fabricated_segment()
                logs[7][0][key] = value
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)

    def test_rank_global_divergence_rejected_even_if_ratios_coherent(self):
        logs, binding = fabricated_segment()
        row = logs[7][0]
        row['weighted_k3_numerator'] += 1.
        row['sampled_old_policy_kl'] = row['weighted_k3_numerator'] / row['effective_weight_mass']
        with self.assertRaisesRegex(EvidenceError, 'disagree across ranks'):
            verify_segment(logs, binding)

    def test_weighted_ratios_and_counts_independently_recomputed(self):
        for name in ('sampled_old_policy_kl', 'negative_tail_count_fraction',
                     'negative_tail_weight_fraction', 'dual_clip_saturated_objective_weight_fraction'):
            with self.subTest(name=name):
                logs, binding = fabricated_segment()
                for rank in range(8):
                    logs[rank][0][name] += .01
                with self.assertRaisesRegex(EvidenceError, 'report arithmetic'):
                    verify_segment(logs, binding)
        logs, binding = fabricated_segment()
        for rank in range(8):
            logs[rank][0]['negative_tail_count'] = 49
        with self.assertRaises(EvidenceError):
            verify_segment(logs, binding)

    def test_eightfold_reaccumulation_cannot_pass_frozen_monitor_counts(self):
        logs, binding = fabricated_segment()
        for rows in logs.values():
            for row in rows:
                if row['kind'] == 'POLICY_MONITOR':
                    for name in ('weighted_k3_numerator', 'effective_weight_mass',
                                 'negative_effective_weight_mass', 'tail_effective_weight_mass',
                                 'effective_token_count', 'negative_token_count', 'negative_tail_count'):
                        row[name] *= 8
        with self.assertRaisesRegex(EvidenceError, 'Global token count'):
            verify_segment(logs, binding)

    def test_eightfold_weight_mass_alone_rejected(self):
        logs, binding = fabricated_segment()
        for rows in logs.values():
            for row in rows:
                if row['kind'] == 'POLICY_MONITOR':
                    for name in ('weighted_k3_numerator', 'effective_weight_mass',
                                 'negative_effective_weight_mass', 'tail_effective_weight_mass'):
                        row[name] *= 8
        with self.assertRaisesRegex(EvidenceError, 'two-row micro weight mass'):
            verify_segment(logs, binding)

    def test_local_hashes_frozen_within_outer(self):
        logs, binding = fabricated_segment()
        logs[7][59]['monitor_identity']['tensor_sha256']['old_log_probs'] = digest('changed old')
        with self.assertRaisesRegex(EvidenceError, 'identity changed within outer'):
            verify_segment(logs, binding)

    def test_frozen_negative_statistics_cannot_drift_with_consistent_ratios(self):
        logs, binding = fabricated_segment()
        for rows in logs.values():
            row = rows[2]
            row['negative_token_count'] -= 1
            row['negative_effective_weight_mass'] -= .1
            row['negative_tail_count_fraction'] = row['negative_tail_count'] / row['negative_token_count']
            row['negative_tail_weight_fraction'] = row['tail_effective_weight_mass'] / row['negative_effective_weight_mass']
        with self.assertRaisesRegex(EvidenceError, 'Frozen mask/advantage statistics'):
            verify_segment(logs, binding)

    def test_local_selection_and_actual_tensor_hash_schema_required(self):
        for variant in ('duplicate', 'out_of_range', 'empty', 'unsorted', 'missing_tensor', 'bad_hash'):
            with self.subTest(variant=variant):
                logs, binding = fabricated_segment()
                identity = logs[7][0]['monitor_identity']
                if variant == 'duplicate':
                    identity['selection_indices'][1] = identity['selection_indices'][0]
                elif variant == 'out_of_range':
                    identity['selection_indices'][3] = 320
                elif variant == 'empty':
                    identity['actor_token_lengths'][0] = 0
                elif variant == 'unsorted':
                    identity['actor_token_lengths'].reverse()
                elif variant == 'missing_tensor':
                    del identity['tensor_sha256']['advantages']
                else:
                    identity['tensor_sha256']['old_log_probs'] = 'fixture'
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)

    def test_binding_run_definition_and_quantile_method_required(self):
        for name, value in [('run_id', 'another run'), ('calibration_sha256', digest('another plan')),
                            ('objective_weight_definition', 'token mean'), ('monitoring_scope', 'full rollout'),
                            ('quantile_method', 'approximate'), ('stage', 'after_minibatch')]:
            with self.subTest(name=name):
                logs, binding = fabricated_segment()
                logs[7][0][name] = value
                with self.assertRaises(EvidenceError):
                    verify_segment(logs, binding)

    def test_zero_negative_mass_is_structural_only_not_health(self):
        logs, binding = fabricated_segment()
        for rows in logs.values():
            for row in rows:
                if row['kind'] == 'POLICY_MONITOR':
                    row.update(negative_token_count=0, negative_tail_count=0,
                               negative_effective_weight_mass=0., tail_effective_weight_mass=0.,
                               negative_tail_count_fraction=0., negative_tail_weight_fraction=0.,
                               dual_clip_saturated_objective_weight_fraction=0.,
                               negative_log_ratio_p95=None, negative_log_ratio_p99=None, negative_log_ratio_max=None)
        self.assertIs(verify_segment(logs, binding)['production_admission'], False)

    def test_inputs_and_rank_local_metadata_remain_unmodified(self):
        logs, binding = fabricated_segment()
        original = copy.deepcopy((logs, binding))
        result = verify_segment(logs, binding)
        self.assertEqual((logs, binding), original)
        result['monitor_identities'][1][0]['actor_token_lengths'][0] = 999
        result['global_monitors'][0]['reasons'].append('changed result')
        self.assertEqual((logs, binding), original)


if __name__ == '__main__':
    unittest.main()
