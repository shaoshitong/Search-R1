"""CPU structural regression for Easy's actual extra frozen tensor key.

Synthetic ledger rows exercise the real protocol2 reader, including its frozen
identity and numerical guards. They do not claim native training acceptance.
"""
import copy
import hashlib
import struct
import unittest

from reproduction.common8.formal_attempt_audit import _audit_transaction_rows
from reproduction.common8.formal_contract import consumed_lr
from reproduction.common8.safety_calibration_evidence import (
    EvidenceError, MONITOR_TENSORS, QUANTILE_METHOD, _identity, objective_weight_definition,
)
from reproduction.common8.topology import monitor_scope
from search_g1.monitor_forward_reuse import source_identity


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def easy_run():
    return dict(profile='common8', easy2560_selection=True, execution_site='hpc3',
                execution_backend='slurm', training_world_size=4, data_selection_mode='mixed_first',
                question_probe=None, actor_logical_micro_batch_per_rank=4,
                actor_micro_batch_per_rank=16, run_id='CPU-structural-Easy-fixture')


def identity():
    # Hash concrete CPU byte strings for the exact producer-selected key set;
    # never add a fabricated hash to actual runtime evidence.
    names = sorted((MONITOR_TENSORS - {'ref_log_prob'}) | {'selection_fallback'})
    return dict(selection_indices=list(range(8)), actor_token_lengths=[4] * 8,
                tensor_sha256={name: hashlib.sha256(struct.pack('<8q', *([i] * 8))).hexdigest()
                               for i, name in enumerate(names)})


def transaction_fixture(step=1):
    run = easy_run(); start = (step - 1) * 20; aid = f'{step:032x}'; gate = digest('gate' + str(step))
    binding = dict(run_id=run['run_id'], calibration_sha256=digest('plan'))
    attempt = dict(logical_update=step, attempt_id=aid, attempt_index=step,
                   optimizer_writes_before=start, optimizer_writes_after=start + 20,
                   optimizer_writes_this_attempt=20, opportunities_processed=20,
                   opportunity_statuses=['written'] * 20,
                   behavior_probability_gate_sha256=gate, complete_outer=True)
    # 32 monitor rows, each four policy tokens, four-row logical means:
    # global effective mass8, half negative mass4, unchanged policy delta0.
    statistics = dict(allowed=True, reasons=[], sampled_old_policy_kl=0., kl_estimate_complete=True,
        effective_weight_mass=8., weighted_k3_numerator=0., negative_effective_weight_mass=4.,
        tail_effective_weight_mass=0., effective_token_count=128, negative_token_count=64,
        negative_tail_count=0, negative_tail_count_fraction=0., negative_tail_weight_fraction=0.,
        dual_clip_saturated_objective_weight_fraction=0., invalid_count=0, overflow_count=0,
        quantile_method=QUANTILE_METHOD, negative_log_ratio_p95=0., negative_log_ratio_p99=0.,
        negative_log_ratio_max=0.)
    rows = {}
    for rank in range(4):
        events = []; previous = None; actual = reused = 0
        for mini in range(20):
            prior = start + mini
            context = dict(logical_update=step, minibatch_index=mini, prior_optimizer_writes=prior,
                           attempt_id=aid, attempt_index=step, behavior_probability_gate_sha256=gate)
            for post in (False, True):
                reuse = not post and mini > 0
                actual += int(not reuse); reused += int(reuse)
                row = dict(kind='POLICY_MONITOR', run_id=run['run_id'], rank=rank,
                    context=copy.deepcopy(context), stage='after_minibatch' if post else 'before_minibatch',
                    optimizer_writes=prior + int(post), calibration_sha256=binding['calibration_sha256'],
                    execution_role='production', monitoring_scope=monitor_scope(run),
                    objective_weight_definition=objective_weight_definition(run),
                    monitor_identity=identity(), rejection_count=0,
                    monitor_model_version=prior + int(post), monitor_input_sha256=digest('input' + str(step)),
                    forward_reused=reuse, actual_monitor_forward=not reuse,
                    forward_source=source_identity(previous) if reuse else None,
                    actual_monitor_forward_count=actual, reused_monitor_forward_count=reused,
                    **copy.deepcopy(statistics))
                events.append(row); previous = row
            events.append(dict(kind='OPTIMIZER_WRITE', run_id=run['run_id'], rank=rank,
                context=copy.deepcopy(context), write_count=prior + 1,
                learning_rates=[consumed_lr(step)], gradient_norm_before=2., gradient_norm_after=1.))
            events.append(dict(kind='OPPORTUNITY', run_id=run['run_id'], rank=rank,
                context=copy.deepcopy(context), opportunity_index=mini, processed=True, status='written',
                optimizer_write_succeeded=True, optimizer_write_rolled_back=False,
                optimizer_call_attempted=True, optimizer_writes_before=prior,
                optimizer_writes_after=prior + 1, guard_stage=None, guard_reasons=[]))
        events.append(dict(kind='OUTER_BUDGET', run_id=run['run_id'], rank=rank,
            opportunities_processed=20, optimizer_writes=20, cumulative_writes=start + 20,
            actual_monitor_forward_count=21, reused_monitor_forward_count=19))
        rows[rank] = events
    return run, binding, attempt, rows


class EasyMonitorIdentityTests(unittest.TestCase):
    def test_actual_extra_key_accepted_only_in_explicit_easy_scope(self):
        value = identity(); _identity(value, monitor_count=8, run=easy_run())
        with self.assertRaisesRegex(EvidenceError, 'Unexpected'):
            _identity(value, monitor_count=8)
        for change in ({'easy2560_selection': False}, {'training_world_size': 8},
                       {'actor_logical_micro_batch_per_rank': 16}, {'execution_backend': 'ssh_local'}):
            with self.subTest(change=change), self.assertRaises(EvidenceError):
                _identity(value, monitor_count=8, run=dict(easy_run(), **change))

    def test_missing_input_or_selection_hash_remains_failure(self):
        for name in ('old_log_probs', 'loss_mask', 'selection_fallback'):
            value = identity(); value['tensor_sha256'].pop(name)
            with self.subTest(name=name), self.assertRaisesRegex(EvidenceError, 'Missing'):
                _identity(value, monitor_count=8, run=easy_run())

    def test_reference_is_optional_but_unknown_key_and_invalid_hash_rejected(self):
        value = identity(); value['tensor_sha256']['ref_log_prob'] = digest('reference')
        _identity(value, monitor_count=8, run=easy_run())
        value['tensor_sha256']['unknown_tensor'] = digest('unknown')
        with self.assertRaisesRegex(EvidenceError, 'Unexpected'):
            _identity(value, monitor_count=8, run=easy_run())
        value = identity(); value['tensor_sha256']['selection_fallback'] = 'not-a-hash'
        with self.assertRaises(EvidenceError):
            _identity(value, monitor_count=8, run=easy_run())

    def test_five_full_native4_protocol2_prefixes_keep_21_real_19_reused(self):
        for step in range(1, 6):
            run, binding, attempt, rows = transaction_fixture(step)
            proof = _audit_transaction_rows(rows, binding, attempt, 0, run=run)
            self.assertEqual(proof['actual_monitor_forward_count'], 21)
            self.assertEqual(proof['reused_monitor_forward_count'], 19)
            self.assertEqual(proof['global_monitor_count'], 40)
            self.assertEqual(proof['guard_triggered_monitor_count'], 0)
            self.assertTrue(all(events[-1]['cumulative_writes'] == step * 20 for events in rows.values()))

    def test_selection_label_hash_cannot_change_inside_frozen_sequence(self):
        run, binding, attempt, rows = transaction_fixture()
        [r for r in rows[3] if r['kind'] == 'POLICY_MONITOR'][-1]['monitor_identity']['tensor_sha256']['selection_fallback'] = digest('changed')
        with self.assertRaisesRegex(EvidenceError, 'frozen sample changed'):
            _audit_transaction_rows(rows, binding, attempt, 0, run=run)

    def test_actual_gradient_finite_checks_remain_enforced(self):
        run, binding, attempt, rows = transaction_fixture()
        [r for r in rows[3] if r['kind'] == 'OPTIMIZER_WRITE'][-1]['gradient_norm_after'] = float('nan')
        with self.assertRaises(EvidenceError):
            _audit_transaction_rows(rows, binding, attempt, 0, run=run)


if __name__ == '__main__':
    unittest.main()
