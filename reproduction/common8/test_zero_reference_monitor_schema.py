"""CPU structural fixtures: reference skipping never removes old-policy checks."""
import copy
import unittest

from reproduction.common8.safety_calibration_evidence import (
    EvidenceError, MONITOR_TENSORS, ZERO_REFERENCE_MONITOR_PRODUCER_SHA256,
    _identity, _reference_required, verify_formal_segment, verify_segment,
)
from reproduction.common8.test_safety_calibration_evidence import fabricated_segment
from reproduction.common8.test_formal_inline_safety import guarded_opportunity_fixture


def contract(binding):
    return dict(schema_version=1, role='bound_actual_zero_reference_monitor',
        run_id=binding['run_id'], calibration_sha256=binding['calibration_sha256'],
        resolved_config_sha256='a' * 64, source_map_sha256='b' * 64,
        producer_source_sha256=ZERO_REFERENCE_MONITOR_PRODUCER_SHA256,
        use_kl_loss=True, reference_kl_coefficient=0.0, state_masking=True)


class ZeroReferenceSchemaTests(unittest.TestCase):
    def test_transactional_twenty_opportunity_path(self):
        from reproduction.common8.formal_attempt_audit import _audit_monitor_rows
        templates, binding = guarded_opportunity_fixture()
        first = binding['complete_attempts'][0]
        attempt = dict(first, logical_update=1, recovery_protocol_version=2,
                       optimizer_writes_after=20, opportunities_processed=20,
                       complete_outer=True, opportunity_statuses=['written'] * 20)
        binding['monitor_reference_contract'] = contract(binding)
        run = dict(profile='common8', training_world_size=8)
        rows = {}
        for rank, values in templates.items():
            template = next(row for row in values if row['kind'] == 'POLICY_MONITOR')
            current = []
            for mini in range(20):
                context = dict(template['context'], minibatch_index=mini, prior_optimizer_writes=mini)
                for stage in ('before_minibatch', 'after_minibatch'):
                    row = copy.deepcopy(template)
                    row.update(context=context, stage=stage, rejection_count=0,
                               optimizer_writes=mini + int(stage == 'after_minibatch'))
                    del row['monitor_identity']['tensor_sha256']['ref_log_prob']
                    current.append(row)
                current.append(dict(kind='OPTIMIZER_WRITE', run_id=binding['run_id'], rank=rank,
                    context=context, write_count=mini + 1, learning_rates=[0.],
                    gradient_norm_before=0.5, gradient_norm_after=0.5))
                current.append(dict(kind='OPPORTUNITY', run_id=binding['run_id'], rank=rank,
                    context=context, opportunity_index=mini, processed=True, status='written',
                    optimizer_write_succeeded=True, optimizer_write_rolled_back=False,
                    optimizer_call_attempted=True, optimizer_writes_before=mini,
                    optimizer_writes_after=mini + 1, guard_stage=None, guard_reasons=[]))
            current.append(dict(kind='OUTER_BUDGET', run_id=binding['run_id'], rank=rank,
                opportunities_processed=20, optimizer_writes=20, cumulative_writes=20))
            rows[rank] = current
        proof = _audit_monitor_rows(rows, binding, attempt, 0, run=run)
        self.assertEqual(proof['global_monitor_count'], 40)
        rows[0][0]['monitor_identity']['tensor_sha256'].pop('old_log_probs')
        with self.assertRaises(EvidenceError):
            _audit_monitor_rows(rows, binding, attempt, 0, run=run)

    def test_exact_identity_and_default_legacy(self):
        identity = dict(selection_indices=[0, 100, 200, 300], actor_token_lengths=[1, 2, 3, 4],
                        tensor_sha256={name: 'a' * 64 for name in MONITOR_TENSORS})
        _identity(identity)
        with self.assertRaises(EvidenceError):
            _identity(identity, reference_required=False)
        del identity['tensor_sha256']['ref_log_prob']
        _identity(identity, reference_required=False)
        with self.assertRaises(EvidenceError):
            _identity(identity)
        for name in tuple(identity['tensor_sha256']):
            broken = copy.deepcopy(identity); del broken['tensor_sha256'][name]
            with self.assertRaises(EvidenceError):
                _identity(broken, reference_required=False)
        broken = copy.deepcopy(identity); broken['tensor_sha256']['old_log_probs'] = 'invalid'
        with self.assertRaises(EvidenceError):
            _identity(broken, reference_required=False)

    def test_contract_rejects_unbound_or_nonzero_or_unknown_producer(self):
        binding = dict(run_id='fixture', calibration_sha256='c' * 64)
        self.assertTrue(_reference_required(binding))
        good = contract(binding)
        self.assertFalse(_reference_required(dict(binding, monitor_reference_contract=good)))
        for field, value in (('run_id', 'other'), ('reference_kl_coefficient', 0.1),
                             ('use_kl_loss', False), ('state_masking', False),
                             ('producer_source_sha256', 'd' * 64), ('resolved_config_sha256', 'invalid')):
            broken = dict(good, **{field: value})
            with self.assertRaises(EvidenceError):
                _reference_required(dict(binding, monitor_reference_contract=broken))

    def test_formal_guarded_opportunity_path_accepts_only_bound_seven_keys(self):
        rows, binding = guarded_opportunity_fixture()
        verify_formal_segment(rows, binding)
        for values in rows.values():
            for row in values:
                if row['kind'] == 'POLICY_MONITOR':
                    del row['monitor_identity']['tensor_sha256']['ref_log_prob']
        with self.assertRaises(EvidenceError):
            verify_formal_segment(rows, binding)
        binding['monitor_reference_contract'] = contract(binding)
        verify_formal_segment(rows, binding)
        rows[0][0]['monitor_identity']['tensor_sha256'].pop('old_log_probs')
        with self.assertRaises(EvidenceError):
            verify_formal_segment(rows, binding)

    def test_historical_calibration_stays_eight_key(self):
        rows, binding = fabricated_segment()
        verify_segment(rows, binding)
        binding['monitor_reference_contract'] = contract(binding)
        for values in rows.values():
            for row in values:
                if row['kind'] == 'POLICY_MONITOR':
                    del row['monitor_identity']['tensor_sha256']['ref_log_prob']
        with self.assertRaises(EvidenceError):
            verify_segment(rows, binding)


if __name__ == '__main__':
    unittest.main()
