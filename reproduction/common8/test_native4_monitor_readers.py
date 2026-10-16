"""CPU monitor arithmetic and complete reader chains; no native GPU claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from reproduction.common8.formal_contract import FORMAL_SAFEGUARDS, SAFETY_MODE
from reproduction.common8.formal_attempt_audit import _monitor_statistics, verify_attempt_history
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.safety_calibration_evidence import EvidenceError, verify_formal_segment, verify_segment
from reproduction.common8.test_formal_inline_safety import guarded_opportunity_fixture, attempt_history_fixture
from reproduction.common8.test_safety_calibration_evidence import fabricated_segment
from reproduction.common8.topology import formal_monitor_geometry
from search_g1.policy_safety import monitor


MASS_FIELDS = ('effective_weight_mass', 'weighted_k3_numerator',
               'negative_effective_weight_mass', 'tail_effective_weight_mass')


def verify_history(base, run, plan, records, schedulers):
    def gate(_base, _run, step, update_record=None, attempt_id=None, attempt_index=None):
        ordinal = update_record['attempt_index'] if update_record is not None else attempt_index
        return dict(proof_sha256=document_sha256(ordinal))
    with patch('reproduction.common8.behavior_runtime.read_gate_receipt', side_effect=gate):
        return verify_attempt_history(base, run, records, schedulers, last=len(records), plan_sha256=plan)


def completed_segment(base, run, plan, records):
    successful = {row['attempt_id'] for row in records}
    rows = {rank: [row for row in map(json.loads,
                (base/'logs'/('optimizer-rank-'+str(rank)+'.jsonl')).read_text().splitlines())
            if (row.get('context') or row)['attempt_id'] in successful]
            for rank in range(run['training_world_size'])}
    binding = dict(schema_version=1, role='production', policy_safety_mode=SAFETY_MODE,
        run_id=run['run_id'], calibration_sha256=plan, outer_count=len(records),
        initial_optimizer_writes=0, candidate_thresholds=FORMAL_SAFEGUARDS,
        complete_attempts=records, training_world_size=run['training_world_size'])
    if 'actor_micro_batch_per_rank' in run:
        binding['actor_micro_batch_per_rank'] = run['actor_micro_batch_per_rank']
    return rows, binding


def scale_mass(rows, factor):
    # Preserve every normalized KL/tail ratio so only the structural geometry
    # check distinguishes a false total mass from valid CPU monitor arithmetic.
    for events in rows.values():
        for row in events:
            if row['kind'] == 'POLICY_MONITOR':
                for key in MASS_FIELDS:
                    row[key] *= factor


class NativeFourMonitorReaderTests(unittest.TestCase):
    def test_transactional_four_rank_mass_eight_and_false_sixteen(self):
        from reproduction.common8.formal_attempt_audit import _audit_transaction_rows
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            run,plan,records,schedulers=attempt_history_fixture(base,last=1,skipped_before=(),world_size=4,actor_micro=4)
            rows,binding=completed_segment(base,run,plan,records)
            attempt=json.loads((base/'logs/actual-attempts.jsonl').read_text().splitlines()[0])
            attempt['recovery_protocol_version']=2
            attempt['behavior_probability_gate_sha256']=records[0]['behavior_probability_gate_sha256']
            for events in rows.values():
                for row in events:
                    if row['kind']=='OPPORTUNITY':
                        row.update(optimizer_write_rolled_back=False,optimizer_call_attempted=True)
            proof=_audit_transaction_rows(rows,binding,attempt,0,run=run)
            self.assertEqual(proof['global_monitor_count'],40)
            scale_mass(rows,2.)
            with self.assertRaisesRegex(EvidenceError,'weight mass'):
                _audit_transaction_rows(rows,binding,attempt,0,run=run)

    def test_bound_geometry_requires_explicit_typed_native4_micro(self):
        for size, micro, mass in ((4, 4, 8.), (8, 2, 16.)):
            run = dict(profile='common8', training_world_size=size, actor_micro_batch_per_rank=micro)
            proof = formal_monitor_geometry(run)
            self.assertEqual(proof['expected_complete_effective_weight_mass'], mass)
            self.assertEqual(proof['monitor_trajectories_per_rank'], 32//size)
        legacy = formal_monitor_geometry(dict(profile='common8', training_world_size=8))
        self.assertEqual(legacy['actor_micro_batch_per_rank'], 2)
        for value in ('missing', None, True, False, 4., '4', 3, 0):
            run = dict(profile='common8', training_world_size=4)
            if value != 'missing': run['actor_micro_batch_per_rank'] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                formal_monitor_geometry(run)

    def test_actual_cpu_unequal_length_masked_means_pass_full_five_reader_chain(self):
        for size, micro, mass in ((4, 4, 8.), (8, 2, 16.)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as root:
                base = Path(root)
                run, plan, records, schedulers = attempt_history_fixture(
                    base, world_size=size, actor_micro=micro)
                history = verify_history(base, run, plan, records, schedulers)
                self.assertEqual(history['completed_outer_count'], 5)
                self.assertEqual(history['actual_attempt_count'], 7)
                self.assertEqual(history['actual_optimizer_writes_per_rank'], 100)
                rows, binding = completed_segment(base, run, plan, records)
                before = copy.deepcopy(rows)
                proof = verify_formal_segment(rows, binding)
                self.assertEqual(proof['complete_outer_count'], 5)
                self.assertEqual(len(proof['global_monitors']), 200)
                self.assertEqual(proof['monitor_weight_binding']['expected_complete_effective_weight_mass'], mass)
                self.assertTrue(all(abs(row['effective_weight_mass']-mass) < 1e-12
                                    for row in proof['global_monitors']))
                self.assertEqual(rows, before)  # Raw legacy strings/statistics are retained.
                self.assertFalse(proof['production_admission'])

    def test_native4_post_write_guard_keeps_one_write_and_nineteen_skips(self):
        rows, binding = guarded_opportunity_fixture(4, 4)
        proof = verify_formal_segment(rows, binding)
        self.assertEqual(proof['completed_outer_actual_writes_per_rank'], 1)
        self.assertEqual(proof['processed_opportunities_per_rank'], 20)
        self.assertEqual(proof['guard_triggered_monitor_count'], 39)
        self.assertEqual(proof['monitor_weight_binding']['expected_complete_effective_weight_mass'], 8.)
        self.assertEqual(FORMAL_SAFEGUARDS['target_old_policy_kl'], .2)
        self.assertEqual(FORMAL_SAFEGUARDS['negative_tail_limit'], .01)

    def test_wrong_mass_fails_both_readers_even_with_unchanged_normalized_ratios(self):
        for size, micro in ((4, 4), (8, 2)):
            for factor in (.5, 2.):
                with self.subTest(size=size, factor=factor), tempfile.TemporaryDirectory() as root:
                    base = Path(root)
                    run, plan, records, schedulers = attempt_history_fixture(
                        base, skipped_before=(), world_size=size, actor_micro=micro)
                    rows, binding = completed_segment(base, run, plan, records)
                    scale_mass(rows, factor)
                    with self.assertRaisesRegex(EvidenceError, 'weight mass'):
                        verify_formal_segment(rows, binding)
                    for rank, events in rows.items():
                        path = base/'logs'/('optimizer-rank-'+str(rank)+'.jsonl')
                        path.write_text(''.join(json.dumps(row)+'\n' for row in events))
                    with self.assertRaisesRegex(EvidenceError, 'weight mass'):
                        verify_history(base, run, plan, records, schedulers)

    def test_native4_missing_bad_or_incorrect_micro_binding_fails_closed(self):
        for value in ('missing', None, True, 4., '4', 2):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as root:
                base = Path(root)
                run, plan, records, schedulers = attempt_history_fixture(base, world_size=4, actor_micro=4)
                rows, binding = completed_segment(base, run, plan, records)
                if value == 'missing':
                    run.pop('actor_micro_batch_per_rank'); binding.pop('actor_micro_batch_per_rank')
                else:
                    run['actor_micro_batch_per_rank'] = binding['actor_micro_batch_per_rank'] = value
                with self.assertRaises(EvidenceError): verify_formal_segment(rows, binding)
                with self.assertRaises(EvidenceError): verify_history(base, run, plan, records, schedulers)

    def test_missing_rank_or_zero_monitor_length_cannot_supply_complete_mass(self):
        rows, binding = guarded_opportunity_fixture(4, 4)
        partial = copy.deepcopy(rows); partial.pop(3)
        with self.assertRaises(EvidenceError): verify_formal_segment(partial, binding)
        for events in rows.values():
            for row in events:
                if row['kind'] == 'POLICY_MONITOR': row['monitor_identity']['actor_token_lengths'][0] = 0
        with self.assertRaises(EvidenceError): verify_formal_segment(rows, binding)

    def test_invalid_or_empty_cpu_inputs_keep_incomplete_diagnostics(self):
        mask = torch.ones((32, 4), dtype=torch.float64)
        weights = mask/16.  # One unit per actual four-row micro.
        advantages = torch.ones_like(mask)
        for bad in ('nonfinite', 'empty'):
            delta = torch.zeros_like(mask); current = mask.clone()
            if bad == 'nonfinite': delta[0, 0] = float('nan')
            else: current.zero_()
            report = monitor(delta, advantages, current, weights, c=3., kl_log_bound=30.,
                             target_kl=.2, negative_tail_limit=.01)
            self.assertFalse(report['allowed']); self.assertFalse(report['kl_estimate_complete'])
            self.assertIsNone(_monitor_statistics(report)['sampled_old_policy_kl'])
            claimed = dict(report, kl_estimate_complete=True)
            with self.assertRaises(EvidenceError): _monitor_statistics(claimed)

    def test_explicit_historical_eight_rank_calibration_still_requires_mass16(self):
        rows, binding = fabricated_segment()
        proof = verify_segment(rows, binding)
        self.assertTrue(all(abs(row['effective_weight_mass']-16.) < 1e-12
                            for row in proof['global_monitors']))
        scale_mass(rows, .5)
        with self.assertRaisesRegex(EvidenceError, 'eight-rank two-row micro weight mass'):
            verify_segment(rows, binding)


if __name__ == '__main__': unittest.main()
