"""Tiny arithmetic and false-coverage rejection fixtures; no production reads."""
import copy
import unittest

from .supplement import CONFIG, RUN, SCIENCE, combine


def inputs():
    prefix = {str(v): {'training/actual_outer_updates': v,
                      'training/consumed_sampling_rounds': v,
                      'timing_s/step': 8} for v in range(1, 6)}
    prefix['1'].update({'training/reused_saved_prepared_actor_batch': 1,
                        'training/replayed_round_timing_excludes_original_sampling': 1})
    for v in range(2, 6):
        prefix[str(v)]['rollout/raw_emitted_tokens'] = 5
        prefix[str(v)]['rollout/generated_tokens'] = 9  # Legitimately distinct legacy convention.
    original = dict(protocol='ablation150_original_zero_write_sampling_cost_v1',
        status='COMPLETE_RECORDED_ACTION_COUNTS_WITH_TIMING_LIMITS', group='C1-root-only',
        run_id=RUN, config_sha256=CONFIG, scientific_source_commit=SCIENCE,
        original_outer=1, original_optimizer_calls_per_rank=0,
        reused_for_committed_outer1=True, additional_sampling_or_training=False,
        coverage=dict(initial_trajectories=512, selected_training_rows=2560,
                      successful_unique_action_batches=32, generation_rpc_records=32, full_log_read=True),
        recorded_action_cost_sum=dict(raw_emitted_tokens=834605, raw_retained_tokens=735524,
            raw_discarded_tail_tokens=99081, context_mismatches=0, prefill_tokens=5422985,
            generation_rpc_wall_seconds=173.586),
        time_scopes=dict(first_RPC_to_compose_end_wall_seconds=401.64, compose_seconds=57.59,
                        exact_native_timing_s_gen=None, measured_GPU_active_seconds=None,
                        allocator_billed_GPU_seconds=None),
        evidence=dict(full_closed_log=dict(sha256='a' * 64)))
    row = dict(outer=5, complete_scalar_prefix=True, scalar_coverage_count=5,
        receipts_complete_through_outer=True, complete_generated_token_prefix=False,
        dev_status='COMPLETE_NATIVE_AGGREGATE', generated_tokens_recorded_sum=20,
        cost_coverage=dict(recovered_outer1_original_sampling_excluded=True,
                           end_to_end_coverage_complete=False),
        recorded_stage_seconds_sum={'timing_s/step': 40, 'timing_s/gen': 10},
        allocated_gpu_hours_from_recorded_outer_wall=40 * 8 / 3600)
    aggregate = dict(protocol='recursive_bran_ablation150_closed_milestone_v1',
        milestone_report_complete=True, requested_label='C1-root-only', milestone=5,
        reuse_of=None, generated_new_training=True, configuration_receipt_sha256='b' * 64,
        conditions={'C1-root-only': dict(run_id=RUN, config_sha256=CONFIG, world_size=8,
                                         milestones={'5': row})},
        native_evidence=dict(native_log=dict(captured_line_sha256='c' * 64)))
    capture = dict(protocol='ablation150_closed_prefix_native_scalar_v1', run_id=RUN,
        config_sha256=CONFIG, scientific_source_commit=SCIENCE,
        configuration_receipt_sha256='b' * 64, recovered_outer1=True, completed_prefix_outer=5,
        outer1_metrics=copy.deepcopy(prefix['1']), prefix_metrics=prefix,
        native_line_SHA_by_outer={str(v): 'c' * 64 for v in range(1, 6)})
    return aggregate, original, capture


class SupplementChecks(unittest.TestCase):
    def test_add_once_and_keep_time_unknown_and_inputs_immutable(self):
        values = inputs()
        before = copy.deepcopy(values)
        result = combine(*values)
        self.assertEqual(result['status'], 'COMPLETE_RECORDED_EMITTED_PREFIX_SUPPLEMENT')
        self.assertEqual(result['supplemental_emitted_prefix'], 834625)
        self.assertEqual(result['recorded_native_outer_wall_seconds_unchanged'], 40)
        self.assertIsNone(result['allocator_billed_GPU_seconds'])
        self.assertFalse(result['end_to_end_cost_complete'])
        self.assertEqual(values, before)
        self.assertEqual(combine(result, values[1], values[2])['status'], 'INCOMPLETE')

    def test_zero_field_is_present_not_a_missing_field(self):
        a, o, c = inputs()
        c['outer1_metrics']['rollout/generated_tokens'] = 0
        c['prefix_metrics']['1']['rollout/generated_tokens'] = 0
        result = combine(a, o, c)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertIsNone(result['supplemental_emitted_prefix'])

    def test_later_gap_or_wrong_report_sum_is_rejected(self):
        a, o, c = inputs()
        c['prefix_metrics']['3'].pop('rollout/raw_emitted_tokens')
        c['prefix_metrics']['3'].pop('rollout/generated_tokens')
        self.assertEqual(combine(a, o, c)['status'], 'INCOMPLETE')
        a, o, c = inputs()
        a['conditions']['C1-root-only']['milestones']['5']['generated_tokens_recorded_sum'] = 834625
        self.assertEqual(combine(a, o, c)['status'], 'INCOMPLETE')

    def test_mismatched_identity_and_line_SHA_are_rejected(self):
        a, o, c = inputs()
        c['run_id'] = 'other'
        self.assertEqual(combine(a, o, c)['status'], 'INCOMPLETE')
        a, o, c = inputs()
        c['native_line_SHA_by_outer']['5'] = 'd' * 64
        self.assertEqual(combine(a, o, c)['status'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
