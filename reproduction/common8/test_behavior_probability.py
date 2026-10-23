"""Tiny CPU protocol fixtures; no model, GPU, training or admission claims."""
import copy
import hashlib
import unittest
from pathlib import Path

import torch

from reproduction.common8.behavior_probability import (
    BehaviorProbabilityError, LOGPROB_ATOL, TRAJECTORY_COUNT, verify, verify_sampling,
)


def sampling_fixture():
    return dict(n=1, do_sample=True, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                ignore_eos=False, logprobs=1, max_tokens=500, min_tokens=0, logits_processors=[],
                best_of=1, logit_bias=None, allowed_token_ids=None, guided_decoding=None,
                truncate_prompt_tokens=None,
                logprob_semantics='selected_token_logprob_at_actual_sampling_call')


def tiny_outer_fixture():
    """Real CPU trace API, full2560 rows with two actor tokens, environment and pad."""
    from reproduction.common8.behavior_trace import make_actor_row, make_trace, make_turn
    sampling = sampling_fixture()
    parents = ['cpu-parent-' + str(index) for index in range(512)]
    ids = ['cpu-trajectory-' + str(index) for index in range(TRAJECTORY_COUNT)]
    source = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (Path(__file__), Path(__file__).with_name('behavior_trace.py'))}
    bindings = dict(expected_run_id='cpu-only-probability', expected_source_sha256=source,
                    expected_slurm_job_id='cpu-only', expected_logical_update=1,
                    expected_parent_group_ids=parents, expected_trajectory_ids=ids,
                    expected_policy_snapshot_id='cpu-only-policy:0', expected_sampling=sampling)
    rows, trajectories = [], []
    for index, trajectory_id in enumerate(ids):
        common = dict(trajectory_id=trajectory_id, special_token_ids=[0, 99],
                      eos_token_id=99, pad_token_id=0, sampling=sampling,
                      policy_snapshot_id=bindings['expected_policy_snapshot_id'])
        first = make_turn(turn_index=0, kept_token_count=1,
                          rollout_call_id='cpu-first-' + str(index),
                          actual_request_token_ids=[11, 12], raw_generated_token_ids=[21],
                          raw_generated_token_count=1, selected_logprobs=[-.25 - .01 * (index % 5)],
                          finish_reason='length', stop_reason=None,
                          observation_token_ids=[31] if index % 2 else [],
                          correction_token_ids=[] if index % 2 else [31], **common)
        second = make_turn(turn_index=1, kept_token_count=1, final_turn=True,
                           rollout_call_id='cpu-final-' + str(index),
                           actual_request_token_ids=[11, 12, 21, 31],
                           raw_generated_token_ids=[22, 99], raw_generated_token_count=2,
                           selected_logprobs=[-.5 - .02 * (index % 5), -.75], finish_reason='stop', stop_reason=99,
                           **common)
        trajectories.append(dict(trajectory_id=trajectory_id, parent_occurrence_index=index // 5,
                                 sample_index=index % 5, turns=[first, second]))
        rows.append(make_actor_row(trajectory_id=trajectory_id,
                    input_ids=[0, 11, 12, 21, 31, 22, 0], attention_mask=[0, 1, 1, 1, 1, 1, 0],
                    info_mask=[0, 1, 1, 1, 0, 1, 0], position_ids=[0, 0, 1, 2, 3, 4, 0], prompt_width=3))
    trace = make_trace(run_id=bindings['expected_run_id'], source_sha256=source,
                       slurm_job_id=bindings['expected_slurm_job_id'], logical_update=1,
                       policy_snapshot_id=bindings['expected_policy_snapshot_id'], sampling=sampling,
                       parent_group_ids=parents, trajectories=trajectories, actor_rows=rows)
    return trace, bindings


class SamplingEvidenceTests(unittest.TestCase):
    def test_neutral_actual_fields_are_not_normalized(self):
        value = sampling_fixture()
        value.update(min_tokens=None, logits_processors=None, best_of=None, logit_bias={})
        self.assertEqual(verify_sampling(value), value)
        self.assertIsNone(value['min_tokens'])

    def test_configuration_drift_and_greedy_sampling_rejected(self):
        for field, value in (('temperature', .8), ('top_p', .95), ('top_k', 50),
                             ('min_p', .1), ('presence_penalty', .1),
                             ('frequency_penalty', .1), ('repetition_penalty', 1.1),
                             ('min_tokens', 1), ('logits_processors', ['custom']),
                             ('best_of', 2), ('n', 2), ('do_sample', False),
                             ('ignore_eos', True), ('logprobs', 0),
                             ('max_tokens', 501),
                             ('logit_bias', {'1': 1.0}), ('allowed_token_ids', [1]),
                             ('guided_decoding', {'regex': '.*'}),
                             ('truncate_prompt_tokens', 100),
                             ('logprob_semantics', 'full_softmax')):
            with self.subTest(field=field):
                bad = sampling_fixture(); bad[field] = value
                with self.assertRaises(BehaviorProbabilityError): verify_sampling(bad)

    def test_missing_actual_evidence_and_nonfinite_config_rejected(self):
        for field in sampling_fixture():
            with self.subTest(field=field):
                bad = sampling_fixture(); del bad[field]
                with self.assertRaises(BehaviorProbabilityError): verify_sampling(bad)
        for value in (float('nan'), float('inf'), True):
            bad = sampling_fixture(); bad['temperature'] = value
            with self.assertRaises(BehaviorProbabilityError): verify_sampling(bad)


class WholeOuterProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace, cls.bindings = tiny_outer_fixture()

    def inputs(self, *, dtype=torch.float32):
        from reproduction.common8.behavior_trace import project_sampling_logprobs
        trace, bindings = copy.deepcopy(self.trace), copy.deepcopy(self.bindings)
        rows = copy.deepcopy(trace['actor_rows'])
        projected = project_sampling_logprobs(trace, actor_rows=rows, **bindings)
        actual = torch.tensor(projected['sampling_log_probs'], dtype=dtype)
        authored = torch.tensor(projected['sampling_mask'], dtype=torch.int64)
        attention = torch.tensor([row['attention_mask'][row['prompt_width']:] for row in rows])
        hf = actual.clone()
        hf[:, 1] = float('nan'); hf[:, 3] = float('nan')
        return [hf, actual, authored, attention], dict(trace=trace, actor_rows=rows,
                                                      audit=projected['audit'], **bindings)

    def test_all2560_included_tokens_and_input_identity_unchanged(self):
        tensors, evidence = self.inputs()
        before = [value.clone() for value in tensors]
        trace_before = copy.deepcopy(evidence['trace'])
        proof = verify(*tensors, **evidence)
        self.assertTrue(proof['accepted'])
        self.assertEqual(proof['trajectory_count'], TRAJECTORY_COUNT)
        self.assertEqual(proof['included_actor_token_count'], 2 * TRAJECTORY_COUNT)
        self.assertEqual(proof['actor_token_counts'], [2] * TRAJECTORY_COUNT)
        self.assertEqual(proof['max_abs_discrepancy'], 0.)
        self.assertEqual(proof['tolerance'], LOGPROB_ATOL)
        for name in ('production_admission', 'healthy_checkpoint_accepted', 'physical_execution_authenticated'):
            self.assertIs(proof[name], False)
        self.assertEqual(evidence['trace'], trace_before)
        for actual, original in zip(tensors, before):
            torch.testing.assert_close(actual, original, rtol=0, atol=0, equal_nan=True)

    def test_bad_last_trajectory_cannot_hide_outside32_sample(self):
        tensors, evidence = self.inputs()
        tensors[0][-1, 2] -= .06
        with self.assertRaisesRegex(BehaviorProbabilityError, 'original0.05'):
            verify(*tensors, **evidence)

    def test_restored_collection_producers_reach_probability_trace_rebuild(self):
        from reproduction.common8.behavior_trace import project_sampling_logprobs, BehaviorTraceError
        from reproduction.common8.policy_calibration_plan import document_sha256
        tensors,evidence=self.inputs()
        producers=[dict(run_id=evidence['expected_run_id'],
                        slurm_job_id=evidence['expected_slurm_job_id'],
                        source_map_sha256=document_sha256(evidence['expected_source_sha256']))]
        evidence['expected_generation_producers']=producers
        evidence['trace']['generation_producers']=copy.deepcopy(producers)
        bindings={k:v for k,v in evidence.items() if k.startswith('expected_')}
        evidence['audit']=project_sampling_logprobs(evidence['trace'],actor_rows=evidence['actor_rows'],**bindings)['audit']
        proof=verify(None,*tensors[1:],require_hf_parity=False,**evidence)
        self.assertTrue(proof['accepted'])
        self.assertFalse(proof['hf_recomputation_performed'])
        evidence['trace']['generation_producers'][0]['slurm_job_id']='unbound-job'
        with self.assertRaises(BehaviorTraceError):
            verify(None,*tensors[1:],require_hf_parity=False,**evidence)

    def test_original_tolerance_accepts_small_discrepancy(self):
        tensors, evidence = self.inputs()
        tensors[0][:, 0] -= .04
        proof = verify(*tensors, **evidence)
        self.assertGreater(proof['max_abs_discrepancy'], .039)
        self.assertLessEqual(proof['max_abs_discrepancy'], .041)

    def test_original_point05_boundary_cannot_be_relaxed(self):
        from reproduction.common8.behavior_trace import audit_outer
        tensors, evidence = self.inputs(dtype=torch.float64)
        evidence['trace']['trajectories'][0]['turns'][0]['selected_logprobs'][0] = 0.
        bindings = {key: value for key, value in evidence.items() if key.startswith('expected_')}
        evidence['audit'] = audit_outer(evidence['trace'], actor_rows=evidence['actor_rows'], **bindings)
        tensors[1][0, 0] = 0.; tensors[0][0, 0] = -.05
        self.assertEqual(verify(*tensors, **evidence)['max_abs_discrepancy'], .05)
        tensors[0][0, 0] = -.050001
        with self.assertRaisesRegex(BehaviorProbabilityError, 'original0.05'):
            verify(*tensors, **evidence)

    def test_nonfinite_actor_values_rejected_but_environment_hf_nan_ignored(self):
        tensors, evidence = self.inputs()
        self.assertTrue(verify(*tensors, **evidence)['accepted'])
        for matrix in (0, 1):
            for value in (float('nan'), float('inf')):
                with self.subTest(matrix=matrix, value=value):
                    bad = [item.clone() for item in tensors]; bad[matrix][-1, 0] = value
                    with self.assertRaises(BehaviorProbabilityError): verify(*bad, **evidence)
        tensors[1][0, 1] = float('nan')
        with self.assertRaisesRegex(BehaviorProbabilityError, 'frozen old'):
            verify(*tensors, **evidence)

    def test_environment_probabilities_never_enter_actor_denominator(self):
        tensors, evidence = self.inputs()
        tensors[1][0, 1] = -100.
        with self.assertRaisesRegex(BehaviorProbabilityError, 'outside actor mask'):
            verify(*tensors, **evidence)
        tensors, evidence = self.inputs()
        tensors[2][0, 1] = 1
        with self.assertRaises(BehaviorProbabilityError): verify(*tensors, **evidence)
        tensors, evidence = self.inputs()
        tensors[2][0, 0] = 0; tensors[2][0, 1] = 1
        tensors[0][0, 1] = 0.
        with self.assertRaises(BehaviorProbabilityError): verify(*tensors, **evidence)

    def test_hf_copy_cannot_impersonate_actual_sampling_logprobs(self):
        tensors, evidence = self.inputs()
        tensors[0][0, 0] = -.27
        tensors[1][0, 0] = -.27
        with self.assertRaisesRegex(BehaviorProbabilityError, 'actual selected'):
            verify(*tensors, **evidence)

    def test_mask_dtype_float_dtype_shape_and_whole_outer_required(self):
        tensors, evidence = self.inputs()
        for matrix, dtype in ((0, torch.int64), (1, torch.int64), (0, torch.float16),
                              (1, torch.bfloat16), (2, torch.float32)):
            with self.subTest(matrix=matrix, dtype=dtype):
                bad = [value.clone() for value in tensors]; bad[matrix] = bad[matrix].to(dtype)
                with self.assertRaises(BehaviorProbabilityError): verify(*bad, **evidence)
        with self.assertRaises(BehaviorProbabilityError):
            verify(*[value[:32] for value in tensors], **evidence)
        bad = [value.clone() for value in tensors]; bad[1] = bad[1][:, :-1]
        with self.assertRaises(BehaviorProbabilityError): verify(*bad, **evidence)
        bad = [value.clone() for value in tensors]; bad[2][0, 0] = 2
        with self.assertRaises(BehaviorProbabilityError): verify(*bad, **evidence)

    def test_reconstructed_audit_source_run_and_missing_trace_fail_closed(self):
        tensors, evidence = self.inputs()
        for name, value in (('audit', None), ('trace', None), ('expected_run_id', 'different'),
                            ('expected_source_sha256', {})):
            with self.subTest(name=name):
                bad = copy.deepcopy(evidence); bad[name] = value
                with self.assertRaises(ValueError): verify(*tensors, **bad)
        evidence['audit']['included_token_count'] += 1
        with self.assertRaisesRegex(BehaviorProbabilityError, 'reconstructed'):
            verify(*tensors, **evidence)

    def test_valid_reorder_requires_same_reorder_of_tensors_rows_and_audit(self):
        from reproduction.common8.behavior_trace import audit_outer
        tensors, evidence = self.inputs()
        order = list(range(TRAJECTORY_COUNT - 1, -1, -1))
        reordered = [value[order] for value in tensors]
        evidence['actor_rows'] = [evidence['actor_rows'][index] for index in order]
        bindings = {key: value for key, value in evidence.items() if key.startswith('expected_')}
        evidence['audit'] = audit_outer(evidence['trace'], actor_rows=evidence['actor_rows'], **bindings)
        proof = verify(*reordered, **evidence)
        self.assertEqual(proof['trajectory_ids'], list(reversed(evidence['expected_trajectory_ids'])))
        with self.assertRaisesRegex(BehaviorProbabilityError, 'actual selected'):
            verify(*tensors, **evidence)


if __name__ == '__main__': unittest.main()
