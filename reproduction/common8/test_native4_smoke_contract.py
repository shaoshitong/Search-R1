"""Bound BF16 journal diagnostics and unchanged same-actor/legacy gates."""
import copy
import unittest
from unittest.mock import patch

from reproduction.common8 import model_smoke as smoke
from reproduction.common8.production_safety_admission import _verify_parent_probability, AdmissionError
from reproduction.common8.test_model_smoke import ModelSmokeTests
from reproduction.common8.test_precision_evidence import fixture_proof


def run_binding():
    return dict(run_id='cpu-only', profile='common8', precision_profile='native4_bf16_varlen_fp32_master',
        training_world_size=4, execution_backend='slurm', execution_site='hpc3',
        actor_micro_batch_per_rank=4, vllm_gpu_memory_utilization=.1,
        behavior_probability_policy='sampling_journal_no_extra_hf_forward')


def precision_fixture():
    proof = fixture_proof()
    for name, row in proof['roles'].items():
        row.update(autocast_states=[True], logit_dtypes=['torch.bfloat16'])
        if name == 'actor':
            row.update(compute_parameter_dtypes=['torch.bfloat16'],
                activation_dtypes=['torch.bfloat16'], mixed_parameter_dtype='torch.bfloat16')
    for name in ('parameter_dtypes', 'hidden_dtypes', 'cache_dtypes', 'logit_dtypes'):
        proof['rollout'][name] = ['torch.bfloat16']
    return proof


def ranks(phase='initial'):
    ids, uuids = [str(i) for i in range(64)], [str(i) for i in range(4)]
    rows = [dict(accepted=True, rank=r, run_id='cpu-only', slurm_job_id='cpu-only',
        phase=phase, cuda_uuid=uuids[r], actual_optimizer_writes=0,
        sync_calls=1, sync_load_format='dtensor', initial_parameters_finite=True,
        separate_actor_logprob_entrypoint=True, internal_separate_logprob_max_abs=0.,
        question_ids=ids[r*16:(r+1)*16], resolved_config_sha256='same',
        runtime_precision=precision_fixture(), parity=dict(tolerance=None,
            comparison_policy=run_binding()['behavior_probability_policy'],
            cross_backend_drift_is_diagnostic=True, actor_reference_max_abs=.737,
            actor_vllm_max_abs=1.251, valid_tokens_per_question=[1]*16)) for r in range(4)]
    return rows, ids, uuids


class Native4SmokeContractTests(unittest.TestCase):
    def test_explicit_profile_records_cross_backend_drift_without_false_parity(self):
        values = ModelSmokeTests().samples()
        values[1][0, 0] -= .737
        values[4][0, 0] -= 1.251
        result = smoke.compare(*values, run=run_binding())
        self.assertIsNone(result['tolerance'])
        self.assertTrue(result['cross_backend_drift_is_diagnostic'])
        self.assertGreater(result['actor_reference_max_abs'], .7)
        self.assertGreater(result['actor_vllm_max_abs'], 1.2)
        with self.assertRaises(AssertionError):
            smoke.compare(*values)
        no_policy = run_binding(); no_policy.pop('behavior_probability_policy')
        with self.assertRaises(AssertionError):
            smoke.compare(*values, run=no_policy)

    def test_journal_mode_still_rejects_bad_tokens_and_nonfinite_values(self):
        for field, value in ((0, float('nan')), (1, float('inf')), (4, float('nan')), (3, 99)):
            values = ModelSmokeTests().samples(); values[field][0, 0] = value
            with self.assertRaises(AssertionError):
                smoke.compare(*values, run=run_binding())

    def test_native4_aggregate_keeps_strict_internal_finite_and_precision_checks(self):
        rows, ids, uuids = ranks()
        self.assertEqual(smoke.aggregate(rows, 'cpu-only', 'cpu-only', 'initial', ids, uuids,
            run=run_binding()), 'same')
        for case in ('internal', 'nonfinite', 'policy', 'precision', 'missing'):
            bad = copy.deepcopy(rows)
            if case == 'internal': bad[0]['internal_separate_logprob_max_abs'] = .051
            elif case == 'nonfinite': bad[0]['parity']['actor_vllm_max_abs'] = float('nan')
            elif case == 'policy': bad[0]['parity']['comparison_policy'] = 'other'
            elif case == 'precision': bad[0]['runtime_precision']['roles']['actor']['compute_parameter_dtypes'] = ['torch.float32']
            else: bad.pop()
            with self.assertRaises(AssertionError):
                smoke.aggregate(bad, 'cpu-only', 'cpu-only', 'initial', ids, uuids, run=run_binding())

    def test_parent_admission_reads_all_four_original_rank_receipts_and_bound_profile(self):
        documents, receipts = {}, {}
        probability = dict(slurm_job_id='cpu-only', question_ids=[str(i) for i in range(64)],
            actual_cuda_runtime_uuids=[str(i) for i in range(4)], resolved_config_sha256='same')
        for phase in ('initial', 'coexistence'):
            rows, _, _ = ranks(phase)
            path = phase + '.json'
            probability[phase+'_proof'] = path; probability[phase+'_proof_sha256'] = phase
            documents[path] = dict(accepted=True, run_id='cpu-only', phase=phase, ranks=rows,
                actual_grpo_updates=0, rank_file_sha256={str(i):'rank'+str(i) for i in range(4)})
            for i, row in enumerate(rows):
                relative = 'logs/policy-safety-admission-receipts/parent/model-smoke-'+phase+'-rank-'+str(i)+'.json'
                documents[relative] = row; receipts[relative] = 'rank'+str(i)
        with patch('reproduction.common8.production_safety_admission._receipt', side_effect=lambda base, path, sha: documents[path]):
            _verify_parent_probability(None, run_binding(), probability, receipts)
            documents['initial.json']['ranks'][0]['internal_separate_logprob_max_abs'] = .051
            with self.assertRaises(AssertionError):
                _verify_parent_probability(None, run_binding(), probability, receipts)


if __name__ == '__main__':
    unittest.main()
