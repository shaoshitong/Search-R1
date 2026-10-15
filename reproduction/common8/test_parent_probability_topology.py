"""Scoped CPU receipt checks for four/eight-rank parent smoke verification."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.common8 import production_safety_admission as admission
from reproduction.common8 import test_model_smoke as smoke_fixture
from reproduction.common8.precision_contract import NATIVE4_BF16_PROFILE, SAMPLING_JOURNAL_POLICY


class ParentProbabilityTopologyTests(unittest.TestCase):
    def fixture(self, base, size, journal=False):
        templates, ids, _ = smoke_fixture.ModelSmokeTests().proofs()
        run = dict(run_id='cpu-only', profile='common8', training_world_size=size)
        if journal:
            run.update(precision_profile=NATIVE4_BF16_PROFILE, execution_backend='slurm',
                execution_site='hpc3', actor_micro_batch_per_rank=4, vllm_gpu_memory_utilization=.1,
                behavior_probability_policy=SAMPLING_JOURNAL_POLICY,
                question_probe=dict(head_kind='qwen_native_periodic_lastblock_binary_v1'))
        uuids = [str(rank) for rank in range(size)]
        probability = dict(slurm_job_id='123', question_ids=ids,
            actual_cuda_runtime_uuids=uuids, resolved_config_sha256='same')
        parent = base/'logs/policy-safety-admission-receipts/parent'
        parent.mkdir(parents=True)
        receipts = {}
        for phase in ('initial', 'coexistence'):
            proofs = []; hashes = {}
            for rank in range(size):
                row = copy.deepcopy(templates[rank])
                count = 64//size
                row.update(rank=rank, cuda_uuid=uuids[rank], phase=phase, slurm_job_id='123',
                    initial_parameters_finite=True, question_ids=ids[rank*count:(rank+1)*count])
                row['parity']['valid_tokens_per_question'] = [1]*count
                if journal:
                    for role in ('actor', 'reference'):
                        precision = row['runtime_precision']['roles'][role]
                        precision.update(autocast_states=[True], logit_dtypes=['torch.bfloat16'])
                        if role == 'actor':
                            precision.update(compute_parameter_dtypes=['torch.bfloat16'],
                                activation_dtypes=['torch.bfloat16'], mixed_parameter_dtype='torch.bfloat16')
                    rollout = row['runtime_precision']['rollout']
                    for key in ('parameter_dtypes', 'hidden_dtypes', 'cache_dtypes', 'logit_dtypes'):
                        rollout[key] = ['torch.bfloat16']
                    row['parity'].update(tolerance=None, cross_backend_difference_is_diagnostic=True,
                        actor_reference_max_abs=.2, actor_vllm_max_abs=.3)
                    row['exact_model_identity'] = dict(actor_reference_masters_equal=True,
                        masters_unchanged_after_smoke=True, vllm=dict(accepted=True, independent_reader=True))
                path = parent/f'model-smoke-{phase}-rank-{rank}.json'
                path.write_text(json.dumps(row))
                relative = path.relative_to(base).as_posix()
                receipts[relative] = hashes[str(rank)] = admission.digest(path)
                proofs.append(row)
            path = parent/f'model-smoke-{phase}-complete.json'
            path.write_text(json.dumps(dict(accepted=True, run_id='cpu-only', phase=phase,
                ranks=proofs, actual_grpo_updates=0, rank_file_sha256=hashes)))
            probability[phase+'_proof'] = path.relative_to(base).as_posix()
            probability[phase+'_proof_sha256'] = admission.digest(path)
        return run, probability, receipts

    def test_four_and_eight_rank_original_fp32_receipts(self):
        for size in (4, 8):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as root:
                base = Path(root)
                admission._verify_parent_probability(base, *self.fixture(base, size))

    def test_missing_rank_rejects_even_when_complete_receipt_sha_is_refreshed(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root); run, probability, receipts = self.fixture(base, 4)
            path = base/probability['initial_proof']
            proof = json.loads(path.read_text()); proof['ranks'].pop(); path.write_text(json.dumps(proof))
            probability['initial_proof_sha256'] = admission.digest(path)
            with self.assertRaises(admission.AdmissionError):
                admission._verify_parent_probability(base, run, probability, receipts)

    def test_native_four_bf16_journal_uses_actual_precision_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root); run, probability, receipts = self.fixture(base, 4, journal=True)
            admission._verify_parent_probability(base, run, probability, receipts)
            run['question_probe']['head_kind'] = 'different'
            with self.assertRaises(admission.AdmissionError):
                admission._verify_parent_probability(base, run, probability, receipts)


if __name__ == '__main__':
    unittest.main()
