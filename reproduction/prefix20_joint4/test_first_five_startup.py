"""CPU receipt graph fixtures; no synthetic model/GPU acceptance is produced."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8.formal_contract import (
    CONTRACT_KIND, FORMAL_COUNTING, FORMAL_SAFEGUARDS, SAFETY_MODE, formal_mathematics,
)
from reproduction.common8 import formal_inline_safety as safety
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.precision_contract import NATIVE4_BF16_PROFILE, SAMPLING_JOURNAL_POLICY
from reproduction.common8.verify_data import digest
from reproduction.h1008_qprobe.first_five_admission import prepare, selected


class FirstFiveStartupTests(unittest.TestCase):
    def run_fixture(self, base):
        run = dict(profile='common8', execution_site='hpc3', execution_backend='slurm',
            training_world_size=4, actor_micro_batch_per_rank=4, vllm_gpu_memory_utilization=.1,
            precision_profile=NATIVE4_BF16_PROFILE, behavior_probability_policy=SAMPLING_JOURNAL_POLICY,
            startup_admission_mode='actual_first_five', data_selection_mode='prefix20_probe',
            question_probe=dict(probe_architecture='qwen2_prefix20_trajectory_em_v1'),
            contract_kind=CONTRACT_KIND, policy_safety_mode=SAFETY_MODE, completed_updates=0,
            target_updates=200, acceptance_update=5, run_id='a' * 32, source_commit='b' * 40,
            source_sha256={'fixture.py': 'c' * 64}, started_unix=20.,
            initial_model='/fixture/official', reference_model='/fixture/official',
            model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
            initial_model_sha256={'config.json': 'd' * 64}, input_sha256={'train.parquet': 'e' * 64},
            initial_probe_version='a' * 32 + ':0:real-calibration',
            checkpoint_dir=str(base / 'checkpoints/search-g1-common8'),
            train_log=str(base / 'logs/training.log'))
        budget = json.loads((Path(__file__).resolve().parents[1] / 'common8/budget.json').read_text())
        budget.update(formal_mathematics(run))
        run['budget'] = budget
        return run

    def receipt_fixture(self, base):
        (base / 'logs').mkdir()
        (base / 'preparation').mkdir()
        run = self.run_fixture(base)
        method = dict(schema_version=1, role=safety.METHOD_ROLE, source_commit=run['source_commit'],
            source_map_sha256=document_sha256(run['source_sha256']), policy_safety_mode=SAFETY_MODE,
            target_updates=200, warmup_steps=57, same_process_after_five=True, fresh_states=True,
            optimizer_counting=FORMAL_COUNTING, safeguards=copy.deepcopy(FORMAL_SAFEGUARDS),
            created_unix=10., negative_tail_semantics='prospective operational candidate; not calibrated health')
        archive = base / 'artifacts/round_0-real/version-manifest.json'
        archive.parent.mkdir(parents=True)
        archive.write_text(json.dumps(dict(run_id=run['run_id'], round=0,
            version=run['initial_probe_version'], encoder=run['initial_model'])))
        initial = dict(role='fresh200_actual_initial_probe', run_id=run['run_id'],
            source_map_sha256=document_sha256(run['source_sha256']), initial_model_sha256=run['initial_model_sha256'],
            input_sha256=run['input_sha256'], version=run['initial_probe_version'],
            archive=archive.relative_to(base).as_posix(), archive_sha256=digest(archive))
        allocation = dict(accepted=True, run_id=run['run_id'], slurm_job_id='7001',
            actual_cuda_runtime_uuids=['fixture-gpu-%d' % rank for rank in range(4)],
            actual_allocation_detail='JobId=7001 JobState=RUNNING SubmitTime=s StartTime=t '
                'UserId=jxu957(14090) Command=fixture-launch WorkDir=fixture-repo NodeList=fixture-node')
        files = {'logs/native-allocation-admission.json': allocation,
                 'preparation/formal-inline-method.json': method, safety.INITIAL: initial}
        for name in ('retrieval-admission', 'retrieval-http-admission', 'evaluator-release-admission',
                     'probe-coexistence-admission'):
            files['logs/' + name + '.json'] = dict(run_id=run['run_id'], fixture_only=True)
        for relative, value in files.items():
            (base / relative).write_text(json.dumps(value))
        receipts = {initial['archive']: digest(archive), safety.INITIAL: digest(base / safety.INITIAL)}
        return run, method, initial, receipts

    def prepare_fixture(self, base):
        run, method, initial, receipts = self.receipt_fixture(base)
        with patch.dict(os.environ, SLURM_JOB_ID='7001'), patch(
                'reproduction.common8.production_safety_admission._verify_parent_probability',
                side_effect=AssertionError('obsolete standalone model smoke route was called')):
            artifact = prepare(base, run, {}, method, initial, receipts)
            safety.verify_plan(base, artifact, run)
        return run, artifact

    def test_real_formal_plan_verification_accepts_prospective_native4_without_smoke_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run, artifact = self.prepare_fixture(base)
            self.assertTrue(selected(run))
            self.assertTrue((base / safety.PLAN).is_file())
            self.assertFalse((base / 'logs/model-smoke-initial').exists())
            self.assertFalse((base / 'logs/model-smoke-coexistence').exists())
            self.assertFalse(artifact['healthy_checkpoint_accepted'])
            self.assertFalse(artifact['short_train_accepted'])
            probability = artifact['probability']
            for field in ('actual_model_initialization_accepted', 'actual_behavior_probability_accepted',
                          'actual_first_outer_accepted', 'standalone_synthetic_gpu_smoke_performed'):
                self.assertIs(probability[field], False)
            self.assertEqual(probability['actual_optimizer_writes'], 0)
            self.assertEqual(probability['training_world_size'], 4)

    def test_original_h1008_route_remains_and_other_experiments_cannot_opt_in(self):
        original = dict(execution_site='h1008', training_world_size=8,
            startup_admission_mode='actual_first_five', question_probe=dict(probe_architecture='qwen2_decoder3_count6'))
        self.assertTrue(selected(original))
        with tempfile.TemporaryDirectory() as directory:
            run = self.run_fixture(Path(directory).resolve())
            mutations = [('profile', 'other'), ('execution_site', 'other'), ('execution_backend', 'ssh_local'),
                         ('training_world_size', 8), ('startup_admission_mode', 'old'),
                         ('data_selection_mode', 'question_probe'), ('full_state_restore', {'completed_outer': 5}),
                         ('question_probe', {'probe_architecture': 'qwen2_decoder3_binary_mix_v1'}),
                         ('question_probe', {'architecture': 'qwen2_prefix20_trajectory_em_v1'})]
            for field, value in mutations:
                with self.subTest(field=field, value=value):
                    self.assertFalse(selected(dict(run, **{field: value})))

    def test_prospective_plan_cannot_be_relabelled_as_actual_model_or_outer_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run, artifact = self.prepare_fixture(base)
            for field in ('actual_model_initialization_accepted', 'actual_first_outer_accepted',
                          'actual_behavior_probability_accepted', 'standalone_synthetic_gpu_smoke_performed'):
                changed = copy.deepcopy(artifact)
                changed['probability'][field] = True
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    safety.verify_plan(base, changed, run)

    def test_source_binding_or_nonzero_optimizer_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run, artifact = self.prepare_fixture(base)
            for mutation in ('source', 'writes', 'cuda_count', 'completed'):
                changed, actual = copy.deepcopy(artifact), copy.deepcopy(run)
                if mutation == 'source':
                    changed['probability']['source_sha256'] = {'foreign.py': 'f' * 64}
                elif mutation == 'writes':
                    changed['probability']['actual_optimizer_writes'] = 1
                elif mutation == 'cuda_count':
                    changed['probability']['actual_cuda_runtime_uuids'] = ['foreign'] * 4
                else:
                    actual['completed_updates'] = 1
                with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                    safety.verify_plan(base, changed, actual)

    def test_real_initial_reward_archive_remains_a_required_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run, method, initial, receipts = self.receipt_fixture(base)
            (base / initial['archive']).unlink()
            with patch.dict(os.environ, SLURM_JOB_ID='7001'), self.assertRaises((AssertionError, FileNotFoundError)):
                prepare(base, run, {}, method, initial, receipts)
            self.assertFalse((base / safety.PLAN).exists())

    def test_existing_attempt_state_cannot_be_reset_by_reduced_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run, method, initial, receipts = self.receipt_fixture(base)
            existing = base / 'logs/actual-updates.jsonl'
            original = b'{"step":1,"fixture_actual_state":true}\n'
            existing.write_bytes(original)
            with patch.dict(os.environ, SLURM_JOB_ID='7001'), self.assertRaisesRegex(ValueError, 'already attempted'):
                prepare(base, run, {}, method, initial, receipts)
            self.assertEqual(existing.read_bytes(), original)
            self.assertFalse((base / safety.PLAN).exists())


if __name__ == '__main__':
    unittest.main()
