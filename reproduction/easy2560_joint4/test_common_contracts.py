"""CPU contract checks for physical native4 micros and unchanged logical4 loss."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8 import formal_inline_safety as safety
from reproduction.common8.formal_contract import (
    SKIP_INITIAL_EVALUATION_POLICY, formal_mathematics, skip_initial_evaluation,
)
from reproduction.common8.precision_contract import (
    NATIVE4_BF16_PROFILE, SAMPLING_JOURNAL_POLICY, bf16_policy, easy2560_native4,
)
from reproduction.common8.safety_calibration_evidence import (
    EvidenceError, OBJECTIVE_WEIGHT_DEFINITION, objective_weight_definition, verify_formal_segment,
)
from reproduction.common8.test_formal_inline_safety import guarded_opportunity_fixture
from reproduction.common8.topology import formal_monitor_geometry
from reproduction.h1008_qprobe.first_five_admission import prepare, selected
from reproduction.prefix20_joint4 import test_first_five_startup


def easy_run(physical=32):
    return dict(profile='common8', execution_site='hpc3', execution_backend='slurm',
        training_world_size=4, easy2560_selection=True, data_selection_mode='mixed_first',
        actor_micro_batch_per_rank=physical, actor_logical_micro_batch_per_rank=4,
        precision_profile=NATIVE4_BF16_PROFILE, vllm_gpu_memory_utilization=.1,
        behavior_probability_policy=SAMPLING_JOURNAL_POLICY,
        startup_admission_mode='actual_first_five')


class EasyCommonContractTests(unittest.TestCase):
    def test_physical_fallbacks_preserve_logical4_objective_and_monitor_mass(self):
        for physical in (32, 16, 8, 4):
            with self.subTest(physical=physical):
                run = easy_run(physical)
                self.assertTrue(bf16_policy(run))
                mathematics = formal_mathematics(run)
                self.assertEqual(mathematics['ppo_micro_batch_size'], physical * 4)
                self.assertEqual(mathematics['per_rank_actor_micro_batch'], physical)
                self.assertEqual(mathematics['per_rank_actor_logical_micro_batch'], 4)
                self.assertEqual(mathematics['objective_reduction'], 'original four-row micro masked_mean')
                geometry = formal_monitor_geometry(run)
                self.assertEqual(geometry['monitor_trajectories_per_rank'], 8)
                self.assertEqual(geometry['expected_complete_effective_weight_mass'], 8.)

    def test_physical32_requires_explicit_easy_scope_and_typed_logical4(self):
        for mutation in (dict(easy2560_selection=False), dict(execution_site='h1008'),
                dict(execution_backend='ssh_local'), dict(training_world_size=8),
                dict(data_selection_mode='prefix20_probe'), dict(question_probe={'enabled': True}),
                dict(actor_logical_micro_batch_per_rank=None), dict(actor_logical_micro_batch_per_rank=4.),
                dict(actor_micro_batch_per_rank=2), dict(actor_micro_batch_per_rank=True)):
            run = dict(easy_run(), **mutation)
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                bf16_policy(run)

    def test_old_native4_contract_and_historical_monitor_strings_are_unchanged(self):
        run = easy_run(4)
        for field in ('easy2560_selection', 'data_selection_mode', 'actor_logical_micro_batch_per_rank'):
            run.pop(field)
        self.assertTrue(bf16_policy(run))
        mathematics = formal_mathematics(run)
        self.assertNotIn('per_rank_actor_logical_micro_batch', mathematics)
        self.assertEqual(mathematics['ppo_micro_batch_size'], 16)
        self.assertEqual(objective_weight_definition(run), OBJECTIVE_WEIGHT_DEFINITION)
        self.assertEqual(objective_weight_definition({}), OBJECTIVE_WEIGHT_DEFINITION)

    def test_easy_actual_first_five_does_not_invent_model_or_zero_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run, method, initial, receipts = test_first_five_startup.FirstFiveStartupTests().receipt_fixture(base)
            run.update(easy_run())
            run.pop('question_probe')
            run['budget'].update(formal_mathematics(run))
            run['pretraining_baseline_policy'] = SKIP_INITIAL_EVALUATION_POLICY
            self.assertTrue(easy2560_native4(run))
            self.assertTrue(selected(run))
            self.assertTrue(skip_initial_evaluation(run))
            with patch.dict(os.environ, SLURM_JOB_ID='7001'), patch(
                    'reproduction.common8.production_safety_admission._verify_parent_probability',
                    side_effect=AssertionError('obsolete standalone model smoke route')):
                artifact = prepare(base, run, {}, method, initial, receipts)
                safety.verify_plan(base, artifact, run)
            probability = artifact['probability']
            self.assertEqual(probability['training_world_size'], 4)
            self.assertFalse(probability['actual_model_initialization_accepted'])
            self.assertFalse(probability['actual_first_outer_accepted'])
            self.assertFalse(probability['standalone_synthetic_gpu_smoke_performed'])
            self.assertFalse(artifact['healthy_checkpoint_accepted'])
            self.assertFalse((base/'logs/model-smoke-initial').exists())
            self.assertFalse((base/'logs/formal-inline-development-0.json').exists())
            for mutation in (dict(full_state_restore={'completed_outer': 5}),
                             dict(startup_admission_mode='old'), dict(data_selection_mode='random')):
                changed = dict(run, **mutation)
                self.assertFalse(selected(changed))
                with self.assertRaises(AssertionError):
                    skip_initial_evaluation(changed)

    def test_reader_preserves_eight_logical_micro_mass_for_physical32(self):
        # Statistics are computed by the existing CPU torch monitor from
        # unequal-length masks in groups of four, then read with physical32.
        rows, binding = guarded_opportunity_fixture(4, 4)
        run = easy_run(32)
        binding.update(run)
        for events in rows.values():
            for row in events:
                if row['kind'] == 'POLICY_MONITOR':
                    row['objective_weight_definition'] = objective_weight_definition(run)
                elif row['kind'] == 'OUTER_BUDGET':
                    row['local_micro'] = 32
        proof = verify_formal_segment(rows, binding)
        self.assertEqual(proof['monitor_weight_binding']['expected_complete_effective_weight_mass'], 8.)
        self.assertEqual(proof['processed_opportunities_per_rank'], 20)
        false_weights = copy.deepcopy(rows)
        for events in false_weights.values():
            for row in events:
                if row['kind'] == 'POLICY_MONITOR':
                    for field in ('effective_weight_mass', 'weighted_k3_numerator',
                                  'negative_effective_weight_mass', 'tail_effective_weight_mass'):
                        row[field] /= 2
        with self.assertRaisesRegex(EvidenceError, 'weight mass'):
            verify_formal_segment(false_weights, binding)
        wrong_label = copy.deepcopy(rows)
        wrong_label[0][0]['objective_weight_definition'] = OBJECTIVE_WEIGHT_DEFINITION
        with self.assertRaisesRegex(EvidenceError, 'weighting/scope'):
            verify_formal_segment(wrong_label, binding)

    def test_worker_console_reader_accepts_logical4_and_rejects_physical_weighting(self):
        from reproduction.common8.worker_evidence import read_console_form
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base/'logs').mkdir()
            rows, _ = guarded_opportunity_fixture(4, 4)
            run = dict(easy_run(), run_id='component', policy_safety_required=True,
                       policy_safety_mode='inline_first_five')
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            for rank, events in rows.items():
                for row in events:
                    if row['kind'] == 'POLICY_MONITOR':
                        row['objective_weight_definition'] = objective_weight_definition(run)
                (base/'logs'/f'optimizer-rank-{rank}.jsonl').write_text(
                    ''.join(json.dumps(row)+'\n' for row in events))
            self.assertIn('SEARCH_G1_OUTER_BUDGET=', read_console_form(base, 'component'))
            rows[0][0]['effective_weight_mass'] /= 2
            (base/'logs/optimizer-rank-0.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in rows[0]))
            with self.assertRaises(AssertionError):
                read_console_form(base, 'component')


if __name__ == '__main__':
    unittest.main()
