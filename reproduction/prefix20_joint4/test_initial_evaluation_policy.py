"""CPU policy/receipt fixtures; no generation or GPU acceptance is simulated."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.common8 import configuration, formal_inline_evaluation as evaluation
from reproduction.common8.formal_contract import (
    SAVE_STEPS, SKIP_INITIAL_EVALUATION_POLICY, formal_mathematics,
    skip_initial_evaluation, val_before_train,
)
from reproduction.common8 import test_formal_inline_evaluation as existing_evaluation_tests


class InitialEvaluationPolicyTests(unittest.TestCase):
    def fixture(self, root):
        base,run,rows,expected=existing_evaluation_tests.FormalInlineEvaluationTests().fixture(root)
        run.update(execution_site='hpc3',execution_backend='slurm',training_world_size=4,
            actor_micro_batch_per_rank=4,startup_admission_mode='actual_first_five',
            data_selection_mode='prefix20_probe',pretraining_baseline_policy=SKIP_INITIAL_EVALUATION_POLICY,
            question_probe=dict(probe_architecture='qwen2_prefix20_trajectory_em_v1',
                model_dir=run['initial_model'],head_safetensors=str(base/'assets/head.safetensors'),
                probe_config=str(base/'assets/probe-config.json'),
                eligible_pool_path=str(base/'assets/eligible.json'),output_dir=str(base/'artifacts/question-probe')))
        run['budget']['world_size']=4
        run['budget'].update(formal_mathematics(run))
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        return base,run,rows,expected

    def test_recipe_disables_initial_stage_only_and_preserves_fixed_training_horizon(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,*_=self.fixture(root)
            _,args=configuration.arguments(base)
            values=dict(arg.split('=',1) for arg in args)
            self.assertEqual(values['+trainer.val_before_train'],'false')
            self.assertEqual(values['+trainer.acceptance_step'],'5')
            self.assertEqual(values['trainer.test_freq'],'50')
            self.assertEqual(values['trainer.save_freq'],'50')
            self.assertEqual(values['trainer.total_training_steps'],'201')
            self.assertEqual(SAVE_STEPS,(5,50,100,150,200))
            self.assertFalse(val_before_train(run))

    def test_dev0_records_truthful_skip_without_calling_validate_or_claiming_before_rows(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,*_=self.fixture(root)
            trainer=SimpleNamespace(global_steps=0,_validate=Mock(side_effect=AssertionError('must start training')))
            with patch.object(evaluation,'active_run',return_value=(base,run)), patch(
                    'reproduction.common8.formal_inline_safety.verify_plan',return_value=(.01,'a'*64)):
                result=evaluation.evaluate_development(trainer)
            trainer._validate.assert_not_called()
            self.assertFalse(result['current_run_evaluation_performed'])
            self.assertFalse(result['historical_comparison'])
            self.assertFalse((base/evaluation.PATHS[0]).exists())
            self.assertTrue((base/evaluation.SKIPPED_INITIAL).is_file())
            self.assertNotIn('weighted_em',result)
            self.assertEqual(trainer.global_steps,0)

    def test_after5_remains_actual_and_after_only_reader_rejects_forged_skip(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,rows,expected=self.fixture(root)
            with patch.object(evaluation,'active_run',return_value=(base,run)), patch(
                    'reproduction.common8.validation_identity.expected',return_value=expected), patch(
                    'reproduction.common8.formal_inline_safety.verify_plan',return_value=(.01,'a'*64)):
                evaluation.record_skipped_initial(base,run)
                trainer=SimpleNamespace(global_steps=5,val_dataloader=object(),_validate=Mock(return_value={'actual':'fixture'}))
                self.assertEqual(evaluation.evaluate_development(trainer),{'actual':'fixture'})
                trainer._validate.assert_called_once_with(inline_development=True)
                evaluation.record_development(5,rows,[row['em'] for row in rows])
                result=evaluation.verify_pair(base,run)
                self.assertIsNone(result['before'])
                self.assertEqual(result['after']['correct_count'],512)
                self.assertEqual(result['changes'],{})
                self.assertFalse(result['paired_same_run'])
                self.assertEqual(set(result['file_sha256']),{'skipped0','5'})
                path=base/evaluation.SKIPPED_INITIAL;original=json.loads(path.read_text())
                for mutation in ({'weighted_em':1.0},{'current_run_evaluation_performed':True},{'run_id':'foreign'}):
                    with self.subTest(mutation=mutation):
                        path.write_text(json.dumps(dict(original,**mutation)))
                        with self.assertRaises(AssertionError):evaluation.verify_pair(base,run)

    def test_other_routes_cannot_opt_in_and_existing_default_or_historical_policy_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            _,run,*_=self.fixture(root)
            for mutation in ({'execution_site':'h1008'},{'training_world_size':8},{'execution_backend':'ssh_local'},
                    {'startup_admission_mode':'standalone'},{'full_state_restore':{'completed_outer':5}},
                    {'question_probe':{'probe_architecture':'qwen2_decoder3_binary_mix_v1'}}):
                with self.subTest(mutation=mutation),self.assertRaises(AssertionError):
                    skip_initial_evaluation(dict(run,**mutation))
            old=copy.deepcopy(run);old.pop('pretraining_baseline_policy')
            self.assertFalse(skip_initial_evaluation(old));self.assertTrue(val_before_train(old))
            old['pretraining_baseline_policy']='reuse_historical'
            self.assertFalse(skip_initial_evaluation(old));self.assertFalse(val_before_train(old))


if __name__=='__main__':unittest.main()
