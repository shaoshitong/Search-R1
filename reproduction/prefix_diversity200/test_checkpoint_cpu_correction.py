"""Tiny real-schema fixtures; no tensors, models, network or remote files."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .checkpoint_i1i6_cpu import RUNTIME, validate_resume_origin
from .checkpoint_cpu_correction import BUG_HELPER_COMMIT, BUG_READER_SHA256, FAILED_STATUS, proof_name, verify_predecessor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResumeSchemaTests(unittest.TestCase):
    def test_scheduler_is_terminal_plus_actual_joint_not_import_result(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory).resolve();base=parent/'current';source=parent/'fresh'
            science=('model_revision','model_files_sha256','candidate_questions','selected_questions',
                'actor_selected_questions','responses_per_question','prefix_policy_tokens','seed',
                'official_model_dir','canonical_train_sha256','validation_questions_sha256',
                'final_questions_sha256','grounding_calibration_sha256','interaction_contract',
                'grounding_manifest_sha256','grounding_refit_rounds','policy_monitor',
                'probe_learning_rate','probe_updates_per_outer','probe_epochs_per_outer',
                'probe_global_batch_average','actor_global_mini_batch','actor_ppo_epochs','fixed_test_manifest_sha256')
            config=dict.fromkeys(science,'scientific fixture')
            config.update(run_id='fixture',actor_version='fixture-official0',grounding_initial_probe_version='round0',
                source_commit='33ef232e605f7864086d1685d9a4d8ea0d67f4ab',fresh_actor_from_official=False,
                initial_official_fresh_run_source=str(source),resume_joint=dict(source_base=str(source),completed_outer=6))
            initial=dict(config,source_commit=RUNTIME,fresh_actor_from_official=True)
            initial_sha=write(source/'preparation/launch-config.json',initial)
            joint=dict(run_id='fixture',outer=6,actor=dict(run_id='fixture',completed_outer=6,
                actual_adam_steps_per_rank=[120]*8,scheduler_completed_outer=6),
                probe=dict(run_id='fixture',completed_outer=6,actual_adam_steps_per_rank=[120]*8))
            joint_path=source/'outers/outer-000006/complete.json';joint_sha=write(joint_path,joint)
            terminal=dict(source_base=str(source),completed_outer=6,source_config_sha256=initial_sha,
                scheduler_completed_outer=6,actor_steps=[120]*8,probe_steps=[120]*8,joint_sha256=joint_sha)
            terminal_path=base/'preparation/joint-source-terminal-proof.json'
            resume=dict(status='COMPLETE_JOINT_BOUNDARY_IMPORTED_NO_RECOMPUTATION',run_id='fixture',
                source_base=str(source),completed_outer=6,actor_steps=[120]*8,probe_steps=[120]*8,
                terminal_proof_sha256=write(terminal_path,terminal))
            resume_path=base/'preparation/joint-resume-receipt.json';write(resume_path,resume)
            self.assertNotIn('scheduler_completed_outer',resume)
            self.assertEqual(validate_resume_origin(base,config,{'completed_outer':50})['scheduler_completed_outer'],6)
            wrong=copy.deepcopy(terminal);wrong['scheduler_completed_outer']=5
            resume['terminal_proof_sha256']=write(terminal_path,wrong);write(resume_path,resume)
            with self.assertRaises(ValueError):validate_resume_origin(base,config,{'completed_outer':50})
            joint['actor']['scheduler_completed_outer']=5
            terminal['joint_sha256']=write(joint_path,joint)
            resume['terminal_proof_sha256']=write(terminal_path,terminal);write(resume_path,resume)
            with self.assertRaises(ValueError):validate_resume_origin(base,config,{'completed_outer':50})


class CorrectionEligibilityTests(unittest.TestCase):
    def fixture(self, base):
        config=dict(base=str(base),run_id='fixture',outer=50,audit_version='v2',
            predecessor_control_name='acceptance50-cpu-control-v1',control_name='acceptance50-cpu-control-v2',
            unit='prefix-diversity-4a1cc2d5-cal1-checkpoint50-cpu-v2',runtime_source_commit='source33ef')
        launch=dict(base=str(base),run_id='fixture',source_commit='source33ef')
        config['runtime_config_sha256']=write(base/'preparation/launch-config.json',launch)
        joint_sha=write(base/'outers/outer-000050/complete.json',{'outer':50})
        prior=dict(status=FAILED_STATUS,error_type='KeyError',error="'scheduler_completed_outer'",
            reader_source_sha256=BUG_READER_SHA256,independent_full_file_sha_performed=False,
            actual_cpu_deserialization_performed=False,elapsed_seconds=1.6,run_id='fixture',outer=50,joint_complete_sha256=joint_sha)
        proof=base/'preparation'/(proof_name(50)+'.json')
        config['predecessor_failed_proof_sha256']=sha=write(proof,prior)
        write(proof.with_name(proof.stem+'-intent.json'),dict(run_id='fixture',outer=50,
            joint_complete_sha256=joint_sha,config_sha256=config['runtime_config_sha256'],source_commit='source33ef'))
        previous=base/'acceptance50-cpu-control-v1'
        write(previous/'deploy-receipt.json',dict(helper_commit=BUG_HELPER_COMMIT,config=dict(config),
            files={'reproduction/prefix_diversity200/checkpoint_cpu_once.py':{'sha256':BUG_READER_SHA256}}))
        write(previous/'reader-start.json',dict(helper_commit=BUG_HELPER_COMMIT,pid=2147483647,proc_stat='unused dead pid'))
        write(previous/'reader-exit.json',dict(helper_commit=BUG_HELPER_COMMIT,status='CPU_ACCEPTANCE_FAILED_NO_REPLAY',
            subprocess_exit=1,exit_code=1,proof_sha256=sha,error=prior['error'],elapsed_seconds=1.6,max_rss_kib=427376))
        write(previous/'reader.log',dict(status=FAILED_STATUS,error=prior['error'],proof={'sha256':sha}))
        return config, proof, previous/'reader.log'

    def test_only_pinned_zero_payload_failure_is_correctable_and_v1_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();config,proof,log=self.fixture(base);before=proof.read_bytes()
            result=verify_predecessor(base,config)
            self.assertFalse(result['prior_large_payload_read']);self.assertEqual(proof.read_bytes(),before)
            self.assertNotEqual(proof_name(50,'v1'),proof_name(50,'v2'))
            write(base/'preparation'/(proof_name(50,'v2')+'.json'),{'status':'new distinct fixture'})
            self.assertEqual(proof.read_bytes(),before)
            log.write_text('HF_FULL_SHA_CPU_BEGIN\n'+log.read_text())
            with self.assertRaises(ValueError):verify_predecessor(base,config)

    def test_changed_proof_or_old_namespace_cannot_authorize_second_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();config,proof,log=self.fixture(base)
            wrong=dict(config,control_name='acceptance50-cpu-control-v1')
            with self.assertRaises(ValueError):verify_predecessor(base,wrong)
            prior=json.loads(proof.read_text());prior['error']='another failure';write(proof,prior)
            with self.assertRaises(ValueError):verify_predecessor(base,config)


if __name__=='__main__':unittest.main()
