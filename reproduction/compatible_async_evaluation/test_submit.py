"""Targeted remote CPU fixtures; sbatch/GPU/native observations are mocked."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reproduction.compatible_async_evaluation import submit as s


class OnceSubmitTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.base=Path(temp.name).resolve()
        self.prep=self.base/'preparation';self.prep.mkdir();(self.base/'logs').mkdir()
        self.execution='14b4b25169354f8b92dc0872769fce49'
        role='compatible_async_checkpoint_evaluation';common=dict(role=role,run_id=s.RUN,checkpoint_step=5,split='validation')
        self.manifest=dict(common,base=str(self.base),source_base=str(s.TRAINER),allocation_mode='shared',world_size=8,cpu_request=96,memory_request='512G',control_repo=str(s.SCIENCE_ROOT),source_root=str(s.SCIENCE_ROOT),recipe_sha256=s.RECIPE_SHA,execution_id=self.execution,forbidden_training_job_ids=['3402'],binding=str(self.prep/'binding.json'),config=str(self.prep/'reference-config.json'),journal=str(self.base/'journal'),ray_tmp_dir=str(self.base/'unused-ray'),physical_gpu_uuids=['GPU-'+str(i) for i in range(8)])
        (self.prep/'reference-config.json').write_text('{}')
        self.binding=dict(common,source_root=str(s.SCIENCE_ROOT),control_source_commit=s.SCIENCE_COMMIT,recipe_sha256=s.RECIPE_SHA,resolved_config_sha256=s.sha(self.prep/'reference-config.json'))
        for name,value in [('binding.json',self.binding),('compatible-evaluation-manifest.json',self.manifest),('prepare-intent.json',dict(common,execution_id=self.execution))]:s.once(self.prep/name,value)
        self.receipt=dict(common,binding_sha256=s.sha(self.prep/'binding.json'),manifest_sha256=s.sha(self.prep/'compatible-evaluation-manifest.json'))
        s.once(self.prep/'prepare-receipt.json',self.receipt);self.prepare_sha=s.sha(self.prep/'prepare-receipt.json')
        text='#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH='+s.shlex.quote(str(s.SCIENCE_ROOT))+'\nexec '+s.shlex.join([str(self.base/'env/bin/python'),'-m','reproduction.compatible_async_evaluation.pipeline','--base',str(self.base)])+'\n'
        (self.prep/'launch.sh').write_text(text)
        self.original={name:s.sha(self.prep/name) for name in ('binding.json','compatible-evaluation-manifest.json','prepare-intent.json','prepare-receipt.json','reference-config.json','launch.sh')}
        host=patch.object(s.socket,'gethostname',return_value='h100-8.local');host.start();self.addCleanup(host.stop)

    def call(self):return s.submit(self.base,self.prepare_sha,'terminal.json','a'*64,'b'*40)
    def native(self,job='3456',comment=None):
        return 'JobId='+job+' NodeList=h100-8 JobName=g1ce-'+self.execution[:12]+' Comment='+(comment or 'g1-canonical-eval:'+self.execution)+' WorkDir='+str(self.base)+' Command='+str(self.prep/'submit-launch.sh')+' JobState=RUNNING'

    def test_intent_is_durable_before_single_sbatch_and_prepared_science_unchanged(self):
        def sbatch(command,**kwargs):
            self.assertTrue((self.prep/'submit-intent.json').is_file())
            intent=s.read(self.prep/'submit-intent.json');self.assertEqual(intent['command'],command)
            self.assertEqual(intent['science_control_commit'],s.SCIENCE_COMMIT)
            self.assertIn('--oversubscribe',command);self.assertIn('--cpus-per-task=96',command);self.assertIn('--mem=512G',command)
            self.assertFalse(any(x.startswith('--gres') or x=='--exclusive' for x in command))
            self.assertIn('CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',(self.prep/'submit-launch.sh').read_text())
            self.assertIn(str(self.prep/'launch.sh'),(self.prep/'submit-launch.sh').read_text())
            return SimpleNamespace(returncode=0,stdout='3456\n',stderr='')
        with patch.object(s,'trainer_guard',return_value={'released':True}),patch.object(s,'foreign_guard',return_value={}),patch.object(s.subprocess,'run',side_effect=sbatch) as run:
            result=self.call();self.assertEqual(result['status'],'SUBMITTED')
            with patch.object(s,'queue_snapshot',return_value=[dict(job_id='3456',job_name='g1ce-'+self.execution[:12],state='RUNNING')]),patch.object(s,'query',return_value=self.native()):
                repeated=self.call();self.assertEqual(repeated['status'],'RECONCILED_EXISTING_JOB')
            self.assertEqual(run.call_count,1)
        self.assertEqual(self.original,{name:s.sha(self.prep/name) for name in self.original})

    def test_timeout_is_unknown_and_never_replayed_even_if_queue_empty(self):
        with patch.object(s,'trainer_guard',return_value={}),patch.object(s,'foreign_guard',return_value={}),patch.object(s.subprocess,'run',side_effect=subprocess.TimeoutExpired(['sbatch'],30)) as run:
            self.assertEqual(self.call()['status'],'UNKNOWN_NO_REPLAY')
            original=s.sha(self.prep/'submit-receipt.json')
            with patch.object(s,'queue_snapshot',return_value=[]):self.assertEqual(self.call()['status'],'UNKNOWN_NO_REPLAY')
            with patch.object(s,'queue_snapshot',return_value=[dict(job_id='3456',job_name='g1ce-'+self.execution[:12],state='RUNNING')]),patch.object(s,'query',return_value=self.native()):self.assertEqual(self.call()['status'],'RECONCILED_EXISTING_JOB')
            self.assertEqual(run.call_count,1);self.assertEqual(s.sha(self.prep/'submit-receipt.json'),original)

    def test_active_trainer_or_unproven_terminal_guard_cannot_create_intent(self):
        for error in ('TRAINER_STILL_ACTIVE_OR_QUEUED','PINNED_ACTUAL_TRAINER_TERMINAL_PROOF_REQUIRED'):
            with patch.object(s,'trainer_guard',side_effect=ValueError(error)),patch.object(s.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,error):self.call()
                run.assert_not_called();self.assertFalse((self.prep/'submit-intent.json').exists())

    def test_foreign_binding_change_refuses_before_submit(self):
        with patch.object(s,'trainer_guard',return_value={}),patch.object(s,'foreign_guard',side_effect=ValueError('FOREIGN_ALLOCATION_BINDING_CHANGED')),patch.object(s.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'FOREIGN_ALLOCATION'):self.call()
            run.assert_not_called();self.assertFalse((self.prep/'submit-intent.json').exists())

    def test_pinned_prepared_artifact_or_old_launch_change_refuses(self):
        (self.prep/'launch.sh').write_text('exec NEW_PIPELINE\n')
        with patch.object(s.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'ORIGINAL_PREPARED_LAUNCH'):self.call()
            run.assert_not_called()
        with self.assertRaisesRegex(ValueError,'PINNED_PREPARE_RECEIPT'):s.prepared(self.base,'0'*64)

    def test_partial_submission_artifact_is_not_replayed(self):
        (self.prep/'submit-launch.sh').write_text('partial')
        with patch.object(s.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'PARTIAL_SUBMISSION'):self.call()
            run.assert_not_called()

    def test_wrong_comment_is_not_reconciled_and_query_failure_is_not_absence(self):
        with patch.object(s,'trainer_guard',return_value={}),patch.object(s,'foreign_guard',return_value={}),patch.object(s.subprocess,'run',side_effect=subprocess.TimeoutExpired(['sbatch'],30)):
            self.call()
        with patch.object(s,'queue_snapshot',return_value=[dict(job_id='3456',job_name='g1ce-'+self.execution[:12],state='RUNNING')]),patch.object(s,'query',return_value=self.native(comment='different')):
            self.assertEqual(s.query_existing(self.base)['status'],'UNKNOWN_NO_REPLAY')
        with patch.object(s,'queue_snapshot',side_effect=subprocess.CalledProcessError(1,['squeue'])):
            with self.assertRaises(subprocess.CalledProcessError):s.query_existing(self.base)

    def test_capture_validates_prepared_namespace_and_never_writes_trainer(self):
        target=self.prep/'trainer-terminal-proof-once-v1.json'
        with patch.object(s,'capture_terminal',return_value={'capture':'fixture'}) as capture:
            self.assertEqual(s.capture_prepared_terminal(self.base,self.prepare_sha,target),{'capture':'fixture'})
            capture.assert_called_once_with(target)
        with patch.object(s,'capture_terminal') as capture:
            for base,sha,target in ((s.TRAINER,self.prepare_sha,s.TRAINER/'preparation/trainer-terminal-proof-once-v1.json'),(self.base,None,target),(self.base,self.prepare_sha,self.base/'wrong.json')):
                with self.assertRaises(ValueError):s.capture_prepared_terminal(base,sha,target)
            capture.assert_not_called()
        with patch.object(Path,'open') as opening:
            with self.assertRaisesRegex(ValueError,'CANNOT_WRITE_TRAINER'):s.submit(s.TRAINER,self.prepare_sha,'terminal','a'*64,'b'*40)
            opening.assert_not_called()

    def test_actual_training_cleanup_receipt_is_mandatory(self):
        with patch.object(s,'TRAINER',self.base):
            with self.assertRaises(ValueError):s.released({'jobs':{}})
            s.once(self.base/'logs/cleanup-status.json',dict(run_id=s.RUN,cleaned=False))
            with self.assertRaisesRegex(ValueError,'CLEANUP_NOT_COMPLETE'):s.released({'jobs':{}})

    def test_capture_needs_actual_native_terminal_not_queue_absence(self):
        with patch.object(s,'queue_snapshot',return_value=[]),patch.object(s,'query',return_value='JobId=3402 NodeList=h100-8 JobState=RUNNING ExitCode=0:0'):
            with self.assertRaisesRegex(ValueError,'SAME_JOB_TERMINAL'):s.observe_terminal()
        with patch.object(s,'queue_snapshot',return_value=[]),patch.object(s,'query',side_effect=subprocess.CalledProcessError(1,['scontrol'])):
            with self.assertRaises(subprocess.CalledProcessError):s.observe_terminal()

    def test_retained_terminal_still_rechecks_current_release_and_full200_bindings(self):
        binding=dict(trainer_job_id='3402',trainer_base=str(s.TRAINER),run_id=s.RUN,trainer_run_manifest_sha256='1'*64,native_admission_sha256='2'*64,process_ledger_sha256='3'*64,complete200_pair_sha256='4'*64)
        release=dict(owned_births_released=True,owned_sessions_released=True,native_cgroup_released=True,owned_gpu_contexts_released=True)
        row=dict(binding,**release,native_state='COMPLETED',native_exit_code='0:0',actual_scontrol_stdout='JobId=3402 NodeList=h100-8 JobState=COMPLETED ExitCode=0:0')
        proof=self.prep/'terminal-fixture.json';s.once(proof,dict(role=s.ROLE+'_trainer_terminal',observations=[row,row]))
        with patch.object(s,'queue_snapshot',return_value=[]),patch.object(s,'trainer_bindings',return_value=({'jobs':{}},binding)),patch.object(s,'released',return_value=release) as current:
            result=s.trainer_guard(proof,s.sha(proof));self.assertTrue(result['owned_births_released']);current.assert_called_once()
        with patch.object(s,'queue_snapshot',return_value=[dict(job_id='3402',job_name='trainer',state='RUNNING')]):
            with self.assertRaisesRegex(ValueError,'STILL_ACTIVE'):s.trainer_guard(proof,s.sha(proof))
        with patch.object(s,'queue_snapshot',return_value=[]),patch.object(s,'trainer_bindings',return_value=({'jobs':{}},dict(binding,complete200_pair_sha256='5'*64))):
            with self.assertRaisesRegex(ValueError,'BINDING_DIFFERS'):s.trainer_guard(proof,s.sha(proof))

if __name__=='__main__':unittest.main()
