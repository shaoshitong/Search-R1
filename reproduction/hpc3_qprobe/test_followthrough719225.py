"""CPU only: real tiny control files with simulated Slurm/child observations.

These fixtures prove once orchestration, not real HPC training or CUDA restore.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import followthrough719225 as worker


class FollowthroughTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();top=Path(self.temp.name).resolve()
        self.root=top/'root';self.base=top/'base';self.directory=self.root/'followthrough/719225'
        self.directory.mkdir(parents=True);(self.base/'logs').mkdir(parents=True)
        (self.base/'preparation').mkdir();self.patches=[]
        for target,name,value in ((worker,'ROOT',self.root),(worker,'BASE',self.base),
                                  (worker,'DIRECTORY',self.directory),(worker.control,'OWNER',os.getuid())):
            p=patch.object(target,name,value);p.start();self.patches.append(p)
        self.actual=dict(job=worker.JOB,state='RUNNING',exit_code='0:0',raw='fixture-only')
        self.run=dict(run_id=worker.RUN,source_commit=worker.PRODUCER)
        self.native=patch.object(worker.control,'producer',return_value=(self.run,dict(producer_job=worker.JOB),'fixture-only'))
        self.native.start()
        self.publish(self.base/'logs/run-manifest.json',self.run)
        self.publish(self.base/'logs/native-allocation-admission.json',dict(fixture_only=True))

    def tearDown(self):
        self.native.stop()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def publish(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def pair(self,step):
        p=self.base/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json'
        self.publish(p,dict(fixture_only=True))
        return self.root/'final-control'/worker.RUN/('archive-'+str(step))

    def observation(self,step,done):
        return dict(status='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0' if done else 'CPU_ARCHIVE_RUNNING',
                    actual_task_exit_code=0 if done else None,runner_alive=not done,
                    actual_payload_bytes=10,actual_payload_files=2,
                    transport_sha256='fixture-sha-'+str(step),transport_path='fixture-transport')

    def submission_binding(self):
        receipt=self.directory/'tiny-source-admission.json';self.publish(receipt,dict(fixture_only=True))
        sha=worker.control.digest(receipt)
        command=['sbatch','--account=jxu957','--qos=formal-user','--partition=emergency_acd',
            '--gres=gpu:4','--cpus-per-task=48','--mem=512G','--exclude=ACD1-31,ACD1-67',
            '--chdir='+str(self.base),str(self.root/'source'/worker.PRODUCER/'reproduction/hpc3_qprobe/train.sh'),
            str(self.base),worker.PRODUCER]
        self.publish(self.base/'preparation/training-submission.json',dict(job=worker.JOB,returncode=0,
            commit=worker.PRODUCER,recovery_of_failed_job='718650'))
        self.publish(self.base/'preparation/training-submit-claim.json',dict(commit=worker.PRODUCER,
            command=command,source_receipt_sha256=sha))
        self.publish(self.base/'preparation/configuration-preflight.json',dict(source_commit=worker.PRODUCER,
            status='ACTUAL_REMOTE_HYDRA_SAVED5_NATIVE4_CONFIGURATION_PASS',training_world_size=4,
            completed_outer=5,next_outer=6,source_receipt_sha256=sha))
        self.publish(self.base/'preparation/checkpoint-restoration.json',dict(source_receipt=str(receipt),source_receipt_sha256=sha))

    def test_fixed_binding_rejects_wrong_job_resources_or_admission_sha(self):
        self.submission_binding()
        with patch.object(worker.control,'caps'):
            self.assertEqual(worker.fixed_binding()['job'],worker.JOB)
            for name,field,value in (('training-submission.json','job','another-job'),
                ('configuration-preflight.json','training_world_size',8),
                ('checkpoint-restoration.json','source_receipt_sha256','changed')):
                path=self.base/'preparation'/name;before=json.loads(path.read_text())
                bad=dict(before);bad[field]=value;self.publish(path,bad)
                with self.subTest(name=name),self.assertRaises(ValueError):worker.fixed_binding()
                self.publish(path,before)
            path=self.base/'preparation/training-submit-claim.json';body=json.loads(path.read_text())
            body['command']=[x.replace('--gres=gpu:4','--gres=gpu:8') for x in body['command']]
            self.publish(path,body)
            with self.assertRaisesRegex(ValueError,'ENTRY_OR_RESOURCES_CHANGED'):worker.fixed_binding()

    def test_queued_without_manifest_never_enters_native_producer_or_archive(self):
        (self.base/'logs/run-manifest.json').unlink();(self.base/'logs/native-allocation-admission.json').unlink()
        with patch.object(worker,'accounting',return_value=dict(self.actual,state='PENDING')),\
             patch.object(worker,'queued_detail',return_value='real detail simulated for CPU fixture'),\
             patch.object(worker.control,'producer') as native,patch.object(worker.control,'launch_archive') as launch:
            result=worker.one_tick('helper',{})
        self.assertEqual(result['phase'],'ACTUAL_PRODUCER_QUEUE_WAIT');native.assert_not_called();launch.assert_not_called()
        self.assertFalse(result['actual_restore_accepted'])

    def test_real_terminal_failure_retains_record_without_restart(self):
        with patch.object(worker,'accounting',return_value=dict(self.actual,state='FAILED',exit_code='1:0')),\
             patch.object(worker.control,'launch_archive') as launch:
            with self.assertRaisesRegex(ValueError,'ACTUAL_PRODUCER_FAILED'):worker.one_tick('helper',{})
        launch.assert_not_called();self.assertEqual(json.loads((self.directory/'producer-terminal-failure.json').read_text())['state'],'FAILED')

    def test_archives_serialize_and_existing_control_is_not_relaunched(self):
        first=self.pair(5);first.mkdir(parents=True);self.pair(50)
        with patch.object(worker,'accounting',return_value=self.actual),\
             patch.object(worker.control,'inspect_archive',return_value=self.observation(5,False)),\
             patch.object(worker.control,'launch_archive') as launch:
            result=worker.one_tick('helper',{})
        launch.assert_not_called();self.assertEqual(list(result['archives']),['5'])

    def test_finished_child_requires_fresh_sha_once_and_new_component_claim(self):
        first=self.pair(5);first.mkdir(parents=True);self.pair(50)
        def observed(base,step,full=False):
            result=self.observation(step,step==5)
            if full:result['fresh_full_sha']={'fixture_actual_byte_check':True}
            return result
        verified={}
        with patch.object(worker,'accounting',return_value=self.actual),\
             patch.object(worker.control,'inspect_archive',side_effect=observed) as check,\
             patch.object(worker.control,'launch_archive') as launch:
            worker.one_tick('helper',verified)
            self.assertEqual(verified,{5:'fixture-sha-5'})
            self.assertEqual(sum(call.kwargs.get('full') is True for call in check.call_args_list),1)
            # Simulate the exclusive existing claim that launch_archive really creates.
            (self.root/'final-control'/worker.RUN/'archive-50').mkdir()
            check.reset_mock();launch.reset_mock();worker.one_tick('helper',verified)
            launch.assert_not_called();self.assertFalse(any(call.kwargs.get('full') is True for call in check.call_args_list))

    def test_dead_archive_without_wait_status_is_not_accepted_or_replayed(self):
        directory=self.pair(5);directory.mkdir(parents=True)
        observed=dict(self.observation(5,False),runner_alive=False,status='CPU_ARCHIVE_STOPPED_WITHOUT_WAIT_RECEIPT')
        with patch.object(worker,'accounting',return_value=self.actual),\
             patch.object(worker.control,'inspect_archive',return_value=observed),\
             patch.object(worker.control,'launch_archive') as launch:
            with self.assertRaisesRegex(ValueError,'UNKNOWN_OUTCOME_INSPECT'):worker.one_tick('helper',{})
        launch.assert_not_called()

    def final_setup(self):
        for step in worker.control.STEPS:self.pair(step).mkdir(parents=True)
        root=self.root/'final-control'/worker.RUN;(root/'audit-controller').mkdir()
        self.final_observed=lambda base,step,full=False:dict(self.observation(step,True),fresh_full_sha={'actual':True})
        return {step:'fixture-sha-'+str(step) for step in worker.control.STEPS}

    def test_final_cpu_child0_with_rejected_native_submit_is_not_success_or_replayed(self):
        verified=self.final_setup()
        with patch.object(worker,'accounting',return_value=dict(self.actual,state='COMPLETED')),\
             patch.object(worker.control,'inspect_archive',side_effect=self.final_observed),\
             patch.object(worker.control,'inspect_audit',return_value=dict(status='SUBMIT_FAILED_DO_NOT_REPLAY',job=None,
                 cpu_controller=dict(status='ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0',actual_task_exit_code=0))),\
             patch.object(worker.control,'start_audit') as submit:
            with self.assertRaisesRegex(ValueError,'ACTUAL_FINAL_SUBMISSION_FAILED'):worker.one_tick('helper',verified)
        submit.assert_not_called()

    def test_final_native_pending_cannot_call_goal_consumer(self):
        verified=self.final_setup()
        with patch.object(worker,'accounting',side_effect=[dict(self.actual,state='COMPLETED'),dict(self.actual,job='final-job',state='PENDING')]),\
             patch.object(worker.control,'inspect_archive',side_effect=self.final_observed),\
             patch.object(worker.control,'inspect_audit',return_value=dict(job='final-job',audit_base='fixture',status='ACTUAL_SLURM_SUBMITTED')),\
             patch('reproduction.hpc3_qprobe.verify_goal.verify') as verify:
            result=worker.one_tick('helper',verified)
        verify.assert_not_called();self.assertEqual(result['phase'],'ACTUAL_FINAL_NATIVE_RESTORE_WAIT')

    def test_completed_native_must_pass_full_goal_consumer_before_result(self):
        verified=self.final_setup();goal=self.base/'logs/qprobe-final-goal-acceptance.json'
        def verify(base,audit_base):
            self.publish(goal,dict(status='QPROBE_FULL200_TEACHER_HEAD_FIXED_VALIDATION_ARCHIVE_NATIVE4_GOAL_PASS',
                run_id=worker.RUN,completed_outer=200,update201_performed=False,fixture_only=True))
        with patch.object(worker,'accounting',side_effect=[dict(self.actual,state='COMPLETED'),dict(self.actual,job='final-job',state='COMPLETED')]),\
             patch.object(worker.control,'inspect_archive',side_effect=self.final_observed),\
             patch.object(worker.control,'inspect_audit',return_value=dict(job='final-job',audit_base='fixture',status='ACTUAL_SLURM_SUBMITTED')),\
             patch('reproduction.hpc3_qprobe.verify_goal.verify',side_effect=verify) as check,\
             patch.object(worker.control,'process',return_value=dict(pid=1,sid=1,start_ticks=2,uid=os.getuid(),state='R')):
            result=worker.one_tick('helper',verified)
        check.assert_called_once();self.assertTrue(result['full_objective_accepted']);self.assertTrue(goal.is_file())

    def test_preexisting_header_only_goal_receipt_cannot_stand_in_for_actual_consumer(self):
        verified=self.final_setup()
        self.publish(self.base/'logs/qprobe-final-goal-acceptance.json',dict(
            status='QPROBE_FULL200_TEACHER_HEAD_FIXED_VALIDATION_ARCHIVE_NATIVE4_GOAL_PASS',
            run_id=worker.RUN,completed_outer=200,update201_performed=False,fixture_only=True))
        with patch.object(worker,'accounting',side_effect=[dict(self.actual,state='COMPLETED'),dict(self.actual,job='final-job',state='COMPLETED')]),\
             patch.object(worker.control,'inspect_archive',side_effect=self.final_observed),\
             patch.object(worker.control,'inspect_audit',return_value=dict(job='final-job',audit_base='fixture',status='ACTUAL_SLURM_SUBMITTED')):
            with self.assertRaises(FileNotFoundError):worker.one_tick('helper',verified)

    def test_known_native_db_lag_wait_is_nonce_bound_and_bounded(self):
        directory=self.root/'final-control'/worker.RUN/'final-audit';directory.mkdir(parents=True)
        audit_base=str(self.root/'audit-nonce');entry=str(self.root/'source/helper/final_restore_audit.sh')
        self.publish(directory/'submit-claim.json',dict(audit_base=audit_base,
            command=['sbatch',entry,audit_base,'helper','receipt','transport']))
        raw='JobId=55 UserId=jxu957(14090) Account=jxu957 QOS=formal-user Partition=emergency_acd '+\
            'WorkDir='+audit_base+' Command='+entry+' JobName=qprobe-final200audit '+\
            'NumCPUs=48 MinMemoryNode=512G NumNodes=1-1 TresPerNode=gres/gpu:4 JobState=PENDING\n'
        worker.REGISTRATION_WAIT.clear()
        with patch.object(worker.subprocess,'check_output',return_value=raw),\
             patch.object(worker.time,'monotonic',return_value=1):
            result=worker.final_registration_wait(dict(job='55',audit_base=audit_base))
        self.assertFalse(result['actual_native_restore_accepted'])
        with patch.object(worker.subprocess,'check_output',return_value=raw.replace(audit_base,'another-workdir')),\
             self.assertRaisesRegex(ValueError,'REGISTRATION_JOB_DIFFERS'):
            worker.final_registration_wait(dict(job='55',audit_base=audit_base))
        with patch.object(worker.subprocess,'check_output',return_value=raw),\
             patch.object(worker.time,'monotonic',return_value=302),self.assertRaisesRegex(ValueError,'REGISTRATION_TIMEOUT'):
            worker.final_registration_wait(dict(job='55',audit_base=audit_base))
        worker.REGISTRATION_WAIT.clear()


if __name__=='__main__':unittest.main()
