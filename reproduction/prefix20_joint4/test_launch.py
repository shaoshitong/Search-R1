"""Prospective launch geometry, storage bounds and native retriever routing."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from reproduction.prefix20_joint4.control import launch_command,storage_plan,validate_slurm_job,PERSONAL_ROOT
from reproduction.common8.retriever import native_gpu_binding


class Prefix20LaunchTests(unittest.TestCase):
    def test_native_launch_has_exact_four_gpu_resources_and_no_teacher_dependency(self):
        command=launch_command(Path('/constructed/base'),'a'*40)
        for argument in ('--gres=gpu:4','--cpus-per-task=48','--mem=512G','--partition=emergency_acd',
                         '--qos=formal-user','--account=jxu957'):
            self.assertIn(argument,command)
        self.assertFalse(any('dependency' in arg or 'features' in arg or 'hpc3_qprobe/train' in arg for arg in command))
        self.assertIn('/reproduction/prefix20_joint4/train.sh',command[-3])

    def test_actual_worker_rejects_wrong_job_account_queue_command_or_gpu_count(self):
        base=Path('/constructed/base');commit='a'*40;submitted=dict(job='123',source_commit=commit)
        actual=dict(JobId='123',JobState='RUNNING',Account='jxu957',QOS='formal-user',Partition='emergency_acd',
            JobName='qprobe-prefix20-joint200',NumNodes='1',NumCPUs='48',UserId='jxu957(14090)',
            Command=str(base/'source'/commit/'reproduction/prefix20_joint4/train.sh'),WorkDir=str(base/'repo'),
            TresPerNode='gres/gpu:4',TRES='mem=512G')
        self.assertEqual(validate_slurm_job(base,commit,submitted,actual)['JobId'],'123')
        for key in ('JobId','Account','QOS','Partition','Command','WorkDir','UserId','TresPerNode','NumCPUs'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_slurm_job(base,commit,submitted,dict(actual,**{key:'wrong'}))

    def test_partial_du_cannot_be_treated_as_full_personal_usage(self):
        observation=dict(role='prefix20_prelaunch_personal_storage_observation',account_experiment_root=PERSONAL_ROOT,
            stdout='1000\t'+PERSONAL_ROOT,stderr='Permission denied',returncode=1,measurement_complete=False,checked_unix=100)
        checkpoint=dict(save_steps=[5,50,100,150,200],per_save_allowance_bytes=1000)
        proof=storage_plan(observation,checkpoint,25*10**11,now=101)
        self.assertFalse(proof['accepted']);self.assertTrue(proof['measurement_is_lower_bound'])
        observation.update(stderr='',returncode=0,measurement_complete=True)
        self.assertTrue(storage_plan(observation,checkpoint,25*10**11,now=101)['accepted'])
        observation['stdout']=str(25*10**11)+'\t'+PERSONAL_ROOT
        self.assertFalse(storage_plan(observation,checkpoint,25*10**11,now=101)['accepted'])
        with self.assertRaisesRegex(ValueError,'FRESH_PERSONAL'):storage_plan(observation,checkpoint,25*10**11,now=22000)

    def test_retriever_queries_only_actual_native_four_cuda_ordinals_and_preserves_eight(self):
        for count in (4,8):
            run=dict(profile='common8',training_world_size=count)
            devices=[str(i) for i in range(count)];uuids=['uuid-'+i for i in devices]
            allocation=dict(visible_devices=devices,actual_cuda_runtime_uuids=uuids)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices),clear=False), \
                    patch('reproduction.common8.native_allocation.cuda_uuids',return_value=uuids) as query:
                self.assertEqual(native_gpu_binding(run,dict(TRES='mem=512G'),allocation),(count,uuids))
                query.assert_called_once_with(count)
                allocation['actual_cuda_runtime_uuids']=list(reversed(uuids))
                with self.assertRaises(AssertionError):native_gpu_binding(run,dict(TRES='mem=512G'),allocation)


if __name__=='__main__':unittest.main()
