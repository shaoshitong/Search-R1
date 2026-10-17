"""One small CPU-only scalar fixture for the native joint27 hourly adapter."""
import copy
import json
import unittest

from .a800_actorhidden_prefix5_p10_hourly_format import (
    HISTORY_FIELDS, NATIVE_JOINT_CONTRACT, NATIVE_PARENT_EXECUTION,
    NATIVE_PARENT_JOINT_SHA, NATIVE_PARENT_SOURCE, format_report,
)


def native_fixture():
    run='387d3f6e228a488f90c418e2ba47e28e'; execution='6cc73ac131734c9e8e4522c99f503c90'
    source='c'*40
    prefix='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'
    parent=prefix+run+'-resume-'+NATIVE_PARENT_EXECUTION
    actual=dict(operator_contract=NATIVE_JOINT_CONTRACT,metadata_only=True,models_or_tensors_read=False,
        no_new_evaluation_or_model_readback=True,run_id=run,execution_id=execution,source_commit=source,
        base=prefix+run+'-resume-'+execution,observed_unix=4600.,complete_outer=27,current_outer=28,
        history_columns=list(HISTORY_FIELDS),complete_outer_scalar_history=[],
        probe_clock_contract=dict(period=10,updates_per_period=50,skip_update_at_outer200=True,
            last_update_outer=20,actual_adam_steps_per_rank=[100]*4),probe_update_history=[],
        actor_attempts=[],qa_milestones=[],pipeline_process=None,
        cpu_qualification_is_not_cuda_restoration=True)
    for outer in range(1,28):
        last=outer//10*10; updates=last//10*50; due=outer%10==0
        actual['complete_outer_scalar_history'].append([
            outer,20*outer,last,updates,.5,0.,1.,1.,512,1024,due,
            512 if due else None,0 if due else None,0,0,512,
            NATIVE_PARENT_JOINT_SHA if outer==27 else '%064x'%outer])
        if due:
            actual['probe_update_history'].append(dict(outer=outer,new_adam_steps_per_rank=50,
                actual_adam_steps_per_rank=[updates]*4,actual_learning_rate=5e-6))
    actor=dict(run_id=run,actor_version=run+'-actor0027',completed_outer=27,
        scheduler_completed_outer=27,actual_adam_steps_per_rank=[540]*4)
    probe=dict(run_id=run,completed_outer=20,actual_adam_steps_per_rank=[100]*4)
    actual.update(actor_latest=actor,probe_latest=probe,latest_joint=dict(outer=27,actor=actor,probe=probe))
    actual['native_joint_cpu_qualification']=dict(
        status='NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS',adapter='native_a800_joint_v1',
        run_id=run,source_base=parent,source_execution_id=NATIVE_PARENT_EXECUTION,
        source_commit=NATIVE_PARENT_SOURCE,completed_outer=27,actor_adam_per_rank=[540]*4,
        scheduler_completed_outer=27,probe_completed_outer=20,probe_adam_per_rank=[100]*4,
        owned_births_all_gone=True,source_joint_sha256=NATIVE_PARENT_JOINT_SHA)
    actual['native_joint_import']=dict(status='NATIVE_A800_JOINT_IMPORTED',run_id=run,
        execution_id=execution,source_commit=source,source_base=parent,
        source_execution_id=NATIVE_PARENT_EXECUTION,completed_outer=27,
        actor_adam_per_rank=[540]*4,probe_adam_per_rank=[100]*4,
        full_state_import=True,optimizer_reset=False,cuda_restore_verified=False)
    record=dict(exit_code=0,remote_mutations=False,run_id=run,execution_id=execution,
        source_commit=source,stdout=json.dumps(actual))
    anchor=dict(run_id=run,execution_id=execution,source_commit=source,base=actual['base'],
        delivered=True,delivered_unix=1000.,complete_outers=27,completed_outer=27,
        actor_adam=540,online_probe_adam=100,reward_mean=.5)
    return record,anchor,actual


class NativeJointHourlyFixtureTest(unittest.TestCase):
    def test_existing_native_metadata_formats_without_remote_calls_or_anchor_mutation(self):
        record,anchor,_=native_fixture(); saved=copy.deepcopy(anchor)
        result=format_report(record,anchor,'a'*64,'b'*64)
        self.assertEqual(result['complete_outers'],27)
        self.assertEqual(result['native_joint_resume']['retained_actor_Adam_per_rank'],[540]*4)
        self.assertEqual(result['window']['new_complete_outers'],0)
        self.assertFalse(result['new_evaluation_or_remote_call'])
        self.assertFalse(result['anchor_mutated'])
        self.assertEqual(anchor,saved)

    def test_same_fixture_rejects_broken_qualification_import_and_existing_guards(self):
        mutations=[
            ('native_joint_cpu_qualification','status','FAILED'),
            ('native_joint_cpu_qualification','adapter','other'),
            ('native_joint_cpu_qualification','run_id','wrong'),
            ('native_joint_cpu_qualification','source_base','/data/wrong'),
            ('native_joint_cpu_qualification','source_execution_id','a'*32),
            ('native_joint_cpu_qualification','source_commit','a'*40),
            ('native_joint_cpu_qualification','completed_outer',26),
            ('native_joint_cpu_qualification','actor_adam_per_rank',[540,540,539,540]),
            ('native_joint_cpu_qualification','scheduler_completed_outer',26),
            ('native_joint_cpu_qualification','probe_completed_outer',10),
            ('native_joint_cpu_qualification','probe_adam_per_rank',[50]*4),
            ('native_joint_cpu_qualification','owned_births_all_gone',False),
            ('native_joint_cpu_qualification','source_joint_sha256','f'*64),
            ('native_joint_import','full_state_import',False),
            ('native_joint_import','optimizer_reset',True),
            ('native_joint_import','run_id','wrong'),
            ('native_joint_import','execution_id','a'*32),
            ('native_joint_import','source_commit','a'*40),
            ('native_joint_import','actor_adam_per_rank',[520]*4),
            ('native_joint_import','probe_adam_per_rank',[50]*4),
        ]
        for section,key,value in mutations:
            with self.subTest(section=section,key=key):
                record,anchor,actual=native_fixture()
                actual[section][key]=value;record['stdout']=json.dumps(actual)
                with self.assertRaises(ValueError):format_report(record,anchor,'a'*64,'b'*64)
        record,anchor,actual=native_fixture()
        anchor['execution_id']='a'*32
        with self.assertRaises(ValueError):format_report(record,anchor,'a'*64,'b'*64)
        record,anchor,actual=native_fixture()
        actual['complete_outer_scalar_history'][26][-1]='f'*64
        record['stdout']=json.dumps(actual)
        with self.assertRaises(ValueError):format_report(record,anchor,'a'*64,'b'*64)


if __name__=='__main__':unittest.main()
