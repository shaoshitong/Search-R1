"""Constructed CPU scheduler tests, never actual Slurm/GPU/GRPO evidence."""
import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from reproduction.common8.scheduler import (
    BASE_LR, CONTRACT_KIND, OuterSchedulerEvidence, SchedulerError, actor_scheduler,
    constant_scheduler, expected_lr, fixed_outer_warmup_scheduler,
)


class SchedulerTests(unittest.TestCase):
    def test_first_outer_twenty_real_adam_writes_use_requested_lr(self):
        import torch
        parameter=torch.nn.Parameter(torch.tensor([1.0],dtype=torch.float64))
        optimizer=torch.optim.AdamW([parameter],lr=1e-6)
        scheduler=constant_scheduler(optimizer)
        self.assertEqual(scheduler.get_last_lr(),[1e-6])
        previous=parameter.detach().clone()
        for _ in range(20):
            self.assertEqual(optimizer.param_groups[0]['lr'],1e-6)
            parameter.grad=torch.ones_like(parameter);optimizer.step()
            self.assertFalse(torch.equal(previous,parameter.detach()))
            previous=parameter.detach().clone()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-6)

    def test_reject_different_base_lr(self):
        import torch
        p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p],lr=2e-6)
        with self.assertRaises(AssertionError):constant_scheduler(optimizer)


def _optimizer():
    import torch
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    return p,torch.optim.AdamW([p],lr=BASE_LR)


class WarmupTests(unittest.TestCase):
    def test_explicit_fixed57_and_fresh_adam_required(self):
        for wrong in (None,0,56,58,57.,True):
            with self.subTest(warmup=wrong):
                _,optimizer=_optimizer()
                with self.assertRaises(SchedulerError):fixed_outer_warmup_scheduler(optimizer,warmup_steps=wrong)
        p,optimizer=_optimizer();p.grad=__import__('torch').ones_like(p);optimizer.step()
        with self.assertRaisesRegex(SchedulerError,'fresh'):
            fixed_outer_warmup_scheduler(optimizer,warmup_steps=57)
    def test_inherited_scheduler_rejected(self):
        _,optimizer=_optimizer();constant_scheduler(optimizer)
        with self.assertRaisesRegex(SchedulerError,'fresh'):
            fixed_outer_warmup_scheduler(optimizer,warmup_steps=57)
    def test_factory_separates_old_diagnostic_and_new_production(self):
        _,optimizer=_optimizer()
        scheduler=actor_scheduler(optimizer,dict(lr_warmup_steps=57,lr_warmup_steps_ratio=0.,completed_updates=0),
                                  execution_role='production')
        self.assertEqual(scheduler._common8_schedule_kind,CONTRACT_KIND)
        _,optimizer=_optimizer()
        self.assertEqual(actor_scheduler(optimizer,dict(lr_warmup_steps=0),
                         execution_role='policy_safety_calibration_discovery').get_last_lr(),[BASE_LR])
        for role,config in [('production',{}),('production',dict(lr_warmup_steps=57,lr_warmup_steps_ratio=.284)),
                            ('production',dict(lr_warmup_steps=57,completed_updates=1)),
                            ('policy_safety_calibration_discovery',dict(lr_warmup_steps=57)),('unknown',{})]:
            _,optimizer=_optimizer()
            with self.assertRaises(SchedulerError):actor_scheduler(optimizer,config,execution_role=role)
    def test_boundary_formula_and_reject_invalid_counter(self):
        self.assertEqual(expected_lr(0),0.)
        self.assertTrue(math.isclose(expected_lr(1),BASE_LR/57,rel_tol=1e-14))
        self.assertTrue(math.isclose(expected_lr(56),BASE_LR*56/57,rel_tol=1e-14))
        self.assertEqual(expected_lr(57),BASE_LR)
        self.assertEqual(expected_lr(58),BASE_LR)
        self.assertEqual(expected_lr(200),BASE_LR)
        for invalid in (-1,201,True,1.):
            with self.assertRaises(SchedulerError):expected_lr(invalid)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name).resolve();(self.base/'logs').mkdir()
        self.run=dict(profile='common8',contract_kind=CONTRACT_KIND,run_id='a'*32,source_commit='b'*40,
                      source_sha256={'constructed.py':'c'*64},completed_updates=0,target_updates=200,
                      budget=dict(contract_kind=CONTRACT_KIND,target_actual_outer_updates=200,
                                  lr_warmup_steps=57,lr_warmup_steps_ratio=0.,learning_rate=BASE_LR,
                                  world_size=8,optimizer_steps_per_outer=20))
        self._save_run()
        (self.base/'logs/native-allocation-admission.json').write_text(json.dumps(
            dict(accepted=True,run_id=self.run['run_id'],slurm_job_id='12345',
                 scope='constructed CPU fixture only, no native allocation')))
        self.birth=patch('reproduction.common8.scheduler._start_ticks',return_value=67890)
        self.birth.start()
        self.parameter,self.optimizer=_optimizer()
        self.scheduler=fixed_outer_warmup_scheduler(self.optimizer,warmup_steps=57)
        self.recorders=[]
    def tearDown(self):
        for recorder in self.recorders:recorder.close()
        self.birth.stop();self.tmp.cleanup()
    def _save_run(self):
        (self.base/'logs/run-manifest.json').write_text(json.dumps(self.run))
    def recorder(self,**kwargs):
        recorder=OuterSchedulerEvidence(self.base,self.optimizer,self.scheduler,
                                       rank=kwargs.get('rank',0),world_size=kwargs.get('world_size',8),
                                       job_id=kwargs.get('job_id','12345'))
        self.recorders.append(recorder);return recorder
    def begin(self,recorder,step=1,writes=0):
        recorder.begin_outer(step,optimizer_writes=writes,batch_gate_sha256='d'*64)
    def writes(self,n):
        import torch
        for _ in range(n):
            self.parameter.grad=torch.ones_like(self.parameter);self.optimizer.step()
    def rows(self):
        return [json.loads(line) for line in (self.base/'logs/scheduler-rank-0.jsonl').read_text().splitlines()]
    def invoke_actual_worker_method(self,actual_writes,*,fail=False,persistent_complete_io_failure=False):
        """Actual wrapper AST with constructed backend and real CPU Adam only."""
        source=Path(__file__).resolve().parents[2]/'verl/workers/fsdp_workers.py'
        tree=ast.parse(source.read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ActorRolloutRefWorker')
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='update_actor')
        method.decorator_list=[]
        for arg in method.args.args:arg.annotation=None
        method.returns=None
        class Data:
            def __init__(self,meta_info=None):
                self.meta_info=meta_info or dict(logical_update=1,global_token_num=1,
                                               behavior_probability_gate=dict(sha256='d'*64))
                self.batch=SimpleNamespace(cuda=lambda **kwargs:self.batch)
            def to(self,*args,**kwargs):return self
        class Manager:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def preprocess_data(self,data):return data
            def postprocess_data(self,data):return data
        class Clock(Manager):
            def __init__(self,*args,**kwargs):self.last=1.
        backend=SimpleNamespace(device=lambda *args:None,
            cuda=SimpleNamespace(current_device=lambda:0,empty_cache=lambda:None),
            distributed=SimpleNamespace(get_rank=lambda:0,get_world_size=lambda:8))
        namespace=dict(os=os,torch=backend,Timer=Clock,logger=SimpleNamespace(error=lambda *args:None),
                       log_gpu_memory_usage=lambda *args,**kwargs:None,DataProto=Data)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'actual worker wrapper CPU fixture','exec'),namespace)
        actor=SimpleNamespace(_optimizer_writes=0)
        def update_policy(data):
            for _ in range(actual_writes):
                self.writes(1);actor._optimizer_writes+=1
            if fail:raise RuntimeError('constructed collective brake after real CPU Adam writes')
            return {'actor/optimizer_steps_this_update':999}  # Deliberately not trusted.
        actor.update_policy=update_policy
        worker=SimpleNamespace(_is_actor=True,_is_offload_param=False,_is_offload_optimizer=False,
            _assert_generation_phase_inactive=lambda:None,
            ulysses_sharding_manager=Manager(),actor=actor,actor_module=SimpleNamespace(),actor_optimizer=self.optimizer,
            actor_lr_scheduler=self.scheduler,config=SimpleNamespace(actor=SimpleNamespace(ppo_epochs=1)),
            flops_counter=SimpleNamespace(estimate_flops=lambda *args:(1.,1.)),world_size=8)
        original_append=OuterSchedulerEvidence._append
        def append(recorder,row):
            if persistent_complete_io_failure and row['kind'] in ('COMPLETE','POST_COMMIT_FAILURE'):
                raise OSError('persistent constructed receipt failure')
            return original_append(recorder,row)
        try:
            with patch.dict(os.environ,{'SEARCH_G1_PROFILE':'common8','SEARCH_G1_BASE':str(self.base),
                                       'SLURM_JOB_ID':'12345'}), patch(
                    'reproduction.common8.scheduler.collective_call',side_effect=lambda callback,**kwargs:callback()), \
                    patch.object(OuterSchedulerEvidence,'_append',append):
                return namespace['update_actor'](worker,Data())
        finally:
            if hasattr(worker,'_outer_scheduler_evidence'):
                self.recorders.append(worker._outer_scheduler_evidence)
    def test_two_hundred_outers_have_twenty_consumed_lr_writes_and_one_advance(self):
        import torch
        recorder=self.recorder();initial=self.parameter.detach().clone();at5=None
        for step in range(1,201):
            self.begin(recorder,step,(step-1)*20)
            for _ in range(20):
                self.assertEqual(self.optimizer.param_groups[0]['lr'],expected_lr(step-1))
                self.writes(1)
            self.assertEqual(self.scheduler.last_epoch,step-1)
            row=recorder.complete_outer(optimizer_writes=step*20)
            self.assertEqual(row['scheduler_last_epoch_after'],step)
            self.assertEqual(row['scheduler_step_count_after'],step+1)
            self.assertEqual(row['actual_optimizer_writes'],20)
            self.assertEqual(row['next_learning_rates'],[expected_lr(step)])
            if step==1:
                self.assertTrue(torch.equal(self.parameter.detach(),initial))
                self.assertEqual(int(self.optimizer.state[self.parameter]['step']),20)
            if step==5:at5=self.parameter.detach().clone()
        self.assertFalse(torch.equal(initial,at5))
        self.assertEqual(len(self.rows()),600)
        self.assertEqual(self.rows()[0]['consumed_learning_rates'],[0.])
        self.assertEqual(self.rows()[171]['consumed_learning_rates'],[BASE_LR])
        self.assertEqual(self.rows()[-1]['completed_outer_after'],200)
    def test_partial_or_post_write_brake_never_advances_or_rolls_back(self):
        for count in (0,1,19,20):
            with self.subTest(count=count):
                if self.recorders:
                    for old in self.recorders:old.close()
                    self.recorders=[];(self.base/'logs/scheduler-rank-0.jsonl').unlink()
                    self.parameter,self.optimizer=_optimizer()
                    self.scheduler=fixed_outer_warmup_scheduler(self.optimizer,warmup_steps=57)
                recorder=self.recorder();self.begin(recorder);self.writes(count)
                row=recorder.reject_outer(optimizer_writes=count,reason='constructed collective brake')
                self.assertEqual(row['actual_optimizer_writes'],count)
                self.assertEqual(row['scheduler_last_epoch_after'],0)
                self.assertEqual(row['scheduler_step_count_after'],1)
                self.assertEqual(row['completed_outer_after'],0)
                self.assertTrue(row['scheduler_unchanged_on_rejection'])
                if count:self.assertEqual(int(self.optimizer.state[self.parameter]['step']),count)
                with self.assertRaises(SchedulerError):self.begin(recorder)
    def test_short_or_extra_count_cannot_advance_even_after_return(self):
        recorder=self.recorder();self.begin(recorder)
        for wrong in (0,1,19,21):
            with self.assertRaisesRegex(SchedulerError,'Exactly20'):
                recorder.complete_outer(optimizer_writes=wrong)
            self.assertEqual(self.scheduler.last_epoch,0)
            self.assertEqual(self.scheduler._step_count,1)
    def test_counter_sequence_duplicate_and_resume_rejected(self):
        recorder=self.recorder()
        for step,writes in ((2,0),(1,1),(0,0)):
            with self.assertRaises(SchedulerError):self.begin(recorder,step,writes)
        self.begin(recorder)
        with self.assertRaises(SchedulerError):self.begin(recorder)
    def test_prior_receipt_is_not_appended_or_overwritten(self):
        recorder=self.recorder();self.begin(recorder)
        original=(self.base/'logs/scheduler-rank-0.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):self.recorder()
        self.assertEqual((self.base/'logs/scheduler-rank-0.jsonl').read_bytes(),original)
    def test_wrong_run_contract_rank_job_and_source_rejected(self):
        for key,value in [('target_updates',300),('completed_updates',1),('contract_kind','legacy')]:
            saved=self.run[key];self.run[key]=value;self._save_run()
            with self.assertRaises(SchedulerError):self.recorder()
            self.run[key]=saved;self._save_run()
        for kwargs in (dict(rank=8),dict(world_size=4),dict(job_id='12346')):
            with self.assertRaises(SchedulerError):self.recorder(**kwargs)
        recorder=self.recorder();self.run['source_commit']='e'*40;self._save_run()
        with self.assertRaisesRegex(SchedulerError,'identity changed'):self.begin(recorder)
    def test_outside_scheduler_advance_or_lr_mutation_rejected(self):
        recorder=self.recorder();self.optimizer.param_groups[0]['lr']=BASE_LR
        with self.assertRaisesRegex(SchedulerError,'outside'):self.begin(recorder)
    def test_prepare_is_durable_and_does_not_advance(self):
        recorder=self.recorder();self.begin(recorder);self.writes(20)
        recorder.prepare_outer(optimizer_writes=20)
        self.assertEqual(self.rows()[-1]['kind'],'COMMIT_PREPARED')
        self.assertEqual(self.rows()[-1]['actual_optimizer_writes'],20)
        self.assertEqual(self.scheduler.last_epoch,0)
        self.assertEqual(self.scheduler._step_count,1)
        recorder.commit_prepared()
        self.assertEqual(self.rows()[-1]['kind'],'COMPLETE')
        self.assertEqual(self.scheduler.last_epoch,1)
    def test_prepare_io_failure_preserves_actual_adam_and_unchanged_scheduler(self):
        recorder=self.recorder();self.begin(recorder);self.writes(20)
        with patch.object(recorder,'_append',side_effect=OSError('constructed fsync failure')):
            with self.assertRaises(OSError):recorder.prepare_outer(optimizer_writes=20)
        row=recorder.reject_outer(optimizer_writes=20,reason='prepare fsync failed')
        self.assertTrue(row['scheduler_unchanged_on_rejection'])
        self.assertEqual(self.scheduler.last_epoch,0)
        self.assertEqual(int(self.optimizer.state[self.parameter]['step']),20)
    def test_post_commit_io_failure_is_fatal_with_actual_advance(self):
        recorder=self.recorder();self.begin(recorder);self.writes(20)
        recorder.prepare_outer(optimizer_writes=20)
        with patch.object(recorder,'_append',side_effect=OSError('constructed COMPLETE fsync failure')):
            with self.assertRaises(OSError):recorder.commit_prepared()
        row=recorder.fail_after_commit(optimizer_writes=20,reason='COMPLETE fsync failed')
        self.assertEqual(row['kind'],'POST_COMMIT_FAILURE')
        self.assertEqual(row['actual_scheduler_last_epoch'],1)
        self.assertEqual(row['actual_scheduler_step_count'],2)
        self.assertTrue(row['fatal_no_automatic_resume'])
        self.assertEqual(int(self.optimizer.state[self.parameter]['step']),20)
        self.assertNotEqual(self.rows()[-1]['kind'],'REJECTED')
        with self.assertRaises(SchedulerError):self.begin(recorder,2,20)
    def test_actual_worker_console_fallback_preserves_actual_counters_after_persistent_io_failure(self):
        import contextlib,io
        output=io.StringIO()
        with contextlib.redirect_stdout(output),self.assertRaises(OSError):
            self.invoke_actual_worker_method(20,persistent_complete_io_failure=True)
        marker='SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='
        rows=[json.loads(line.split(marker,1)[1]) for line in output.getvalue().splitlines() if marker in line]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['actual_scheduler_last_epoch'],1)
        self.assertEqual(rows[0]['actual_scheduler_step_count'],2)
        self.assertEqual(rows[0]['writes_after'],20)
        self.assertEqual(rows[0]['actual_learning_rates'],[expected_lr(1)])
        self.assertEqual(self.rows()[-1]['kind'],'COMMIT_PREPARED')
        from reproduction.common8.runtime import verify_no_fatal_scheduler_console
        console=self.base/'logs/train.log';console.write_text(output.getvalue())
        with self.assertRaisesRegex(AssertionError,'Fatal scheduler console'):
            verify_no_fatal_scheduler_console(self.base,dict(train_log=str(console)))
    def test_console_fallback_gate_requires_owned_regular_current_console(self):
        from reproduction.common8.runtime import verify_no_fatal_scheduler_console
        import hashlib
        console=self.base/'logs/train.log';console.write_text('constructed successful CPU console\n')
        proof=verify_no_fatal_scheduler_console(self.base,dict(train_log=str(console)))
        self.assertEqual(proof['observed_sha256'],hashlib.sha256(console.read_bytes()).hexdigest())
        console.unlink();console.symlink_to(self.base/'logs/run-manifest.json')
        with self.assertRaises(AssertionError):
            verify_no_fatal_scheduler_console(self.base,dict(train_log=str(console)))
    def test_ledger_rows_bind_run_job_rank_source_gate_and_process(self):
        recorder=self.recorder();self.begin(recorder);self.writes(20)
        recorder.complete_outer(optimizer_writes=20)
        for row in self.rows():
            self.assertEqual(row['run_id'],self.run['run_id']);self.assertEqual(row['slurm_job_id'],'12345')
            self.assertEqual(row['rank'],0);self.assertEqual(row['world_size'],8)
            self.assertEqual(row['source_commit'],self.run['source_commit'])
            self.assertEqual(row['batch_gate_sha256'],'d'*64)
            self.assertEqual(row['process_pid'],os.getpid());self.assertEqual(row['process_start_ticks'],67890)
            self.assertEqual(row['contract_kind'],CONTRACT_KIND)
    def test_actual_worker_wrapper_dynamic_counter_not_metric_is_consumed(self):
        output=self.invoke_actual_worker_method(20)
        self.assertEqual(output.meta_info['metrics']['actor/lr_consumed'],0.)
        self.assertEqual(output.meta_info['metrics']['actor/lr'],expected_lr(1))
        self.assertEqual(self.rows()[-1]['actual_optimizer_writes'],20)
    def test_actual_worker_wrapper_partial_exception_preserves_adam_without_advance(self):
        with self.assertRaisesRegex(RuntimeError,'collective brake'):
            self.invoke_actual_worker_method(3,fail=True)
        self.assertEqual(self.rows()[-1]['kind'],'REJECTED')
        self.assertEqual(self.rows()[-1]['actual_optimizer_writes'],3)
        self.assertEqual(self.scheduler.last_epoch,0)
        self.assertEqual(int(self.optimizer.state[self.parameter]['step']),3)
    def test_actual_worker_return_without_twenty_real_writes_cannot_advance(self):
        with self.assertRaisesRegex(SchedulerError,'Exactly20'):
            self.invoke_actual_worker_method(1)
        self.assertEqual(self.rows()[-1]['kind'],'REJECTED')
        self.assertEqual(self.rows()[-1]['actual_optimizer_writes'],1)
        self.assertEqual(self.scheduler.last_epoch,0)


if __name__=='__main__':unittest.main()
