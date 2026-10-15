"""Constructed CPU/Gloo regressions; never native GPU/GRPO acceptance evidence.

These tests are source artifacts until the separately required fresh review has
finished. They execute the actual actor opportunity loop and optimizer method,
with synthetic model inputs and real CPU Adam/Gloo collectives.
"""
import ast
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.common8.recoverable_attempt import AttemptProgress
from reproduction.common8.scheduler import (BASE_LR,CONTRACT_KIND,OuterSchedulerEvidence,
    SchedulerError,expected_lr,fixed_outer_warmup_scheduler)


def _fixture(base):
    (base/'logs').mkdir()
    run=dict(profile='common8',contract_kind=CONTRACT_KIND,policy_safety_mode='inline_first_five',
        run_id='a'*32,source_commit='b'*40,source_sha256={'constructed.py':'c'*64},
        completed_updates=0,target_updates=200,budget=dict(contract_kind=CONTRACT_KIND,
        target_actual_outer_updates=200,lr_warmup_steps=57,lr_warmup_steps_ratio=0.,
        learning_rate=BASE_LR,world_size=8,optimizer_steps_per_outer=20))
    (base/'logs/run-manifest.json').write_text(json.dumps(run))
    (base/'logs/native-allocation-admission.json').write_text(json.dumps(
        dict(accepted=True,run_id=run['run_id'],slurm_job_id='12345',scope='constructed CPU only')))


def _rank_result(identity,before,count,rank):
    return dict(identity,rank=rank,status='complete' if count else 'recoverable_guard',
        optimizer_writes_before=before,optimizer_writes_after=before+count,
        optimizer_writes_this_attempt=count,opportunities_processed=20,
        opportunity_statuses=['written']*count+['skipped_guard']*(20-count),
        guard_stage=None if count else 'all_update_opportunities_skipped',
        guard_reasons=[] if count else ['no_successful_adam_write'],complete_outer=bool(count),
        scheduler_advances=int(bool(count)),completed_outer=identity['logical_update']-int(not count))


class CompleteAccountingTests(unittest.TestCase):
    def test_real_dataproto_ray_serialization_and_dp_collection_preserve_all_eight_rank_results(self):
        # Actual offline Ray/DataProto serialization + collection; no Ray service
        # or GPU execution. Target dependencies are required, never skipped.
        import importlib.util
        for dependency in ('ray', 'tensordict', 'numpy', 'torch'):
            self.assertIsNotNone(importlib.util.find_spec(dependency),
                                 'Required target DataProto dependency is missing: '+dependency)
        import numpy as np
        from ray import cloudpickle
        from verl.protocol import DataProto
        from verl.single_controller.base.decorator import collect_dp_compute_data_proto
        from verl.single_controller.base.worker_group import ResourcePool, WorkerGroup
        group=WorkerGroup(ResourcePool([8]));group._workers=list(range(8))
        progress=AttemptProgress();before=0
        for count in (0,13):
            identity=progress.begin()
            outputs=[DataProto(meta_info={'metrics':{'constructed_rank':rank}},
                non_tensor_batch={'recoverable_outer_result':np.array(
                    [_rank_result(identity,before,count,rank)],dtype=object)}) for rank in range(8)]
            returned=[cloudpickle.loads(cloudpickle.dumps(output)) for output in outputs]
            collected=collect_dp_compute_data_proto(group,returned).to('cpu')
            self.assertIsNone(collected.batch)
            self.assertEqual(len(collected),8)
            actual=collected.non_tensor_batch['recoverable_outer_result'].tolist()
            self.assertEqual([row['rank'] for row in actual],list(range(8)))
            self.assertEqual([row['optimizer_writes_this_attempt'] for row in actual],[count]*8)
            result=progress.finish(actual)
            self.assertEqual(result['complete_outer'],count>0)
            self.assertEqual(result['scheduler_advances'],int(count>0))
            before+=count
        identity=progress.begin()
        divergent=[_rank_result(identity,before,1,rank) for rank in range(8)]
        divergent[7]['optimizer_writes_after']+=1
        outputs=[DataProto(non_tensor_batch={'recoverable_outer_result':np.array([row],dtype=object)})
                 for row in divergent]
        collected=collect_dp_compute_data_proto(group,
            [cloudpickle.loads(cloudpickle.dumps(output)) for output in outputs])
        with self.assertRaises(AssertionError):progress.finish(collected.non_tensor_batch['recoverable_outer_result'].tolist())

    def test_every_whole_zero_attempt_is_fresh_and_cannot_advance_first_five(self):
        progress=AttemptProgress();seen=set();writes=0
        for complete,count in enumerate((1,9,20,3,17),1):
            rejected=progress.begin();seen.add(rejected['attempt_id'])
            result=progress.finish([_rank_result(rejected,writes,0,rank) for rank in range(8)])
            self.assertFalse(result['complete_outer']);self.assertEqual(progress.completed_outer,complete-1)
            accepted=progress.begin();self.assertNotIn(accepted['attempt_id'],seen);seen.add(accepted['attempt_id'])
            result=progress.finish([_rank_result(accepted,writes,count,rank) for rank in range(8)])
            writes+=count
            self.assertTrue(result['all_ranks_agreed']);self.assertEqual(progress.completed_outer,complete)
            self.assertEqual(progress.cumulative_writes,writes)
        self.assertEqual(writes,50);self.assertEqual(progress.aborted_attempts,5)
        self.assertEqual(progress.begin()['logical_update'],6)

    def test_one_rank_status_or_actual_write_disagreement_is_fatal(self):
        for field,value in [('status','recoverable_guard'),('optimizer_writes_after',2),
                            ('opportunities_processed',19),('scheduler_advances',0),('rank',6)]:
            with self.subTest(field=field):
                progress=AttemptProgress();identity=progress.begin()
                rows=[_rank_result(identity,0,1,rank) for rank in range(8)]
                rows[7][field]=value
                with self.assertRaises(AssertionError):progress.finish(rows)
                self.assertEqual(progress.completed_outer,0);self.assertEqual(progress.cumulative_writes,0)


class ActualAdamSchedulerTests(unittest.TestCase):
    def test_two_hundred_complete_outers_keep_one_optimizer_and_warmup_with_skips(self):
        import torch
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);_fixture(base)
            parameter=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float32))
            optimizer=torch.optim.AdamW([parameter],lr=BASE_LR)
            scheduler=fixed_outer_warmup_scheduler(optimizer,warmup_steps=57)
            with patch('reproduction.common8.scheduler._start_ticks',return_value=67890):
                evidence=OuterSchedulerEvidence(base,optimizer,scheduler,rank=0,world_size=8,job_id='12345')
            writes=0;attempt=0;initial_optimizer=id(optimizer);initial_parameter=id(parameter)
            try:
                for logical in range(1,201):
                    attempt+=1
                    evidence.begin_outer(logical,optimizer_writes=writes,batch_gate_sha256='d'*64,
                        attempt_id=format(attempt,'032x'),attempt_index=attempt)
                    evidence.abort_attempt(optimizer_writes=writes,opportunities_processed=20,
                        guard_stage='all_update_opportunities_skipped',guard_reasons=['no_successful_adam_write'])
                    self.assertEqual(evidence.completed_outer,logical-1)
                    self.assertEqual(scheduler.last_epoch,logical-1)
                    attempt+=1
                    evidence.begin_outer(logical,optimizer_writes=writes,batch_gate_sha256='e'*64,
                        attempt_id=format(attempt,'032x'),attempt_index=attempt)
                    successes=(logical%20)+1
                    for _ in range(successes):
                        self.assertEqual(optimizer.param_groups[0]['lr'],expected_lr(logical-1))
                        parameter.grad=torch.ones_like(parameter);optimizer.step();writes+=1
                    row=evidence.complete_outer(optimizer_writes=writes,opportunities_processed=20)
                    self.assertEqual(row['actual_optimizer_writes'],successes)
                    self.assertEqual(int(optimizer.state[parameter]['step']),writes)
                    self.assertEqual(id(optimizer),initial_optimizer);self.assertEqual(id(parameter),initial_parameter)
                    if logical==5:
                        self.assertEqual(evidence.completed_outer,5)
                        self.assertLess(writes,100)
                self.assertEqual(evidence.completed_outer,200);self.assertEqual(scheduler.last_epoch,200)
                self.assertLess(writes,4000);self.assertEqual(evidence.actual_writes,writes)
                rows=[json.loads(line) for line in evidence.path.read_text().splitlines()]
                self.assertEqual(len([row for row in rows if row['kind']=='COMPLETE']),200)
                self.assertEqual(len([row for row in rows if row['kind']=='ABORTED_ATTEMPT']),200)
                self.assertEqual(len({row['process_pid'] for row in rows}),1)
                self.assertEqual(len({row['process_start_ticks'] for row in rows}),1)
            finally:evidence.close()

    def test_abandoned_partial_writes_retained_but_all_twenty_with_success_must_complete(self):
        import torch
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);_fixture(base)
            p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p],lr=BASE_LR)
            scheduler=fixed_outer_warmup_scheduler(optimizer,warmup_steps=57)
            with patch('reproduction.common8.scheduler._start_ticks',return_value=67890):
                evidence=OuterSchedulerEvidence(base,optimizer,scheduler,rank=0,world_size=8,job_id='12345')
            try:
                evidence.begin_outer(1,optimizer_writes=0,batch_gate_sha256='d'*64,attempt_id='1'*32,attempt_index=1)
                for _ in range(3):p.grad=torch.ones_like(p);optimizer.step()
                with self.assertRaises(SchedulerError):
                    evidence.abort_attempt(optimizer_writes=3,opportunities_processed=20,
                        guard_stage='guard',guard_reasons=['guard'])
                evidence.abort_attempt(optimizer_writes=3,opportunities_processed=7,
                    guard_stage='abandoned',guard_reasons=['constructed abandonment'])
                self.assertEqual(int(optimizer.state[p]['step']),3);self.assertEqual(scheduler.last_epoch,0)
                evidence.begin_outer(1,optimizer_writes=3,batch_gate_sha256='e'*64,attempt_id='2'*32,attempt_index=2)
                p.grad=torch.ones_like(p);optimizer.step()
                evidence.complete_outer(optimizer_writes=4,opportunities_processed=20)
                self.assertEqual(int(optimizer.state[p]['step']),4);self.assertEqual(scheduler.last_epoch,1)
            finally:evidence.close()


class LedgerDirectoryDurabilityTests(unittest.TestCase):
    def test_first_worker_and_driver_attempt_receipts_sync_file_then_parent(self):
        from reproduction.common8.worker_evidence import _append_local
        from reproduction.common8.recoverable_attempt import record_attempt
        for kind in ('worker','driver'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as name:
                base=Path(name);_fixture(base);seen=[];real=os.fsync
                def sync(fd):
                    seen.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
                    return real(fd)
                result=dict(attempt_index=1,attempt_id='1'*32,all_ranks_agreed=True,real_rank_count=8)
                with patch.dict(os.environ,SEARCH_G1_BASE=str(base),SLURM_JOB_ID='12345'),patch.object(os,'fsync',side_effect=sync):
                    if kind=='worker':_append_local('OPPORTUNITY',dict(rank=0,processed=True))
                    else:record_attempt(result)
                self.assertEqual(seen,['file','directory'])

    def test_first_receipt_parent_sync_failure_is_not_acknowledged(self):
        from reproduction.common8.worker_evidence import _append_local
        from reproduction.common8.recoverable_attempt import record_attempt
        for kind in ('worker','driver'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as name:
                base=Path(name);_fixture(base);real=os.fsync
                def sync(fd):
                    if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('constructed parent durability failure')
                    return real(fd)
                with patch.dict(os.environ,SEARCH_G1_BASE=str(base),SLURM_JOB_ID='12345'),patch.object(os,'fsync',side_effect=sync):
                    with self.assertRaisesRegex(OSError,'parent durability'):
                        if kind=='worker':_append_local('OPPORTUNITY',dict(rank=0,processed=True))
                        else:record_attempt(dict(attempt_index=1,attempt_id='1'*32,all_ranks_agreed=True,real_rank_count=8))
                # Readable new bytes are explicitly NOT acknowledged success.
                receipt=base/'logs'/('optimizer-rank-0.jsonl' if kind=='worker' else 'actual-attempts.jsonl')
                self.assertTrue(receipt.is_file())


class _Config(dict):
    def __getattr__(self,name):return self[name]


class _Batch(dict):
    def __len__(self):return self['responses'].shape[0]
    def __getitem__(self,key):
        if isinstance(key,str):return dict.__getitem__(self,key)
        return _Batch({name:value[key] for name,value in self.items()})
    def clone(self):return _Batch({name:value.clone() for name,value in self.items()})
    def split(self,count):return [self[slice(start,start+count)] for start in range(0,len(self),count)]


def _actual_actor_methods(torch):
    """Use actual source methods without importing GPU-only flash attention."""
    from torch.distributed.fsdp import FullyShardedDataParallel
    from search_g1.numerics import require_finite
    from verl.utils.py_functional import append_to_dict
    source=Path(__file__).resolve().parents[2]/'verl/workers/actor/dp_actor.py'
    actor=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='DataParallelPPOActor')
    methods=[node for node in actor.body if isinstance(node,ast.FunctionDef) and node.name in ('__init__','_update_policy_attempt','_optimizer_step','_invalidate_monitor_forward_cache','_commit_optimizer_write','_rollback_optimizer_write')]
    for method in methods:
        method.decorator_list=[];method.returns=None
        for arg in method.args.args:arg.annotation=None
    from search_g1.compute_precision import ComputePrecision
    class FixtureBase:
        def __init__(self,config):self.config=config
    namespace=dict(torch=torch,os=os,json=json,FSDP=FullyShardedDataParallel,require_finite=require_finite,
        append_to_dict=append_to_dict,ComputePrecision=ComputePrecision,BasePPOActor=FixtureBase,
        verl_F=SimpleNamespace(entropy_from_logits=lambda value:value))
    # Compile the real constructor and methods together: zero-argument super()
    # needs its actual class cell. No manual write-counter fixture injection.
    actor.body=methods;actor.decorator_list=[]
    exec(compile(ast.Module(body=[actor],type_ignores=[]),'actual actor opportunity CPU fixture','exec'),namespace)
    actual=namespace['DataParallelPPOActor']
    return dict(Actor=actual,**{name:getattr(actual,name) for name in ('_update_policy_attempt','_optimizer_step')})


def _gloo_actor_worker(rank,world,endpoint,base_name,whole_skip):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from search_g1.numerics import require_finite
    from search_g1.policy_safety import RecoverableGuard,monitor
    from reproduction.common8.worker_evidence import append
    dist.init_process_group('gloo',rank=rank,world_size=world,init_method=endpoint,timeout=timedelta(seconds=45))
    try:
        # Native topology is constructed; all guard collectives below are the
        # real eight-rank Gloo group, independent from this local source adapter.
        class TorchAdapter:
            distributed=SimpleNamespace(get_world_size=lambda:8,get_rank=dist.get_rank,is_initialized=dist.is_initialized)
            cuda=SimpleNamespace(current_device=lambda:0)
            compile=staticmethod(lambda function,**kwargs:function)
            def __getattr__(self,name):return getattr(torch,name)
        methods=_actual_actor_methods(TorchAdapter())
        module=torch.nn.Linear(1,1,bias=False);module.weight.data.fill_(1.)
        optimizer=torch.optim.AdamW(module.parameters(),lr=1e-6)
        safety=_Config(enabled=True,monitor_trajectories_per_rank=4)
        config=_Config(use_dynamic_bsz=False,ppo_mini_batch_size=16,ppo_micro_batch_size=2,
                       ppo_epochs=1,state_masking=True,use_kl_loss=False,policy_safety=safety,grad_clip=1.,
                       ulysses_sequence_parallel_size=1)
        actor=methods['Actor'](config,module,optimizer)
        assert actor._optimizer_writes==0 and not optimizer.state
        actor._recoverable_guard_mode=True
        actor._optimizer_step=lambda:methods['_optimizer_step'](actor)
        def observe(frozen,safety,tail,stage):
            opportunity=actor._gradient_diagnostic_context['minibatch_index']
            injected=whole_skip
            delta=(module.weight.detach()-1.).double()
            if rank==0 and injected:delta=delta+2.
            report=monitor(delta,torch.ones_like(delta),torch.ones_like(delta),torch.ones_like(delta),
                c=3.,kl_log_bound=30.,target_kl=.2,negative_tail_limit=.01)
            append('POLICY_MONITOR',dict(rank=rank,context=dict(actor._gradient_diagnostic_context),
                stage=stage,optimizer_writes=actor._optimizer_writes,**report),collective=True)
            if not report['allowed']:raise RecoverableGuard(stage=stage,reasons=report['reasons'])
        actor._monitor_old_policy=observe
        def update(mini,**kwargs):
            observe(kwargs['frozen'],safety,.01,'before_minibatch')
            opportunity=actor._gradient_diagnostic_context['minibatch_index']
            # A failure on only one rank must skip the same current opportunity
            # everywhere, then continue through all later opportunities.
            loss=module.weight.sum()
            require_finite({'loss':loss*(float('nan') if rank==0 and opportunity==2 else 1.)},
                stage='before_backward',recoverable=True)
            optimizer.zero_grad();loss.backward();norm=actor._optimizer_step()
            kwargs['metrics'].update({'actor/pg_loss':[float(loss.detach())],'actor/grad_norm':[float(norm)]})
            observe(kwargs['frozen'],safety,.01,'after_minibatch')
        actor._run_minibatch=update
        batch=_Batch({name:torch.ones(320,1,dtype=torch.long if name in ('responses','input_ids','attention_mask','position_ids','loss_mask') else torch.float32)
                      for name in ('responses','input_ids','attention_mask','position_ids','loss_mask','old_log_probs','advantages')})
        class Data(SimpleNamespace):
            def __len__(self):return 320
        data=Data(batch=batch,meta_info=dict(logical_update=1,attempt_id='1'*32,attempt_index=1,
            temperature=1.,behavior_probability_gate=dict(sha256='d'*64)),
            select=lambda **kwargs:SimpleNamespace(batch=batch))
        with patch.dict(os.environ,{'SEARCH_G1_PROFILE':'common8','SEARCH_G1_BASE':base_name,'SLURM_JOB_ID':'12345'}), \
             patch('search_g1.policy_safety.bind_actual_config'), \
             patch('search_g1.policy_safety.bind_policy_phase',return_value=(.01,'c'*64,'production')), \
             patch('reproduction.common8.behavior_runtime.verify_worker_input'), \
             patch('reproduction.common8.formal_inline_safety.publish_optimizer_state'):
            metrics=methods['_update_policy_attempt'](actor,data)
        rows=actor._last_opportunities
        # Current transactional protocol ends this attempt at its first real
        # guard. Earlier accepted Adam writes remain; skipped work is not a full
        # outer and a fresh rollout is required, rather than reusing stale data.
        processed=1 if whole_skip else 3
        assert len(rows)==processed and [row['opportunity_index'] for row in rows]==list(range(processed))
        expected=0 if whole_skip else 2
        assert actor._optimizer_writes==expected
        assert (not optimizer.state) if whole_skip else int(optimizer.state[module.weight]['step'])==expected
        assert rows[-1]['status']=='skipped_guard' and not rows[-1]['optimizer_write_succeeded']
        if not whole_skip:
            assert all(row['status']=='written' for row in rows[:-1])
            assert rows[-1]['optimizer_writes_before']==rows[-1]['optimizer_writes_after']==2
            assert metrics['actor/update_opportunities_this_attempt']==3
        (Path(base_name)/f'constructed-result-{rank}.json').write_text(json.dumps(
            dict(actual_writes=actor._optimizer_writes,opportunities=len(rows),statuses=[row['status'] for row in rows])))
    finally:dist.destroy_process_group()


class ActualGlooOpportunityTests(unittest.TestCase):
    def test_collective_real_guard_ends_attempt_after_retained_adam(self):
        import torch.multiprocessing as mp
        for whole_skip in (False,True):
            with self.subTest(whole_skip=whole_skip),tempfile.TemporaryDirectory() as name:
                base=Path(name);_fixture(base)
                mp.spawn(_gloo_actor_worker,args=(8,'file://'+str(base/'gloo-init'),str(base),whole_skip),nprocs=8,join=True)
                results=[json.loads((base/f'constructed-result-{rank}.json').read_text()) for rank in range(8)]
                self.assertTrue(all(v==results[0] for v in results))
                for rank in range(8):
                    rows=[json.loads(line) for line in (base/f'logs/optimizer-rank-{rank}.jsonl').read_text().splitlines()]
                    self.assertEqual(sum(row['kind']=='POLICY_MONITOR' for row in rows),1 if whole_skip else 5)
                    self.assertEqual(sum(row['kind']=='OPPORTUNITY' for row in rows),1 if whole_skip else 3)
                    self.assertEqual(sum(row['kind']=='OPTIMIZER_WRITE' for row in rows),0 if whole_skip else 2)

    def test_post_write_corruption_and_unapproved_stage_are_fatal(self):
        import torch
        from search_g1.numerics import require_finite
        from search_g1.policy_safety import RecoverableGuard
        for stage,recoverable in [('after_optimizer_write',False),('after_optimizer_write',True),
                                  ('frozen_monitor_identity',True)]:
            with self.subTest(stage=stage,recoverable=recoverable):
                try:require_finite({'state':torch.tensor(float('nan'))},stage=stage,recoverable=recoverable)
                except RecoverableGuard:self.fail('Fatal state/identity corruption cannot become recoverable')
                except RuntimeError:pass
                else:self.fail('Corrupt state did not reject')


if __name__=='__main__':unittest.main()
