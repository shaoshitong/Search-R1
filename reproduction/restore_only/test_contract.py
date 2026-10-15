"""CPU execution-boundary fixtures; never actual GPU reload evidence."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.restore_only import contract as c


class RestoreOnlyContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.receipt=self.root/'receipt.json';self.receipt.write_text('{}')
        producer=Path(os.environ.get('SEARCH_G1_RESTORE_SOURCE_ROOT',Path(__file__).parents[2]))/'reproduction/h1008_qprobe/pipeline.py'
        calls=[node for node in ast.walk(ast.parse(producer.read_text())) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='prepare']
        self.producer_micro=next(keyword.value.value for call in calls for keyword in call.keywords if keyword.arg=='actor_micro_batch_per_rank')
        self.parent=dict(run_id=c.RUN,source_commit=c.RUNTIME,precision_profile='native8_bf16_varlen_fp32_master',
            execution_site='h1008',actor_micro_batch_per_rank=self.producer_micro,initial_model='official',reference_model='official',model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',question_probe=dict(output_dir='old',initial_head_checkpoint='old',snapshot_path='old'))
        self.saved=dict(run_id=c.RUN,step=200,status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',world_size=8,
            checkpoint='source-checkpoint200',attempt_progress=dict(cumulative_writes=3999),source_base=str(c.SOURCE),reward_probe_version='saved',reward_probe_round=4)
    def test_cp200_acceptance_binding_uses_actual_saved_writes_not_opportunities(self):
        old=copy.deepcopy(self.parent)
        run=c.bind_step(self.parent,self.saved,self.receipt,self.root/'output')
        self.assertEqual(run['full_state_restore']['completed_outer'],200)
        self.assertEqual(run['full_state_restore']['actual_optimizer_writes_per_rank'],3999)
        self.assertFalse(run['restore_only_acceptance']['training_updates_allowed'])
        self.assertEqual(self.parent,old)
    def test_frozen_h1008_producer_budget_local4_global32_accumulation4(self):
        from reproduction.common8.topology import actor_global_micro
        self.assertEqual(self.producer_micro,4)
        run=dict(self.parent,profile='common8',execution_backend='slurm',training_world_size=8)
        self.assertEqual(actor_global_micro(run),32)
        self.assertEqual(128//actor_global_micro(run),4)
    def test_signal_during_popen_registration_is_deferred_until_owned_ledger_exists(self):
        from reproduction.restore_only.controller import DeferredInterrupt,start_owned
        import signal
        interrupt=DeferredInterrupt();events=[]
        class Owned:
            def start(self,*args):
                events.append('popen')
                interrupt.handler(signal.SIGTERM,None)
                events.append('birth-ledger-registered')
                return object()
            def observe(self,name):events.append('observed')
        with self.assertRaises(KeyboardInterrupt):start_owned(interrupt,Owned(),'child',[],{},'log','cwd')
        self.assertEqual(events,['popen','birth-ledger-registered','observed'])
    def test_wrong_run_precision_world_or_unsaved_point_refused(self):
        for target,key,value in (('saved','run_id','other'),('saved','world_size',1),('saved','step',201),('parent','source_commit','wrong'),('parent','precision_profile','float32'),('parent','actor_micro_batch_per_rank',2)):
            saved=copy.deepcopy(self.saved);parent=copy.deepcopy(self.parent)
            (saved if target=='saved' else parent)[key]=value
            with self.assertRaises(ValueError):c.bind_step(parent,saved,self.receipt,self.root/'output')
    def test_training_bind_cp200_refusal_is_preserved_in_source(self):
        source=(Path(os.environ.get('SEARCH_G1_RESTORE_SOURCE_ROOT',Path(__file__).parents[2]))/'search_g1/qprobe_checkpoint.py').read_text()
        self.assertIn("_need(receipt['step']<200,'TRAINING_TARGET_ALREADY_COMPLETE')",source)
    def test_runtime_contains_no_fit_generation_backward_or_optimizer_step_call(self):
        tree=ast.parse((Path(__file__).parent/'runtime.py').read_text())
        forbidden={'fit','generate_sequences','update_actor','advance_generation_phase','backward','step','bootstrap','update_completed_outer'}
        actual=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
        self.assertEqual(actual,[])
    def test_live_candidate_stream_roundtrip_is_real_object(self):
        from types import SimpleNamespace
        from search_g1.qprobe_runtime import QProbeCandidateStream
        from search_g1.training_state import compact_stream_state
        from search_g1.full_state_restore import equal_tree
        saved=dict(candidate_order=[2,0,1],candidate_cursor=2,epoch=4,consumed_source_rows=14,pending_source=None,pending_offset=0)
        loader=SimpleNamespace(dataset=[0,1,2])
        actual=QProbeCandidateStream(loader,restored_state=saved)
        self.assertEqual(actual.cursor,2);self.assertEqual(actual.consumed,14)
        equal_tree(compact_stream_state(actual.state()),saved)
        actual._source.cursor=1
        with self.assertRaises(AssertionError):equal_tree(compact_stream_state(actual.state()),saved)
    def test_actual_terminal_guard_precedes_namespace_and_cuda_admission(self):
        from types import SimpleNamespace
        from reproduction.restore_only import controller
        base=self.root/'never-created'
        binding=dict(base=str(base),execution_id='a'*32,trainer_terminal_proof='proof',trainer_terminal_proof_sha256='pin')
        guards=SimpleNamespace(trainer_guard=__import__('unittest.mock',fromlist=['Mock']).Mock(side_effect=ValueError('ACTIVE_TRAINER')))
        native=__import__('unittest.mock',fromlist=['Mock']).Mock()
        with patch.object(controller,'read',return_value=binding),patch.object(controller,'validate_binding',return_value=binding),patch.object(controller,'execution_controls',return_value=(guards,native)),patch.object(controller.socket,'gethostname',return_value='h100-8'),patch.dict(os.environ,SLURM_JOB_ID='999'):
            with self.assertRaisesRegex(ValueError,'ACTIVE_TRAINER'):controller.run(self.receipt)
        self.assertFalse(base.exists());native.native_admission.assert_not_called()
    def test_child_cannot_bypass_live_coordinator_before_importing_ray(self):
        from reproduction.restore_only import controller
        base=self.root/'output';stepbase=base/'step-5'
        binding=dict(base=str(base),execution_id='a'*32)
        intent=dict(binding_sha256=c.sha(self.receipt),coordinator_birth=dict(pid=123))
        with patch.object(controller,'read',side_effect=[binding,intent,{}]),patch.object(controller,'validate_binding',return_value=binding),patch('reproduction.a800.owned_processes.same',return_value=False),patch.dict(os.environ,SEARCH_G1_BASE=str(stepbase)):
            with self.assertRaisesRegex(ValueError,'CHILD_REQUIRES_LIVE_BOUND_COORDINATOR'):controller.child(self.receipt,stepbase,5)
    def test_reused_foreign_pid_is_not_killed_or_claimed_as_owned_gpu_context(self):
        from types import SimpleNamespace
        from reproduction.restore_only import controller
        birth=dict(pid=123,start_ticks=7,sid=123,uid=0)
        owned=SimpleNamespace(jobs={'x':{'ledger':{'123':birth}}})
        with patch('reproduction.a800.owned_processes.same',return_value=False),patch('reproduction.a800.owned_processes.identity',return_value=dict(birth,start_ticks=8)),patch.object(controller.subprocess,'check_output',return_value='123\n'):
            self.assertTrue(controller.wait_owned_release(owned)['owned_gpu_contexts_released'])
    def test_probe_rpc_readback_is_actual_state_not_load_acknowledgement(self):
        import torch
        from search_g1.qprobe_sequence_rpc import execute_probe
        from search_g1.full_state_restore import equal_tree
        class Head:
            version='v';completed_outer=5;update_count=11;actor_snapshot_id='actor:5'
            def load_state_dict(self,state):self.saved=copy.deepcopy(state)
            def state_dict(self):return self.saved
        state=dict(model={'w':torch.tensor([1.])},optimizer={'state':{'step':torch.tensor(2.)}},
            private_rng=torch.tensor([3]),replay=[{'k':1}])
        head=Head();self.assertTrue(execute_probe(head,'load_state_dict',{'state':state})['result'])
        actual=execute_probe(head,'state_dict',{})['result'];equal_tree(actual,state)
        head.saved['model']['w'].add_(1)
        with self.assertRaises(AssertionError):equal_tree(execute_probe(head,'state_dict',{})['result'],state)

if __name__=='__main__':unittest.main()
