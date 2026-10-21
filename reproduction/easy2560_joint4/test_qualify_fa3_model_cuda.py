"""CPU mechanics only; no 7B model, CUDA qualification, remote I/O or rollout."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from reproduction.easy2560_joint4 import qualify_fa3_model_cuda as entry
from reproduction.easy2560_joint4.test_qualify_sparse_cuda import fixture as raw_fixture
from search_g1 import sparse_policy_qualification as numerical
from search_g1.test_sparse_policy_qualification import TinyActor, actual_policy_loss


class Actor(TinyActor):
    def __init__(self, offset=0., fail_call=None):
        super().__init__(fail_call=fail_call)
        self._selected_policy_logits = False
        self.flash3_enabled = True
        self.flash3_forward = self.flash3_backward = 0
        self.attention_modes = []
        self.offset = offset

    def _forward_micro_batch(self, *args, **kwargs):
        self.attention_modes.append(self.flash3_enabled)
        _, value = super()._forward_micro_batch(*args, **kwargs)
        if self.flash3_enabled:
            self.flash3_forward += 1
            value = value+self.offset*args[0]['loss_mask']
            def backward(gradient):
                if not self.flash3_enabled: raise RuntimeError('Attention mode changed before backward')
                self.flash3_backward += 1
                return gradient
            value.register_hook(backward)
        return None, value


class Adapter:
    def __init__(self, actor): self.actor=actor

    @contextmanager
    def flash3_mode(self, module, *, enabled):
        previous=self.actor.flash3_enabled; self.actor.flash3_enabled=enabled
        try: yield module
        finally: self.actor.flash3_enabled=previous

    def execution_report(self, module, synchronize=False):
        return dict(attention_layers=1,flash3_enabled=self.actor.flash3_enabled,
            forward_success_returns=dict(varlen=self.actor.flash3_forward,dense=0),
            kernel_backward_success_returns=self.actor.flash3_backward,cpu_fixture_only=True)


def source_fixture(root, step):
    base=root/'new';source=root/'source';checkpoint=source/'checkpoint'/('global_step_'+str(step))
    attempt='a'*32;run_id='same-native4-run';source_commit='d'*40
    pool=source/'artifacts/mixed-candidate-pools'/(attempt+'-pool-0')
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write(source/'logs/run-manifest.json',dict(run_id=run_id,source_commit=source_commit))
    write(checkpoint/'complete.json',dict(step=step,run_id=run_id,finite_all_saved_tensors=True,validated_tensor_count=339))
    write(checkpoint/'config.json',dict(model_type='qwen2',num_hidden_layers=28,hidden_size=3584,
        num_attention_heads=28,num_key_value_heads=4))
    write(checkpoint/'model.safetensors.index.json',dict(weight_map={str(n):'test-shard' for n in range(339)}))
    update=dict(step=step,run_id=run_id,attempt_id=attempt,attempt_index=step+2)
    (source/'logs/actual-updates.jsonl').write_text(json.dumps(update)+'\n'+ '{"step":')
    write(source/'artifacts/mixed-selection'/(attempt+'.json'),dict(step=step,source_commit=source_commit,
        policy_snapshot_id=run_id+':'+str(step)+':policy-'+attempt,pool_retry=0,pool_path=str(pool),
        fixed_candidate_budget=True,attempted_candidates=2560,selected_groups=512,responses_per_group=5))
    pool.mkdir(parents=True);(pool/'0.pt').write_bytes(b'CPU-fixture-placeholder-not-a-tensor')
    receipt=source/'restore-receipt.json'
    write(receipt,dict(kind='easy2560_native4',status='EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS',
        step=step,world_size=4,run_id=run_id,source_base=str(source),source_commit=source_commit,
        checkpoint=str(checkpoint),hf_saved_tensor_count=339))
    run=dict(run_id=run_id,training_world_size=4,full_state_restore=dict(kind='easy2560_native4',completed_outer=step,
        source_base=str(source),source_checkpoint=str(checkpoint),source_receipt=str(receipt),
        source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
    write(base/'logs/run-manifest.json',run)
    return base,source,pool,run


class FA3QualificationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(95)
        self.micro=dict(features=torch.randn(8,6,5,dtype=torch.float64),responses=torch.randint(11,(8,6)),
            loss_mask=torch.tensor([[0,1,1,0,1,0],[1,0,1,1,0,0]]*4),
            old_log_probs=torch.full((8,6),-2.4,dtype=torch.float64),
            advantages=torch.linspace(-.7,.8,48,dtype=torch.float64).reshape(8,6))
        self.loss=actual_policy_loss()

    def qualify(self, actor):
        with patch.object(numerical,'_policy_loss_function',return_value=self.loss):
            return entry.qualify_flash3_policy(actor,self.micro,1.,adapter=Adapter(actor),expected_world_size=1)

    def test_original_fa2_repeats_and_fa3_all_gradients_rng_and_parameters(self):
        actor=Actor();before_rng=numerical._rng(torch.device('cpu'))
        before=entry.parameter_digest(list(actor.actor_module.named_parameters()))
        modes=[m.training for m in actor.actor_module.modules()]
        result=self.qualify(actor)
        self.assertTrue(result['accepted']);self.assertTrue(result['fa2_repeats_pass'])
        self.assertEqual(result['thresholds'],dict(numerical.THRESHOLDS))
        self.assertEqual(actor.attention_modes,[False,False,False,True])
        self.assertEqual((actor.flash3_forward,actor.flash3_backward),(1,1))
        self.assertTrue(actor.flash3_enabled)
        self.assertEqual(len(set(actor.noise)),1)
        self.assertTrue(entry._rng_equal(before_rng,numerical._rng(torch.device('cpu'))))
        self.assertEqual(before,result['local_parameter_bytes_after'])
        self.assertEqual(modes,[m.training for m in actor.actor_module.modules()])
        expected=sum(p.numel() for p in actor.actor_module.parameters())
        for comparison in result['comparisons'].values():
            self.assertEqual(comparison['gradients']['global_compared_gradient_elements'],expected)
            self.assertEqual(comparison['log_probabilities']['global_authored_tokens'],int(self.micro['loss_mask'].sum()))
        self.assertTrue(all(p.grad is None for p in actor.actor_module.parameters()))
        json.dumps(result,allow_nan=False)

    def test_strict_original_thresholds_reject_offset_without_fallback(self):
        result=self.qualify(Actor(offset=.005))
        self.assertFalse(result['accepted']);self.assertTrue(result['fa2_repeats_pass'])
        self.assertIn('log_probability_max_abs',result['comparisons']['FA2-1_FA3-1']['failed_checks'])
        self.assertFalse(result['automatic_fallback'])

    def test_nonfinite_fa3_keeps_original_ppo_rejection_and_cleans_state(self):
        actor=Actor(offset=float('nan'));saved=numerical._rng(torch.device('cpu'))
        with self.assertRaisesRegex(ValueError,'Nonfinite effective Dual-Clip input'):self.qualify(actor)
        self.assertEqual(len(actor.calls),4);self.assertTrue(actor.flash3_enabled)
        self.assertTrue(entry._rng_equal(saved,numerical._rng(torch.device('cpu'))))
        self.assertTrue(all(p.grad is None for p in actor.actor_module.parameters()))

    def test_failure_cleans_mode_rng_gradients_and_does_not_retry(self):
        actor=Actor(fail_call=2);saved=numerical._rng(torch.device('cpu'))
        before=entry.parameter_digest(list(actor.actor_module.named_parameters()))
        with self.assertRaisesRegex(RuntimeError,'independent diagnostic worker failure'):self.qualify(actor)
        self.assertEqual(len(actor.calls),2);self.assertTrue(actor.flash3_enabled)
        self.assertTrue(entry._rng_equal(saved,numerical._rng(torch.device('cpu'))))
        self.assertEqual(before,entry.parameter_digest(list(actor.actor_module.named_parameters())))
        self.assertTrue(all(p.grad is None for p in actor.actor_module.parameters()))

    def test_update_apis_are_forbidden_and_empty_optimizer_stays_empty(self):
        model=torch.nn.Linear(2,2);optimizer=torch.optim.Adam(model.parameters())
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        worker=SimpleNamespace(actor_optimizer=optimizer,actor_lr_scheduler=scheduler)
        counters=entry.block_updates(worker)
        with self.assertRaisesRegex(RuntimeError,'UPDATE_FORBIDDEN'):optimizer.step()
        with self.assertRaisesRegex(RuntimeError,'UPDATE_FORBIDDEN'):scheduler.step()
        self.assertEqual(counters,dict(optimizer_step=1,scheduler_step=1));self.assertFalse(optimizer.state)

    def test_dynamic_cp9_and_later_routes_real_successful_source_step(self):
        for step in (9,27):
            with tempfile.TemporaryDirectory() as name:
                base,source,pool,run=source_fixture(Path(name).resolve(),step)
                bound,raw,proof=entry.source_binding(base)
                self.assertEqual(bound,run);self.assertEqual(raw,pool/'0.pt');self.assertEqual(proof['source_step'],step)
                self.assertEqual(proof['hf_saved_tensor_count'],339)
                original=raw_fixture();original['identity'].update(run_id=run['run_id'],logical_update=step,
                    attempt_id=proof['source_attempt_id'],attempt_index=proof['source_attempt_index'],
                    policy_snapshot_id=proof['source_policy_snapshot_id'])
                entry.validate_raw_identity(original,run,proof)
                micro,selected=entry.prepare_micro_batch(original,3)
                self.assertEqual(micro['responses'].shape[0],8)
                self.assertIn('NOT_production_grounding',selected['advantages']['role'])
                original['identity']['logical_update']=step+1
                with self.assertRaisesRegex(ValueError,'POOL_IDENTITY_DIFFERS'):entry.validate_raw_identity(original,run,proof)
                (pool/'selection-failure.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'FOREIGN_SOURCE_POOL'):entry.source_binding(base)

    def test_cp_before_nine_and_foreign_tensor_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();base,_,_,_=source_fixture(root,8)
            with self.assertRaisesRegex(ValueError,'AT_LEAST_NINE'):entry.source_binding(base)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();base,source,_,_=source_fixture(root,9)
            foreign=source/'0.pt';foreign.write_bytes(b'foreign')
            with self.assertRaisesRegex(ValueError,'SOURCE_POOL_MEMBER'):entry.source_binding(base,foreign)


if __name__=='__main__':unittest.main()
