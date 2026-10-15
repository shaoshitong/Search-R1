"""Real native Qwen/Adam tests; tiny dimensions keep CPU recovery checks cheap."""
import copy
from dataclasses import replace
from unittest import TestCase, mock
import unittest

import torch

from search_g1.qprobe_periodic_model import restore_engine, fit_phase, REFIT_OUTERS, SAVE_OUTERS
from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig, QwenLastBlockQuestionProbe, QwenBlockWarmup


class Continuity(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.config = QwenBlockQProbeConfig(max_tokens=3, hidden_size=8,
            num_attention_heads=2, num_key_value_heads=1, intermediate_size=16,
            source_layer=1, num_hidden_layers=2, epochs=2, batch_size=2,
            micro_batch_size=1, top_k=2)
        self.inputs = torch.randn(4,3,8)
        self.lengths, self.counts = torch.tensor([3,2,3,1]), torch.tensor([0,1,4,5])
        self.ids = ['a','b','c','d']
        state = QwenLastBlockQuestionProbe(self.config).block.state_dict()
        self.initial = QwenBlockWarmup(self.config, state)
        self.initial.fit(self.inputs,self.lengths,self.counts,self.ids)

    def phase(self, engine):
        return dict(start_steps=engine._adam_steps(),epoch=0,batch=0,order=None,
                    minibatches=[],summaries=[])

    def assert_state(self,a,b):
        from search_g1.full_state_restore import equal_tree
        equal_tree(a.model.state_dict(),b.model.state_dict())
        equal_tree(a.optimizer.state_dict(),b.optimizer.state_dict())
        self.assertTrue(torch.equal(a.rng.get_state(),b.rng.get_state()))
        self.assertEqual(a.update_count,b.update_count)

    def test_exact_boundaries(self):
        self.assertEqual(REFIT_OUTERS,(0,25,50,75,100,125,150,175))
        self.assertEqual(SAVE_OUTERS,(25,50,75,100,125,150,175,200))

    def test_restore_never_copies_student_or_official_parameters(self):
        with mock.patch.object(QwenLastBlockQuestionProbe,'copy_official_block',
                               side_effect=AssertionError('forbidden reinitialization')):
            restored=restore_engine(self.initial.state_dict(),epochs=2)
        self.assert_state(self.initial,restored)

    def test_continuous_adam_and_no_replay(self):
        restored=restore_engine(self.initial.state_dict(),epochs=2)
        before=restored._adam_steps()
        events=[]
        result=fit_phase(restored,self.inputs,self.lengths,self.counts,self.ids,
                         self.phase(restored),lambda kind,p:events.append(kind))
        self.assertEqual(restored._adam_steps(),before+4)
        self.assertEqual(result['replay_groups'],0)
        self.assertEqual(events.count('epoch'),2)
        self.assertEqual(events.count('after_step'),4)

    def test_mid_epoch_failure_continues_exact_next_update(self):
        source=self.initial.state_dict()
        uninterrupted=restore_engine(source,epochs=2)
        fit_phase(uninterrupted,self.inputs,self.lengths,self.counts,self.ids,
                  self.phase(uninterrupted),lambda *a:None)
        interrupted=restore_engine(source,epochs=2)
        phase=self.phase(interrupted)
        checkpoint={}
        def save(kind,p):
            if kind=='after_step' and p['batch']==1 and p['epoch']==0:
                raise RuntimeError('injected error after a real Adam update')
            if kind=='failure':
                checkpoint.update(head=interrupted.state_dict(),phase=copy.deepcopy(p))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            fit_phase(interrupted,self.inputs,self.lengths,self.counts,self.ids,phase,save)
        restored=restore_engine(checkpoint['head'],epochs=2)
        fit_phase(restored,self.inputs,self.lengths,self.counts,self.ids,
                  checkpoint['phase'],lambda *a:None)
        self.assert_state(uninterrupted,restored)

    def test_invalid_adam_phase_is_rejected(self):
        engine=restore_engine(self.initial.state_dict(),epochs=2)
        phase=self.phase(engine);phase['start_steps']-=1
        with self.assertRaisesRegex(ValueError,'COUNTER_MISMATCH'):
            fit_phase(engine,self.inputs,self.lengths,self.counts,self.ids,phase,lambda *a:None)

    def test_probe_gradient_never_reaches_student_features(self):
        features=self.inputs.clone().requires_grad_(True)
        engine=restore_engine(self.initial.state_dict(),epochs=2)
        from search_g1.qprobe_token_model import token_binary_loss
        prediction=engine.model(features,self.lengths)
        loss,_=token_binary_loss(prediction,self.counts,engine.config)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertTrue(any(p.grad is not None for p in engine.model.parameters()))

    def test_student_outers_do_not_fit_or_change_probe_between_boundaries(self):
        from search_g1.qprobe_periodic_runtime import PeriodicRuntime
        runtime=PeriodicRuntime.__new__(PeriodicRuntime)
        runtime.head=mock.Mock()
        runtime.head.version='fixed-weights'
        runtime.head._adam_steps.return_value=400
        runtime.completed_outer=0;runtime.completed_attempts=set()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            runtime.output_dir=Path(root)
            for outer in range(1,201):
                runtime.pending=dict(outer=outer,labels=[{}]*512,
                    prediction_version='fixed-weights',attempt_id='attempt-'+str(outer),prequential={})
                event=runtime.complete_outer(outer,'actor:'+str(outer))
                self.assertEqual(event['head_update']['head_optimizer_updates'],0)
                self.assertEqual(runtime.version,'fixed-weights')
                self.assertEqual(runtime.execution_phase,'PROBE_PENDING' if outer in REFIT_OUTERS else
                    ('FINAL_EVALUATION_PENDING' if outer==200 else 'STUDENT_READY'))
        runtime.head.optimizer.step.assert_not_called()

    def test_real_trainer_boundary_verifies_checkpoint_before_refit(self):
        from search_g1.qprobe_periodic_refit import save_then_refit
        trainer=mock.Mock();trainer.global_steps=25
        trainer.config.trainer.default_local_dir='/data/own/checkpoints'
        events=[]
        trainer._save_checkpoint.side_effect=lambda:events.append('save-full-state')
        trainer.qprobe_runtime.refit_after_checkpoint.side_effect=lambda *a:events.append('rollout-and-fit')
        with mock.patch('reproduction.profile.checkpoint_complete',side_effect=lambda *a:events.append('verify-full-state')), \
                mock.patch('search_g1.qprobe_checkpoint.publish_completed_pair',side_effect=lambda *a,**k:events.append('accepted-pre-pair')):
            save_then_refit(trainer,object())
        self.assertEqual(events,['save-full-state','verify-full-state','accepted-pre-pair','rollout-and-fit'])

    def test_failed_checkpoint_verification_never_starts_refit(self):
        from search_g1.qprobe_periodic_refit import save_then_refit
        trainer=mock.Mock();trainer.global_steps=25
        trainer.config.trainer.default_local_dir='/data/own/checkpoints'
        with mock.patch('reproduction.profile.checkpoint_complete',side_effect=ValueError('actual SHA mismatch')):
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                save_then_refit(trainer,object())
        trainer.qprobe_runtime.refit_after_checkpoint.assert_not_called()

    def test_driver_uses_only_the_first_admitted_slurm_gpu(self):
        from search_g1.qprobe_periodic_driver import validate_route
        run=dict(run_id='own',execution_site='hpc3',training_world_size=4,
            question_probe=dict(head_kind='qwen_native_periodic_lastblock_binary_v1',head_device='cuda:0'))
        allocation=dict(run_id='own',accepted=True,node='ACD1-9',slurm_job_id='42',
            training_world_size=4,actual_cuda_runtime_uuids=['GPU-a','GPU-b','GPU-c','GPU-d'])
        claim=dict(run_id='own',slurm_job_id='42')
        route=validate_route(run,allocation,claim,'ACD1-9','42')
        self.assertEqual(route['gpu_uuid'],'GPU-a')
        self.assertEqual(route['ray_gpu_resources_reserved'],0)
        with self.assertRaisesRegex(ValueError,'ALLOCATION_MISMATCH'):
            validate_route(run,allocation,claim,'ACD1-9','foreign-job')

    def test_hard_kill_cannot_silently_rewind_adam(self):
        from search_g1.qprobe_periodic_refit import verify_phase_recovery
        saved=dict(outer=25,pre_receipt_sha256='pre',head=dict(update_count=420))
        pointer=dict(sha256='state')
        verify_phase_recovery(saved,pointer,actual_sha='state',outer=25,pre_receipt_sha='pre',
                              ledger=dict(possible_adam_steps=420))
        with self.assertRaisesRegex(ValueError,'UNSAVED_ADAM_WRITES'):
            verify_phase_recovery(saved,pointer,actual_sha='state',outer=25,pre_receipt_sha='pre',
                                  ledger=dict(possible_adam_steps=421))
        with self.assertRaisesRegex(ValueError,'DIFFERENT_STUDENT'):
            verify_phase_recovery(saved,pointer,actual_sha='state',outer=50,pre_receipt_sha='pre')


if __name__=='__main__':unittest.main(verbosity=2)
