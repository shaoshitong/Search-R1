"""One bounded CPU check of padding, the actual loss, and warmup scheduling."""
import hashlib
import json
import math
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from search_g1.qprobe_token_model import (
    TokenBinaryQProbeConfig, TokenBinaryQuestionProbe, TokenBinaryWarmup, token_binary_loss,
)


class TokenBinaryWarmupTests(unittest.TestCase):
    def test_failed_once_only_fit_retains_actual_logged_adam_updates(self):
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            trainer = TokenBinaryWarmup(TokenBinaryQProbeConfig(max_tokens=1,feature_dim=3,batch_size=2))
            initial = {key:value.detach().clone() for key,value in trainer.model.state_dict().items()}
            rng_before = trainer.rng.get_state().clone(); events = []
            def fail_after_recording(event):
                events.append(event)
                if event['stage']=='head_optimizer_update':
                    raise RuntimeError('simulated once-only publication failure')
            with self.assertRaisesRegex(RuntimeError,'once-only publication failure'):
                trainer.fit(torch.ones(4,1,3),[1]*4,[0,1,2,5],['a','b','c','d'],
                            progress_callback=fail_after_recording)
            self.assertEqual(trainer.update_count,1)
            self.assertEqual(trainer._adam_steps(),1)
            self.assertFalse(trainer.fitted)
            self.assertEqual(events[-1]['head_adam_steps_after'],1)
            self.assertTrue(trainer.optimizer.state)
            self.assertTrue(any(not torch.equal(value,initial[key]) for key,value in trainer.model.state_dict().items()))
            self.assertFalse(torch.equal(rng_before,trainer.rng.get_state()))
        finally:
            torch.set_num_threads(old_threads)

    def test_real_tokens_padding_loss_and_two_hundred_actual_adam_updates(self):
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            config = TokenBinaryQProbeConfig(max_tokens=3, feature_dim=3)
            fresh = TokenBinaryWarmup(config)
            model = fresh.model.eval()
            features = torch.tensor([[[1.,2.,3.],[-1.,.2,4.],[99.,99.,99.]],
                                     [[.1,-.2,.3],[80.,80.,80.],[70.,70.,70.]]], requires_grad=True)
            lengths = torch.tensor([2,1]); counts = torch.tensor([1,0])
            with mock.patch.object(model.attention,'forward',wraps=model.attention.forward) as attention:
                prediction = model(features,lengths)
                self.assertEqual(attention.call_args.args[0].shape[1],2)  # Only trailing all-padding was trimmed.
            self.assertEqual(prediction['raw_outputs'].shape,(2,2))
            self.assertEqual(model.output.in_features,3*128)
            arbitrary_padding = features.detach().clone()
            arbitrary_padding[0,2] = float('nan'); arbitrary_padding[1,1:] = -12345.
            torch.testing.assert_close(model(arbitrary_padding,lengths)['raw_outputs'],prediction['raw_outputs'],rtol=0,atol=0)
            torch.testing.assert_close(model(features[:,:2],lengths)['raw_outputs'],prediction['raw_outputs'],rtol=0,atol=0)
            changed = features.detach().clone(); changed[0,1,0] += 2.
            self.assertFalse(torch.allclose(model(changed,lengths)['raw_outputs'],prediction['raw_outputs']))
            self.assertTrue(bool(((prediction['predicted_statistic']>=0)&(prediction['predicted_statistic']<=math.sqrt(.3))).all()))
            loss, parts = token_binary_loss(prediction,counts,config)
            target = torch.tensor([math.sqrt(.2),0.])
            bce = F.binary_cross_entropy_with_logits(prediction['raw_outputs'][:,0],torch.tensor([1.,0.]))
            raw = (prediction['raw_outputs'][:,1].sigmoid()*math.sqrt(.3)-target).square().mean()
            manual = bce+.3*raw/.3
            torch.testing.assert_close(loss,manual)
            torch.testing.assert_close(parts['binary_bce'],bce)
            torch.testing.assert_close(parts['regression_mse_raw'],raw)
            actual_gradient, = torch.autograd.grad(loss,model.output.weight,retain_graph=True)
            manual_gradient, = torch.autograd.grad(manual,model.output.weight,retain_graph=True)
            torch.testing.assert_close(actual_gradient,manual_gradient)
            loss.backward()
            self.assertGreater(float(features.grad[0,1].abs().sum()),0.)
            self.assertEqual(float(features.grad[0,2].abs().sum()),0.)
            self.assertEqual(float(features.grad[1,1:].abs().sum()),0.)
            self.assertEqual(float(model.output.weight.grad[:,2*128:].abs().sum()),0.)
            with self.assertRaisesRegex(ValueError,'BMAX'):
                model(torch.zeros(1,4,3),torch.tensor([4]))
            variance_config=TokenBinaryQProbeConfig(max_tokens=1,feature_dim=3,target_statistic='sample_variance')
            variance_prediction=TokenBinaryQuestionProbe(variance_config)(torch.zeros(1,1,3),[1])
            variance_loss, variance_parts=token_binary_loss(variance_prediction,torch.tensor([2]),variance_config)
            expected_variance_mse=(variance_prediction['predicted_statistic']-.3).square().mean()
            torch.testing.assert_close(variance_parts['regression_mse_normalized'],expected_variance_mse/(.3*.3))
            torch.testing.assert_close(variance_loss,variance_parts['binary_bce']+.3*variance_parts['regression_mse_normalized'])

            # One-token CPU fixtures exercise the actual 5120/256*10 schedule.
            train_config = TokenBinaryQProbeConfig(max_tokens=1,feature_dim=3)
            global_rng = torch.get_rng_state().clone()
            trainer = TokenBinaryWarmup(train_config)
            expected_rng = torch.Generator().set_state(trainer.rng.get_state())
            for _ in range(10):torch.randperm(5120,generator=expected_rng)
            ids = [f'question-{i}' for i in range(5120)]
            k = torch.arange(5120)%6; tokens = torch.zeros(5120,1,3)
            calls, events = [], []
            forward = trainer.model.forward
            def capture(features,lengths):
                calls.append((trainer.model.training,trainer._adam_steps(),len(features)))
                return forward(features,lengths)
            with mock.patch.object(trainer.model,'forward',side_effect=capture):
                report = trainer.fit(tokens,torch.ones(5120,dtype=torch.long),k,ids,progress_callback=events.append)
            self.assertEqual(report['actual_head_adam_updates'],200)
            self.assertEqual(report['expected_head_adam_updates'],200)
            self.assertEqual(len(report['minibatch_history']),200)
            self.assertEqual([row['epoch'] for row in report['epoch_history']],list(range(11)))
            self.assertEqual([count for training,count,_ in calls if training],list(range(200)))
            self.assertEqual([count for training,count,_ in calls if not training],
                             [step for step in range(0,201,20) for _ in range(20)])
            self.assertTrue(all(size==256 for _,_,size in calls))
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(expected_rng.get_state(),trainer.rng.get_state()))
            self.assertEqual(len(events),211)
            chosen = sorted(range(5120),key=lambda i:hashlib.sha256(ids[i].encode()).hexdigest())[:512]
            top_hash = hashlib.sha256(json.dumps([ids[i] for i in chosen],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            truth = (k>0)&(k<5); variance = k.float()*(5-k.float())/20
            for epoch,row in enumerate(report['epoch_history']):
                self.assertEqual(row['training_groups'],5120)
                self.assertEqual(row['actual_head_adam_updates'],epoch*20)
                self.assertEqual(row['target_statistic'],'sample_std')
                self.assertEqual(row['auxiliary_weight'],.3)
                self.assertAlmostEqual(row['total_loss'],row['binary_bce']+row['regression_contribution'],places=6)
                self.assertAlmostEqual(row['regression_contribution'],.3*row['regression_mse_normalized'],places=6)
                confusion = row['binary_confusion'];self.assertEqual(sum(confusion.values()),5120)
                top = row['top_k'];self.assertEqual(top['requested_k'],512)
                self.assertEqual(top['selected_question_ids_sha256'],top_hash)
                self.assertNotIn('ids',top)
                self.assertAlmostEqual(top['mixed_precision'],truth[chosen].float().mean().item())
                self.assertAlmostEqual(top['true_sample_std_mean'],variance[chosen].sqrt().mean().item())
                self.assertAlmostEqual(top['true_sample_variance_mean'],variance[chosen].mean().item())
            for index,row in enumerate(report['minibatch_history']):
                self.assertEqual((row['epoch'],row['minibatch_index'],row['batch_size']),(index//20+1,index%20,256))
                self.assertEqual((row['head_adam_steps_before'],row['head_adam_steps_after']),(index,index+1))
                self.assertAlmostEqual(row['pre_step_loss'],row['binary_bce']+row['regression_contribution'],places=6)
            self.assertEqual(trainer.state_dict()['update_count'],200)
            with self.assertRaisesRegex(ValueError,'FRESH'):
                trainer.fit(tokens,torch.ones(5120,dtype=torch.long),k,ids)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
