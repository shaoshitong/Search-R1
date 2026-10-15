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


if __name__=='__main__':unittest.main(verbosity=2)
