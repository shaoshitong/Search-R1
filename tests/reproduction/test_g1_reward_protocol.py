import os
from unittest.mock import patch
import unittest
import numpy as np
import torch
from verl import DataProto
from verl.trainer.main_ppo_eco import EcoRewardManager

class Tokenizer:
    def decode(self,ids):return ''.join(chr(int(x)) for x in ids)

class ActualG1RewardProtocol(unittest.TestCase):
    def score(self,segments,mode):
        prompt='Question: capital of France?';response=''.join(s for s,m in segments)
        mask=[1]*len(prompt)+[int(m) for s,m in segments for _ in s]
        tensors=dict(prompts=torch.tensor([[ord(c) for c in prompt]]),responses=torch.tensor([[ord(c) for c in response]]),attention_mask=torch.ones(1,len(prompt+response),dtype=torch.long),info_mask=torch.tensor([mask]))
        data=DataProto.from_dict(tensors,non_tensors=dict(data_source=np.array(['nq'],dtype=object),reward_model=np.array([{'ground_truth':{'target':['Paris']}}],dtype=object)))
        env=dict(ECO_USE_CF='0',ECO_NO_SELF='1',ECO_TRUST_SIGNAL='probe',ECO_PROBE_GATE='1',ECO_PROBE_BASE='0',ECO_WRONG_GROUND_SCALE='0',ECO_BAL_GROUND='0.6',ECO_BAL_NOSEARCH_BONUS='0.5',ECO_W_COST='0.02')
        with patch.dict(os.environ,env):
            manager=EcoRewardManager(Tokenizer(),0,reward_mode=mode)
            # Deliberate finite probe fixtures test actual reward logic, not trained probes.
            manager._probe_call=lambda items:[dict(id=x['id'],flip=.7,boundary=.2,version='fixture') for x in items]
            return float(manager(data).sum())
    def test_nested_and_previous_answers_are_invalid_in_real_manager(self):
        cases=[[('<search><answer>Paris</answer></search>',True)],
               [('<answer>Paris</answer>',True),('observation',False),('<search>q</search>',True)],
               [('<search>q</search>',True),('<information><answer>Paris</answer></information>',False)]]
        for segments in cases:
            self.assertEqual(self.score(segments,'eval'),0)
            self.assertAlmostEqual(self.score(segments,'eco'),-.1,places=6)
    def test_valid_direct_and_searched_answers_keep_g1_formula(self):
        self.assertEqual(self.score([('<answer>Paris</answer>',True)],'eval'),1)
        self.assertAlmostEqual(self.score([('<answer>Paris</answer>',True)],'eco'),1+.5*.2,places=6)
        segments=[('<search>capital of France</search>',True),('<information>Paris</information>',False),('<think>Found it.</think><answer>Paris</answer>',True)]
        self.assertAlmostEqual(self.score(segments,'eco'),1+.6*.7*(1-.2),places=6)
    def test_all_correct_reward_is_one_after_balancing_but_mixed_stays_soft(self):
        ids=np.array(['all','mixed']*5,dtype=object)
        replies=['<answer>Paris</answer>' if i!=5 else '<answer>Rome!</answer>' for i in range(10)]
        prompt='Question: capital of France?'
        width=max(map(len,replies))
        responses=torch.zeros(10,width,dtype=torch.long)
        attention=torch.zeros(10,len(prompt)+width,dtype=torch.long)
        for i,r in enumerate(replies):
            responses[i,:len(r)]=torch.tensor([ord(c) for c in r]);attention[i,:len(prompt)+len(r)]=1
        data=DataProto.from_dict(dict(prompts=torch.tensor([[ord(c) for c in prompt]]*10),responses=responses,
            attention_mask=attention,info_mask=attention.clone()),non_tensors=dict(uid=ids,
            data_source=np.array(['nq']*10,dtype=object),
            reward_model=np.array([{'ground_truth':{'target':['Paris']}}]*10,dtype=object)))
        env=dict(ECO_USE_CF='0',ECO_NO_SELF='1',ECO_TRUST_SIGNAL='probe',ECO_WRONG_GROUND_SCALE='0',ECO_BAL_NOSEARCH_BONUS='0.5')
        with patch.dict(os.environ,env):
            manager=EcoRewardManager(Tokenizer(),0,reward_mode='eco',all_correct_reward_one=True)
            requests=[]
            def probe(items):
                requests.extend(x['id'] for x in items)
                return [dict(id=x['id'],flip=.7,boundary=.2,version='fixture') for x in items]
            manager._probe_call=probe
            scores=manager(data).sum(-1)
            self.assertEqual(requests,['1','3','5','7','9'])
            self.assertTrue(torch.equal(scores[::2],torch.ones(5)))
            self.assertAlmostEqual(float(scores[1]),1.1,places=6)
            self.assertAlmostEqual(float(scores[5]),.05,places=6)
            from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
            adv,_=compute_grpo_outcome_advantage(scores[:,None],torch.ones(10,1),ids,all_correct_virtual_zero=False)
            self.assertTrue(torch.equal(adv[::2],torch.zeros(5,1)))
            self.assertLess(float(adv[5]),0)
            self.assertGreater(float(adv[1]),0)

if __name__=='__main__':unittest.main()
