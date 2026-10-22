"""Small CPU-only contracts; never loads official7B or remote assets."""
import copy,math,unittest
import torch
from torch import nn
from .packed_variance_probe import view,plan,pack_group,PackedVarianceHead,PackedVarianceProbe
from .online_packed_variance import metrics

class PackedVarianceContract(unittest.TestCase):
 def group(self,q='q'):
  return dict(qid=q,trajectory_ids=[q+'-t%d'%i for i in range(10)],labels=[0,1]*5)
 def test_population_targets(self):
  for em,want in[([0]*5,0),([1]*5,0),([1,0,0,0,0],.16),([1,1,0,0,0],.24),([1]*5+[0]*5,.25)]:
   mean=sum(em)/len(em);self.assertAlmostEqual(mean*(1-mean),want)
  g=self.group();v=view(g,seed=17,epoch=1,fixed_m=10);self.assertAlmostEqual(v['target_variance'],.25);self.assertEqual(v['target4variance'],1)
 def test_labelblind_variable_M_subset_and_order(self):
  g=self.group();flip=dict(g,labels=[1-y for y in g['labels']]);allwrong=dict(g,labels=[0]*10);ms=set()
  for epoch in range(1,101):
   a=view(g,seed=19,epoch=epoch);b=view(flip,seed=19,epoch=epoch);c=view(allwrong,seed=19,epoch=epoch);ms.add(a['M'])
   self.assertTrue(3<=a['M']<=10);self.assertEqual(len(set(a['trajectory_ids'])),a['M']);self.assertTrue(set(a['trajectory_ids'])<=set(g['trajectory_ids']));self.assertEqual(a['trajectory_ids'],b['trajectory_ids']);self.assertEqual(a['trajectory_ids'],c['trajectory_ids']);self.assertEqual(a['M'],c['M']);self.assertEqual(c['target_variance'],0)
   mean=sum(a['labels'])/a['M'];self.assertAlmostEqual(a['target_variance'],mean*(1-mean))
  self.assertEqual(ms,set(range(3,11)))
  gs={str(i):self.group(str(i))for i in range(128)};a=plan(gs,1);b=plan(gs,2);self.assertEqual(len({v['qid']for v in a}),128);self.assertEqual(a,plan(gs,1));self.assertNotEqual([v['qid']for v in a],[v['qid']for v in b]);self.assertTrue(any(x['trajectory_ids']!=next(z for z in b if z['qid']==x['qid'])['trajectory_ids']for x in a))
 def test_one_question_and_short_padding_label_exclusion(self):
  g=self.group();v=view(g,seed=4,epoch=0,fixed_m=3);records={tid:dict(prompt=[10,11,12],policy20=[20+i]* (i+1))for i,tid in enumerate(v['trajectory_ids'])};packed=pack_group(v,records,[99],pad_token=0)
  self.assertEqual(packed['input_ids'].count(10),1);self.assertEqual(len(packed['input_ids']),3+3*21);self.assertEqual(sum(packed['attention_mask']),3+3+1+2+3);self.assertEqual(sum(packed['token_types'][i]==0 for i in range(len(packed['token_types']))),3)
  changed=dict(v,labels=[1-y for y in v['labels']],target_variance=0,target4variance=0,true_mixed=False);self.assertEqual(pack_group(changed,records,[99],pad_token=0),packed)
  active=[p for p,m in zip(packed['position_ids'],packed['attention_mask'])if m];self.assertEqual(active,list(range(len(active))))
 def test_gradient_detaches_actor_and_updates_block_head(self):
  class Block(nn.Module):
   def __init__(self):super().__init__();self.linear=nn.Linear(16,16)
   def forward(self,x,**kw):return(self.linear(x),)
  class Rotary(nn.Module):
   def forward(self,x,p):return None
  torch.manual_seed(6);head=PackedVarianceHead(hidden=16,dimension=16,max_positions=64);model=PackedVarianceProbe(Block(),Rotary(),False,head);source=torch.randn(2,16,16,requires_grad=True);mask=torch.ones(2,16,dtype=torch.bool);mask[0,-3:]=False;batch=dict(features=source,attention_mask=mask,position_ids=torch.arange(16).repeat(2,1),token_types=torch.ones(2,16,dtype=torch.long));raw=torch.sigmoid(model(batch));target=torch.tensor([.64,.96]);loss=((raw-target)**2).mean();loss.backward();self.assertIsNone(source.grad);self.assertGreater(sum(float(p.grad.abs().sum())for p in model.block.parameters()),0);self.assertGreater(sum(float(p.grad.abs().sum())for p in head.parameters()if p.grad is not None),0);self.assertTrue(torch.isfinite(loss));self.assertTrue(bool(((raw>=0)&(raw<=1)).all()))
 def test_selection_is_predicted_variance_not_labels(self):
  rows=[]
  for i in range(2048):
   labels=[1,0,0,0,0]if i<663 else[0]*5;rows.append(dict(qid='%04d'%i,labels=labels,true_mixed=i<663,target_variance=.16 if i<663 else 0,predicted_variance=i/10000))
  x=metrics(rows,True);self.assertEqual(x['Top512']['Mixed'],0);self.assertEqual(x['predeclared_G5_classification_diagnostic']['variance_threshold'],.08);self.assertFalse(x['strict_EM_used_for_ranking']);self.assertEqual(x['selection_score'],'predicted_population_EM_variance')
if __name__=='__main__':unittest.main()
