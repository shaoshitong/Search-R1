"""CPU objective/gradient equivalence for unchanged B32 and four real views."""
import copy,unittest
import torch
from torch import nn
from . import direct_update_fast as fast
from .joint_mixed_direct_pretrain import direct_update,warm_gradient_buckets
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from . import mixed_group_checkpoint as ck,probe_ablation_train as u


class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.block=nn.Linear(3,3);self.group_head=nn.Linear(3,1)
 def encode_trajectories(self,b):return dict(hidden=self.block(b['features'].detach()),attention_mask=b['attention_mask'],position_ids=b['position_ids'])
 def group_logits(self,e):return self.group_head(e['hidden'].mean((1,2))).squeeze(-1)


class Reader:
 def __init__(self,groups):self.rows={tid:(g['qid'],torch.arange(6,dtype=torch.float32).reshape(2,3)*.03+(i%7+j)*.02)for i,g in enumerate(groups)for j,tid in enumerate(g['trajectory_ids'])};self.calls=[];self.inputs=[]
 def get(self,ids,device):
  self.calls.append(list(ids));features=torch.stack([self.rows[t][1]for t in ids]).to(device).requires_grad_(True);self.inputs.append(features)
  return dict(features=features,attention_mask=torch.ones(len(ids),2,dtype=torch.bool,device=device),position_ids=torch.arange(2,device=device).expand(len(ids),-1),trajectory_ids=list(ids),qids=[self.rows[t][0]for t in ids],capture_layer=26)


def setup():
 torch.manual_seed(8);groups=[dict(qid='q%03d'%i,question_hash='h%d'%i,trajectory_ids=['q%d-t%d'%(i,j)for j in range(10)],labels=([0]*10 if i<8 else[1]*10 if i<16 else[0,1]*5))for i in range(64)];m=Tiny();ema=copy.deepcopy(m).eval().requires_grad_(False);b=torch.optim.Adam(m.block.parameters(),lr=5e-6);h=torch.optim.Adam(m.group_head.parameters(),lr=5e-6);warm_gradient_buckets(m);s=BalancedHardG10Sampler(groups,seed=1729,batch_questions=32,steps_per_epoch=2);return m,ema,b,h,s,Reader(groups),{g['qid']:g for g in groups}


class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_micro1_4_8_16_32_matches_scalar_objective_own_views_both_gradients_and_Adam(self):
  device=torch.device('cpu');m,e,b,h,s,r,l=setup();ref,batch=direct_update(m,e,b,h,r,l,s,device)
  for micro in(1,4,8,16,32):
   with self.subTest(micro=micro):
    mm,ee,bb,hh,ss,rr,ll=setup();value,actual=fast.direct_update_fast(mm,ee,bb,hh,rr,ll,ss,device,micro_questions=micro)
    self.assertEqual(batch,actual);self.assertEqual(len(rr.calls),32//micro);self.assertTrue(all(len(ids)==10*micro for ids in rr.calls));self.assertEqual(value['GPU_to_CPU_metric_vectors'],1);self.assertEqual(value['confusion'],ref['confusion']);self.assertAlmostEqual(value['loss'],ref['loss'],places=6);self.assertEqual(value['native_steps'],1);self.assertEqual(value['head_steps'],1);self.assertEqual(value['mixed_views'],value['nonmixed_views']);self.assertGreater(value['native_gradient_L2'],0);self.assertGreater(value['group_head_gradient_L2'],0)
    for a,z in zip(mm.parameters(),m.parameters()):torch.testing.assert_close(a,z,rtol=1e-5,atol=1e-7)
    for a,z in zip(ee.parameters(),e.parameters()):torch.testing.assert_close(a,z,rtol=1e-5,atol=1e-7)
    for o,p in((bb,b),(hh,h)):
     for aa,zz in zip(o.state.values(),p.state.values()):
      for key in('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(aa[key],zz[key],rtol=2e-5,atol=1e-8)
    self.assertTrue(all(x.grad is None for x in rr.inputs));self.assertEqual(ss.state_dict()['accepted_updates'],1)
    for role in('mixed','nonmixed'):
     self.assertEqual(set(ss.difficulty[role]),set(s.difficulty[role]))
     for q,loss in ss.difficulty[role].items():self.assertAlmostEqual(loss,s.difficulty[role][q],places=6)
 def test_fast_same_micro_complete_restore_next_update_is_exact(self):
  m,e,b,h,s,r,l=setup();device=torch.device('cpu');binding=dict(config_sha256='synthetic',matched_binding_sha256='synthetic');provenance=dict(initialization_source='official_initial_Qwen2.5-7B-Instruct_block27',synthetic=True);_,batch=fast.direct_update_fast(m,e,b,h,r,l,s,device,micro_questions=8);seen=set(batch['source_qids']);saved=ck.snapshot(m,e,b,h,device,s,1,binding,provenance,seen,'training');_,next_batch=fast.direct_update_fast(m,e,b,h,r,l,s,device,micro_questions=8);expected=ck.snapshot(m,e,b,h,device,s,2,binding,provenance,seen|set(next_batch['source_qids']),'training')['accepted_hashes'];ck.restore(saved,m,e,b,h,device,s,binding);_,again=fast.direct_update_fast(m,e,b,h,r,l,s,device,micro_questions=8);self.assertEqual(next_batch,again);self.assertEqual(expected,ck.snapshot(m,e,b,h,device,s,2,binding,provenance,seen|set(again['source_qids']),'training')['accepted_hashes'])


if __name__=='__main__':unittest.main()
