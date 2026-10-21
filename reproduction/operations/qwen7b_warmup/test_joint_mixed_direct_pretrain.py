"""Bounded synthetic worker math and full-state replay; no real assets/model."""
import copy,gc,hashlib,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from . import mixed_group_checkpoint as ck,probe_ablation_train as u
from .joint_mixed_direct_pretrain import direct_update,selected_views,warm_gradient_buckets,official_sources,OFFICIAL_REVISION
from .balanced_hard_g10_sampler import BalancedHardG10Sampler


class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.block=nn.Linear(3,3);self.group_head=nn.Linear(3,1);self.register_buffer('scratch',torch.tensor([0.]),persistent=False)
 def encode_trajectories(self,b):
  # Stochastic toy exercises all restored RNG families; real production has no
  # dropout. Detach demonstrates that only the native probe receives gradients.
  noise=(torch.rand(())+np.random.rand()+random.random())*.001
  self.scratch.add_(1)
  return dict(hidden=self.block(b['features'].detach())+noise,attention_mask=b['attention_mask'],position_ids=b['position_ids'])
 def group_logits(self,e):return self.group_head(e['hidden'].mean((1,2))).squeeze(-1)


class Reader:
 def __init__(self,groups):
  self.rows={t:(g['qid'],torch.arange(6,dtype=torch.float32).reshape(2,3)*.03+(j+int(g['qid'][1:]))*.02)for g in groups for j,t in enumerate(g['trajectory_ids'])};self.last=[]
 def get(self,ids,device):
  features=torch.stack([self.rows[t][1]for t in ids]).to(device).requires_grad_(True);self.last.append(features)
  return dict(features=features,attention_mask=torch.ones(len(ids),2,dtype=torch.bool,device=device),position_ids=torch.arange(2,device=device).expand(len(ids),-1),trajectory_ids=list(ids),qids=[self.rows[t][0]for t in ids],capture_layer=26)


def setup():
 random.seed(9);np.random.seed(9);torch.manual_seed(9)
 groups=[dict(qid='q%d'%i,question_hash='h%d'%i,trajectory_ids=['q%d-t%d'%(i,j)for j in range(10)],labels=([0]*10 if i==0 else[1]*10 if i==1 else[0,1]*5))for i in range(8)]
 model=Tiny();ema=copy.deepcopy(model).eval().requires_grad_(False);bo=torch.optim.Adam(model.block.parameters(),lr=5e-6);ho=torch.optim.Adam(model.group_head.parameters(),lr=5e-6);warm_gradient_buckets(model)
 sampler=BalancedHardG10Sampler(groups,seed=1729,batch_questions=4,steps_per_epoch=2);reader=Reader(groups);lookup={g['qid']:g for g in groups}
 return model,ema,bo,ho,sampler,reader,lookup


DEVICE=torch.device('cpu');BIND=dict(config_sha256='synthetic-config',matched_binding_sha256='synthetic-binding');PROV=dict(initialization_source='official_initial_Qwen2.5-7B-Instruct_block27',synthetic_fixture=True)

def snap(model,ema,bo,ho,sampler,cursor,seen=()):return ck.snapshot(model,ema,bo,ho,DEVICE,sampler,cursor,BIND,PROV,set(seen),'training')


class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)

 def test_explicit_external_official_inventory_full_asset_SHA_binding(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   root=Path(tmp);assets=root/'assets';assets.mkdir();payloads={'config.json':json.dumps(dict(num_hidden_layers=28,hidden_size=3584,num_attention_heads=28,num_key_value_heads=4,intermediate_size=18944)).encode(),'model.safetensors.index.json':json.dumps(dict(weight_map={'model.layers.27.toy':'shard.safetensors'})).encode(),'shard.safetensors':b'synthetic-not-real-weights'};files={};siblings=[]
   for name,raw in payloads.items():
    p=assets/name;p.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest();files[name]=dict(bytes=len(raw),sha256=digest);item=dict(rfilename=name)
    if name.endswith('safetensors'):item['lfs']=dict(sha256=digest)
    else:item['blobId']=hashlib.sha1(('blob %d\0'%len(raw)).encode()+raw).hexdigest()
    siblings.append(item)
   inv=root/'external-inventory.json';inv.write_text(json.dumps(dict(id='Qwen/Qwen2.5-7B-Instruct',sha=OFFICIAL_REVISION,siblings=siblings)));c=dict(model_dir=str(assets),model_files=files,official_inventory=dict(path=str(inv),bytes=inv.stat().st_size,sha256=u.sha(inv)));proofs=official_sources(c);self.assertEqual(set(proofs),set(files));self.assertFalse((assets/'official-inventory.json').exists())
   (assets/'shard.safetensors').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'mismatch'):official_sources(c)

 def test_microquestion_gradient_matches_full_question_objective_and_true_EM_order(self):
  model,ema,bo,ho,sampler,reader,lookup=setup();ref=copy.deepcopy(model);rs=copy.deepcopy(sampler);rng=u.rng_state(DEVICE);batch=rs.next_batch();by=selected_views(batch,lookup,4);full=[];expected_losses=[None]*16
  for q in batch['source_qids']:
   enc=ref.encode_trajectories(reader.get(lookup[q]['trajectory_ids'],DEVICE));idx=torch.tensor([v['source_indices']for _,v in by[q]]);z=ref.group_logits({k:v[idx]for k,v in enc.items()});target=torch.tensor([float(v['true_mixed'])for _,v in by[q]]);per=torch.nn.functional.binary_cross_entropy_with_logits(z,target,reduction='none');full.append(per.mean()/4)
   for(i,_),l in zip(by[q],per.detach().tolist()):expected_losses[i]=l
  sum(full).backward();torch.nn.utils.clip_grad_norm_(ref.parameters(),1.);rb=torch.optim.Adam(ref.block.parameters(),lr=5e-6);rh=torch.optim.Adam(ref.group_head.parameters(),lr=5e-6);rb.step();rh.step();u.restore_rng(rng,DEVICE)
  before=u.model_hash(model);m,actual=direct_update(model,ema,bo,ho,reader,lookup,sampler,DEVICE)
  self.assertEqual(actual,batch);self.assertAlmostEqual(m['loss'],sum(expected_losses)/16,places=7);self.assertNotEqual(before,u.model_hash(model));self.assertFalse(m['trajectory_loss_computed']);self.assertEqual(m['mixed_views'],m['nonmixed_views']);self.assertGreater(m['native_gradient_L2'],0);self.assertGreater(m['group_head_gradient_L2'],0)
  for p,q in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-8)
  self.assertTrue(all(t.grad is None for t in reader.last));self.assertEqual(sampler.difficulty['mixed'],{q:sum(expected_losses[i]for i,v in enumerate(batch['grouping']['groups'])if v['qid']==q)/4 for q,r in zip(batch['source_qids'],batch['roles'])if r=='mixed'})

 def test_CP0_two_Adam0_CP1_full_reload_and_exact_next_update(self):
  model,ema,bo,ho,sampler,reader,lookup=setup();seen=set();cp0=snap(model,ema,bo,ho,sampler,0);self.assertEqual(ck.validate(cp0),0);self.assertEqual(cp0['native_optimizer']['state'],{});self.assertEqual(cp0['head_optimizer']['state'],{})
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   root=Path(tmp);r0=ck.save(root,cp0,pin=True,owned_allocation_required=False);_,b1=direct_update(model,ema,bo,ho,reader,lookup,sampler,DEVICE);seen.update(b1['source_qids']);cp1=snap(model,ema,bo,ho,sampler,1,seen);r1=ck.save(root,cp1,pin=True,owned_allocation_required=False)
   self.assertTrue(r1['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']);self.assertEqual(u.read(root/'latest.json'),r1);self.assertTrue(Path(r0['path']).exists())
   expected_metrics,b2=direct_update(model,ema,bo,ho,reader,lookup,sampler,DEVICE);expected=snap(model,ema,bo,ho,sampler,2,seen|set(b2['source_qids']))['accepted_hashes'];loaded=torch.load(u.checked(r1),map_location='cpu',mmap=True,weights_only=False)
   cursor,restored_seen=ck.restore(loaded,model,ema,bo,ho,DEVICE,sampler,BIND);self.assertEqual(cursor,1);self.assertEqual(restored_seen,seen);actual_metrics,b3=direct_update(model,ema,bo,ho,reader,lookup,sampler,DEVICE);actual=snap(model,ema,bo,ho,sampler,2,seen|set(b3['source_qids']))['accepted_hashes']
   self.assertEqual(b2,b3);self.assertEqual(actual,expected);self.assertEqual(expected_metrics['loss'],actual_metrics['loss']);del loaded;gc.collect()

 def test_no_latest_published_before_new_complete_CPU_reload_and_clock_guard(self):
  model,ema,bo,ho,sampler,reader,lookup=setup();cp0=snap(model,ema,bo,ho,sampler,0)
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   root=Path(tmp);r0=ck.save(root,cp0,pin=True,owned_allocation_required=False);_,batch=direct_update(model,ema,bo,ho,reader,lookup,sampler,DEVICE);cp1=snap(model,ema,bo,ho,sampler,1,batch['source_qids'])
   with patch.object(ck.torch,'load',side_effect=RuntimeError('synthetic interrupted full reload')):
    with self.assertRaisesRegex(RuntimeError,'interrupted'):ck.save(root,cp1,pin=True,owned_allocation_required=False)
   self.assertEqual(u.read(root/'latest.json'),r0);self.assertFalse((root/'checkpoints/cursor-000001.pt').exists());self.assertTrue(list((root/'checkpoints').glob('.mixed-stage-*/failure.json')))
   broken=copy.deepcopy(cp1);broken['native_steps']=0
   with self.assertRaisesRegex(ValueError,'clocks'):ck.validate(broken)
   wrong=copy.deepcopy(batch);wrong['grouping']['groups'][0]['labels'][0]^=1
   with self.assertRaisesRegex(ValueError,'own positional'):selected_views(wrong,lookup,4)


if __name__=='__main__':unittest.main()
