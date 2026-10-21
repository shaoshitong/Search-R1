"""Focused CPU qualification of new two-segment gather and exact CP0 recovery."""
import copy,tempfile,unittest
from pathlib import Path
import torch
from . import count53312_train as m,count53312_incremental_cache as cache,direct_hidden_count as d,probe_ablation_train as u,direct_count_large_small_checkpoint as small

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 def source(self):
  h,e,o=d.initialize_head(device=torch.device('cpu'));o.param_groups[0]['lr']=1e-4;rng=u.rng_state(torch.device('cpu'));values=m.hashes(h,e,o,torch.device('cpu'))
  return dict(kind='LARGE_FROZEN_E1_HEAD_ONLY_FULL_STATE',head=copy.deepcopy(d.new_head_state(h,e,o)),rng=rng,cursor=0,epoch=0,config={'sha256':'a'*64},cache={'sha256':'b'*64},source_all_four_rank_states=[dict(rank=i,model_sha256='c'*64,ema_sha256='d'*64,optimizer_sha256='e'*64,steps=10464,rng=copy.deepcopy(rng))for i in range(4)],frozen_legacy_hashes={'raw':'c'*64,'buffers':'f'*64},initial_head_hashes=d.new_head_hashes(h,e,o),accepted_hashes=values,data_cursor={'cursor':0,'orders_sha256':'1'*64})
 def test_two_segment_gather_exact_order_mask_dtype_and_no_full_cuda(self):
  old={'hidden':torch.arange(20*3*8).reshape(20,3,8).to(torch.bfloat16),'mask':torch.ones(20,3,dtype=torch.bool),'logits':torch.arange(20).float()};new={k:v.clone()for k,v in old.items()};new['hidden'].add_(1000);new['mask'][2]=False;new['logits'].add_(100)
  loc={'a':(0,0),'b':(0,10),'c':(1,0),'d':(1,10)};h,mask,z=cache.gather([old,new],loc,['c','b','a'],torch.device('cpu'))
  for i,(seg,start)in enumerate([(1,0),(0,10),(0,0)]):
   for actual,key in ((h,'hidden'),(mask,'mask'),(z,'logits')):self.assertTrue(torch.equal(actual[10*i:10*i+10],[old,new][seg][key][start:start+10]))
  self.assertEqual(h.dtype,torch.bfloat16);self.assertEqual(mask.dtype,torch.bool);self.assertFalse(m.memory_plan()['full_bank_CUDA_resident']);self.assertEqual(m.memory_plan()['total_hidden_bytes'],83768115200)
  with self.assertRaises(ValueError):cache.gather([old,new],loc,['a','a'],torch.device('cpu'))
 def test_actual_small_CP0_double_update_durable_recovery(self):
  source=self.source();device=torch.device('cpu');h,e,o=d.initialize_head(seed=999,device=device);d.restore_new_head(source['head'],h,e,o,device);u.restore_rng(source['rng'],device)
  with tempfile.TemporaryDirectory(prefix='count53312-synthetic-',dir='/tmp')as tmp:
   root=Path(tmp);p=root/'actual-source-CP0.pt';torch.save(source,p);loaded=torch.load(p,map_location='cpu',mmap=True,weights_only=False);small.geometry(loaded);self.assertEqual(small.component_hashes(loaded),source['accepted_hashes']);c={'config_reference':{'sha256':'1'*64},'orders':{'sha256':'2'*64},'baseline_CP0':m.proof(p),'baseline_CP0_four_hashes':source['accepted_hashes']};cache_ref={'sha256':'3'*64}
   torch.manual_seed(91);hidden=torch.randn(10,2,3584);mask=torch.ones(10,2,dtype=torch.bool);mask[-1]=False;labels=[0,1,0,1,0,1,0,1,0,1];group={'qid':'q','trajectory_ids':['t'+str(i)for i in range(10)],'labels':labels};indices=[[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]];views=[dict(qid='q',trajectory_ids=[group['trajectory_ids'][i]for i in ids],labels=[labels[i]for i in ids],weight_numerator=1,weight_denominator=4)for ids in indices];z=torch.randn(1,10)
   def step():
    o.zero_grad(set_to_none=True);emb=h.encode(hidden,mask).reshape(1,10,128);loss=d.count_loss(z,emb,[group],views,h);loss['count_NLL'].backward();o.step()
    with torch.no_grad():
     for dst,src in zip(e.parameters(),h.parameters()):dst.mul_(.99).add_(src,alpha=.01)
   # Tensor creation advances RNG only before the fixture snapshot, just as
   # production resets actual CP0 RNG after native-cache construction.
   u.restore_rng(source['rng'],device);self.assertEqual(m.hashes(h,e,o,device),source['accepted_hashes']);zero=m.state_blob(h,e,o,device,0,c,loaded,cache_ref);m.save(root,zero,c,loaded,cache_ref);step();expected=m.hashes(h,e,o,device);m.restore(zero,h,e,o,device,c,loaded,cache_ref);step();self.assertEqual(m.hashes(h,e,o,device),expected)
   one=m.state_blob(h,e,o,device,1,c,loaded,cache_ref);ref=m.save(root,one,c,loaded,cache_ref);step();next_expected=m.hashes(h,e,o,device);saved=torch.load(ref['path'],map_location='cpu',mmap=True,weights_only=False);m.restore(saved,h,e,o,device,c,loaded,cache_ref);step();self.assertEqual(m.hashes(h,e,o,device),next_expected);self.assertEqual(d.optimizer_steps(o),2);self.assertEqual(small.archive_signature(saved),small.archive_signature(loaded))
   bad=copy.deepcopy(one);bad['data_cursor']['cursor']=2
   with self.assertRaises(ValueError):m.verify_state(bad,c,loaded,cache_ref)
 def test_budget_entry_and_no_rollback_guards(self):
  self.assertEqual(m.FIT,53312);self.assertEqual(m.TOTAL,1666*20);entry=m.entry('/remote/source','a'*64);self.assertIn('--mem=128G',entry);self.assertIn('--time=03:00:00',entry);self.assertNotIn('dependency',entry);self.assertNotIn('emergency',entry)
  self.assertEqual(m.resume_cursor({'cursor':7},{'cursor':7},{'cursor':7,'phase':'accepted_update_complete'}),7)
  with self.assertRaises(ValueError):m.resume_cursor({'cursor':6},{'cursor':7})
  with self.assertRaises(ValueError):m.resume_cursor({'cursor':6},{'cursor':6},{'cursor':7,'phase':'started'})

if __name__=='__main__':unittest.main()
