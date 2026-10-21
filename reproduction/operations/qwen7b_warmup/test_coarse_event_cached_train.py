"""Pure CPU synthetic-state qualification; never uses real tensors or scoring."""
import copy,random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from . import coarse_event_cached_train as m,direct_hidden_count as d,probe_ablation_train as u,prior_logit_event_loss as loss,direct_count_large_small_checkpoint as small

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 def baseline(self):
  head,ema,opt=d.initialize_head(device=torch.device('cpu'));opt.param_groups[0]['lr']=1e-4;rng=u.rng_state(torch.device('cpu'));initial=m.hashes(head,ema,opt,torch.device('cpu'))
  archive=[dict(rank=i,model_sha256='a'*64,ema_sha256='b'*64,optimizer_sha256='c'*64,steps=10464,rng=copy.deepcopy(rng))for i in range(4)]
  return dict(kind='LARGE_FROZEN_E1_HEAD_ONLY_FULL_STATE',head=copy.deepcopy(d.new_head_state(head,ema,opt)),rng=rng,cursor=0,epoch=0,config=dict(path='/remote/baseline-config',bytes=1,sha256='d'*64),cache=dict(path='/remote/cache-complete',bytes=1,sha256='e'*64),source_all_four_rank_states=archive,frozen_legacy_hashes=dict(raw='a'*64,buffers='f'*64),initial_head_hashes=d.new_head_hashes(head,ema,opt),accepted_hashes=initial,data_cursor=dict(cursor=0,orders_sha256='1'*64))
 def config(self,source,mode,cp0):
  return dict(mode=mode,config_reference=dict(path='/remote/'+mode+'-config',bytes=1,sha256=('2'if mode=='binary'else'3')*64),cache=source['cache'],baseline_CP0=cp0,baseline_CP0_four_hashes=source['accepted_hashes'],orders=dict(sha256='1'*64),frozen_train_prior=loss.FULL938_PRIVATE)
 def test_waiting_is_explicit_no_fake_input_acceptance(self):
  v=m.waiting_inputs(None);self.assertEqual(v['status'],'WAITING_FOR_VERIFIED_BASELINE_CACHE_CP0');self.assertFalse(v['prepared_inputs_passed']);self.assertFalse(v['GPU_submitted'])
  self.assertNotEqual(m.target('binary'),m.target('three_event'));entry=m.render_entry('/remote/base','/remote/code','binary','x'*64)
  self.assertIn('--time=02:00:00',entry);self.assertIn('--mem=128G',entry);self.assertNotIn('dependency',entry);self.assertNotIn('emergency',entry)
 def test_actual_small_CP0_restore_replay_save_reload_next_update(self):
  source=self.baseline();device=torch.device('cpu');torch.manual_seed(31);hidden=torch.randn(10,2,3584);mask=torch.ones(10,2,dtype=torch.bool);mask[-1]=False;frozen=torch.randn(10);idx=torch.tensor([[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]);own=torch.tensor([0,1,0,1,0,1,0,1,0,1]);labels=own[idx]
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);path=root/'actual-small-CP0.pt';torch.save(source,path);small.zip_guard(path);loaded=torch.load(path,map_location='cpu',mmap=True,weights_only=False);small.geometry(loaded);self.assertEqual(small.component_hashes(loaded),source['accepted_hashes']);cp0=m.proof(path)
   for mode in m.MODES:
    base=root/mode;base.mkdir();c=self.config(source,mode,cp0);head,ema,opt=d.initialize_head(seed=999,device=device);d.restore_new_head(loaded['head'],head,ema,opt,device);u.restore_rng(loaded['rng'],device)
    self.assertEqual(m.hashes(head,ema,opt,device),source['accepted_hashes']);self.assertEqual(opt.param_groups[0]['lr'],1e-4)
    pi=loss.coarse_prior_from_counts([5,1,2,2,1,3],mode,dtype=torch.float32)
    def step():
     emb=head.encode(hidden,mask);return m.objective_update(head,ema,opt,emb[idx],frozen[idx],labels,pi,mode,1)
    zero=m.checkpoint_blob(head,ema,opt,device,0,0,c,source);m.save_checkpoint(base,zero,c,source);step();expected=m.hashes(head,ema,opt,device);random.random();np.random.rand();torch.rand(3)
    m.restore_blob(zero,head,ema,opt,device,c,source);step();self.assertEqual(m.hashes(head,ema,opt,device),expected)
    one=m.checkpoint_blob(head,ema,opt,device,1,0,c,source);ref=m.save_checkpoint(base,one,c,source);step();next_expected=m.hashes(head,ema,opt,device)
    saved=torch.load(ref['path'],map_location='cpu',mmap=True,weights_only=False);m.restore_blob(saved,head,ema,opt,device,c,source);step();self.assertEqual(m.hashes(head,ema,opt,device),next_expected)
    self.assertEqual(d.optimizer_steps(opt),2);self.assertEqual(small.archive_signature(source),small.archive_signature(one))
    bad=copy.deepcopy(one);bad['mode']='three_event'if mode=='binary'else'binary'
    with self.assertRaises(ValueError):m.verify_blob(bad,c,source)
 def test_guard_prior_data_cursor_and_highest_accepted_state(self):
  source=self.baseline();head,ema,opt=d.initialize_head(device=torch.device('cpu'));opt.param_groups[0]['lr']=1e-4;d.restore_new_head(source['head'],head,ema,opt,torch.device('cpu'));u.restore_rng(source['rng'],torch.device('cpu'));c=self.config(source,'binary',dict(path='/remote/CP0',bytes=1,sha256='4'*64));state=m.checkpoint_blob(head,ema,opt,torch.device('cpu'),0,0,c,source)
  for key,value in [('data_cursor',dict(cursor=1,orders_sha256='1'*64)),('frozen_train_prior',dict(path='/wrong'))]:
   bad=copy.deepcopy(state);bad[key]=value
   with self.assertRaises(ValueError):m.verify_blob(bad,c,source)
  self.assertEqual(m.durable_resume_cursor(dict(cursor=7),dict(cursor=7),dict(cursor=7,phase='accepted_update_complete')),7)
  with self.assertRaises(ValueError):m.durable_resume_cursor(dict(cursor=6),dict(cursor=7),dict(cursor=7,phase='accepted_update_complete'))
  with self.assertRaises(ValueError):m.durable_resume_cursor(dict(cursor=6),dict(cursor=6),dict(cursor=7,phase='started'))
  bad=copy.deepcopy(state);bad['legacy_weights']=torch.zeros(1)
  with self.assertRaises(ValueError):m.verify_blob(bad,c,source)
  bad=copy.deepcopy(state);bad['head']['optimizer']['param_groups'][0]['lr']=5e-6
  with self.assertRaises(ValueError):m.verify_blob(bad,c,source)

if __name__=='__main__':unittest.main()
