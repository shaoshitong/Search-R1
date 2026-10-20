"""CPU synthetic qualification only; no official model, feature or GPU claim."""
import copy,gc,tempfile,unittest
from pathlib import Path
import torch
from . import probe_ablation_train as u
from .direct_hidden_count import DirectCountHead,initialize_head,count_loss,new_head_state,restore_new_head,new_head_hashes,optimizer_steps,native_hidden_forward
from .direct_hidden_count_update import bounded_update
from .g10_grouping import build_training_views

class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.block=torch.nn.Linear(3,6,dtype=torch.float64);self.output=torch.nn.Linear(30,1,dtype=torch.float64);self.drop=torch.nn.Dropout(.2)
  self.register_buffer('native_clock',torch.zeros(1,dtype=torch.float64),persistent=False)
def forward(legacy,data,head):
 legacy.native_clock.add_(1);h=torch.tanh(legacy.block(legacy.drop(data['features'].detach())))+legacy.native_clock.clone()*.00001
 z=legacy.output(h.masked_fill(~data['attention_mask'][...,None],0).flatten(1)).squeeze(-1)
 return z,head.encode(h[:,2:],data['attention_mask'][:,2:])
class Reader:
 def get(self,ids,device):
  values=[];masks=[]
  for tid in ids:
   q,t=map(int,tid[1:].split(':t'));values.append((torch.arange(15,dtype=torch.float64).reshape(5,3)*.01+q*.1+t*.02).tolist());masks.append([True]*2+[j<t%4 for j in range(3)])
  return dict(features=torch.tensor(values,dtype=torch.float64,device=device),attention_mask=torch.tensor(masks,device=device))
def setup():
 torch.manual_seed(22);device=torch.device('cpu');legacy=Toy();ema=copy.deepcopy(legacy).requires_grad_(False);opt=torch.optim.Adam(legacy.parameters(),lr=5e-6)
 legacy.output(legacy.block(torch.ones(2,5,3,dtype=torch.float64)).flatten(1)).sum().backward();opt.step()
 for s in opt.state.values():s['step'].fill_(10464) # Synthetic populated E1-shaped counter only.
 archive=[dict(rank=i,rng=copy.deepcopy(u.rng_state(device))) for i in range(4)]
 saved=copy.deepcopy(u.checkpoint_blob(legacy,ema,opt,device,0,dict(config_sha256='toy',matched_binding_sha256='toy'),archive))
 head,he,ho=initialize_head(dimension=6,embedding_dimension=4);head.double();he.double()
 own=[dict(qid='q%d'%i,trajectory_ids=['q%d:t%d'%(i,j) for j in range(10)],labels=[int(j<k) for j in range(10)]) for i,k in enumerate((1,5,9))]
 return legacy,ema,opt,head,he,ho,own,saved,copy.deepcopy(new_head_state(head,he,ho))
def update(values):
 m,e,o,h,he,ho,own,*_=values
 return bounded_update(m,e,o,h,he,ho,Reader(),own,torch.device('cpu'),grouping_kwargs=dict(mode='random',views_per_question=4,seed=1729,epoch=0),micro_trajectories=3,forward_adapter=forward)
def restore(values):
 m,e,o,h,he,ho,own,old,new=values;u.restore_blob(old,m,e,o,torch.device('cpu'));restore_new_head(new,h,he,ho,torch.device('cpu'))
def hashes(values):
 m,e,o,h,he,ho,*_=values;return dict(legacy=u.state_hashes(m,e,o,torch.device('cpu')),new=new_head_hashes(h,he,ho))
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_G5_permutation_invariant_symmetric_count_math(self):
  torch.manual_seed(9);head=DirectCountHead(6,4).double();x=torch.randn(7,5,4,dtype=torch.float64);a=head.probabilities(x);b=head.probabilities(x[:,[4,1,0,3,2]])
  torch.testing.assert_close(a['count_probabilities'],b['count_probabilities'],rtol=1e-12,atol=1e-12)
  torch.testing.assert_close(a['mixed_probability'],a['count_probabilities'][:,1:5].sum(-1),rtol=0,atol=0)
  torch.testing.assert_close(a['count_probabilities'].sum(-1),torch.ones(7,dtype=torch.float64),rtol=1e-12,atol=1e-12)
 def test_short_padding_and_zero_prefix_cannot_change_embedding(self):
  head=DirectCountHead(6,4).double();x=torch.randn(3,20,6,dtype=torch.float64);mask=torch.zeros(3,20,dtype=torch.bool);mask[0,:1]=True;mask[1,:7]=True
  changed=x.clone();changed[~mask]=float('nan');a=head.encode(x,mask);b=head.encode(changed,mask)
  torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertTrue(bool(torch.isfinite(b).all()))
 def test_old_parameters_Adam_and_RNG_exact_when_new_head_initialized(self):
  values=setup();m,e,o,h,he,ho,own,saved,new=values
  self.assertEqual(u.state_hashes(m,e,o,torch.device('cpu')),saved['accepted_hashes']);self.assertEqual(u.steps(o),10464);self.assertEqual(optimizer_steps(ho),0)
  self.assertEqual(new_head_hashes(h,he,ho)['raw'],new_head_hashes(h,he,ho)['ema'])
 def test_count_uses_hidden_not_old_p_logits_and_question_mass1(self):
  values=setup();*_,own,old,new=values;h=values[3];views=build_training_views(own,mode='random',views_per_question=4,seed=1729)['groups'];z=torch.randn(3,10,dtype=torch.float64,requires_grad=True);emb=torch.randn(3,10,4,dtype=torch.float64,requires_grad=True)
  a=count_loss(z,emb,own,views,h);b=count_loss(z+9,emb,own,views,h)
  torch.testing.assert_close(a['mixed_probability'],b['mixed_probability'],rtol=0,atol=0)
  self.assertNotEqual(float(a['trajectory_loss'].detach()),float(b['trajectory_loss'].detach()));self.assertEqual(float(a['view_weights'].sum()),3.)
  for v,k in zip(views,a['true_count'].tolist()):self.assertEqual(sum(v['labels']),k)
 def test_actual_joint_double_next_update_old_and_new_full_states(self):
  values=setup();out=update(values);expected=hashes(values);restore(values);update(values)
  self.assertEqual(hashes(values),expected);self.assertEqual(out['legacy_Adam_steps'],10465);self.assertEqual(out['new_head_Adam_steps'],1);self.assertTrue(out['joint_micro_replay_exact'])
 def test_explicit_native_adapter_preserves_legacy_output_and_detaches_input(self):
  class Block(torch.nn.Linear):
   def forward(self,x,**kwargs):return super().forward(x)
  class Legacy(torch.nn.Module):
   def __init__(self):
    super().__init__();self.qmax=2;self.prefix=3;self.cache_position=False;self.block=Block(6,6,dtype=torch.float64);self.output=torch.nn.Linear(30,1,dtype=torch.float64)
   def rotary(self,x,positions):return x,positions
   def forward(self,batch):
    h=self.block(batch['features'].detach());return self.output(h.masked_fill(~batch['attention_mask'][...,None],0).flatten(1)).squeeze(-1).float()
  legacy=Legacy();head=DirectCountHead(6,4).double();x=torch.randn(2,5,6,dtype=torch.float64,requires_grad=True);mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool);batch=dict(features=x,attention_mask=mask,position_ids=torch.arange(5).repeat(2,1))
  z,h=native_hidden_forward(legacy,batch,head);torch.testing.assert_close(z,legacy(batch),rtol=0,atol=0)
  (z.sum()+h.sum()).backward();self.assertIsNone(x.grad);self.assertIsNotNone(legacy.block.weight.grad);self.assertIsNotNone(head.attention.weight.grad)
 def test_changed_recomputed_hidden_fails_before_either_optimizer_write(self):
  class Changed(Reader):
   def __init__(self):self.calls=0
   def get(self,ids,device):
    self.calls+=1;v=super().get(ids,device)
    if self.calls>10:v['features'].add_(1)
    return v
  values=setup();m,e,o,h,he,ho,own,*_=values;before=u.model_hash(m)
  with self.assertRaisesRegex(ValueError,'Two-pass joint'):
   bounded_update(m,e,o,h,he,ho,Changed(),own,torch.device('cpu'),grouping_kwargs=dict(mode='random',views_per_question=4,seed=1729,epoch=0),micro_trajectories=3,forward_adapter=forward)
  self.assertEqual(u.model_hash(m),before);self.assertEqual(u.steps(o),10464);self.assertEqual(optimizer_steps(ho),0)
 def test_bounded_hidden_VJP_matches_joined_reference_gradients(self):
  values=setup();m,e,o,h,he,ho,own,*_=values;update(values);actual=[p.grad.clone() for p in list(m.parameters())+list(h.parameters())];expected=hashes(values);restore(values)
  o.zero_grad(set_to_none=True);ho.zero_grad(set_to_none=True);m.train();h.train();z=[];emb=[];tids=[t for g in own for t in g['trajectory_ids']]
  for start in range(0,len(tids),3):
   a,b=forward(m,Reader().get(tids[start:start+3],torch.device('cpu')),h);z.append(a);emb.append(b)
  views=build_training_views(own,mode='random',views_per_question=4,seed=1729,epoch=0)['groups'];loss=count_loss(torch.cat(z).reshape(3,10),torch.cat(emb).reshape(3,10,4),own,views,h);loss['loss'].backward()
  for a,p in zip(actual,list(m.parameters())+list(h.parameters())):torch.testing.assert_close(a,p.grad,rtol=1e-11,atol=1e-12)
  o.step();ho.step()
  with torch.no_grad():
   for target,value in zip(e.parameters(),m.parameters()):target.mul_(.99).add_(value,alpha=.01)
   for target,value in zip(he.parameters(),h.parameters()):target.mul_(.99).add_(value,alpha=.01)
  # Floating reduction order can differ; values/gradients are checked, not a
  # monolithic-bitexact claim. Counter/native/RNG clocks remain exact.
  self.assertEqual(u.steps(o),10465);self.assertEqual(optimizer_steps(ho),1)
 def test_new_head_full_checkpoint_restore_continues_instead_of_reset(self):
  values=setup();update(values);m,e,o,h,he,ho,own,*_=values;device=torch.device('cpu')
  state=copy.deepcopy(dict(legacy=u.checkpoint_blob(m,e,o,device,1,dict(config_sha256='toy',matched_binding_sha256='toy'),values[7]['source_all_rank_states']),group_head=new_head_state(h,he,ho)))
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'joint.pt';torch.save(state,p);update(values);expected=hashes(values);saved=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
   u.restore_blob(saved['legacy'],m,e,o,device);restore_new_head(saved['group_head'],h,he,ho,device);update(values)
   self.assertEqual(hashes(values),expected);self.assertEqual(u.steps(o),10466);self.assertEqual(optimizer_steps(ho),2);del saved;gc.collect()
 def test_prepare_is_E1_all938_not_CP256_and_has_no_submit_action(self):
  from . import direct_hidden_count_prepare as p,deploy_direct_hidden_count_prepare as d
  old=dict(source_checkpoint={'sha256':'02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'},source_adam_steps=10464,update_budget=256,source_world_size=4,current_world_size=1,grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),domain_scope={'question_counts':{'nq':8192}},evaluations=['fixed'])
  selection={k:{'sha256':k} for k in ('new_bank','append_plan','source_plan','source_config')};pub=dict(commit='source',files={});c=p.configured(old,selection,pub,123)
  self.assertEqual(c['source_checkpoint']['sha256'],old['source_checkpoint']['sha256']);self.assertEqual(c['source_cursor'],0);self.assertEqual(c['source_adam_steps'],10464);self.assertEqual(c['update_budget'],938);self.assertEqual(c['new_head']['new_Adam_steps'],0);self.assertFalse(c['legacy_optimizer_reset']);self.assertEqual(c['evaluation_cumulative_cursors'],[256,512,768,938]);self.assertEqual(c['evaluations'],old['evaluations'])
  for action in ('init','chunk','extract','prepare','status'):compile(d.remote_source(dict(base=str(p.TARGET),target=str(p.TARGET),pin=p.a.D8,callback_commit=p.a.CALLBACK,reference=d.REFERENCE),action),action,'exec')
  with self.assertRaises(KeyError):d.remote_source({},'submit')
if __name__=='__main__':unittest.main()
