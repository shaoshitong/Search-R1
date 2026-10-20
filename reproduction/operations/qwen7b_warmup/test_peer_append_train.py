"""Focused CPU toy tests of append counters, real next update, and eval state."""
import copy,gc,inspect,json,random,shlex,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import peer_append_train as r,deploy_peer_append_train as d,probe_ablation_train as u
from . import probe_g10_train as t

class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.linear=torch.nn.Linear(2,1,dtype=torch.float64);self.drop=torch.nn.Dropout(.15)
  self.register_buffer('native_clock',torch.zeros(1,dtype=torch.float64),persistent=False)
 def forward(self,data):
  self.native_clock.add_(1);return self.linear(self.drop(data['features'])).flatten()+self.native_clock.clone()*.0001
class Reader:
 def get(self,ids,device):return {'features':torch.tensor([[int(t.split(':')[0][1:])*.1+.2,int(t.split(':')[1][1:])*.07] for t in ids],dtype=torch.float64,device=device)}

def toy():
 random.seed(19);np.random.seed(19);torch.manual_seed(19);device=torch.device('cpu')
 model=Toy();ema=copy.deepcopy(model).requires_grad_(False);opt=torch.optim.Adam(model.parameters(),lr=5e-6)
 model({'features':torch.ones(2,2,dtype=torch.float64)}).sum().backward();opt.step()
 # Synthetic populated Adam counter only. This is never a production checkpoint.
 for state in opt.state.values():state['step'].fill_(10720)
 own=[dict(qid='q%d'%i,trajectory_ids=['q%d:t%d'%(i,j) for j in range(10)],labels=[int(j<k) for j in range(10)]) for i,k in enumerate((1,5,9))]
 archive=[dict(rank=i,rng=copy.deepcopy(u.rng_state(device))) for i in range(4)]
 binding=dict(config_sha256='toy',matched_binding_sha256='toy',plan_sha256='toy')
 saved=copy.deepcopy(r.full_blob(model,ema,opt,device,256,binding,archive))
 c=dict(recipe={'loss_kwargs':dict(trajectory_weight=1.,three_event_weight=1.)},grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),micro_trajectories=3,max_virtual_views=128,max_inter_view_pairs=4096)
 return model,ema,opt,own,saved,c,binding

class AppendTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_first_append_double_replay_preserves_real_six_components(self):
  m,e,o,own,s,c,b=toy();device=torch.device('cpu');lookup={g['qid']:g for g in own}
  r.train_update(m,e,o,Reader(),lookup,list(lookup),c,device,0);expected=u.state_hashes(m,e,o,device)
  u.restore_blob(s,m,e,o,device);self.assertEqual(u.state_hashes(m,e,o,device),s['accepted_hashes'])
  r.train_update(m,e,o,Reader(),lookup,list(lookup),c,device,0)
  self.assertEqual(u.state_hashes(m,e,o,device),expected);self.assertEqual(u.steps(o),10721)
  self.assertEqual(s['cursor'],256);self.assertEqual(s['steps'],10720)
 def test_fullsave_reload_next_update_and_truthful_current_topology(self):
  m,e,o,own,s,c,b=toy();device=torch.device('cpu');lookup={g['qid']:g for g in own}
  r.train_update(m,e,o,Reader(),lookup,list(lookup),c,device,0)
  v=copy.deepcopy(r.full_blob(m,e,o,device,257,b,s['source_all_rank_states']))
  self.assertEqual(v['source_world_size'],1);self.assertEqual(v['current_world_size'],1);self.assertFalse(v['hardware_topology_changed']);self.assertEqual(v['original_E1_archive_world_size'],4)
  with tempfile.TemporaryDirectory() as directory:
   receipt=u.save_checkpoint(Path(directory),v)
   r.train_update(m,e,o,Reader(),lookup,list(lookup),c,device,0);expected=u.state_hashes(m,e,o,device)
   loaded=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
   u.restore_blob(loaded,m,e,o,device);r.train_update(m,e,o,Reader(),lookup,list(lookup),c,device,0)
   self.assertEqual(u.state_hashes(m,e,o,device),expected);self.assertEqual(u.steps(o),10722)
   self.assertEqual([u.rng_hash(x['rng']) for x in loaded['source_all_rank_states']],[u.rng_hash(x['rng']) for x in s['source_all_rank_states']])
   del loaded;gc.collect()  # Release NFS mmap before removing its toy directory.
 def test_no_rank_losses_or_fake_mask(self):
  m,e,o,own,s,c,b=toy();c['recipe']['loss_kwargs']['inter_rank_weight']=1.
  with self.assertRaisesRegex(ValueError,'No failed ranking'):r.train_update(m,e,o,Reader(),{},[],c,torch.device('cpu'),0)
  self.assertNotIn('input_pair_mask',inspect.getsource(r.train_update))
 def test_config_preserves_loss_grouping_evaluations_and_full_source(self):
  old=dict(source_adam_steps=10464,update_budget=256,global_questions=32,fit_questions=8192,fit_real_trajectories=81920,recipe=dict(id='old',loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),optimizer_reset=False,source_checkpoint={'old':'E1'},source_config={'old':'E1config'},domain_scope={'question_counts':{'nq':8192}},evaluations=['unchanged'],save_every=16)
  sel={k:{'path':k,'sha256':k} for k in ('source_checkpoint','source_config','source_complete','source_plan','new_bank','new_bank_independent_audit','append_plan')};sel['loss_kwargs']=old['recipe']['loss_kwargs'];before=copy.deepcopy(old)
  c=r.configured(old,dict(commit='new',files={'entry':{'sha256':'h'}}),sel)
  self.assertEqual(old,before);self.assertEqual(c['evaluations'],old['evaluations']);self.assertEqual(c['grouping'],old['grouping']);self.assertEqual(c['recipe']['loss_kwargs'],old['recipe']['loss_kwargs']);self.assertEqual(c['source_cursor'],256);self.assertEqual(c['update_budget']-256,682);self.assertEqual(c['source_adam_steps'],10720);self.assertEqual(c['evaluation_cumulative_cursors'],[512,768,938]);self.assertEqual(c['original_E1_checkpoint'],old['source_checkpoint'])
 def test_cumulative_plan_loop_never_restarts_history(self):
  source=inspect.getsource(r.run)
  self.assertIn("plan['batches'][256]",source);self.assertIn('cursor=256',source);self.assertIn('cursor=257',source);self.assertIn('while cursor<938',source)
  self.assertNotIn('t.run(',source);self.assertNotIn('10464+cursor',source)
  self.assertIn('validate_source_checkpoint_metadata',source);self.assertIn('source[\'accepted_hashes\']',source)
  self.assertLess(source.index('child.wait(timeout=180)'),source.index('while cursor<938'))
 def test_evaluation_checkpoint_first_state_restored_and_idempotent(self):
  m,e,o,own,s,c,b=toy();device=torch.device('cpu')
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder);cp=base/'cp.pt';cp.write_bytes(b'toy checkpoint');ref=r.proof(cp);ref['cursor']=512;r.atomic(base/'latest.json',ref)
   features=base/'features.json';features.write_text('[]');ev=[(dict(name='fixed',features=r.proof(features)),[])]
   class ER:
    def __init__(self,*_):pass
    def close(self):pass
   def evaluate(ema,reader,groups,dev,destination):
    random.random();np.random.rand();torch.rand(2);ema.native_clock.add_(9);Path(destination).write_bytes(b'own predictions');return {'toy':1}
   from . import replay_feature_reader
   with patch.object(r,'TARGET',base),patch.object(r,'checked',side_effect=lambda v:Path(v['path'])),patch.object(replay_feature_reader,'ReplayFeatureReader',ER),patch.object(u,'evaluate',side_effect=evaluate) as fn:
    before=u.state_hashes(m,e,o,device);first=r.evaluate_at(512,e,m,o,device,ev,b);self.assertEqual(u.state_hashes(m,e,o,device),before)
    self.assertEqual(first,r.evaluate_at(512,e,m,o,device,ev,b));self.assertEqual(fn.call_count,1)
    with self.assertRaisesRegex(ValueError,'checkpoint must precede'):r.evaluate_at(768,e,m,o,device,ev,b)
 def test_remote_prepare_submission_isolation_and_bounded_code(self):
  script=r.render(r.TARGET/'code-synthetic','a'*64);subprocess.run(['bash','-n'],input=script,text=True,check=True)
  self.assertIn('--qos=normal',script);self.assertIn('--gres=gpu:1',script);self.assertNotIn('dependency',script);self.assertNotIn('@@',script)
  values=dict(base=str(r.TARGET),target=str(r.TARGET),train=str(r.CONTROL),pin=r.D8,callback_commit=r.CALLBACK)
  for action in ('init','chunk','extract','prepare','submit','status'):
   source=d.remote_source(values,action);compile(source,action,'exec');self.assertLess(len(('python3 -c '+shlex.quote(source)).encode()),16000)
  self.assertIn('Unknown previous submission outcome',d.SUBMIT);self.assertIn('q7b-peer-append682',d.SUBMIT);self.assertIn('append-only.lock',d.SUBMIT)
 def test_independent_child_uses_own_cp257_and_base(self):
  self.assertIn("peer_append_train','--action','audit-first'",inspect.getsource(r.run))
  self.assertIn('cursor-000257.pt',inspect.getsource(r.audit_first));self.assertIn('10721',inspect.getsource(r.audit_first))
if __name__=='__main__':unittest.main()
