"""Narrow independent inter value/gradient, accounting, and isolated entry tests."""
import copy,inspect,shlex,subprocess,unittest
from fractions import Fraction
from unittest.mock import patch
import torch
import torch.nn.functional as F
from . import peer_rank_matched as r,deploy_peer_rank_matched as d
from .g10_grouping import build_training_views
from .g10_question_loss import g10_question_loss
from .probe_ablation_loss import group_event_scores

def groups(counts=(5,1,0)):
 return [dict(qid='q'+str(i),trajectory_ids=['q%d:t%d'%(i,j) for j in range(10)],labels=[int(j<k) for j in range(10)]) for i,k in enumerate(counts)]

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def original(self):
  return dict(source_checkpoint={'sha256':r.w.SOURCE_SHA},source_adam_steps=10464,update_budget=256,global_questions=32,fit_questions=8192,fit_real_trajectories=81920,plan={'sha256':r.PLAN_SHA},grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),recipe=dict(id='original',loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),optimizer_reset=False,max_virtual_views=128,max_inter_view_pairs=4096,source_commit='old',code_hashes={'old':'old'},source_all_four_rank_RNG_preserved=True,evaluations=[{'unchanged':'fixed-HF'}])
 def test_exact_two_recipe_leaves_and_no_reset(self):
  for mode in r.RECIPES:
   old=self.original();before=copy.deepcopy(old);pub=dict(commit='new',files={'entry.py':{'sha256':'x'}});sig={'path':'real-signatures','sha256':'x'} if mode=='bothrank1' else None
   new=r.matched_config(old,pub,mode,sig);self.assertEqual(old,before)
   self.assertEqual(new['recipe']['loss_kwargs'],dict(trajectory_weight=1.,three_event_weight=1.,**r.RECIPES[mode]))
   for key in ('plan','source_checkpoint','evaluations','grouping','source_all_four_rank_RNG_preserved'):self.assertEqual(new[key],old[key])
   self.assertEqual('within_pair_inputs' in new,mode=='bothrank1');self.assertFalse(new['optimizer_reset'])
  bad=self.original();bad['plan']['sha256']='wrong'
  with self.assertRaises(ValueError):r.matched_config(bad,pub,'inter1')
  with self.assertRaises(ValueError):r.matched_config(self.original(),pub,'inter1',{'fake':'mask'})
  with self.assertRaises(ValueError):r.matched_config(self.original(),pub,'bothrank1')
 def test_inter_value_and_gradient_matches_independent_ordered_q_loop(self):
  own=groups((5,1,9));views=build_training_views(own,mode='per-question-balanced',views_per_question=4,seed=29)['groups'];torch.manual_seed(3)
  z=torch.randn(3,10,dtype=torch.float64,requires_grad=True)
  actual=g10_question_loss(z,own,views,trajectory_weight=0.,inter_rank_weight=1.)
  by={g['qid']:i for i,g in enumerate(own)};scores=[]
  for v in views:
   indices=[own[by[v['qid']]]['trajectory_ids'].index(t) for t in v['trajectory_ids']]
   score=group_event_scores(z[by[v['qid']],indices][None])['mixed_event_logit'][0]
   scores.append(score)
  terms=[];mass=Fraction(0);eligible=0
  for i,p in enumerate(views):
   if not p['true_mixed']:continue
   for j,n in enumerate(views):
    if n['true_mixed'] or p['qid']==n['qid']:continue
    weight=Fraction(p['weight_numerator'],p['weight_denominator'])*Fraction(n['weight_numerator'],n['weight_denominator'])
    mass+=weight;eligible+=1;terms.append(F.softplus(1-scores[i]+scores[j])*float(weight))
  expected=sum(terms,z.sum()*0)/(3*2)
  torch.testing.assert_close(actual['loss'],expected,rtol=1e-12,atol=1e-12)
  ga=torch.autograd.grad(actual['loss'],z,retain_graph=True)[0];gb=torch.autograd.grad(expected,z)[0]
  torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
  self.assertEqual(actual['inter_actual_eligible_view_pairs'],eligible);self.assertEqual(actual['inter_eligible_weighted_pair_mass_exact'],str(mass))
  self.assertEqual(actual['denominators']['inter_ordered_original_question_pairs'],6)
  self.assertNotAlmostEqual(float(expected.detach()),float((sum(terms)/float(mass)).detach()))
 def test_same_q_positive_negative_views_are_not_inter_pairs(self):
  own=groups((5,));views=build_training_views(own,mode='per-question-balanced',views_per_question=4)['groups']
  self.assertEqual(sum(v['true_mixed'] for v in views),2)
  z=torch.randn(1,10,dtype=torch.float64,requires_grad=True);a=g10_question_loss(z,own,views,trajectory_weight=0.,inter_rank_weight=1.)
  self.assertEqual(float(a['loss'].detach()),0.);self.assertEqual(a['inter_actual_eligible_view_pairs'],0);self.assertEqual(float(torch.autograd.grad(a['loss'],z)[0].abs().sum()),0.)
 def test_128_real_views_and_exact_question_mass_pair_ceiling(self):
  own=groups(tuple(i%11 for i in range(32)));views=build_training_views(own,mode='random',views_per_question=4,seed=1729,epoch=0)['groups'];self.assertEqual(len(views),128)
  mass={g['qid']:Fraction(0) for g in own}
  for v in views:mass[v['qid']]+=Fraction(v['weight_numerator'],v['weight_denominator'])
  self.assertEqual(set(mass.values()),{Fraction(1)});pos=sum(v['true_mixed'] for v in views);self.assertLessEqual(pos*(128-pos),4096)
  self.assertEqual(64*64,self.original()['max_inter_view_pairs'])
 def test_inter_callback_uses_no_within_mask_or_signature(self):
  from . import probe_g10_train as t
  old=self.original();old['micro_trajectories']=5;lookup={g['qid']:g for g in groups()}
  with patch.object(t,'bounded_question_update',return_value={}) as fake:
   out=r.callback('inter1')(None,None,None,None,lookup,list(lookup),old,None,0)
   self.assertIsNone(fake.call_args.kwargs['distinguishable_pair_mask']);self.assertFalse(out['within_mask_constructed'])
  with self.assertRaises(ValueError):r.callback('bothrank1')
 def test_disjoint_jobs_bases_and_remote_source_compile(self):
  self.assertNotEqual(r.target('inter1'),r.target('bothrank1'));self.assertNotEqual(r.target('inter1'),r.w.TARGET)
  for recipe in r.RECIPES:
   script=r.render(r.target(recipe),r.target(recipe)/'code-test','a'*64,recipe);subprocess.run(['bash','-n'],input=script,text=True,check=True)
   self.assertIn('--recipe '+recipe,script);self.assertIn('--qos=normal',script);self.assertNotIn('afterok',script);self.assertNotIn('@@',script)
   values=dict(base=str(r.target(recipe)),target=str(r.target(recipe)),recipe=recipe,pin=r.w.D8,callback_commit=r.w.CALLBACK,reference_commit=r.REFERENCE)
   for action in ('init','chunk','extract','prepare','submit','status'):
    source=d.remote_source(values,action);compile(source,action,'exec');self.assertLess(len(('python3 -c '+shlex.quote(source)).encode()),16000)
   self.assertIn('q7b-peer8192-'+recipe,d.remote_source(values,'submit'));self.assertIn('Unknown previous submission outcome',d.remote_source(values,'submit'))
  self.assertIn('matched_rank_reference_commit',d.EXTRACT)
 def test_child_and_cp_audit_are_own_base_not_within1(self):
  text=inspect.getsource(r.train);self.assertIn("peer_rank_matched','--recipe',recipe",text);self.assertNotIn('w.train(',text);self.assertNotIn('w.audit_first(',text)
  audit=inspect.getsource(r.audit_first);self.assertIn('base=target(recipe)',audit);self.assertIn('source_all_rank_states',audit);self.assertIn('actual!=replay',audit)
if __name__=='__main__':unittest.main()
