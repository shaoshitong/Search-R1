import copy,inspect,subprocess,unittest
from . import peer_within1_matched as w,deploy_peer_within1_matched as d
class Tests(unittest.TestCase):
 def original(self):
  return dict(source_checkpoint={'sha256':w.SOURCE_SHA},source_adam_steps=10464,update_budget=256,global_questions=32,fit_questions=8192,fit_real_trajectories=81920,grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),recipe={'id':'UNCHANGED','loss_kwargs':dict(trajectory_weight=1.,three_event_weight=1.)},optimizer_reset=False,source_commit=w.D8,code_hashes={'D8':'old'},plan={'sha256':'original'},source_all_four_rank_RNG_preserved=True)
 def test_same_causal_even_BF16_drift_excluded(self):
  labels=[1,0]+[0]*8;actual=[[False]*10 for _ in range(10)];actual[0][1]=True;causal=['same','same']+[str(i) for i in range(8)];self.assertFalse(w.gated_pairs(labels,actual,causal)[0][1])
 def test_different_causal_but_same_actual_input_excluded(self):
  self.assertFalse(w.gated_pairs([1]+[0]*9,[[False]*10 for _ in range(10)],[str(i) for i in range(10)])[0][1])
 def test_only_actual_different_causal_and_own_opposite_EM(self):
  labels=[1,0,1]+[0]*7;actual=[[True]*10 for _ in range(10)];mask=w.gated_pairs(labels,actual,[str(i) for i in range(10)]);self.assertTrue(mask[0][1]);self.assertFalse(mask[1][0]);self.assertFalse(mask[0][2]);self.assertFalse(mask[0][0])
 def test_exact_matched_configuration_except_documented_bindings(self):
  old=self.original();before=copy.deepcopy(old);new=w.matched_config(old,{'commit':'new','files':{'r.py':{'sha256':'x'}}},{'path':'remote','sha256':'s'});self.assertEqual(old,before);self.assertEqual(new['plan'],old['plan']);self.assertEqual(new['recipe']['id'],old['recipe']['id']);self.assertEqual(new['recipe']['loss_kwargs'],dict(trajectory_weight=1.,three_event_weight=1.,within_rank_weight=1.));self.assertTrue(new['source_all_four_rank_RNG_preserved']);old['source_checkpoint']['sha256']='wrong'
  with self.assertRaises(ValueError):w.matched_config(old,{'commit':'new','files':{}},{})
 def test_actual_render_and_unique_job(self):
  script=w.render(w.TARGET,w.TARGET/'code-abc','a'*64);subprocess.run(['bash','-n'],input=script,text=True,check=True);self.assertNotIn('@@',script);self.assertNotIn('afterok',script);self.assertIn('--time=02:00:00',script);self.assertIn('--qos=normal',script);self.assertIn('q7b-peer8192-within1',script)
  exports='\n'.join(x for x in script.splitlines() if x.startswith('export '));out=subprocess.check_output(['bash','-c',exports+'\nprintf "%s" "$QWEN7B_CODE_ROOT"'],text=True);self.assertEqual(out,str(w.TARGET/'code-abc'))
 def test_transport_boundaries_and_no_unknown_duplicate(self):
  v=dict(base=str(w.TARGET),target=str(w.TARGET),pin=w.D8,callback_commit=w.CALLBACK)
  for action in ('init','chunk','extract','prepare','submit','status'):compile(d.remote_source(v,action),action,'exec')
  self.assertIn("'file_origins'",d.INIT);self.assertIn("'..' not in rel.parts",d.EXTRACT);self.assertIn("name='q7b-peer8192-within1'",d.SUBMIT);self.assertIn('Unknown previous submission outcome',d.SUBMIT)
 def test_callback_bounded_update_and_independent_cpu_process(self):
  source=inspect.getsource(w);self.assertIn('t.bounded_question_update',source);self.assertIn('causal_signature(row)[0]',source);self.assertIn("CUDA_VISIBLE_DEVICES=''",source);self.assertIn('FIRST_CP1_INDEPENDENT_FULL_SHA_AND_CPU_ALL_COMPONENT_RELOAD_PASSED',source);self.assertNotIn('u.save_checkpoint=',source)
if __name__=='__main__':unittest.main()
