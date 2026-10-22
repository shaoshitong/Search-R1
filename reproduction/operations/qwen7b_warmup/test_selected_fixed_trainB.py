"""Pure-CPU source/identity/legacy-replay tests; no actual CP, pool or inference."""
import copy,math,unittest
from . import score_selected_fixed_trainB as m
class TrainBTest(unittest.TestCase):
 def config(self):
  x=m.template();x.update(status=m.STATUS,root_authorized_actual_execution=True)
  def ref(p):return dict(path=str(p),bytes=1,sha256='a'*64)
  for key in('source_publication','full_fixture','causal_fixture','E1_dev_evaluation','E1_dev_predictions'):
   if key=='source_publication':x[key]=ref(m.BASE/'source-publication.json')
   else:x[key]['bytes']=1;x[key]['sha256']='a'*64
  x['causal_fixture']['sha256']=m.CAUSAL_SHA;x['E1_dev_evaluation']['sha256']='3d832d00b2d40eaf5ffad4568c1785c84270b0a81e033c57e6c7030a93cd0de8';x['E1_dev_predictions']['sha256']='1d821ef5059710213681b4eab3f0b55ad499485ae0b2a774a7684ffc6e9910ea'
  for model in x['models'][1:]:
   for key in('checkpoint','CPU_acceptance','training_publication','training_complete','evaluation','saved_predictions'):model[key]['bytes']=1;model[key]['sha256']='a'*64
  return x
 def test_future_template_rejected_and_bound_fixed_models_accepted(self):
  with self.assertRaises(ValueError):m.descriptor(m.template())
  m.descriptor(self.config())
 def test_no_trainA_or_threshold_or_epoch_selection(self):
  for key,value in [('fixed_train_SHA256','f'*64),('trainA_used',True),('epoch_or_threshold_selected_on_trainB',True),('threshold_fitted',True)]:
   x=self.config();x[key]=value
   with self.assertRaises(ValueError):m.descriptor(x)
  for key,value in [('epoch',9),('state','EMA'),('prefix',80),('threshold',.1)]:
   x=self.config();x['models'][1][key]=value
   with self.assertRaises(ValueError):m.descriptor(x)
 def views(self,n=2048):return [m.explicit(dict(qid='q%04d'%i,trajectory_ids=['q%04d-t%d'%(i,j)for j in range(5)],labels=[0,1,0,1,0]))for i in range(n)]
 def test_saved_predictions_match_exact_G5_not_count_only(self):
  vs=self.views();m.match_saved(copy.deepcopy(vs),vs);bad=copy.deepcopy(vs);bad[0]['trajectory_ids'].reverse()
  with self.assertRaises(ValueError):m.match_saved(bad,vs)
  bad=copy.deepcopy(vs);bad[0]['labels']=[1]*5
  with self.assertRaises(ValueError):m.match_saved(bad,vs)
 def test_real_causal_segments_preserved_against_flattened_P80(self):
  prompt=[11,12];r=dict(qid='q',label=1,prompt=prompt,segments=[dict(policy_length=2,request_length=2,context_tokens=prompt+[21,22]),dict(policy_length=1,request_length=6,context_tokens=prompt+[21,22,300,301,23])]);s=dict(qid='q',prompt=prompt,policy_token_ids=[21,22,23]);m.compare_causal_compact({'t':r},{'t':s},{'t':1});self.assertNotEqual(r['segments'][1]['context_tokens'],prompt+s['policy_token_ids'])
  s['policy_token_ids'][-1]=99
  with self.assertRaises(ValueError):m.compare_causal_compact({'t':r},{'t':s},{'t':1})
 def test_fixed32_adapter_replay_same_identity_and_probability(self):
  rows=[dict(v,logit=.25)for v in self.views(32)]
  # Actual old E1_DEV schema has strict_em rather than labels (the root failure).
  saved=[dict(predicted_mixed=1/(1+math.exp(-.25)),qid=v['qid'],strict_em=v['labels'],trajectory_ids=v['trajectory_ids'],true_mixed=v['true_mixed'])for v in self.views(32)]
  self.assertEqual(set(saved[0]),{'predicted_mixed','qid','strict_em','trajectory_ids','true_mixed'});self.assertEqual(m.identity(rows),m.identity(saved));self.assertTrue(m.replay_alignment(rows,saved)['passed'])
  both=copy.deepcopy(saved);both[0]['labels']=list(both[0]['strict_em']);self.assertTrue(m.replay_alignment(rows,both)['passed']);both[0]['labels'][0]=1-both[0]['labels'][0]
  with self.assertRaises(ValueError):m.identity(both)
  with self.assertRaises(ValueError):m.replay_alignment(rows,both)
  saved[0]['predicted_mixed']+=.01;self.assertFalse(m.replay_alignment(rows,saved)['passed']);saved[0]['trajectory_ids'].reverse()
  with self.assertRaises(ValueError):m.replay_alignment(rows,saved)
if __name__=='__main__':unittest.main()
