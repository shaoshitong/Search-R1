"""Targeted pure-CPU metadata/token tests; never loads models, CPs or real pools."""
import copy,unittest
from . import packed_full169028_trajectory_permutation_audit as audit

class FullP80PermutationTest(unittest.TestCase):
 def setUp(self):
  self.group=dict(qid='q',trajectory_ids=['t%d'%j for j in range(5)],labels=[0,1,1,0,0])
  self.records={tid:dict(qid='q',trajectory_id=tid,prompt=[11,12,13],policy_token_ids=[100+j]*[80,2,19,79,1][j])for j,tid in enumerate(self.group['trajectory_ids'])}
 def test_exact_own_G5_identity_question_once_P80_order_and_padding(self):
  original=copy.deepcopy((self.group,self.records));identities=[]
  for order in audit.ORDERS:
   v=audit.protocol.reorder(self.group,order);packed=audit.training.pack_group(v,self.records,audit.SEPARATOR,80);identities.append(audit.input_identity(v,self.records))
   self.assertEqual(dict(zip(v['trajectory_ids'],v['labels'])),dict(zip(self.group['trajectory_ids'],self.group['labels'])))
   self.assertEqual((v['M'],v['target_variance'],v['true_mixed']),(5,.24,True));self.assertEqual(packed['input_ids'][:3],[11,12,13]);self.assertEqual(packed['token_types'].count(0),3);self.assertEqual(len(packed['input_ids']),3+5*84)
   for index,tid in enumerate(v['trajectory_ids']):
    start=3+84*index;tokens=self.records[tid]['policy_token_ids'];n=len(tokens)
    self.assertEqual(packed['input_ids'][start:start+4],audit.SEPARATOR);self.assertEqual(packed['input_ids'][start+4:start+4+n],tokens);self.assertEqual(packed['attention_mask'][start+4:start+84],[True]*n+[False]*(80-n));self.assertEqual(packed['position_ids'][start+4+n:start+84],[0]*(80-n))
  self.assertEqual(len({r['unordered_own_ID_EM_tokens']for r in identities}),1);self.assertEqual(len({r['packed_input_sha256']for r in identities}),3);self.assertEqual((self.group,self.records),original)
 def test_labels_never_enter_inputs(self):
  a=audit.protocol.reorder(self.group,'original');b=audit.protocol.reorder(dict(self.group,labels=[1-y for y in self.group['labels']]),'original')
  self.assertEqual(audit.training.pack_group(a,self.records,audit.SEPARATOR,80),audit.training.pack_group(b,self.records,audit.SEPARATOR,80))
 def config(self):
  x=audit.future_template();x.update(status=audit.STATUS,root_authorized_actual_execution=True,epoch=4,state='raw',selected_at=1)
  ref=lambda p:dict(path=str(p),bytes=1,sha256='a'*64)
  x['fixture']=ref(audit.FIXTURE/'fixture-complete.json');x['audit_source_publication']=ref(audit.BASE/'source-publication.json')
  for r in x['replications']:
   parent=audit.PARENTS[r['seed_id']]
   for k,p in dict(checkpoint=parent/'checkpoint-epoch-04.pt',CPU_acceptance=audit.A/'synthetic-CPU-accepted.json',training_complete=parent/'complete.json',training_publication=parent/'source-publication.json',source_evaluation=parent/'evaluation-epoch-04.json',source_saved_predictions=parent/'epoch-04-fixed_validation2048_G5-predictions.jsonl.gz',ready=parent/'ready.json').items():r[k]=ref(p)
  return x
 def test_future_template_and_malformed_geometry_state_orders_rejected(self):
  with self.assertRaises(ValueError):audit.descriptor(audit.future_template())
  audit.descriptor(self.config())
  for field,value in [('prefix',20),('layer',21),('epoch',None),('state','best'),('orders',['original','reverse']),('root_authorized_actual_execution',False),('choose_favorable_trajectory_order',True)]:
   x=self.config();x[field]=value
   with self.assertRaises(ValueError):audit.descriptor(x)
 def test_same_epoch_and_exact_v3_paths_both_seeds(self):
  x=self.config();x['replications'][1]['checkpoint']['path']=str(audit.PARENTS['seed2']/'checkpoint-epoch-05.pt')
  with self.assertRaises(ValueError):audit.descriptor(x)
 def test_saved_evaluation_original_identity_and_score_alignment(self):
  rows=[dict(qid='q%04d'%i,trajectory_ids=['t%d'%j for j in range(5)],labels=[0,1,0,1,0],true_mixed=True,raw_original=.1)for i in range(2048)]
  saved={r['qid']:dict(r,raw_score=.4)for r in rows};result=audit.saved_alignment(rows,saved,'raw');self.assertTrue(result['passed']);self.assertEqual(result['absolute_score_delta_max'],0)
  saved['q0000']['raw_score']=.41;self.assertFalse(audit.saved_alignment(rows,saved,'raw')['passed']);saved['q0000']['trajectory_ids']=list(reversed(saved['q0000']['trajectory_ids']))
  with self.assertRaises(ValueError):audit.saved_alignment(rows,saved,'raw')
if __name__=='__main__':unittest.main()
