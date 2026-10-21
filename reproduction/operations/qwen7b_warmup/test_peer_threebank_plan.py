"""Pure metadata boundaries; no real data or model loads."""
import copy,hashlib,json,unittest
from . import peer_threebank_plan as p

def g(domain,i):
 q=json.dumps([domain,str(i)]);return dict(qid=q,question_hash=hashlib.sha256(q.encode()).hexdigest(),trajectory_ids=[q+'-t'+str(j) for j in range(10)],labels=[(i+j)%2 for j in range(10)])
def fixture():
 core=[g('nq',i) for i in range(4)];extra=[g('nq',i) for i in range(4,10)]+[g('hotpotqa',i) for i in range(6)];qs=[v['qid'] for v in core]
 old=dict(cohort_qids=qs,batches=[qs[2:],qs[:2]],batch_epochs=[0,0],questions=4,updates=2,global_questions=2,views_per_question=4,view_weight=.25,effective_total_weight_per_original_question=1.,labels_used_for_question_order=False,all_questions_once=True,seed=1729)
 controlled=[{k:extra[0][k] for k in ('qid','question_hash')}]
 return core,extra,old,controlled

def run(core,extra,old,controlled,excluded=()):return p.build_partitions(core,extra,old,controlled,list(excluded),expected_core=4,expected_added=12,hold_per_domain=2,batch=2)
class ThreeBankTests(unittest.TestCase):
 def test_label_blind_domain_quota_and_control_fit_preserved(self):
  core,extra,old,ctl=fixture();a=run(core,extra,old,ctl);changed=copy.deepcopy(extra)
  for v in changed:v['labels']=[1-y for y in v['labels']]
  self.assertEqual(a,run(core,changed,old,ctl));self.assertEqual(a['selection']['summary']['domains'],{'nq':2,'hotpotqa':2});self.assertEqual(a['controlled_overlap_in_new'],1);self.assertEqual(a['controlled_overlap_retained_in_fit'],1)
  self.assertNotIn(ctl[0]['qid'],[x['qid'] for x in a['holdout_identity']]);self.assertIn(ctl[0]['qid'],[x['qid'] for x in a['fit_identity']])
 def test_exact_old_batch_prefix_all_once_and_fixed_new5(self):
  core,extra,old,ctl=fixture();a=run(core,extra,old,ctl);plan=a['plan'];self.assertEqual(plan['batches'][:2],old['batches']);self.assertEqual(plan['questions'],12);self.assertEqual(plan['updates'],6);self.assertEqual(plan['additional_updates'],4)
  flat=[q for b in plan['batches'] for q in b];self.assertEqual(len(flat),len(set(flat)));self.assertEqual(set(flat),{g['qid'] for g in a['fit_identity']});self.assertFalse(set(flat)&{g['qid'] for g in a['holdout_identity']})
  lookup={g['qid']:g for g in extra}
  for g in a['selection']['groups']:
   self.assertEqual(g['trajectory_ids'],lookup[g['qid']]['trajectory_ids'][5:]);self.assertEqual(g['source_indices'],[5,6,7,8,9]);self.assertNotIn('labels',g)
 def test_no_eval_overlap_or_identity_alias_or_budget_padding(self):
  core,extra,old,ctl=fixture()
  with self.assertRaises(ValueError):run(core,extra,old,ctl,[{k:core[0][k] for k in ('qid','question_hash')}])
  other=copy.deepcopy(extra);other[0]['question_hash']=core[0]['question_hash']
  with self.assertRaises(ValueError):run(core,other,old,ctl)
  with self.assertRaises(ValueError):run(core,extra[:-1],old,ctl)
  bad=copy.deepcopy(old);bad['batches'][0][0]=bad['batches'][1][0]
  with self.assertRaises(ValueError):run(core,extra,bad,ctl)
 def test_labels_join_own_ids_only_after_identity_selection(self):
  core,extra,old,ctl=fixture();a=run(core,extra,old,ctl);lookup={g['qid']:g for g in core+extra};rows=p.labeled(a['holdout_identity'],lookup)
  for row in rows:self.assertEqual(row['labels'],lookup[row['qid']]['labels'])
  wrong=copy.deepcopy(lookup);wrong[rows[0]['qid']]['trajectory_ids'].reverse()
  with self.assertRaises(ValueError):p.labeled(a['holdout_identity'],wrong)
if __name__=='__main__':unittest.main()
