"""Small CPU-only semantic tests; no model, features, remote IO or Slurm."""
import copy,json,unittest
from .detached_score_window import DetachedScoreWindow

def row(q,step,attempt,mixed=True,score=.25):
 labels=[1,0,0,0,0]if mixed else[1]*5
 return dict(qid=str(q),trajectory_ids=[str(q)+'-t'+str(i)for i in range(5)],labels=labels,true_mixed=mixed,cached_score=score,scored_step=step,scored_batch_attempt=attempt,score_model='raw',view=dict(indices=[0,1,2,3,4]))

class CacheTests(unittest.TestCase):
 def test_full_window_warmfill_preupdate_versions(self):
  q=DetachedScoreWindow();last=None
  for step in range(33):
   current=[row(step*64+i,step,step)for i in range(64)];last=q.prepare(current,model_version=step)
   self.assertEqual(last['K'],(last['N_eff']*500+2017)//2018);q.commit(post_update_step=step+1)
   self.assertTrue(all(x['scored_step']==step for x in q.state_dict()['rows'][-64:]))
  self.assertEqual((last['N_eff'],len(last['cached_rows']),last['K']),(2018,1954,500));self.assertEqual(len(q.state_dict()['rows']),2018);self.assertEqual(last['diagnostics']['duplicate_window_qids'],0)
 def test_current_qid_removal_fifo_labelblind(self):
  states=[]
  for flip in(False,True):
   q=DetachedScoreWindow(capacity=6)
   q.prepare([row(i,0,0,mixed=not flip)for i in range(4)],0);q.commit(1)
   w=q.prepare([row(2,1,1,mixed=flip,score=-.8),row(4,1,1,mixed=flip)],1)
   self.assertEqual([x['qid']for x in w['rows']],['0','1','3','2','4']);self.assertEqual(w['diagnostics']['removed_previous_current_qids'],1)
   q.commit(2);q.prepare([row(5,2,2,mixed=not flip),row(6,2,2,mixed=not flip)],2);q.commit(3)
   states.append([x['qid']for x in q.state_dict()['rows']])
  self.assertEqual(states,[['1','3','2','4','5','6']]*2)
 def test_zero_cache_same_selection_fraction_skip_update(self):
  q=DetachedScoreWindow(capacity=0)
  w=q.prepare([row(i,0,0)for i in range(64)],0);self.assertEqual((w['N_eff'],w['K'],w['cached_scores']),(64,16,[]));q.commit(0)
  self.assertEqual(q.state_dict()['rows'],[]);q.prepare([row(i,0,1)for i in range(64)],0);q.commit(1);self.assertEqual(q.state_dict()['last_model_version'],1)
 def test_checkpoint_exact_including_prepared_phase(self):
  q=DetachedScoreWindow(capacity=6);q.prepare([row(i,0,0)for i in range(4)],0);q.commit(1)
  state=q.state_dict();p=DetachedScoreWindow.from_state_dict(json.loads(json.dumps(state)));self.assertEqual(state,p.state_dict());self.assertEqual(q.ring_sha256,p.ring_sha256)
  current=[row(0,1,1),row(8,1,1)];self.assertEqual(q.prepare(current,1),p.prepare(current,1));pending=q.state_dict();restored=DetachedScoreWindow.from_state_dict(pending)
  self.assertEqual(q.commit(2),restored.commit(2));self.assertEqual(q.state_dict(),restored.state_dict());self.assertEqual(q.state_dict()['rows'][-1]['scored_step'],1)
 def test_reject_duplicate_qids_wrong_view_and_graph_features(self):
  q=DetachedScoreWindow();r=row('x',0,0)
  with self.assertRaises(ValueError):q.prepare([r,r],0)
  r=row('x',0,0);r['labels']=[0]*5
  with self.assertRaises(ValueError):q.prepare([r],0)
  r=row('x',0,0);r['view']['features']=[1.]
  with self.assertRaises(ValueError):q.prepare([r],0)
  r=row('x',0,0);r['cached_score']=float('nan')
  with self.assertRaises(ValueError):q.prepare([r],0)

if __name__=='__main__':unittest.main()
