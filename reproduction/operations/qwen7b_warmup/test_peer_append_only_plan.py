import copy,unittest
from .peer_append_only_plan import _extend,digest,validate_cursor,validate_source_checkpoint_metadata

def fixture():
    groups=[dict(qid='q%d'%i,question_hash=digest(['q',i]),trajectory_ids=['q%d-t%d'%(i,j) for j in range(10)]) for i in range(96)]
    q=[g['qid'] for g in groups[:32]]
    old=dict(cohort_qids=q,batches=[q[16:],q[:16]],batch_epochs=[0,0],questions=32,updates=2,
        global_questions=16,seed=1729,views_per_question=4,view_weight=.25,
        effective_total_weight_per_original_question=1.,labels_used_for_question_order=False,all_questions_once=True)
    return old,dict(groups=groups)

class AppendOnly(unittest.TestCase):
    def test_old_actual_order_preserved_new_questions_exactly_once(self):
        old,ident=fixture();before=copy.deepcopy(old)
        p=_extend(old,ident,historical_questions=32,total_questions=96,batch=16)
        self.assertEqual(old,before);self.assertEqual(p['batches'][:2],old['batches'])
        self.assertEqual(p['updates'],6);self.assertEqual(p['additional_updates'],4)
        flat=[q for b in p['batches'][2:] for q in b]
        self.assertEqual(len(flat),64);self.assertEqual(set(flat),{'q%d'%i for i in range(32,96)})
        self.assertEqual(p,_extend(old,ident,historical_questions=32,total_questions=96,batch=16))
    def test_reject_history_rewrite_or_label_injection(self):
        old,ident=fixture();bad=copy.deepcopy(old);bad['batches'][1][0]=bad['batches'][0][0]
        with self.assertRaises(ValueError):_extend(bad,ident,historical_questions=32,total_questions=96,batch=16)
        ident['groups'][40]['labels']=[0]*10
        with self.assertRaises(ValueError):_extend(old,ident,historical_questions=32,total_questions=96,batch=16)
    def test_reject_changed_identity_or_missing_real_trajectory(self):
        old,ident=fixture();ident['groups'][0],ident['groups'][1]=ident['groups'][1],ident['groups'][0]
        with self.assertRaises(ValueError):_extend(old,ident,historical_questions=32,total_questions=96,batch=16)
        old,ident=fixture();ident['groups'][-1]['trajectory_ids'][0]=ident['groups'][0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):_extend(old,ident,historical_questions=32,total_questions=96,batch=16)
    def test_cursor_is_cumulative_no_replay_of_completed_prefix(self):
        self.assertEqual(validate_cursor(257,10721,new_config_sha='a',saved_config_sha='a'),1)
        self.assertEqual(validate_cursor(938,11402,new_config_sha='a',saved_config_sha='a'),682)
        for cursor,steps in [(1,10721),(256,10720),(257,10465),(939,11403)]:
            with self.assertRaises(ValueError):validate_cursor(cursor,steps,new_config_sha='a',saved_config_sha='a')
        with self.assertRaises(ValueError):validate_cursor(257,10721,new_config_sha='a',saved_config_sha='a',highest_intent=258)
    def test_source_metadata_never_forged_as_E1_or_reset(self):
        hashes={k:digest(k) for k in ('raw','ema','adam','raw_buffers','ema_buffers','rng')}
        saved=dict(raw={},ema={},optimizer={},raw_buffers={},ema_buffers={},rng={},accepted_hashes=hashes,
            source_all_rank_states=[dict(rank=i,rng={}) for i in range(4)],cursor=256,steps=10720,
            config_sha256='config',plan_sha256='plan',actor_updates=0,optimizer_reset=False,readout_reset=False,block_reset=False)
        receipt={k:saved[k] for k in ('cursor','steps','config_sha256','accepted_hashes')}
        v=validate_source_checkpoint_metadata(saved,receipt,source_config_sha='config',source_plan_sha='plan')
        self.assertTrue(v['full_restore_and_next_update_replay_still_required'])
        for key,value in [('cursor',0),('steps',10464),('optimizer_reset',True),('source_all_rank_states',[])]:
            bad=copy.deepcopy(saved);bad[key]=value
            with self.assertRaises(ValueError):validate_source_checkpoint_metadata(bad,receipt,source_config_sha='config',source_plan_sha='plan')

if __name__=='__main__':unittest.main()
