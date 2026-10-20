"""Actual consumer boundary tests; synthetic metadata, no tensor payloads."""
import copy
import json
import unittest
from . import peer_newseed_train as p


def fixture():
    groups=[dict(qid=json.dumps(['nq',i]),question_hash=('%064x'%i),
        trajectory_ids=['own-%d-%d'%(i,j) for j in range(10)],labels=[0,1,0,1,0,1,0,1,0,1]) for i in range(p.Q)]
    chosen=[{k:g[k] for k in ('qid','question_hash','trajectory_ids')} for g in groups]
    summary=dict(labels_consulted=False,selected_questions=p.Q,selected_trajectories=p.Q*10,
        controlled6528_and_eval_qids_and_hashes_excluded=True,
        ordered_qid_sha256=p.raw.digest([g['qid'] for g in groups]),
        trajectory_ids_sha256=p.raw.digest([t for g in groups for t in g['trajectory_ids']]))
    identity=dict(role='train',groups=chosen,summary=summary,domain_scope=dict(kind='NQ_only_enrichment'))
    complete=dict(status=p.raw.STATUS,role='train',actor_updates=0,probe_updates=0,strict_EM_mismatches=0,
        final_labels_or_rollouts_read=False,real_trajectories_per_question=10)
    return complete,groups,identity


class PeerConsumerTests(unittest.TestCase):
    def test_exact_identity_join_and_all_unique_old_new(self):
        c,g,i=fixture();selected=p.select_groups(c,g,i)
        self.assertEqual(len(selected),8192);self.assertEqual(sum(len(x['trajectory_ids']) for x in selected),81920)

    def test_question_plan_is_label_independent_and_once_only(self):
        _,g,_=fixture();one=p.build_plan(g)
        changed=copy.deepcopy(g)
        for row in changed:row['labels']=[1-y for y in row['labels']]
        self.assertEqual(one,p.build_plan(changed))
        self.assertEqual(len(one['batches']),256)
        self.assertTrue(all(len(batch)==32 for batch in one['batches']))
        self.assertEqual(sorted(q for batch in one['batches'] for q in batch),sorted(one['cohort_qids']))

    def test_reject_noncomplete_own_em_and_label_aware_selection(self):
        c,g,i=fixture();c['strict_EM_mismatches']=1
        with self.assertRaises(ValueError):p.select_groups(c,g,i)
        c['strict_EM_mismatches']=0;i['groups'][0]['labels']=[0]*10
        with self.assertRaises(ValueError):p.select_groups(c,g,i)

    def test_reject_duplicate_or_wrong_new_trajectory(self):
        c,g,i=fixture();g[0]['trajectory_ids'][5]=g[0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):p.select_groups(c,g,i)

    def test_reject_domain_or_identity_drift(self):
        c,g,i=fixture();i['groups'][0]['question_hash']='f'*64
        with self.assertRaises(ValueError):p.select_groups(c,g,i)

    def test_existing_native_runner_keeps_explicit_default_loaders(self):
        import inspect
        from . import probe_g10_train as native
        args=inspect.signature(native.run).parameters
        self.assertIs(args['bank_loader'].default,native.accepted_g10)
        self.assertIs(args['reader_factory'].default,native.native_reader)


if __name__=='__main__':unittest.main()
