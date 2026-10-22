"""CPU-only fake fixtures; never opens real final data or checkpoint tensors."""
import copy,sys,types,unittest
from unittest.mock import patch
from . import score_frozen_final4096 as s


class FinalFreezeTests(unittest.TestCase):
    def freeze(self):
        ref=dict(path='/remote/fake.json',bytes=1,sha256='a'*64)
        return dict(status='CANDIDATE_FROZEN_BEFORE_FINAL_SCORES',freeze_epoch=1.,
            final_labels_or_scores_used_for_selection=False,further_candidate_selection_on_final_allowed=False,
            primary_validation_questions=2048,baseline=dict(checkpoint=s.E1,state_key='ema',layer=27,prefix=20),
            candidate=dict(checkpoint=ref,state_key='EMA',layer=21,prefix=80),
            viewed_validation_evaluation=ref,scorer_publication=ref)

    def test_freeze_required_before_final_read(self):
        f=self.freeze(); self.assertTrue(s.freeze_contract(f))
        for key,value in [('status','TEMPLATE_ONLY_NOT_CANDIDATE_FREEZE'),
                          ('final_labels_or_scores_used_for_selection',True),
                          ('further_candidate_selection_on_final_allowed',True)]:
            bad=copy.deepcopy(f);bad[key]=value
            with self.assertRaises(ValueError):s.freeze_contract(bad)

    def test_own_g5_labelblind_ids(self):
        own=[dict(qid=str(i),trajectory_ids=['%d-%d'%(i,j)for j in range(10)])for i in range(2048)]
        fixed=[dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'][::2])for g in own]
        p=dict(planned_G10=own,labels_observed=False,fixed_G5=dict(groups=fixed,summary=dict(role='final',labels_consulted=False,selected_g5_sha256=s.digest(fixed))))
        self.assertEqual(s.plan_groups(p)[0],fixed)
        bad=copy.deepcopy(p);bad['fixed_G5']['groups'][0]['trajectory_ids'][0]='foreign'
        bad['fixed_G5']['summary']['selected_g5_sha256']=s.digest(bad['fixed_G5']['groups'])
        with self.assertRaises(ValueError):s.plan_groups(bad)

    def test_paired_identical_scores_zero_difference(self):
        import numpy as np
        # Stub only the established metric helper to keep this test torch-free.
        def metrics(rows):
            y=np.array([r['true_mixed']for r in rows]);z=np.array([r['logit']for r in rows]);p=z>=0
            top=sorted(rows,key=lambda r:(-r['logit'],r['qid']))[:512]
            return dict(balanced_accuracy=float(.5*(p[y].mean()+(~p[~y]).mean())),selected512=dict(Mixed_fraction=sum(r['true_mixed']for r in top)/512))
        stub=types.ModuleType(s.__package__+'.online_mixed_prefix_explore');stub.metrics=metrics
        rows=[dict(qid='%04d'%i,true_mixed=bool(i%3),labels=[0,1,1,1,1]if i%3 else[0]*5,
            baseline_logit=(i%31-15)/5,candidate_logit=(i%31-15)/5)for i in range(2048)]
        with patch.dict(sys.modules,{stub.__name__:stub}):result=s.paired_summary(rows,repeats=25)
        difference=result['paired_difference_candidate_minus_E1_EMA']
        self.assertEqual(difference['Mixed_balanced_accuracy'],0.)
        self.assertEqual(difference['Top512_true_Mixed_fraction'],0.)
        self.assertEqual(difference['percentile95_question_bootstrap']['Mixed_balanced_accuracy'],[0.,0.])
        self.assertEqual(difference['percentile95_question_bootstrap']['Top512_true_Mixed_fraction'],[0.,0.])
        s.canonical(result)


if __name__=='__main__':unittest.main()
