"""Small CPU proofs of exact original-question class mass and train-only plans."""
from fractions import Fraction
import unittest
import torch
from .g10_question_loss import g10_question_loss
from .g10_weighted_balanced import build_training_views,build_weighted_batch_plan


def group(k,qid):
    return dict(qid=qid,trajectory_ids=[qid+':t'+str(i) for i in range(10)],labels=[1]*k+[0]*(10-k))


class WeightedBalancedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def assert_weights(self,source,result):
        by={g['qid']:g for g in source};mass={q:Fraction() for q in by};seen={q:set() for q in by};positive=Fraction()
        for v in result['groups']:
            own=by[v['qid']];ix=tuple(v['source_indices']);self.assertNotIn(ix,seen[v['qid']]);seen[v['qid']].add(ix)
            self.assertEqual(v['trajectory_ids'],[own['trajectory_ids'][i] for i in ix]);self.assertEqual(v['labels'],[own['labels'][i] for i in ix])
            self.assertEqual(v['true_mixed'],0<sum(v['labels'])<5)
            w=Fraction(v['weight_numerator'],v['weight_denominator']);self.assertGreater(w,0);mass[v['qid']]+=w
            if v['true_mixed']:positive+=w
        self.assertTrue(all(v==1 for v in mass.values()));self.assertTrue(all(len(v)==4 for v in seen.values()));return positive

    def test_weighted_balance_is_not_view_count_balance(self):
        source=[group(k,'q'+str(i)) for i,k in enumerate((0,1,5,9))]
        result=build_training_views(source,weighting='heterogeneous-mass-rational')
        self.assertEqual(self.assert_weights(source,result),2)
        self.assertTrue(result['summary']['weighted_class_mass_balanced']);self.assertFalse(result['summary']['count_balanced'])
        self.assertEqual(result['summary']['mixed_views'],6);self.assertEqual(result['summary']['heterogeneous_question_Mixed_mass'],'2/3')
        # The existing per-question objective accepts these exact rational views.
        logits=torch.linspace(-1,1,40,dtype=torch.float64).reshape(4,10).requires_grad_(True)
        loss=g10_question_loss(logits,source,result['groups'],trajectory_weight=1.,three_event_weight=1.)['loss']
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_default_equal_quota_satisfies_both_count_and_weight_balance(self):
        source=[group(k,'q'+str(i)) for i,k in enumerate((0,1,5,9))]
        result=build_training_views(source)
        self.assertEqual(self.assert_weights(source,result),2)
        self.assertTrue(result['summary']['count_balanced'])
        self.assertTrue(result['summary']['weighted_class_mass_balanced'])
        self.assertEqual(result['summary']['mixed_views'],8)
        self.assertEqual(result['summary']['weighting'],'equal-view-quota')
        self.assertTrue(all(Fraction(v['weight_numerator'],v['weight_denominator'])==Fraction(1,4) for v in result['groups']))
        for row in result['summary']['quota_allocation']['quota_by_question']:
            self.assertLessEqual(row['minimum'],row['selected']);self.assertLessEqual(row['selected'],row['maximum'])
        # K5 has only two real nonMixed subsets, so at least two must beMixed.
        k5=next(v for v in result['summary']['quota_allocation']['quota_by_question'] if v['qid']=='q2')
        self.assertEqual(k5['minimum'],2)
        self.assertEqual(result,build_training_views(list(reversed(source))))

    def test_boundary_uses_four_positive_views_without_zero_weights(self):
        source=[group(k,'q'+str(i)) for i,k in enumerate((0,10,1,5))]
        result=build_training_views(source)
        self.assertEqual(self.assert_weights(source,result),2)
        self.assertTrue(result['summary']['weighted_class_mass_balanced'])
        for row in result['per_question']:
            self.assertIn(row['mixed_views'],(0,4))

    def test_impossible_reports_actual_maximum_without_changing_labels(self):
        source=[group(k,'q'+str(i)) for i,k in enumerate((0,0,10,5))]
        with self.assertRaisesRegex(ValueError,'impossible'):build_training_views(source)
        result=build_training_views(source,on_unbalance='report')
        self.assertEqual(self.assert_weights(source,result),1)
        self.assertFalse(result['summary']['weighted_class_mass_balanced'])
        self.assertEqual(result['summary']['maximum_attainable_weighted_Mixed_fraction'],'1/4')
        self.assertEqual(result['summary']['weighted_mixed_fraction'],.25)

    def test_actual_pilot_counts_all_128_same_plan_eight_feasible_batches(self):
        source=[group(0 if i<62 else 1+(i%9),'q%03d'%i) for i in range(128)]
        plan=build_weighted_batch_plan(source)
        self.assertEqual(plan,build_weighted_batch_plan(list(reversed(source))))
        self.assertEqual(sorted(plan['batch_homogeneous_counts']),[7,7,8,8,8,8,8,8])
        by={g['qid']:g for g in source};seen=[]
        for qids in plan['batches']:
            own=[by[q] for q in qids];seen.extend(qids)
            balanced=build_training_views(own);random=build_training_views(own,mode='random')
            self.assertEqual(self.assert_weights(own,balanced),8);self.assert_weights(own,random)
            self.assertEqual(balanced['summary']['mixed_views'],32)
            self.assertEqual(balanced['summary']['nonmixed_views'],32)
            self.assertTrue(all(Fraction(v['weight_numerator'],v['weight_denominator'])==Fraction(1,4) for v in balanced['groups']))
            self.assertEqual({v['qid'] for v in balanced['groups']},{v['qid'] for v in random['groups']})
        self.assertEqual(len(seen),len(set(seen)));self.assertEqual(set(seen),set(by))
        self.assertTrue(plan['labels_used_for_plan']);self.assertFalse(plan['validation_or_final_labels_read'])

    def test_infeasible_plan_report_and_train_only_guards(self):
        source=[group(0 if i<96 else 5,'q%03d'%i) for i in range(128)]
        with self.assertRaisesRegex(ValueError,'impossible'):build_weighted_batch_plan(source)
        plan=build_weighted_batch_plan(source,on_unbalance='report')
        self.assertFalse(plan['exact_weighted_balance_feasible_in_every_batch'])
        self.assertEqual(plan['maximum_global_weighted_Mixed_fraction'],'1/4')
        self.assertEqual(plan['batch_homogeneous_counts'],[12]*8)
        for function in (build_training_views,build_weighted_batch_plan):
            with self.assertRaisesRegex(ValueError,'training-only'):function(source,role='validation')
        with self.assertRaisesRegex(ValueError,'exactly4'):build_training_views(source,views_per_question=6)
        bad=group(5,'bad');bad['labels'][0]=True
        with self.assertRaises(ValueError):build_training_views([bad])


if __name__=='__main__':unittest.main()
