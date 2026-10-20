import itertools
import unittest
from fractions import Fraction

from . import g20_grouping as g


def fixture(k, name):
    return dict(qid=name, trajectory_ids=['%s-%02d'%(name,i) for i in range(20)],
        labels=[1]*k+[0]*(20-k))


class G20GroupingTest(unittest.TestCase):
    def test_unrank_matches_all_true_combinations(self):
        actual = [g.unrank(tuple(range(20)),5,i) for i in range(g.choose(20,5))]
        self.assertEqual(actual,list(itertools.combinations(range(20),5)))

    def test_category_bijections_extreme_and_central_K(self):
        all_indices = set(itertools.combinations(range(20),5))
        for k in (0,1,4,5,10,15,19,20):
            row=fixture(k,'q')
            positive,negative=g.class_capacity(row)
            groups=[]
            for mixed,capacity in ((True,positive),(False,negative)):
                indices=[g.category_subset(row,mixed,i) for i in range(capacity)]
                self.assertEqual(len(indices),len(set(indices)))
                self.assertTrue(all((0<sum(row['labels'][i] for i in ix)<5)==mixed for ix in indices))
                groups.extend(indices)
            self.assertEqual(set(groups),all_indices)
            self.assertEqual(len(groups),len(all_indices))

    def test_strict_balance_weight_and_real_labels(self):
        rows=[fixture(k,str(i)) for i,k in enumerate((0,20,1,10))]
        out=g.build_training_views(rows,mode='global-weighted-balanced')
        self.assertEqual(out['summary']['weighted_mixed_fraction'],.5)
        for row in rows:
            own=[v for v in out['groups'] if v['qid']==row['qid']]
            self.assertEqual(sum(Fraction(v['weight_numerator'],v['weight_denominator']) for v in own),1)
            self.assertEqual(len(set(tuple(v['source_indices']) for v in own)),4)
            for view in own:
                self.assertEqual(view['labels'],[row['labels'][i] for i in view['source_indices']])
                self.assertEqual(view['trajectory_ids'],[row['trajectory_ids'][i] for i in view['source_indices']])

    def test_infeasible_fails_without_dropping_questions(self):
        with self.assertRaisesRegex(ValueError,'impossible'):
            g.build_training_views([fixture(0,'a'),fixture(20,'b'),fixture(10,'c')],mode='global-weighted-balanced')

    def test_random_reproducible_but_epoch_changes(self):
        rows=[fixture(10,'a'),fixture(3,'b')]
        a=g.build_training_views(rows,seed=3,epoch=0)
        self.assertEqual(a,g.build_training_views(rows,seed=3,epoch=0))
        self.assertNotEqual(a['summary']['selected_views_sha256'],g.build_training_views(rows,seed=3,epoch=1)['summary']['selected_views_sha256'])

    def test_fixed_validation_is_label_blind(self):
        a=[fixture(0,'q')]; b=[fixture(20,'q')]
        self.assertEqual(g.fixed_g5(a),g.fixed_g5(b))
        self.assertNotIn('labels',g.fixed_g5(a)['groups'][0])
        with self.assertRaisesRegex(ValueError,'training-only'):
            g.build_training_views(a,role='validation')

    def test_reject_duplicated_or_missing_real_trajectories(self):
        a=fixture(10,'q');a['trajectory_ids'][0]=a['trajectory_ids'][1]
        with self.assertRaises(ValueError):g.validate([a])
        a=fixture(10,'q');a['labels'][0]=.5
        with self.assertRaises(ValueError):g.validate([a])
        with self.assertRaises(ValueError):g.validate([fixture(10,'q')]*2)

    def test_common_plan_preserves_cohort_and_every_batch_feasible(self):
        rows=[fixture(k,str(i)) for i,k in enumerate([0,20]*6+[1,10,19]*4)]
        plan=g.build_batch_plan(rows,batch_questions=4)
        self.assertEqual(set(plan['cohort_qids']),{r['qid'] for r in rows})
        self.assertEqual(len(plan['cohort_qids']),24)
        by={r['qid']:r for r in rows}
        for batch in plan['batches']:
            out=g.build_training_views([by[q] for q in batch],mode='global-weighted-balanced')
            self.assertTrue(out['summary']['weighted_class_mass_balanced'])
        with self.assertRaisesRegex(ValueError,'training-only'):
            g.build_batch_plan(rows,role='validation')
        with self.assertRaisesRegex(ValueError,'full batches'):
            g.build_batch_plan(rows[:-1],batch_questions=4)


if __name__=='__main__':unittest.main()
