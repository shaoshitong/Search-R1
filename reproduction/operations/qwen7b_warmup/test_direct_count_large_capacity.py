"""Narrow CPU metadata tests; no pretrained model/tensor/cache loading."""
import copy
import unittest
from . import direct_count_large_capacity as m


def fixture():
    fit=['fit-%05d'%i for i in range(m.FIT)]
    held=['held-%05d'%i for i in range(m.HELD)]
    def row(q,domain='nq'):
        return dict(qid=q,question_hash='hash-'+q,trajectory_ids=[q+'-t%d'%j for j in range(10)],
            labels=[0,1,0,0,0,1,0,1,1,0],data_source=domain)
    groups=[row(q)for q in fit]+[row(q,'nq'if i<2560 else'hotpotqa')for i,q in enumerate(held)]
    selected=[dict(qid=g['qid'],question_hash=g['question_hash'],trajectory_ids=g['trajectory_ids'][5:],
        source_indices=[5,6,7,8,9],domain=g['data_source'])for g in groups[m.FIT:]]
    fixed=[dict(qid=g['qid'],question_hash=g['question_hash'],trajectory_ids=g['trajectory_ids'][5:],labels=g['labels'][5:])for g in groups[m.FIT:]]
    plan={'batches':[fit[i:i+32]for i in range(0,len(fit),32)]}
    return plan,groups,{'groups':selected},fixed


class MetadataTests(unittest.TestCase):
    def test_exact_real_geometry_and_cost(self):
        p=m.memory_plan()
        self.assertEqual(p['hidden_shape'],[351360,20,3584])
        self.assertEqual(p['hidden_bytes'],50370969600)
        self.assertEqual(p['fit_updates_total'],18760)
        self.assertGreater(p['GPU_reserve_after_cache_required_bytes'],15*1024**3)

    def test_identity_views_and_no_label_selection(self):
        plan,groups,selection,fixed=fixture()
        _,fit,held,counts=m.validate_cohorts(plan,groups,selection,fixed)
        self.assertFalse(set(fit)&set(held));self.assertEqual(counts,{'NQ':2560,'HotpotQA':2560})
        changed=copy.deepcopy(fixed);changed[0]['trajectory_ids'][0]=groups[0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):m.validate_cohorts(plan,groups,selection,changed)
        changed=copy.deepcopy(fixed);changed[0]['labels'][0]^=1
        with self.assertRaises(ValueError):m.validate_cohorts(plan,groups,selection,changed)

    def test_orders_retain_original_first_epoch_and_full_question_mass(self):
        plan,groups,selection,fixed=fixture();fit=[q for b in plan['batches']for q in b]
        orders=m.epoch_orders(fit,plan['batches'])
        self.assertEqual(orders[0],plan['batches'])
        self.assertEqual(len(orders),20)
        for epoch in orders:self.assertEqual(sorted(q for batch in epoch for q in batch),sorted(fit))
        self.assertTrue(all(epoch==plan['batches']for epoch in orders))


if __name__=='__main__':unittest.main()
