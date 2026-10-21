"""Focused real-subset identity, exact balance and sampler recovery checks."""
import copy
import json
import unittest
from fractions import Fraction
from .balanced_hard_g10_sampler import BalancedHardG10Sampler


def source():
    return [dict(qid='q%03d'%i, question_hash='h%03d'%i,
        trajectory_ids=['q%03d:t%d'%(i,j) for j in range(10)],
        labels=[int(j<k) for j in range(10)])
        for i,k in enumerate(list(range(11))*4)]


class SamplerTests(unittest.TestCase):
    def test_exact_class_and_question_weight_own_id_binding(self):
        groups=source(); by={g['qid']:g for g in groups}
        s=BalancedHardG10Sampler(groups,batch_questions=8)
        for _ in range(10):
            batch=s.next_batch(); views=batch['grouping']['groups']
            self.assertEqual(len(set(batch['source_qids'])),8)
            self.assertEqual(sum(v['true_mixed'] for v in views),16)
            mass={q:Fraction() for q in batch['source_qids']}
            for v in views:
                g=by[v['qid']]; ix=v['source_indices']
                self.assertEqual(len(set(ix)),5)
                self.assertEqual(v['trajectory_ids'],[g['trajectory_ids'][j] for j in ix])
                self.assertEqual(v['labels'],[g['labels'][j] for j in ix])
                self.assertEqual(v['true_mixed'],0<sum(v['labels'])<5)
                mass[v['qid']]+=Fraction(v['weight_numerator'],v['weight_denominator'])
            self.assertEqual(set(mass.values()),{Fraction(1)})
            s.observe(batch,[.4]*32)

    def test_exact_resume_including_hardness_and_next_batch(self):
        groups=source(); s=BalancedHardG10Sampler(groups,batch_questions=8,steps_per_epoch=2)
        for step in range(6):
            b=s.next_batch()
            s.observe(b,[1+int(v['qid'][1:])*.01 for v in b['grouping']['groups']])
        state=json.loads(json.dumps(s.state_dict()))
        restored=BalancedHardG10Sampler(list(reversed(groups)),batch_questions=8,steps_per_epoch=2)
        restored.load_state_dict(state)
        for _ in range(5):
            first,second=s.next_batch(),restored.next_batch()
            self.assertEqual(first,second)
            self.assertEqual(first['grouping']['summary']['mixed_views'],16)
            self.assertGreater(sum(first['grouping']['summary']['hard_pool_actual_counts'].values()),0)
            losses=[.2+i*.01 for i in range(32)]
            s.observe(first,losses);restored.observe(second,losses)
            self.assertEqual(s.state_dict(),restored.state_dict())

    def test_eval_guard_no_fabrication_and_transaction(self):
        groups=source()
        with self.assertRaisesRegex(ValueError,'training-only'):
            BalancedHardG10Sampler(groups,role='validation')
        with self.assertRaisesRegex(ValueError,'Evaluation'):
            BalancedHardG10Sampler(groups,forbidden_qids=['q000'])
        with self.assertRaisesRegex(ValueError,'Evaluation'):
            BalancedHardG10Sampler(groups,forbidden_hashes=['h000'])
        bad=copy.deepcopy(groups)
        for g in bad:g['labels']=[0]*10
        with self.assertRaisesRegex(ValueError,'genuine'):
            BalancedHardG10Sampler(bad,batch_questions=8)
        s=BalancedHardG10Sampler(groups,batch_questions=8); b=s.next_batch()
        with self.assertRaises(ValueError):s.next_batch()
        with self.assertRaises(ValueError):s.state_dict()
        with self.assertRaises(ValueError):s.observe(b,[float('nan')]*32)
        self.assertEqual(s.accepted_updates,0)
        changed=copy.deepcopy(b);changed['grouping']['groups'][0]['labels'][0]^=1
        with self.assertRaises(ValueError):s.observe(changed,[1.]*32)
        s.observe(b,[1.]*32)
        state=s.state_dict();state['accepted_updates']+=s.steps_per_epoch
        with self.assertRaisesRegex(ValueError,'epoch/cursor'):
            s.load_state_dict(state)

    def test_k5_negative_views_are_authentic_even_when_repeated(self):
        groups=[dict(qid='q%d'%i,trajectory_ids=['q%d:t%d'%(i,j)for j in range(10)],
                     labels=[1]*5+[0]*5) for i in range(8)]
        s=BalancedHardG10Sampler(groups,batch_questions=8);b=s.next_batch()
        negatives=[v for v in b['grouping']['groups']if not v['true_mixed']]
        self.assertEqual(len(negatives),16)
        self.assertTrue(all(v['source_indices'] in (list(range(5)),list(range(5,10)))for v in negatives))
        self.assertGreater(b['grouping']['summary']['repeated_subset_draws'],0)
        s.observe(b,[.5]*32)


if __name__=='__main__':unittest.main()
