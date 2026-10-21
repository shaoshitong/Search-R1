import gzip
import json
from pathlib import Path
import tempfile
import unittest

from . import direct_count_large_independent_audit as a


class IndependentAuditTests(unittest.TestCase):
    def test_metrics_ties_confusion_and_exact_quarter(self):
        rows=[dict(qid='q%d'%i,K=k,q=q)for i,(k,q)in enumerate([(0,.8),(5,.8),(2,.8),(1,.2),(0,.2),(5,.2),(4,.8),(3,.2)])]
        m=a.metrics(rows)
        self.assertEqual(m['confusion'],dict(TP=2,FN=2,TN=2,FP=2))
        self.assertEqual(m['Mixed_BA'],.5)
        self.assertEqual(m['Mixed_AUC'],.5)
        self.assertEqual(m['allwrong_FPR'],.5)
        self.assertEqual(m['allcorrect_FPR'],.5)
        self.assertEqual(m['Top25percent']['questions'],2)
        self.assertEqual(m['Top25percent']['Mixed_count'],0)
        self.assertEqual(m['Top25percent']['ordered_qids_sha256'],a.digest(['q0','q1']))

    def test_missing_classes_are_na_and_scores_fail_closed(self):
        m=a.metrics([dict(qid='q',K=2,q=.5)])
        self.assertIsNone(m['Mixed_BA']);self.assertIsNone(m['Mixed_AUC'])
        self.assertIsNone(m['allwrong_FPR']);self.assertIsNone(m['allcorrect_FPR'])
        for value in (float('nan'),float('inf'),-.001,1.001):
            with self.assertRaises(ValueError):a.metrics([dict(qid='q',K=1,q=value)])
        with self.assertRaises(ValueError):a.metrics([dict(qid='q',K=True,q=.5)])

    def test_nq_halves_are_identity_only_and_permutation_stable(self):
        qids=['nq-%04d'%i for i in range(2560)]
        x=a.expected_halves(qids);y=a.expected_halves(list(reversed(qids)))
        self.assertEqual(x,y);self.assertEqual([len(h)for h in x],[1280,1280])
        self.assertFalse(set(x[0])&set(x[1]));self.assertEqual(set(x[0]+x[1]),set(qids))
        with self.assertRaises(ValueError):a.expected_halves(qids[:-1]+[qids[0]])

    def test_gate_requires_strict_whole_and_both_noninferior_halves(self):
        def v(ba,top):return dict(Mixed_BA=ba,Top25percent=dict(Mixed_count=top))
        base=dict(NQ=v(.5,200),half0=v(.5,100),half1=v(.5,100))
        candidate=dict(NQ=v(.51,201),half0=v(.5,101),half1=v(.51,100))
        self.assertTrue(a.gate(candidate,base)['pass_internal_node'])
        candidate['NQ']=v(.5,201)
        self.assertFalse(a.gate(candidate,base)['pass_internal_node'])
        candidate['NQ']=v(.51,201);candidate['half1']=v(.49,101)
        self.assertFalse(a.gate(candidate,base)['pass_internal_node'])

    def test_row_own_em_order_and_optional_ids_are_checked(self):
        fixed={'q0':dict(labels=[0,1,0,0,0],trajectory_ids=['t%d'%i for i in range(5)],domain='NQ'),
            'q1':dict(labels=[1]*5,trajectory_ids=['u%d'%i for i in range(5)],domain='HotpotQA')}
        rows=[dict(qid=q,K=sum(fixed[q]['labels']),q=.8,domain=fixed[q]['domain'],count_NLL=.2,predicted_K=1)for q in fixed]
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'rows.gz'
            def write(values):
                with gzip.open(p,'wt')as f:
                    for v in values:f.write(json.dumps(v)+'\n')
            write(rows);self.assertEqual(a.verify_rows(p,list(fixed),fixed),rows)
            bad=[dict(r)for r in rows];bad[0]['K']=0;write(bad)
            with self.assertRaises(ValueError):a.verify_rows(p,list(fixed),fixed)
            write(list(reversed(rows)))
            with self.assertRaises(ValueError):a.verify_rows(p,list(fixed),fixed)
            bad=[dict(r)for r in rows];bad[0]['trajectory_ids']=['wrong']*5;write(bad)
            with self.assertRaises(ValueError):a.verify_rows(p,list(fixed),fixed)
            write([rows[0],rows[0]])
            with self.assertRaises(ValueError):a.verify_rows(p,list(fixed),fixed)


if __name__=='__main__':unittest.main()
