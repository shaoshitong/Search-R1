"""Only tiny stdlib aggregate fixtures, no real data/checkpoint or torch import."""
import unittest
from . import audit_prefix_confidence_internal as a


def metric(ba,top):
    return dict(pool=dict(mixed_balanced_accuracy=ba),selected512=dict(mixed_questions=top))


class InternalAuditTest(unittest.TestCase):
    def test_independent_joint_gate_requires_raw_and_matched_control(self):
        raw,control=metric(.64,238),metric(.63,229)
        self.assertFalse(a.gate(raw,control,metric(.6336,232))['passed'])
        self.assertTrue(a.gate(raw,control,metric(.65,240))['passed'])
        self.assertFalse(a.gate(raw,metric(.66,241),metric(.65,240))['passed'])
        self.assertFalse(a.gate(raw,metric(.65,240),metric(.65,240))['passed'])

    def test_same_split_is_order_invariant_and_whole_question(self):
        qids=['q%d'%i for i in range(8192)];fit,hold=a.split(qids);f2,h2=a.split(list(reversed(qids)))
        self.assertEqual((len(fit),len(hold)),(6144,2048));self.assertFalse(set(fit)&set(hold))
        self.assertEqual(set(fit),set(f2));self.assertEqual(set(hold),set(h2))

    def test_independent_stdlib_metrics_known_own_labels(self):
        rows=[dict(qid=str(i),trajectory_ids=['%d-%d'%(i,j) for j in range(5)],
                   strict_em=[0,1,0,1,0] if i%2 else [0]*5,score=.8 if i%2 else .2) for i in range(2048)]
        got=a.n.summary(rows);self.assertEqual(got['pool']['mixed_balanced_accuracy'],1)
        self.assertEqual(got['pool']['mixed_roc_auc'],1);self.assertEqual(got['selected512']['mixed_questions'],512)
        self.assertEqual(got['selected512']['k_histogram'],[0,0,512,0,0,0])


if __name__=='__main__':unittest.main()
