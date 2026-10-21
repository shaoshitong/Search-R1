"""Small fixed screen math checks; synthetic values only."""
import copy,unittest
from . import direct_count_large_screen as m


class ScreenTests(unittest.TestCase):
    def test_AW_AC_and_fixed_halves(self):
        rows=[dict(qid=str(i),q=q,K=k)for i,(q,k)in enumerate([(0.8,0),(.2,0),(.8,5),(.8,1),(.1,2),(.1,5),(.6,3),(.8,4)])]
        v=m.metrics(rows);self.assertEqual(v['allwrong_FPR'],.5);self.assertEqual(v['allcorrect_FPR'],.5);self.assertEqual(v['Mixed_BA'],.625)
        qids=['q%d'%i for i in range(2560)]
        self.assertEqual(m.fixed_halves(qids),m.fixed_halves(list(reversed(qids))))
        self.assertFalse(set(m.fixed_halves(qids)[0])&set(m.fixed_halves(qids)[1]))

    def test_joint_gate_and_frozen_ties(self):
        def metric(ba,top):return dict(Mixed_BA=ba,Top25percent=dict(Mixed_count=top))
        base=dict(NQ=metric(.6,240),half0=metric(.6,120),half1=metric(.6,120))
        rows=[dict(epoch=e,model=s,NQ=metric(.61,241),half0=metric(.6,120),half1=metric(.6,120))for e in m.EPOCHS for s in m.MODES]
        chosen=m.select_candidate(rows,base);self.assertEqual((chosen['epoch'],chosen['model']),(0,'EMA'))
        weaker=copy.deepcopy(rows[0]);weaker['half0']['Mixed_BA']=.59;self.assertFalse(m.qualifies(weaker,base))
        equal=copy.deepcopy(rows[0]);equal['NQ']['Top25percent']['Mixed_count']=240;self.assertFalse(m.qualifies(equal,base))
        with self.assertRaises(ValueError):m.select_candidate(rows[:-1],base)


if __name__=='__main__':unittest.main()
