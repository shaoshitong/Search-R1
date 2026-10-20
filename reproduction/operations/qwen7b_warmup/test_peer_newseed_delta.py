import copy
import unittest
from . import peer_newseed_delta as m


def item(n):
    return dict(input='inputs/shard-00/batch-%06d.json'%n,input_sha256=str(n)*64,
                seal_sha256=str(n+1)*64,questions=128,trajectories=640)


class DeltaTest(unittest.TestCase):
    def test_exact_batch_difference_without_old_duplication(self):
        a,b,c=item(0),item(1),item(2)
        self.assertEqual(m.partition([c,a,b],[a,b]),[c])
        self.assertEqual(m.partition([a,b],[b,a]),[])

    def test_imported_changed_or_missing_fail_closed(self):
        a,b=item(0),item(1);bad=copy.deepcopy(a);bad['seal_sha256']='x'*64
        for current in ([bad,b],[b],[a,a,b]):
            with self.assertRaises(ValueError):m.partition(current,[a])

    def test_root_source_contract(self):
        c=dict(run_id=m.RUN,seed=m.SEED,source_seed=1729,model_revision=m.REVISION,
               sampling_attempt_id=m.ATTEMPT,capture_prefix_tokens=20,feature_layer=26,
               question_max_tokens=273,actor_optimizer_updates=0,warmup_only=True,actor_hidden_native=False)
        m.contract(c)
        for k,value in [('seed',1729),('model_revision','wrong'),('capture_prefix_tokens',5),('actor_optimizer_updates',1)]:
            bad=dict(c);bad[k]=value
            with self.assertRaises(ValueError):m.contract(bad)


if __name__=='__main__':unittest.main()
