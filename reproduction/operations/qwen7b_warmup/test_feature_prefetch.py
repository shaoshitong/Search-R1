"""Sampler lookahead identity and asynchronous producer/consumer smoke test."""
import json,time,unittest
from collections import OrderedDict
import torch
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from .replay_feature_reader_fast import PrefetchedReplayFeatureReader
from . import probe_ablation_train as u

class Reader(PrefetchedReplayFeatureReader):
    def _decode(self,ids):
        time.sleep(.003)
        return dict(features=torch.ones(len(ids),2,2),attention_mask=torch.ones(len(ids),2,dtype=torch.bool),position_ids=torch.zeros(len(ids),2,dtype=torch.long),qids=[x.split(':')[0]for x in ids])

def groups():
    return [dict(qid='q%d'%i,trajectory_ids=['q%d:r%d'%(i,j)for j in range(10)],labels=([0]*5+[1]*5 if i%2 else[0]*10),question_hash=('%064x'%i))for i in range(80)]

class PrefetchTests(unittest.TestCase):
    def test_real_sampler_unchanged_and_epoch_crossing_identity(self):
        sampler=BalancedHardG10Sampler(groups(),steps_per_epoch=3)
        reader=Reader.__new__(Reader);reader.layer,reader.dimension=26,2;reader.records={};reader.cache=OrderedDict();reader.prototypes={};reader.closed=False;reader.synthetic_fixture=True
        try:
            for _ in range(7):
                before=u.digest(sampler.state_dict());reader.prepare(sampler)
                self.assertEqual(before,u.digest(sampler.state_dict()))
                batch=sampler.next_batch();reader.accept_batch(batch)
                ids=[tid for q in batch['source_qids']for tid in sampler.groups[q]['trajectory_ids']]
                data=reader.get(ids,'cpu');self.assertEqual(data['qids'],[q for q in batch['source_qids']for _ in range(10)])
                sampler.observe(batch,[.5]*128)
            self.assertEqual(sampler.accepted_updates,7)
            self.assertGreater(reader.prefetch_statistics['consumed_question_groups'],0)
        finally:reader.close()

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
