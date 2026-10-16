"""Exported membership, cross-cycle defer, and exact pending-state recovery."""
import copy
from types import SimpleNamespace
import unittest

import pandas as pd
import torch

from reproduction.prefix20_online.pool import identity,initialize_records
from search_g1.prefix20_data import Prefix20CandidateStream,eligible_indices
from search_g1.training_state import compact_stream_state

try:
    from verl import DataProto
except ModuleNotFoundError:
    DataProto=None


def fixture():
    texts=['Alpha x','Alpha  x','Beta','Diagnostic',' Diagnostic  ',
           'Gamma','Delta','Epsilon','Zeta','Eta']
    rows=[dict(prompt=[dict(role='user',content=text)],data_source='constructed',
               extra_info=dict(index=i),reward_model=dict(ground_truth=dict(target=str(i))))
          for i,text in enumerate(texts)]
    cached=[dict(identity(rows[i]),split='train' if i in (1,2) else 'test') for i in (1,2,3)]
    pool,_=initialize_records(rows,cached,train_sha256='constructed',epoch_questions=2,diagnostic_questions=1)
    return rows,pool


class Prefix20MembershipTests(unittest.TestCase):
    def test_exact_preferred_representative_and_both_diagnostic_exclusions(self):
        rows,pool=fixture();frame=pd.DataFrame(rows)
        indices=eligible_indices(frame,'prompt',pool['records'])
        self.assertIn(1,indices);self.assertNotIn(0,indices)
        self.assertNotIn(3,indices);self.assertNotIn(4,indices)
        bad=copy.deepcopy(pool['records']);bad[0]['source_index']=0
        with self.assertRaisesRegex(ValueError,'ABSENT_OR_CHANGED'):eligible_indices(frame,'prompt',bad)
        bad=copy.deepcopy(pool['records']);bad[0]['normalized_question_hash']='0'*64
        with self.assertRaisesRegex(ValueError,'ABSENT_OR_CHANGED'):eligible_indices(frame,'prompt',bad)


@unittest.skipUnless(DataProto is not None,'native DataProto dependencies absent')
class Prefix20StreamTests(unittest.TestCase):
    def source(self):
        rows,pool=fixture()
        class Dataset:
            prompt_key='prompt'
            dataframe=pd.DataFrame(rows)
            def __len__(self):return len(self.dataframe)
            def __getitem__(self,index):return dict(token=torch.tensor([index]),gold=str(index))
        from verl.utils.dataset.rl_dataset import collate_fn
        return SimpleNamespace(dataset=Dataset(),collate_fn=collate_fn),pool

    def stream(self,loader,pool,state=None):
        value=Prefix20CandidateStream.__new__(Prefix20CandidateStream)
        value._initialize(loader,pool['records'],state)
        return value

    def test_fresh_zero_cross_cycle_once_and_explicit_canonical_mapping(self):
        loader,pool=self.source();stream=self.stream(loader,pool)
        self.assertEqual((stream.cursor,stream.consumed,stream.pending),(0,0,None))
        seen=[]
        for _ in range(6):
            parents,_=stream.collect_pool(6)
            indices=parents.batch['pool_index'].tolist()
            self.assertEqual(len(set(indices)),6)
            seen.extend(zip(indices,parents.batch['pool_cycle'].tolist()))
            encoded=stream.materialize(parents,list(range(6)))
            canonical=encoded.batch['canonical_source_index'].tolist()
            self.assertEqual(encoded.batch['token'].flatten().tolist(),canonical)
            self.assertEqual(encoded.non_tensor_batch['gold'].tolist(),[str(i) for i in canonical])
        size=len(pool['records'])
        for cycle in range(5):
            indices=[i for i,c in seen if c==cycle]
            self.assertEqual(len(indices),size)
            self.assertEqual(set(indices),set(range(size)))
        self.assertEqual(stream.consumed,36)

    def test_pending_restore_and_driver_rng_produce_exact_same_future(self):
        loader,pool=self.source();stream=self.stream(loader,pool)
        stream.collect_pool(3)
        state=copy.deepcopy(compact_stream_state(stream.state()));rng=torch.get_rng_state().clone()
        self.assertIsNotNone(state['pending_source'])
        expected=[stream.collect_pool(6)[0] for _ in range(3)]
        after=torch.get_rng_state().clone()
        torch.set_rng_state(rng);restored=self.stream(loader,pool,state)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        actual=[restored.collect_pool(6)[0] for _ in range(3)]
        for a,b in zip(expected,actual):
            for key in ('pool_index','pool_cycle','canonical_source_index'):
                self.assertTrue(torch.equal(a.batch[key],b.batch[key]))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(restored.consumed,stream.consumed)
        bad=copy.deepcopy(state);bad['prefix20_pool_binding']['ordered_membership_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'POOL_BINDING'):self.stream(loader,pool,bad)
        with self.assertRaisesRegex(ValueError,'POOL_BINDING'):
            changed=copy.deepcopy(pool);changed['records'].reverse();self.stream(loader,changed,state)


if __name__=='__main__':unittest.main()
