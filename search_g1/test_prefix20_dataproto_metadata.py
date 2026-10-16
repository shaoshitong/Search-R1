"""Actual DataProto producer/chunk/dispatch regression; requires runtime deps.

Run with the existing training CPU environment. No model, Ray cluster, GPU or
saved rollout is needed; no protocol/model/dispatch test doubles are used.
"""
import copy
import unittest
import numpy as np
import torch
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.single_controller.base.decorator import _split_args_kwargs_data_proto
from search_g1.prefix20_advantage import strict_em_from_batch,anchored_grpo


class AnswerTokenizer:
    def decode(self,ids):
        return ''.join('<answer>right</answer>' if int(token)==1 else '<answer>wrong</answer>' if int(token)==0 else ''
                       for token in ids)


def fixture():
    # Authentic observed category counts, synthetic CPU tokens/logprob values.
    groups=[[1,1,0,0,0]]*273+[[1]*5]*120+[[0]*5]*119
    labels=[value for group in groups for value in group];n=len(labels)
    response=torch.tensor([[value,99] for value in labels],dtype=torch.long)
    logprob=-(torch.arange(n*2,dtype=torch.float32).reshape(n,2)%71+1)/17
    tensors=dict(prompts=torch.arange(n*2).reshape(n,2),responses=response,
        attention_mask=torch.tensor([[1,1,1,0]],dtype=torch.long).repeat(n,1),
        info_mask=torch.tensor([[0,0,1,0]],dtype=torch.long).repeat(n,1),old_log_probs=logprob,
        token_level_rewards=torch.stack((torch.arange(n,dtype=torch.float32)%19/20+.2,torch.zeros(n)),dim=1))
    ids=['selected-q%04d'%(row//5) for row in range(n)]
    trajectories=['%s-r%d'%(ids[row],row%5) for row in range(n)]
    metadata=dict(uid=np.asarray(ids,dtype=object),behavior_trajectory_id=np.asarray(trajectories,dtype=object),
        candidate_stable_id=np.asarray(ids,dtype=object),selection_source=np.asarray(['predicted_mixed']*n,dtype=object),
        reward_model=np.asarray([dict(ground_truth={'target':['right']}) for _ in range(n)],dtype=object))
    data=DataProto(batch=TensorDict(tensors,batch_size=[n]),non_tensor_batch=metadata,
        meta_info={'behavior_identity':{'trajectory_ids':trajectories,'policy_snapshot_id':'fixture-original-policy'}})
    return data,labels


class Prefix20DataProtoMetadataTests(unittest.TestCase):
    def test_actual_old_int8_failure_and_correct_producer_four_rank_chunk(self):
        data,labels=fixture();tokens={k:v.clone() for k,v in data.batch.items()};ids=copy.deepcopy(data.non_tensor_batch)
        wire=strict_em_from_batch(data,AnswerTokenizer())
        self.assertEqual(wire.dtype,object);self.assertEqual(wire.tolist(),labels)
        self.assertTrue(all(type(value) is int for value in wire.tolist()))
        data.non_tensor_batch['prefix20_strict_em']=np.asarray(labels,dtype=np.int8)
        with self.assertRaisesRegex(AssertionError,'dtype=object'):data.chunk(4)
        data.non_tensor_batch['prefix20_strict_em']=wire
        chunks=data.chunk(4);self.assertEqual([len(chunk) for chunk in chunks],[640]*4)
        restored=DataProto.concat(chunks)
        self.assertEqual(restored.non_tensor_batch['prefix20_strict_em'].tolist(),labels)
        for key,value in ids.items():self.assertEqual(restored.non_tensor_batch[key].tolist(),value.tolist())
        for key,value in tokens.items():self.assertTrue(torch.equal(restored.batch[key],value),key)

    def test_actual_dispatch_args_kwargs_preserve_balanced_labels_ids_and_native_lp_bytes(self):
        data,labels=fixture()
        order=torch.randperm(2560,generator=torch.Generator().manual_seed(1729))
        data.reorder(order)
        wire=strict_em_from_batch(data,AnswerTokenizer());data.non_tensor_batch['prefix20_strict_em']=wire
        expected_labels=[labels[int(i)] for i in order]
        self.assertEqual(wire.tolist(),expected_labels)
        tensors={k:v.clone() for k,v in data.batch.items()};metadata=copy.deepcopy(data.non_tensor_batch)
        expected_advantage=anchored_grpo(data.batch['token_level_rewards'],torch.ones(2560,2),
            data.non_tensor_batch['uid'],np.asarray(expected_labels,dtype=np.int8))
        actual_advantage=anchored_grpo(data.batch['token_level_rewards'],torch.ones(2560,2),
            data.non_tensor_batch['uid'],wire)
        for left,right in zip(expected_advantage,actual_advantage):torch.testing.assert_close(left,right,rtol=0,atol=0)
        args,kwargs=_split_args_kwargs_data_proto(4,data,same=data)
        for chunks in (args[0],kwargs['same']):
            self.assertEqual([len(chunk) for chunk in chunks],[640]*4)
            restored=DataProto.concat(chunks)
            for key,value in tensors.items():self.assertTrue(torch.equal(restored.batch[key],value),key)
            for key,value in metadata.items():
                self.assertEqual(restored.non_tensor_batch[key].dtype,object)
                self.assertEqual(restored.non_tensor_batch[key].tolist(),value.tolist())
            self.assertEqual(restored.meta_info,data.meta_info)
        correct=np.asarray(labels).reshape(512,5).sum(1)
        self.assertEqual((int(((correct>0)&(correct<5)).sum()),int((correct==5).sum()),int((correct==0).sum())),(273,120,119))


if __name__=='__main__':unittest.main()
