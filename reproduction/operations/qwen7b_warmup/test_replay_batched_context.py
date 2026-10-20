"""Tiny synthetic CPU causal/masking tests; no actual weights/features are loaded."""
import copy
from types import SimpleNamespace
import unittest
import torch
from . import replay_probe_layers as r
from .replay_batched_context import BatchedContextCapture, compare_replay_features, plan_context_batches


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.config=SimpleNamespace(num_hidden_layers=28,hidden_size=3,pad_token_id=99)
        self.calls=[];self.eval()
    def forward(self,input_ids,attention_mask,position_ids,use_cache,output_hidden_states,return_dict):
        assert not use_cache and output_hidden_states and return_dict and not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(),attention_mask.clone(),position_ids.clone()))
        tokens=input_ids.float()*attention_mask
        base=torch.stack((tokens,tokens.cumsum(1),position_ids.float()),dim=-1)
        return SimpleNamespace(hidden_states=tuple(base+i*.125 for i in range(29)))


def prefix(tid='q0-r0',zero=False):
    row=dict(trajectory_id=tid,qid='q0',actor_version='official7b-'+r.REVISION,
        initial_prompt_token_ids=[11,12,13],policy_token_ids=[21,22,31,32],
        policy_position_ids=[3,4,8,9],retained_policy_length=4,
        prefix_segments=[dict(turn_index=0,policy_start=0,policy_length=2,request_length=3,position_start=3,
            context_tokens=[11,12,13,21,22]),dict(turn_index=1,policy_start=2,policy_length=2,
            request_length=8,position_start=8,context_tokens=[11,12,13,21,22,90,91,92,31,32])],
        terminal=True,future_information_included=False)
    if zero:row.update(policy_token_ids=[],policy_position_ids=[],retained_policy_length=0,prefix_segments=[])
    return r.normalize_prefix(row)


class BatchedReplayTests(unittest.TestCase):
    def adapter(self,model=None,**kw):return BatchedContextCapture(model or TinyCausal(),expected_hidden_size=3,**kw)
    def test_stable_length_bucket_padded_budget_and_oversize_not_truncated(self):
        contexts=[[1]*n for n in (5,1,4,30,2)]
        batches=plan_context_batches(contexts,batch_size=2,max_context_tokens=8)
        self.assertEqual(batches,[[1,4],[2],[0],[3]])
        for batch in batches:
            self.assertTrue(len(batch)*max(len(contexts[i]) for i in batch)<=8 or len(batch)==1)
        cap=self.adapter(batch_size=2,max_context_tokens=8);values=cap.capture_many(contexts)
        self.assertEqual([v[26].shape[0] for v in values],[5,1,4,30,2])
        self.assertEqual(cap.last_capture['oversized_single_contexts'],1)
        self.assertEqual(cap.last_capture['contexts_truncated'],0)
        self.assertEqual([len(b) for b in plan_context_batches([[1]]*33,batch_size=32,max_context_tokens=16384)],[32,1])
        with self.assertRaises(ValueError):plan_context_batches([[1]],batch_size=33)

    def test_scalar_batch_actual_positions_padding_masks_and_order(self):
        contexts=[[7,8,9,10],[1],[3,4],[11,12,13]]
        scalar=self.adapter(batch_size=1);batch=self.adapter(batch_size=16)
        a=scalar.capture_many(contexts);b=batch.capture_many(contexts)
        for context,first,second in zip(contexts,a,b):
            for layer in r.LAYERS:
                self.assertTrue(torch.equal(first[layer],second[layer]))
                self.assertEqual(second[layer].dtype,torch.bfloat16)
                self.assertFalse(second[layer].requires_grad)
                self.assertEqual(second[layer].shape,(len(context),3))
                self.assertEqual(float(second[layer][0,0]),float(torch.tensor(context[0]+(layer+1)*.125).bfloat16()))
        self.assertLess(batch.total_model_calls,scalar.total_model_calls)
        ids,mask,positions=batch.model.calls[0]
        for row,length in enumerate((1,2,3,4)):
            self.assertEqual(mask[row].tolist(),[1]*length+[0]*(4-length))
            self.assertEqual(positions[row,:length].tolist(),list(range(length)))
            self.assertTrue(bool((ids[row,length:]==99).all()))

    def test_multi_segment_initial_question_only_and_zero_prefix_legacy_gather(self):
        rows=[prefix(),prefix('q0-r1',zero=True)];scalar=self.adapter(batch_size=1);batch=self.adapter(batch_size=16)
        expected=[r.replay_features(row,lambda c,p:scalar.capture_many([c],[p])[0],hidden_size=3) for row in rows]
        actual=batch.replay_many(rows);summary=compare_replay_features(expected,actual)
        for layer in r.LAYERS:
            self.assertTrue(torch.equal(expected[0][layer]['features'],actual[0][layer]['features']))
            self.assertEqual(actual[0][layer]['position_ids'][r.QMAX:r.QMAX+4].tolist(),[3,4,8,9])
            self.assertEqual(int(actual[1][layer]['attention_mask'].sum()),3)
            self.assertTrue(torch.equal(actual[1][layer]['features'][r.QMAX:],torch.zeros(r.PREFIX,3,dtype=torch.bfloat16)))
            self.assertEqual(summary['layers'][str(layer)]['all_retained']['max_absolute_error'],0.)
        self.assertEqual(batch.total_contexts,3)
        self.assertFalse(summary['native_bitexact_claim'])
        self.assertFalse(summary['scalar_batch_bitexact_claim'])

    def test_streaming_window_does_not_materialize_bank(self):
        seen=[]
        def source():
            for i in range(5):seen.append(i);yield prefix('t%d'%i,zero=True)
        adapter=self.adapter();iterator=adapter.iter_replay(source(),window_trajectories=2)
        first=next(iterator);self.assertEqual(first[0]['trajectory_id'],'t0');self.assertEqual(seen,[0,1])
        rest=list(iterator);self.assertEqual(len(rest),4);self.assertEqual(seen,list(range(5)))
        with self.assertRaises(ValueError):adapter.replay_many(prefix(str(i)) for i in range(65))

    def test_reject_changed_positions_noninteger_tokens_or_trainable_actor(self):
        cap=self.adapter()
        for contexts,positions in (([[1,2]],[[1,2]]),([[1]],[]),([[True]],None),([[]],None)):
            with self.assertRaises(ValueError):cap.capture_many(contexts,positions)
        cap.model.train()
        with self.assertRaises(ValueError):cap.capture_many([[1]])
        model=TinyCausal();model.anchor.requires_grad_(True)
        with self.assertRaises(ValueError):self.adapter(model)

    def test_compare_rejects_mask_or_position_drift(self):
        values=self.adapter().replay_many([prefix()]);wrong=copy.deepcopy(values)
        wrong[0][26]['position_ids'][0]=1
        with self.assertRaises(ValueError):compare_replay_features(values,wrong)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
