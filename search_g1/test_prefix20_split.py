"""Constructed CPU protocol fixtures; no physical rollout acceptance claim."""
import copy
import unittest
from search_g1.prefix20_split import make_turn,verify_turn,logical_sampling
from reproduction.common8 import behavior_trace as bt
from reproduction.common8.test_sampling_capture import stop_provenance_fixture,complete_sampling_fixture
from reproduction.common8.test_behavior_trace import tiny_outer_fixture


def split_fixture(tid='real',source=None):
    base=complete_sampling_fixture(dict(n=1,best_of=1,do_sample=True,temperature=1.,top_p=1.,top_k=-1,
        min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False,
        logprobs=1,min_tokens=0,logits_processors=[],logprob_semantics=bt.LOGPROB_SEMANTICS,max_tokens=20))
    prefix=[20]*19+[22]
    records=[]
    for i,(request,tokens,maximum,finish) in enumerate((([10,11],prefix,20,'length'),
            ([10,11]+prefix,[21,99],480,'stop'))):
        sampling=dict(base,max_tokens=maximum)
        row=dict(schema_version=1,request_id=f'{tid}-request{i}',row_id=tid,rollout_ordinal=0,
            rollout_call_id=f'{tid}-call{i}',actual_request_token_ids=request,
            raw_generated_token_ids=tokens,raw_generated_token_count=len(tokens),
            selected_logprobs=[-.2 if i==0 else -.3]*len(tokens),finish_reason=finish,stop_reason=None,
            eos_token_id=99,pad_token_id=0,sampling=sampling,policy_snapshot_id='cpu-policy0')
        row['effective_stop_provenance']=stop_provenance_fixture(row,source_sha256=source)
        records.append(row)
    fragments={20:'x',22:'<answer>A</ans',21:'wer>',99:'<EOS>'}
    decode=lambda ids:''.join(fragments[i] for i in ids)
    executor=lambda ids:''.join(fragments[i] for i in ids if i!=99)
    cut=bt.raw_prefix_to_first_closing_tag(prefix+[21,99],decode,eos_token_id=99,pad_token_id=0,
        special_token_ids=[0,99],executor_decode=executor,effective_stop_token_ids=[99])
    return make_turn(records,trajectory_id=tid,turn_index=0,special_token_ids=[0,99],prefix_cut=cut,final_turn=True)


class SplitProtocolTest(unittest.TestCase):
    def test_cross_boundary_tag_retains_native20_and480_samplers(self):
        turn=split_fixture();request,raw,lp,environment=verify_turn(turn)
        self.assertEqual(request,[10,11]);self.assertEqual(len(raw),22);self.assertEqual(len(lp),22)
        self.assertEqual(environment,[]);self.assertEqual(turn['kept_token_count'],21)
        self.assertEqual(turn['prefix_cut']['execution_text'],'x'*19+'<answer>A</answer>')
        self.assertEqual([c['sampling']['max_tokens'] for c in turn['native_chunks']],[20,480])
        self.assertIn('budget_semantics',turn['sampling'])
        self.assertNotIn('budget_semantics',turn['native_chunks'][0]['sampling'])

    def test_changed_suffix_and_terminal_restart_rejected(self):
        turn=split_fixture();turn['native_chunks'][1]['actual_request_token_ids'][2]=77
        with self.assertRaisesRegex(ValueError,'suffix_request_changed'):verify_turn(turn)
        turn=split_fixture();turn['native_chunks'][0]['finish_reason']='stop'
        with self.assertRaisesRegex(ValueError,'short_or_terminal_chunk'):verify_turn(turn)
        turn=split_fixture();turn['native_chunks'][1]['sampling']['max_tokens']=500
        with self.assertRaisesRegex(ValueError,'remaining_action_budget'):verify_turn(turn)

    def test_existing_complete_outer_gate_projects_every_prefix_token(self):
        trace,bindings=tiny_outer_fixture();source=bindings['expected_source_sha256']
        for trajectory in trace['trajectories']:
            tid=trajectory['trajectory_id'];turn=split_fixture(tid,source)
            trajectory['turns']=[turn]
        policy=trace['trajectories'][0]['turns'][0]['sampling']
        trace['sampling']=policy;bindings['expected_sampling']=policy
        trace['actor_rows']=[bt.make_actor_row(trajectory_id=t['trajectory_id'],
            input_ids=[0,10,11]+[20]*19+[22,21,0],attention_mask=[0]+[1]*23+[0],
            info_mask=[0]+[1]*23+[0],position_ids=[0]+list(range(23))+[0],prompt_width=3)
            for t in trace['trajectories']]
        projection=bt.project_sampling_logprobs(trace,**bindings)
        self.assertEqual(projection['audit']['included_token_count'],2560*21)
        self.assertEqual(projection['sampling_log_probs'][0],[-.2]*20+[-.3,0.])
        self.assertTrue(projection['audit']['selected_logprob_alignment_verified'])


if __name__=='__main__':unittest.main()
