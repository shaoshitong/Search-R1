"""CPU contracts for real native chunk continuity, not a substitute GPU gate."""
import copy
import unittest
from types import SimpleNamespace

from reproduction.prefix_selection200.rollout import (
    BOUNDARY, CONTRACT, append_chunk, finish_action, first_action_cut,
    next_request, prefix_record, registered_sampling, seed_for, start_action, stream_action_present, validate_selection,
)


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    def __call__(self, text, **_kwargs):
        return {'input_ids': [ord(char) for char in text]}
    def encode(self,text,**_kwargs):
        return [ord(char) for char in text]


def decode(tokens):
    return ''.join(chr(token) for token in tokens if token)


def state():
    return dict(run_id='run', outer=1, actor_version='actor1', trajectory_id='trajectory',
                candidate_ordinal=0, response_index=0,
                group=dict(qid='question', question_hash='qhash'),
                prompt=[65, 66], response=[], info_mask=[], log_probs=[], turns=[],
                ordinal=0, chunks_started=0, action=None, retrieval_calls=0,
                terminal=False, terminal_reason=None, sampling_contract=CONTRACT, action_boundary=BOUNDARY)


BUDGET = dict(total_tokens=4096, action_tokens=500, answer_reserve_tokens=500)


def record(s, request, tokens, finish_reason='length', aborted=False):
    return append_chunk(s, request, tokens, [-.125] * len(tokens),
                        dict(max_tokens=request['max_tokens'], seed=request['seed'], logprobs=1,
                             temperature=1.,top_p=1.,top_k=-1,n=1),
                        finish_reason=finish_reason, stop_reason=None, terminal_ids=[0],
                        decode=decode, action_boundary_abort=aborted)


def full_record_fixture(s):
    """The actual persisted actor fields from a completed native state fixture."""
    return dict(actor_version=s['actor_version'],policy_snapshot_id=s['actor_version'],
        sampling_contract=CONTRACT,initial_prompt_token_ids=s['prompt'],
        response_token_ids=s['response'],response_info_mask=s['info_mask'],
        response_log_probs=s['log_probs'],strict_em=1,behavior_turns=s['turns'],
        sampling_journal=[chunk for turn in s['turns'] for chunk in turn['actual_sampling_chunks']],
        reward_model={'ground_truth':{'target':['ok']},'style':'rule'},data_source='nq')


def assert_actor_fields(test, row):
    """Always check causal assembly; use real actor adapter when integrated.

    This rollout-only branch intentionally does not own actor.py. Cherry-pick
    into the integration branch automatically adds its real CPU validator to
    these exact same native-state fixtures, with no absolute external path.
    """
    try:
        from reproduction.prefix_selection200.actor import validate_record
    except ModuleNotFoundError as error:
        if error.name != 'reproduction.prefix_selection200.actor':
            raise
    else:
        test.assertIs(validate_record(row,row['actor_version']),row)
    response,mask,lp,journal=[],[],[],[]
    for turn in row['behavior_turns']:
        request=turn['actual_request_token_ids']
        test.assertEqual(request,row['initial_prompt_token_ids']+response)
        raw,rawlp=[],[]
        for chunk in turn['actual_sampling_chunks']:
            test.assertEqual(chunk['actual_request_token_ids'],request+raw)
            raw.extend(chunk['raw_generated_token_ids']);rawlp.extend(chunk['selected_logprobs'])
            journal.append(chunk)
        n=turn['kept_token_count']
        observation=turn['observation_token_ids']+turn['correction_token_ids']
        test.assertEqual(raw,turn['raw_generated_token_ids'])
        response.extend(raw[:n]+observation);mask.extend([1]*n+[0]*len(observation))
        lp.extend(rawlp[:n]+[0.]*len(observation))
    test.assertEqual(response,row['response_token_ids'])
    test.assertEqual(mask,row['response_info_mask'])
    test.assertEqual(lp,row['response_log_probs'])
    test.assertEqual(journal,row['sampling_journal'])


class NativePrefixContinuityTest(unittest.TestCase):
    def test_prefix_quota_is_pause_and_continuation_retains_ids_lp(self):
        s=state(); start_action(s,BUDGET)
        first=next_request(s,'prefix',{})
        original=list(range(97,117))
        self.assertEqual(first['max_tokens'],20)
        self.assertIsNone(record(s,first,original))
        before=prefix_record(s)
        # Simulate phase serialization/reload and a different TP1 owner GPU.
        resumed=copy.deepcopy(s)
        second=next_request(resumed,'continue',{})
        self.assertEqual(second['max_tokens'],480)
        self.assertEqual(second['actual_request_token_ids'],[65,66]+original)
        self.assertNotEqual(first['seed'],second['seed'])
        ending=[ord(c) for c in '<answer>ok</answer>']
        cut=record(resumed,second,ending,finish_reason=None,aborted=True)
        finish_action(resumed,cut,('',True,1,0),CharacterTokenizer(),BUDGET)
        self.assertTrue(resumed['terminal'])
        self.assertEqual(resumed['response'][:20],original)
        self.assertEqual(resumed['log_probs'],[-.125]*len(resumed['response']))
        self.assertEqual(prefix_record(resumed)['policy_token_ids'],before['policy_token_ids'])
        chunks=resumed['turns'][0]['actual_sampling_chunks']
        self.assertEqual([c['sampling']['max_tokens'] for c in chunks],[20,480])
        self.assertEqual(chunks[1]['finish_reason'],None)
        self.assertTrue(chunks[1]['native_action_boundary_abort'])
        assert_actor_fields(self,full_record_fixture(resumed))

    def test_same_prefix_cross_tool18_plus2_then_native498_actor_packet(self):
        s=state();start_action(s,BUDGET)
        first=next_request(s,'prefix',{})
        cut=record(s,first,[ord(c) for c in '<search>x</search>'],None,True)
        finish_action(s,cut,('<information>tool</information>',False,1,1),CharacterTokenizer(),BUDGET)
        start_action(s,BUDGET)
        second=next_request(s,'prefix',{})
        self.assertEqual(second['max_tokens'],2)
        self.assertIsNone(record(s,second,[97,98]))
        scored=prefix_record(s)
        continued=copy.deepcopy(s)
        third=next_request(continued,'continue',{})
        self.assertEqual(third['max_tokens'],498)
        self.assertEqual(third['actual_request_token_ids'],second['actual_request_token_ids']+[97,98])
        cut=record(continued,third,[ord(c) for c in '<answer>ok</answer>'],None,True)
        finish_action(continued,cut,('',True,1,0),CharacterTokenizer(),BUDGET)
        after=prefix_record(continued)
        self.assertEqual(after['policy_token_ids'],scored['policy_token_ids'])
        self.assertEqual(after['policy_position_ids'],scored['policy_position_ids'])
        self.assertEqual([c['sampling']['max_tokens'] for t in continued['turns']
                          for c in t['actual_sampling_chunks']],[20,2,498])
        assert_actor_fields(self,full_record_fixture(continued))

    def test_first_action_is_first_regex_match_not_future_search_priority(self):
        tokens=[ord(c) for c in '<answer>A</answer><search>future</search>']
        cut=first_action_cut(tokens,decode)
        self.assertEqual(cut['action'],'answer')
        self.assertEqual(decode(cut['kept_token_ids']),'<answer>A</answer>')
        self.assertEqual(cut['execution_text'],'<answer>A</answer>')
        self.assertGreater(cut['discarded_token_count'],0)

    def test_token_straddling_close_tag_is_not_retokenized(self):
        vocab={1:'<answer>',2:'x',3:'</answer>suffix',4:'future'}
        cut=first_action_cut([1,2,3,4],lambda ids:''.join(vocab[i] for i in ids))
        self.assertEqual(cut['kept_token_ids'],[1,2,3])
        self.assertEqual(cut['execution_text'],'<answer>x</answer>')

    def test_stream_close_tag_crosses_native_chunk_boundary(self):
        partial=[ord(c) for c in 'a'*100+'<search>query</se']
        self.assertFalse(stream_action_present(partial,decode))
        self.assertTrue(stream_action_present(partial+[ord(c) for c in 'arch>'],decode))

    def test_sampling_metadata_reads_registered_clone_not_only_caller(self):
        caller=SimpleNamespace(n=1,temperature=1.,max_tokens=20,seed=9,stop_token_ids=[1])
        actual=SimpleNamespace(n=1,temperature=1.,max_tokens=20,seed=9,
                               stop_token_ids=[1,2],all_stop_token_ids={1,2,3})
        request=dict(request_id='r',actual_request_token_ids=[20,21])
        group=SimpleNamespace(request_id='r',prompt_token_ids=[20,21],sampling_params=actual)
        engine=SimpleNamespace(scheduler=[SimpleNamespace(waiting=[group],running=[],swapped=[])])
        evidence=registered_sampling(engine,request,caller)
        self.assertEqual(evidence['stop_token_ids'],[1,2])
        self.assertEqual(evidence['caller_sampling']['stop_token_ids'],[1])
        self.assertEqual(evidence['effective_all_stop_token_ids'],[1,2,3])

    def test_tool_context_not_counted_as_policy_and_positions_are_original(self):
        s=state();start_action(s,BUDGET)
        request=next_request(s,'prefix',{})
        tokens=[ord(c) for c in '<search>x</search>']
        cut=record(s,request,tokens,finish_reason=None,aborted=True)
        finish_action(s,cut,('\n<information>observation</information>\n',False,1,1),CharacterTokenizer(),BUDGET)
        self.assertFalse(s['terminal'])
        self.assertEqual(s['ordinal'],1)
        self.assertEqual(sum(s['info_mask']),len(tokens))
        self.assertGreater(len(s['response']),len(tokens))
        self.assertTrue(all(p==0 for p,m in zip(s['log_probs'],s['info_mask']) if not m))
        start_action(s,BUDGET)
        request2=next_request(s,'prefix',{})
        self.assertEqual(request2['max_tokens'],20-len(tokens))
        self.assertIsNone(record(s,request2,[ord('a')]*request2['max_tokens']))
        row=prefix_record(s)
        self.assertEqual(len(row['policy_token_ids']),20)
        self.assertEqual(len(row['prefix_segments']),2)
        second=row['prefix_segments'][1]
        self.assertEqual(second['position_start'],len(s['prompt'])+len(s['response']))
        self.assertEqual(second['context_tokens'][:second['request_length']],s['prompt']+s['response'])
        self.assertNotIn('strict_em',row)
        self.assertNotIn('reward_model',row)

    def test_actual_first_token_eos_is_legal_zero_prefix_without_resampling(self):
        s=state();start_action(s,BUDGET);request=next_request(s,'prefix',{})
        cut=record(s,request,[0],finish_reason='stop')
        finish_action(s,cut,('',True,0,0),CharacterTokenizer(),BUDGET)
        row=prefix_record(s)
        self.assertEqual(row['policy_token_ids'],[])
        self.assertEqual(row['prefix_segments'],[])
        self.assertEqual(row['initial_prompt_token_ids'],[65,66])
        self.assertEqual(s['chunks_started'],1)
        self.assertEqual(s['turns'][0]['selected_logprobs'],[-.125])
        self.assertEqual(s['terminal_reason'],'natural_eos')

    def test_native_eos_after_short_prefix_is_actual_terminal(self):
        s=state();start_action(s,BUDGET);request=next_request(s,'prefix',{})
        cut=record(s,request,[97,98,0],finish_reason='stop')
        finish_action(s,cut,('',True,0,0),CharacterTokenizer(),BUDGET)
        self.assertEqual(prefix_record(s)['policy_token_ids'],[97,98])
        self.assertEqual(s['log_probs'],[-.125,-.125])

    def test_native_seed_distinct_per_trajectory_turn_chunk_and_reproducible(self):
        self.assertEqual(seed_for('run',1,'t',0,0),seed_for('run',1,'t',0,0))
        values={seed_for('run',1,t,turn,chunk) for t in ('a','b') for turn in range(2) for chunk in range(2)}
        self.assertEqual(len(values),8)

    def test_missing_actual_probability_is_rejected_without_hf_recomputation(self):
        s=state();start_action(s,BUDGET);request=next_request(s,'prefix',{})
        with self.assertRaisesRegex(ValueError,'log probabilities'):
            append_chunk(s,request,[97],[],{},finish_reason='stop',stop_reason=None,terminal_ids=[0],decode=decode)

    def test_selected_original_g5_and_actor_scan_binding(self):
        receipt=dict(run_id='run',outer=1,actor_version='actor',scan_sha256='sha')
        selection=dict(run_id='run',outer=1,actor_version='actor',prefix_scan_sha256='sha',selected=[
            dict(qid='q',candidate_ordinal=3,trajectory_ids=['t'+str(j) for j in range(5)])])
        self.assertEqual(len(validate_selection(selection,receipt,expected_questions=1)),1)
        wrong=copy.deepcopy(selection);wrong['selected'][0]['trajectory_ids'][4]='t0'
        with self.assertRaisesRegex(ValueError,'incomplete'):
            validate_selection(wrong,receipt,expected_questions=1)
        wrong=copy.deepcopy(selection);wrong['actor_version']='newactor'
        with self.assertRaisesRegex(ValueError,'bound'):
            validate_selection(wrong,receipt,expected_questions=1)


if __name__=='__main__':unittest.main()
