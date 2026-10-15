"""Focused CPU fixtures for no-repeat pool cycling and true prefix provenance."""
import unittest

from .pool import initialize_records, next_selection, identity
from .rollout import prompt_tokens, compact_prefix


def source(index, content=None):
    return dict(prompt=[dict(role='user', content=content or ('Question: %d' % index))],
                data_source='fixture', extra_info=dict(index=index),
                reward_model=dict(ground_truth=dict(target=[str(index)])))


def cached(row, split):
    binding=identity(row)
    return dict(qid=binding['qid'], question_hash=binding['question_hash'], split=split)


class PoolTest(unittest.TestCase):
    def test_cached_first_epoch_consumes_pool_and_fixed_diagnostic_stays_excluded(self):
        rows=[source(i) for i in range(9)]
        # Canonical duplicate content must choose the actual cached member.
        rows.append(source(9, 'Question: 0'))
        old=[cached(rows[0],'train'),cached(rows[1],'train'),
             cached(rows[7],'test'),cached(rows[8],'test')]
        pool,state=initialize_records(rows,old,train_sha256='fixture',
            seed=11,epoch_questions=2,diagnostic_questions=2)
        self.assertEqual(len(pool['records']),7)
        self.assertEqual(state['cursor'],2)
        teacher=set(pool['epoch1_cached_qids']); firstcycle=set(teacher)
        for epoch in range(2,20):
            selection,state=next_selection(pool,state,epoch)
            qids=[row['qid'] for row in selection['rows']]
            self.assertEqual(len(qids),len(set(qids)))
            self.assertFalse(set(qids)&{old[-2]['qid'],old[-1]['qid']})
            for row in selection['rows']:
                if row['pool_cycle']==0:
                    self.assertNotIn(row['qid'],firstcycle)
                    firstcycle.add(row['qid'])
        self.assertEqual(len(firstcycle),7)

    def test_each_complete_cycle_visits_all_questions_once_including_boundary_epoch(self):
        rows=[source(i) for i in range(9)]
        old=[cached(rows[0],'train'),cached(rows[1],'train'),
             cached(rows[7],'test'),cached(rows[8],'test')]
        pool,state=initialize_records(rows,old,train_sha256='fixture',
            seed=11,epoch_questions=2,diagnostic_questions=2)
        cycles={0:list(pool['epoch1_cached_qids'])}
        for epoch in range(2,30):
            selection,state=next_selection(pool,state,epoch)
            for row in selection['rows']:
                cycles.setdefault(row['pool_cycle'],[]).append(row['qid'])
        for cycle,ids in cycles.items():
            self.assertEqual(len(ids),len(set(ids)))
            if cycle<state['cycle']:
                self.assertEqual(len(ids),len(pool['records']))

    def test_sequence_cannot_skip_epochs_or_rewind(self):
        rows=[source(i) for i in range(9)]
        old=[cached(rows[0],'train'),cached(rows[1],'train'),
             cached(rows[7],'test'),cached(rows[8],'test')]
        pool,state=initialize_records(rows,old,train_sha256='fixture',
            seed=11,epoch_questions=2,diagnostic_questions=2)
        with self.assertRaises(ValueError): next_selection(pool,state,3)
        _,after=next_selection(pool,state,2)
        with self.assertRaises(ValueError): next_selection(pool,after,2)

    def test_official_chat_prefill_keeps_question_context(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages;self.kwargs=kwargs
                return 'actual_question_prompt'
            def __call__(self,text,**kwargs):
                self.text=text;self.encode_kwargs=kwargs
                return {'input_ids':[1,2,3,4]}
        tokenizer=Tokenizer(); messages=[dict(role='user',content='Question stays in causal context')]
        self.assertEqual(prompt_tokens(tokenizer,messages),[1,2,3,4])
        self.assertEqual(tokenizer.messages,messages)
        self.assertTrue(tokenizer.kwargs['add_generation_prompt'])
        self.assertFalse(tokenizer.encode_kwargs['add_special_tokens'])

    def test_prefix_uses_only_retained_generated_tokens_across_same_trajectory(self):
        tid='test-trajectory'
        turns=[dict(trajectory_id=tid,turn_index=0,actual_request_token_ids=[100,101],
                    raw_generated_token_ids=list(range(30)),raw_generated_token_count=30,
                    kept_token_count=12,prefix_cut=dict(kept_token_ids=list(range(12)))),
               dict(trajectory_id=tid,turn_index=1,actual_request_token_ids=[100,101]+list(range(12))+[900,901],
                    raw_generated_token_ids=list(range(40,55)),raw_generated_token_count=15,
                    kept_token_count=15,prefix_cut=dict(kept_token_ids=list(range(40,55))))]
        segments,tokens,positions=compact_prefix(turns,tid,20)
        self.assertEqual(tokens,list(range(12))+list(range(40,48)))
        self.assertEqual(len(tokens),20)
        self.assertEqual(segments[1]['context_tokens'][-8:],list(range(40,48)))
        self.assertIn(900,segments[1]['context_tokens'])
        self.assertEqual(positions,list(range(2,14))+list(range(16,24)))
        self.assertNotIn(20,tokens)


if __name__=='__main__': unittest.main()
