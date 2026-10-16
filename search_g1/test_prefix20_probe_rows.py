import copy
import unittest
from search_g1.prefix20_probe_worker import validate_rows


class ProbeRowsTest(unittest.TestCase):
    def row(self):
        return dict(trajectory_id='real1',initial_prompt_token_ids=[2,3,4],
                    policy_token_ids=[5,6],policy_position_ids=[3,4],
                    prefix_segments=[dict(turn_index=0,policy_start=0,policy_length=2,
                        position_start=3,request_length=3,context_tokens=[2,3,4,5,6])])

    def test_real_context_and_labels_are_separate(self):
        row=self.row();validate_rows([row],labels_required=False)
        row['strict_em']=1
        with self.assertRaisesRegex(ValueError,'MUST_NOT_RECEIVE_FINAL_LABELS'):
            validate_rows([row],labels_required=False)
        validate_rows([row],labels_required=True)

    def test_future_token_or_other_questions_prefix_rejected(self):
        row=self.row();row['prefix_segments'][0]['context_tokens'].append(99)
        with self.assertRaisesRegex(ValueError,'OWN_CAUSAL_REQUEST'):validate_rows([row],labels_required=False)
        row=self.row();row['initial_prompt_token_ids']=[2,9,4]
        with self.assertRaisesRegex(ValueError,'INITIAL_QUESTION_CONTEXT_CHANGED'):validate_rows([row],labels_required=False)
        row=self.row();row['policy_position_ids']=[4,5]
        with self.assertRaisesRegex(ValueError,'ROPE_POSITION'):validate_rows([row],labels_required=False)


if __name__=='__main__':unittest.main()
