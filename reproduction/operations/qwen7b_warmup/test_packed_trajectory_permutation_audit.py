"""Small CPU token/identity tests; no actor, checkpoint, GPU or remote access."""
import copy
import unittest
from .packed_variance_probe import pack_group
from .packed_trajectory_permutation_audit import ORDERS, SEPARATOR, input_identity, reorder

class PermutationInputTest(unittest.TestCase):
    def setUp(self):
        self.group=dict(qid='q',trajectory_ids=['t%d'%i for i in range(5)],labels=[0,1,1,0,0])
        self.records={t:dict(qid='q',trajectory_id=t,prompt=[11,12,13],policy20=[100+i]*[20,2,7,20,1][i]) for i,t in enumerate(self.group['trajectory_ids'])}

    def test_own_id_label_token_mapping_and_variance_only_order_changes(self):
        source=copy.deepcopy((self.group,self.records)); expected=dict(zip(self.group['trajectory_ids'],self.group['labels'])); identities=[]
        for order in ORDERS:
            view=reorder(self.group,order); self.assertEqual(dict(zip(view['trajectory_ids'],view['labels'])),expected)
            self.assertEqual(view['target_variance'],.24); self.assertEqual(view['target4variance'],.96); self.assertTrue(view['true_mixed'])
            identities.append(input_identity(view,self.records,pack_group))
        self.assertEqual(len({x['unordered_own_ID_EM_tokens'] for x in identities}),1)
        self.assertEqual(len({x['packed_input_sha256'] for x in identities}),3)
        self.assertEqual((self.group,self.records),source)

    def test_question_once_exact_policy_order_and_short_padding_mask(self):
        for order in ORDERS:
            view=reorder(self.group,order); packed=pack_group(view,self.records,SEPARATOR)
            self.assertEqual(packed['input_ids'][:3],[11,12,13]); self.assertEqual(packed['token_types'].count(0),3)
            self.assertEqual(len(packed['input_ids']),3+5*(4+20))
            for i,tid in enumerate(view['trajectory_ids']):
                begin=3+i*24; policy=self.records[tid]['policy20']; n=len(policy)
                self.assertEqual(packed['input_ids'][begin:begin+4],SEPARATOR)
                self.assertEqual(packed['input_ids'][begin+4:begin+4+n],policy)
                self.assertEqual(packed['attention_mask'][begin+4:begin+24],[True]*n+[False]*(20-n))
                self.assertEqual(packed['position_ids'][begin+4+n:begin+24],[0]*(20-n))

    def test_labels_do_not_enter_packed_inputs_and_duplicate_id_rejected(self):
        original=reorder(self.group,'original'); changed=reorder(dict(self.group,labels=[1-y for y in self.group['labels']]),'original')
        self.assertEqual(pack_group(original,self.records,SEPARATOR),pack_group(changed,self.records,SEPARATOR))
        bad=dict(self.group,trajectory_ids=['t0']*5)
        with self.assertRaises(AssertionError): reorder(bad,'reverse')

if __name__=='__main__': unittest.main()
