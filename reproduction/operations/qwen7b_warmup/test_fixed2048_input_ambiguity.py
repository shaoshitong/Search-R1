"""Tiny metadata/math checks only; no model/tensor/data loads."""
import unittest
from .audit_fixed2048_input_ambiguity import input_keys,grouping,metrics,audit_score

class AmbiguityTest(unittest.TestCase):
    def test_token_match_is_not_feature_match(self):
        row=dict(question_length=3,policy_length=2,policy_positions=[3,9],policy_token_sha256='a'*64,retained_feature_sha256='b'*64)
        m=dict(capture_layer=26,dimension=3584,question_max_tokens=273,prefix_tokens=20,model_revision='official')
        token,strict=input_keys(row,m)
        token2,strict2=input_keys(dict(row,retained_feature_sha256='c'*64),m)
        self.assertEqual(token,token2);self.assertNotEqual(strict,strict2)
        self.assertNotEqual(strict,input_keys(dict(row,policy_positions=[3,10]),m)[1])
        self.assertNotEqual(strict,input_keys(dict(row,question_length=4),m)[1])
    def test_duplicate_conflicts_and_metrics(self):
        g=grouping(['a','a','b','b','c'],[0,1,0,0,1]);self.assertEqual(g['unique_inputs'],3);self.assertEqual(g['duplicate_pairs'],2);self.assertEqual(g['opposite_EM_identical_key_pairs'],1)
        rows=[dict(qid='a',strict_em=[0]*5,predicted_mixed=.1),dict(qid='b',strict_em=[0,0,0,0,1],predicted_mixed=.9),dict(qid='c',strict_em=[1]*5,predicted_mixed=.2)]
        v=metrics(rows,{'a','b'});self.assertEqual(v['MixedBA'],1);self.assertEqual(v['AUC'],1);self.assertEqual(v['globalTop512']['K_histogram'],[1,1,0,0,0,0])
    def test_selected_ids_are_not_reselected_by_stratum_labels(self):
        rows=[dict(qid=str(i),strict_em=[0,0,0,0,i%2],predicted_mixed=i/600)for i in range(600)]
        def selected(v):return {r['qid']for r in sorted(v,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]}
        original=selected(rows);changed=[dict(r,strict_em=[1-y for y in r['strict_em']])for r in rows];self.assertEqual(original,selected(changed))
        fp={r['qid']:{f:grouping([1]*5,r['strict_em'])for f in('tokens_positions','strict_features_masks_positions')}for r in rows};out=audit_score(rows,fp)
        self.assertEqual(out['strata']['tokens_positions']['unique1']['globalTop512']['selected_questions'],512)
        self.assertFalse(out['strata']['tokens_positions']['unique1']['per_stratum_reselection_performed'])

if __name__=='__main__':unittest.main()
