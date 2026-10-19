import unittest
from .core import aggregate, classify, digest, intervene, prepare_row, CF_CONFIG

M=dict(method='test',scale='3B',seed=1729,checkpoint_hash='ckpt',cf_config_sha256=digest(CF_CONFIG))

def row(ordinal=0, **kwargs):
    data=dict(qid='repeated-qid',data_source='nq',evaluation_ordinal=ordinal,question='Who?',gold=['Ann'],
        model_spans=['<think>Look it up.</think><search>Who?</search>', '<answer>Ann</answer>'],
        seen_docs=['Ann saw Anna. Annette said Ann.'],actual_retrieval_calls=1,source_shard_sha256='source')
    data.update(kwargs);return data

class StrictProtocol(unittest.TestCase):
    def test_following_vs_arbitrary(self):
        self.assertEqual(classify('Ann','Tokyo',['Tokyo'],['Ann']),'True')
        self.assertEqual(classify('Ann','Mars',['Tokyo'],['Ann']),'Amb')
        self.assertEqual(classify('Ann','Ann',['Tokyo'],['Ann']),'Hall')
        self.assertEqual(classify('Ann',None,['Tokyo'],['Ann']),'Amb')
        self.assertEqual(classify('Ann','I cannot answer',['Tokyo'],['Ann']),'Amb')

    def test_boundary_overlap_determinism(self):
        p=intervene(['Ann met Anna in New York and York.'],['Ann','New York','York'],['nq',0,'x'])
        self.assertEqual(p['replacement_count'],3)
        self.assertIn('Anna',p['modified_docs'][0])
        self.assertEqual(p,intervene(['Ann met Anna in New York and York.'],['Ann','New York','York'],['nq',0,'x']))
        self.assertEqual(intervene(['1999'],['19'],['nq',0,'x'])['status'],'UNPERTURBABLE')
        q=intervene(['2003 and 2,003'],['2003','2,003'],['nq',1,'y'])
        self.assertEqual(q['replacement_count'],2)
        self.assertNotIn(q['injected_value'],['2003','2,003'])

    def test_legal_final_answer_only(self):
        p=prepare_row(row(model_spans=['<search><answer>Ann</answer></search>']),M)
        self.assertEqual(p['original_em'],0)
        p=prepare_row(row(model_spans=['<answer>Ann</answer>'],seen_docs=[],actual_retrieval_calls=0),M)
        self.assertEqual(p['classification'],'Direct')

    def test_preserve_occurrences_and_config_separation(self):
        a=prepare_row(row(),M);b=prepare_row(row(1),M)
        self.assertNotEqual(a['request_id'],b['request_id'])
        self.assertNotEqual(a['request_id'],prepare_row(row(),dict(M,checkpoint_hash='other'))['request_id'])
        self.assertNotEqual(a['request_id'],prepare_row(row(),dict(M,cf_config_sha256='other'))['request_id'])
        # Cross-method qid serialization must not change the injected entity.
        c=prepare_row(row(qid='different-historical-serialization'),M)
        self.assertEqual(a['injected_value'],c['injected_value'])
        self.assertNotEqual(a['request_id'],c['request_id'])

    def test_aggregate_idempotent_global_counts(self):
        p=[prepare_row(row(i),M) for i in range(3)]
        p.append(prepare_row(row(3,model_spans=['<answer>Ann</answer>'],seen_docs=[],actual_retrieval_calls=0),M))
        r=[]
        for a,answer in zip(p,[p[0]['injected_value'],'Ann','unrelated']):
            r.append(dict(request_id=a['request_id'],checkpoint_hash='ckpt',cf_config_sha256=M['cf_config_sha256'],
                source_record_hash=a['source_record_hash'],status='CLASSIFIED',cf_answer=answer,
                classification=classify(a['original_answer'],answer,a['cf_aliases'],a['gold'])))
        s=aggregate(p,r+r)
        self.assertAlmostEqual(s['TC'],(1+1+.3+.6)/4)
        self.assertAlmostEqual(s['D_Inv'],1/3)
        self.assertEqual(s['status'],'complete')
        # Network/OOM errors are not Amb; incomplete coverage is provisional.
        r[0]=dict(r[0],status='INFERENCE_ERROR')
        s=aggregate(p,r)
        self.assertEqual(s['counts']['Amb'],1)
        self.assertEqual(s['coverage']['pending_or_error'],1)
        self.assertEqual(s['status'],'provisional')

    def test_zero_denominator_unperturbable_and_missing_source(self):
        s=aggregate([],[])
        self.assertIsNone(s['TC']);self.assertIsNone(s['D_Inv'])
        p=prepare_row(row(seen_docs=['Anna']),M)
        self.assertEqual(p['status'],'UNPERTURBABLE')
        s=aggregate([p],[]);self.assertEqual(s['TC_denominator'],0)
        del p['question']
        self.assertEqual(prepare_row(p,M)['status'],'SOURCE_INCOMPLETE')

if __name__=='__main__':unittest.main()
