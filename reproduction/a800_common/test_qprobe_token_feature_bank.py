"""Tiny CPU-only provider tests: no model, Torch, token arrays or remote access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.a800_common.qprobe_token_feature_bank import canonical_records,emit_bank,once


def text_sha(value):return hashlib.sha256(value.encode()).hexdigest()


class FakeProvider:
    def __init__(self,index):self.index=index;self.calls=[]
    def get_features(self,ids):self.calls.extend(ids);return ['reused:'+qid for qid in ids]


class FakeWriter:
    def __init__(self):self.ids=[];self.values=[];self.finished=False
    def append(self,ids,features):self.ids.extend(ids);self.values.extend(features)
    def finish(self):self.finished=True;return dict(count=len(self.ids))


class FeatureBankTest(unittest.TestCase):
    def setUp(self):
        self.rows=[('question '+str(i),'id'+str(i)) for i in range(5)]
        self.bindings=[dict(qid=qid,question_hash=text_sha(q),length=i+1,token_ids_sha256=text_sha('tokens'+qid)) for i,(q,qid) in enumerate(self.rows)]
        self.reused={row['qid']:dict(row,feature_sha256=text_sha(row['qid'])) for row in self.bindings if row['qid'] in ('id1','id4')}

    def test_canonical_order_and_only_missing_rows_encoded(self):
        records=canonical_records(self.rows,self.bindings,self.reused,text_sha)
        provider=FakeProvider(self.reused);writer=FakeWriter();encoded=[];events=[]
        def extract(questions,bindings):
            encoded.extend(questions)
            self.assertEqual([text_sha(q) for q in questions],[x['question_hash'] for x in bindings])
            return ['encoded:'+x['qid'] for x in bindings]
        manifest,counts=emit_bank(self.rows,records,provider,writer,extract,batch_size=2,progress=events.append)
        self.assertEqual(writer.ids,[qid for _,qid in self.rows])
        self.assertEqual(writer.values,['encoded:id0','reused:id1','encoded:id2','encoded:id3','reused:id4'])
        self.assertEqual(provider.calls,['id1','id4'])
        self.assertEqual(encoded,[self.rows[i][0] for i in (0,2,3)])
        self.assertEqual(counts,dict(rows=5,reused_rows=2,newly_encoded_rows=3))
        self.assertEqual(manifest,dict(count=5));self.assertTrue(writer.finished)
        self.assertEqual(events[-1]['completed_rows'],5)

    def test_rejects_bad_order_bindings_and_unknown_reuse(self):
        for field,value in (('qid','changed'),('question_hash','0'*64),('length',0),('token_ids_sha256','bad')):
            bindings=[dict(row) for row in self.bindings];bindings[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):canonical_records(self.rows,bindings,self.reused,text_sha)
        with self.assertRaises(ValueError):canonical_records(self.rows,self.bindings,dict(self.reused,unknown=self.bindings[0]),text_sha)
        with self.assertRaises(ValueError):canonical_records(self.rows[::-1],self.bindings,self.reused,text_sha)
        with self.assertRaises(ValueError):canonical_records(self.rows+[self.rows[0]],self.bindings+[self.bindings[0]],self.reused,text_sha)

    def test_extractor_failure_never_finishes_bank(self):
        provider=FakeProvider(self.reused);writer=FakeWriter()
        with self.assertRaises(ValueError):emit_bank(self.rows,self.bindings,provider,writer,lambda q,b:[],batch_size=2)
        self.assertFalse(writer.finished)

    def test_once_preserves_first_attempt_and_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'claim.json';once(path,dict(attempt=1))
            with self.assertRaises(FileExistsError):once(path,dict(attempt=2))
            self.assertEqual(json.loads(path.read_bytes()),dict(attempt=1))


if __name__=='__main__':unittest.main()
