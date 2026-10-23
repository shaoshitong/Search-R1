"""CPU path/header checks; no Slurm, HTTP, GPU or model loading."""
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.dynrpo7b.train import validate_retriever_scope


class RestoreRetrieverScopeTests(unittest.TestCase):
    def test_fresh_local_and_bound_restore_original_producer(self):
        base=Path('/data/user/jxu957/experiments/dynrpo7b-formal-new')
        old=Path('/data/user/jxu957/experiments/dynrpo7b-formal-old')
        local=dict(service_ready_path=str(base/'retrieval/logs/service-ready.json'),service_job_id='11')
        original=dict(service_ready_path=str(old/'retrieval/logs/service-ready.json'),service_job_id='12',
                      service_ready_sha256='a'*64,endpoint='http://original-node:18126/retrieve')
        self.assertIs(validate_retriever_scope(base,local),local)
        with self.assertRaises(ValueError):validate_retriever_scope(base,original)
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'boundary.json'
            header=dict(kind='dynrpo7b_native4',status='DYNRPO7B_ROLLING_READY',source_base=str(old))
            receipt.write_text(json.dumps(header))
            self.assertIs(validate_retriever_scope(base,original,receipt),original)
            self.assertEqual(original['service_job_id'],'12')
            self.assertEqual(original['service_ready_path'],str(old/'retrieval/logs/service-ready.json'))
            for change in (dict(kind='other'),dict(status='not_committed'),dict(source_base=str(old.parent/'foreign'))):
                receipt.write_text(json.dumps(dict(header,**change)))
                with self.assertRaises(ValueError):validate_retriever_scope(base,original,receipt)

    def test_second_resume_keeps_parent_actual_producer_and_rejects_foreign(self):
        original=dict(service_ready_path='/data/original/retrieval/logs/service-ready.json',
                      service_job_id='12',service_ready_sha256='a'*64,
                      endpoint='http://original-node:18126/retrieve')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=root/'R1'
            (first/'logs').mkdir(parents=True)
            (first/'logs/run-manifest.json').write_text(json.dumps(dict(external_retriever=original)))
            first_receipt=root/'original-boundary.json'
            first_receipt.write_text(json.dumps(dict(kind='dynrpo7b_native4',status='DYNRPO7B_ROLLING_READY',
                                                     source_base='/data/original')))
            self.assertIs(validate_retriever_scope(first,original,first_receipt),original)
            second_receipt=root/'R1-boundary.json'
            second_receipt.write_text(json.dumps(dict(kind='dynrpo7b_native4',status='DYNRPO7B_ROLLING_READY',
                                                      source_base=str(first))))
            self.assertIs(validate_retriever_scope(root/'R2',original,second_receipt),original)
            for change in (dict(service_job_id='99'),dict(service_ready_sha256='b'*64),
                           dict(service_ready_path='/data/foreign/retrieval/logs/service-ready.json'),
                           dict(endpoint='http://foreign-node:18126/retrieve')):
                with self.assertRaises(ValueError):
                    validate_retriever_scope(root/'R2',dict(original,**change),second_receipt)


if __name__=='__main__':
    unittest.main()
