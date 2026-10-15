import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import final_control as c

class ArchiveBindingTests(unittest.TestCase):
    def test_legacy_route_unchanged_and_recovery_requires_full_failed_chain(self):
        with tempfile.TemporaryDirectory() as d, patch.object(c,'OWNER',__import__('os').getuid()):
            b=Path(d).resolve()/c.SIGMA3_BASENAME;(b/'preparation').mkdir(parents=True);(b/'failed-job-721449').mkdir()
            def write(name,value):(b/name).write_text(json.dumps(value))
            sha=c.SIGMA3_PRODUCER_SOURCE
            prior=dict(job='721449',returncode=0,commit=sha);claim=dict(commit=sha,command=['original'])
            write('preparation/training-submission.json',prior);write('preparation/training-submit-claim.json',claim)
            run=dict(source_commit=sha)
            self.assertEqual(c.producer_submission(b,run),(prior,claim,{}))
            log=b/'failed-job-721449/training.log';log.write_text('original failed storage log')
            failure=dict(accounting='721449|FAILED|1:0|14090|ACD1-56',original_submission=prior,original_claim=claim,actor_started=False,formal_outers=0,original_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
            write('failed-job-721449/failure-and-recovery.json',failure)
            rec=dict(job='721485',returncode=0,commit=sha,recovery_of='721449')
            rc=dict(commit=sha,recovery_of='721449',failure=failure)
            write('preparation/recovery721449-submit-claim.json',rc)
            with self.assertRaisesRegex(ValueError,'unknown'):c.producer_submission(b,run)
            write('preparation/recovery721449-submission.json',rec)
            with patch.object(c,'slurm_query',return_value=failure['accounting']):
                selected,_,chain=c.producer_submission(b,run)
                self.assertEqual(selected['job'],'721485');self.assertEqual(chain['original_failed_job'],'721449')
                log.write_text('changed')
                with self.assertRaisesRegex(ValueError,'chain differs'):c.producer_submission(b,run)

if __name__=='__main__':unittest.main()
