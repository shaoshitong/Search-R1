import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import recover_sigma3_storage721449 as r

class RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'run';source=root/'root/source'/r.COMMIT;blocked=root/'blocked'
        (base/'logs').mkdir(parents=True);(base/'preparation').mkdir();source.mkdir(parents=True);blocked.mkdir()
        (source/'tiny.py').write_text('valid fixture')
        publish=root/'root/preparation';publish.mkdir()
        (publish/('source-'+r.COMMIT+'-verification.json')).write_text(json.dumps(dict(commit=r.COMMIT,status='PUBLISHED_SOURCE_FULL_SHA_PASS',files={'tiny.py':hashlib.sha256((source/'tiny.py').read_bytes()).hexdigest()})))
        expected=[str(source/'reproduction/hpc3_qprobe/train.sh'),str(base),r.COMMIT]
        for name,value in [('training-submission.json',dict(job='721449',returncode=0,commit=r.COMMIT)),('training-submit-claim.json',dict(commit=r.COMMIT,command=expected))]:
            (base/'preparation'/name).write_text(json.dumps(value))
        (base/'logs/training-721449.log').write_text('ValueError: PERSONAL_STORAGE_USAGE_NOT_VERIFIED: du: cannot read directory '+repr(str(blocked))+': Permission denied\n')
        return base,blocked

    def run_fixture(self,root,change=None):
        base,blocked=self.fixture(root)
        if change:change(base)
        def output(args,**kwargs):return '721449|FAILED|1:0|14090|ACD1-56\n' if args[0]=='sacct' else ''
        def command(args,**kwargs):
            return subprocess.CompletedProcess(args,0,'900001\n' if args[0]=='sbatch' and '--test-only' not in args else '123\t'+str(blocked)+'\n','')
        with patch.object(r,'BASE',base),patch.object(r,'ROOT',root/'root'),patch.object(r,'BLOCKED',blocked),patch.object(r.os,'getuid',return_value=14090),patch.object(r.subprocess,'check_output',side_effect=output),patch.object(r.subprocess,'run',side_effect=command) as execution:
            result=r.recover()
            self.assertEqual(result['job'],'900001')
            self.assertEqual(result['recovery_of'],'721449')
            with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):r.recover()
            return execution.call_args_list

    def test_only_once_after_bound_failure_and_readable_directory(self):
        with tempfile.TemporaryDirectory() as d:
            calls=self.run_fixture(Path(d))
            submits=[x for x in calls if x.args[0][0]=='sbatch' and '--test-only' not in x.args[0]]
            self.assertEqual(len(submits),1)
            self.assertIn('--gres=gpu:4',submits[0].args[0])

    def test_actor_manifest_forbids_fresh_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'ACTOR_WORK_ALREADY_EXISTS'):
                self.run_fixture(Path(d),lambda b:(b/'logs/run-manifest.json').write_text('{}'))

    def test_wrong_error_refuses_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'NOT_OBSERVED_STORAGE'):
                self.run_fixture(Path(d),lambda b:(b/'logs/training-721449.log').write_text('CUDA OOM'))

if __name__=='__main__':unittest.main()
