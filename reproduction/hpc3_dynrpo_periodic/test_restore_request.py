"""Focused metadata binding and fail-closed formal entry regression."""
import copy
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import unittest
from unittest import mock

from reproduction.hpc3_dynrpo_periodic import restore_request as restore


class RestoreRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.prefix = Path(self.temporary.name).resolve()/'run'
        self.source = Path(str(self.prefix)+'-r13')
        self.target = Path(str(self.prefix)+'-r14')
        self.commit = 'a'*40
        self.receipt_path = self.source/'artifacts/question-probe/pending-probe-pairs/global_step_5/source-receipt.json'
        self.receipt = dict(status=restore.PAIR_STATUS,kind='question_probe_pair',step=5,world_size=4,
                            run_id='source-run',source_base=str(self.source))
        self.write(self.receipt_path,self.receipt)
        self.write(self.source/'logs/run-manifest.json',dict(run_id='source-run',
            question_probe=dict(head_kind='qwen_native_periodic_lastblock_binary_v1')))
        self.write(self.source/'logs/native-allocation-admission.json',
                   dict(run_id='source-run',slurm_job_id='724530',accepted=True))
        self.write(self.source/'preparation/formal-submission.json',dict(job='724530'))
        self.write(self.source/'logs/cleanup-status.json',dict(run_id='source-run',cleaned=True))
        self.write(self.source/'logs/pipeline-status.json',dict(run_id='source-run',stage='FAILED'))
        self.write(self.target/'preparation/source-publication.json',dict(commit=self.commit,pushed_and_fetched=True))
        self.request = dict(schema_version=1,target_base=str(self.target),target_commit=self.commit,
            source_base=str(self.source),source_job='724530',source_run_id='source-run',
            source_receipt=str(self.receipt_path),source_receipt_sha256=restore.sha(self.receipt_path))
        self.write(self.target/'preparation/restore-request.json',self.request)
        self.patch = mock.patch.object(restore,'DEFAULT_BASE',self.prefix)
        self.patch.start()
        self.queue = mock.patch.object(restore.subprocess,'check_output',side_effect=self.slurm)
        self.queue.start()

    def tearDown(self):
        self.queue.stop(); self.patch.stop(); self.temporary.cleanup()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def slurm(self,argv,**kwargs):
        if argv[0]=='sacct':
            return '724530|FAILED|1:0|'+pwd.getpwuid(os.getuid()).pw_name+'|\n'
        return ''

    def test_metadata_bindings_and_terminal_source(self):
        request,receipt=restore.validate_request(self.target,self.commit)
        self.assertEqual(request,self.request);self.assertEqual(receipt['step'],5)
        for key,value in (('target_base',str(self.source)),('target_commit','b'*40),
                          ('source_run_id','other'),('source_job','99'),
                          ('source_receipt_sha256','0'*64)):
            changed=copy.deepcopy(self.request);changed[key]=value
            self.write(self.target/'preparation/restore-request.json',changed)
            with self.subTest(key=key),self.assertRaises(ValueError):
                restore.validate_request(self.target,self.commit)
        self.write(self.target/'preparation/restore-request.json',self.request)
        with mock.patch.object(restore.subprocess,'check_output',return_value='724530|RUNNING|0:0|mou176|'):
            with self.assertRaises(ValueError):restore.validate_request(self.target,self.commit)

    def test_full_restore_load_never_runs_on_login_or_reinitializes(self):
        with mock.patch.dict(os.environ,{},clear=True),mock.patch.object(restore.subprocess,'run') as launch:
            with self.assertRaisesRegex(ValueError,'512G_SLURM'):
                restore.prepare(self.target,self.commit)
            launch.assert_not_called()
        run=dict(source_commit=self.commit,run_id='source-run',full_state_restore=dict(
            source_receipt=self.request['source_receipt'],source_receipt_sha256=self.request['source_receipt_sha256'],
            source_base=str(self.source),completed_outer=5))
        self.write(self.target/'logs/run-manifest.json',run)
        with mock.patch.object(restore.subprocess,'run') as launch:
            self.assertEqual(restore.prepare(self.target,self.commit),'EXISTING_FULL_STATE_RESTORE_BINDING')
            launch.assert_not_called()
        run['full_state_restore']={}
        self.write(self.target/'logs/run-manifest.json',run)
        with self.assertRaises(ValueError):restore.prepare(self.target,self.commit)

    def test_bad_present_restore_request_cannot_fall_back_to_fresh_bind(self):
        envbin=self.target/'env/bin';envbin.mkdir(parents=True)
        source=self.target/'source'/self.commit;source.mkdir(parents=True)
        flock=envbin/'flock';flock.write_text('#!/bin/sh\nexit 0\n');flock.chmod(0o755)
        calls=self.target/'python-calls'
        python=envbin/'python'
        python.write_text('#!/bin/sh\necho "$*" >> '+str(calls)+'\n'
                          'case "$*" in *restore_request*) exit 21;; esac\nexit 0\n')
        python.chmod(0o755)
        entry=Path(restore.__file__).with_name('run.sh')
        result=subprocess.run(['bash',str(entry),'formal',str(self.target),self.commit],capture_output=True,text=True)
        self.assertEqual(result.returncode,21,result.stderr)
        text=calls.read_text()
        self.assertIn('restore_request',text)
        self.assertNotIn('hpc3_dynrpo_periodic.bind',text)
        self.assertNotIn('run_pipeline',text)


if __name__=='__main__':unittest.main()
