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
                            attempt_progress=dict(cumulative_writes=100),
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
        (self.source/'logs/actual-updates.jsonl').write_text(json.dumps(dict(
            step=5,update_status='complete',full_budget_complete=True))+'\n')
        for rank in range(4):
            (self.source/'logs'/('optimizer-rank-'+str(rank)+'.jsonl')).write_text(json.dumps(dict(
                kind='OPTIMIZER_WRITE',rank=rank,run_id='source-run',write_count=100))+'\n')
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

    def test_cleanup_failure_requires_actual_original_node_reconciliation(self):
        self.write(self.source/'logs/pipeline-status.json',
                   dict(run_id='source-run',stage='OWNED_CLEANUP_FAILED'))
        with self.assertRaisesRegex(ValueError,'ORIGINAL_NODE_CLEANUP_UNPROVEN'):
            restore.validate_request(self.target,self.commit)
        self.write(self.source/'logs/native-allocation-admission.json',
                   dict(run_id='source-run',slurm_job_id='724530',accepted=True,node='ACD1-19'))
        proof=dict(run_id='source-run',cleaned=True,failed_job='724530',
                   original_allocation_hostname_verified=True,all_owned_sessions_absent=True,
                   recorded_cuda_contexts_absent=True,execution_hostname='ACD1-19',cleanup_job_id='724800')
        self.write(self.source/'logs/cleanup-status.json',proof)
        self.assertEqual(restore.validate_request(self.target,self.commit)[1]['step'],5)
        for key,value in (('failed_job','99'),('execution_hostname','ACD1-20'),
                          ('cleanup_job_id','724530'),('all_owned_sessions_absent',False),
                          ('recorded_cuda_contexts_absent',False),
                          ('original_allocation_hostname_verified',False)):
            self.write(self.source/'logs/cleanup-status.json',dict(proof,**{key:value}))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'ORIGINAL_NODE_CLEANUP_UNPROVEN'):
                restore.validate_request(self.target,self.commit)

    def test_newer_real_updates_cannot_be_discarded_without_bound_user_approval(self):
        (self.source/'logs/actual-updates.jsonl').write_text(json.dumps(dict(
            step=6,update_status='complete',full_budget_complete=True))+'\n')
        for rank in range(4):
            (self.source/'logs'/('optimizer-rank-'+str(rank)+'.jsonl')).write_text(json.dumps(dict(
                kind='OPTIMIZER_WRITE',rank=rank,run_id='source-run',write_count=120))+'\n')
        with self.assertRaisesRegex(ValueError,'EXPLICIT_USER_APPROVAL'):
            restore.validate_request(self.target,self.commit)
        approval=dict(approved_by='user',approved_unix=1,human_instruction='Resume CP5 and recompute6',
                      source_job='724530',checkpoint_outer=5,source_completed_outer=6,
                      source_optimizer_writes_per_rank=[120]*4,recompute_complete_outers=[6],lossless_resume=False)
        self.write(self.target/'preparation/restore-request.json',dict(self.request,
                   unsaved_suffix_authorization=approval))
        restore.validate_request(self.target,self.commit)
        for key,value in (('source_job','99'),('checkpoint_outer',4),('recompute_complete_outers',[]),
                          ('source_optimizer_writes_per_rank',[100]*4),('lossless_resume',True)):
            self.write(self.target/'preparation/restore-request.json',dict(self.request,
                       unsaved_suffix_authorization=dict(approval,**{key:value})))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'EXPLICIT_USER_APPROVAL'):
                restore.validate_request(self.target,self.commit)

    def test_partial_rank_write_also_requires_explicit_approval(self):
        path=self.source/'logs/optimizer-rank-2.jsonl'
        path.write_text(json.dumps(dict(kind='OPTIMIZER_WRITE',rank=2,run_id='source-run',write_count=101))+'\n')
        with self.assertRaisesRegex(ValueError,'EXPLICIT_USER_APPROVAL'):
            restore.validate_request(self.target,self.commit)

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
