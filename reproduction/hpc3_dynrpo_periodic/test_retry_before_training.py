"""Rejection tests for the one narrowly certified deferred-cleanup retry."""
import copy
import json
import os
from pathlib import Path
import pwd
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from reproduction.hpc3_dynrpo_periodic import retry_before_training as retry
from reproduction.hpc3_dynrpo_periodic import final_audit as audit


class DeferredRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.prefix = Path(self.temp.name).resolve()/'run'
        self.source, self.target = self.prefix, Path(str(self.prefix)+'-r1')
        for base in (self.source,self.target):
            (base/'preparation').mkdir(parents=True)
            (base/'logs').mkdir()
            (base/'warmup').mkdir()
        self.commit = 'a'*40
        self.uid, self.user = os.getuid(), pwd.getpwuid(os.getuid()).pw_name
        self.failed, self.cleanup = '724056','724148'
        self.node = 'ACD1-12'
        self.log = 'Traceback\n  File "reproduction/common8/refit.py", line4\nModuleNotFoundError: No module named \'joblib\'\n'
        self.ledger = dict(run_id='old',jobs={})
        arguments = {
            'gpu-preflight': ['-m','torch.distributed.run','--standalone','--nproc_per_node=4','reproduction/common8/gpu_preflight.py'],
            'model-initial': ['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','reproduction.common8.model_smoke','--phase','initial'],
            'retrieval-stage': ['-m','reproduction.common8.stage_retrieval','--base',str(self.source),'--port','18016'],
            'retriever': ['-m','reproduction.common8.retriever'],
            'initial-calibration': ['-m','reproduction.common8.refit','--step','0','--model',str(retry.MODEL)],
        }
        for index,(role,argv) in enumerate(arguments.items()):
            identity=dict(pid=100+index,start_ticks=index+1,sid=100+index,uid=self.uid)
            self.ledger['jobs'][role]=dict(leader=identity,ledger={str(identity['pid']):identity},
                arguments=[str(self.source/'env/bin/python'),*argv],log=str(self.source/'logs'/(role+'.log')),
                state='CLEANUP_FAILED' if role=='retriever' else 'STOPPED',
                exit_code=None if role=='retriever' else (1 if role=='initial-calibration' else 0))
        root=self.source/'source'/('b'*40)
        self.write(root/'reproduction/hpc3_dynrpo_periodic/reconcile_cleanup.py','owned cleanup',raw=True)
        self.script='#!/bin/sh\n'+' '.join(['env','SEARCH_G1_BASE='+str(self.source),
            'PYTHONPATH='+str(root),retry.CONTROL_PYTHON,'-m',
            'reproduction.hpc3_dynrpo_periodic.reconcile_cleanup','--failed-job',self.failed])+'\n'
        self.job=dict(JobId=self.cleanup,JobState='PENDING',UserId=self.user+'('+str(self.uid)+')',
            JobName='dynrpo-thread5-cleanup-'+self.failed,WorkDir=str(self.source),ReqNodeList=self.node,
            NumNodes='1',NodeList='(null)')

    def tearDown(self):
        self.temp.cleanup()

    def write(self,path,value,raw=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value if raw else json.dumps(value))
        return path

    def validate_job(self,job=None,script=None):
        return retry.validate_cleanup_job(job or self.job,script or self.script,
            cleanup_job=self.cleanup,failed_job=self.failed,source=self.source,old_node=self.node,
            username=self.user,uid=self.uid)

    def certificate(self):
        self.write(self.target/'preparation/source-publication.json',dict(commit=self.commit,pushed_and_fetched=True))
        snapshot=self.write(self.target/'preparation/deferred-source-process-ledger.json',self.ledger)
        head=self.write(self.target/'warmup/continued-head.pt','immutable-probe400',raw=True)
        proof=dict(kind=retry.DEFERRED_KIND,source_base=str(self.source),target_base=str(self.target),
            source_job=self.failed,source_run_id='old',source_node=self.node,cleaned=False,
            target_source_commit=self.commit,head_sha256=retry.sha(head),
            target_publication_sha256=retry.sha(self.target/'preparation/source-publication.json'),
            source_ledger_snapshot=str(snapshot),source_ledger_sha256=retry.sha(snapshot),
            final_acceptance_requires_actual_source_cleanup=True,
            cpu_probe_readback=dict(actual_saved_model_adam_rng_readback=True))
        record=dict(source_base=str(self.source),source_job=self.failed,source_run_id='old',
            student_updates_before_failure=0,inherited_probe_adam_steps=400,
            inherited_probe_sha256=retry.sha(head),probe_optimizer_reset=False,deferred_cleanup=proof)
        self.write(self.target/'preparation/zero-update-retry.json',record)
        return record

    def test_exact_import_failure_and_pretraining_roles_only(self):
        retry.validate_pretraining_ledger(self.ledger,self.source,'old',self.uid,self.log)
        for mutation in ('trainer','wrong_argv','wrong_uid','different_error','unfinished_smoke'):
            ledger=copy.deepcopy(self.ledger);log=self.log
            if mutation=='trainer':ledger['jobs']['trainer']=ledger['jobs']['retriever']
            if mutation=='wrong_argv':ledger['jobs']['initial-calibration']['arguments'][-3]='25'
            if mutation=='wrong_uid':ledger['jobs']['retriever']['leader']['uid']+=1
            if mutation=='different_error':log=log.replace('joblib','numpy')
            if mutation=='unfinished_smoke':ledger['jobs']['model-initial']['state']='STARTED'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                retry.validate_pretraining_ledger(ledger,self.source,'old',self.uid,log)

    def test_any_student_claim_or_state_refuses_zero_update_retry(self):
        retry.pristine_student(self.source)
        for name in ('logs/training-launch-claim.json','logs/actual-updates.jsonl',
                     'logs/optimizer-rank-0.jsonl','checkpoints/actor/global_step_1/model.pt',
                     'artifacts/question-probe/periodic-refits/25/latest-state.pt'):
            path=self.write(self.source/name,'',raw=True)
            with self.subTest(name=name),self.assertRaises(ValueError):retry.pristine_student(self.source)
            path.unlink()

    def test_cleanup_job_must_be_live_owned_original_node_and_exact_script(self):
        self.validate_job()
        self.validate_job(dict(self.job,JobState='RUNNING',NodeList=self.node))
        for change in (dict(JobState='CANCELLED'),dict(UserId='foreign(9)'),dict(ReqNodeList='ACD1-13'),
                       dict(WorkDir=str(self.target)),dict(JobName='foreign'),dict(JobState='RUNNING',NodeList='ACD1-13')):
            with self.subTest(change=change),self.assertRaises(ValueError):self.validate_job(dict(self.job,**change))
        for script in (self.script.replace('--failed-job '+self.failed,'--failed-job 99'),
                       self.script+'echo altered\n',self.script.replace('reconcile_cleanup','train')):
            with self.assertRaises(ValueError):self.validate_job(script=script)

    def test_terminal_release_and_source_identity_are_required(self):
        run=dict(run_id='old')
        self.write(self.source/'logs/native-allocation-admission.json',dict(slurm_job_id=self.failed,
            run_id='old',node=self.node,accepted=True))
        self.write(self.source/'logs/pipeline-status.json',dict(run_id='old',stage='OWNED_CLEANUP_FAILED'))
        self.write(self.source/'logs/process-ledger.json',self.ledger)
        self.write(self.source/'logs/run-manifest.json',run)
        self.write(self.source/'logs/initial-calibration.log',self.log,raw=True)
        rows=[self.failed+'|FAILED|1:0|'+self.user+'|'+self.node+'|',
              self.failed+'.extern|COMPLETED|0:0||'+self.node+'|']
        job=' '.join(k+'='+v for k,v in self.job.items())
        with mock.patch.object(retry.subprocess,'check_output',side_effect=[job,self.script]):
            proof,_=retry.deferred_source(self.source,self.failed,self.cleanup,rows,run)
        self.assertFalse(proof['cleaned'])
        for bad in (rows[:1],[rows[0].replace('FAILED','COMPLETED'),rows[1]],
                    [rows[0].replace(self.user,'foreign'),rows[1]],
                    [rows[0],rows[1].replace('COMPLETED','RUNNING')]):
            with self.assertRaises(ValueError):retry.deferred_source(self.source,self.failed,self.cleanup,bad,run)

    def test_certificate_binds_target_publication_head_and_immutable_ledger(self):
        self.certificate()
        self.assertEqual(retry.excluded_nodes(self.target,self.commit),['ACD1-12','ACD1-21'])
        for mutation in ('publication','head','ledger','commit'):
            self.certificate()
            if mutation=='publication':self.write(self.target/'preparation/source-publication.json',dict(commit='c'*40,pushed_and_fetched=True))
            if mutation=='head':self.write(self.target/'warmup/continued-head.pt','changed',raw=True)
            if mutation=='ledger':self.write(self.target/'preparation/deferred-source-process-ledger.json',dict(run_id='changed'))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                retry.excluded_nodes(self.target,'c'*40 if mutation=='commit' else self.commit)

    def test_pure_cpu_actual_allocation_rejects_oldnode_and_missing_exclusion(self):
        job=dict(JobId='new',JobState='RUNNING',UserId=self.user+'('+str(self.uid)+')',NumNodes='1')
        params=dict(job_id='new',uid=self.uid,username=self.user,required=['ACD1-12','ACD1-21'])
        retry.validate_retry_allocation(job,'ACD1-55',['ACD1-55'],['ACD1-12','ACD1-21'],**params)
        for node,excluded in (('ACD1-12',['ACD1-12','ACD1-21']),('ACD1-55',['ACD1-21'])):
            with self.assertRaises(ValueError):
                retry.validate_retry_allocation(job,node,[node],excluded,**params)
        with self.assertRaises(ValueError):
            retry.validate_retry_allocation(dict(job,JobState='PENDING'),'ACD1-55',['ACD1-55'],params['required'],**params)

    def test_actual_submission_includes_oldnode_exclusion_without_dependency(self):
        from reproduction.hpc3_dynrpo_periodic import control
        self.certificate();captured=[]
        def remote(command,payload=b''):
            program=shlex.split(command)[2]
            def submit(arguments,**kwargs):
                captured.append(arguments)
                return SimpleNamespace(returncode=0,stdout='999\n',stderr='')
            with mock.patch.object(control.subprocess,'check_output',return_value=''), \
                    mock.patch.object(control.subprocess,'run',side_effect=submit):
                exec(program,{})
            return SimpleNamespace(returncode=0,stdout=b'{}',stderr=b'')
        with mock.patch.object(control,'BASE',self.target),mock.patch.object(control,'run',side_effect=remote):
            control.submit('formal',self.commit,Path(self.temp.name),'acd_u')
        self.assertIn('--exclude=ACD1-12,ACD1-21',captured[0])
        self.assertFalse(any(arg.startswith('--dependency') for arg in captured[0]))

    def test_invalid_certificate_creates_neither_submit_claim_nor_slurm_job(self):
        from reproduction.hpc3_dynrpo_periodic import control
        self.certificate()
        self.write(self.target/'warmup/continued-head.pt','changed-after-certificate',raw=True)
        def remote(command,payload=b''):
            program=shlex.split(command)[2]
            with mock.patch.object(control.subprocess,'check_output',return_value=''), \
                    mock.patch.object(control.subprocess,'run') as submit:
                with self.assertRaises(ValueError):exec(program,{})
                submit.assert_not_called()
            return SimpleNamespace(returncode=1,stdout=b'',stderr=b'Certificate rejected')
        with mock.patch.object(control,'BASE',self.target),mock.patch.object(control,'run',side_effect=remote):
            with self.assertRaises(RuntimeError):
                control.submit('formal',self.commit,Path(self.temp.name),'acd_u')
        self.assertFalse((self.target/'preparation/formal-submit-claim.json').exists())
        self.assertFalse((self.target/'preparation/formal-submission.json').exists())

    def test_final_audit_requires_fresh_oldcleanup_with_original_run_and_node(self):
        record=self.certificate();proof=record['deferred_cleanup']
        source_run=dict(run_id='old',source_commit='b'*40)
        self.write(self.source/'logs/run-manifest.json',source_run)
        proof['source_run_manifest_sha256']=retry.sha(self.source/'logs/run-manifest.json')
        self.write(self.target/'preparation/zero-update-retry.json',record)
        self.write(self.source/'preparation/formal-submission.json',dict(job=self.failed,commit='b'*40,returncode=0))
        current=dict(run_id='current',source_commit=self.commit)
        self.write(self.target/'logs/run-manifest.json',current)
        self.write(self.target/'preparation/formal-submission.json',dict(job='999',commit=self.commit,returncode=0))
        self.write(self.target/'logs/pipeline-status.json',dict(run_id='current',stage='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP'))
        self.write(self.target/'logs/cleanup-status.json',dict(run_id='current',cleaned=True))
        self.write(self.target/'logs/process-ledger.json',dict(run_id='current',jobs={'trainer':dict(state='STOPPED')}))
        stopped=copy.deepcopy(self.ledger)
        for value in stopped['jobs'].values():value['state']='STOPPED'
        self.write(self.source/'logs/process-ledger.json',stopped)
        cleanup=dict(run_id='old',cleaned=True,failed_job=self.failed,execution_hostname=self.node,
            original_allocation_hostname_verified=True,all_owned_sessions_absent=True,recorded_cuda_contexts_absent=True)
        calls=[]
        class Backend:
            def terminal_job(_self,base,run,submission,require_success=True):
                calls.append((str(base),require_success))
                return dict(job=submission['job'],freshly_queried=True)
        def invoke():
            value=audit.Auditor(self.target,reader=audit.Reader(self.prefix),backend=Backend())
            value.chain=[dict(base=self.target,run=current)];value.run_id='current'
            return value.cleanup()
        self.write(self.source/'logs/cleanup-status.json',cleanup)
        result=invoke()
        self.assertEqual(len(result),2)
        self.assertIn((str(self.source),False),calls)
        # Cleanup legitimately changed the current process ledger; the immutable
        # admission snapshot remains its original precleanup SHA.
        self.assertNotEqual(retry.sha(self.source/'logs/process-ledger.json'),proof['source_ledger_sha256'])
        for mutation in (dict(cleaned=False),dict(run_id='foreign'),dict(failed_job='99'),dict(execution_hostname='ACD1-13')):
            self.write(self.source/'logs/cleanup-status.json',dict(cleanup,**mutation))
            with self.subTest(mutation=mutation),self.assertRaises(audit.Incomplete):invoke()


if __name__=='__main__':
    unittest.main(verbosity=2)
