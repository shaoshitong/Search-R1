"""Synthetic protocol and real byte/rename tests; never actual HPC acceptance."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
from reproduction.hpc3_qprobe import recover_sigma3_archive5 as recovery
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader
from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as holder


class RecoveryFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.top=Path(self.temp.name).resolve()
        self.root=self.top/'root';self.base=self.top/'sigma3';self.commit='a'*40
        self.archive=self.root/'final-control'/recovery.RUN/'archive-5'
        self.preserved=self.archive.with_name('failed-archive-5-first-9c')
        self.intent=self.archive.with_name('archive5-recovery-first-9c')
        self.follow=self.root/'followthrough/721485';self.new=self.follow.with_name('721485-archive5-recovery-20261015')
        self.payload=self.root/'archives'/recovery.RUN/'global_step_5'
        self.source=dict(commit=self.commit,publication_path=str(self.root/'publication.json'),publication_sha256='b'*64)
        self.oldsource=dict(commit=recovery.OLD,publication_path=str(self.root/'oldpublication.json'),publication_sha256='c'*64)
        self.binding=dict(base=str(self.base),run_id=recovery.RUN,producer_job=recovery.JOB,producer_commit=recovery.PRODUCER)
        self.run=dict(run_id=recovery.RUN,source_commit=recovery.PRODUCER,source_sha256={'search_g1/science.py':'d'*64})
        self.pair=dict(path=str(self.base/'pair5.json'),sha256=recovery.PAIR_SHA,step=5,run_id=recovery.RUN)
        self.stack=ExitStack()
        values=dict(ROOT=self.root,BASE=self.base,ARCHIVE=self.archive,PRESERVED=self.preserved,INTENT=self.intent,
                    OLD_FOLLOW=self.follow,DIRECTORY=self.new,PAYLOAD=self.payload)
        for key,value in values.items():self.stack.enter_context(patch.object(recovery,key,value))
        for target in (recovery.control,reader.shared):self.stack.enter_context(patch.object(target,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(recovery.control,'owner'))
        self.stack.enter_context(patch.object(recovery.control,'caps'))
        self.stack.enter_context(patch.object(recovery.control,'live',return_value=False))
        self.stack.enter_context(patch.object(recovery,'__file__',str(self.root/'source'/self.commit/'reproduction/hpc3_qprobe/recover_sigma3_archive5.py')))
        self.stack.enter_context(patch.object(recovery.control,'source',return_value=(self.root/'source'/self.commit,self.source)))
        self.stack.enter_context(patch.object(reader,'publication',return_value=(self.root/'source'/recovery.OLD,self.oldsource)))
        self.stack.enter_context(patch.object(recovery.control,'producer',return_value=(self.run,self.binding,'721485|RUNNING|0:0|jxu957|14090|jxu957|formal-user|emergency_acd\n')))
        self.stack.enter_context(patch.object(recovery.control,'pair',return_value=(Path(self.pair['path']),self.pair)))
        self.stack.enter_context(patch.object(recovery,'fixed_binding',return_value=self.binding))
        self.write(Path(self.source['publication_path']),dict(files=self.run['source_sha256']))
        token='e'*32
        claim=dict(schema_version=1,role='detached_cpu_pair_archive',token=token,step=5,source=self.oldsource,
            producer=self.binding,pair=self.pair,controller=recovery.OLD_CONTROLLER,
            command=recovery.control.command(self.root/'source'/recovery.OLD,'archive-worker',self.archive/'submit-claim.json',token),
            training_permitted=False,direct_gpu_device_access_permitted=False,native_gpu_audit_submission_permitted=False)
        sha=self.write(self.archive/'submit-claim.json',claim)
        self.stack.enter_context(patch.object(recovery,'CLAIM_SHA',sha))
        shared=dict(token=token,claim_sha256=sha)
        self.write(self.archive/'runner-binding.json',dict(shared,actual_process=recovery.OLD_RUNNER,controller=recovery.OLD_CONTROLLER))
        self.write(self.archive/'task-binding.json',dict(shared,actual_process=recovery.OLD_CHILD,supervisor=recovery.OLD_RUNNER))
        self.write(self.archive/'worker-result.json',dict(shared,actual_task_exit_code=1,actual_task_birth=recovery.OLD_CHILD,
            actual_supervisor=recovery.OLD_RUNNER,status='CPU_CHILD_FAILED_OR_MISSING_RECEIPT',
            role='detached_cpu_pair_archive',finished_unix=recovery.FINISHED))
        task='Traceback (most recent call last):\n  File "'+str(self.root/'source'/recovery.OLD/'reproduction/hpc3_qprobe/archive_completed_pair.py')+'", line 102, in archive\n'
        task+="    if measured.returncode or measured.stderr.strip():raise ValueError('Fresh owner storage measurement failed')\nValueError: Fresh owner storage measurement failed\n"
        (self.archive/'task.log').write_text(task);(self.archive/'worker.log').write_text('original supervisor stderr retained\n')
        self.write(self.follow/'claim.json',dict(source=self.oldsource,binding=self.binding,cwd=str(self.root/'source'/recovery.OLD),
            training_submission_permitted=False,actor_model_reload_permitted=False,direct_gpu_access_permitted=False,
            native_final_restore_only_submission_permitted=True,
            command=[str(self.base/'env/bin/python'),'-m','reproduction.hpc3_qprobe.followthrough_sigma3','supervise','--commit',recovery.OLD]))
        self.write(self.follow/'process.json',dict(actual_process=recovery.OLD_CONTROLLER,commit=recovery.OLD,control=str(self.follow)))
        previous=dict(phase='ACTUAL_PRODUCER_OR_ARCHIVE_WAIT',accounting=dict(job=recovery.JOB,state='RUNNING',exit_code='0:0'),
            archives={'5':dict(actual_payload_bytes=0,actual_payload_files=0)})
        self.write(self.follow/'result.json',dict(exit_code=1,error=recovery.ERROR,actual_process=recovery.OLD_CONTROLLER,
            result=previous,checked_unix=recovery.STOPPED,training_submission_performed=False,actor_model_loaded=False,direct_gpu_device_access_performed=False,native_final_restore_only_submission_permitted=True))
        self.write(self.follow/'state.json',dict(error=recovery.ERROR,phase='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS',full_objective_accepted=False,previous_actual_state=previous))
        self.du=self.top/'du.json'
        self.write(self.du,dict(completed_unix=time.time()-1))
        du_proof=reader.document(self.du)[1]
        self.stack.enter_context(patch.object(recovery,'successful_du',return_value=du_proof))
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()
    def start(self):
        # Real migration/fileset/hash; external launch/birth are the only process boundary.
        with patch.object(recovery.subprocess,'Popen',return_value=Mock(pid=900001)) as launch,\
             patch.object(recovery.control,'process',return_value=dict(pid=900001,start_ticks=123,sid=900001,uid=os.getuid())):
            answer=recovery.start(self.commit,self.du)
        return answer,launch
    def test_exact_pre_copy_failure_full_bytes_preserved_and_second_start_never_replays(self):
        # Exercise consume using the actual saved du row/time, without an observer
        # on this machine. Dedicated tests below validate the full du adapter.
        proof=recovery.failure_evidence();before=proof['archive_files']
        answer,launch=self.start()
        self.assertEqual(recovery.consume(self.commit)['status'],recovery.STATUS)
        self.assertEqual(answer['actual_process']['pid'],900001);self.assertEqual(launch.call_count,1)
        self.assertFalse(self.archive.exists());self.assertEqual(recovery.snapshot(self.preserved),before)
        self.assertEqual(recovery.failure_evidence(self.preserved),proof)
        migration=json.loads((self.intent/'migration.json').read_text())
        self.assertEqual(migration['before_files'],migration['after_files'])
        with patch.object(recovery.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'existing recovery intent'):recovery.start(self.commit,self.du)
            launch.assert_not_called()
    def test_live_unknown_wrong_wait_and_other_exception_refuse_before_intent(self):
        with patch.object(recovery.control,'live',return_value=True):
            with self.assertRaisesRegex(ValueError,'live or unknown'):recovery.failure_evidence()
        wait_path=self.archive/'worker-result.json';wait=json.loads(wait_path.read_text());wait['actual_task_exit_code']=0;self.write(wait_path,wait)
        with self.assertRaisesRegex(ValueError,'held failure/birth'):recovery.failure_evidence()
        wait['actual_task_exit_code']=1;self.write(wait_path,wait)
        (self.archive/'task.log').write_text('ValueError: quota exceeded\n')
        with self.assertRaisesRegex(ValueError,'exact original storage'):recovery.failure_evidence()
        self.assertFalse(self.intent.exists())
    def test_payload_even_empty_and_symlink_refuse_without_launch(self):
        self.payload.mkdir(parents=True)
        with patch.object(recovery.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'payload forbids'):recovery.start(self.commit,self.du)
            launch.assert_not_called()
        self.assertFalse(self.intent.exists())
    def test_same_length_preserved_log_change_changes_fingerprint(self):
        self.start()
        before=recovery.consume(self.commit)
        self.assertTrue(before['original_failure_preserved'])
        path=self.preserved/'worker.log';raw=path.read_bytes();path.write_bytes(raw.replace(b'original',b'tampered'))
        with self.assertRaisesRegex(ValueError,'original failure bytes changed'):recovery.consume(self.commit)
    def test_snapshot_rejects_added_symlink_and_new_evidence_member_is_bound(self):
        initial=recovery.snapshot(self.archive);(self.archive/'extra.txt').write_text('retained extra evidence')
        self.assertNotEqual(initial,recovery.snapshot(self.archive))
        (self.archive/'bad').symlink_to(self.archive/'task.log')
        with self.assertRaisesRegex(ValueError,'unsafe evidence member'):recovery.snapshot(self.archive)
    def test_unknown_intent_after_crash_is_not_replayed(self):
        self.intent.mkdir()
        with patch.object(recovery.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'existing recovery intent'):recovery.start(self.commit,self.du)
            launch.assert_not_called()


class DuAndRouteTests(unittest.TestCase):
    def test_actual_held_du_adapter_binding_and_admission_time_freshness(self):
        with tempfile.TemporaryDirectory() as d:
            top=Path(d).resolve();folder=top/'observer';folder.mkdir()
            now=1000.;root=str(recovery.ROOT.parent);stdout='1234\t'+root+'\n'
            actual=dict(source=dict(commit='a'*40),allocated_bytes=1234,stdout=stdout,stderr='',actual_du_exit_code=0,
                        started_unix=900.,completed_unix=990.)
            result=folder/'result.json';result.write_text(json.dumps(actual))
            adapter=dict(root=root,uid=os.getuid(),returncode=0,stderr='',stdout=stdout,allocated_bytes=1234,
                command=['du','-s','-B1',root],started_unix=900.,completed_unix=990.,
                original_result=dict(path=str(result),sha256=hashlib.sha256(result.read_bytes()).hexdigest()))
            path=top/'adapter.json';path.write_text(json.dumps(adapter))
            observer=Mock(CONTROL=folder);observer.inspect.return_value=dict(status='ACTUAL_DU_COMPLETED0_USAGE_MEASURED',
                actual_live_birth=False,actual_du_live_birth=False,result=actual)
            with patch.object(recovery.control,'OWNER',os.getuid()),patch.object(reader.shared,'OWNER',os.getuid()),\
                 patch.object(reader,'load_fixed',return_value=(observer,dict(commit='a'*40))):
                self.assertEqual(recovery.successful_du(path,now=now)['path'],str(path))
                with self.assertRaisesRegex(ValueError,'fresh successful'):recovery.successful_du(path,now=1600.)
                observer.inspect.return_value['actual_du_live_birth']=True
                with self.assertRaisesRegex(ValueError,'actual held du'):recovery.successful_du(path,now=now)
    def test_explicit_recovery_route_and_normal_route_are_disjoint(self):
        self.assertEqual(reader.followthrough_route(reader.FOLLOWTHROUGH)[0],'followthrough_sigma3')
        commit='a'*40
        self.assertEqual(reader.followthrough_route(commit,commit)[0],'recover_sigma3_archive5')
        self.assertNotEqual(holder.scope(commit),holder.scope())
        for pin,recovery_pin in ((commit,None),(reader.FOLLOWTHROUGH,commit),(reader.FOLLOWTHROUGH,reader.FOLLOWTHROUGH)):
            with self.assertRaises(ValueError):reader.followthrough_route(pin,recovery_pin)
    def test_recovery_missing200_still_no_scope_or_source_or_launch(self):
        with patch.object(holder.reader,'early_history',side_effect=ValueError('actual fresh1..200 absent')),\
             patch.object(holder,'caps'),patch.object(holder,'DIRECTORY',Path('/nonexistent-final-sigma3-test-scope')),\
             patch.object(holder.ctl,'source') as source,patch.object(holder.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'fresh1..200'):
                holder.start('b'*40,followthrough_commit='a'*40,recovery_commit='a'*40)
            source.assert_not_called();launch.assert_not_called()


class RecoveredConsumerCompatibility(unittest.TestCase):
    def test_holder_recovered_stdout_preserves_all_admitted_anchors_and_rejects_changed_failure(self):
        from reproduction.hpc3_qprobe.test_independent_goal_control_sigma3 import output_fixture
        source,admission,proof=output_fixture('b'*40);commit='a'*40
        reference=dict(status=recovery.STATUS,intent=dict(sha256='1'*64),migration=dict(sha256='2'*64))
        proof.update(followthrough_commit=commit,recovery_commit=commit)
        admission.update(recovery_commit=commit,archive5_recovery=reference)
        admission['followthrough_source']['commit']=commit
        proof['actual_followthrough_full_CPU_consumer'].update(source=admission['followthrough_source'],archive5_recovery=reference)
        holder.accept_output(proof,'b'*40,commit,admission,commit)
        bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer']['archive5_recovery']['intent']['sha256']='3'*64
        with self.assertRaisesRegex(ValueError,'admitted followthrough/raw'):
            holder.accept_output(bad,'b'*40,commit,admission,commit)
        with self.assertRaisesRegex(ValueError,'exact published'):
            holder.accept_output(proof,'b'*40,commit,admission)
    def test_native_pin_recovered_source_accepts_only_original_science_bytes(self):
        commit='a'*40;name='search_g1/qprobe_model.py';sha='b'*64
        run=dict(source_sha256={name:sha});native=dict(audit_commit=commit,audit_id='actual')
        audit=dict(source_commit=commit,run_id=reader.RUN,final_restore_audit=dict(audit_id='actual'),source_sha256={name:sha})
        reader.native_source_gate(run,native,audit,dict(files={name:sha}),commit,commit)
        with self.assertRaisesRegex(ValueError,'scientific source differs'):
            reader.native_source_gate(run,native,audit,dict(files={name:'c'*64}),commit,commit)
        native['audit_commit']=reader.FOLLOWTHROUGH
        with self.assertRaisesRegex(ValueError,'authenticated sigma3 followthrough source'):
            reader.native_source_gate(run,native,audit,dict(files={name:sha}),commit,commit)


if __name__=='__main__':unittest.main()
