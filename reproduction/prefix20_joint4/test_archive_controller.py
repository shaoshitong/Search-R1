"""Local CPU byte/process mechanics only; no SSH/remote/GPU operation."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

try:
    from . import checkpoint_archive as core
    from . import archive_supervisor as held
    from . import archive_controller as control
    from . import archive_launcher as launch
except ImportError:
    import checkpoint_archive as core
    import archive_supervisor as held
    import archive_controller as control
    import archive_launcher as launch


class Fixture:
    def __init__(self,root):
        self.root=Path(root).resolve();self.personal=self.root/'experiments';self.hk=self.root/'backup'
        self.base=self.personal/'qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016';self.base.mkdir(parents=True)
        self.commit='b'*40;self.run_id='a'*32;self.job='727050'
        self.patches=[patch.object(core,'PERSONAL_ROOT',self.personal),patch.object(core,'ARCHIVES',self.hk/'checkpoints'),
            patch.object(held,'HK_ROOT',self.hk),patch.object(held,'source_host',return_value=None)]
        for item in self.patches:item.start()
        code=Path(__file__).parent
        self.manifest={name:dict(bytes=(code/name).stat().st_size,sha256=core.sha(code/name)) for name in held.BUNDLE_NAMES}
        self.config=dict(schema_version=1,base=str(self.base),training_commit=self.commit,source_job=self.job,
            code_commit='c'*40,code_manifest=self.manifest,controller_root=str(self.hk/'archive-controller'/('job-'+self.job)),
            hpc_identity=held.HPC_IDENTITY,known_hosts=held.KNOWN_HOSTS,ssh_peer=held.PEER,poll_seconds=10,
            source_wait_deadline_seconds=600,transport_stall_seconds=60)
        Path(self.config['controller_root']).mkdir(parents=True)
        for source in (False,True):
            directory=held.code_root(self.config,source=source);directory.mkdir(parents=True)
            for name in held.BUNDLE_NAMES:(directory/name).write_bytes((code/name).read_bytes())
            core.publish(directory/'code-publication.json',dict(code_commit=self.config['code_commit'],files=self.manifest))
        self.plan=held.operation_plan(self.config,self.run_id,5,'snapshot')

    def close(self):
        for item in reversed(self.patches):item.stop()

    @staticmethod
    def born(pid=None):
        return dict(host='constructed-local-process-fixture',boot_id='f'*36,pid_namespace='pid:[1]',
            pid=os.getpid() if pid is None else pid,start_ticks=1,sid=os.getsid(0),uid=os.getuid(),state='S')


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.f=Fixture(self.temporary.name)
        self.addCleanup(self.temporary.cleanup);self.addCleanup(self.f.close)

    def test_fixed_routes_and_method_commits_reject_wrong_bindings(self):
        f=self.f;held.validate_config(f.config)
        for fields in (dict(hpc_identity=dict(held.HPC_IDENTITY,hostname='ACD-Login-2')),
            dict(ssh_peer='mou176@10.120.48.32'),dict(training_commit='short'),dict(source_job=''),dict(poll_seconds=True)):
            with self.subTest(fields=fields),self.assertRaises(ValueError):held.validate_config(dict(f.config,**fields))
        for step in (0,201):
            with self.subTest(step=step),self.assertRaises(ValueError):held.operation_plan(f.config,f.run_id,step,'snapshot')
        argv=held.ssh_argv('true')
        self.assertIn('-oStrictHostKeyChecking=yes',argv);self.assertIn('-oUserKnownHostsFile='+held.KNOWN_HOSTS,argv)
        self.assertIn(held.PEER,argv);self.assertIn(held.HPC_IDENTITY['boot_id'],argv[-1])

    def recovery_fixture(self):
        f=self.f;old=f.config;root=Path(old['controller_root']);born=dict(f.born(),host='xielab3')
        supervisor=dict(born,pid=born['pid']+1,start_ticks=2)
        core.publish(root/'config.json',old)
        core.publish(root/'controller-launch-claim.json',dict(config_sha256=core.document_sha(old)))
        core.publish(root/'controller-birth.json',born)
        core.publish(root/'controller-supervisor-birth.json',supervisor)
        core.publish(root/'launcher-observed-birth.json',supervisor)
        (root/'controller.log').write_text('constructed controller output\n')
        core.publish(root/'controller-exit.json',dict(config_sha256=core.document_sha(old),controller=born,
            returncode=-15,stdout_sha256=core.sha(root/'controller.log')))
        core.publish(root/'failed-controller-stop-request.json',dict(reason='constructed exact own birth TERM'))
        core.publish(root/'progress.json',dict(run_id=None,completed_steps=[]))
        (root/'observation-errors.jsonl').write_text(json.dumps(dict(mode='probe',returncode=1,
            stderr="TypeError: __init__() got an unexpected keyword argument 'capture_output'"))+'\n')
        prior=f.base/'preparation/archive-control/d9-controller-failure-no-operations.json';prior.parent.mkdir(parents=True,exist_ok=True)
        core.publish(prior,dict(status='constructed earlier no operations proof'))
        recovery=dict(reason='source_system_python36_capture_output_rejected',old_code_commit=old['code_commit'],
            old_config_sha256=core.sha(root/'config.json'),old_exit_sha256=core.sha(root/'controller-exit.json'),
            old_stop_request_sha256=core.sha(root/'failed-controller-stop-request.json'),
            source_no_operations_proof_path=str(prior),source_no_operations_proof_sha256=core.sha(prior))
        new=dict(old,code_commit='d'*40,recovery_of=recovery,controller_root=str(root/('recovery-'+old['code_commit'])))
        directory=held.code_root(new);directory.mkdir(parents=True)
        for name in held.BUNDLE_NAMES:(directory/name).write_bytes((held.code_root(old)/name).read_bytes())
        core.publish(directory/'code-publication.json',dict(code_commit=new['code_commit'],files=new['code_manifest']))
        return new,root

    def test_all_source_commands_use_bound_environment_python_and_no_pyc(self):
        f=self.f;c=control.Controller(f.config,'RAM only');commands=[]
        def run(argv,**kwargs):
            commands.append(shlex.split(argv[-1]));return subprocess.CompletedProcess(argv,0,b'{}',b'')
        with patch.object(held,'ssh_argv',side_effect=lambda command:['constructed',command]),patch.object(control.subprocess,'run',side_effect=run):
            c.remote('probe',f.config)
        self.assertEqual(commands[0][:2],[str(f.base/'env/bin/python'),'-B'])
        self.assertEqual(held.source_argv(f.config,'a.py','stream')[:2],commands[0][:2])
        self.assertTrue(held.bound_command('true').startswith('python3 -B -c '))
        with patch.object(held,'ssh_argv',side_effect=lambda command:['constructed',command]),\
             patch.object(control.subprocess,'run',return_value=subprocess.CompletedProcess([],0,core.document_bytes(
                 dict(status='SIDECAR_FULL_SHA_PASS',code_root=str(held.code_root(f.config,source=True)))),b'')) as invoke:
            c.deploy_source_sidecar()
        self.assertEqual(shlex.split(invoke.call_args[0][0][-1])[:3],[str(f.base/'env/bin/python'),'-B','-c'])

    def test_recovery_config_pins_new_revision_and_rejects_reusing_old_code(self):
        new,root=self.recovery_fixture();held.validate_config(new)
        for changed in (dict(controller_root=str(root)),dict(code_commit=new['recovery_of']['old_code_commit'])):
            with self.subTest(changed=changed),self.assertRaises(ValueError):held.validate_config(dict(new,**changed))
        changed=copy.deepcopy(new);changed['recovery_of']['reason']='generic retry'
        with self.assertRaises(ValueError):held.validate_config(changed)

    def test_source_launch_and_default_held_worker_use_training_environment_python(self):
        f=self.f;invocations=[]
        def spawn(argv,**kwargs):
            invocations.append(argv);return SimpleNamespace(pid=os.getpid(),wait=lambda:7)
        with patch.object(held.subprocess,'Popen',side_effect=spawn),patch.object(held,'identity',side_effect=f.born):
            held.launch_operation(f.plan)
        self.assertEqual(invocations[0][:2],[str(f.base/'env/bin/python'),'-B'])
        self.assertIn('supervise',invocations[0])
        with patch.object(held.subprocess,'Popen',side_effect=spawn):
            result=held.supervise(held.operation_root(f.plan),observe=f.born)
        self.assertEqual(invocations[1][:2],[str(f.base/'env/bin/python'),'-B'])
        self.assertIn('operation-worker',invocations[1]);self.assertEqual(result['worker_returncode'],7)

    def test_recovery_launch_persists_admission_then_claims_new_revision_only_once(self):
        new,old=self.recovery_fixture();root=Path(new['controller_root']);admitted=dict(status='constructed admission')
        invocations=[]
        def spawn(argv,**kwargs):
            self.assertTrue((root/'recovery-admission.json').exists());self.assertTrue((root/'controller-launch-claim.json').exists())
            invocations.append(argv);return SimpleNamespace(pid=os.getpid(),stdin=io.BytesIO())
        with patch.object(held.socket,'gethostname',return_value='xielab3'),patch.object(held,'identity',side_effect=self.f.born),\
             patch.object(held,'recovery_admission',return_value=admitted) as admission,patch.object(held.subprocess,'Popen',side_effect=spawn):
            self.assertEqual(held.launch_controller(new,'RAM')['status'],'DISPATCHED_CONTROLLER_SUPERVISOR')
            self.assertEqual(held.launch_controller(new,'RAM')['status'],'CLAIMED_BIRTH_PENDING_OR_UNKNOWN')
        self.assertEqual(len(invocations),1);self.assertEqual(admission.call_count,1)
        self.assertEqual(core._read_json(root/'recovery-admission.json'),admitted)
        self.assertNotIn('RAM',''.join(p.read_text() for p in root.iterdir() if p.is_file()))

    def test_recovery_admits_only_actual_nonzero_exit_and_dead_bound_births(self):
        new,root=self.recovery_fixture();observer=lambda config,password:dict(proof='constructed fresh empty namespaces')
        with patch.object(held,'same_birth',return_value=False):value=held.recovery_admission(new,'RAM',source_observer=observer)
        self.assertEqual(value['old_actual_exit']['returncode'],-15);self.assertTrue(value['old_births_dead'])
        self.assertFalse(value['training_job_changed']);self.assertFalse(Path(new['controller_root']).exists())
        with patch.object(held,'same_birth',return_value=True),self.assertRaisesRegex(ValueError,'STILL_ALIVE'):
            held.recovery_admission(new,'RAM',source_observer=observer)

    def test_recovery_refuses_wrong_exit_hash_and_previous_run_or_source_operations(self):
        new,root=self.recovery_fixture();wrong=copy.deepcopy(new);wrong['recovery_of']['old_exit_sha256']='e'*64
        with self.assertRaises(ValueError):held.recovery_admission(wrong,'RAM')
        core.publish(root/'run-binding.json',dict(run_id=self.f.run_id))
        with patch.object(held,'same_birth',return_value=False),self.assertRaisesRegex(ValueError,'HAS_RUN_OR_OPERATIONS'):
            held.recovery_admission(new,'RAM')

    def test_recovery_zero_exit_is_never_retry_authorization(self):
        new,root=self.recovery_fixture();exited=core._read_json(root/'controller-exit.json');exited['returncode']=0
        held.atomic(root/'controller-exit.json',exited);new['recovery_of']['old_exit_sha256']=core.sha(root/'controller-exit.json')
        with self.assertRaisesRegex(ValueError,'NONZERO_EXIT'):held.recovery_admission(new,'RAM')

    def timestamp_collision_fixture(self):
        spec=importlib.util.spec_from_file_location('collision_archive_fixture',Path(__file__).with_name('test_checkpoint_archive.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.archive=core
        pair=module.Fixture(self.f.root/'collision',step=5);self.addCleanup(pair.close)
        old=dict(self.f.config,base=str(pair.base),training_commit=pair.commit)
        root=Path(old['controller_root']);born=dict(self.f.born(),host='xielab3')
        supervisor=dict(born,pid=born['pid']+1,start_ticks=2)
        core.publish(root/'config.json',old);core.publish(root/'controller-launch-claim.json',dict(config_sha256=core.document_sha(old)))
        for name,value in (('controller-birth.json',born),('controller-supervisor-birth.json',supervisor),
                           ('launcher-observed-birth.json',supervisor)):
            core.publish(root/name,value)
        (root/'controller.log').write_text("TypeError: type object got multiple values for keyword argument 'checked_unix'\n")
        core.publish(root/'controller-exit.json',dict(config_sha256=core.document_sha(old),controller=born,
            returncode=1,stdout_sha256=core.sha(root/'controller.log')))
        core.publish(root/'controller-result.json',dict(status='ARCHIVE_CONTROLLER_INCOMPLETE',error_type='TypeError',
            error="type object got multiple values for keyword argument 'checked_unix'",completed_steps=[],
            run_id=pair.run_id,source_job=old['source_job'],training_job_changed=False))
        core.publish(root/'run-binding.json',dict(base=str(pair.base),training_commit=pair.commit,run_id=pair.run_id,
            source_job=old['source_job'],actual_prepared_sha256='e'*64))
        wrapped=core.snapshot(pair.base,5,commit=pair.commit,run_id=pair.run_id)
        output=dict(kind='snapshot',step=5,run_id=pair.run_id,result=wrapped,checked_unix=time.time())
        plan=held.operation_plan(old,pair.run_id,5,'snapshot')
        result=dict(worker_returncode=0,accepted_output=True,plan_sha256=core.document_sha(plan),
            worker_output_sha256=core.document_sha(output),worker_output_bytes=len(core.document_bytes(output)))
        observed=dict(status='COMPLETED',kind='snapshot',step=5,run_id=pair.run_id,source_complete_present=False,
            operation_root=str(held.operation_root(plan)),operation_result=result,output=output,
            operation_result_sha256=core.document_sha(result))
        core.publish(root/'step-5-snapshot-operation.json',observed)
        directory=core.ARCHIVES/pair.run_id/'global_step_5';directory.mkdir(parents=True);(directory/'transport').mkdir()
        (directory/'source-snapshot.json').write_bytes(Path(wrapped['snapshot_file']).read_bytes())
        core.publish(directory/'archive-claim.json',dict(run_id=pair.run_id,step=5,code_commit=old['code_commit'],
            source_snapshot_sha256=wrapped['snapshot_sha256']))
        core.publish(directory/'storage-admission.json',dict(constructed=True))
        (directory/'transport/0000.partial').write_bytes(b'actual failed first payload prefix')
        core.publish(directory/'transport/0000.transport-result.json',dict(status='INCOMPLETE_STREAM_PRESERVED',error_type='TypeError'))
        (directory/'transport/0000.ssh-stderr').write_bytes(b'')
        recovery=dict(reason='terminal_archive_progress_timestamp_collision',old_code_commit=old['code_commit'],
            old_config_sha256=core.sha(root/'config.json'),old_exit_sha256=core.sha(root/'controller-exit.json'),
            old_result_sha256=core.sha(root/'controller-result.json'),source_snapshot_sha256=wrapped['snapshot_sha256'],run_id=pair.run_id)
        new=dict(old,code_commit='d'*40,recovery_of=recovery,controller_root=str(root/('recovery-'+old['code_commit'])))
        return new,old,root,directory,observed,pair

    def test_terminal_timestamp_collision_preserves_partial_and_adopts_existing_snapshot(self):
        new,old,root,directory,observed,pair=self.timestamp_collision_fixture()
        before={str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        observer=lambda config,password:observed
        with patch.object(held,'same_birth',return_value=False),patch.object(held,'_rename_no_replace',side_effect=os.rename):
            accepted=held.recovery_admission(new,'RAM',source_observer=observer)
        self.assertEqual(accepted['status'],'TERMINAL_TIMESTAMP_COLLISION_ARCHIVE_RECOVERY_ADMITTED')
        self.assertFalse(accepted['training_job_changed']);self.assertFalse(directory.exists())
        preserved=root/'failed-step5-transport'
        self.assertEqual({str(p.relative_to(preserved)):p.read_bytes() for p in preserved.rglob('*') if p.is_file()},before)
        self.assertEqual(core.sha(preserved/'source-snapshot.json'),new['recovery_of']['source_snapshot_sha256'])
        self.assertTrue((root/'timestamp-collision-recovery-claim.json').exists())
        with patch.object(held,'same_birth',return_value=False),self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):
            held.recovery_admission(new,'RAM',source_observer=observer)
        plan=held.operation_plan(new,pair.run_id,5,'snapshot');operation=held.operation_root(plan);operation.mkdir(parents=True)
        core.publish(operation/'plan.json',plan)
        with patch.object(held,'verify_bundle',return_value={}),patch.object(core,'snapshot',side_effect=AssertionError('never repeat full source snapshot')):
            adopted=held.operation_worker(operation)
        self.assertTrue(adopted['result']['adopted_existing_snapshot'])
        self.assertEqual(adopted['result']['snapshot_sha256'],new['recovery_of']['source_snapshot_sha256'])

    def test_timestamp_collision_live_birth_rejected_before_source_read_or_rename(self):
        new,old,root,directory,observed,pair=self.timestamp_collision_fixture()
        with patch.object(held,'same_birth',return_value=True),\
             patch.object(held,'completed_snapshot_observation',side_effect=AssertionError('no remote read')):
            with self.assertRaisesRegex(ValueError,'STILL_ALIVE'):held.recovery_admission(new,'RAM')
        self.assertTrue(directory.exists());self.assertFalse((root/'failed-step5-transport').exists())
        self.assertFalse((root/'timestamp-collision-recovery-claim.json').exists())

    def test_timestamp_collision_rejects_unknown_source_or_changed_snapshot_without_mutation(self):
        new,old,root,directory,observed,pair=self.timestamp_collision_fixture()
        for field,value in (('status','UNKNOWN_OPERATION_OUTCOME'),('source_complete_present',True)):
            wrong=dict(observed,**{field:value})
            with patch.object(held,'same_birth',return_value=False),self.assertRaises(ValueError):
                held.recovery_admission(new,'RAM',source_observer=lambda c,p:wrong)
        changed=copy.deepcopy(new);changed['recovery_of']['source_snapshot_sha256']='e'*64
        with patch.object(held,'same_birth',return_value=False),self.assertRaises(ValueError):
            held.recovery_admission(changed,'RAM',source_observer=lambda c,p:observed)
        self.assertTrue(directory.exists());self.assertFalse((root/'timestamp-collision-recovery-claim.json').exists())

    def test_source_no_operations_observation_requires_fresh_identity_and_exit_zero(self):
        new,root=self.recovery_fixture();paths=[str(self.f.base/p) for p in ('preparation/archive-control/operations','preparation/checkpoint-archives')]
        proof=dict(status='FRESH_SOURCE_NO_ARCHIVE_OPERATIONS_PASS',base=str(self.f.base),source_job=self.f.job,
            training_commit=self.f.commit,observer_uid=14090,identity=held.HPC_IDENTITY,run_manifest_may_exist=True,
            prior_proof_sha256=new['recovery_of']['source_no_operations_proof_sha256'],checked_unix=time.time(),
            roots=[dict(path=p,exists=False,empty=True) for p in paths])
        result=subprocess.CompletedProcess([],0,core.document_bytes(proof),b'')
        with patch.object(held.subprocess,'run',return_value=result) as invoke:value=held.fresh_no_source_operations(new,'do not persist')
        self.assertEqual(value['returncode'],0);self.assertNotIn('do not persist',json.dumps(value))
        # The secret is removed from the very environment passed to subprocess.
        self.assertNotIn('SSHPASS',invoke.call_args.kwargs['env'])
        for bad in (dict(proof,checked_unix=time.time()-120),dict(proof,observer_uid=0)):
            with patch.object(held.subprocess,'run',return_value=subprocess.CompletedProcess([],0,core.document_bytes(bad),b'')),self.assertRaises(ValueError):
                held.fresh_no_source_operations(new,'RAM')
        with patch.object(held.subprocess,'run',side_effect=subprocess.TimeoutExpired([],45)),self.assertRaisesRegex(ValueError,'UNKNOWN_NO_LAUNCH'):
            held.fresh_no_source_operations(new,'RAM')
        with patch.object(held.subprocess,'run',return_value=subprocess.CompletedProcess([],1,b'',b'failure')),self.assertRaisesRegex(ValueError,'FAILED_NO_LAUNCH'):
            held.fresh_no_source_operations(new,'RAM')

    def test_fresh_source_empty_namespaces_allow_existing_run_manifest_and_reject_operations(self):
        new,root=self.recovery_fixture();f=self.f
        (f.base/'job.id').write_text(f.job+'\n');core.publish(f.base/'preparation/training-submission.json',
            dict(job=f.job,source_commit=f.commit,returncode=0))
        (f.base/'logs').mkdir();core.publish(f.base/'logs/run-manifest.json',dict(run_id=f.run_id))
        program=held.NO_OPERATIONS_PROGRAM.replace("Path('/data/user/jxu957/experiments')",'Path('+repr(str(f.personal))+')')
        builtin_open=open;real_readlink=os.readlink
        def local_open(path,*args,**kwargs):
            if str(path)=='/proc/sys/kernel/random/boot_id':return io.StringIO(held.HPC_IDENTITY['boot_id'])
            return builtin_open(path,*args,**kwargs)
        def check():
            with patch.object(sys,'stdin',io.StringIO(json.dumps(new))),patch.object(sys,'stdout',io.StringIO()) as stdout,\
                 patch.object(os,'getuid',return_value=14090),patch.object(socket_module,'gethostname',return_value=held.HPC_IDENTITY['hostname']),\
                 patch.object(os,'readlink',side_effect=lambda p,*a,**kw:held.HPC_IDENTITY['pid_namespace'] if str(p)=='/proc/self/ns/pid' else real_readlink(p,*a,**kw)),\
                 patch('builtins.open',side_effect=local_open):
                exec(program,{})
                return json.loads(stdout.getvalue())
        import socket as socket_module
        self.assertTrue(check()['run_manifest_may_exist'])
        operations=f.base/'preparation/archive-control/operations';operations.mkdir();core.publish(operations/'some-claim.json',{})
        with self.assertRaises(AssertionError):check()

    def test_existing_recovery_once_claim_only_queries_and_never_spawns(self):
        new,old=self.recovery_fixture();root=Path(new['controller_root']);root.mkdir()
        core.publish(root/'config.json',new);core.publish(root/'controller-launch-claim.json',dict(config_sha256=core.document_sha(new)))
        with patch.object(held.socket,'gethostname',return_value='xielab3'),patch.object(held.subprocess,'Popen',side_effect=AssertionError('no replay')),\
             patch.object(held,'recovery_admission',side_effect=AssertionError('no re-admission')):
            self.assertEqual(held.launch_controller(new,'RAM')['status'],'CLAIMED_BIRTH_PENDING_OR_UNKNOWN')

    def test_accept_persists_exact_original_archive_wrapper_and_never_overwrites_it(self):
        f=self.f;wrapper=dict(archive_receipt_file='/constructed/archive/archive-receipt.json',
            archive_receipt_sha256='e'*64,archive_receipt=dict(checked_unix=123,constructed=True))
        plan=held.operation_plan(f.config,f.run_id,5,'accept',snapshot_sha='f'*64,archive_wrapper=wrapper)
        root=held.operation_root(plan);root.mkdir(parents=True);core.publish(root/'plan.json',plan)
        source=f.base/'preparation/checkpoint-archives/global_step_5';source.mkdir(parents=True)
        core.publish(source/'source-snapshot.json',{})
        with patch.object(core,'load_snapshot',return_value={}),patch.object(core,'validate_archive_proof',return_value={}),\
             patch.object(core,'accept_archive',return_value={'constructed':'accepted'}):held.operation_worker(root)
        imported=source/'archive-receipt-import.json';original_bytes=imported.read_bytes()
        self.assertEqual(core._read_json(imported),wrapper);self.assertEqual(core.sha(imported),core.document_sha(wrapper))
        changed=copy.deepcopy(plan);changed['archive_wrapper']['archive_receipt']['checked_unix']=124
        held.atomic(root/'plan.json',changed)
        with patch.object(core,'load_snapshot',return_value={}),patch.object(core,'validate_archive_proof',return_value={}),\
             patch.object(core,'accept_archive',side_effect=AssertionError('must refuse changed import')),\
             self.assertRaisesRegex(ValueError,'IMPORTED_ARCHIVE_WRAPPER_DIFFERS'):held.operation_worker(root)
        self.assertEqual(imported.read_bytes(),original_bytes)

    def test_partial_source_snapshot_claim_never_spawns_another_worker(self):
        f=self.f;source=f.base/'preparation/checkpoint-archives/global_step_5';source.mkdir(parents=True)
        core.publish(source/'claim.json',dict(pid=123,run_id=f.run_id))
        with patch.object(held.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            state=held.launch_operation(f.plan)
        self.assertEqual(state['status'],'EXISTING_SOURCE_SNAPSHOT_CLAIM_REQUIRES_INSPECTION')
        self.assertFalse(held.operation_root(f.plan).exists())

    def test_existing_held_claim_is_queried_and_binding_mismatch_is_rejected(self):
        f=self.f;root=held.operation_root(f.plan);root.mkdir(parents=True)
        core.publish(root/'plan.json',f.plan);core.publish(root/'claim.json',dict(plan_sha256=core.document_sha(f.plan)))
        with patch.object(held.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            self.assertEqual(held.launch_operation(f.plan)['status'],'CLAIMED_BIRTH_PENDING_OR_UNKNOWN')
        changed=copy.deepcopy(f.plan);changed['config']['training_commit']='d'*40
        with self.assertRaisesRegex(ValueError,'BINDING'):held.launch_operation(changed)

    def test_source_supervisor_records_actual_child_exit_zero_and_nonzero(self):
        f=self.f
        for step,exit_code in ((5,0),(50,7)):
            plan=held.operation_plan(f.config,f.run_id,step,'snapshot');root=held.operation_root(plan);root.mkdir(parents=True)
            core.publish(root/'plan.json',plan);core.publish(root/'claim.json',dict(plan_sha256=core.document_sha(plan)))
            output=dict(kind='snapshot',run_id=f.run_id,step=step,result={'constructed':True})
            program=('import json,pathlib,sys;pathlib.Path('+repr(str(root/'worker-output.json'))+').write_text('+repr(json.dumps(output))+');sys.exit('+str(exit_code)+')')
            result=held.supervise(root,observe=f.born,worker_command=[sys.executable,'-B','-c',program])
            self.assertEqual(result['worker_returncode'],exit_code)
            self.assertEqual(result['accepted_output'],exit_code==0)
            state=held.operation_status(plan)
            self.assertEqual(state['status'],'COMPLETED' if exit_code==0 else 'FAILED')

    def test_unknown_supervisor_birth_is_not_a_failed_operation_or_relaunch_permission(self):
        f=self.f;root=held.operation_root(f.plan);root.mkdir(parents=True)
        core.publish(root/'plan.json',f.plan);core.publish(root/'claim.json',dict(plan_sha256=core.document_sha(f.plan)))
        core.publish(root/'supervisor-birth.json',f.born())
        with patch.object(held,'same_birth',return_value=False):state=held.operation_status(f.plan)
        self.assertEqual(state['status'],'UNKNOWN_OPERATION_OUTCOME')
        self.assertNotIn('operation_result',state)

    def test_stream_counts_real_bytes_and_requires_actual_exit_zero(self):
        f=self.f;seed=b'constructed stream bytes\x00';data=seed*65536;row=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for label,exit_code in (('accepted',0),('bad-exit',9)):
            target=f.root/label;progress=[]
            # A short argv also runs on Linux MAX_ARG_STRLEN while transferring
            # more than one receiver block through an actual local pipe.
            program='import sys;sys.stdin.buffer.read();sys.stdout.buffer.write('+repr(seed)+'*65536);sys.stdout.buffer.flush();sys.exit('+str(exit_code)+')'
            if exit_code:
                with self.assertRaisesRegex(ValueError,'TRANSPORT_NOT_ACCEPTED'):
                    control.stream_download([sys.executable,'-B','-c',program],{},target,row,dict(os.environ),progress.append)
                self.assertFalse(target.exists())
            else:
                result=control.stream_download([sys.executable,'-B','-c',program],{},target,row,dict(os.environ),progress.append)
                self.assertEqual(target.read_bytes(),data);self.assertEqual(result['returncode'],0)
                self.assertEqual(progress[-1]['received_bytes'],len(data))
            receipt=core._read_json(target.with_name(target.name+'.transport-result.json'))
            self.assertEqual(receipt['returncode'],exit_code);self.assertEqual(receipt['received_bytes'],len(data))
            self.assertTrue(target.with_name(target.name+'.partial-'+row['sha256']).exists())

    def test_short_or_oversized_stream_preserves_partial_and_never_publishes_payload(self):
        f=self.f;row=dict(bytes=4,sha256=hashlib.sha256(b'abcd').hexdigest())
        for name,data in (('short',b'ab'),('long',b'abcde')):
            destination=f.root/name
            program='import sys;sys.stdin.buffer.read();sys.stdout.buffer.write('+repr(data)+')'
            with self.subTest(name=name),self.assertRaises(ValueError):
                control.stream_download([sys.executable,'-B','-c',program],{},destination,row,dict(os.environ),lambda _:None)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_name(destination.name+'.transport-result.json').exists())

    def test_stalled_transport_is_measured_failed_local_reader_and_preserves_partial(self):
        f=self.f;row=dict(bytes=4,sha256=hashlib.sha256(b'abcd').hexdigest());destination=f.root/'stall'
        program='import sys,time;sys.stdin.buffer.read();time.sleep(30)'
        with self.assertRaisesRegex(ValueError,'TRANSPORT_NOT_ACCEPTED'):
            control.stream_download([sys.executable,'-B','-c',program],{},destination,row,dict(os.environ),lambda _:None,stall_seconds=.01)
        result=core._read_json(destination.with_name(destination.name+'.transport-result.json'))
        self.assertTrue(result['timed_out']);self.assertNotEqual(result['returncode'],0);self.assertFalse(destination.exists())

    def test_ssh_launch_timeout_queries_existing_operation_and_never_replays_snapshot(self):
        f=self.f;c=control.Controller(f.config,'constructed RAM secret');c.run_id=f.run_id;calls=[]
        observations=[dict(status='RUNNING',operation_root=str(held.operation_root(f.plan)),source_progress=dict(verified_bytes=10)),
            dict(status='COMPLETED',operation_root=str(held.operation_root(f.plan)),source_progress=dict(verified_bytes=20),
                operation_result=dict(worker_returncode=0,accepted_output=True),output=dict(result={'snapshot':'constructed'}))]
        def remote(mode,payload):
            calls.append(mode)
            if mode=='launch-operation':raise control.ObservationUnknown('constructed observation timeout')
            return observations.pop(0)
        c.remote=remote
        with patch.object(control.time,'sleep',return_value=None):result=c.wait_operation(f.plan)
        self.assertEqual(result,{'snapshot':'constructed'});self.assertEqual(calls.count('launch-operation'),1)
        events=[json.loads(line) for line in c.events.read_text().splitlines()]
        self.assertEqual([r['actual_progress']['verified_bytes'] for r in events],[10,20])
        self.assertNotIn('constructed RAM secret',''.join(p.read_text() for p in c.root.iterdir() if p.is_file()))

    def test_unstarted_after_timeout_stays_unknown_without_relaunch(self):
        f=self.f;c=control.Controller(f.config,'secret');c.run_id=f.run_id;calls=[]
        def remote(mode,payload):
            calls.append(mode)
            if mode=='launch-operation':raise control.ObservationUnknown('timeout')
            return dict(status='UNSTARTED',operation_root=str(held.operation_root(f.plan)))
        c.remote=remote
        with self.assertRaisesRegex(control.ObservationUnknown,'NO_REPLAY'):c.wait_operation(f.plan)
        self.assertEqual(calls,['launch-operation','operation-status'])

    def test_run_id_is_bound_once_from_actual_prepared_receipt(self):
        f=self.f;c=control.Controller(f.config,'secret')
        self.assertFalse(c.bind_run(dict(run_id=None)))
        observation=dict(run_id=f.run_id,prepared_sha256='e'*64)
        self.assertTrue(c.bind_run(observation));self.assertTrue(c.bind_run(observation))
        with self.assertRaisesRegex(ValueError,'RUN_BINDING_CHANGED'):c.bind_run(dict(observation,run_id='f'*32))

    def test_terminal_source_without200_cannot_mark_archive_controller_complete(self):
        f=self.f;c=control.Controller(f.config,'secret');c.deploy_source_sidecar=lambda:None
        c.probe=lambda:dict(run_id=None,source_terminal=True,pair_receipts={})
        with patch.object(control.socket,'gethostname',return_value='xielab3'),patch.object(held,'identity',side_effect=f.born):
            with self.assertRaisesRegex(RuntimeError,'WITHOUT_REQUIRED_COMPLETED_PAIR'):c.run()
        result=core._read_json(c.root/'controller-result.json')
        self.assertEqual(result['status'],'ARCHIVE_CONTROLLER_INCOMPLETE');self.assertNotIn(200,result['completed_steps'])
        self.assertFalse(result['actual_cuda_restore_accepted']);self.assertFalse(result['final_native51713_accepted'])

    def test_controller_requires_five_ordered_steps_and_scopes_its_success(self):
        f=self.f;c=control.Controller(f.config,'secret');c.deploy_source_sidecar=lambda:None
        c.probe=lambda:dict(run_id=f.run_id,prepared_sha256='e'*64,source_terminal=False,pair_receipts={str(s):True for s in core.STEPS})
        c.archive_step=lambda step:c.completed.append(step)
        with patch.object(control.socket,'gethostname',return_value='xielab3'),patch.object(held,'identity',side_effect=f.born):result=c.run()
        self.assertEqual(result['steps'],list(core.STEPS));self.assertFalse(result['actual_cuda_restore_accepted'])
        self.assertFalse(result['final_native51713_accepted']);self.assertEqual(result['hpc3_restore_targets_created'],0)

    def test_controller_self_result_without_actual_exit_is_not_complete(self):
        f=self.f;root=Path(f.config['controller_root'])
        core.publish(root/'config.json',f.config)
        core.publish(root/'controller-launch-claim.json',dict(config_sha256=core.document_sha(f.config)))
        core.publish(root/'controller-birth.json',f.born())
        core.publish(root/'controller-result.json',dict(status='ALL_FIVE_CHECKPOINT_ARCHIVES_ACCEPTED'))
        with patch.object(held.socket,'gethostname',return_value='xielab3'),patch.object(held,'same_birth',return_value=False):
            self.assertEqual(held.controller_status(f.config)['status'],'UNKNOWN_CONTROLLER_OUTCOME')
        core.publish(root/'controller-exit.json',dict(config_sha256=core.document_sha(f.config),returncode=1))
        with patch.object(held.socket,'gethostname',return_value='xielab3'):
            self.assertEqual(held.controller_status(f.config)['status'],'FAILED')

    def test_exit_zero_marker_without200_is_not_accepted(self):
        f=self.f
        value=dict(status='ALL_FIVE_CHECKPOINT_ARCHIVES_ACCEPTED',scope='checkpoint bytes and CPU archive transport only',
            run_id=f.run_id,source_job=f.job,base=str(f.base),training_commit=f.commit,code_commit=f.config['code_commit'],
            steps=[5,50,100,150],actual_cuda_restore_accepted=False,final_native51713_accepted=False,
            training_job_changed=False,hpc3_archives_created=0,hpc3_restore_targets_created=0)
        self.assertFalse(held.controller_result_accepted(f.config,value))
        value['steps']=list(core.STEPS);self.assertTrue(held.controller_result_accepted(f.config,value))
        value['run_id']=None;self.assertFalse(held.controller_result_accepted(f.config,value))

    def test_local_once_entry_queries_after_timeout_without_reloading_password_or_relaunching(self):
        f=self.f;bridge=f.root/'bridge.py';bridge.write_text('# constructed bridge fixture\n');calls=[]
        def campus(command,payload,timeout):
            calls.append(json.loads(payload))
            return subprocess.CompletedProcess([],124,'','safe timeout') if len(calls)==1 else subprocess.CompletedProcess([],0,'{"status":"RUNNING"}','')
        fake=SimpleNamespace(_campus_long=campus)
        loader=SimpleNamespace(exec_module=lambda module:None)
        args=SimpleNamespace(mode='launch',repo=str(f.root/'unused-repo'),bridge_source=str(bridge),code_commit=f.config['code_commit'],
            training_commit=f.commit,base=str(f.base),source_job=f.job,output=str(f.root/'local-launch.json'),credentials_file='never-read')
        with patch.object(launch,'published_bundle',return_value=({},f.manifest)) as published,\
             patch.object(launch,'password_from_private_memory',return_value='sensitive-constructed-secret') as secret,\
             patch.object(launch,'MODERN_BRIDGE_SHA',core.sha(bridge)),\
             patch.object(launch.importlib.util,'spec_from_file_location',return_value=SimpleNamespace(loader=loader)),\
             patch.object(launch.importlib.util,'module_from_spec',return_value=fake):
            self.assertEqual(launch.run(args),124);self.assertEqual(launch.run(args),0)
        self.assertEqual(published.call_count,1);self.assertEqual(secret.call_count,1)
        self.assertEqual([p['mode'] for p in calls],['launch','status'])
        self.assertEqual(calls[0]['password'],'sensitive-constructed-secret');self.assertIsNone(calls[1]['password'])
        records=list(f.root.glob('local-launch.json.observation-*.json'));self.assertEqual(len(records),2)
        for path in records+[Path(args.output),Path(args.output+'.launch-claim.json')]:
            self.assertNotIn('sensitive-constructed-secret',path.read_text())

    def test_source_probe_requires_actual_same_job_native4_and_preserves_unknown_accounting(self):
        f=self.f;(f.base/'logs').mkdir();(f.base/'job.id').write_text(f.job+'\n')
        core.publish(f.base/'preparation/training-submission.json',dict(job=f.job,source_commit=f.commit,returncode=0))
        born=dict(JobId=f.job,JobState='RUNNING',Account='jxu957',QOS='formal-user',Partition='emergency_acd',
            UserId='jxu957(14090)',NumNodes='1',NumCPUs='48',JobName='qprobe-prefix20-joint200',TresPerNode='gres/gpu:4',TRES='cpu=48,mem=512G,node=1',
            Command=str(f.base/'source'/f.commit/'reproduction/prefix20_joint4/train.sh'),WorkDir=str(f.base/'repo'))
        run=dict(run_id=f.run_id,source_commit=f.commit,training_world_size=4,data_selection_mode='prefix20_probe',target_updates=200,
            question_probe=dict(probe_architecture=core.ARCHITECTURE))
        prepared=dict(source_commit=f.commit,run_id=f.run_id,actual_slurm_job_binding=born)
        core.publish(f.base/'logs/run-manifest.json',run);core.publish(f.base/'logs/prefix20-configuration-prepared.json',prepared)
        failed=f.job+'|FAILED|1:0|jxu957|14090|jxu957|formal-user|qprobe-prefix20-joint200|start|end|ACD1-26\n'+f.job+'.extern|COMPLETED|0:0||||||||\n'
        with patch.object(held.subprocess,'run',side_effect=[subprocess.CompletedProcess([],0,'',''),subprocess.CompletedProcess([],0,failed,'')]):
            value=held.probe_source(f.config)
        self.assertEqual(value['run_id'],f.run_id);self.assertTrue(value['source_terminal']);self.assertFalse(value['pair_receipts']['200'])
        with patch.object(held.subprocess,'run',side_effect=[subprocess.TimeoutExpired([],25),subprocess.CompletedProcess([],0,failed,'')]):
            value=held.probe_source(f.config)
        self.assertFalse(value['source_terminal']);self.assertEqual(value['queue']['status'],'OBSERVATION_TIMEOUT')
        prepared['actual_slurm_job_binding']['TresPerNode']='gres/gpu:8'
        held.atomic(f.base/'logs/prefix20-configuration-prepared.json',prepared)
        with self.assertRaisesRegex(ValueError,'NATIVE4_BINDING'):held.probe_source(f.config)

    def test_real_small_payload_snapshot_stream_archive_source_accept_integration(self):
        # Reuse the independent core's real byte-file fixture. Source functions
        # run locally; SSH is replaced solely by an actual local stdout stream.
        fixture_file=Path(__file__).with_name('test_checkpoint_archive.py')
        spec=importlib.util.spec_from_file_location('standalone_archive_fixture',fixture_file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.archive=core
        with tempfile.TemporaryDirectory() as temporary:
            pair=module.Fixture(temporary,step=5)
            try:
                config=dict(self.f.config,base=str(pair.base),training_commit=pair.commit)
                c=control.Controller(config,'constructed RAM secret');c.run_id=pair.run_id;current={};calls=[]
                code=held.code_root(config,source=True);code.mkdir(parents=True)
                for name in held.BUNDLE_NAMES:(code/name).write_bytes(Path(__file__).with_name(name).read_bytes())
                core.publish(code/'code-publication.json',dict(code_commit=config['code_commit'],files=config['code_manifest']))
                def wait(plan):
                    calls.append(plan['kind'])
                    if plan['kind']=='snapshot':
                        value=core.snapshot(pair.base,5,commit=pair.commit,run_id=pair.run_id);current.update(value);return value
                    root=held.operation_root(plan);root.mkdir(parents=True);core.publish(root/'plan.json',plan)
                    return held.operation_worker(root)['result']
                def local_stream(command):
                    args=shlex.split(command);self.assertEqual(args[:2],[str(pair.base/'env/bin/python'),'-B'])
                    index=int(args[args.index('--index')+1]);row=current['snapshot']['files'][index]
                    program='import pathlib,sys;sys.stdin.buffer.read();sys.stdout.buffer.write(pathlib.Path('+repr(row['source_path'])+').read_bytes());sys.stdout.buffer.flush()'
                    return [sys.executable,'-B','-c',program]
                c.wait_operation=wait
                with patch.object(held,'ssh_argv',side_effect=local_stream),patch.object(held,'identity',side_effect=self.f.born),\
                     patch.object(control.socket,'gethostname',return_value='xielab3'):
                    c.archive_step(5)
                self.assertEqual(calls,['snapshot','accept']);self.assertEqual(c.completed,[5])
                events=[json.loads(line) for line in c.events.read_text().splitlines()]
                received=[row for row in events if row['stage']=='RECEIVING_EXACT_SNAPSHOT_FILE']
                # Real stream_download progress carries checked_unix at both
                # byte growth and exit0; the old event merge raised TypeError.
                self.assertTrue(received)
                self.assertTrue(any(row.get('status')=='ACTUAL_TRANSPORT_EXIT0_BYTES_SHA_PASS' for row in received))
                self.assertTrue(all(row['checked_unix']>0 and row['run_id']==pair.run_id and
                    row['source_job']==config['source_job'] and row['completed_steps']==[] for row in received))
                self.assertEqual(events[-1]['completed_steps'],[5])
                archive_dir=core.ARCHIVES/pair.run_id/'global_step_5'
                transport=core._read_json(archive_dir/'transport-complete.json')
                self.assertTrue(transport['all_actual_stream_exit_codes_zero'])
                self.assertEqual(transport['accepted_payload_bytes'],current['snapshot']['bytes'])
                source=core._read_json(Path(current['snapshot_file']).parent/'complete.json')
                imported=core._read_json(Path(current['snapshot_file']).parent/'archive-receipt-import.json')
                archived=core._read_json(archive_dir/'archive-receipt.json')
                self.assertEqual(imported['archive_receipt'],archived)
                self.assertEqual(imported['archive_receipt_sha256'],core.sha(archive_dir/'archive-receipt.json'))
                self.assertNotEqual(imported['archive_receipt']['checked_unix'],source['checked_unix'])
                self.assertTrue(source['source_readback']['all_source_files_fresh_sha_pass'])
                self.assertTrue(source['original_sources_retained']);self.assertFalse(source['actual_cuda_restore_accepted'])
                for name in pair.model_shas:
                    original=pair.model/name;blob=core.ARCHIVES/pair.run_id/'immutable-dependencies'/core.sha(original)
                    payload=archive_dir/'payload'/str(original.relative_to('/'))
                    self.assertEqual(blob.stat().st_ino,payload.stat().st_ino)
            finally:pair.close()


if __name__=='__main__':unittest.main()
