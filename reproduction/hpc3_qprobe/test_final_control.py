"""Local CPU controller mechanisms only; no fixture proves HPC/GPU execution."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import final_control as control


class FinalControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.top=Path(self.temp.name).resolve()
        self.root=self.top/'qprobe-teacher-warmup200-root';self.root.mkdir()
        self.base=self.top/'qprobe-teacher-warmup200-cpu-fixture';self.base.mkdir()
        self.patch_root=patch.object(control,'ROOT',self.root);self.patch_owner=patch.object(control,'OWNER',os.getuid())
        self.patch_root.start();self.patch_owner.start()
        self.run_id='a'*32;self.commit='b'*40;self.job='12345'
        self.run=dict(run_id=self.run_id,source_commit=self.commit,training_world_size=4,
            actor_micro_batch_per_rank=4,execution_site='hpc3',checkpoint_dir=str(self.base/'checkpoints/search-g1-common8'))
        for folder in ('logs','preparation'):(self.base/folder).mkdir()
        self.write(self.base/'logs/run-manifest.json',self.run)
        self.write(self.base/'preparation/training-submission.json',dict(job=self.job,commit=self.commit,returncode=0))
        cmd=['sbatch',str(self.root/'source'/self.commit/'reproduction/hpc3_qprobe/train.sh'),str(self.base),self.commit]
        self.write(self.base/'preparation/training-submit-claim.json',dict(commit=self.commit,command=cmd))
        self.allocation=dict(accepted=True,run_id=self.run_id,slurm_job_id=self.job,training_world_size=4,
            requested_gres='gres/gpu:4',visible_devices=['0','1','2','3'],actual_allocated_indices=[1,2,3,7],
            actual_cuda_runtime_uuids=['GPU-00000000-0000-0000-0000-%012d'%i for i in range(4)],node='ACD1-1',
            actual_allocation_detail='JobId=12345 UserId=jxu957(14090) Account=jxu957 QOS=formal-user '+
                'Partition=emergency_acd WorkDir='+str(self.base)+' Command='+cmd[-3]+
                ' NumNodes=1 NumCPUs=48 MinMemoryNode=512G NodeList=ACD1-1 '+
                'AllocTRES=cpu=48,mem=512G,node=1,gres/gpu=4\n Nodes=ACD1-1 GRES=gpu:4(IDX:1-3,7)\n')
        self.write(self.base/'logs/native-allocation-admission.json',self.allocation)
        self.raw='12345|RUNNING|0:0|jxu957|14090|jxu957|formal-user|emergency_acd\n'
        self.owner_context=patch.object(control,'owner');self.owner_context.start()

    def tearDown(self):
        self.owner_context.stop();self.patch_root.stop();self.patch_owner.stop();self.temp.cleanup()

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def producer(self):
        with patch.object(control,'slurm_query',return_value=self.raw):return control.producer(self.base)

    def receipt(self,step):
        path=self.base/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json'
        value=dict(schema_version=1,kind='question_probe_pair',status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',
            step=step,world_size=4,run_id=self.run_id,actor_snapshot_id=self.run_id+':'+str(step),source_base=str(self.base),
            checkpoint=str(Path(self.run['checkpoint_dir'])/'actor'/('global_step_'+str(step))),
            source_run_manifest=dict(path=str(self.base/'logs/run-manifest.json'),full_read_sha256=control.digest(self.base/'logs/run-manifest.json')),
            attempt_progress=dict(completed_outer=step),driver_state=dict(completed_outer=step),
            ranks=[dict(rank=i,completed_outer=step) for i in range(4)])
        self.write(path,value);return path,value

    def test_exclusive_claim_is_atomic_and_cannot_replace_original(self):
        path=self.top/'claim.json';control.publish(path,dict(token='first'))
        with self.assertRaises(FileExistsError):control.publish(path,dict(token='second'))
        self.assertEqual(control.read(path),dict(token='first'))
        self.assertEqual(list(self.top.glob('claim.json.*.partial')),[])

    def test_actual_producer_owner_state_source_and_completed0_required(self):
        self.producer()
        with patch.object(control,'slurm_query',return_value=self.raw),self.assertRaisesRegex(ValueError,'COMPLETED0'):
            control.producer(self.base,complete=True)
        for raw in (self.raw.replace('RUNNING','PENDING'),self.raw.replace('14090','21930'),
                    self.raw+self.raw,self.raw.replace('RUNNING|0:0','COMPLETED|1:0')):
            with self.subTest(raw=raw),patch.object(control,'slurm_query',return_value=raw),self.assertRaises(ValueError):
                control.producer(self.base)
        with patch.object(control,'slurm_query',return_value=self.raw.replace('RUNNING','COMPLETED')):
            control.producer(self.base,complete=True)
        bad=copy.deepcopy(self.run);bad['slurm_job_id']='999';self.write(self.base/'logs/run-manifest.json',bad)
        with self.assertRaises(ValueError):self.producer()

    def test_real_manifest_shape_uses_native_allocation_without_inventing_job_field(self):
        manifest=self.base/'logs/run-manifest.json';before=manifest.read_bytes()
        run,binding,_=self.producer()
        self.assertNotIn('slurm_job_id',run);self.assertEqual(manifest.read_bytes(),before)
        self.assertEqual(binding['producer_job'],self.job)
        self.assertEqual(binding['native_allocation_sha256'],
                         control.digest(self.base/'logs/native-allocation-admission.json'))
        self.assertEqual(list(self.root.iterdir()),[])

    def test_allocation_must_bind_the_actual_producer_and_exact_four_gpu_identity(self):
        path=self.base/'logs/native-allocation-admission.json'
        invalid=[('accepted',False),('run_id','c'*32),('slurm_job_id','999'),('training_world_size',8),
                 ('training_world_size',True),('requested_gres','gres/gpu:8'),
                 ('visible_devices',['0','1','2','3','4','5','6','7']),
                 ('actual_allocated_indices',[1,2,3,3]),('actual_allocated_indices',[1,2,3,True]),
                 ('actual_cuda_runtime_uuids',[self.allocation['actual_cuda_runtime_uuids'][0]]*4),
                 ('actual_cuda_runtime_uuids',['GPU-invalid']*4)]
        for key,value in invalid:
            bad=copy.deepcopy(self.allocation);bad[key]=value;self.write(path,bad)
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'allocation binding'):
                self.producer()
        path.unlink()
        with self.assertRaises(FileNotFoundError):self.producer()

    def test_changed_native_admission_cannot_adopt_an_existing_control_claim(self):
        _,binding,_=self.producer();control.control_root(binding)
        changed=copy.deepcopy(self.allocation);changed['checked_unix']=123456
        self.write(self.base/'logs/native-allocation-admission.json',changed)
        _,new_binding,_=self.producer()
        with self.assertRaisesRegex(ValueError,'another producer'):control.control_root(new_binding)

    def test_native_detail_must_match_original_source_owner_resources_and_gpu_indices(self):
        path=self.base/'logs/native-allocation-admission.json';detail=self.allocation['actual_allocation_detail']
        for before,after in [('JobId=12345','JobId=999'),('jxu957(14090)','mou176(21930)'),
                             ('QOS=formal-user','QOS=other'),('Partition=emergency_acd','Partition=other'),
                             ('WorkDir='+str(self.base),'WorkDir=/other'),
                             ('Command=','Command=/other'),('NumCPUs=48','NumCPUs=96'),
                             ('MinMemoryNode=512G','MinMemoryNode=1T'),
                             ('gres/gpu=4','gres/gpu=8'),('IDX:1-3,7','IDX:0,2-3,7')]:
            changed=copy.deepcopy(self.allocation);changed['actual_allocation_detail']=detail.replace(before,after)
            self.write(path,changed)
            with self.subTest(before=before),self.assertRaises(ValueError):self.producer()

    def test_only_real_five_pair_paths_counters_and_four_ranks(self):
        path,value=self.receipt(5);control.pair(self.base,self.run,5)
        for key,bad in [('world_size',8),('step',200),('source_base','/other'),('actor_snapshot_id',self.run_id+':4')]:
            changed=dict(value);changed[key]=bad;self.write(path,changed)
            with self.subTest(key=key),self.assertRaises(ValueError):control.pair(self.base,self.run,5)
        for step in (None,0,1,199,201,True):
            with self.assertRaises(ValueError):control.pair(self.base,self.run,step)

    def test_existing_attempt_or_payload_prevents_another_process(self):
        self.receipt(5);run,binding,_=self.producer();directory=control.control_root(binding)/'archive-5';directory.mkdir()
        with patch.object(control,'source',return_value=(self.root,{})),patch.object(control,'slurm_query',return_value=self.raw), \
             patch.object(control.subprocess,'Popen') as popen,self.assertRaisesRegex(ValueError,'already attempted'):
            control.launch_archive(self.base,self.commit,5)
        popen.assert_not_called();directory.rmdir()
        payload=self.root/'archives'/self.run_id/'global_step_5';payload.mkdir(parents=True)
        with patch.object(control,'source',return_value=(self.root,{})),patch.object(control,'slurm_query',return_value=self.raw), \
             patch.object(control.subprocess,'Popen') as popen,self.assertRaisesRegex(ValueError,'payload already exists'):
            control.launch_archive(self.base,self.commit,5)
        popen.assert_not_called()

    def test_cpu_launch_claim_precedes_popen_and_records_actual_detached_birth(self):
        self.receipt(5);run,binding,_=self.producer();repository=self.root/'source'/self.commit
        publication=dict(commit=self.commit,publication_path='/fixture/publication',publication_sha256='c'*64)
        leader=dict(pid=41,start_ticks=71,sid=41,uid=control.OWNER,state='S')
        child_birth=dict(pid=42,start_ticks=72,sid=42,uid=control.OWNER,state='S')
        def start(command,**kwargs):
            claim=self.root/'final-control'/self.run_id/'archive-5/submit-claim.json'
            self.assertTrue(claim.is_file());value=control.read(claim)
            self.assertEqual(value['command'],command);self.assertEqual(value['controller'],control.birth(leader))
            self.assertTrue(kwargs['start_new_session']);self.assertEqual(kwargs['stdin'],subprocess.DEVNULL)
            return type('Child',(),{'pid':42})()
        with patch.object(control,'source',return_value=(repository,publication)), \
             patch.object(control,'slurm_query',return_value=self.raw), \
             patch.object(control,'process',side_effect=[leader,child_birth]), \
             patch.object(control.subprocess,'Popen',side_effect=start):result=control.launch_archive(self.base,self.commit,5)
        self.assertEqual(result['actual_process'],control.birth(child_birth))
        self.assertEqual(result['status'],'ACTUAL_DETACHED_CPU_ARCHIVE_STARTED')

    def test_missing_child_result_still_persists_actual_wait_status_without_pass(self):
        directory=self.root/'worker';directory.mkdir();claim_path=directory/'submit-claim.json'
        claim=dict(source=dict(commit=self.commit));control.publish(claim_path,claim)
        supervisor=dict(pid=41,start_ticks=71,sid=41,uid=control.OWNER,state='S')
        child_birth=dict(pid=42,start_ticks=72,sid=41,uid=control.OWNER,state='S')
        child=type('Child',(),{'pid':42,'wait':lambda self:0})()
        with patch.object(control,'worker_binding',return_value=(claim,directory)), \
             patch.object(control.subprocess,'Popen',return_value=child), \
             patch.object(control,'process',side_effect=[child_birth,supervisor,supervisor,supervisor]), \
             self.assertRaisesRegex(ValueError,'valid result'):control.archive_worker(claim_path,'token')
        result=control.read(directory/'worker-result.json')
        self.assertEqual(result['actual_task_exit_code'],0)
        self.assertEqual(result['status'],'CPU_CHILD_FAILED_OR_MISSING_RECEIPT')
        self.assertIn('result_validation_error',result)

    def test_generic_supervisor_enforces_role_specific_gpu_submission_scope(self):
        for audit in (False,True):
            for correct in (False,True):
                with self.subTest(audit=audit,correct=correct):
                    directory=self.root/('supervisor-'+str(audit)+'-'+str(correct));directory.mkdir()
                    claim_path=directory/'submit-claim.json';claim={'source':{'commit':self.commit}};control.publish(claim_path,claim)
                    supervisor=dict(pid=41,start_ticks=71,sid=41,uid=control.OWNER,state='S')
                    child_birth=dict(pid=42,start_ticks=72,sid=41,uid=control.OWNER,state='S')
                    status='CPU_FINAL_SUBMIT_CONTROLLER_RETURNED' if audit else 'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS'
                    result=dict(token='token',claim_sha256=control.digest(claim_path),actual_process=control.birth(child_birth),
                        status=status,training_performed=False,direct_gpu_device_access_performed=False,actor_model_loaded=False,
                        native_gpu_audit_submission_attempted=audit if correct else not audit)
                    control.publish(directory/'task-result.json',result)
                    child=type('Child',(),{'pid':42,'wait':lambda self:0})()
                    with patch.object(control,'worker_binding',return_value=(claim,directory)), \
                         patch.object(control.subprocess,'Popen',return_value=child), \
                         patch.object(control,'process',side_effect=[child_birth,supervisor,supervisor,supervisor]):
                        invoke=control.audit_worker if audit else control.archive_worker
                        if correct:invoke(claim_path,'token')
                        else:
                            with self.assertRaisesRegex(ValueError,'valid result'):invoke(claim_path,'token')
                    proof=control.read(directory/'worker-result.json')
                    self.assertEqual(proof['actual_task_exit_code'],0)
                    self.assertEqual(proof['status'].endswith('CHILD_COMPLETED0'),correct)

    def test_inspect_is_read_only_and_measures_real_partial_bytes(self):
        _,binding,_=self.producer();root=control.control_root(binding);directory=root/'archive-5';directory.mkdir()
        claim=dict(producer=binding,step=5,token='token');control.publish(directory/'submit-claim.json',claim)
        payload=self.root/'archives'/self.run_id/'global_step_5';payload.mkdir(parents=True)
        (payload/'weights.partial').write_bytes(b'x'*19);(payload/'other').write_bytes(b'y'*7)
        before=sorted(str(p) for p in self.top.rglob('*'))
        with patch.object(control,'slurm_query',return_value=self.raw),patch.object(control.subprocess,'Popen') as popen:
            result=control.inspect_archive(self.base,5)
        self.assertEqual(result['actual_payload_bytes'],26);self.assertEqual(result['partial_copy_bytes'],19)
        self.assertEqual(result['status'],'LAUNCH_OUTCOME_UNKNOWN_DO_NOT_REPLAY');popen.assert_not_called()
        self.assertEqual(before,sorted(str(p) for p in self.top.rglob('*')))

    def test_cpu_entry_caps_cuda_mask_and_no_training_or_signal_calls(self):
        env={key:'1' for key in control.CAPS};env['CUDA_VISIBLE_DEVICES']=''
        with patch.dict(os.environ,env):control.caps()
        with patch.dict(os.environ,dict(env,CUDA_VISIBLE_DEVICES='0')),self.assertRaises(ValueError):control.caps()
        with patch.dict(os.environ,dict(env,OMP_NUM_THREADS='8')),self.assertRaises(ValueError):control.caps()
        syntax=ast.parse(Path(control.__file__).read_text());calls=[n.func.attr for n in ast.walk(syntax)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'fit','generate_sequences','update_actor','init_workers','from_pretrained','kill','killpg'})
        shell=Path(control.__file__).with_suffix('.sh').read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES=''",shell);self.assertIn('QPROBE_SOURCE="$SOURCE"',shell)
        for key in control.CAPS:self.assertIn(key+'=1',shell)

    def audit_fixture(self):
        _,binding,_=self.producer();root=control.control_root(binding)
        repository=self.root/'source'/self.commit
        transport=dict(path='/synthetic/transport.json',target_receipt=dict(path='/synthetic/target.json',sha256='c'*64))
        proof=dict(source=dict(commit=self.commit),archives={'200':transport})
        return repository,binding,proof,root

    def test_audit_submit_has_real_test_only_and_preserves_unknown_without_replay(self):
        repository,binding,proof,root=self.audit_fixture()
        cpu_birth=dict(pid=99,start_ticks=101,sid=99,uid=control.OWNER,state='S')
        test=subprocess.CompletedProcess(['sbatch'],0,'test only accepted','')
        with patch.object(control,'final_ready',return_value=(repository,self.run,binding,proof)), \
             patch.object(control,'process',return_value=cpu_birth), \
             patch.object(control.subprocess,'run',side_effect=[test,subprocess.TimeoutExpired(['sbatch'],30)]) as submit:
            value=control.submit_audit(self.base,self.commit)
        self.assertEqual(value['status'],'SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY');self.assertIsNone(value['returncode'])
        claim=control.read(root/'final-audit/submit-claim.json');command=claim['command']
        for arg in ('--gres=gpu:4','--cpus-per-task=48','--mem=512G','--partition=emergency_acd',
                    '--account=jxu957','--qos=formal-user','--exclude=ACD1-31,ACD1-67','--job-name=qprobe-final200audit'):
            self.assertIn(arg,command)
        self.assertEqual(command[-2:],['/synthetic/target.json','/synthetic/transport.json'])
        self.assertIn('--test-only',submit.call_args_list[0].args[0]);self.assertNotIn('--test-only',submit.call_args_list[1].args[0])
        with patch.object(control,'final_ready',return_value=(repository,self.run,binding,proof)),patch.object(control.subprocess,'run') as replay, \
             self.assertRaisesRegex(ValueError,'already attempted'):control.submit_audit(self.base,self.commit)
        replay.assert_not_called()

    def test_final_submit_never_prepares_work_when_actual_final_evidence_is_missing(self):
        with patch.object(control,'source',return_value=(self.root,{})),patch.object(control,'slurm_query',return_value=self.raw), \
             patch.object(control.subprocess,'run') as submit,self.assertRaisesRegex(ValueError,'COMPLETED0'):
            control.submit_audit(self.base,self.commit)
        submit.assert_not_called();self.assertFalse((self.root/'final-control').exists())

    def test_typed_final_producer_consumes_exact_held_failure_and_preserves_failed_accounting(self):
        from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader
        _,binding,_=self.producer();binding['producer_job']='719225';binding['base']=str(self.base)
        raw='719225|FAILED|1:0|jxu957|14090|jxu957|formal-user|emergency_acd\n'
        recovery_commit='c'*40;publication=dict(commit=recovery_commit,publication_sha256='d'*64)
        directory=self.root/'observations/typed-recovery';directory.mkdir(parents=True)
        consumed=dict(status='held-CPU0',source=publication,actual_child_exit_code=0,goal_complete=False,
            original_accounting_remains_failed=True,
            native4_restore_and_all_archives_and_independent_final_still_required=True,
            failure=dict(kind='ledger-only',producer=binding,fingerprint='e'*64),
            result_path=str(directory/'result.json'),result_sha256='f'*64,
            output=dict(path=str(directory/'consumer.stdout'),sha256='1'*64,bytes=100),
            structural_acceptance=dict(path=str(directory/'structural-acceptance.json'),sha256='2'*64,bytes=200))
        helper=SimpleNamespace(HELD_STATUS='held-CPU0',FAILURE_KIND='ledger-only',DIRECTORY=directory,
            consume=lambda commit:copy.deepcopy(consumed))
        with patch.object(control,'producer',return_value=(self.run,binding,raw)) as producer, \
                patch.object(reader,'BASE',self.base),patch.object(reader,'RUN',self.run_id), \
                patch.object(reader,'PRODUCER',self.commit), \
                patch.object(reader,'load_fixed',return_value=(helper,publication)) as load:
            run,actual,accounting,ref=control.final_producer(self.base,recovery_commit=recovery_commit)
            producer.assert_called_once_with(self.base,complete=False)
            load.assert_called_once_with(recovery_commit,'final_cpu_ledger_recovery719225','ctl')
            self.assertEqual(accounting,raw);self.assertEqual(actual,binding)
            self.assertEqual(ref['original_producer_state'],'FAILED');self.assertEqual(ref['original_producer_exit_code'],'1:0')
            self.assertFalse(ref['goal_complete'])
            for key,bad in (('goal_complete',True),('actual_child_exit_code',1),
                    ('original_accounting_remains_failed',False),('source',dict(commit='0'*40))):
                original=copy.deepcopy(consumed);consumed[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):
                    control.final_producer(self.base,recovery_commit=recovery_commit)
                consumed.clear();consumed.update(original)
            for bad_raw in (raw.replace('FAILED','RUNNING'),raw.replace('1:0','2:0')):
                producer.return_value=(self.run,binding,bad_raw)
                with self.assertRaisesRegex(ValueError,'cannot replace healthy'):
                    control.final_producer(self.base,recovery_commit=recovery_commit)

    def test_normal_final_producer_never_loads_a_recovery_summary(self):
        from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader
        with patch.object(control,'producer',return_value=(self.run,{},'actual COMPLETED0')) as producer, \
                patch.object(reader,'load_fixed') as load:
            self.assertEqual(control.final_producer(self.base)[-1],None)
            producer.assert_called_once_with(self.base,complete=True);load.assert_not_called()

    def test_typed_cpu_controller_and_native_gpu_entry_have_distinct_published_sources(self):
        repository,binding,proof,root=self.audit_fixture()
        native=control.RECOVERY_NATIVE_AUDIT_SOURCE
        proof.update(native_audit_source=dict(commit=native),producer_final_cpu_recovery=dict(commit='c'*40))
        cpu_birth=dict(pid=99,start_ticks=101,sid=99,uid=control.OWNER,state='S')
        test=subprocess.CompletedProcess(['sbatch'],0,'test only accepted','')
        submitted=subprocess.CompletedProcess(['sbatch'],0,'67890\n','')
        with patch.object(control,'final_ready',return_value=(repository,self.run,binding,proof)), \
                patch.object(control,'process',return_value=cpu_birth), \
                patch.object(control.subprocess,'run',side_effect=[test,submitted]) as submit:
            result=control.submit_audit(self.base,self.commit,recovery_commit='c'*40)
        self.assertEqual(result['job'],'67890')
        claim=control.read(root/'final-audit/submit-claim.json');cmd=claim['command']
        self.assertEqual(cmd[-5],str(self.root/'source'/native/'reproduction/hpc3_qprobe/final_restore_audit.sh'))
        self.assertEqual(cmd[-3],native);self.assertEqual(claim['proof']['source']['commit'],self.commit)
        self.assertEqual(claim['recovery_commit'],'c'*40);self.assertIn('remains FAILED1:0',claim['dependency_policy'])
        self.assertIn('--gres=gpu:4',cmd);self.assertEqual(submit.call_count,2)

    def test_recovery_claim_cannot_replay_an_existing_normal_or_unknown_native_intent(self):
        repository,binding,proof,root=self.audit_fixture();(root/'final-audit').mkdir()
        proof['producer_final_cpu_recovery']=dict(commit='c'*40)
        with patch.object(control,'final_ready',return_value=(repository,self.run,binding,proof)), \
                patch.object(control.subprocess,'run') as submit,self.assertRaisesRegex(ValueError,'already attempted'):
            control.submit_audit(self.base,self.commit,recovery_commit='c'*40)
        submit.assert_not_called()

    def test_recovery_preflight_rechecks_all_five_archives_and_binds_original_native_science(self):
        from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader
        from reproduction.hpc3_qprobe import verify_goal as goal
        from reproduction.hpc3_qprobe import final_restore_audit as native
        from search_g1 import qprobe_saved_checkpoint
        _,binding,_=self.producer();repository=self.root/'source'/self.commit
        structural=self.root/'separate/structural-acceptance.json'
        self.write(structural,dict(structural_accepted=True,actual_outer_updates=200,run_id=self.run_id))
        recovery=dict(commit='c'*40,structural_acceptance=dict(path=str(structural),sha256=control.digest(structural)))
        self.write(self.base/'logs/pipeline-status.json',dict(run_id=self.run_id,stage='FAILED'))
        self.write(self.base/'logs/cleanup-status.json',dict(run_id=self.run_id,cleaned=True))
        self.write(self.base/'logs/final-identity.json',dict(run_id=self.run_id,step=200,final=True,
            question_ids=['q']*51713,em=[0]*51713,f1=[0.0]*51713,counts=dict(nq=51707,a=1,b=1,c=1,d=1,e=1,f=1)))
        rows=[dict(step=i,run_id=self.run_id) for i in range(1,201)]
        (self.base/'logs/actual-updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        cpu_publication=dict(commit=self.commit,publication_path=str(self.root/'cpu-source.json'))
        self.write(Path(cpu_publication['publication_path']),dict(files={'observer-only':'changed'}))
        native_publication=dict(commit=control.RECOVERY_NATIVE_AUDIT_SOURCE,
            publication_path=str(self.root/'native-source.json'))
        protected={'reproduction/common8/runtime.py':'original protected source SHA'}
        self.write(Path(native_publication['publication_path']),dict(files=protected))
        target=self.root/'target.json';self.write(target,{})
        transports={}
        for step in control.STEPS:
            path,_=self.receipt(step);transport=self.root/('transport-'+str(step)+'.json')
            self.write(transport,dict(source_receipt=dict(path=str(path),sha256=control.digest(path))))
            transports[step]=dict(actual_task_exit_code=0,status='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0',
                transport_path=str(transport),fresh_full_sha=dict(path=str(transport),target_receipt=dict(path=str(target))))
        with patch.object(control,'source',return_value=(repository,cpu_publication)), \
                patch.object(control,'final_producer',return_value=(self.run,binding,'actual FAILED1:0',recovery)), \
                patch.object(qprobe_saved_checkpoint,'history_parts',return_value=['authentic original parts']), \
                patch.object(goal,'part_updates',return_value=(rows,['original history proof'])), \
                patch.object(control,'inspect_archive',side_effect=lambda base,step,full:transports[step]) as archives, \
                patch.object(reader,'publication',return_value=(self.root/'native-source',native_publication)) as source, \
                patch.object(native,'bind_audit') as bind:
            result=control.final_ready(self.base,self.commit,recovery_commit='c'*40)
            self.assertEqual([call.args[1] for call in archives.call_args_list],[5,50,100,150,200])
            self.assertTrue(all(call.kwargs['full'] is True for call in archives.call_args_list))
            source.assert_called_once_with(control.RECOVERY_NATIVE_AUDIT_SOURCE)
            self.assertEqual(bind.call_args.args[4],control.RECOVERY_NATIVE_AUDIT_SOURCE)
            self.assertEqual(bind.call_args.args[5],protected)
            self.assertEqual(result[-1]['actual_producer_accounting'],'actual FAILED1:0')
            self.assertEqual(result[-1]['full_run_acceptance_path'],str(structural))
            bind.reset_mock();transports[100]['actual_task_exit_code']=1
            with self.assertRaisesRegex(ValueError,'all five actual completed'):
                control.final_ready(self.base,self.commit,recovery_commit='c'*40)
            bind.assert_not_called()

    def test_recovery_reconcile_uses_actual_EA_native_entry_and_same_once_reference(self):
        _,binding,proof,root=self.audit_fixture();directory=root/'final-audit';directory.mkdir()
        native=control.RECOVERY_NATIVE_AUDIT_SOURCE;proof['native_audit_source']=dict(commit=native)
        reference=dict(commit='c'*40,failure_fingerprint='d'*64)
        audit_base=self.root/'audit-base'
        claim=dict(producer=binding,audit_base=str(audit_base),proof=proof,token='token',
            recovery_commit='c'*40,producer_final_cpu_recovery=reference)
        control.publish(directory/'submit-claim.json',claim)
        self.write(directory/'submission.json',dict(returncode=None,status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY'))
        detail=('JobId=67890 UserId=jxu957(14090) Account=jxu957 QOS=formal-user Partition=emergency_acd '+
            'JobName=qprobe-final200audit WorkDir='+str(audit_base)+' Command='+
            str(self.root/'source'/native/'reproduction/hpc3_qprobe/final_restore_audit.sh')+
            ' NumCPUs=48 MinMemoryNode=512G NumNodes=1 TresPerNode=gres:gpu:4 JobState=PENDING NodeList=(null)')
        with patch.object(control,'final_producer',return_value=(self.run,binding,'actual FAILED1:0',reference)), \
                patch.object(control,'slurm_query',side_effect=[detail,'']) as query, \
                patch.object(control.subprocess,'run') as submit:
            result=control.reconcile_audit(self.base,'67890',recovery_commit='c'*40)
            self.assertEqual(result['status'],'ACTUAL_SLURM_RECONCILED');submit.assert_not_called()
            self.assertEqual([call.args[0][0] for call in query.call_args_list],['scontrol','squeue'])
            with self.assertRaisesRegex(ValueError,'intent'):
                control.inspect_audit(self.base,recovery_commit='e'*40)

    def test_cpu_intent_mismatch_is_rejected_before_native_intent_exists(self):
        _,binding,_,root=self.audit_fixture();directory=root/'audit-controller';directory.mkdir()
        reference=dict(commit='c'*40,failure_fingerprint='d'*64)
        control.publish(directory/'submit-claim.json',dict(producer=binding,recovery_commit='wrong',
            producer_final_cpu_recovery=reference))
        with patch.object(control,'final_producer',return_value=(self.run,binding,'actual FAILED1:0',reference)), \
                patch.object(control,'controller_state') as state:
            with self.assertRaisesRegex(ValueError,'CPU submit producer/recovery intent'):
                control.inspect_audit(self.base,recovery_commit='c'*40)
            state.assert_not_called()

    def test_short_audit_start_detaches_before_any_big_sha_preflight(self):
        repository=self.root/'source'/self.commit
        publication=dict(commit=self.commit,publication_path='/fixture/publication',publication_sha256='c'*64)
        leader=dict(pid=41,start_ticks=71,sid=41,uid=control.OWNER,state='S')
        child_birth=dict(pid=42,start_ticks=72,sid=42,uid=control.OWNER,state='S')
        with patch.object(control,'source',return_value=(repository,publication)), \
             patch.object(control,'slurm_query',return_value=self.raw.replace('RUNNING','COMPLETED')), \
             patch.object(control,'process',side_effect=[leader,child_birth]), \
             patch.object(control.subprocess,'Popen',return_value=type('Child',(),{'pid':42})()) as popen, \
             patch.object(control,'final_ready') as heavy,patch.object(control,'submit_audit') as slurm:
            result=control.start_audit(self.base,self.commit)
        self.assertEqual(result['status'],'ACTUAL_DETACHED_FINAL_SUBMIT_CPU_STARTED');heavy.assert_not_called();slurm.assert_not_called()
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertIn('audit-worker',popen.call_args.args[0])
        self.assertTrue((self.root/'final-control'/self.run_id/'audit-controller/submit-claim.json').is_file())

    def test_cpu_submit_worker_keeps_unknown_slurm_outcome_distinct_from_its_own_success(self):
        directory=self.root/'audit-test';directory.mkdir();claim_path=directory/'submit-claim.json'
        claim=dict(producer={'base':str(self.base)},source={'commit':self.commit});control.publish(claim_path,claim)
        unknown=dict(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY',returncode=None)
        actual=dict(pid=42,start_ticks=72,sid=41,uid=control.OWNER,state='S')
        with patch.object(control,'worker_binding',return_value=(claim,directory)), \
             patch.object(control,'process',return_value=actual),patch.object(control,'submit_audit',return_value=unknown):
            result=control.audit_exec(claim_path,'token')
        self.assertEqual(result,unknown)
        task=control.read(directory/'task-result.json')
        self.assertEqual(task['submission'],unknown);self.assertIsNone(task['submission']['returncode'])
        self.assertFalse(task['training_performed']);self.assertFalse(task['direct_gpu_device_access_performed'])
        self.assertTrue(task['native_gpu_audit_submission_attempted'])

    def test_unknown_audit_reconcile_checks_actual_owner_source_workdir_and_resources(self):
        repository,binding,proof,root=self.audit_fixture();directory=root/'final-audit';directory.mkdir()
        audit_base=str(self.top/'qprobe-teacher-warmup200-final200-audit-fixture')
        claim=dict(producer=binding,token='token',audit_base=audit_base,proof=proof)
        control.publish(directory/'submit-claim.json',claim)
        observed=dict(JobId='56789',UserId='jxu957(14090)',Account='jxu957',QOS='formal-user',Partition='emergency_acd',
            JobName='qprobe-final200audit',WorkDir=audit_base,Command=str(repository/'reproduction/hpc3_qprobe/final_restore_audit.sh'),
            NumCPUs='48',MinMemoryNode='512G',NumNodes='1',TresPerNode='gres:gpu:4',JobState='PENDING',NodeList='(null)')
        for field,bad in [('UserId','mou176(21930)'),('WorkDir','/other'),('NumCPUs','96'),('TresPerNode','gres:gpu:8'),('NodeList','ACD1-31')]:
            changed=dict(observed);changed[field]=bad
            with patch.object(control,'producer',return_value=(self.run,binding,self.raw)), \
                 patch.object(control,'slurm_query',return_value=' '.join(k+'='+v for k,v in changed.items())), \
                 patch.object(control.subprocess,'run') as submit,self.assertRaises(ValueError):control.reconcile_audit(self.base,'56789')
            submit.assert_not_called();self.assertFalse((directory/'reconciled-submission.json').exists())
        detail=' '.join(k+'='+v for k,v in observed.items())
        duplicate=dict(observed,JobId='56790')
        with patch.object(control,'producer',return_value=(self.run,binding,self.raw)), \
             patch.object(control,'slurm_query',side_effect=[detail,'56790|qprobe-final200audit\n',
                        ' '.join(k+'='+v for k,v in duplicate.items())]), \
             self.assertRaisesRegex(ValueError,'more than one actual job'):control.reconcile_audit(self.base,'56789')
        self.assertFalse((directory/'reconciled-submission.json').exists())
        with patch.object(control,'producer',return_value=(self.run,binding,self.raw)), \
             patch.object(control,'slurm_query',side_effect=[detail,'56789|qprobe-final200audit\n']), \
             patch.object(control.subprocess,'run') as submit:result=control.reconcile_audit(self.base,'56789')
        self.assertEqual(result['status'],'ACTUAL_SLURM_RECONCILED');self.assertIsNone(result['returncode']);submit.assert_not_called()


if __name__=='__main__':unittest.main()
