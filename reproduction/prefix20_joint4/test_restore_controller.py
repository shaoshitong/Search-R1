"""CPU receipt/controller regressions only; no CUDA/model reload simulation."""
import copy
from contextlib import nullcontext
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FILE = Path(__file__).with_name('restore_controller.py')
spec = importlib.util.spec_from_file_location('draft_restore_controller', FILE)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_fixture(source):
    commit, run_id = 'a'*40, 'b'*32
    run = dict(run_id=run_id, source_commit=commit, training_world_size=4, target_updates=200,
               data_selection_mode='prefix20_probe')
    updates = [dict(step=i, run_id=run_id) for i in range(1, 201)]
    cpu = dict(kind='prefix20_joint4_independent_cpu_recorded_evidence_audit', accepted=True,
        status='RECORDED_EVIDENCE_PASS', source_base=str(source), source_commit=commit, run_id=run_id,
        torch_imported=False, model_reloads=0, gpu_allocations=0, training_updates=0,
        goal_complete=False,
        actual_history=dict(completed_outers=200, no_update201=True, actual_actor_adam_writes_per_rank=3812,
                            candidate_questions_consumed=201*5120),
        evaluations=dict(fixed_boundaries=[dict(step=s) for s in control.archive.STEPS],
                         final_native51713=dict(rows=51713, per_source={str(i):{} for i in range(7)})),
        saved_checkpoints=[dict(outer=s,full_recorded_checkpoint_sha_checked=True) for s in control.archive.STEPS])
    return run, updates, cpu, commit, run_id


def actual_fixture():
    source = '/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-final'
    gate = dict(source_base=source,source_commit='a'*40,source_run_id='b'*32,
                actual_actor_writes=3812,candidate_questions_consumed=201*5120,
                source_attempt_progress=dict(completed_outer=200,cumulative_writes=3812),source_hf_saved_tensor_count=123)
    allocation=dict(accepted=True,run_id=gate['source_run_id'],training_world_size=4,slurm_job_id='999',actual_cuda_runtime_uuids=['GPU-'+str(i) for i in range(4)])
    membership=dict(pending_rows_verified=64,pending_offset=12,consumed_source_rows=gate['candidate_questions_consumed'],
                    candidate_cursor=256,epoch=3,ordered_membership_sha256='d'*64)
    driver=dict(pid=100,start_ticks=300,uid=14090,sid=400,state='S')
    def readback(phase):
        actors=[]
        for rank in range(4):
            actors.append(dict(rank=rank,phase=phase,actual_cuda_uuid=allocation['actual_cuda_runtime_uuids'][rank],
                process_identity=dict(pid=110+rank,start_ticks=310+rank,uid=14090,sid=410+rank,state='S'),
                actual_optimizer_writes=3812,actual_counters=dict(completed_outer=200),
                scheduler_last_epoch=200,scheduler_step_count=201,
                model_hf_exact_tensors_checked_on_rank0=123 if rank==0 else 0,
                model_equal=True,optimizer_equal=True,scheduler_equal=True,boundary_rng_equal=True,forbidden_call_counts={}))
        probes=[dict(rank=r,completed_outer=200,actual_steps=2000,model_optimizer_rng_exact_equal=True,
                     private_rng_activated_and_equal=True,actor_rng_unchanged=True,forbidden_call_counts={}) for r in range(4)]
        return dict(role=control.ROLE,phase=phase,source_step=200,source_run_id=gate['source_run_id'],
            source_receipt_sha256='c'*64,actual_actor_readbacks=actors,actual_probe_readbacks=probes,
            candidate_dataequal=True,selector_rng_equal=True,driver_rng_equal=True,eligible_mapping_equal=True,
            partial_pending_membership=membership,verified_saved_candidate_membership=membership,
            source_attempt_progress=dict(completed_outer=200,cumulative_writes=3812),driver_process_identity=driver,
            optimizer_write_increment=0,scheduler_advance_increment=0,probe_update_increment=0,candidate_consumption_increment=0,
            generation_calls=0,training_checkpoint_save_calls=0,independent_second_readback_equal=phase=='B',
            checked_unix=1 if phase=='A' else 2)
    a,b=readback('A'),readback('B')
    complete=dict(role=control.ROLE,accepted=True,completed_outer=200,actual_native4_cuda_restore=True,
        actual_two_independent_readbacks=True,no_update201=True,all_update_consumption_increments=0,actual_slurm_job_id='999')
    released=dict(role=control.ROLE,all_recorded_processes_terminal=True,assigned_gpu_contexts_released=True,
                  recorded_process_births=[r['process_identity'] for r in b['actual_actor_readbacks']]+[driver])
    return dict(a=a,b=b,complete=complete,released=released,allocation=allocation,gate=gate,
                target=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-restore200-cpu'),
                restore_commit='e'*40,job='999',now=3)


class SourceAndBudgetTests(unittest.TestCase):
    def test_real_local_archive_bytes_and_original_wrapper_are_consumed(self):
        from reproduction.prefix20_joint4.test_checkpoint_archive import Fixture,archive
        with tempfile.TemporaryDirectory() as directory:
            fixture=Fixture(directory,step=200)
            try:
                snapshot=fixture.snapshot();target=fixture.mirror(snapshot);wrapper=fixture.archive(snapshot,target)
                archive.accept_archive(fixture.base,snapshot['snapshot_file'],snapshot['snapshot_sha256'],wrapper,
                                       commit=fixture.commit,run_id=fixture.run_id,step=200)
                imported=Path(snapshot['snapshot_file']).parent/'archive-receipt-import.json'
                write(imported,wrapper)
                with patch.object(control,'archive',archive),patch.object(archive,'STEPS',(200,)):
                    records,final=control.archive_gate(fixture.base,fixture.commit,fixture.run_id)
                    self.assertEqual(final['step'],200);self.assertEqual(len(records),1)
                    self.assertEqual(records[0]['files'],len(snapshot['snapshot']['files']))
                    self.assertFalse(final['actual_cuda_restore_accepted'])
                    bad=copy.deepcopy(wrapper);bad['archive_receipt_sha256']='0'*64;write(imported,bad)
                    with self.assertRaisesRegex(ValueError,'RECEIPT_SHA'):
                        control.archive_gate(fixture.base,fixture.commit,fixture.run_id)
            finally:fixture.close()

    def test_actual_200_and_native51713_gate_does_not_require_goalcomplete(self):
        source=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-final')
        run,updates,cpu,commit,rid=source_fixture(source)
        self.assertFalse(cpu['goal_complete'])
        self.assertEqual(control.history_gate(run,updates,cpu,source=source,commit=commit,run_id=rid)['completed_outers'],200)

    def test_missing200_extra201_wrong_run_or_incomplete_fixed_final_reject(self):
        source=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-final')
        for case in ('missing200','201','run','boundaries','51720','seven','checkpoint','torch','actorwrites'):
            run,updates,cpu,commit,rid=source_fixture(source)
            if case=='missing200':updates.pop()
            elif case=='201':updates.append(dict(step=201,run_id=rid))
            elif case=='run':run['source_commit']='f'*40
            elif case=='boundaries':cpu['evaluations']['fixed_boundaries'].pop()
            elif case=='51720':cpu['evaluations']['final_native51713']['rows']=51720
            elif case=='seven':cpu['evaluations']['final_native51713']['per_source'].pop('6')
            elif case=='checkpoint':cpu['saved_checkpoints'][0]['full_recorded_checkpoint_sha_checked']=False
            elif case=='torch':cpu['torch_imported']=True
            else:cpu['actual_history']['actual_actor_adam_writes_per_rank']=0
            with self.subTest(case=case),self.assertRaises(ValueError):
                control.history_gate(run,updates,cpu,source=source,commit=commit,run_id=rid)

    def test_missing200_rejects_before_archive_or_slurm_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory).resolve()/'qprobe-teacher-warmup200-prefix20-joint4-final'
            run,updates,cpu,commit,rid=source_fixture(source)
            write(source/'logs/run-manifest.json',run)
            (source/'logs/actual-updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in updates[:-1]))
            path=source/'audit.json';write(path,cpu)
            with patch.object(control,'namespace',return_value=(source,None)),patch.object(control.archive,'binding'),\
                    patch.object(control,'archive_gate') as archives,patch.object(control,'terminal_job') as slurm:
                with self.assertRaisesRegex(ValueError,'COMPLETE200'):
                    control.source_gate(source,commit,rid,path,control.sha(path))
                archives.assert_not_called();slurm.assert_not_called()

    def test_storage_requires_actual_complete_fresh_upper_bound_and_2tb(self):
        proof=dict(measurement_complete=True,account_experiment_root=str(control.PERSONAL_ROOT),checked_unix=100,
                   allocated_bytes_upper_bound=10**12)
        value=control.budget_projection(proof,10**11,10000,now=101)
        self.assertEqual(value['personal_budget_bytes'],2*10**12)
        self.assertEqual(value['unique_hpc3_restore_target_copies'],1)
        self.assertLess(value['projected_bytes'],2*10**12)
        for case in ('incomplete','stale','future','overbudget','missingactual'):
            bad=copy.deepcopy(proof)
            if case=='incomplete':bad['measurement_complete']=False
            elif case=='stale':bad['checked_unix']=-22000
            elif case=='future':bad['checked_unix']=102
            elif case=='overbudget':bad['allocated_bytes_upper_bound']=19*10**11
            else:bad.pop('allocated_bytes_upper_bound')
            with self.subTest(case=case),self.assertRaises(ValueError):control.budget_projection(bad,10**11,10000,now=101)

    def test_accounting_running_failed_foreign_or_queued_is_not_completion(self):
        line='999|COMPLETED|0:0|jxu957|14090|jxu957|formal-user|prefix20-restore200-only|2026-10-20T01:00:00|2026-10-20T01:01:00|ACD1-1\n'
        def account(raw=line,queue=''):
            return [dict(stdout=raw),dict(stdout=queue)]
        with patch.object(control,'query',side_effect=account()):self.assertTrue(control.terminal_job('999','prefix20-restore200-only')['actual_terminal'])
        for raw,queue in ((line.replace('COMPLETED','RUNNING'),''),(line.replace('0:0','1:0'),''),
                          (line.replace('14090','1'),''),(line,'999\n'),(line.replace('restore200-only','other'),'')):
            with patch.object(control,'query',side_effect=account(raw,queue)),self.assertRaises(ValueError):
                control.terminal_job('999','prefix20-restore200-only')


class SubmissionTests(unittest.TestCase):
    def test_launch_is_four_gpus_emergency_restore_only_with_no_dependency_or_training(self):
        cmd=control.command_for(Path('/target'), 'a'*40)
        self.assertIn('--gres=gpu:4',cmd);self.assertNotIn('--gres=gpu:8',cmd)
        self.assertIn('--partition=emergency_acd',cmd)
        self.assertIn('--account=jxu957',cmd);self.assertIn('--export=NONE',cmd)
        self.assertTrue(cmd[-2].endswith('/restore_only.sh'))
        self.assertFalse(any('train.sh' in value or 'dependency=' in value for value in cmd))

    def test_unknown_sbatch_is_claimed_once_and_cannot_retry(self):
        from reproduction.prefix20_joint4 import control as old_control
        from reproduction.prefix20_joint4 import restore_only_assets as assets
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'preparation').mkdir();(base/'logs').mkdir()
            copy_path=base/'preparation/restore-copy-complete.json';write(copy_path,dict(copied_bytes=1))
            gate=dict(source_base='/source',source_commit='a'*40,source_run_id='b'*32,
                      cpu_final_audit=dict(path='/source/audit.json',sha256='c'*64))
            permit=dict(source_gate=gate,restore_source_commit='d'*40,registration_sha256='e'*64)
            write(base/control.PREPARED,dict(copy_receipt=dict(sha256=control.sha(copy_path))))
            with patch.object(control,'registry_lock',return_value=nullcontext()),\
                 patch.object(control,'registered',return_value=(permit,{},{})),\
                 patch.object(control,'source_gate',return_value=(gate,{})),\
                 patch.object(control,'budget_gate',return_value={}),\
                 patch.object(old_control,'verify_source',return_value=(base,dict(files={}))),\
                 patch.object(old_control,'bind_repo'),patch.object(assets,'verify_target_copy'),\
                 patch.object(control.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',30)) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):control.submit(base)
                self.assertEqual(launch.call_count,1)
                unknown,_=control.document(base/control.SUBMISSION)
                self.assertEqual(unknown['outcome'],'UNKNOWN_INSPECT_SLURM_NO_RETRY')
                self.assertTrue((base/control.SUBMIT_CLAIM).is_file())
                with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):control.submit(base)
                self.assertEqual(launch.call_count,1)

    def test_publication_is_exclusive_and_never_replaces_first_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'claim.json'
            control.publish(path,dict(first=True))
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):control.publish(path,dict(first=False))
            self.assertEqual(path.read_bytes(),before)


class PositivePermitFlowTests(unittest.TestCase):
    """Real small receipts, SHA reads, source copies, registration and permits.

    Only remote namespaces, Slurm, storage observation/environment bootstrap,
    and the heavyweight tensor-copy adapter are substituted. Source history,
    all five archive gates, source publication/repo binding, registry lock,
    immutable controller claims and every permit validator run unchanged.
    """
    def setUp(self):
        from reproduction.prefix20_joint4.test_checkpoint_archive import Fixture,archive
        from reproduction.prefix20_joint4 import control as source_control
        from reproduction.prefix20_joint4 import restore_only_assets as assets
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        f=Fixture(temporary.name,step=200);self.addCleanup(f.close)
        self.f,self.assets,self.source_control=f,assets,source_control
        self.source=f.base;self.target=f.personal/'qprobe-teacher-warmup200-prefix20-restore200-permit-fixture'
        self.restore_commit='e'*40;self.job='727900';self.copy_calls=[]
        # Extend the independent core's tiny byte graph to all five real
        # source/archive/source receipts in one immutable source namespace.
        for step in archive.STEPS:
            if step!=200:
                cp=self.source/'checkpoints/search-g1-common8/actor'/('global_step_'+str(step))
                pair=self.source/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))
                shutil.copytree(f.cp,cp);shutil.copytree(f.pair,pair)
                receipt=copy.deepcopy(f.receipt);attempts=['%032x'%i for i in range(1,step+1)]
                reward=copy.deepcopy(f.reward);round_id=min(step//50,3)*50
                reward.update(round=round_id,version=f.run_id+':'+str(round_id)+':fixture')
                f.write_json(pair/'reward-probe-manifest.json',reward)
                receipt.update(step=step,checkpoint=str(cp),actor_snapshot_id=f.run_id+':'+str(step),
                    question_probe_state_path=str(cp/'training_state/question-probe-state.pt'),
                    head_path=str(cp/'training_state/prefix20-rank-0.pt'),head_version='prefix20-epoch68-outer-%04d'%step,
                    reward_probe_version=reward['version'],reward_probe_round=round_id,
                    attempt_progress=dict(completed_outer=step,attempt_index=step,cumulative_writes=step*20,
                        aborted_attempts=0,full_budget_outer=step),driver_state=dict(completed_outer=step,source_consumed=step*5120))
                receipt['files']=[f.file_record(cp/row['relative_path'],relative_path=row['relative_path']) for row in f.receipt['files']]
                for name in ('source_run_manifest','boundary_driver','reward_probe_manifest'):
                    receipt[name]=f.file_record(pair/Path(f.receipt[name]['path']).name)
                receipt['boundary_worker_files']=[dict(f.file_record(pair/'boundary-worker-rng'/('rank-%d.pt'%rank)),rank=rank) for rank in range(4)]
                receipt['ranks']=[dict(row,completed_outer=step,scheduler_last_epoch=step,scheduler_step_count=step+1,
                    actual_optimizer_writes=step*20,last_attempt_index=step,used_attempt_ids=attempts) for row in f.receipt['ranks']]
                f.write_json(pair/'source-receipt.json',receipt)
            f.step=step;snapshot=f.snapshot();mirrored=f.mirror(snapshot);wrapper=f.archive(snapshot,mirrored)
            archive.accept_archive(self.source,snapshot['snapshot_file'],snapshot['snapshot_sha256'],wrapper,
                commit=f.commit,run_id=f.run_id,step=step)
            archive.publish(Path(snapshot['snapshot_file']).parent/'archive-receipt-import.json',wrapper)
        f.step=200
        _,updates,cpu,_,_=source_fixture(self.source)
        updates=[dict(row,run_id=f.run_id) for row in updates]
        cpu.update(source_commit=f.commit,run_id=f.run_id)
        cpu['actual_history'].update(actual_actor_adam_writes_per_rank=4000,candidate_questions_consumed=200*5120)
        write(self.source/'logs/run-manifest.json',f.run)
        (self.source/'logs/actual-updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in updates))
        write(self.source/'logs/cleanup-status.json',dict(run_id=f.run_id,cleaned=True))
        (self.source/'job.id').write_text('727800\n')
        write(self.source/'preparation/training-submission.json',dict(job='727800',source_commit=f.commit,returncode=0))
        self.cpu_path=self.source/'preparation/cpu-final-audit.json';write(self.cpu_path,cpu);self.cpu_sha=control.sha(self.cpu_path)
        runtime=self.source/'source'/self.restore_commit;runtime.mkdir(parents=True)
        files={}
        for name,raw in (('tiny.py',b'# constructed published source\n'),
                         ('reproduction/prefix20_joint4/restore_only.sh',b'#!/bin/sh\nexit 0\n')):
            path=runtime/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            files[name]=dict(bytes=len(raw),sha256=control.sha(path))
        write(runtime/'source-publication.json',dict(source_commit=self.restore_commit,
            fresh_source_all_files_sha_verified=True,files=files))
        self.runtime=runtime

        def environment(base):
            (base/'logs').mkdir(exist_ok=True)
            return dict(constructed_local_environment=True)
        def budget(source,snapshot,manifest):
            self.assertEqual(source,self.source)
            return control.budget_projection(dict(measurement_complete=True,account_experiment_root=str(f.personal),
                checked_unix=control.time.time(),allocated_bytes_upper_bound=10**6),snapshot['bytes'],
                sum(row['bytes'] for row in manifest['files'].values()))
        def copy_target(base,source_receipt,commit):
            permit=control.validate_copy_permit(base,source_receipt,commit)
            self.assertTrue((self.source/control.REGISTRATION).is_file())
            self.copy_calls.append((str(base),str(source_receipt),commit))
            original=f.cp/'weights.safetensors';member=assets.copy_verified(original,base/'copied/tiny-state.bin',control.sha(original))
            result=dict(role=control.ROLE,target_base=str(base),copied_bytes=member['bytes'],files=[member],
                original_receipt_sha256=permit['source_gate']['source_pair_receipt_sha256'])
            path=base/'preparation/restore-copy-complete.json';control.publish(path,result)
            control.publish(base/'logs/run-manifest.json',dict(source_commit=commit))
            return path
        def verify_copy(base,path):
            value,_=control.document(path)
            self.assertEqual(value['target_base'],str(base))
            for row in value['files']:
                actual=assets.record(Path(row['path']))
                if any(actual[k]!=row[k] for k in ('inode','bytes','mtime_ns','full_read_sha256')):
                    raise ValueError('CONSTRUCTED_COPY_ADAPTER_MEMBER_CHANGED')
            return value
        def query(command):
            self.assertEqual(command,['du','-s','-B1',str(self.target)])
            allocated=sum(p.stat().st_blocks*512 for p in self.target.rglob('*') if p.is_file())
            return dict(command=command,returncode=0,stdout=str(allocated)+'\t'+str(self.target)+'\n',stderr='')
        def terminal(job,name):
            self.assertEqual((job,name),('727800','qprobe-prefix20-joint200'))
            return dict(job=job,actual_terminal=True)
        def submit(command,**kwargs):
            self.assertEqual(command,control.command_for(self.target,self.restore_commit))
            self.assertTrue((self.target/control.SUBMIT_CLAIM).is_file())
            self.assertTrue((self.target/control.EXECUTE_PERMIT).is_file())
            return subprocess.CompletedProcess(command,0,self.job+';constructed-cluster\n','')
        patches=[patch.object(control,'archive',archive),patch.object(control,'PERSONAL_ROOT',f.personal),
            patch.object(source_control,'own_base',side_effect=archive.own_base),
            patch.object(source_control,'prepare_environment',side_effect=environment),
            patch.object(control,'budget_gate',side_effect=budget),patch.object(assets,'prepare',side_effect=copy_target),
            patch.object(assets,'verify_target_copy',side_effect=verify_copy),patch.object(control,'query',side_effect=query),
            patch.object(control,'terminal_job',side_effect=terminal),patch.object(control.subprocess,'run',side_effect=submit)]
        for item in patches:self.addCleanup(item.stop);item.start()

    def prepare(self):
        return control.prepare_target(self.source,self.f.commit,self.f.run_id,self.target,self.restore_commit,self.cpu_path,self.cpu_sha)

    def test_prepare_submit_check_execute_preserve_real_registration_and_sha_bindings(self):
        prepared=self.prepare();permit,registration,_=control.registered(self.target)
        self.assertEqual(len(prepared['source_gate']['archives']),5)
        self.assertEqual(permit['registration_sha256'],control.sha(self.source/control.REGISTRATION))
        self.assertEqual(registration['source_gate']['source_run_id'],self.f.run_id)
        self.assertEqual(permit['source_gate']['cpu_final_audit']['sha256'],self.cpu_sha)
        self.assertEqual((self.target/'repo/tiny.py').read_bytes(),(self.runtime/'tiny.py').read_bytes())
        self.assertEqual(prepared['copy_receipt']['sha256'],control.sha(self.target/'preparation/restore-copy-complete.json'))
        self.assertFalse(prepared['actual_cuda_restore_accepted']);self.assertEqual(len(self.copy_calls),1)
        submitted=control.submit(self.target);self.assertEqual(submitted['job'],self.job)
        execute,_=control.document(self.target/control.EXECUTE_PERMIT)
        self.assertEqual(execute['submit_claim_sha256'],control.sha(self.target/control.SUBMIT_CLAIM))
        self.assertEqual(execute['registration_sha256'],permit['registration_sha256'])
        self.assertEqual(execute['copy_receipt_sha256'],prepared['copy_receipt']['sha256'])
        with patch.dict(os.environ,SLURM_JOB_ID=self.job),patch.object(sys,'argv',
             ['restore_controller','check-execute','--base',str(self.target)]),patch.object(sys,'stdout',io.StringIO()) as out:
            control.main()
        checked=json.loads(out.getvalue());self.assertTrue(checked['controller_permit_validated'])
        self.assertFalse(checked['actual_cuda_restore_accepted']);self.assertNotIn('torch',sys.modules)
        with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):self.prepare()
        with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):control.submit(self.target)
        self.assertEqual(len(self.copy_calls),1)

    def test_registration_and_copy_permit_tampering_reject_before_submission(self):
        self.prepare();registration=self.source/control.REGISTRATION;permit=self.target/control.COPY_PERMIT
        for path,key,value in ((registration,'target_base',str(self.target)+'-other'),
                               (permit,'registration_sha256','0'*64),(permit,'restore_source_commit','f'*40)):
            original=path.read_bytes();changed=json.loads(original);changed[key]=value;write(path,changed)
            try:
                with self.subTest(key=key),self.assertRaises(ValueError):control.registered(self.target)
            finally:path.write_bytes(original)
        with self.assertRaisesRegex(ValueError,'COPY_PERMIT_SOURCE_OR_COMMIT_DIFFERS'):
            control.validate_copy_permit(self.target,self.source/'other-pair.json',self.restore_commit)
        self.assertFalse((self.target/control.SUBMIT_CLAIM).exists())

    def test_execution_refuses_tampered_claim_copy_receipt_permit_or_actual_job(self):
        self.prepare();control.submit(self.target)
        cases=((control.SUBMIT_CLAIM,'retry_allowed',True),
               ('preparation/restore-copy-complete.json','copied_bytes',999),
               (control.EXECUTE_PERMIT,'world_size',8),(control.EXECUTE_PERMIT,'training_permitted',True),
               (control.SUBMISSION,'job','999999'),(control.SUBMISSION,'submit_claim_sha256','0'*64))
        with patch.dict(os.environ,SLURM_JOB_ID=self.job):
            for relative,key,value in cases:
                path=self.target/relative;original=path.read_bytes();changed=json.loads(original);changed[key]=value;write(path,changed)
                try:
                    with self.subTest(file=relative,key=key),self.assertRaises(ValueError):
                        control.validate_execute_permit(self.target,self.restore_commit)
                finally:path.write_bytes(original)
            self.assertTrue(control.validate_execute_permit(self.target,self.restore_commit)['restore_only'])
        with patch.dict(os.environ,SLURM_JOB_ID='999999'),self.assertRaisesRegex(ValueError,'ACTUAL_SUBMITTED_RESTORE_JOB_DIFFERS'):
            control.validate_execute_permit(self.target,self.restore_commit)

    def test_global_registry_refuses_a_second_existing_target_and_preserves_original_claim(self):
        self.prepare();registration=self.source/control.REGISTRATION;original=registration.read_bytes()
        other=self.f.personal/'qprobe-teacher-warmup200-prefix20-restore200-other';other.mkdir()
        with self.assertRaisesRegex(ValueError,'MULTIPLE_OR_REPLACED'):control.registered(self.target)
        another_source=self.f.personal/'qprobe-teacher-warmup200-prefix20-joint4-other';another_source.mkdir()
        with self.assertRaisesRegex(ValueError,'ANOTHER_HPC3_RESTORE_TARGET_EXISTS'):
            control.prepare_target(another_source,self.f.commit,self.f.run_id,
                self.f.personal/'qprobe-teacher-warmup200-prefix20-restore200-third',self.restore_commit,self.cpu_path,self.cpu_sha)
        self.assertEqual(registration.read_bytes(),original);self.assertEqual(len(self.copy_calls),1)


class ActualAdapterTests(unittest.TestCase):
    def test_two_actual_receipts_adapt_without_allocating_or_importing_torch(self):
        value=control.adapt_actual_readbacks(**actual_fixture())
        self.assertTrue(value['accepted']);self.assertEqual(value['completed_outer'],200)
        self.assertEqual(len(value['actual_rank_readbacks']),4)
        self.assertTrue(all(r['readback_count']==2 for r in value['actual_rank_readbacks']))
        self.assertNotIn('torch',sys.modules)

    def test_mutation_missing_phase_model_rng_data_or_release_rejects(self):
        for case in ('optimizer','scheduler','probe','generation','save','consume','boolzero','twice','actor','modelcount',
                     'probe1999','privaterng','data','mapping','membership','cursor','source','birth','release','complete','201'):
            fixture=actual_fixture();row=fixture['b']
            if case in ('optimizer','scheduler','probe','generation','save','consume','boolzero'):
                key=dict(optimizer='optimizer_write_increment',scheduler='scheduler_advance_increment',probe='probe_update_increment',
                         generation='generation_calls',save='training_checkpoint_save_calls',consume='candidate_consumption_increment',
                         boolzero='generation_calls')[case]
                row[key]=False if case=='boolzero' else 1
            elif case=='twice':row['independent_second_readback_equal']=False
            elif case=='actor':row['actual_actor_readbacks'][2]['optimizer_equal']=False
            elif case=='modelcount':row['actual_actor_readbacks'][0]['model_hf_exact_tensors_checked_on_rank0']=0
            elif case=='probe1999':row['actual_probe_readbacks'][0]['actual_steps']=1999
            elif case=='privaterng':row['actual_probe_readbacks'][1]['private_rng_activated_and_equal']=False
            elif case=='data':row['candidate_dataequal']=False
            elif case=='mapping':row['eligible_mapping_equal']=False
            elif case=='membership':row['verified_saved_candidate_membership']=dict(row['partial_pending_membership'],pending_offset=13)
            elif case=='cursor':row['partial_pending_membership']=dict(row['partial_pending_membership'],consumed_source_rows=1)
            elif case=='source':row['source_receipt_sha256']='f'*64
            elif case=='birth':row['actual_actor_readbacks'][0]['process_identity']['pid']=9999
            elif case=='release':fixture['released']['assigned_gpu_contexts_released']=False
            elif case=='complete':fixture['complete']['actual_native4_cuda_restore']=False
            else:fixture['complete']['no_update201']=False
            with self.subTest(case=case),self.assertRaises(ValueError):control.adapt_actual_readbacks(**fixture)

    def test_adapter_schema_is_accepted_by_existing_cpu_consumer(self):
        from reproduction.prefix20_joint4.cpu_audit.audit_prefix20 import Reads,recovery
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            fixture=actual_fixture();fixture['gate']['source_base']=str(source)
            result=control.adapt_actual_readbacks(**fixture)
            path=root/'actual-recovery.json';write(path,result)
            run=dict(run_id=fixture['gate']['source_run_id'],source_commit=fixture['gate']['source_commit'])
            consumed=recovery(Reads(source),run,path,control.sha(path))
            self.assertTrue(consumed['accepted'])
            result['actual_rank_readbacks'][0]['probe_actual_adam_steps']=1999
            write(path,result)
            with self.assertRaises(ValueError):recovery(Reads(source),run,path,control.sha(path))
            self.assertNotIn('torch',sys.modules)

    def test_direct_copy_and_shell_execution_require_controller_before_reload(self):
        root=FILE.parent
        assets=(root/'restore_only_assets.py').read_text()
        start=assets.index('def prepare(')
        self.assertLess(assets.index('validate_copy_permit(base',start),assets.index('from search_g1.qprobe_checkpoint import read_source',start))
        execute=(root/'restore_only.py').read_text()
        start=execute.index('def execute(')
        self.assertLess(execute.index('validate_execute_permit(base',start),execute.index('allocation = admit(base)',start))
        shell=(root/'restore_only.sh').read_text()
        self.assertLess(shell.index('restore_controller check-execute'),shell.index('restore_only execute'))


if __name__=='__main__':unittest.main()
