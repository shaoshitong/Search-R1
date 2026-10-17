"""Actual small-file negative fixtures; never CUDA recovery or tensor evidence."""
import builtins
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import final_restore_acceptance as core


class FinalRestoreAcceptanceTests(unittest.TestCase):
    def put(self, path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        path.write_bytes(raw)
        return dict(path=str(path), bytes=len(raw), sha256=core.digest(raw))

    def fixture(self, directory, *, with_accessories=False):
        root = Path(directory).resolve(); self.base = root/'source'; self.target = root/'restore'
        self.base.mkdir(); self.target.mkdir(); self.run_id = 'a'*32
        self.run = dict(run_id=self.run_id, source_commit='b'*40, actor_micro_batch_per_rank=16,
            actor_logical_micro_batch_per_rank=4, training_world_size=4, question_probe=None,
            all_correct_virtual_zero=False,native_ray_execution_id='c'*32)
        source_run = self.put(self.base/'logs/run-manifest.json', self.run)
        self.cp = self.base/'checkpoints/search-g1-common8/actor/global_step_200'
        self.saved_progress = dict(completed_outer=200, cumulative_writes=3137, attempt_index=202,
                                  aborted_attempts=2, full_budget_outer=150)
        self.progress = dict(actual_completed_updates=200, actual_attempt_count=202, rank_count=4,
            cumulative_actual_optimizer_writes_per_rank=3137, no_update201=True,
            no_post200_driver_scheduler_worker_attempts=True,
            authenticated_history=dict(current_job='730961',current_source_commit=self.run['source_commit']))
        names = ['training_state/rank-'+str(rank)+'.pt' for rank in range(4)] + [
            'training_state/driver.pt', 'training_state/manifest.json', 'model.safetensors.index.json',
            'config.json','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors',
            'model-00003-of-00003.safetensors']
        # Actual CP5 saver:19 archived members,11 producer hashes,7 accessories.
        accessories = ['added_tokens.json','generation_config.json','merges.txt','special_tokens_map.json',
                       'tokenizer.json','tokenizer_config.json','vocab.json'] if with_accessories else []
        names += accessories
        members = {name: dict(path=str(self.cp/name),relative_path=name,bytes=100,
            full_read_sha256=core.digest(name.encode())) for name in names}
        states = [dict(rank=rank, path=members[f'training_state/rank-{rank}.pt']['path'],
            run_id=self.run_id, actual_optimizer_writes=3137,
            sha256=members[f'training_state/rank-{rank}.pt']['full_read_sha256']) for rank in range(4)]
        self.state = dict(completed_outer=200,question_probe=None,worker_files=states)
        state_ref = self.put(self.base/'small-archive/training-state-manifest.json', self.state)
        members['training_state/manifest.json'].update(bytes=state_ref['bytes'],full_read_sha256=state_ref['sha256'])
        self.complete = dict(run_id=self.run_id,step=200,checkpoint=str(self.cp),finite_all_saved_tensors=True,
            validated_tensor_count=435,actual_optimizer_writes_per_rank=3137,
            files_sha256={name: row['full_read_sha256'] for name,row in members.items() if name not in accessories})
        complete_ref = self.put(self.base/'small-archive/checkpoint-complete.json', self.complete)
        members['complete.json'] = dict(path=str(self.cp/'complete.json'),relative_path='complete.json',
                                       bytes=complete_ref['bytes'],full_read_sha256=complete_ref['sha256'])
        self.snapshot = dict(base=str(self.base),run_id=self.run_id,source_commit=self.run['source_commit'],step=200,
            files={'checkpoint/'+name:dict(bytes=row['bytes'],sha256=row['full_read_sha256']) for name,row in members.items()})
        snapshot_ref = self.put(self.base/'small-archive/source-snapshot.json', self.snapshot)
        self.archive = dict(base=str(self.base),run_id=self.run_id,source_commit=self.run['source_commit'],step=200,
            status=core.ARCHIVE_STATUS,fresh_destination_sha_verified=True,destination_fsynced=True,
            fresh_source_after_archive_sha_verified=True,snapshot_sha256=snapshot_ref['sha256'])
        archive_ref = self.put(self.base/'small-archive/accepted-archive.json', self.archive)
        self.checkpoint_summary = dict(step=200,source_checkpoint=str(self.cp),
            existing_full_file_sha_acceptance_authenticated=True,source_snapshot=snapshot_ref,
            accepted_archive=archive_ref,checkpoint_complete=complete_ref,training_state_manifest=state_ref)
        used = [f'{index:032x}' for index in range(1,203)]
        self.receipt = dict(status=core.SOURCE_STATUS,kind=core.ROLE,restore_only=True,step=200,world_size=4,
            source_base=str(self.base),checkpoint=str(self.cp),run_id=self.run_id,source_commit=self.run['source_commit'],
            current_source_commit=self.run['source_commit'],source_job='730961',actual_cuda_restore_accepted=False,
            question_probe=None,single_original_job_prefix=False,inherited_history_assembled=True,
            post_callback_rng_captured=False,reward_refresh_pending=False,no_update201=True,
            sampling_rng_stage='saved_checkpoint_before_final51713',checkpoint_archive=copy.deepcopy(self.checkpoint_summary),
            hf_saved_tensor_count=435,files=list(members.values()),file_count=len(members),
            attempt_progress=copy.deepcopy(self.saved_progress),actual_optimizer_writes_per_rank=3137,
            authenticated_history=copy.deepcopy(self.progress['authenticated_history']),
            ranks=[dict(rank=rank,completed_outer=200,scheduler_last_epoch=200,scheduler_step_count=201,
                actual_optimizer_writes=3137,last_attempt_index=202,used_attempt_ids=used,
                policy_rejections=9,optimizer_tensors_finite=True,full_budget_outers=150,
                optimizer_calls=3140,optimizer_rollbacks=3) for rank in range(4)],
            driver_state=dict(completed_outer=200,probe_round=150,probe_version='actual-round150-version',
                reward_refresh_pending=False,source_consumed=517120,pending_offset=7,actual_driver_rng_saved=True),
            driver_file=members['training_state/driver.pt'],reward_probe_round=150,
            reward_probe_version='actual-round150-version',source_run_manifest=source_run)
        self.source_path = self.base/'preparation/final200-source.json'
        self.source_ref = self.put(self.source_path,self.receipt)
        self.target_run = dict(self.run,source_commit='d'*40,restore_only_role=core.ROLE,
            native_ray_execution_id='e'*32,
            source_sha256={'reproduction/easy2560_joint4/final_restore_only.py':'d'*64},
            full_state_restore=dict(kind=core.ROLE,source_receipt=str(self.source_path),
                source_receipt_sha256=self.source_ref['sha256'],source_base=str(self.base),source_checkpoint=str(self.cp),
                completed_outer=200))
        self.put(self.target/'logs/run-manifest.json',self.target_run)
        published=dict(source_commit='d'*40,fresh_source_all_files_sha_verified=True,
            files={'reproduction/easy2560_joint4/final_restore_only.py':dict(bytes=123,sha256='d'*64)})
        self.put(self.target/'source'/('d'*40)/'source-publication.json',published)
        self.put(self.target/'preparation/run-source.json',dict(source_commit='d'*40,
            files=published['files'],runtime_repo=str(self.target/'repo')))
        self.allocation = dict(accepted=True,run_id=self.run_id,training_world_size=4,actual_job_state='RUNNING',
            slurm_job_id='799999',requested_gres='gres/gpu:4',node='ACD1-1',actual_allocated_indices=[0,1,2,3],
            actual_cuda_runtime_uuids=['GPU-fixture-'+str(rank) for rank in range(4)],
            actual_allocation_detail=' Nodes=ACD1-1 GRES=gpu:4(IDX:0-3) ',visible_devices=['0','1','2','3'],
            physical_gpu_inventory=[[str(rank),'GPU-fixture-'+str(rank),'80000',str(rank)] for rank in range(4)])
        allocation_ref = self.put(self.target/'logs/native-allocation-admission.json',self.allocation)
        self.host = dict(hostname='ACD1-1',boot_id='12345678-1234-1234-1234-123456789abc',pid_namespace='pid:[4026531836]')
        self.births = [dict(pid=rank+100,start_ticks=rank+1000,sid=99,uid=14090,state='S') for rank in range(5)]
        self.phases = {}; refs = {}
        for phase, stamp in (('A',10),('B',20)):
            rows = []
            for rank in range(4):
                saved = self.receipt['ranks'][rank]
                rows.append(dict(rank=rank,phase=phase,run_id=self.run_id,source_step=200,source_checkpoint=str(self.cp),
                    slurm_job_id='799999',source_receipt_sha256=self.source_ref['sha256'],
                    source_state_full_sha256=members[f'training_state/rank-{rank}.pt']['full_read_sha256'],
                    actual_cuda_uuid=self.allocation['actual_cuda_runtime_uuids'][rank],process_identity=self.births[rank],
                    actual_optimizer_writes=3137,scheduler_last_epoch=200,scheduler_step_count=201,
                    model_hf_exact_tensors_checked_on_rank0=435 if rank==0 else 0,
                    model_equal=True,optimizer_equal=True,scheduler_equal=True,saved_rng_equal=True,
                    actual_counters={key: saved[key] for key in core.COUNTERS},forbidden_call_counts={},checked_unix=stamp-1))
            self.phases[phase] = dict(schema_version=1,role=core.ROLE,phase=phase,run_id=self.run_id,
                source_receipt_sha256=self.source_ref['sha256'],host_identity=self.host,driver_process_identity=self.births[4],
                actual_actor_readbacks=rows,driver=dict(completed_outer=200,actual_driver_rng_readback_equal=True,
                    actual_candidate_stream_readback_equal=True,actual_candidate_source_cursor=517120,
                    attempt_progress=copy.deepcopy(self.saved_progress),pending_offset=7,eligible_mapping_equal=True,
                    actual_probe_round=150,actual_probe_version=self.receipt['reward_probe_version'],
                    reward_refresh_pending=False,question_probe=None,
                    source_driver_full_sha256=members['training_state/driver.pt']['full_read_sha256']),
                forbidden_call_counts={},independent_second_readback_equal=phase=='B',checked_unix=stamp,
                **{key:0 for key in core.ZERO})
            refs[phase] = self.put(self.target/'logs'/f'final-restore-only-readback-{phase}.json',self.phases[phase])
        result = dict(actual_all4_model_optimizer_scheduler_rng_dataequal=True,actual_cuda_restore_accepted=True,
                      actual_two_independent_readbacks=True,no_update201=True)
        self.completion = dict(role=core.ROLE,run_id=self.run_id,source_receipt_sha256=self.source_ref['sha256'],
            ray_get_returned=True,task_result=result,task_result_sha256=core.digest(core.canonical(result)),
            driver_process_identity=self.births[4],host_identity=self.host,checked_unix=21)
        completion_ref = self.put(self.target/'logs/final-restore-execution-completion.json',self.completion)
        self.release = dict(role=core.ROLE,all_recorded_processes_terminal=True,assigned_gpu_contexts_released=True,
            assigned_gpu_applications_after=[],host_identity=self.host,recorded_process_births=self.births,
            execution_completion=completion_ref,checked_unix=22,
            terminal_observations=[dict(process_identity=birth,terminal=True,observation='absent',
                                        observed_process_identity=None) for birth in self.births])
        release_ref = self.put(self.target/'logs/final-restore-owned-release.json',self.release)
        ledgers={name:dict(path=str(self.base/'logs'/name),bytes=123,sha256=core.digest(name.encode())) for name in core.LEDGERS}
        self.no201 = dict(last_completed_outer=200,update201_present=False,checkpoint201_present=False,
                         source_ledgers=ledgers,checked_unix=1)
        before_ref = self.put(self.target/'preparation/source-no201-before.json',self.no201)
        after_ref = self.put(self.target/'preparation/source-no201-after.json',dict(self.no201,checked_unix=23))
        self.proof = dict(result,schema_version=1,status=core.GPU_STATUS,kind=core.ROLE,role=core.ROLE,
            restore_only_role=core.ROLE,target_base=str(self.target),target_source_commit='d'*40,source_base=str(self.base),
            source_commit=self.run['source_commit'],run_id=self.run_id,source_step=200,source_checkpoint=str(self.cp),
            source_job='730961',source_receipt=self.source_ref,allocation=allocation_ref,readbacks=refs,
            execution_completion=completion_ref,owned_release=release_ref,source_no201_before=before_ref,
            source_no201_after=after_ref,attempt_progress=self.saved_progress,actual_optimizer_writes_per_rank=3137,
            candidate_source_cursor=517120,checked_unix=24)
        self.proof_path=self.target/'logs/final-restore-only-complete.json';self.put(self.proof_path,self.proof)

    def verify(self):
        self.put(self.proof_path,self.proof)
        return core.verify_actual_final_restore(self.proof_path,base=self.base,run=self.run,
                                                checkpoint_summary=self.checkpoint_summary,progress=self.progress)

    def phase_update(self,phase):
        self.proof['readbacks'][phase]=self.put(self.target/'logs'/f'final-restore-only-readback-{phase}.json',self.phases[phase])

    def test_all_linked_evidence_without_any_large_tensor_retention_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            original=builtins.__import__
            def guard(name,*args,**kwargs):
                if name=='torch' or name.startswith('torch.'):raise AssertionError('CPU consumer imported Torch')
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guard): result=self.verify()
            self.assertTrue(result['actual_cuda_restore_accepted'])
            self.assertFalse(result['actual_restore_performed_by_this_cpu_reader'])
            self.assertFalse(self.cp.exists())
            self.assertEqual((result['actual_optimizer_writes_per_rank'],result['candidate_source_cursor']),(3137,517120))

    def test_changed_readback_file_fails_even_when_top_booleans_remain_true(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory);self.phases['B']['driver']['pending_offset']=8
            self.put(self.proof['readbacks']['B']['path'],self.phases['B'])
            with self.assertRaisesRegex(ValueError,'SHA_OR_SIZE'):self.verify()

    def test_complete_may_omit_accessories_but_archive_and_receipt_cover_all(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,with_accessories=True)
            self.assertEqual((len(self.receipt['files']),len(self.complete['files_sha256'])),(19,11))
            self.assertNotIn('tokenizer.json',self.complete['files_sha256'])
            self.assertTrue(self.verify()['actual_cuda_restore_accepted'])
            accessory=next(row for row in self.receipt['files'] if row['relative_path']=='tokenizer.json')
            accessory['full_read_sha256']='f'*64
            self.proof['source_receipt']=self.put(self.source_path,self.receipt)
            with self.assertRaisesRegex(ValueError,'MEMBER_ARCHIVE_SHA'):self.verify()

    def test_actual_native_allocation_minor_namespace_shape_and_false_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            self.allocation.update(actual_allocated_indices=[4,5,6,7],
                actual_allocation_detail=' Nodes=ACD1-1 CPU_IDs=0-15 Mem=512000 GRES=gpu:4(IDX:4-7)',
                physical_gpu_inventory=[[str(rank),'GPU-fixture-'+str(rank),'81559',str(rank+4)] for rank in range(4)])
            self.proof['allocation']=self.put(self.target/'logs/native-allocation-admission.json',self.allocation)
            self.assertTrue(self.verify()['actual_cuda_restore_accepted'])
            self.allocation['physical_gpu_inventory'][0][3]='99'
            self.proof['allocation']=self.put(self.target/'logs/native-allocation-admission.json',self.allocation)
            with self.assertRaisesRegex(ValueError,'CUDA_MAPPING'):self.verify()

    def test_assigned_gpu_permutation_and_same_allocation_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            self.allocation['slurm_job_id']=self.receipt['source_job']
            self.proof['allocation']=self.put(self.target/'logs/native-allocation-admission.json',self.allocation)
            for phase in ('A','B'):
                for rank,row in enumerate(self.phases[phase]['actual_actor_readbacks']):
                    row['slurm_job_id']=self.receipt['source_job']
                    row['actual_cuda_uuid']=self.allocation['actual_cuda_runtime_uuids'][(rank+2)%4]
                self.phase_update(phase)
            self.assertTrue(self.verify()['actual_cuda_restore_accepted'])

    def test_rehashed_wrong_counter_and_actual_optimizer_calls_rejected(self):
        for field in ('actual_optimizer_writes','optimizer_calls','optimizer_rollbacks'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                for phase in ('A','B'):
                    self.phases[phase]['actual_actor_readbacks'][2]['actual_counters'][field]+=1;self.phase_update(phase)
                with self.assertRaisesRegex(ValueError,'COUNTER'):self.verify()

    def test_rank_model_adam_scheduler_or_rng_false_cannot_pass(self):
        for field in ('model_equal','optimizer_equal','scheduler_equal','saved_rng_equal'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory);self.phases['B']['actual_actor_readbacks'][3][field]=False;self.phase_update('B')
                with self.assertRaisesRegex(ValueError,'RANK_READBACK'):self.verify()

    def test_duplicate_gpu_and_reused_worker_birth_rejected(self):
        for mode in ('uuid','birth'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                for phase in ('A','B'):
                    rows=self.phases[phase]['actual_actor_readbacks']
                    if mode=='uuid':rows[1]['actual_cuda_uuid']=rows[0]['actual_cuda_uuid']
                    else:rows[1]['process_identity']=rows[0]['process_identity']
                    self.phase_update(phase)
                with self.assertRaises(ValueError):self.verify()

    def test_different_b_or_host_birth_not_second_readback(self):
        for mode in ('birth','boot','phase'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                if mode=='birth':self.phases['B']['actual_actor_readbacks'][1]['process_identity']=dict(self.births[1],start_ticks=9999)
                elif mode=='boot':self.phases['B']['host_identity']=dict(self.host,boot_id='aaaaaaaa-1234-1234-1234-123456789abc')
                else:self.phases['B']['phase']='A'
                self.phase_update('B')
                with self.assertRaises(ValueError):self.verify()

    def test_driver_cursor_probe_and_saved_driver_sha_must_match(self):
        for field,value in (('actual_candidate_source_cursor',512000),('actual_probe_round',200),
                            ('source_driver_full_sha256','f'*64),('pending_offset',0)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                for phase in ('A','B'):self.phases[phase]['driver'][field]=value;self.phase_update(phase)
                with self.assertRaisesRegex(ValueError,'DRIVER_READBACK'):self.verify()

    def test_rank_source_state_sha_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            for phase in ('A','B'):
                self.phases[phase]['actual_actor_readbacks'][0]['source_state_full_sha256']='e'*64;self.phase_update(phase)
            with self.assertRaisesRegex(ValueError,'RANK_READBACK'):self.verify()

    def test_just_completion_or_terminal_booleans_are_insufficient(self):
        for mode in ('completion','observations','alive'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                if mode=='completion':
                    self.completion.pop('task_result');self.proof['execution_completion']=self.put(
                        self.target/'logs/final-restore-execution-completion.json',self.completion)
                else:
                    if mode=='observations':self.release.pop('terminal_observations')
                    else:self.release['terminal_observations'][0]['observation']='live'
                    self.proof['owned_release']=self.put(self.target/'logs/final-restore-owned-release.json',self.release)
                with self.assertRaises(ValueError):self.verify()

    def test_source_ledger_changes_or_update201_artifacts_fail(self):
        for mode in ('changed','missing','201'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                if mode=='201':self.put(self.target/'logs/actual-updates.jsonl',dict(step=201))
                else:
                    after=copy.deepcopy(self.no201)
                    if mode=='changed':after['source_ledgers']['optimizer-rank-1.jsonl']['sha256']='f'*64
                    else:after['source_ledgers'].pop('scheduler-rank-3.jsonl')
                    self.proof['source_no201_after']=self.put(self.target/'preparation/source-no201-after.json',after)
                with self.assertRaises(ValueError):self.verify()

    def test_extra_mutating_call_or_missing_real_second_phase_rejected(self):
        for field in core.ZERO:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory);self.phases['B'][field]=1;self.phase_update('B')
                with self.assertRaisesRegex(ValueError,'READBACK_PHASE'):self.verify()

    def test_unrelated_cpu_progress_or_checkpoint_archive_not_accepted(self):
        for mode in ('progress','archive'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                if mode=='progress':self.progress['cumulative_actual_optimizer_writes_per_rank']=4000
                else:self.checkpoint_summary['source_checkpoint']=str(self.base/'foreign200')
                with self.assertRaises(ValueError):self.verify()

    def test_legacy_scope_symlink_and_duplicate_json_rejected(self):
        for mode in ('scope','symlink','duplicate'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory)
                if mode=='scope':self.proof['kind']='question_probe_pair'
                elif mode=='symlink':
                    p=Path(self.proof['readbacks']['A']['path']);other=p.with_suffix('.real');p.rename(other);p.symlink_to(other)
                else:
                    self.proof_path.write_text('{"target_base":"'+str(self.target)+'","target_base":"duplicate"}')
                    with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON'):core.verify_actual_final_restore(
                        self.proof_path,base=self.base,run=self.run,checkpoint_summary=self.checkpoint_summary,progress=self.progress)
                    continue
                with self.assertRaises(ValueError):self.verify()


if __name__=='__main__':unittest.main()
