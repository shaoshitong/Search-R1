"""CPU identity/argument fixtures; no bridge, credentials, GPU or remote calls."""
import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from reproduction.operations.easy7b_20261021 import watch_cp34_recovery_local as watch
from reproduction.operations.easy7b_20261021 import observe_cp34_recovery_hpc as remote

RUNTIME='a'*40
OPERATOR='b'*40
JOB='799999'


def fixture():
    receipt='c'*64
    checkpoint=watch.SOURCE_BASE+'/checkpoints/search-g1-common8/actor/global_step_34'
    binding=dict(kind='easy2560_native4',completed_outer=34,actual_optimizer_writes_per_rank=680,
        source_job='746125',source_commit=watch.SOURCE_COMMIT,source_base=watch.SOURCE_BASE,
        source_checkpoint=checkpoint,source_receipt_sha256=receipt,candidate_source_cursor=87040)
    run=dict(run_id=remote.RUN,source_commit=RUNTIME,training_world_size=4,
        actor_micro_batch_per_rank=8,actor_logical_micro_batch_per_rank=4,full_state_restore=binding)
    rows=[dict(rank=rank,run_id=remote.RUN,slurm_job_id=JOB,source_receipt_sha256=receipt,
        source_job='746125',source_commit=watch.SOURCE_COMMIT,source_checkpoint=checkpoint,
        source_step=34,actual_optimizer_writes=680,scheduler_last_epoch=34,world_size=4,
        actual_cuda_uuid='GPU-'+str(rank),model_hf_exact_tensors_checked_on_rank0=339 if rank==0 else 0,
        actual_model_collective_readback_equal=True,actual_adam_scheduler_readback_equal=True,
        actual_rng_readback_equal=True) for rank in range(4)]
    workers=dict(run_id=remote.RUN,source_commit=RUNTIME,slurm_job_id=JOB,source_step=34,
        actual_world_size=4,actual_optimizer_writes_per_rank=680,source_receipt_sha256=receipt,ranks=rows)
    driver=dict(run_id=remote.RUN,slurm_job_id=JOB,source_receipt_sha256=receipt,source_step=34,
        completed_outer=34,source_job='746125',source_commit=watch.SOURCE_COMMIT,source_checkpoint=checkpoint,
        actual_candidate_source_cursor=87040,actual_driver_dataequal=True,
        attempt_progress=dict(completed_outer=34,cumulative_writes=680),
        actual_candidate_stream_readback_equal=True,actual_driver_rng_readback_equal=True)
    stream=dict(completed_outer=34,candidate_source_cursor=87040,current_world_size=4,
        actual_candidate_stream_readback_equal=True,actual_driver_rng_readback_equal=True)
    documents={name:dict(present=True,value=value) for name,value in (
        ('logs/run-manifest.json',run),('logs/full-state-restored-workers.json',workers),
        ('logs/full-state-restored-driver.json',driver),('logs/easy2560-restored-stream-before-next-outer.json',stream))}
    value=dict(documents=documents,actual_restore_ranks={str(rank):dict(present=True,value=copy.deepcopy(rows[rank]))
        for rank in range(4)},new_job_record={'matches_requested_job':True},
        queue=dict(rc=0,stdout=JOB+'|RUNNING|node\n745198|RUNNING|node\n'),accounting=dict(rc=0,stdout=''),
        post_restore_update_records=[],post_restore_attempt_records=[],post_restore_scheduler_records={},
        post_restore_optimizer_records={},actual_flash3_execution={})
    for step in range(35,40):
        value['post_restore_update_records'].append(dict(step=step,run_id=remote.RUN,update_status='complete',
            actual_optimizer_writes=20,full_budget_complete=True,opportunities_processed=20,
            opportunities_unattempted=0,optimizer_writes_before=(step-1)*20,optimizer_writes_after=step*20))
        value['post_restore_attempt_records'].append(dict(logical_update=step,run_id=remote.RUN,slurm_job_id=JOB,
            status='complete',complete_outer=True,all_ranks_agreed=True,real_rank_count=4,full_budget_complete=True,
            optimizer_writes_this_attempt=20,optimizer_calls_this_attempt=20,optimizer_rollbacks_this_attempt=0,
            scheduler_advances=1,optimizer_writes_after=step*20))
    for rank in range(4):
        value['post_restore_scheduler_records'][str(rank)]=[dict(kind='COMPLETE',rank=rank,
            logical_update=step,run_id=remote.RUN,slurm_job_id=JOB,actual_optimizer_writes=20,
            full_budget_complete=True,scheduler_advances=1,scheduler_last_epoch_after=step,
            writes_before=(step-1)*20,writes_after=step*20) for step in range(35,40)]
        value['post_restore_optimizer_records'][str(rank)]=[dict(rank=rank,run_id=remote.RUN,logical_update=step,
            optimizer_writes=20,cumulative_writes=step*20,full_budget_complete=True,opportunities_processed=20,
            local_micro=8,local_logical_micro=4,local_mini=32,gradient_accumulation=4,
            actual_write_counts=list(range((step-1)*20+1,step*20+1))) for step in range(35,40)]
        value['actual_flash3_execution'][str(rank)]=dict(latest_actual_record=dict(rank=rank,run_id=remote.RUN,
            actual_flash3_training_executed=True,cuda_synchronized=True,logical_update=39,actual_optimizer_writes=780,
            forward_success_returns={'varlen':25000},kernel_backward_success_returns=11000))
    value['new_training_progress']=dict(completed_outer_metrics=[{'step':39,'reward':.6}],
        candidate_pools=[dict(step=40,completed_questions=512,saved_chunk_files=1)],
        **{'preparation/rolling-checkpoints/latest.json':dict(status='EASY2560_ROLLING_CHECKPOINT_READY',
            run_id=remote.RUN,source_commit=RUNTIME,step=39)})
    value['grounding_cache_release']=dict(latest=dict(role='probe_post_judge_cuda_cache_release',
        cuda_synchronized=True,released_reserved_bytes=1000,before={'reserved_bytes':2000},after={'reserved_bytes':1000}))
    value['allocator_resources']=dict(present=True,value=dict(run_id=remote.RUN,source_commit=RUNTIME,
        slurm_job_id=JOB,torch_allocator_env={'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}))
    return value


class CP34ObserverTests(unittest.TestCase):
    def test_queued_job_without_manifest_or_complete_binding_is_pending_not_an_error(self):
        for manifest in ('absent',None,{'run_id':remote.RUN,'full_state_restore':None}):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as td:
                base=Path(td);(base/'logs').mkdir();(base/'job.id').write_text(JOB)
                if manifest!='absent': (base/'logs/run-manifest.json').write_text(json.dumps(manifest))
                reader=remote.existing.private_progress_reader()
                with patch.object(remote,'BASE',base), patch.object(remote.existing,'private_progress_reader',return_value=reader), \
                     patch.object(reader,'command',return_value=dict(rc=0,stdout=JOB+'|PENDING|node\n745198|RUNNING|node\n',stderr='')):
                    value=remote.observe(JOB,RUNTIME,OPERATOR)
                self.assertTrue(value['restore_manifest_pending'])
                summary=watch.summarize(value,JOB,RUNTIME)
                self.assertTrue(summary['restore_manifest_pending'])
                self.assertEqual(summary['inspection_issues'],[])
                self.assertFalse(summary['requested_followup_observed'])

    def test_real_small_file_reader_keeps_checkpoint_binding_without_reading_tensor_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); (base/'logs').mkdir(); (base/'job.id').write_text(JOB)
            run=fixture()['documents']['logs/run-manifest.json']['value']
            run['source_sha256']={'large-repeated-map':'not-to-be-persisted'}
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            pool=base/'artifacts/mixed-candidate-pools/pool40';pool.mkdir(parents=True)
            (pool/'progress.json').write_text(json.dumps(dict(step=40,complete_ordinals=[0],reserved=[])))
            (pool/'0.pt').write_bytes(b'synthetic payload must never be opened by observer')
            (pool/'0-before-generation.pt').write_bytes(b'RNG sidecar is not a raw candidate chunk')
            (pool/'0-after-generation.pt').write_bytes(b'RNG sidecar is not a raw candidate chunk')
            reader=remote.existing.private_progress_reader()
            original_open=Path.open
            def guarded(path,*args,**kwargs):
                if path.suffix=='.pt': raise AssertionError('Observer opened a tensor payload')
                return original_open(path,*args,**kwargs)
            with patch.object(remote,'BASE',base), patch.object(remote.existing,'private_progress_reader',return_value=reader), \
                 patch.object(reader,'command',return_value=dict(rc=0,stdout='',stderr='')), patch.object(Path,'open',guarded):
                value=remote.observe(JOB,RUNTIME,OPERATOR)
            compact=value['documents']['logs/run-manifest.json']['value']
            self.assertNotIn('source_sha256',compact)
            self.assertEqual(compact['full_state_restore']['source_checkpoint'],run['full_state_restore']['source_checkpoint'])
            self.assertEqual(value['new_training_progress']['candidate_pools'][0]['saved_chunk_files'],1)
            self.assertEqual(value['new_training_progress']['candidate_pools'][0]['completed_questions'],1)
            self.assertFalse(value['new_training_progress']['candidate_pools'][0]['progress_pending'])
            self.assertTrue(value['new_job_record']['matches_requested_job'])

    def test_rng_sidecars_without_progress_are_pending_not_completed_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);(base/'logs').mkdir();(base/'job.id').write_text(JOB)
            pool=base/'artifacts/mixed-candidate-pools/attempt-pool-0';pool.mkdir(parents=True)
            for name in ('0-before-generation.pt','0-after-generation.pt'):
                (pool/name).write_bytes(b'boundary sidecar, not candidate output')
            # A different manifest must never be used to guess step/completion.
            (pool/'manifest.json').write_text(json.dumps(dict(step=40,complete_ordinals=list(range(512)))))
            reader=remote.existing.private_progress_reader();original_open=Path.open
            def guarded(path,*args,**kwargs):
                if path.suffix=='.pt':raise AssertionError('Observer read tensor payload')
                return original_open(path,*args,**kwargs)
            def observation():
                with patch.object(remote,'BASE',base),patch.object(remote.existing,'private_progress_reader',return_value=reader), \
                     patch.object(reader,'command',return_value=dict(rc=0,stdout='',stderr='')),patch.object(Path,'open',guarded):
                    return remote.observe(JOB,RUNTIME,OPERATOR)['new_training_progress']['candidate_pools'][0]
            pending=observation()
            self.assertTrue(pending['progress_pending']);self.assertIsNone(pending['step'])
            self.assertIsNone(pending['completed_questions']);self.assertEqual(pending['saved_chunk_files'],0)
            self.assertIsNone(pending['progress_error'])
            (pool/'0.pt').write_bytes(b'actual synthetic complete raw candidate chunk')
            (pool/'progress.json').write_text(json.dumps(dict(step=35,complete_ordinals=list(range(512)),
                reserved=list(range(512)),policy_snapshot_id='saved-policy',source_consumed=89600)))
            complete=observation()
            self.assertFalse(complete['progress_pending']);self.assertEqual(complete['step'],35)
            self.assertEqual(complete['completed_questions'],512);self.assertEqual(complete['saved_chunk_files'],1)

    def test_actual_cp34_restoration_matches_all4_workers_driver_and_stream_without_mutation(self):
        value=fixture(); before=copy.deepcopy(value)
        self.assertTrue(watch.restoration(value,JOB,RUNTIME)); self.assertEqual(value,before)
        self.assertFalse(watch.restoration(value,JOB,OPERATOR))

    def test_old_cp12_wrong_tensor_job_source_or_cursor_never_pass(self):
        for target,key,bad in (('binding','completed_outer',12),('binding','source_job','745371'),
            ('driver','actual_candidate_source_cursor',30720),('stream','candidate_source_cursor',87041),
            ('driver','source_commit',RUNTIME),('workers','source_commit',OPERATOR)):
            value=fixture()
            record=(value['documents']['logs/run-manifest.json']['value']['full_state_restore'] if target=='binding'
                else value['documents']['logs/'+{'driver':'full-state-restored-driver.json','stream':
                    'easy2560-restored-stream-before-next-outer.json','workers':'full-state-restored-workers.json'}[target]]['value'])
            record[key]=bad
            with self.subTest(target=target,key=key): self.assertFalse(watch.restoration(value,JOB,RUNTIME))
        value=fixture()
        value['documents']['logs/full-state-restored-workers.json']['value']['ranks'][0]['model_hf_exact_tensors_checked_on_rank0']=435
        value['actual_restore_ranks']['0']['value']['model_hf_exact_tensors_checked_on_rank0']=435
        self.assertFalse(watch.restoration(value,JOB,RUNTIME))

    def test_five_complete_native4_budgets_seal_and_saved40_required(self):
        value=fixture(); summary=watch.summarize(value,JOB,RUNTIME)
        self.assertTrue(summary['requested_followup_observed']); self.assertFalse(summary['overall_goal_complete'])
        self.assertEqual(summary['required_completed_outers'],[35,36,37,38,39])
        for change in ('missing_write','unsealed','unsaved40','no_fa3','no_cache'):
            altered=copy.deepcopy(value)
            if change=='missing_write': altered['post_restore_optimizer_records']['3'][-1]['actual_write_counts'].pop()
            elif change=='unsealed': altered['new_training_progress']['preparation/rolling-checkpoints/latest.json']['step']=38
            elif change=='unsaved40': altered['new_training_progress']['candidate_pools'][0]['saved_chunk_files']=0
            elif change=='no_fa3': altered['actual_flash3_execution']['2']['latest_actual_record']['kernel_backward_success_returns']=0
            else: altered['grounding_cache_release']={}
            with self.subTest(change=change): self.assertFalse(watch.summarize(altered,JOB,RUNTIME)['requested_followup_observed'])

    def test_actual_optimizer_call_and_rollback_fields_must_be_present_and_exact(self):
        for field,bad in (('optimizer_calls_this_attempt','missing'),('optimizer_calls_this_attempt',21),
                          ('optimizer_rollbacks_this_attempt','missing'),('optimizer_rollbacks_this_attempt',False)):
            value=fixture();row=value['post_restore_attempt_records'][0]
            if bad=='missing': row.pop(field)
            else: row[field]=bad
            with self.subTest(field=field,bad=bad):
                summary=watch.summarize(value,JOB,RUNTIME)
                self.assertFalse(summary['complete_attempts_observed'])
                self.assertFalse(summary['requested_followup_observed'])

    def test_command_uses_operator_tree_but_passes_distinct_runtime_identity(self):
        args=shlex.split(watch.command(JOB,RUNTIME,OPERATOR)); code=args[-1]
        self.assertEqual(args[-2],'-c'); compile(code,'remote-read-only-observer','exec')
        self.assertIn(str(remote.BASE/'source'/OPERATOR),code)
        self.assertIn(','.join(map(repr,(JOB,RUNTIME,OPERATOR))),code)
        remote.arguments(JOB,RUNTIME,OPERATOR)
        with self.assertRaises(ValueError): remote.arguments(JOB,'short',OPERATOR)

    def test_default_login2_requires_explicit_pin_before_loading_bridge(self):
        with patch.object(watch.shared,'load_bridge') as bridge, redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            watch.main(['--runtime-commit',RUNTIME,'--operator-commit',OPERATOR,'--job',JOB])
        self.assertEqual(error.exception.code,2); bridge.assert_not_called()
        pin=dict(hostname='ACD-Login-2',boot_id='3c637b25-17cf-404c-971b-30bc4c4e9fbb',pid_namespace='pid:[4026531836]')
        self.assertEqual(watch.shared.resolve_login_route('10.120.48.32',json.dumps(pin)),('10.120.48.32',pin))


if __name__=='__main__': unittest.main()
