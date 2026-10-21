"""CPU fixtures for actual native4 restore, scheduler and FA3 audit schemas."""
import unittest

from reproduction.operations.easy7b_20261021 import watch_fa3_transition_local as watch


def actual_schema():
    job, run = '746125', watch.RUN
    ranks = [dict(rank=rank, run_id=run, slurm_job_id=job, source_step=12,
        actual_optimizer_writes=240, scheduler_last_epoch=12,
        actual_model_collective_readback_equal=True, actual_adam_scheduler_readback_equal=True,
        actual_rng_readback_equal=True) for rank in range(4)]
    value = dict(epoch=1., jobs=dict(new_training=job, retrieval_service='745198'),
        queue=dict(rc=0, stdout=job+'|RUNNING|ACD1-55|None\n745198|RUNNING|ACD1-26|None\n'),
        accounting=dict(rc=0, stdout=''), documents={},
        actual_restore_ranks={str(row['rank']):dict(present=True, value=row) for row in ranks},
        new_training_slurm_log=dict(bytes=100, tail='Ordinary native4 startup\n'),
        actual_flash3_execution={}, post_restore_scheduler_records={},
        post_restore_update_records=[], post_restore_attempt_records=[])
    def doc(name, obj):value['documents'][name] = dict(present=True, value=obj)
    doc('logs/run-manifest.json', dict(run_id=run, full_state_restore=dict(
        completed_outer=12, actual_optimizer_writes_per_rank=240, source_job='745371')))
    doc('logs/full-state-restored-workers.json', dict(run_id=run, slurm_job_id=job, source_step=12,
        actual_world_size=4, actual_optimizer_writes_per_rank=240, ranks=ranks))
    # actual_cuda_restore_accepted remains false in this genuine producer
    # schema; success comes from actual rank readbacks plus driver dataequal.
    doc('logs/full-state-restored-driver.json', dict(run_id=run, slurm_job_id=job, source_step=12,
        completed_outer=12, source_job='745371', actual_driver_dataequal=True,
        actual_candidate_source_cursor=30720, actual_cuda_restore_accepted=False))
    doc('logs/easy2560-restored-stream-before-next-outer.json', dict(completed_outer=12,
        candidate_source_cursor=30720, actual_candidate_stream_readback_equal=True,
        actual_driver_rng_readback_equal=True))
    doc('logs/pipeline-status.json', dict(stage='formal'))
    metrics = []
    for step in range(13, 18):
        metrics.append(dict(step=step, optimizer_writes_this_outer=20, optimizer_writes=step*20, reward=.6))
        value['post_restore_update_records'].append(dict(step=step, run_id=run, update_status='complete',
            actual_optimizer_writes=20, full_budget_complete=True, opportunities_processed=20,
            opportunities_unattempted=0, optimizer_writes_before=(step-1)*20, optimizer_writes_after=step*20))
        value['post_restore_attempt_records'].append(dict(logical_update=step, run_id=run,
            slurm_job_id=job, status='complete', complete_outer=True, all_ranks_agreed=True,
            real_rank_count=4, full_budget_complete=True, optimizer_writes_this_attempt=20,
            optimizer_rollbacks_this_attempt=0, scheduler_advances=1, optimizer_writes_after=step*20))
    for rank in range(4):
        value['actual_flash3_execution'][str(rank)] = dict(latest_actual_record=dict(
            rank=rank, run_id=run, logical_update=17, actual_optimizer_writes=340,
            actual_flash3_training_executed=True, cuda_synchronized=True,
            forward_success_returns=dict(varlen=5600, dense=0), kernel_backward_success_returns=2800))
        value['post_restore_scheduler_records'][str(rank)] = [dict(kind='COMPLETE', logical_update=step,
            rank=rank, run_id=run, slurm_job_id=job, actual_optimizer_writes=20, full_budget_complete=True,
            scheduler_advances=1, scheduler_last_epoch_after=step,
            writes_before=(step-1)*20, writes_after=step*20) for step in range(13, 18)]
    value['new_training_progress'] = dict(completed_outer_metrics=metrics,
        candidate_pools=[dict(step=18, completed_questions=512, saved_chunk_files=1)],
        training_log=dict(bytes=200, tail='epoch 0, step 18\n'), trainer_log=dict(bytes=0, tail=''))
    value['new_training_progress']['preparation/rolling-checkpoints/latest.json'] = dict(
        status='EASY2560_ROLLING_CHECKPOINT_READY', run_id=run, step=17)
    return value


class FollowupTests(unittest.TestCase):
    def test_real_native4_schema_can_succeed_with_false_driver_summary_flag(self):
        result = watch.summarize(actual_schema(), 17, 18)
        self.assertTrue(result['requested_followup_observed'])
        self.assertEqual(result['phase'], 'formal')
        self.assertEqual(result['all4_scheduler_epochs'], {str(rank):17 for rank in range(4)})
        self.assertFalse(result['overall_goal_complete'])

    def test_missing_logs_or_next_tensor_metadata_cannot_succeed(self):
        for kind in ('logs', 'saved'):
            value = actual_schema()
            if kind == 'logs':
                value['new_training_slurm_log'] = {'observation_error':'Not yet present'}
                value['new_training_progress']['training_log'] = {}
            else:value['new_training_progress']['candidate_pools'][0]['saved_chunk_files'] = 0
            self.assertFalse(watch.summarize(value, 17, 18)['requested_followup_observed'])

    def test_all4_source12_counter_cursor_and_dataequal_are_required(self):
        for kind in ('rank_counter', 'scheduler', 'cursor', 'dataequal'):
            value = actual_schema()
            if kind == 'rank_counter':value['actual_restore_ranks']['2']['value']['actual_optimizer_writes'] = 239
            elif kind == 'scheduler':value['actual_restore_ranks']['2']['value']['scheduler_last_epoch'] = 11
            elif kind == 'cursor':value['documents']['logs/full-state-restored-driver.json']['value']['actual_candidate_source_cursor'] = 0
            else:value['documents']['logs/full-state-restored-driver.json']['value']['actual_driver_dataequal'] = False
            self.assertFalse(watch.summarize(value, 17, 18)['requested_followup_observed'])

    def test_completed_status_alone_or_missing_rank_scheduler_is_insufficient(self):
        value = actual_schema()
        value['post_restore_update_records'][0]['actual_optimizer_writes'] = 19
        result = watch.summarize(value, 17, 18)
        self.assertFalse(result['requested_followup_observed'])
        self.assertTrue(result['inspection_issues'])
        value = actual_schema(); value['post_restore_scheduler_records']['1'] = []
        self.assertFalse(watch.summarize(value, 17, 18)['requested_followup_observed'])

    def test_guard_or_abort_is_reported_for_inspection(self):
        value = actual_schema()
        value['post_restore_attempt_records'][0].update(status='recoverable_guard',
            optimizer_writes_this_attempt=0, complete_outer=False)
        result = watch.summarize(value, 17, 18)
        self.assertFalse(result['requested_followup_observed'])
        self.assertTrue(any('guarded/partial/aborted' in issue for issue in result['inspection_issues']))

    def test_counter340_and_epoch17_are_required_on_every_rank(self):
        for kind in ('counter', 'epoch', 'backward'):
            value = actual_schema()
            if kind == 'counter':value['actual_flash3_execution']['3']['latest_actual_record']['actual_optimizer_writes'] = 339
            elif kind == 'epoch':value['post_restore_scheduler_records']['3'][-1]['scheduler_last_epoch_after'] = 16
            else:value['actual_flash3_execution']['3']['latest_actual_record']['kernel_backward_success_returns'] = 0
            self.assertFalse(watch.summarize(value, 17, 18)['requested_followup_observed'])

    def test_slurm_startup_error_and_terminal_job_fail_readonly(self):
        for kind in ('error', 'terminal'):
            value = actual_schema()
            if kind == 'error':value['new_training_slurm_log']['tail'] = 'ModuleNotFoundError: missing module\n'
            else:value['queue']['stdout'] = value['queue']['stdout'].replace('746125|RUNNING', '746125|FAILED')
            result = watch.summarize(value, 17, 18)
            self.assertTrue(result['inspection_issues'])
            self.assertFalse(result['requested_followup_observed'])
            self.assertFalse(result['production_mutated'])

    def test_remote_wrapper_is_readonly_and_supports_old_operator(self):
        text = watch.command(watch.OPERATOR_COMMIT)
        self.assertIn('new_training_slurm_log', text)
        self.assertIn('post_restore_scheduler_records', text)
        self.assertNotIn('sbatch', text)
        self.assertNotIn('scancel', text)
        self.assertNotIn('torch.load', text)


if __name__ == '__main__':unittest.main()
