"""Count complete twenty-opportunity attempts separately from retained Adam writes."""
import json
import os
import uuid
from pathlib import Path

from reproduction.common8.topology import execution_job_id, world_size


class AttemptProgress:
    """Complete counters and retained physical Adam counters stay independent."""
    def __init__(self, run=None):
        self.world_size=world_size(run)
        self.completed_outer=0
        self.cumulative_writes=0
        self.attempt_index=0
        self.pending=None
        self.aborted_attempts=0
        self.full_budget_outer=0

    def begin(self):
        assert self.pending is None and self.completed_outer<200
        self.attempt_index+=1
        self.pending=dict(logical_update=self.completed_outer+1,attempt_index=self.attempt_index,
                          attempt_id=uuid.uuid4().hex)
        return dict(self.pending)

    def finish(self, rank_results):
        """The driver reads every real Ray return before advancing anything."""
        assert self.pending is not None and type(rank_results) is list and len(rank_results)==self.world_size
        assert all(type(row['rank']) is int for row in rank_results)
        assert sorted(row['rank'] for row in rank_results)==list(range(self.world_size))
        shared=('status','logical_update','attempt_id','attempt_index','optimizer_writes_before',
                'optimizer_writes_after','optimizer_writes_this_attempt','opportunities_processed',
                'opportunity_statuses','guard_stage','guard_reasons','complete_outer',
                'scheduler_advances','completed_outer')
        if rank_results[0].get('recovery_protocol_version')==2:
            shared+=('recovery_protocol_version','full_budget_complete','opportunities_unattempted')
        representative=rank_results[0]
        assert all(all(row[key]==representative[key] for key in shared) for row in rank_results)
        if representative.get('recovery_protocol_version')==2:
            for key in ('optimizer_calls_this_attempt','optimizer_rollbacks_this_attempt'):
                assert all(row.get(key)==representative.get(key) for row in rank_results)
        assert all(representative[key]==value for key,value in self.pending.items())
        before,after=representative['optimizer_writes_before'],representative['optimizer_writes_after']
        count=representative['optimizer_writes_this_attempt']
        processed=representative['opportunities_processed']
        assert type(before) is int and before==self.cumulative_writes and type(after) is int
        assert type(count) is int and after-before==count and 0<=count<=20
        assert type(processed) is int and 1<=processed<=20
        version=representative.get('recovery_protocol_version',1)
        if version==1: assert processed==20
        statuses=representative['opportunity_statuses']
        assert type(statuses) is list and len(statuses)==processed
        assert all(value in ('written','written_with_post_guard','skipped_guard','rolled_back_guard') for value in statuses)
        assert sum(value in ('written','written_with_post_guard') for value in statuses)==count
        complete=processed==20 and count>0
        assert representative['complete_outer'] is complete
        full_budget=count==20 and processed==20
        if version==2:
            assert representative['full_budget_complete'] is full_budget
            assert representative['opportunities_unattempted']==20-processed
            assert representative['status']==('complete' if complete else 'partial' if count>0 else 'recoverable_guard')
            assert representative['retained_partial_adam_writes'] is (count>0 and not complete)
            calls=representative['optimizer_calls_this_attempt'];rollbacks=representative['optimizer_rollbacks_this_attempt']
            assert type(calls) is int and type(rollbacks) is int and 0<=rollbacks<=calls<=processed
            assert rollbacks==statuses.count('rolled_back_guard') and calls==count+rollbacks
        else:
            assert representative['status']==('complete' if complete else 'recoverable_guard')
        assert representative['scheduler_advances']==int(complete)
        assert representative['completed_outer']==self.completed_outer+int(complete)
        self.cumulative_writes=after
        if complete:self.completed_outer+=1
        else:self.aborted_attempts+=1
        self.full_budget_outer+=int(full_budget)
        self.pending=None
        result={key:value for key,value in representative.items() if key!='rank'}
        result.update(all_ranks_agreed=True,real_rank_count=self.world_size,
                      cumulative_aborted_attempts=self.aborted_attempts,
                      cumulative_full_budget_outers=self.full_budget_outer)
        return result


def record_attempt(result):
    """Every attempt is durable; skipped attempts are never update/reward rows."""
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['policy_safety_mode']=='inline_first_five' and result['all_ranks_agreed'] is True
    path=base/'logs/actual-attempts.jsonl'
    previous=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    restored=run.get('full_state_restore',{})
    start=restored.get('attempt_index',restored.get('completed_outer',0))+1
    assert [row['attempt_index'] for row in previous]==list(range(start,result['attempt_index']))
    assert result['attempt_id'] not in {row['attempt_id'] for row in previous}
    assert type(result['real_rank_count']) is int and result['real_rank_count']==world_size(run)
    row=dict(run_id=run['run_id'],slurm_job_id=execution_job_id(run),**result)
    if run.get('execution_backend', 'slurm')=='ssh_local':
        row.update(execution_backend='ssh_local',execution_id=run['execution_id'])
    descriptor=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        content=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
        assert os.write(descriptor,content)==len(content)
        os.fsync(descriptor)
        parent_fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
    finally:os.close(descriptor)
    return row
