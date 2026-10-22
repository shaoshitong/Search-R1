"""Explicit diagnostic constant LR and fresh200 outer-counted warmup.

Scheduler receipts describe real optimizer writes and actual torch counters.
They are not GPU, accuracy, checkpoint or full-run acceptance declarations.
"""
import hashlib
import json
import math
import os
import re
import socket
import time
from pathlib import Path

from reproduction.common8.topology import execution_job_id, ranks, world_size as training_world_size

BASE_LR = 1e-6
WARMUP_OUTERS = 57
TARGET_OUTERS = 200
WRITES_PER_OUTER = 20
CONTRACT_KIND = 'fresh200_warmup57'


class SchedulerError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise SchedulerError(message)


def expected_lr(completed_outer):
    _need(type(completed_outer) is int and 0 <= completed_outer <= TARGET_OUTERS,
          'Scheduler completed-outer counter is outside fresh200')
    return BASE_LR * min(completed_outer / WARMUP_OUTERS, 1.)


def constant_scheduler(optimizer):
    """Legacy constant-LR diagnostic or synthetic mechanics only."""
    from torch.optim.lr_scheduler import LambdaLR
    assert optimizer.param_groups and all(group['lr'] == BASE_LR for group in optimizer.param_groups)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    assert all(group['lr'] == BASE_LR for group in optimizer.param_groups)
    scheduler._common8_schedule_kind = 'diagnostic_constant_lr'
    return scheduler


def fixed_outer_warmup_scheduler(optimizer, *, warmup_steps):
    """Fresh LambdaLR: completed0 consumes LR0; completed57 consumes base LR."""
    from torch.optim.lr_scheduler import LambdaLR
    _need(type(warmup_steps) is int and warmup_steps == WARMUP_OUTERS,
          'Formal warmup must be exactly57 explicitly counted outer updates')
    _need(optimizer.param_groups and all(group['lr'] == BASE_LR for group in optimizer.param_groups),
          'Wrong fresh optimizer base LR')
    _need(not optimizer.state and all('initial_lr' not in group for group in optimizer.param_groups),
          'Formal scheduler/Adam must initialize fresh')
    scheduler = LambdaLR(optimizer, lr_lambda=lambda completed: min(completed / WARMUP_OUTERS, 1.))
    _need(scheduler.last_epoch == 0 and scheduler._step_count == 1 and
          all(group['lr'] == 0. for group in optimizer.param_groups),
          'Actual torch constructor did not establish zero-index warmup')
    scheduler._common8_schedule_kind = CONTRACT_KIND
    return scheduler


def actor_scheduler(optimizer, config, *, execution_role):
    """Explicit production57 contract; old diagnostics remain constant0."""
    _need(type(config.get('completed_updates', 0)) is int and config.get('completed_updates', 0) == 0 and
          type(config.get('lr_warmup_steps_ratio', 0.)) in (int, float) and
          config.get('lr_warmup_steps_ratio', 0.) == 0.,
          'Fresh common8 cannot inherit counters or derive warmup from a ratio')
    if execution_role == 'production':
        return fixed_outer_warmup_scheduler(optimizer, warmup_steps=config.get('lr_warmup_steps'))
    _need(execution_role in ('policy_safety_calibration_discovery', 'policy_safety_calibration_confirmation') and
          config.get('lr_warmup_steps', 0) == 0,
          'Only original calibration roles may use diagnostic constant LR')
    return constant_scheduler(optimizer)


def collective_actor_initialization(optimizer_factory, config, *, execution_role):
    """Catch fresh optimizer/scheduler construction before rollout collectives."""
    optimizer, scheduler = None, None
    def initialize():
        nonlocal optimizer, scheduler
        optimizer = optimizer_factory()
        scheduler = actor_scheduler(optimizer, config, execution_role=execution_role)
        return dict(execution_role=execution_role, schedule_kind=scheduler._common8_schedule_kind,
                    learning_rates=_learning_rates(optimizer), last_epoch=scheduler.last_epoch,
                    step_count=scheduler._step_count, fresh_adam_entries=len(optimizer.state))
    collective_call(initialize, stage='optimizer_scheduler_initialization')
    return optimizer, scheduler


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def collective_call(callback, *, stage):
    """Agree on local I/O/check results before any following training collective.

    callback returns a JSON-compatible state common to all ranks. A local
    failure is broadcast to peers; no rank proceeds merely because its own
    receipt succeeded. This cannot recover a dead process or failed transport.
    """
    import torch
    import torch.distributed as dist
    _need(dist.is_initialized(), 'Scheduler admission requires a real process group')
    error, state, fingerprint = None, None, None
    try:
        state = callback()
        fingerprint = hashlib.sha256(_canonical(state)).hexdigest()
    except BaseException as caught:
        error = caught
    device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
    accepted = torch.tensor([int(error is None)], dtype=torch.int64, device=device)
    dist.all_reduce(accepted, op=dist.ReduceOp.MIN)
    if not accepted.item():
        raise SchedulerError('Collective scheduler '+stage+' rejected; local='+
                             (type(error).__name__+': '+str(error) if error is not None else 'peer failure')) from error
    fingerprints = [None] * dist.get_world_size()
    dist.all_gather_object(fingerprints, fingerprint)
    _need(all(value == fingerprint for value in fingerprints),
          'Scheduler '+stage+' states differ across ranks')
    return state


def _start_ticks():
    text = Path('/proc/self/stat').read_text()
    return int(text[text.rfind(')') + 2:].split()[19])


def _learning_rates(optimizer):
    values = [float(group['lr']) for group in optimizer.param_groups]
    _need(values and all(math.isfinite(value) and value >= 0. for value in values),
          'Invalid actual optimizer learning rate')
    return values


def _identity(base, *, job_id):
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    budget = run['budget']
    _need(run['profile'] == 'common8' and run.get('contract_kind', budget.get('contract_kind')) == CONTRACT_KIND and
          run.get('policy_calibration_role', 'production') == 'production' and
          run['completed_updates'] == 0 and run['target_updates'] == TARGET_OUTERS and
          budget['target_actual_outer_updates'] == TARGET_OUTERS and
          type(budget['lr_warmup_steps']) is int and budget['lr_warmup_steps'] == WARMUP_OUTERS and
          budget['lr_warmup_steps_ratio'] == 0. and budget['learning_rate'] == BASE_LR and
          budget['world_size'] == training_world_size(run) and budget['optimizer_steps_per_outer'] == WRITES_PER_OUTER,
          'Run is not the fresh200 fixed57 production scheduler contract')
    _need(re.fullmatch('[0-9a-f]{32}', run['run_id']) and
          re.fullmatch('[0-9a-f]{40}', run['source_commit']) and
          isinstance(run['source_sha256'], dict) and bool(run['source_sha256']) and
          all(isinstance(name, str) and isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha)
              for name, sha in run['source_sha256'].items()), 'Invalid run/source scheduler identity')
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    _need(allocation['accepted'] is True and allocation['run_id'] == run['run_id'] and
          allocation['slurm_job_id'] == job_id, 'Scheduler is not bound to this actual allocation')
    local_identity = {}
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        _need(job_id is None and execution_job_id(run) is None and
              allocation['execution_backend'] == 'ssh_local' and allocation['execution_id'] == run['execution_id'] and
              allocation['node'] == socket.gethostname(),
              'Scheduler local execution identity changed')
        local_identity = dict(execution_backend='ssh_local', execution_id=run['execution_id'], hostname=socket.gethostname())
    else:
        _need(isinstance(job_id, str) and re.fullmatch('[1-9][0-9]*', job_id),
              'Scheduler requires the actual native allocation job ID')
    return dict(run_id=run['run_id'], source_commit=run['source_commit'], slurm_job_id=job_id,
                source_map_sha256=hashlib.sha256(_canonical(run['source_sha256'])).hexdigest(),
                policy_safety_mode=run.get('policy_safety_mode'),
                scheduler_contract_sha256=hashlib.sha256(_canonical(budget)).hexdigest(), **local_identity)


class OuterSchedulerEvidence:
    """One rank, one fresh scheduler, one exclusive append-only durable ledger."""
    def __init__(self, base, optimizer, scheduler, *, rank, world_size, job_id):
        self.base = Path(base).resolve()
        run = json.loads((self.base / 'logs/run-manifest.json').read_text())
        from search_g1.actor_performance import lean_run
        self.lean_performance_profile = lean_run(run)
        _need(type(rank) is int and rank in ranks(run) and world_size == training_world_size(run),
              'Wrong real scheduler rank topology')
        self.identity = _identity(self.base, job_id=job_id)
        restore=run.get('full_state_restore')
        if restore:
            from search_g1.full_state_restore import verified_runtime_prefix
            prefix=verified_runtime_prefix(self.base,run)
            if restore.get('kind') in ('question_probe_pair','easy2560_native4'):
                _need(prefix['completed_outer']==restore['completed_outer'] and
                      prefix['writes']==restore['actual_optimizer_writes_per_rank'] and optimizer.state,
                      'Actual paired optimizer/readback is required')
            else:
                _need(prefix['completed_outer']==5 and prefix['writes']==100 and optimizer.state,
                      'Actual restored optimizer/readback is required')
        else:prefix=dict(completed_outer=0,writes=0,attempt_index=0)
        k=prefix['completed_outer']
        _need(getattr(scheduler, '_common8_schedule_kind', None)==CONTRACT_KIND and
              scheduler.last_epoch==k and scheduler._step_count==k+1 and
              _learning_rates(optimizer)==[expected_lr(k)]*len(optimizer.param_groups) and
              (bool(optimizer.state) if restore else not optimizer.state),
              'Outer scheduler must match actual fresh/restored state')
        self.optimizer,self.scheduler=optimizer,scheduler
        self.rank,self.world_size=rank,world_size
        self.completed_outer,self.pending,self.rejected=k,None,False
        self.recoverable=self.identity['policy_safety_mode']=='inline_first_five'
        self.actual_writes,self.last_attempt_index,self.used_attempt_ids=prefix['writes'],prefix['attempt_index'],set()
        if restore:
            if restore.get('kind') in ('question_probe_pair','easy2560_native4'):
                self.used_attempt_ids=set(prefix['used_attempt_ids'])
            else:
                origin=Path(restore['source_base'])
                attempts=[json.loads(line) for line in (origin/'logs/actual-attempts.jsonl').read_text().splitlines()]
                self.used_attempt_ids={v['attempt_id'] for v in attempts}
        self.prepared = None
        self.process_pid, self.process_start_ticks = os.getpid(), _start_ticks()
        _need(self.process_start_ticks > 0, 'Missing actual process birth ticks')
        self.path = self.base / 'logs' / ('scheduler-rank-%d.jsonl' % rank)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL |
                          getattr(os, 'O_NOFOLLOW', 0), 0o600)
        self.inode = os.fstat(self.fd).st_ino
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _append(self, row):
        _need(self.fd is not None and self.path.stat().st_ino == self.inode and not self.path.is_symlink(),
              'Exclusive scheduler ledger changed')
        header = dict(schema_version=1, role='fresh200_outer_scheduler',
                                contract_kind=CONTRACT_KIND, rank=self.rank, world_size=self.world_size,
                                process_pid=self.process_pid, process_start_ticks=self.process_start_ticks,
                                base_learning_rate=BASE_LR, warmup_steps=WARMUP_OUTERS,
                                observed_unix=time.time(),
                                **self.identity)
        for key in header.keys() & row.keys():
            _need(header[key] == row[key], 'Scheduler receipt attempted to replace immutable identity: '+key)
        header.update(row)
        data = _canonical(header) + b'\n'
        _need(os.write(self.fd, data) == len(data), 'Incomplete scheduler evidence write')
        os.fsync(self.fd)

    def _check_state(self):
        _need(_identity(self.base, job_id=self.identity['slurm_job_id']) == self.identity,
              'Scheduler run/source/contract identity changed during training')
        _need(self.scheduler.last_epoch == self.completed_outer and
              self.scheduler._step_count == self.completed_outer + 1 and
              _learning_rates(self.optimizer) == [expected_lr(self.completed_outer)] * len(self.optimizer.param_groups),
              'Scheduler/counter/LR changed outside a completed outer')

    def begin_outer(self, logical_update, *, optimizer_writes, batch_gate_sha256,
                    attempt_id=None, attempt_index=None):
        _need(not self.rejected and self.pending is None and
              type(logical_update) is int and logical_update == self.completed_outer + 1 and
              logical_update <= TARGET_OUTERS and type(optimizer_writes) is int and
              optimizer_writes == (self.actual_writes if self.recoverable else self.completed_outer * WRITES_PER_OUTER) and
              re.fullmatch('[0-9a-f]{64}', batch_gate_sha256), 'Wrong sequential outer/write/behavior-gate binding')
        if self.recoverable:
            _need(isinstance(attempt_id,str) and re.fullmatch('[0-9a-f]{32}',attempt_id) and
                  attempt_id not in self.used_attempt_ids and type(attempt_index) is int and
                  attempt_index==self.last_attempt_index+1,'Reused or nonsequential formal attempt identity')
        else:
            _need(attempt_id is None and attempt_index is None,'Legacy scheduler cannot admit recoverable attempts')
        self._check_state()
        row = dict(logical_update=logical_update, writes_before=optimizer_writes,
                   writes_after=optimizer_writes, actual_optimizer_writes=0,
                   completed_outer_before=self.completed_outer, completed_outer_after=self.completed_outer,
                   consumed_learning_rates=_learning_rates(self.optimizer),
                   next_learning_rates=_learning_rates(self.optimizer),
                   scheduler_last_epoch_before=self.scheduler.last_epoch,
                   scheduler_last_epoch_after=self.scheduler.last_epoch,
                   scheduler_step_count_before=self.scheduler._step_count,
                   scheduler_step_count_after=self.scheduler._step_count,
                   batch_gate_sha256=batch_gate_sha256, scheduler_advances=0)
        if self.recoverable:
            row.update(attempt_id=attempt_id,attempt_index=attempt_index,opportunities_processed=0,
                       optimizer_opportunities=WRITES_PER_OUTER)
        self._append(dict(kind='BEGIN', **row))
        self.pending = row
        if self.recoverable:
            self.last_attempt_index=attempt_index
            self.used_attempt_ids.add(attempt_id)

    def agreement_state(self):
        """Exclude rank/process identity, include the actual shared counters."""
        return dict(enabled=True, **self.identity,
                    completed_outer=self.completed_outer, pending=self.pending,
                    cumulative_actual_optimizer_writes=self.actual_writes,
                    last_attempt_index=self.last_attempt_index,
                    prepared=self.prepared, last_epoch=self.scheduler.last_epoch,
                    step_count=self.scheduler._step_count,
                    learning_rates=_learning_rates(self.optimizer))

    def prepare_outer(self, *, optimizer_writes, opportunities_processed=None):
        _need(self.pending is not None and not self.rejected, 'No active accepted outer to complete')
        _need(self.prepared is None, 'Duplicate scheduler commit preparation')
        self._check_state()
        before = self.pending['writes_before']
        if self.recoverable:
            _need(type(opportunities_processed) is int and 1<=opportunities_processed<=WRITES_PER_OUTER and
                  type(optimizer_writes) is int and 1<=optimizer_writes-before<=opportunities_processed,
                  'At least one retained safe Adam write required before scheduler boundary advance')
        else:
            _need(type(optimizer_writes) is int and optimizer_writes - before == WRITES_PER_OUTER,
                  'Exactly20 dynamically observed real optimizer writes required before scheduler advance')
        row = dict(self.pending)
        row.update(kind='COMMIT_PREPARED', writes_after=optimizer_writes,
                   actual_optimizer_writes=optimizer_writes-before)
        if self.recoverable:
            row['opportunities_processed']=opportunities_processed
            row['full_budget_complete']=optimizer_writes-before==WRITES_PER_OUTER
            row['update_status']='complete' if row['full_budget_complete'] else 'partial'
        self._append(row)
        self.prepared = row
        return row

    def commit_prepared(self):
        """After collective preparation: advance once, then persist the fact.

        A failure here is fatal post-commit evidence, never a brake rejection
        with zero advance. Real Adam writes and scheduler state are retained.
        """
        _need(self.pending is not None and self.prepared is not None and not self.rejected,
              'No durable scheduler commit preparation')
        self._check_state()
        self.scheduler.step()
        after = self.completed_outer + 1
        _need(self.scheduler.last_epoch == after and self.scheduler._step_count == after + 1 and
              _learning_rates(self.optimizer) == [expected_lr(after)] * len(self.optimizer.param_groups),
              'Actual scheduler did not advance exactly once')
        row = dict(self.prepared)
        row.update(kind='COMPLETE', completed_outer_after=after,
                   next_learning_rates=_learning_rates(self.optimizer),
                   scheduler_last_epoch_after=self.scheduler.last_epoch,
                   scheduler_step_count_after=self.scheduler._step_count, scheduler_advances=1)
        self._append(row)
        self.completed_outer, self.pending, self.prepared = after, None, None
        self.actual_writes=row['writes_after']
        return row

    def complete_outer(self, *, optimizer_writes, opportunities_processed=None):
        """Local mechanics helper; the real worker uses collective stages."""
        self.prepare_outer(optimizer_writes=optimizer_writes,opportunities_processed=opportunities_processed)
        return self.commit_prepared()

    def failure_snapshot(self, *, optimizer_writes, reason, commit_started=True):
        """Actual scalar fallback also usable when the durable ledger fails."""
        row = dict(kind='POST_COMMIT_FAILURE' if commit_started else 'PRE_COMMIT_EVIDENCE_FAILURE',
                   logical_update=self.pending['logical_update'] if self.pending else self.scheduler.last_epoch,
                   writes_after=optimizer_writes, reason=str(reason),
                   actual_scheduler_last_epoch=self.scheduler.last_epoch,
                   actual_scheduler_step_count=self.scheduler._step_count,
                   actual_learning_rates=_learning_rates(self.optimizer),
                   retained_pending=self.pending, retained_prepared=self.prepared,
                   fatal_no_automatic_resume=True, commit_started=commit_started, run_id=self.identity['run_id'],
                   rank=self.rank, process_pid=self.process_pid, process_start_ticks=self.process_start_ticks)
        return row

    def fail_after_commit(self, *, optimizer_writes, reason):
        """Best-effort fatal receipt after commit started, with actual counters."""
        row = self.failure_snapshot(optimizer_writes=optimizer_writes, reason=reason)
        self.rejected = True
        self._append(row)
        return row

    def reject_outer(self, *, optimizer_writes, reason):
        _need(self.pending is not None, 'No pending outer rejection to preserve')
        before = self.pending['writes_before']
        _need(type(optimizer_writes) is int and before <= optimizer_writes <= before + WRITES_PER_OUTER,
              'Invalid actual partial optimizer count')
        unchanged = (self.scheduler.last_epoch == self.pending['scheduler_last_epoch_before'] and
                     self.scheduler._step_count == self.pending['scheduler_step_count_before'] and
                     _learning_rates(self.optimizer) == self.pending['consumed_learning_rates'])
        row = dict(self.pending)
        row.update(kind='REJECTED', writes_after=optimizer_writes,
                   actual_optimizer_writes=optimizer_writes - before, reason=str(reason),
                   next_learning_rates=_learning_rates(self.optimizer),
                   scheduler_last_epoch_after=self.scheduler.last_epoch,
                   scheduler_step_count_after=self.scheduler._step_count,
                   scheduler_advances=self.scheduler._step_count - self.pending['scheduler_step_count_before'],
                   scheduler_unchanged_on_rejection=unchanged)
        self._append(row)
        self.rejected, self.pending, self.prepared = True, None, None
        _need(unchanged, 'Rejected outer had an unauthorized scheduler advance')
        return row

    def abort_attempt(self, *, optimizer_writes, opportunities_processed, guard_stage, guard_reasons):
        """Recover only a wholly skipped or abandoned attempt; retain real Adam."""
        from search_g1.actor_performance import rollback_metadata
        _need(self.recoverable and self.pending is not None and self.prepared is None and not self.rejected,
              'No explicitly recoverable active attempt')
        before=self.pending['writes_before']
        _need(type(opportunities_processed) is int and 0<=opportunities_processed<=WRITES_PER_OUTER and
              type(optimizer_writes) is int and 0<=optimizer_writes-before<=opportunities_processed and
              (opportunities_processed<WRITES_PER_OUTER or optimizer_writes==before),
              'Zero-write attempts are retried without advancing the scheduler')
        self._check_state()
        row=dict(self.pending)
        row.update(kind='ABORTED_ATTEMPT',writes_after=optimizer_writes,
            actual_optimizer_writes=optimizer_writes-before,opportunities_processed=opportunities_processed,
            guard_stage=guard_stage,guard_reasons=list(guard_reasons),recoverable_guard=True,
            retained_partial_adam_writes=optimizer_writes>before,
            **rollback_metadata(lean=getattr(self,'lean_performance_profile',False)),recovery_protocol_version=2,
            next_learning_rates=_learning_rates(self.optimizer),scheduler_advances=0,
            scheduler_unchanged_on_rejection=True)
        self._append(row)
        self.actual_writes,self.pending,self.prepared=optimizer_writes,None,None
        return row

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
