"""Small constructed CPU accounting fixtures; no model, tensors or GPU acceptance."""
import ast
import copy
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8.recoverable_attempt import AttemptProgress
from reproduction.common8.scheduler import (
    BASE_LR, CONTRACT_KIND, OuterSchedulerEvidence, SchedulerError, expected_lr,
)


class Guard(Exception):
    def __init__(self, stage, reasons):
        self.stage, self.reasons = stage, reasons


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def actor_class():
    """Execute the production accounting/loop with scalar stand-ins, not GPU imports."""
    source = Path(__file__).resolve().parents[2] / 'verl/workers/actor/dp_actor.py'
    actor = next(node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == 'DataParallelPPOActor')
    actor.bases = []; actor.decorator_list = []
    actor.body = [node for node in actor.body if isinstance(node, ast.FunctionDef)
                  and node.name in ('update_policy', '_update_policy_attempt')]
    for method in actor.body:
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
    def append_metrics(metrics, values):
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)
    namespace = dict(torch=SimpleNamespace(distributed=SimpleNamespace(
        is_initialized=lambda: False, get_rank=lambda: 0, get_world_size=lambda: 4)),
        FSDP=type('FixtureFSDP', (), {}), append_to_dict=append_metrics)
    exec(compile(ast.Module(body=[actor], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['DataParallelPPOActor']


class CompleteOuterAccountingTests(unittest.TestCase):
    def check_case(self, writes, processed, complete, *, rollback=False, constructed_zero=False):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name); (base / 'logs').mkdir()
            run = dict(profile='common8', training_world_size=4, actor_micro_batch_per_rank=4,
                contract_kind=CONTRACT_KIND, policy_safety_mode='inline_first_five', run_id='a'*32,
                source_commit='b'*40, source_sha256={'fixture.py':'c'*64}, completed_updates=0,
                target_updates=200, budget=dict(contract_kind=CONTRACT_KIND, target_actual_outer_updates=200,
                    lr_warmup_steps=57, lr_warmup_steps_ratio=0., learning_rate=BASE_LR,
                    world_size=4, optimizer_steps_per_outer=20))
            (base / 'logs/run-manifest.json').write_text(json.dumps(run))
            (base / 'logs/native-allocation-admission.json').write_text(json.dumps(
                dict(accepted=True, run_id=run['run_id'], slurm_job_id='12345')))
            progress = AttemptProgress(run); identity = progress.begin()
            policy = ModuleType('search_g1.policy_safety')
            policy.RecoverableGuard = Guard; policy.recoverable_mode = lambda *a, **k: True
            policy.bind_actual_config = lambda *a, **k: None
            policy.agree_attempt_result = lambda value, **kwargs: value
            policy._phase_agreement = lambda *a, **k: None
            evidence = ModuleType('reproduction.common8.worker_evidence'); rows = []
            evidence.append = lambda kind, value, **kwargs: rows.append(dict(kind=kind, **value))
            inline = ModuleType('reproduction.common8.formal_inline_safety'); proofs = []
            inline.publish_optimizer_state = lambda *a, **k: proofs.append(k)
            behavior = ModuleType('reproduction.common8.behavior_runtime')
            behavior.verify_worker_input = lambda *a, **k: None
            actor = actor_class()()
            actor.config = Config(use_dynamic_bsz=False, ppo_mini_batch_size=32,
                ppo_micro_batch_size=4, ppo_epochs=1, state_masking=True, use_kl_loss=False)
            actor.actor_module = SimpleNamespace(train=lambda: None)
            optimizer = SimpleNamespace(param_groups=[dict(lr=0.)], state={})
            optimizer.zero_grad = lambda **kwargs: None; actor.actor_optimizer = optimizer
            actor._optimizer_writes = actor._optimizer_calls = actor._optimizer_rollbacks = 0
            actor._invalidate_monitor_forward_cache = lambda: None
            actor._commit_optimizer_write = lambda group: None
            def undo(group, before):
                if actor._optimizer_writes > before:
                    actor._optimizer_writes = before; actor._optimizer_rollbacks += 1
                    return True
                return False
            actor._rollback_optimizer_write = undo
            def minibatch(index, **kwargs):
                if index < writes:
                    actor._optimizer_writes += 1; actor._optimizer_calls += 1
                else:
                    if rollback:
                        actor._optimizer_writes += 1; actor._optimizer_calls += 1
                    raise Guard('after_minibatch' if rollback else 'before_minibatch', ['fixture_guard'])
            actor._run_minibatch = minibatch
            batch = type('Batch', (), {'__len__':lambda self:640,
                                      'split':lambda self, count:list(range(20))})()
            data = type('Data', (SimpleNamespace,), {'__len__':lambda self:640})(batch={},meta_info=dict(identity, temperature=1.,
                behavior_probability_gate=dict(sha256='d'*64)), select=lambda **kwargs:SimpleNamespace(batch=batch))
            if constructed_zero:
                # Guard reuse stops immediately in production. This boundary
                # fixture tests a supplied all-skipped20 return, not that loop.
                def zero_return(value):
                    actor._last_opportunities = [dict(status='skipped_guard', guard_stage='before_minibatch',
                        guard_reasons=['fixture_guard']) for _ in range(20)]
                    return {}
                actor._update_policy_attempt = zero_return
            modules = {'search_g1.policy_safety':policy,
                'reproduction.common8.worker_evidence':evidence,
                'reproduction.common8.formal_inline_safety':inline,
                'reproduction.common8.behavior_runtime':behavior}
            with patch.dict(os.environ, SEARCH_G1_PROFILE='common8', SEARCH_G1_BASE=str(base),
                            SEARCH_G1_TRAINING_WORLD_SIZE='4', SLURM_JOB_ID='12345'), patch.dict('sys.modules', modules):
                metrics = actor.update_policy(data)
                result = actor._last_attempt_result
                scheduler = SimpleNamespace(_common8_schedule_kind=CONTRACT_KIND, last_epoch=0, _step_count=1)
                def advance():
                    scheduler.last_epoch += 1; scheduler._step_count += 1
                    optimizer.param_groups[0]['lr'] = expected_lr(scheduler.last_epoch)
                scheduler.step = advance
                with patch('reproduction.common8.scheduler._start_ticks', return_value=123):
                    ledger = OuterSchedulerEvidence(base, optimizer, scheduler, rank=0, world_size=4, job_id='12345')
                try:
                    ledger.begin_outer(1, optimizer_writes=0, batch_gate_sha256='d'*64, **{
                        key:identity[key] for key in ('attempt_id','attempt_index')})
                    if complete:
                        ending = ledger.complete_outer(optimizer_writes=writes, opportunities_processed=processed)
                    else:
                        with self.assertRaises(SchedulerError):
                            ledger.prepare_outer(optimizer_writes=writes, opportunities_processed=processed)
                        ending = ledger.abort_attempt(optimizer_writes=writes, opportunities_processed=processed,
                            guard_stage=result['guard_stage'], guard_reasons=result['guard_reasons'])
                    returned = [dict(copy.deepcopy(result), rank=rank, scheduler_advances=int(complete),
                        completed_outer=ledger.completed_outer) for rank in range(4)]
                    actual = progress.finish(returned)
                    self.assertEqual(actual['opportunities_processed'], processed)
                    self.assertEqual(actual['complete_outer'], complete)
                    self.assertEqual(actual['status'], 'complete' if complete else 'partial' if writes else 'recoverable_guard')
                    self.assertEqual(actual['optimizer_writes_this_attempt'], writes)
                    self.assertEqual(actual['optimizer_calls_this_attempt'], writes+int(rollback))
                    self.assertEqual(actual['optimizer_rollbacks_this_attempt'], int(rollback))
                    self.assertEqual(actual['full_budget_complete'], writes==20)
                    self.assertEqual(actual['retained_partial_adam_writes'], bool(writes and not complete))
                    self.assertEqual((progress.cumulative_writes, ledger.actual_writes, actor._optimizer_writes), (writes,)*3)
                    self.assertEqual((progress.completed_outer, ledger.completed_outer, scheduler.last_epoch), (int(complete),)*3)
                    self.assertEqual(ending['scheduler_advances'], int(complete))
                    self.assertEqual(progress.begin()['logical_update'], 1+int(complete))
                    self.assertEqual(progress.attempt_index, 2)
                    self.assertNotEqual(progress.pending['attempt_id'], identity['attempt_id'])
                    self.assertEqual(optimizer.param_groups[0]['lr'], expected_lr(int(complete)))
                    self.assertEqual([row['kind'] for row in rows if row['kind'] in ('OUTER_BUDGET','ABORTED_ATTEMPT')],
                                     ['OUTER_BUDGET' if complete else 'ABORTED_ATTEMPT'])
                    self.assertEqual(len(proofs), int(complete))
                    if writes:
                        self.assertEqual(metrics['actor/skipped_update_opportunities'], processed-writes)
                finally:
                    ledger.close()

    def test_all_twenty_returned_adam_complete(self):
        self.check_case(20, 20, True)

    def test_last_guard_nineteen_retained_twenty_processed_complete(self):
        self.check_case(19, 20, True, rollback=True)

    def test_early_guard_retains_three_without_logical_or_scheduler_advance(self):
        self.check_case(3, 4, False, rollback=True)

    def test_twenty_processed_zero_adam_does_not_complete(self):
        self.check_case(0, 20, False, constructed_zero=True)


if __name__ == '__main__':
    unittest.main()
