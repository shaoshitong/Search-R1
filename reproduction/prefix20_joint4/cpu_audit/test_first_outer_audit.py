"""Tiny adversarial fixtures; no GPU, Torch, remote access or real tensors."""
import copy
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('first_outer_operator', HERE / 'first_outer_audit.py')
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('first-outer reader tried to import Torch')


GUARD = NoTorch()


def setUpModule():
    sys.meta_path.insert(0, GUARD)


def tearDownModule():
    sys.meta_path.remove(GUARD)


class FirstWindow(unittest.TestCase):
    def fixture(self):
        run = dict(run_id='a' * 32, source_commit='c' * 40)
        attempts = []
        for index, (count, statuses) in enumerate(((0, ['skipped_guard']),
                (3, ['written', 'written', 'written', 'rolled_back_guard'])), 1):
            complete = count > 0
            attempts.append(dict(run_id=run['run_id'], slurm_job_id='727251',
                attempt_id=('d' if index == 1 else 'e') * 32, attempt_index=index,
                logical_update=1, all_ranks_agreed=True, real_rank_count=4,
                optimizer_writes_before=0, optimizer_writes_after=count,
                optimizer_writes_this_attempt=count, opportunities_processed=len(statuses),
                opportunity_statuses=statuses, complete_outer=complete,
                scheduler_advances=int(complete), completed_outer=int(complete),
                recovery_protocol_version=2, full_budget_complete=False,
                opportunities_unattempted=20 - len(statuses),
                status='partial' if complete else 'recoverable_guard'))
        update = dict(run_id=run['run_id'], step=1, attempt_id='e' * 32, attempt_index=2,
                      actual_optimizer_writes=3, optimizer_writes_before=0, optimizer_writes_after=3)
        snapshots = {}
        for rank in range(4):
            scheduler = []
            worker = []
            for a in attempts:
                for kind in (('BEGIN', 'COMMIT_PREPARED', 'COMPLETE') if a['complete_outer'] else
                             ('BEGIN', 'ABORTED_ATTEMPT')):
                    terminal = kind != 'BEGIN'
                    after = int(kind == 'COMPLETE')
                    scheduler.append(dict(run_id=run['run_id'], source_commit=run['source_commit'],
                        slurm_job_id='727251', rank=rank, world_size=4, logical_update=1,
                        attempt_id=a['attempt_id'], attempt_index=a['attempt_index'], kind=kind,
                        scheduler_last_epoch_before=0, scheduler_last_epoch_after=after,
                        scheduler_step_count_before=1, scheduler_step_count_after=1 + after,
                        actual_optimizer_writes=a['optimizer_writes_this_attempt'] if terminal else 0,
                        writes_before=0, writes_after=a['optimizer_writes_after'] if terminal else 0,
                        consumed_learning_rates=[0.]))
                context = dict(attempt_id=a['attempt_id'], attempt_index=a['attempt_index'], logical_update=1)
                for index, status in enumerate(a['opportunity_statuses']):
                    wrote = status == 'written'
                    before = min(index, a['optimizer_writes_this_attempt'])
                    if wrote:
                        worker.append(dict(run_id=run['run_id'], rank=rank, kind='OPTIMIZER_WRITE',
                            context=dict(context, minibatch_index=index), write_count=index + 1,
                            learning_rates=[0.]))
                    worker.append(dict(run_id=run['run_id'], rank=rank, kind='OPPORTUNITY',
                        context=context, opportunity_index=index, status=status,
                        optimizer_write_succeeded=wrote, optimizer_writes_before=before,
                        optimizer_writes_after=before + int(wrote)))
                worker.append(dict(run_id=run['run_id'], rank=rank,
                    kind='OUTER_BUDGET' if a['complete_outer'] else 'ABORTED_ATTEMPT', **context))
            snapshots['logs/scheduler-rank-%d.jsonl' % rank] = scheduler
            snapshots['logs/optimizer-rank-%d.jsonl' % rank] = worker
        return run, attempts, update, snapshots

    def test_aborted_then_partial_three_write_outer_is_valid_not_full20(self):
        run, attempts, update, snapshots = self.fixture()
        selected, history = operator.first_attempt_window(attempts, update, run, '727251')
        self.assertEqual(history['aborted_attempts_before_first_complete'], 1)
        self.assertEqual(history['actual_actor_adam_writes_per_rank'], 3)
        self.assertFalse(history['first_actor_full20_write_budget'])
        ranks = operator.validate_rank_window(snapshots, selected, run, '727251')
        self.assertEqual([r['actual_retained_adam_writes'] for r in ranks], [3] * 4)

    def test_later_inflight_attempt_does_not_expand_first_window(self):
        run, attempts, update, snapshots = self.fixture()
        for rank in range(4):
            snapshots['logs/scheduler-rank-%d.jsonl' % rank].append(dict(
                run_id=run['run_id'], source_commit=run['source_commit'], slurm_job_id='727251',
                rank=rank, world_size=4, logical_update=2, attempt_id='f' * 32,
                attempt_index=3, kind='BEGIN'))
            snapshots['logs/optimizer-rank-%d.jsonl' % rank].append(dict(run_id=run['run_id'],
                rank=rank, kind='POLICY_MONITOR', context=dict(logical_update=2,
                attempt_id='f' * 32, attempt_index=3)))
        ranks = operator.validate_rank_window(snapshots, attempts, run, '727251')
        self.assertEqual([r['later_worker_rows_not_audited'] for r in ranks], [1] * 4)

    def test_causally_incomplete_terminal_snapshot_is_waiting_not_rejection(self):
        run, attempts, update, snapshots = self.fixture()
        snapshots['logs/scheduler-rank-2.jsonl'].pop()
        with self.assertRaisesRegex(operator.Waiting, 'WAITING_CAUSAL'):
            operator.validate_rank_window(snapshots, attempts, run, '727251')

    def test_wrong_actual_rank_adam_sequence_rejected(self):
        run, attempts, update, snapshots = self.fixture()
        row = next(r for r in snapshots['logs/optimizer-rank-1.jsonl']
                   if r['kind'] == 'OPTIMIZER_WRITE' and r['write_count'] == 3)
        row['write_count'] = 4
        with self.assertRaisesRegex(operator.Rejected, 'PHYSICAL_ADAM_WRITE_SEQUENCE'):
            operator.validate_rank_window(snapshots, attempts, run, '727251')

    def test_scheduler_first_attempt_cannot_be_relabelled_outer2(self):
        run, attempts, update, snapshots = self.fixture()
        snapshots['logs/scheduler-rank-0.jsonl'][0]['logical_update'] = 2
        with self.assertRaisesRegex(operator.Rejected, 'ACTUAL_SCHEDULER_COUNTER'):
            operator.validate_rank_window(snapshots, attempts, run, '727251')

    def test_physical_actor_write_cannot_consume_nonzero_lr_at_outer1(self):
        run, attempts, update, snapshots = self.fixture()
        row = next(r for r in snapshots['logs/optimizer-rank-3.jsonl'] if r['kind'] == 'OPTIMIZER_WRITE')
        row['learning_rates'] = [1e-6]
        with self.assertRaisesRegex(operator.Rejected, 'PHYSICAL_WRITE_INDEX_OR_CONSUMED_LR'):
            operator.validate_rank_window(snapshots, attempts, run, '727251')

    def test_opportunity_counter_cannot_claim_an_extra_unretained_write(self):
        run, attempts, update, snapshots = self.fixture()
        row = next(r for r in snapshots['logs/optimizer-rank-1.jsonl']
                   if r['kind'] == 'OPPORTUNITY' and r['status'] == 'rolled_back_guard')
        row['optimizer_writes_after'] = 4
        with self.assertRaisesRegex(operator.Rejected, 'PHYSICAL_OPPORTUNITY_COUNTER'):
            operator.validate_rank_window(snapshots, attempts, run, '727251')

    def test_observed_update201_even_outside_first_window_is_rejected(self):
        run, attempts, update, snapshots = self.fixture()
        snapshots['logs/optimizer-rank-0.jsonl'].append(dict(run_id=run['run_id'], rank=0,
            kind='POLICY_MONITOR', context=dict(logical_update=201, attempt_id='f' * 32, attempt_index=3)))
        with self.assertRaisesRegex(operator.Rejected, 'UPDATE201'):
            operator.validate_rank_window(snapshots, attempts, run, '727251')

    def test_zero_write_cannot_be_marked_completed(self):
        run, attempts, update, snapshots = self.fixture()
        attempts[0]['complete_outer'] = True
        with self.assertRaisesRegex(operator.Rejected, 'COMPLETION_OR_SCHEDULER'):
            operator.first_attempt_window(attempts, update, run, '727251')

    def test_foreign_attempt_job_is_rejected(self):
        run, attempts, update, snapshots = self.fixture()
        attempts[0]['slurm_job_id'] = 'old-job'
        with self.assertRaisesRegex(operator.Rejected, 'RUN_JOB_OR_SEQUENCE'):
            operator.first_attempt_window(attempts, update, run, '727251')


class LiveSnapshots(unittest.TestCase):
    def test_partial_next_row_is_ignored_with_explicit_prefix_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'logs').mkdir()
            path = base / 'logs/x.jsonl'
            complete = b'{"step":1}\n'
            path.write_bytes(complete + b'{"step":2')
            rows, record = operator.live_rows(base, 'logs/x.jsonl')
            self.assertEqual(rows, [dict(step=1)])
            self.assertEqual(record['read_complete_prefix_bytes'], len(complete))
            self.assertEqual(record['ignored_incomplete_tail_bytes'], 9)

    def test_append_during_read_is_not_a_fatal_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'logs').mkdir()
            path = base / 'logs/x.jsonl'
            path.write_bytes(b'{"step":1}\n')
            original = os.read

            def append_after_read(fd, count):
                result = original(fd, count)
                with path.open('ab') as stream:
                    stream.write(b'{"step":2}\n')
                return result

            with patch.object(operator.os, 'read', side_effect=append_after_read):
                rows, record = operator.live_rows(base, 'logs/x.jsonl')
            self.assertEqual(rows, [dict(step=1)])
            self.assertEqual(record['rows'], 1)
            self.assertEqual(path.read_bytes().count(b'\n'), 2)

    def test_missing_first_update_is_waiting_before_helpers_or_tensor_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'logs').mkdir()
            with patch.object(operator, 'load_helpers', side_effect=AssertionError('helpers touched too soon')):
                with self.assertRaisesRegex(operator.Waiting, 'WAITING_FIRST_ACTUAL_COMPLETE'):
                    operator.audit(str(base), 'a' * 40, 'b' * 32)

    def test_missing_causal_raw_artifacts_is_waiting_before_optional_dependencies(self):
        run, attempts, update, snapshots = FirstWindow().fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(operator.Waiting, 'WAITING_CAUSAL_FIRST_OUTER_ARTIFACTS'):
                operator.require_causal_artifacts(Path(directory).resolve(), attempts)

    def test_wrong_helper_hash_is_rejected_without_importing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base / 'source' / ('a' * 40) / 'reproduction/prefix20_joint4/cpu_audit'
            source.mkdir(parents=True)
            (source / 'audit_prefix20.py').write_text('import torch\n')
            with self.assertRaisesRegex(operator.Rejected, 'BOUND_CPU_HELPER_SHA_DIFFERS'):
                operator.load_helpers(base, 'a' * 40, dict(source_sha256={}))
            self.assertNotIn('torch', sys.modules)

    def test_real_bound_helpers_load_and_final_missing200_gate_stays_strict(self):
        original = HERE
        self.assertTrue((original / 'audit_prefix20.py').is_file())
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target = base / 'source' / ('a' * 40) / 'reproduction/prefix20_joint4/cpu_audit'
            target.mkdir(parents=True)
            hashes = {}
            for name in operator.HELPER_NAMES:
                raw = (original / name).read_bytes()
                (target / name).write_bytes(raw)
                hashes['reproduction/prefix20_joint4/cpu_audit/' + name] = hashlib.sha256(raw).hexdigest()
            (base / 'logs').mkdir()
            (base / 'logs/actual-updates.jsonl').write_text(json.dumps(dict(step=1, run_id='b' * 32)) + '\n')
            try:
                helper, identity = operator.load_helpers(base, 'a' * 40, dict(source_sha256=hashes))
                with self.assertRaisesRegex(helper.Rejected, 'MISSING_COMPLETE200'):
                    helper.early_history(helper.Reads(base), 'a' * 40, 'b' * 32)
                self.assertEqual(set(identity), set(operator.HELPER_NAMES))
                self.assertNotIn('torch', sys.modules)
            finally:
                for name in list(sys.modules):
                    if name == '_first_outer_bound_cpu' or name.startswith('_first_outer_bound_cpu.'):
                        del sys.modules[name]

    def test_receipt_is_small_immutable_and_never_final_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'logs').mkdir()
            path = base / 'logs/result.json'
            result = dict(status='FIRST_OUTER_RECORDED_EVIDENCE_PASS', accepted=True,
                          final_acceptance=False, goal_complete=False)
            operator.write_once(str(path), str(base), result)
            with self.assertRaises(FileExistsError):
                operator.write_once(str(path), str(base), dict(accepted=False))
            self.assertEqual(json.loads(path.read_text()), result)


if __name__ == '__main__':
    unittest.main()
