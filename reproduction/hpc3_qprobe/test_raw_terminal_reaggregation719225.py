"""Original failed RAW evidence, all guarded tasks and fresh byte conjunction."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import raw_terminal_reaggregation719225 as recovery
from reproduction.hpc3_qprobe import raw_audit_followthrough719225 as original
from reproduction.hpc3_qprobe import test_independent_goal_acceptance719225 as fixtures

write = fixtures.write


class TerminalTests(unittest.TestCase):
    def fixture(self):
        parts = [dict(first=1, last=5, producer=dict(run_id='run', producer_job='718650')),
                 dict(first=6, last=200, producer=dict(run_id='run', producer_job='719225'))]
        rows = [[], []]; index = 0
        for outer in range(1, 201):
            for count in ([0, 3] if outer == 6 else [20]):
                index += 1; complete = count > 0
                rows[int(outer > 5)].append(dict(attempt_id='%032x'%index, attempt_index=index,
                    logical_update=outer, run_id='run', slurm_job_id='719225' if outer > 5 else '718650',
                    optimizer_writes_this_attempt=count, complete_outer=complete,
                    opportunities_processed=20, scheduler_advances=int(complete),
                    completed_outer=outer if complete else outer-1, status='complete' if complete else 'recoverable_guard'))
        tasks = original.tasks_from_rows(parts, rows)
        publication = dict(commit=recovery.RAW)
        process = dict(pid=100, sid=100, start_ticks=101, uid=recovery.reader.OWNER)
        claim = dict(source=publication, parts=parts, command=[str(recovery.ORIGINAL_RAW_EXECUTABLE),
            '-m', 'reproduction.hpc3_qprobe.raw_audit_followthrough719225', '--commit', recovery.RAW, '--worker'],
            training_control_permitted=False, goal_complete=False, created_unix=1)
        result = dict(status=recovery.FAILED_STATUS, process=process, source=publication,
            goal_complete=False, completed_attempt_audits=len(tasks), started_unix=2, completed_unix=3,
            error_type='FileNotFoundError', error='[Errno 2] No such file or directory')
        state = dict(process=process, phase='ACTIVE_LEDGER_READ_WAIT', completed_attempt_audits=len(tasks))
        return claim, process, result, state, publication, parts, tasks

    def validate(self, values, *, alive=lambda row:False):
        claim, process, result, state, publication, parts, tasks = values
        return recovery.validate_terminal(claim, process, result, state, json.dumps(result)+'\n',
            publication, parts, tasks, alive=alive)

    def test_all200_and_guard_children_do_not_prove_original_internal_error_site(self):
        values = self.fixture(); self.assertEqual(len(values[-1]), 201)
        proof = self.validate(values)
        self.assertFalse(proof['original_exception_call_site_persisted'])
        self.assertFalse(proof['precise_original_aggregation_call_failure_proven'])
        values[3]['phase'] = 'ACTUAL_RAW_ATTEMPT_COMPLETED0'
        self.validate(values)

    def test_live_unknown_successful_or_missing_last_child_result_rejects(self):
        values = self.fixture()
        with self.assertRaisesRegex(ValueError, 'live, unknown'):
            self.validate(values, alive=lambda row:True)
        cases = ((2, 'status', original.STATUS), (2, 'completed_attempt_audits', 200),
            (2, 'completed_attempt_audits', True), (2, 'source', {'commit':'wrong'}),
            (2, 'error_type', 'ValueError'), (0, 'training_control_permitted', True),
            (0, 'command', ['invented']), (3, 'process', dict(values[1], start_ticks=102)))
        for index, key, value in cases:
            bad = copy.deepcopy(values); bad[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                self.validate(bad)

    def test_omitted_guard_duplicate_success_and_logical201_cannot_be_hidden(self):
        values = self.fixture()
        for changed in (values[-1][:-1], values[-1][:5]+values[-1][6:],
                        values[-1]+[dict(values[-1][-1], step=201, attempt_index=202)]):
            bad = list(copy.deepcopy(values)); bad[-1] = changed; bad[2]['completed_attempt_audits'] = len(changed)
            with self.assertRaisesRegex(ValueError, 'task set'):
                self.validate(bad)

    def test_original_missing_stale_read_class_is_not_fatal_other_exception(self):
        allowed = str(recovery.BASE/'logs/process-ledger.json')
        for kind, message in (('FileNotFoundError', '[Errno 2] No such file or directory'),
            ('FileNotFoundError', '[Errno 2] No such file or directory: '+repr(allowed)),
            ('OSError', '[Errno 116] Stale file handle')):
            self.assertFalse(recovery.stopped_read_error(kind, message)['precise_original_aggregation_call_failure_proven'])
        for kind, message in (('ValueError', 'omitted actual guarded attempt'),
            ('FileNotFoundError', "[Errno 2] No such file or directory: '/unrelated'"),
            ('FileNotFoundError', '[Errno 2] No such file or directory\nRuntimeError: bad'),
            ('OSError', '[Errno 5] Input/output error')):
            with self.assertRaises(ValueError):
                recovery.stopped_read_error(kind, message)

    def test_full_supervisor_output_must_preserve_same_failed_result(self):
        values = self.fixture(); claim, process, result, state, pub, parts, tasks = values
        with self.assertRaisesRegex(ValueError, 'same failure result'):
            recovery.validate_terminal(claim, process, result, state, json.dumps(dict(result, error='changed')),
                pub, parts, tasks, alive=lambda row:False)

    def test_healthy_or_missing200_refuses_before_own_or_raw_source_or_fresh_reads(self):
        for message in ('healthy producer', 'missing actual200'):
            with patch.object(recovery, 'caps'), patch.object(recovery.ctl, 'final_producer', side_effect=ValueError(message)), \
                    patch.object(recovery.reader, 'publication') as source, patch.object(recovery.reader, 'load_fixed') as pinned, \
                    patch.object(recovery.reader, 'fresh_attempt') as fresh:
                with self.assertRaisesRegex(ValueError, message):
                    recovery.verify('a'*40, 'b'*40)
                source.assert_not_called(); pinned.assert_not_called(); fresh.assert_not_called()

    def test_exact_retained_namespaces_missing_extra_or_normal_pass_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(recovery.reader, 'OWNER', os.getuid()), \
                patch.object(recovery.ctl, 'OWNER', os.getuid()), patch.object(recovery.ctl, 'live', return_value=False):
            root = Path(folder).resolve(); values = self.fixture()
            claim, process, result, state, pub, parts, tasks = values
            process['uid'] = os.getuid()
            result['process'] = process; state['process'] = process
            claim_file = write(root/'claim.json', claim); result['claim_sha256'] = claim_file['sha256']
            write(root/'process.json', process); write(root/'result.json', result); write(root/'state.json', state)
            write(root/'supervisor.log', (json.dumps(result)+'\n').encode(), binary=True)
            (root/'accepted').mkdir(); (root/'attempts').mkdir()
            for task in tasks:
                write(root/'accepted'/('%06d.json'%task['attempt_index']), {})
                (root/'attempts'/task['attempt_id']).mkdir()
            raw = SimpleNamespace(plan=lambda:parts, actual_tasks=lambda actual:tasks)
            typed = ({'run_id':recovery.reader.RUN}, {'producer_job':'719225'}, 'actual FAILED1:0', {'commit':'b'*40})
            with patch.object(recovery, 'ORIGINAL', root), patch.object(recovery.reader, 'load_fixed', return_value=(raw, pub)), \
                    patch.object(recovery.ctl, 'final_producer', return_value=typed):
                _, actual_tasks, proof, _ = recovery.original_failure('b'*40)
                self.assertEqual(actual_tasks, tasks); self.assertEqual(proof['normal_return_attempts'], 201)
                missing = root/'accepted/000006.json'; original_bytes = missing.read_bytes(); missing.unlink()
                with self.assertRaisesRegex(ValueError, 'missing or extra'):
                    recovery.original_failure('b'*40)
                missing.write_bytes(original_bytes)
                extra = root/'accepted/000202.json'; extra.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'missing or extra'):
                    recovery.original_failure('b'*40)
                extra.unlink()
                write(root/'result.json', dict(result, status=original.STATUS))
                with self.assertRaisesRegex(ValueError, 'live, unknown, successful'):
                    recovery.original_failure('b'*40)

    def test_fresh_full_guarded_conjunction_positive_and_changed_final_failure_rejects(self):
        claim, process, _, _, pub, _, tasks = self.fixture()
        repository = Path(recovery.__file__).resolve().parents[2]
        failure = dict(source=pub, original_process=process, fingerprint='a'*64, producer_final_cpu_recovery={'commit':'b'*40})
        anchors = dict(result={'path':'/original/result', 'sha256':'c'*64}, claim={'path':'/original/claim', 'sha256':'d'*64})
        records = [{'task':task} for task in tasks]; originals = [{'attempt_id':task['attempt_id']} for task in tasks]
        attempts = [dict(attempt_id=task['attempt_id'], attempt_index=task['attempt_index']) for task in tasks]
        terminal = (None, tasks, failure, anchors)
        with patch.object(recovery, 'caps'), patch.object(recovery, 'original_failure', return_value=terminal) as original_read, \
                patch.object(recovery.reader, 'publication', return_value=(repository, {'commit':'f'*40})), \
                patch.dict(os.environ, QPROBE_SOURCE=str(repository)), \
                patch.object(recovery, 'reconsume_children', return_value=(records, originals)), \
                patch.object(recovery.reader, 'document', return_value=({'run_id':recovery.reader.RUN}, {})), \
                patch.object(recovery.reader, 'SharedReads') as shared, patch.object(recovery, 'recheck_anchors') as recheck, \
                patch('reproduction.hpc3_qprobe.verify_goal.history_parts', return_value=[]), \
                patch('reproduction.hpc3_qprobe.verify_goal.verify_attempt_parts', return_value=(attempts, [], [])):
            value = recovery.verify('f'*40, 'b'*40)
            self.assertEqual(value['status'], recovery.STATUS); self.assertEqual(value['fully_skipped_attempts'], 1)
            self.assertFalse(value['goal_complete']); self.assertTrue(value['original_raw_status_remains_failed'])
            shared.return_value.finish.assert_called_once(); recheck.assert_called_once_with(anchors)
            original_read.side_effect = [terminal, (None, tasks, dict(failure, fingerprint='e'*64), anchors)]
            with self.assertRaisesRegex(ValueError, 'changed during'):
                recovery.verify('f'*40, 'b'*40)


class FreshReceiptTests(unittest.TestCase):
    def fixture(self, root):
        base, task, output = fixtures.AttemptTests().fixture(root)
        task['attempt_index'] = 1
        directory = root/'raw/attempts'/task['attempt_id']; parent = dict(pid=100, sid=100, start_ticks=101, uid=os.getuid())
        child = dict(pid=101, sid=100, start_ticks=102, uid=os.getuid())
        publication = {'commit':recovery.RAW}
        claim = write(directory/'claim.json', dict(task=task, process=parent, source=publication))
        write(directory/'child.json', child)
        stdout = write(directory/'audit.stdout', output); stderr = write(directory/'audit.stderr', b'', binary=True)
        result = write(directory/'result.json', dict(task=task, process=parent, child=child, source=publication,
            actual_child_exit_code=0, claim_sha256=claim['sha256'], stdout=stdout, stderr=stderr))
        record = dict(task=task, result_path=result['path'], result_sha256=result['sha256'],
            output_sha256=stdout['sha256'], raw_sha256=output['raw_file']['sha256'],
            private_trace_sha256=output['private_trace_file']['sha256'], included_actor_token_count=10)
        accepted = write(root/'raw/accepted/000001.json', record)
        raw = SimpleNamespace(consume_task=lambda *args:copy.deepcopy(record))
        return base, task, raw, parent, directory, record, accepted, output

    def test_original_child_receipts_and_fresh_raw_private_sources_are_fully_bound(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(recovery.reader, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'; base, task, raw, parent, directory, record, _, output = self.fixture(root)
            with patch.object(recovery, 'ORIGINAL', root/'raw'), patch.object(recovery.reader, 'BASE', base), \
                    patch.object(recovery.reader, 'ROOT', root), patch.object(recovery.ctl, 'live', return_value=False):
                shared = recovery.reader.SharedReads()
                records, originals = recovery.reconsume_children(raw, [task], parent, shared); shared.finish()
                self.assertEqual(records, [record]); self.assertEqual(originals[0]['attempt_id'], task['attempt_id'])
                for name in ('claim.json', 'child.json', 'audit.stderr', 'audit.stdout', 'result.json'):
                    path = directory/name; data = path.read_bytes(); path.write_bytes(data+b' ')
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'changed'):
                        shared.finish()
                    path.write_bytes(data)
                    # Refresh a new retained full-reader after deliberate mutation.
                    shared = recovery.reader.SharedReads()
                    recovery.reconsume_children(raw, [task], parent, shared)

    def test_missing_changed_accepted_bad_wait_or_live_child_is_not_reaggregated(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(recovery.reader, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'; base, task, raw, parent, directory, record, accepted, _ = self.fixture(root)
            with patch.object(recovery, 'ORIGINAL', root/'raw'), patch.object(recovery.reader, 'BASE', base), \
                    patch.object(recovery.reader, 'ROOT', root), patch.object(recovery.ctl, 'live', return_value=False):
                write(Path(accepted['path']), dict(record, output_sha256='a'*64))
                with self.assertRaisesRegex(ValueError, 'accepted child differs'):
                    recovery.reconsume_children(raw, [task], parent, recovery.reader.SharedReads())
                write(Path(accepted['path']), record)
                result = json.loads((directory/'result.json').read_text()); result['actual_child_exit_code'] = False
                write(directory/'result.json', result)
                with self.assertRaisesRegex(ValueError, 'integer wait0'):
                    recovery.reconsume_children(raw, [task], parent, recovery.reader.SharedReads())


if __name__ == '__main__':
    unittest.main()
