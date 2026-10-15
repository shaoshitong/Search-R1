"""Constructed CPU protocol tests, never evidence of native training/recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import final_cpu_ledger_recovery719225 as recovery


class TerminalFailureTests(unittest.TestCase):
    def fixture(self):
        run = dict(run_id=recovery.reader.RUN, source_commit=recovery.reader.PRODUCER,
                   training_world_size=4)
        jobs = {}
        for index, (name, code) in enumerate((('trainer', 0), ('final-acceptance', 1))):
            leader = dict(pid=100+index, sid=100+index, start_ticks=7+index, uid=14090,
                          state='S', command='original command')
            jobs[name] = dict(leader=leader, ledger={str(leader['pid']):copy.deepcopy(leader)},
                state='STOPPED', exit_code=code, log=str(recovery.BASE/'logs'/(name+'.log')))
        jobs['final-acceptance']['arguments'] = [str(recovery.BASE/'env/bin/python'), '-m',
            'reproduction.common8.verify_full', '--base', str(recovery.BASE)]
        pipeline = dict(run_id=run['run_id'], stage='FAILED', unix=1,
            error=repr(AssertionError('final-acceptance failed; see project log')))
        cleanup = dict(run_id=run['run_id'], cleaned=True, checked_unix=2)
        final = dict(run_id=run['run_id'], step=200, final=True, question_ids=['q']*51713,
            data_sources=['nq']*51713, em=[0]*51713, f1=[0.0]*51713,
            counts=dict(nq=51707, a=1, b=1, c=1, d=1, e=1, f=1))
        source = recovery.ROOT/'source'/recovery.reader.PRODUCER/'reproduction/common8'
        output = ('Traceback (most recent call last):\n'
            '  File "'+str(source/'verify_full.py')+'", line 158, in verify_full\n'
            '    scheduler_receipts = verify_scheduler_receipts(...)\n'
            '  File "'+str(source/'runtime.py')+'", line 181, in verify_scheduler_receipts\n'
            '    ledger = json.loads(path.read_text())\n'
            "FileNotFoundError: [Errno 2] No such file or directory: '"+
            str(recovery.BASE/'logs/process-ledger.json')+"'\n")
        return run, dict(run_id=run['run_id'], jobs=jobs), pipeline, cleanup, final, output

    def test_exact_failure_and_terminal_births_are_admitted(self):
        values = self.fixture()
        result = recovery.terminal_documents(*values, alive=lambda _:False)
        self.assertEqual(result['errno'], 2)
        # OwnedProcesses may record exec/state changes without changing birth.
        values[1]['jobs']['trainer']['ledger']['100']['command'] = 'after exec'
        values[1]['jobs']['trainer']['ledger']['100']['state'] = 'Z'
        recovery.terminal_documents(*values, alive=lambda _:False)

    def test_other_exit_cleanup_run_and_live_processes_are_rejected(self):
        mutations = (
            lambda v:v[1]['jobs']['trainer'].update(exit_code=1),
            lambda v:v[1]['jobs']['final-acceptance'].update(exit_code=0),
            lambda v:v[1]['jobs']['trainer'].update(state='CLEANUP_FAILED'),
            lambda v:v[3].update(cleaned=False),
            lambda v:v[3].update(checked_unix=0),
            lambda v:v[2].update(stage='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP'),
            lambda v:v[2].update(error='RuntimeError(\'GPU sampler failed\')'),
            lambda v:v[0].update(training_world_size=8),
            lambda v:v[4].update(step=201),
            lambda v:v[1]['jobs']['trainer']['ledger']['100'].update(start_ticks=99),
            lambda v:v[1]['jobs']['final-acceptance'].update(arguments=['python', 'other.py']),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                values = self.fixture(); mutate(values)
                with self.assertRaises(ValueError):
                    recovery.terminal_documents(*values, alive=lambda _:False)
        with self.assertRaisesRegex(ValueError, 'live/unbound'):
            recovery.terminal_documents(*self.fixture(), alive=lambda _:True)

    def test_unrelated_or_chained_or_wrong_source_traceback_is_rejected(self):
        original = self.fixture()[-1]
        bad = (
            original.replace('process-ledger.json', 'another-ledger.json'),
            original.replace('FileNotFoundError', 'PermissionError'),
            original.replace('[Errno 2]', '[Errno 116]'),
            original.replace(recovery.reader.PRODUCER, 'a'*40),
            original.replace('verify_scheduler_receipts\n', 'another_function\n'),
            original+'RuntimeError: CUDA out of memory\n',
            original+original,
            original.replace('FileNotFoundError:', 'During handling of the above exception\nFileNotFoundError:'),
        )
        for value in bad:
            with self.subTest(output=value[-150:]), self.assertRaises(ValueError):
                recovery.ledger_read_traceback(value)

    def test_observed_bare_read_enoent_requires_exact_original_ledger_call_chain(self):
        source = recovery.ROOT/'source'/recovery.reader.PRODUCER/'reproduction/common8'
        pathlib = Path(recovery.sysconfig.get_path('stdlib'))/'pathlib.py'
        start = ('Traceback (most recent call last):\n  File "'+str(source/'verify_full.py')+
                 '", line 158, in verify_full\n')
        bare = 'FileNotFoundError: [Errno 2] No such file or directory\n'
        json_read = (start+'  File "'+str(source/'runtime.py')+
            '", line 181, in verify_scheduler_receipts\n'+
            '  File "'+str(pathlib)+'", line 1135, in read_text\n'+bare)
        digest_read = (start+'  File "'+str(source/'runtime.py')+
            '", line 315, in verify_scheduler_receipts\n'+
            '  File "'+str(source/'verify_data.py')+'", line 17, in digest\n'+
            '  File "'+str(source/'verify_data.py')+'", line 17, in <lambda>\n'+bare)
        for value in (json_read, digest_read):
            proof = recovery.ledger_read_traceback(value)
            self.assertFalse(proof['filename_present'])
            self.assertIn(proof['read_site'], ('json-read', 'digest-read'))
        bad = (json_read.replace('line 181,', 'line 182,'),
            json_read.replace(str(pathlib), '/unrelated/python/pathlib.py'),
            digest_read.replace('line 17,', 'line 18,'),
            digest_read.replace('verify_data.py', 'another_reader.py'),
            digest_read.replace('in <lambda>', 'in another_function'),
            start+'  File "'+str(source/'runtime.py')+'", line 315, in verify_scheduler_receipts\n'+bare)
        for value in bad:
            with self.subTest(trace=value[-200:]), self.assertRaisesRegex(ValueError, 'read chain'):
                recovery.ledger_read_traceback(value)

    def test_explicit_digest_open_enoent_keeps_exact_file_and_original_tail(self):
        source = recovery.ROOT/'source'/recovery.reader.PRODUCER/'reproduction/common8'
        pathlib = Path(recovery.sysconfig.get_path('stdlib'))/'pathlib.py'
        value = ('Traceback (most recent call last):\n  File "'+str(source/'verify_full.py')+
            '", line 158, in verify_full\n  File "'+str(source/'runtime.py')+
            '", line 315, in verify_scheduler_receipts\n  File "'+str(source/'verify_data.py')+
            '", line 16, in digest\n  File "'+str(pathlib)+'", line 1119, in open\n'+
            "FileNotFoundError: [Errno 2] No such file or directory: '"+
            str(recovery.BASE/'logs/process-ledger.json')+"'\n")
        proof = recovery.ledger_read_traceback(value)
        self.assertTrue(proof['filename_present']); self.assertEqual(proof['read_site'], 'open')
        for bad in (value.replace('line 16,', 'line 18,'),
                    value.replace(str(pathlib), '/other/pathlib.py'),
                    value.replace('FileNotFoundError:', '  File "'+str(pathlib)+
                        '", line 1119, in open\nFileNotFoundError:'),
                    value.replace('process-ledger.json', 'wrong-file.json')):
            with self.subTest(trace=bad[-150:]), self.assertRaises(ValueError):
                recovery.ledger_read_traceback(bad)

    def test_only_actual_failed1_with_owner_and_end_time_passes_accounting(self):
        good = '719225|FAILED|1:0|jxu957|14090|jxu957|formal-user|emergency_acd|2026-10-15T12:00:00\n'
        with patch.object(recovery.subprocess, 'check_output', return_value=good):
            self.assertEqual(recovery.accounting()['state'], 'FAILED')
        for bad in (good.replace('FAILED', 'RUNNING'), good.replace('1:0', '0:0'),
                    good.replace('14090', '7'), good.replace('2026-10-15T12:00:00', 'Unknown'), good+good):
            with patch.object(recovery.subprocess, 'check_output', return_value=bad), self.assertRaises(ValueError):
                recovery.accounting()

    def test_healthy_running_or_missing200_creates_no_scope_and_launches_nothing(self):
        for failure in ('running', 'missing200'):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder).resolve()/'must-stay-absent'
                with patch.object(recovery, 'DIRECTORY', directory), patch.object(recovery, 'caps'), \
                        patch.object(recovery.ctl, 'owner'), patch.object(recovery.ctl, 'source') as source, \
                        patch.object(recovery.ctl, 'publish') as publish, \
                        patch.object(recovery.subprocess, 'Popen') as launch, \
                        patch.object(recovery, 'accounting') as accounting, \
                        patch.object(recovery.reader, 'early_history') as history:
                    if failure == 'running':
                        accounting.side_effect = ValueError('actual producer is running')
                    else:
                        accounting.return_value = dict(state='FAILED', exit_code='1:0')
                        history.side_effect = ValueError('missing200')
                    with self.assertRaises(ValueError): recovery.start('a'*40)
                    self.assertFalse(directory.exists())
                    source.assert_not_called(); publish.assert_not_called(); launch.assert_not_called()
                    if failure == 'running': history.assert_not_called()

    def test_unknown_existing_once_scope_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(recovery, 'DIRECTORY', Path(folder).resolve()), \
                    patch.object(recovery, 'caps'), \
                    patch.object(recovery, 'inspect', return_value=dict(status='UNKNOWN_CPU_RECOVERY_NO_REPLAY')) as inspect, \
                    patch.object(recovery, 'failure_evidence') as evidence, \
                    patch.object(recovery.subprocess, 'Popen') as launch:
                self.assertEqual(recovery.start('a'*40)['status'], 'UNKNOWN_CPU_RECOVERY_NO_REPLAY')
                inspect.assert_called_once_with('a'*40); evidence.assert_not_called(); launch.assert_not_called()


class HeldStructuralTests(unittest.TestCase):
    def fixture(self, directory):
        commit = 'a'*40; source = dict(commit=commit, publication_sha256='b'*64)
        failure = dict(fingerprint='c'*64, kind=recovery.FAILURE_KIND)
        parent = dict(pid=100, sid=100, start_ticks=9, uid=14090)
        child = dict(pid=102, sid=100, start_ticks=10, uid=14090)
        claim = dict(source=source, failure=failure, created_unix=1)
        structural = dict(structural_accepted=True, actual_outer_updates=200, run_id=recovery.reader.RUN)
        (directory/'structural-acceptance.json').write_text(json.dumps(structural))
        proof = dict(status=recovery.STATUS, source=source, failure_fingerprint=failure['fingerprint'],
            run_id=recovery.reader.RUN, completed_outer=200, world_size=4, final_native_questions=51713,
            actual_process=child, structural_acceptance=dict(path=str(directory/'structural-acceptance.json'),
                sha256=recovery.ctl.digest(directory/'structural-acceptance.json')),
            actor_model_loaded=False, training_performed=False, gpu_allocation_requested=False,
            original_receipts_overwritten=False, update201_performed=False, goal_complete=False,
            original_accounting_remains_failed=True,
            native4_restore_and_all_archives_and_independent_final_still_required=True)
        for name, value in (('claim', claim), ('process', parent), ('child', child)):
            (directory/(name+'.json')).write_text(json.dumps(value))
        (directory/'consumer.stdout').write_text(json.dumps(proof)); (directory/'consumer.stderr').write_bytes(b'')
        outputs = {name:dict(path=str(directory/('consumer.'+name)),
            bytes=(directory/('consumer.'+name)).stat().st_size,
            sha256=recovery.ctl.digest(directory/('consumer.'+name))) for name in ('stdout', 'stderr')}
        result = dict(source=source, claim_sha256=recovery.ctl.digest(directory/'claim.json'),
            process=parent, child=child, started_unix=2, completed_unix=3,
            actual_child_exit_code=0, supervisor_exit_code=0, error=None, **outputs)
        (directory/'result.json').write_text(json.dumps(result))
        return commit, result, proof, failure

    def test_real_wait_terminal_child_and_unchanged_stdout_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder).resolve(); commit, result, _, failure = self.fixture(directory)
            read = lambda p:json.loads(Path(p).read_text())
            original_file_reader = recovery.reader.read_file
            def local_file(*args, **kwargs):
                with patch.object(recovery.reader, 'OWNER', __import__('os').getuid()):
                    return original_file_reader(*args, **kwargs)
            with patch.object(recovery, 'DIRECTORY', directory), patch.object(recovery, 'caps'), \
                    patch.object(recovery.ctl, 'read', side_effect=read), \
                    patch.object(recovery.ctl, 'source', return_value=(directory, result['source'])), \
                    patch.object(recovery.ctl, 'live', return_value=False) as live, \
                    patch.object(recovery, 'preserved_failure', return_value=failure), \
                    patch.object(recovery.reader, 'read_file', side_effect=local_file):
                accepted = recovery.consume(commit)
                self.assertEqual(accepted['status'], recovery.HELD_STATUS)
                self.assertTrue(accepted['original_accounting_remains_failed'])
                self.assertFalse(accepted['goal_complete'])
                bad = copy.deepcopy(result); bad['actual_child_exit_code'] = 1
                (directory/'result.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'held CPU wait'): recovery.consume(commit)
                (directory/'result.json').write_text(json.dumps(result))
                live.return_value = True
                with self.assertRaisesRegex(ValueError, 'held CPU wait'): recovery.consume(commit)
                live.return_value = False
                p = directory/'consumer.stdout'; p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError, 'fresh original'): recovery.consume(commit)

    def test_stdout_cannot_claim_goal_complete_successful_producer_or_new_work(self):
        with tempfile.TemporaryDirectory() as folder:
            commit, _, value, failure = self.fixture(Path(folder).resolve())
            for key, bad in (('goal_complete', True), ('original_accounting_remains_failed', False),
                    ('actor_model_loaded', True), ('training_performed', True),
                    ('gpu_allocation_requested', True), ('update201_performed', True),
                    ('native4_restore_and_all_archives_and_independent_final_still_required', False)):
                proof = copy.deepcopy(value); proof[key] = bad
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'separate structural'):
                    recovery.accept_output(proof, commit, failure['fingerprint'])

    def test_changed_original_failure_or_preserved_bytes_are_rejected(self):
        claim = dict(failure=dict(fingerprint='a'*64, files={}))
        with patch.object(recovery, 'failure_evidence', return_value=(dict(fingerprint='b'*64), {})):
            with self.assertRaisesRegex(ValueError, 'changed after once'): recovery.preserved_failure(claim)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder).resolve(); path = directory/'failure/log.txt'
            path.parent.mkdir(); path.write_bytes(b'original failure')
            failure = dict(fingerprint='a'*64, files={'log.txt':dict(
                sha256=recovery.ctl.digest(path), bytes=path.stat().st_size)})
            with patch.object(recovery, 'DIRECTORY', directory), \
                    patch.object(recovery, 'failure_evidence', return_value=(failure, {})), \
                    patch.object(recovery.reader, 'OWNER', __import__('os').getuid()):
                recovery.preserved_failure(dict(failure=failure))
                path.write_bytes(b'changed! failure')
                with self.assertRaisesRegex(ValueError, 'fresh original'):
                    recovery.preserved_failure(dict(failure=failure))

    def test_task_requires_actual_bound_child_and_revalidates_all_history_and_five_pairs(self):
        from reproduction.common8 import verify_full as verifier
        from reproduction.hpc3_qprobe import verify_goal as goal
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder).resolve(); commit, result, _, failure = self.fixture(directory)
            # A successful child writes a new separate receipt only after all
            # readers returned. The real long training is deliberately absent.
            (directory/'structural-acceptance.json').unlink()
            child = result['child']; source = result['source']; run = dict(run_id=recovery.reader.RUN)
            full = dict(structural_accepted=True, run_id=recovery.reader.RUN, actual_outer_updates=200,
                final_source_em={'nq':0.0}, final_source_f1={'nq':0.0}, final_weighted_em=0.0, final_weighted_f1=0.0)
            def read(path):
                return run if Path(path).name == 'run-manifest.json' else json.loads(Path(path).read_text())
            def pair(base, run, step):
                path = directory/('pair-'+str(step)+'.json')
                return path, dict(path=str(path), sha256='d'*64, step=step)
            with patch.object(recovery, 'DIRECTORY', directory), patch.object(recovery, 'caps'), \
                    patch.dict(recovery.os.environ, QPROBE_SOURCE=str(directory)), \
                    patch.object(recovery.ctl, 'source', return_value=(directory, source)), \
                    patch.object(recovery.ctl, 'process', return_value=dict(child, state='R')), \
                    patch.object(recovery.ctl, 'read', side_effect=read), \
                    patch.object(recovery, 'preserved_failure', return_value=failure) as preserve, \
                    patch.object(verifier, 'verify_full', return_value=full) as verify_full, \
                    patch.object(goal, 'history_parts', return_value=['actual parts']) as history, \
                    patch.object(goal, 'part_updates', return_value=(['actual updates'], ['history proof'])), \
                    patch.object(goal, 'verify_evaluations', return_value=(['fixed five'],
                        dict(weighted_em=0.0, weighted_f1=0.0), {'nq':0.0}, {'nq':0.0})), \
                    patch.object(goal, 'verify_attempt_parts', return_value=(list(range(200)), ['evidence'], ['formal'])), \
                    patch.object(recovery.ctl, 'pair', side_effect=pair) as pairs:
                value = recovery.task(commit)
                verify_full.assert_called_once_with(recovery.BASE, publish_receipt=False)
                history.assert_called_once_with(recovery.BASE, run)
                self.assertEqual([call.args[2] for call in pairs.call_args_list], [5, 50, 100, 150, 200])
                self.assertEqual(preserve.call_count, 2)
                self.assertEqual(value['actual_normal_return_attempts'], 200)
                self.assertFalse(value['goal_complete'])
                self.assertEqual(json.loads((directory/'structural-acceptance.json').read_text()), full)

    def test_nonpublishing_full_wrapper_keeps_original_receipt_untouched(self):
        from reproduction.common8 import verify_full as verifier
        from reproduction.common8 import calibration_runtime
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve(); (base/'logs').mkdir()
            path = base/'logs/full-run-acceptance.json'; path.write_text('original failure evidence')
            proof = dict(constructed_test_only=True)
            with patch.object(verifier, '_verify_full', return_value=proof), \
                    patch.object(calibration_runtime, 'write_once') as publish:
                self.assertIs(verifier.verify_full(base, publish_receipt=False), proof)
                publish.assert_not_called()
                self.assertEqual(path.read_text(), 'original failure evidence')
                self.assertIs(verifier.verify_full(base), proof)
                publish.assert_called_once_with(path, proof)
            with patch.object(verifier, '_verify_full', side_effect=ValueError('failed original check')), \
                    patch.object(calibration_runtime, 'write_once') as publish:
                with self.assertRaisesRegex(ValueError, 'failed original check'): verifier.verify_full(base)
                publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
