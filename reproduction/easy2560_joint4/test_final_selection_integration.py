"""Final-consumer dispatch tests, separate from saved trace semantic tests.

The verifier entry point is injected to observe its exact 200 authenticated
inputs. These fixtures intentionally have no generated candidate/token data;
``test_selection_evidence`` covers the real saved-content verifier.
"""
import builtins
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import final_cpu_acceptance as final
from reproduction.easy2560_joint4.test_authenticated_history import bind_fixture, NEW
from reproduction.easy2560_joint4.test_final_cpu_acceptance import FinalAcceptanceTests


MODULE = 'reproduction.easy2560_joint4.selection_evidence'


class FinalSelectionIntegrationTests(unittest.TestCase):
    def fixture(self, directory):
        fixture = FinalAcceptanceTests('runTest')
        fixture.fixture(directory)
        return fixture

    def verifier(self, calls, *, fail_at=None, before=None):
        def verify(base, run, update, *, expected_job):
            if before is not None:
                before()
            calls.append((base, copy.deepcopy(run), copy.deepcopy(update), expected_job))
            if update['step'] == fail_at:
                raise ValueError('fixture-selected-trace-failure-' + str(fail_at))
            return dict(step=update['step'], attempt_id=update['attempt_id'],
                        source_commit=run['source_commit'], slurm_job_id=expected_job,
                        fixture_scope='dispatch only; real verifier tested separately')
        return SimpleNamespace(verify_attempt=verify)

    def test_full200_dispatches_every_exact_completed_update_once_after_no201(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory); calls = []; validated = []
            real_no201 = final.no201
            def no201(*args, **kwargs):
                result = real_no201(*args, **kwargs)
                validated.append(result)
                return result
            def before():
                self.assertEqual(len(validated), 1)
                self.assertTrue(validated[0]['no_update201'])
            with patch.object(final, 'no201', side_effect=no201), \
                 patch.dict('sys.modules', {MODULE: self.verifier(calls, before=before)}):
                report = final.verify_final(f.base, load_canonical=f.load_canonical)
            self.assertEqual(len(calls), 200)
            self.assertEqual([row[2] for row in calls], f.updates)
            self.assertTrue(all(base == f.base and run == f.run and job == f.job
                                for base, run, _, job in calls))
            selection = report['selection_evidence']
            self.assertEqual(selection['verified_completed_outers'], 200)
            self.assertTrue(selection['full_training_history_verified'])
            self.assertEqual([row['step'] for row in selection['per_outer']], list(range(1, 201)))
            self.assertEqual(selection['source_segments'], [dict(
                first_outer=1, last_outer=200, completed_outers=200,
                base=str(f.base), run_id=f.run_id, source_commit=f.commit, slurm_job_id=f.job)])
            self.assertFalse(report['overall_goal_complete'])

    def test_authenticated_prefix1_to5_and_current6_to200_keep_original_source_and_job(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, _ = bind_fixture(Path(directory), final_mode=True)
            calls = []; authenticated = []; real_restore_chain = final.restore_chain
            def restore_chain(*args, **kwargs):
                chain = real_restore_chain(*args, **kwargs)
                authenticated.append(chain)
                return chain
            def before():
                self.assertEqual(len(authenticated), 1)
            with patch.object(final, 'restore_chain', side_effect=restore_chain), \
                 patch.dict('sys.modules', {MODULE: self.verifier(calls, before=before)}):
                report = final.verify_final(new.base, load_canonical=new.load_canonical, training_commit=NEW)
            self.assertEqual([row[2]['step'] for row in calls], list(range(1, 201)))
            self.assertTrue(all(base == old.base and run == old.run and job == old.job
                                for base, run, _, job in calls[:5]))
            self.assertTrue(all(base == new.base and run == new.run and job == new.job
                                for base, run, _, job in calls[5:]))
            self.assertEqual([row[2] for row in calls[:5]], old.updates)
            self.assertEqual([row[2] for row in calls[5:]], new.updates[5:])
            self.assertEqual([(row['first_outer'], row['last_outer'], row['slurm_job_id'])
                              for row in report['selection_evidence']['source_segments']],
                             [(1, 5, old.job), (6, 200, new.job)])

    def test_failure_at_any_completed_outer_propagates_without_a_final_report(self):
        for outer in (1, 5, 6, 100, 200):
            with self.subTest(outer=outer), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); calls = []
                with patch.dict('sys.modules', {MODULE: self.verifier(calls, fail_at=outer)}), \
                     patch.object(final, 'evaluations', side_effect=AssertionError('selection must fail first')):
                    with self.assertRaisesRegex(ValueError, 'fixture-selected-trace-failure-' + str(outer)):
                        final.verify_final(f.base, load_canonical=f.load_canonical)
                self.assertEqual([row[2]['step'] for row in calls], list(range(1, outer + 1)))

    def test_missing200_extra201_or_partial_tail_fail_before_any_selection_trace_reads(self):
        for mutation in ('missing200', 'extra201', 'partial'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                f = self.fixture(directory); calls = []
                rows = f.updates[:-1] if mutation == 'missing200' else f.updates
                if mutation == 'extra201': rows = rows + [dict(rows[-1], step=201)]
                f.ledger('logs/actual-updates.jsonl', rows,
                         suffix=b'{"partial":' if mutation == 'partial' else b'')
                with patch.dict('sys.modules', {MODULE: self.verifier(calls)}), \
                     patch.object(final, 'selection_evidence', wraps=final.selection_evidence) as aggregation:
                    with self.assertRaisesRegex(ValueError, 'MISSING200|PARTIAL_OR_OVERSIZED'):
                        final.verify_final(f.base, load_canonical=f.load_canonical)
                aggregation.assert_not_called()
                self.assertEqual(calls, [])

    def test_direct_final_aggregation_rejects_missing_duplicate_or_relabelled_completed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory)
            for mutation in ('missing', 'duplicate_step', 'duplicate_attempt_id',
                             'duplicate_attempt_index', 'foreign_run'):
                with self.subTest(mutation=mutation):
                    rows = copy.deepcopy(f.updates)
                    if mutation == 'missing': rows.pop()
                    elif mutation == 'duplicate_step': rows[-1]['step'] = 199
                    elif mutation == 'duplicate_attempt_id': rows[-1]['attempt_id'] = rows[-2]['attempt_id']
                    elif mutation == 'duplicate_attempt_index': rows[-1]['attempt_index'] = rows[-2]['attempt_index']
                    else: rows[-1]['run_id'] = 'f' * 32
                    calls = []
                    with patch.object(final, 'chained_lines', return_value=rows), \
                         patch.dict('sys.modules', {MODULE: self.verifier(calls)}):
                        with self.assertRaisesRegex(ValueError, 'SELECTION_FULL200|SELECTION_DUPLICATE'):
                            final.selection_evidence(f.base, f.run)
                    self.assertEqual(calls, [])

    def test_final_aggregation_rejects_current_job_differing_from_authenticated_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, _ = bind_fixture(Path(directory), final_mode=True)
            chain = final.restore_chain(new.base, new.run)
            chain['current_job'] = old.job
            calls = []
            with patch.dict('sys.modules', {MODULE: self.verifier(calls)}):
                with self.assertRaisesRegex(ValueError, 'SELECTION_RESTORE_CHAIN_DIFFERS'):
                    final.selection_evidence(new.base, new.run, chain)
            self.assertEqual(calls, [])

    def test_full_final_dispatch_is_cpu_only_and_does_not_start_processes_or_import_models(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.fixture(directory); calls = []; original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'ray', 'vllm', 'transformers'):
                    raise AssertionError('unexpected model/GPU import: ' + name)
                return original_import(name, *args, **kwargs)
            with patch.dict('sys.modules', {MODULE: self.verifier(calls)}), \
                 patch('builtins.__import__', side_effect=guarded_import), \
                 patch('subprocess.Popen', side_effect=AssertionError('CPU reader must not launch a process')):
                report = final.verify_final(f.base, load_canonical=f.load_canonical)
            self.assertEqual(len(calls), 200)
            for evidence in (report, report['selection_evidence']):
                self.assertEqual((evidence['new_training_updates'], evidence['gpu_allocations'],
                                  evidence['model_reloads']), (0, 0, 0))
            self.assertFalse(report['actual_restore_performed'])
            self.assertFalse(report['actual_cuda_restore_accepted'])


if __name__ == '__main__':
    unittest.main()
