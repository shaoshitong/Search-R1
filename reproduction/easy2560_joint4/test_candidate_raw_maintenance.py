"""CPU file/receipt lifecycle and exact-job observation tests; no live Slurm."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import candidate_raw_maintenance as runner
from reproduction.easy2560_joint4 import retire_candidate_raw as helper
from reproduction.easy2560_joint4 import test_retire_candidate_raw as raw_fixtures


def live():
    return dict(observation='LIVE', source_job='730080', state='RUNNING',
                authority='squeue', terminal_confirmed=False)


def terminal(state='COMPLETED', exit_code='0:0'):
    return dict(observation='TERMINAL', source_job='730080', state=state,
                exit_code=exit_code, authority='sacct', terminal_confirmed=True)


class JobObservationTests(unittest.TestCase):
    def command(self, responses, job='730080'):
        calls = []
        def execute(argv, **kwargs):
            calls.append(argv)
            self.assertIn(job, argv); self.assertNotIn('--allusers', argv)
            response = responses[len(calls)-1]
            if isinstance(response, Exception): raise response
            return NS(returncode=response[0], stdout=response[1], stderr=response[2])
        return execute, calls

    def test_same_live_handle_does_not_query_accounting(self):
        command, calls = self.command([(0, '730080|RUNNING\n', '')])
        observed = runner.observe_job('730080', command)
        self.assertEqual(observed['observation'], 'LIVE'); self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError): runner.observe_job('123', command)

    def test_terminal_requires_exact_main_job_accounting_and_end_time(self):
        command, calls = self.command([(0, '', ''), (0,
            '730080.batch|COMPLETED|0:0|2026-10-17T23:00:00\n'
            '730080|COMPLETED|0:0|2026-10-17T23:00:00\n', '')])
        observed = runner.observe_job('730080', command)
        self.assertEqual(observed['observation'], 'TERMINAL')
        self.assertTrue(observed['terminal_confirmed']); self.assertEqual(len(calls), 2)
        for output in ('730080.batch|COMPLETED|0:0|2026-10-17\n',
                       '730080|RUNNING|0:0|Unknown\n', '730080|FAILED|1:0|Unknown\n'):
            command, _ = self.command([(0, '', ''), (0, output, '')])
            self.assertEqual(runner.observe_job('730080', command)['observation'], 'UNKNOWN')

    def test_observation_errors_and_foreign_queue_rows_never_become_terminal(self):
        for response in ((2, '', 'controller unavailable'),
                         subprocess.TimeoutExpired(['squeue'], 20),
                         (0, '730080|\n', ''),
                         (0, '730080_1|RUNNING\n', '')):
            command, _ = self.command([response])
            observed = runner.observe_job('730080', command)
            self.assertEqual(observed['observation'], 'UNKNOWN')
            self.assertFalse(observed['terminal_confirmed'])

    def test_purged_queue_rc1_falls_through_to_exact_terminal_accounting_only(self):
        for accounting, expected in (
                ('730080|COMPLETED|0:0|2026-10-17T23:00:00\n', 'TERMINAL'),
                ('730080|RUNNING|0:0|Unknown\n', 'UNKNOWN'),
                ('730080.batch|COMPLETED|0:0|2026-10-17T23:00:00\n', 'UNKNOWN'),
                ('', 'UNKNOWN')):
            command, calls = self.command([(1, '', 'invalid job id'), (0, accounting, '')])
            observed = runner.observe_job('730080', command)
            self.assertEqual(observed['observation'], expected)
            self.assertEqual(len(calls), 2)

    def test_explicit_new_job_only_queries_exact_same_handle(self):
        command, calls = self.command([(1, '', 'invalid job id'),
            (0, '731234|COMPLETED|0:0|2026-10-18T01:00:00\n', '')], job='731234')
        observed = runner.observe_job('731234', command, expected_job='731234')
        self.assertEqual(observed['source_job'], '731234'); self.assertTrue(observed['terminal_confirmed'])
        self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(ValueError, 'CANNOT_OBSERVE_ANOTHER_JOB'):
            runner.observe_job('730080', command, expected_job='731234')
        self.assertEqual(len(calls), 2)


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = raw_fixtures.RetirementTests(); self.fixture.setUp(); self.base = self.fixture.base
        self.patch = patch.object(runner, 'RUN_ID', self.fixture.run_id); self.patch.start()
    def tearDown(self): self.patch.stop(); self.fixture.tearDown()

    def test_frontier_unchanged_never_regenerates_plan_and_new_frontier_uses_cheap_absent_skip(self):
        state = runner.initial_state()
        before = sum((self.base/self.fixture.pools[step]/name).stat().st_blocks*512
                     for step in (1, 2) for name in helper.RAW_NAMES)
        with patch.object(helper, 'build_plan', wraps=helper.build_plan) as build:
            self.assertFalse(runner.cycle(self.base, helper, state, live()))
            self.assertEqual(build.call_count, 1)
            first_path = state['operations'][0]['plan_path']
            self.assertEqual(state['confirmed_unlinked_allocated_bytes'], before)
            self.assertFalse(runner.cycle(self.base, helper, state, live()))
            self.assertEqual(build.call_count, 1)
            self.fixture.add_outer(5); self.fixture.ledger()
            original = helper.read_file
            def no_old_trace(base, relative, **kwargs):
                if 'behavior-traces' in str(relative):
                    self.assertNotIn(self.fixture.rows[0]['attempt_id'], str(relative))
                    self.assertNotIn(self.fixture.rows[1]['attempt_id'], str(relative))
                return original(base, relative, **kwargs)
            with patch.object(helper, 'read_file', side_effect=no_old_trace):
                runner.cycle(self.base, helper, state, live())
            self.assertEqual(build.call_count, 2)
        self.assertNotEqual(state['operations'][1]['plan_path'], first_path)
        self.assertTrue((self.base/self.fixture.pools[4]/'0.pt').exists())
        self.assertTrue((self.base/self.fixture.pools[5]/'0.pt').exists())
        self.assertFalse(state['filesystem_quota_free_bytes_measured'])
        # Actual filesystem allocation differs from tiny fixture logical size.
        self.assertGreater(state['confirmed_unlinked_allocated_bytes'], state['retired_logical_bytes'])

    def test_dry_observe_and_unknown_observation_have_no_retirement_or_claim(self):
        state = runner.initial_state()
        with patch.object(helper, 'build_plan', side_effect=AssertionError('must not plan')):
            runner.cycle(self.base, helper, state, live(), dry_observe=True)
            self.assertEqual(state['phase'], 'DRY_OBSERVE_ONLY')
            runner.cycle(self.base, helper, state,
                dict(observation='UNKNOWN', source_job='730080', terminal_confirmed=False))
            self.assertEqual(state['phase'], 'WAITING_OBSERVATION')
        self.assertFalse((self.base/'preparation/candidate-raw-retirement').exists())
        self.assertFalse((self.base/'preparation/candidate-raw-maintenance-runner').exists())

    def test_waits_for_three_full_successes_not_partial_completed_counters(self):
        self.fixture.rows[0].update(full_budget_complete=False, update_status='partial')
        self.fixture.rows[1].update(full_budget_complete=False, update_status='partial')
        self.fixture.ledger(); state = runner.initial_state()
        with patch.object(helper, 'build_plan', side_effect=AssertionError('only two full successes')):
            runner.cycle(self.base, helper, state, live())
        self.assertEqual(state['full_success_outer_count'], 2)
        self.assertTrue((self.base/self.fixture.pools[1]/'0.pt').exists())

    def test_abnormal_partial_raw_set_stops_before_other_eligible_pool_deletion(self):
        (self.base/self.fixture.pools[1]/'512.pt').unlink()
        state = runner.initial_state()
        with self.assertRaisesRegex(ValueError, 'PLAN_HAS_ABNORMAL_PRESERVED_POOLS'):
            runner.cycle(self.base, helper, state, live())
        self.assertEqual(state['operations'][0]['phase'], 'PLAN_SAVED')
        self.assertTrue((self.base/self.fixture.pools[1]/'0.pt').exists())
        self.assertTrue((self.base/self.fixture.pools[2]/'0.pt').exists())

    def test_terminal_failure_never_runs_plan_and_unconfirmed_terminal_is_rejected(self):
        state = runner.initial_state()
        with patch.object(helper, 'build_plan', side_effect=AssertionError('terminal failure')):
            self.assertTrue(runner.cycle(self.base, helper, state, terminal('FAILED', '1:0')))
        self.assertEqual(state['phase'], 'SOURCE_JOB_TERMINAL_FAILURE')
        with self.assertRaisesRegex(ValueError, 'UNCONFIRMED_TERMINAL'):
            runner.cycle(self.base, helper, state, dict(terminal(), authority='squeue', terminal_confirmed=False))

    def test_persistent_claim_keeps_exact_birth_and_refuses_even_a_dead_previous_claim(self):
        observations = iter([live(), terminal()]); sleeps = []
        birth = dict(pid=123, start_ticks=456, uid=14090, boot_id='boot-fixture', pid_namespace='pid:[fixture]')
        with patch.object(runner, 'process_identity', return_value=birth):
            result = runner.persistent(self.base, helper, helper_commit='c'*40, helper_sha256='d'*64,
                observer=lambda job: next(observations), sleeper=sleeps.append)
        root = self.base/'preparation/candidate-raw-maintenance-runner'
        claim = json.loads((root/'launch-claim.json').read_text())
        self.assertEqual(claim['process_identity'], birth)
        self.assertEqual(result['phase'], 'SOURCE_JOB_COMPLETED')
        self.assertEqual(len(result['operations']), 1); self.assertEqual(sleeps, [180])
        old_result = (root/'result.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'EXISTING_LAUNCH_CLAIM'):
            runner.persistent(self.base, helper, helper_commit='c'*40, helper_sha256='d'*64)
        self.assertEqual((root/'result.json').read_bytes(), old_result)

    def test_apply_failure_is_preserved_and_not_retried(self):
        calls = []
        def failed(*args, **kwargs): calls.append((args, kwargs)); raise OSError('constructed apply failure')
        with patch.object(runner, 'process_identity', return_value={'pid': 1, 'start_ticks': 2, 'boot_id': 'fixture'}), \
                patch.object(helper, 'apply_plan', side_effect=failed):
            with self.assertRaisesRegex(OSError, 'constructed apply failure'):
                runner.persistent(self.base, helper, helper_commit='c'*40, helper_sha256='d'*64,
                                  observer=lambda job: live(), sleeper=lambda seconds: None)
        self.assertEqual(len(calls), 1)
        root = self.base/'preparation/candidate-raw-maintenance-runner'
        failure = json.loads((root/'failure.json').read_text())
        self.assertEqual(failure['phase'], 'MAINTENANCE_FAILED_NO_AUTOMATIC_RETRY')
        self.assertTrue(Path(failure['operations'][0]['plan_path']).is_file())
        self.assertFalse((root/'result.json').exists())

    def test_stdlib_only_and_helper_exact_bytes_are_bound(self):
        source = Path(runner.__file__).read_text()
        imported = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): imported.append(node.module)
        self.assertFalse(any(name.startswith(('torch', 'verl', 'ray', 'numpy')) for name in imported))
        commit = 'c'*40; root = self.base/'preparation/candidate-raw-maintenance-code'/commit
        root.mkdir(parents=True); body = Path(helper.__file__).read_bytes(); sha = hashlib.sha256(body).hexdigest()
        (root/'retire_candidate_raw.py').write_bytes(body)
        manifest = dict(operator_commit=commit, training_commit=runner.TRAINING_COMMIT,
            run_id=runner.RUN_ID, source_job=runner.SOURCE_JOB, base=str(self.base),
            files={'retire_candidate_raw.py': dict(bytes=len(body), sha256=sha)})
        (root/'source-manifest.json').write_text(json.dumps(manifest))
        loaded = runner.load_helper(self.base, commit, sha)
        self.assertEqual(loaded.REVIEWED_TRAINING_COMMIT, runner.TRAINING_COMMIT)
        self.assertFalse((root/'__pycache__').exists())
        (root/'retire_candidate_raw.py').write_bytes(body+b'\n')
        with self.assertRaisesRegex(ValueError, 'HELPER_SOURCE_SHA_DIFFERS'):
            runner.load_helper(self.base, commit, sha)

    def deploy_fixture_helper(self, bound):
        commit = 'c'*40; root = self.base/'preparation/candidate-raw-maintenance-code'/commit
        root.mkdir(parents=True); body = Path(helper.__file__).read_bytes(); sha = hashlib.sha256(body).hexdigest()
        (root/'retire_candidate_raw.py').write_bytes(body)
        manifest = dict(operator_commit=commit, base=str(self.base), **bound,
                        files={'retire_candidate_raw.py': dict(bytes=len(body), sha256=sha)})
        (root/'source-manifest.json').write_text(json.dumps(manifest))
        return commit, sha

    def test_new_binding_reaches_helper_plan_apply_persistent_claim_and_observer(self):
        bound = self.fixture.use_published_source(); observed_jobs = []
        def observe(job):
            observed_jobs.append(job)
            return dict(terminal(), source_job=bound['source_job'])
        birth = dict(pid=123, start_ticks=456, uid=14090, boot_id='fixture', pid_namespace='pid:[fixture]')
        with patch.object(runner, 'process_identity', return_value=birth):
            result = runner.persistent(self.base, helper, helper_commit='c'*40, helper_sha256='d'*64,
                                       observer=observe, **bound)
        root = self.base/'preparation/candidate-raw-maintenance-runner'
        claim = json.loads((root/'launch-claim.json').read_text())
        self.assertEqual(observed_jobs, [bound['source_job']]); self.assertEqual(result['phase'], 'SOURCE_JOB_COMPLETED')
        self.assertTrue(all(claim[key] == value for key, value in bound.items()))
        self.assertEqual(result['retired_logical_bytes'], sum(len(('raw-'+str(s)+'-'+n).encode())
                         for s in (1, 2) for n in helper.RAW_NAMES))
        self.assertTrue((self.base/self.fixture.pools[3]/'0.pt').exists())
        self.assertFalse((self.base/self.fixture.pools[1]/'0.pt').exists())

    def test_explicit_cli_dry_observe_uses_new_manifest_scope_and_job_without_claim_or_plan(self):
        bound = self.fixture.use_published_source(); commit, sha = self.deploy_fixture_helper(bound)
        loaded = runner.load_helper(self.base, commit, sha, **bound)
        self.assertEqual(loaded.REVIEWED_TRAINING_COMMIT, helper.REVIEWED_TRAINING_COMMIT)
        with self.assertRaisesRegex(ValueError, 'HELPER_MANIFEST_RUN_BINDING_DIFFERS'):
            runner.load_helper(self.base, commit, sha)
        argv = ['candidate_raw_maintenance.py', '--dry-observe', '--base', str(self.base),
                '--run-id', bound['run_id'], '--source-job', bound['source_job'],
                '--training-commit', bound['training_commit'], '--helper-commit', commit, '--helper-sha256', sha]
        output = io.StringIO()
        with patch('sys.argv', argv), patch.object(runner, 'scope', return_value=self.base), \
             patch.object(runner, 'observe_job', return_value=dict(live(), source_job=bound['source_job'])) as observe, \
             redirect_stdout(output): runner.main()
        observe.assert_called_once_with(bound['source_job'], expected_job=bound['source_job'])
        result = json.loads(output.getvalue())
        self.assertEqual(result['phase'], 'DRY_OBSERVE_ONLY'); self.assertEqual(result['run_id'], bound['run_id'])
        self.assertFalse((self.base/'preparation/candidate-raw-maintenance-runner').exists())
        self.assertFalse((self.base/'preparation/candidate-raw-retirement').exists())

    def test_new_namespace_requires_full_explicit_identity_and_jxu_scope(self):
        path = '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018'
        bound = dict(run_id='b'*32, training_commit='8a5bac94b8230a99b81977f5290589e3d3708dfe', source_job='731234')
        with patch.object(runner.os, 'getuid', return_value=14090):
            self.assertEqual(runner.scope(path, **bound), Path(path))
            with self.assertRaisesRegex(ValueError, 'CANONICAL_JXU957_SCOPE_REQUIRED'):
                runner.scope('/data/user/jxu957/experiments/unrelated', **bound)
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_RUN_SOURCE_JOB_REQUIRED'):
                runner.scope(path, **dict(bound, training_commit='HEAD'))
        state = runner.initial_state()
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_JOB_DIFFERS'):
            runner.cycle(self.base, helper, state, live(), **bound)


if __name__ == '__main__': unittest.main()
