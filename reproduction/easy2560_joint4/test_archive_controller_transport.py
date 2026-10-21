"""CPU-only transport regressions; no SSH, Slurm, or archive writes."""
import io
import json
from contextlib import contextmanager
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

from reproduction.easy2560_joint4 import archive_controller as controller
from reproduction.prefix20_joint4 import archive_supervisor as supervisor


BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')


class PinnedTransportTests(unittest.TestCase):
    def test_observation_snapshot_stream_and_accept_keep_one_pinned_login(self):
        commands = [
            shlex.join([str(BASE/'env/bin/python'), '-B', '-c',
                        controller.checkpoint_status_code(BASE, 5, '745371')]),
            controller.source_command(BASE, 'snapshot', '--base', BASE,
                                      '--commit', 'a'*40, '--step', 5),
            controller.source_command(BASE, 'stream', '--snapshot',
                                      BASE/'preparation/checkpoint-archives/global_step_5/source-snapshot.json'),
            controller.source_command(BASE, 'accept', '--snapshot',
                                      BASE/'preparation/checkpoint-archives/global_step_5/source-snapshot.json'),
        ]
        for command in commands:
            with self.subTest(command=command):
                argv = controller.ssh_argv(command)
                self.assertEqual(argv[-2], 'jxu957@10.120.48.31')
                self.assertIn('-oStrictHostKeyChecking=yes', argv)
                self.assertIn('-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn', argv)
                self.assertIn('-oUserKnownHostsFile='+supervisor.KNOWN_HOSTS, argv)
                self.assertIn('-oControlMaster=no', argv)
                self.assertNotIn('StrictHostKeyChecking=no', ' '.join(argv))
                wrapper = shlex.split(argv[-1])
                self.assertEqual(wrapper[:3], ['python3', '-B', '-c'])
                self.assertIn(repr(command), wrapper[3])
                self.assertIn('UNKNOWN_HOST_IDENTITY_MISMATCH', wrapper[3])
                self.assertIn('ACD-Login-1', wrapper[3])
                self.assertIn(supervisor.HPC_IDENTITY['boot_id'], wrapper[3])

    def test_transport_executes_only_after_exact_source_identity(self):
        command = "python -c 'print(123)'"
        code = shlex.split(controller.ssh_argv(command)[-1])[3]
        expected = supervisor.HPC_IDENTITY
        for actual_host, actual_boot, accepted in (
                (expected['hostname'], expected['boot_id'], True),
                ('ACD-Login-2', expected['boot_id'], False),
                (expected['hostname'], 'different-boot-id', False)):
            with self.subTest(host=actual_host, boot=actual_boot), patch(
                    'socket.gethostname', return_value=actual_host), patch(
                    'os.readlink', return_value=expected['pid_namespace']), patch(
                    'builtins.open', mock_open(read_data=actual_boot+'\n')), patch(
                    'os.execv') as execute, patch('sys.stderr', new_callable=io.StringIO) as error:
                if accepted:
                    exec(code, {})
                    execute.assert_called_once_with('/bin/bash', ['bash', '-c', command])
                    self.assertEqual(error.getvalue(), '')
                else:
                    with self.assertRaises(SystemExit) as caught:
                        exec(code, {})
                    self.assertEqual(caught.exception.code, 86)
                    execute.assert_not_called()
                    self.assertEqual(json.loads(error.getvalue())['status'], 'UNKNOWN_HOST_IDENTITY_MISMATCH')


class ObservationRetryTests(unittest.TestCase):
    @staticmethod
    def success():
        return subprocess.CompletedProcess([], 0, json.dumps(dict(
            ready=False, job_state='RUNNING', queue_returncode=0, queue_error='')), '')

    def exercise(self, responses, *, attempts=3):
        events = []
        env = {'SSHPASS': 'not-real-fixture-secret'}
        with tempfile.TemporaryDirectory() as td, patch.object(
                controller.subprocess, 'run', side_effect=responses) as invoke, patch.object(
                controller.time, 'sleep') as sleep, patch.object(
                controller, 'event', side_effect=lambda root, **value: events.append(value)):
            result = controller.observe_checkpoint(BASE, 5, '745371', env, Path(td),
                attempts=attempts, retry_seconds=15, completed_steps=())
            calls = list(invoke.call_args_list)
            sleeps = list(sleep.call_args_list)
        self.assertEqual(env, {'SSHPASS': 'not-real-fixture-secret'})
        for call in calls:
            self.assertEqual(call.args[0][-2], 'jxu957@10.120.48.31')
            self.assertIs(call.kwargs['env'], env)
        self.assertNotIn('not-real-fixture-secret', json.dumps(events))
        return result, calls, sleeps, events

    def test_transient_auth_denial_retries_observation_then_reports_real_running(self):
        failed = subprocess.CompletedProcess([], 5, '', 'Permission denied, please try again.')
        result, calls, sleeps, events = self.exercise([failed, self.success()])
        self.assertEqual(result['job_state'], 'RUNNING')
        self.assertFalse(result['ready'])
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.args for call in sleeps], [(15,)])
        self.assertEqual([row['phase'] for row in events], ['SOURCE_OBSERVATION_RETRY'])
        self.assertEqual(calls[0].args[0], calls[1].args[0])

    def test_connection_error_and_timeout_have_one_bounded_read_only_retry_chain(self):
        failed = subprocess.CompletedProcess([], 255, '', 'Connection reset by peer')
        timeout = subprocess.TimeoutExpired(['ssh'], 90)
        result, calls, sleeps, events = self.exercise([failed, timeout, self.success()])
        self.assertEqual(result['job_state'], 'RUNNING')
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(sleeps), 2)
        self.assertEqual([row['phase'] for row in events], ['SOURCE_OBSERVATION_RETRY']*2)

    def test_exhausted_transport_error_stays_unknown_and_never_returns_failed_job(self):
        failed = subprocess.CompletedProcess([], 255, '', 'Connection reset by peer')
        with tempfile.TemporaryDirectory() as td, patch.object(
                controller.subprocess, 'run', side_effect=[failed]*3) as invoke, patch.object(
                controller.time, 'sleep') as sleep, patch.object(controller, 'event'):
            with self.assertRaises(RuntimeError) as caught:
                controller.observe_checkpoint(BASE, 5, '745371', {}, Path(td),
                    attempts=3, retry_seconds=15)
            self.assertIn('OBSERVATION', str(caught.exception))
            self.assertIn('255', str(caught.exception))
            self.assertEqual(invoke.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_identity_hostkey_and_remote_execution_errors_are_not_retried(self):
        for code, stderr in ((86, 'UNKNOWN_HOST_IDENTITY_MISMATCH'),
                             (6, 'Host key is unknown'), (1, 'AssertionError')):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as td, patch.object(
                    controller.subprocess, 'run', return_value=subprocess.CompletedProcess(
                        [], code, '', stderr)) as invoke, patch.object(
                    controller.time, 'sleep') as sleep, patch.object(controller, 'event'):
                with self.assertRaises(RuntimeError):
                    controller.observe_checkpoint(BASE, 5, '745371', {}, Path(td),
                        attempts=3, retry_seconds=15)
                self.assertEqual(invoke.call_count, 1)
                sleep.assert_not_called()

    def test_successful_transport_with_invalid_payload_is_not_retried(self):
        invalid = subprocess.CompletedProcess([], 0, 'not-json', '')
        with tempfile.TemporaryDirectory() as td, patch.object(
                controller.subprocess, 'run', return_value=invalid) as invoke, patch.object(
                controller.time, 'sleep') as sleep, patch.object(controller, 'event'):
            with self.assertRaises((ValueError, RuntimeError)):
                controller.observe_checkpoint(BASE, 5, '745371', {}, Path(td),
                    attempts=3, retry_seconds=15)
            self.assertEqual(invoke.call_count, 1)
            sleep.assert_not_called()


class RecoveryRootTests(unittest.TestCase):
    COMMIT = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
    JOB = '745371'
    SUFFIX = 'transport-r1-20261021'

    def parent(self, storage):
        root = storage/'archive-controller'/('job-'+self.JOB)
        root.mkdir(parents=True)
        claim = dict(base=str(BASE), source_commit=self.COMMIT, source_job=self.JOB,
            birth=dict(pid=99999999, start_ticks=12345, boot_id='fixture-boot', uid=1002),
            credentials_persisted=False)
        (root/'claim.json').write_text(json.dumps(claim)+'\n')
        (root/'failure.json').write_text(json.dumps(dict(error_type='RuntimeError',
            error='transient SSH permission denied', completed_steps=[]))+'\n')
        (root/'progress.json').write_text('{"phase":"WAITING_CHECKPOINT"}\n')
        return root, claim

    @contextmanager
    def source_process(self, *, state=None, ticks=12345, boot='fixture-boot', uid=1002):
        """Emulate only the recorded /proc birth; all fixture files remain real."""
        original_read = Path.read_text
        original_stat = Path.stat
        def read(path, *args, **kwargs):
            if str(path) == '/proc/self/stat':
                return '123 (python) '+' '.join(['S']+['0']*18+['23456'])
            if str(path) == '/proc/99999999/stat':
                if state is None:
                    raise FileNotFoundError(str(path))
                return '99999999 (python) '+' '.join([state]+['0']*18+[str(ticks)])
            if str(path) == '/proc/sys/kernel/random/boot_id':
                return boot+'\n'
            return original_read(path, *args, **kwargs)
        def stat(path, *args, **kwargs):
            if str(path) == '/proc/99999999':
                return SimpleNamespace(st_uid=uid)
            return original_stat(path, *args, **kwargs)
        with patch.object(Path, 'read_text', new=read), patch.object(Path, 'stat', new=stat):
            yield

    def test_recovery_child_preserves_failed_claim_and_all_parent_evidence(self):
        with tempfile.TemporaryDirectory() as td, patch.object(controller, 'HK_ROOT', Path(td)):
            root, _ = self.parent(Path(td))
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            with self.source_process():
                child = controller.controller_root(BASE, self.COMMIT, self.JOB, self.SUFFIX)
            self.assertEqual(child, root/('recovery-'+self.SUFFIX))
            self.assertFalse(child.exists())
            self.assertEqual({p.name:p.read_bytes() for p in root.iterdir()}, before)
            self.assertEqual(controller.controller_root(BASE, self.COMMIT, self.JOB), root)

    def test_invalid_suffix_cannot_escape_or_reuse_the_parent_directory(self):
        with tempfile.TemporaryDirectory() as td, patch.object(controller, 'HK_ROOT', Path(td)):
            for suffix in ('', '../escape', 'UPPER', '-leading', 'x'*65):
                with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                    controller.controller_root(BASE, self.COMMIT, self.JOB, suffix)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_recovery_requires_a_real_parent_failure(self):
        with tempfile.TemporaryDirectory() as td, patch.object(controller, 'HK_ROOT', Path(td)):
            root, _ = self.parent(Path(td))
            (root/'failure.json').unlink()
            with self.assertRaises(FileNotFoundError):
                controller.controller_root(BASE, self.COMMIT, self.JOB, self.SUFFIX)
            self.assertFalse((root/('recovery-'+self.SUFFIX)).exists())

    def test_recovery_rejects_foreign_base_source_commit_job_and_missing_error(self):
        mutations = (dict(base='/data/user/jxu957/experiments/easy2560-corollout-joint4-other'),
                     dict(source_commit='a'*40), dict(source_job='different-job'))
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td, patch.object(
                    controller, 'HK_ROOT', Path(td)):
                root, claim = self.parent(Path(td)); claim.update(mutation)
                (root/'claim.json').write_text(json.dumps(claim))
                with self.assertRaises(ValueError):
                    controller.controller_root(BASE, self.COMMIT, self.JOB, self.SUFFIX)
        with tempfile.TemporaryDirectory() as td, patch.object(controller, 'HK_ROOT', Path(td)):
            root, _ = self.parent(Path(td)); (root/'failure.json').write_text('{}')
            with self.assertRaises(ValueError):
                controller.controller_root(BASE, self.COMMIT, self.JOB, self.SUFFIX)

    def test_recovery_rejects_same_live_process_birth(self):
        with tempfile.TemporaryDirectory() as td, patch.object(controller, 'HK_ROOT', Path(td)):
            root, _ = self.parent(Path(td)); before = (root/'claim.json').read_bytes()
            with self.source_process(state='S'), self.assertRaisesRegex(ValueError, 'STILL_LIVE'):
                controller.controller_root(BASE, self.COMMIT, self.JOB, self.SUFFIX)
            self.assertEqual((root/'claim.json').read_bytes(), before)
            self.assertFalse((root/('recovery-'+self.SUFFIX)).exists())

    def test_reused_pid_or_zombie_is_not_a_live_matching_controller(self):
        for state, ticks in (('S', 54321), ('Z', 12345)):
            with self.subTest(state=state, ticks=ticks), tempfile.TemporaryDirectory() as td, patch.object(
                    controller, 'HK_ROOT', Path(td)):
                root, _ = self.parent(Path(td))
                with self.source_process(state=state, ticks=ticks):
                    self.assertEqual(controller.controller_root(BASE, self.COMMIT, self.JOB, self.SUFFIX),
                                     root/('recovery-'+self.SUFFIX))

    def test_recovery_failure_records_once_without_replaying_snapshot_or_parent(self):
        with tempfile.TemporaryDirectory() as td, patch.object(controller, 'HK_ROOT', Path(td)):
            root, _ = self.parent(Path(td))
            before = {name:(root/name).read_bytes() for name in ('claim.json', 'failure.json', 'progress.json')}
            failed = subprocess.CompletedProcess([], 255, '', 'Connection reset during snapshot')
            opened_locks = []
            original_open = Path.open
            def tracked_open(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                if path.name in ('controller-family.lock', 'controller.lock'):
                    opened_locks.append(stream)
                return stream
            with self.source_process(), patch.object(Path, 'open', new=tracked_open), patch.object(
                    controller.socket, 'gethostname', return_value='xielab3'), patch.object(
                    controller.os, 'getuid', return_value=1002), patch.object(
                    controller, 'observe_checkpoint', return_value={'ready':True}), patch.object(
                    controller.subprocess, 'run', return_value=failed) as snapshot, patch.object(
                    controller.subprocess, 'Popen') as stream, patch.object(controller, 'event'):
                try:
                    with self.assertRaisesRegex(RuntimeError, 'SNAPSHOT_FAILED'):
                        controller.run(BASE, self.COMMIT, self.JOB, 'not-real-fixture-secret',
                            first_step=5, controller_suffix=self.SUFFIX)
                finally:
                    # The real controller has a process lifetime; this fixture
                    # invokes run in-process and explicitly releases its locks.
                    for handle in opened_locks:
                        handle.close()
            snapshot.assert_called_once()
            stream.assert_not_called()
            child = root/('recovery-'+self.SUFFIX)
            self.assertEqual({name:(root/name).read_bytes() for name in before}, before)
            claim = json.loads((child/'claim.json').read_text())
            failure = json.loads((child/'failure.json').read_text())
            self.assertEqual(claim['recovery_parent'], str(root))
            self.assertEqual(claim['source_job'], self.JOB)
            self.assertEqual(failure['completed_steps'], [])
            self.assertFalse((child/'result.json').exists())
            self.assertNotIn('not-real-fixture-secret', '\n'.join(
                path.read_text() for path in child.glob('*.json')))


if __name__ == '__main__':
    unittest.main()
