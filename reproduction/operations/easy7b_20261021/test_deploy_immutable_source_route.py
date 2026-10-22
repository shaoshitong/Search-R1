"""Deployment route plumbing only; Git/network/credential transport are mocked."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.operations.easy7b_20261021 import deploy_immutable_source as deploy
from reproduction.operations.easy7b_20261021.watch_fa3_transition_local import DEFAULT_LOGIN_IDENTITY


ALTERNATE = dict(hostname='ACD-Login-2', boot_id='3c637b25-17cf-404c-971b-30bc4c4e9fbb',
                 pid_namespace='pid:[4026531836]')


class DeploymentRouteTests(unittest.TestCase):
    def run_mocked_entry(self, extra):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); repo = root/'work/repo'; repo.mkdir(parents=True)
            (repo/'README.md').write_text('committed source fixture\n')
            entry = repo/'reproduction/operations/easy7b_20261021/deploy_immutable_source.py'
            bridge = SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=0, stdout='{}', stderr='')))
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
            def git(command):
                operation = command[3]
                return {'status':b'', 'rev-parse':b'a'*40+b'\n',
                    'ls-remote':b'a'*40+b' refs/heads/codex/easy2560-qwen7b-20261021\n',
                    'ls-files':b'README.md\0', 'archive':b'archive fixture'}[operation]
            output = root/'receipt.json'
            with patch.object(deploy, '__file__', str(entry)), \
                 patch.object(deploy.subprocess, 'check_output', side_effect=git), \
                 patch('reproduction.easy2560_joint4.control.own_base', side_effect=Path), \
                 patch.object(deploy.importlib.util, 'spec_from_file_location', return_value=spec), \
                 patch.object(deploy.importlib.util, 'module_from_spec', return_value=bridge), \
                 redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                deploy.main(['--base', '/data/user/jxu957/experiments/easy2560-corollout-joint4-fixture',
                             '--output', str(output)] + extra)
            self.assertEqual(result.exception.code, False)
            bridge.run.assert_called_once()
            return bridge.run.call_args, json.loads(output.read_text())

    def test_default_route_keeps_the_original_login1_pin(self):
        call, receipt = self.run_mocked_entry([])
        self.assertEqual(call.kwargs, dict(login_host='10.120.48.31', observation_timeout=55,
                                          expected_host_identity=DEFAULT_LOGIN_IDENTITY))
        self.assertEqual(receipt['login_identity'], DEFAULT_LOGIN_IDENTITY)

    def test_explicit_login2_passes_the_complete_pin_without_fallback(self):
        call, receipt = self.run_mocked_entry(['--login-host', '10.120.48.32',
                                             '--login-identity-json', json.dumps(ALTERNATE)])
        self.assertEqual(call.kwargs, dict(login_host='10.120.48.32', observation_timeout=55,
                                          expected_host_identity=ALTERNATE))
        self.assertEqual(receipt['login_host'], '10.120.48.32')
        self.assertEqual(receipt['login_identity'], ALTERNATE)

    def test_missing_or_partial_pin_fails_before_git_or_bridge(self):
        for extra in ([], ['--login-identity-json', '{}']):
            with self.subTest(extra=extra), patch.object(deploy.subprocess, 'check_output') as git, \
                 patch.object(deploy.importlib.util, 'spec_from_file_location') as bridge, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                deploy.main(['--base', '/unused', '--output', '/unused', '--login-host', '10.120.48.32']+extra)
            self.assertEqual(result.exception.code, 2)
            git.assert_not_called(); bridge.assert_not_called()


if __name__ == '__main__':unittest.main()
