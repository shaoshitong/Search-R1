"""CPU route plumbing only; no credential bridge or network is executed."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.operations.easy7b_20261021 import watch_fa3_transition_local as watcher


ALTERNATE_HOST = '10.120.48.32'
ALTERNATE_IDENTITY = dict(hostname='ACD-Login-2',
    boot_id='3c637b25-17cf-404c-971b-30bc4c4e9fbb', pid_namespace='pid:[4026531836]')


class LoginRouteTests(unittest.TestCase):
    def test_original_default_identity_is_unchanged_and_not_mutable_by_caller(self):
        host, identity = watcher.resolve_login_route()
        self.assertEqual(host, '10.120.48.31')
        self.assertEqual(identity, dict(hostname='ACD-Login-1',
            boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]'))
        identity['boot_id'] = 'caller mutation'
        self.assertNotEqual(watcher.resolve_login_route()[1]['boot_id'], 'caller mutation')

    def test_cli_passes_explicit_alternate_pin_to_existing_bridge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'work').mkdir()
            fake_file = root/'work/repo/reproduction/operations/easy7b_20261021/watcher.py'
            bridge = SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=0, stdout='{}', stderr='')))
            with patch.object(watcher, '__file__', str(fake_file)), \
                 patch.object(watcher, 'load_bridge', return_value=bridge), \
                 patch.object(watcher, 'command', return_value='read-only observer command'), \
                 patch.object(watcher, 'summarize', return_value={'requested_followup_observed':True}), \
                 redirect_stdout(io.StringIO()):
                result = watcher.main(['--login-host', ALTERNATE_HOST,
                    '--login-identity-json', json.dumps(ALTERNATE_IDENTITY), '--observations', '1'])
            self.assertEqual(result, 0)
            bridge.run.assert_called_once()
            self.assertEqual(bridge.run.call_args.kwargs, dict(login_host=ALTERNATE_HOST,
                expected_host_identity=ALTERNATE_IDENTITY, observation_timeout=55))
            summary_path, = (root/'outputs/easy2560-qwen7b-20261021').glob('*/latest-summary.json')
            saved = json.loads(summary_path.read_text())
            self.assertEqual(saved['login_host'], ALTERNATE_HOST)
            self.assertEqual(saved['login_identity'], ALTERNATE_IDENTITY)

    def test_alternate_without_pin_is_rejected_before_loading_credential_bridge(self):
        with patch.object(watcher, 'load_bridge') as bridge, redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            watcher.main(['--login-host', ALTERNATE_HOST])
        self.assertEqual(error.exception.code, 2)
        bridge.assert_not_called()

    def test_partial_or_nonstring_identity_never_becomes_an_unpinned_route(self):
        for identity in ({}, {'hostname':'ACD-Login-2'},
                         dict(ALTERNATE_IDENTITY, boot_id=None),
                         dict(ALTERNATE_IDENTITY, pid_namespace=''),
                         dict(ALTERNATE_IDENTITY, unexpected='field'), []):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                watcher.resolve_login_route(ALTERNATE_HOST, json.dumps(identity))


if __name__ == '__main__':
    unittest.main()
