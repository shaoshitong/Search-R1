"""One CPU integration regression; all SSH/process boundaries are mocked."""
import contextlib
import importlib.abc
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from reproduction.operations.qwen7b_resample import finalize_data as finalizer
from reproduction.operations.qwen7b_resample import pool_control


class FakeControlLoader(importlib.abc.Loader):
    def create_module(self, spec):return None
    def exec_module(self, module):
        def route(node, command):
            assert (node, command) == ('h100-10', '/usr/bin/python3 -')
            return ['ssh', 'mgmt-host', 'mock-retained-node10-command']
        module.route = route


class FinalizerTransportTest(unittest.TestCase):
    def test_actual_control_package_import_reaches_mock_ssh(self):
        original_spec = importlib.util.spec_from_file_location
        def spec(name, path, *args, **kwargs):
            if name == 'pool_ctl' and str(path) == '/Users/shaoshitong/Documents/remote_codex_control/control.py':
                return importlib.util.spec_from_loader(name, FakeControlLoader())
            return original_spec(name, path, *args, **kwargs)
        outcome = subprocess.CompletedProcess(['ssh'], 0, '{"status":"MOCK_TRANSPORT_ONLY"}', '')
        with mock.patch.object(importlib.util, 'spec_from_file_location', side_effect=spec), \
             mock.patch.dict(os.environ, {'Q7B_MGMT_CONTROL_PATH': ''}), \
             mock.patch.object(pool_control.subprocess, 'run', return_value=outcome) as ssh:
            bare = finalizer.load('finalizer_pool_control', Path(pool_control.__file__))
            with self.assertRaisesRegex(ImportError, 'relative import'):
                bare.transport('no remote execution')
            ssh.assert_not_called()
            standalone = finalizer.load('standalone_finalizer_transport_test', Path(finalizer.__file__))
            for module in (finalizer, standalone):
                with self.subTest(package=module.__package__), mock.patch.object(sys, 'argv', ['finalize_data.py', 'status']), \
                     mock.patch.object(sys, 'path', list(sys.path)), mock.patch.object(module, 'digest', return_value=module.AUDIT_SHA), \
                     mock.patch.object(module, 'write') as save, contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as ended:module.main()
                    self.assertEqual(ended.exception.code, 0)
                    save.assert_called_once()
            self.assertEqual(ssh.call_count, 2)
            for call in ssh.call_args_list:
                self.assertEqual(call.args[0], ['ssh', 'mgmt-host', 'mock-retained-node10-command'])
                self.assertIn("ACTION='status'", call.kwargs['input'])


if __name__ == '__main__':unittest.main()
