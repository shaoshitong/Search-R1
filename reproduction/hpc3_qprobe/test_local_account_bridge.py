"""Verify endpoint pinning without opening a connection or reading credentials."""
import ast
import shlex
import subprocess
import types
import unittest
from unittest.mock import Mock, patch

from reproduction.hpc3_qprobe import local_account_bridge as bridge


class LoginEndpointTest(unittest.TestCase):
    def invoke(self, **kwargs):
        campus = Mock(return_value=subprocess.CompletedProcess([], 0, 'safe-output', ''))
        module = types.SimpleNamespace(campus=campus)
        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        credential = '\u8d26\u53f7 jxu957 \u5bc6\u7801: `fixture-password`'
        with patch.object(bridge.importlib.util, 'spec_from_file_location', return_value=spec), \
             patch.object(bridge.importlib.util, 'module_from_spec', return_value=module), \
             patch.object(bridge.Path, 'read_text', return_value=credential):
            result = bridge.run('jxu957', 'printf readonly', '', '/fixture/control', **kwargs)
        call = campus.call_args
        python_code = shlex.split(call.args[1])[2]
        parsed = ast.parse(python_code)
        run_call = next(n for n in ast.walk(parsed) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
        ssh = ast.literal_eval(run_call.args[0])
        self.assertEqual(call.args[0], 'hkust3')
        self.assertNotIn('fixture-password', call.args[1])
        self.assertEqual(call.kwargs['observation_timeout'], 65)
        self.assertEqual(result.stdout, 'safe-output')
        self.assertIn('-oStrictHostKeyChecking=yes', ssh)
        self.assertIn('-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn', ssh)
        self.assertIn('-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts', ssh)
        self.assertEqual(call.kwargs['input'].splitlines()[0], 'fixture-password')
        return ssh, call.kwargs['input']

    def test_default_preserves_existing_gateway_and_exec_channel(self):
        ssh, payload = self.invoke()
        self.assertEqual(ssh[-2:], ['jxu957@10.120.81.30', 'printf readonly'])
        self.assertEqual(payload, 'fixture-password\n')

    def test_observed_login_nodes_can_be_pinned_without_relaxing_trust(self):
        for host in bridge.LOGIN_HOSTS[1:]:
            with self.subTest(host=host):
                ssh, _ = self.invoke(login_host=host)
                self.assertEqual(ssh[-2], 'jxu957@'+host)

    def test_pinned_shell_preserves_stdin_and_exit_propagation(self):
        ssh, payload = self.invoke(login_host='10.120.48.31', shell=True)
        self.assertEqual(ssh[-1], 'jxu957@10.120.48.31')
        self.assertIn('printf readonly\n_qprobe_bridge_status=$?\nexit $_qprobe_bridge_status\n', payload)

    def test_unapproved_endpoint_and_account_fail_before_credential_access(self):
        with patch.object(bridge.Path, 'read_text') as credential:
            for host in ('example.com', '10.120.48.31;id', '', None):
                with self.subTest(host=host), self.assertRaises(ValueError):
                    bridge.run('jxu957', 'readonly', '', '/fixture', login_host=host)
            with self.assertRaises(ValueError):
                bridge.run('other-account', 'readonly', '', '/fixture')
            credential.assert_not_called()


if __name__ == '__main__':
    unittest.main()
