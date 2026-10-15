"""Verify endpoint pinning without opening a connection or reading credentials."""
import ast
import io
import shlex
import subprocess
import types
import unittest
from unittest.mock import Mock, patch

from reproduction.hpc3_qprobe import local_account_bridge as bridge


class LoginEndpointTest(unittest.TestCase):
    HOST = dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
                pid_namespace='pid:[4026531836]')

    def invoke(self, **kwargs):
        campus = Mock(return_value=subprocess.CompletedProcess([], 0, 'safe-output', ''))
        module = types.SimpleNamespace(campus=campus)
        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        credential = '\u8d26\u53f7 jxu957 \u5bc6\u7801: `fixture-password`'
        with patch.object(bridge.importlib.util, 'spec_from_file_location', return_value=spec), \
             patch.object(bridge.importlib.util, 'module_from_spec', return_value=module), \
             patch.object(bridge.Path, 'read_text', return_value=credential), \
             patch.object(bridge,'_campus_long',side_effect=lambda command,payload,timeout:campus('hkust3',command,input=payload,observation_timeout=timeout)):
            result = bridge.run('jxu957', 'printf readonly', '', '/fixture/control', **kwargs)
        call = campus.call_args
        python_code = shlex.split(call.args[1])[2]
        parsed = ast.parse(python_code)
        run_call = next(n for n in ast.walk(parsed) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
        ssh = ast.literal_eval(run_call.args[0])
        self.assertEqual(call.args[0], 'hkust3')
        self.assertNotIn('fixture-password', call.args[1])
        self.assertEqual(call.kwargs['observation_timeout'], kwargs.get('observation_timeout',55)+10)
        observed=next(k.value for k in run_call.keywords if k.arg=='timeout')
        self.assertEqual(ast.literal_eval(observed),kwargs.get('observation_timeout',55))
        self.assertEqual(result.stdout, 'safe-output')
        self.assertIn('-oStrictHostKeyChecking=yes', ssh)
        self.assertIn('-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn', ssh)
        self.assertIn('-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts', ssh)
        self.assertEqual(call.kwargs['input'].splitlines()[0], 'fixture-password')
        return ssh, call.kwargs['input']

    def test_primary255_fallback_uses_same_long_deadline_and_keeps_strict_route(self):
        first=subprocess.CompletedProcess([],255,'','unavailable')
        second=subprocess.CompletedProcess([],0,'bound-result','')
        with patch.object(bridge.subprocess,'run',side_effect=[first,second]) as run:
            self.assertIs(bridge._campus_long('bound command','RAM stdin',1210),second)
            self.assertEqual(run.call_count,2)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['timeout'],1210)
                self.assertEqual(call.kwargs['input'],'RAM stdin')
                self.assertIn('-oStrictHostKeyChecking=yes',call.args[0])
                self.assertEqual(call.args[0][-2:],['hkust3','bound command'])
            self.assertIn('-oProxyJump=none',run.call_args_list[1].args[0])

    def test_long_observation_timeout_is_unknown_and_never_replays_on_fallback(self):
        for failure in (subprocess.TimeoutExpired('fixture',1210,output=b'partial evidence'),):
            with patch.object(bridge.subprocess,'run',side_effect=failure) as run:
                result=bridge._campus_long('bound command','RAM stdin',1210)
                self.assertEqual(result.returncode,124);self.assertEqual(result.stdout,'partial evidence')
                self.assertIn('inspect the existing',result.stderr);self.assertEqual(run.call_count,1)
        with patch.object(bridge.subprocess,'run',side_effect=[subprocess.CompletedProcess([],255,'',''),
                subprocess.TimeoutExpired('fixture',1210)]) as run:
            self.assertEqual(bridge._campus_long('bound command','RAM stdin',1210).returncode,124)
            self.assertEqual(run.call_count,2)

    def test_explicit_long_observation_preserves_endpoint_channel_and_RAM_credentials(self):
        ssh,_=self.invoke(login_host='10.120.48.31',expected_host_identity=self.HOST,observation_timeout=1200)
        self.assertEqual(ssh[-2],'jxu957@10.120.48.31')
        self.assertIn('-oControlMaster=no',ssh)

    def test_invalid_timeout_refuses_before_credential_or_transport(self):
        with patch.object(bridge.Path,'read_text') as credential:
            for timeout in (0,1801,True,55.5,'1200',None):
                with self.subTest(timeout=timeout),self.assertRaises(ValueError):
                    bridge.run('jxu957','readonly','','/fixture',observation_timeout=timeout)
            credential.assert_not_called()

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

    def test_kernel_guard_executes_only_after_exact_identity_match(self):
        ssh, _ = self.invoke(login_host='10.120.48.31', expected_host_identity=self.HOST)
        wrapper = shlex.split(ssh[-1]);self.assertEqual(wrapper[:2], ['python3', '-c'])
        for actual, should_exec in (
            (self.HOST, True),
            (dict(self.HOST, boot_id='new-boot'), False),
            (dict(self.HOST, hostname='ACD-Login-2'), False),
            (dict(self.HOST, pid_namespace='pid:[999]'), False),
        ):
            with self.subTest(actual=actual), \
                 patch('builtins.open', return_value=io.StringIO(actual['boot_id'])), \
                 patch('socket.gethostname', return_value=actual['hostname']), \
                 patch('os.readlink', return_value=actual['pid_namespace']), \
                 patch('os.execv') as execute, patch('sys.stderr', new_callable=io.StringIO):
                if should_exec:
                    exec(compile(wrapper[2], '<host-guard>', 'exec'), {})
                    execute.assert_called_once_with('/bin/bash', ['bash', '-c', 'printf readonly'])
                else:
                    with self.assertRaises(SystemExit) as error:
                        exec(compile(wrapper[2], '<host-guard>', 'exec'), {})
                    self.assertEqual(error.exception.code, 86)
                    execute.assert_not_called()

    def test_identity_cannot_be_ambiguous_or_attached_to_gateway(self):
        with patch.object(bridge.Path, 'read_text') as credential:
            for host, identity in (
                ('10.120.81.30', self.HOST),
                ('10.120.48.32', self.HOST),
                ('10.120.48.31', dict(self.HOST, pid_namespace='unbound')),
                ('10.120.48.31', dict(self.HOST, boot_id='invalid')),
                ('10.120.48.31', dict(hostname='ACD-Login-1')),
            ):
                with self.subTest(host=host, identity=identity), self.assertRaises(ValueError):
                    bridge.run('jxu957', 'readonly', '', '/fixture', login_host=host,
                               expected_host_identity=identity)
            credential.assert_not_called()


if __name__ == '__main__':
    unittest.main()
