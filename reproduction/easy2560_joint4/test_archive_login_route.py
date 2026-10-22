"""CPU-only route/pin checks; no SSH, Slurm or archive operations."""
import io
import json
from pathlib import Path
import shlex
import types
import unittest
from unittest.mock import mock_open, patch

from reproduction.easy2560_joint4 import archive_login_route as route
from reproduction.prefix20_joint4 import archive_supervisor as frozen_default
from reproduction.operations.easy7b_20261021 import archive_post_callback5_hk as helper
from reproduction.operations.easy7b_20261021 import archive_post_callback_future_hk as future
from reproduction.operations.easy7b_20261021 import fa3_archive_consumers_hk as consumers

LOGIN2 = dict(hostname='ACD-Login-2',boot_id='3c637b25-17cf-404c-971b-30bc4c4e9fbb',
              pid_namespace='pid:[4026531836]')


class ExplicitArchiveRouteTests(unittest.TestCase):
    def test_default_is_byte_identical_to_frozen_supervisor(self):
        command='printf "%s\\n" archive-fixture'
        self.assertEqual(route.ssh_argv(command),frozen_default.ssh_argv(command))

    def test_complete_explicit_pin_is_required_and_no_auto_fallback(self):
        bad = [('10.120.48.32',None),('10.120.48.33',json.dumps(LOGIN2)),
               ('-oProxyCommand=bad',json.dumps(LOGIN2)),('10.120.48.32','{}')]
        for field,value in (('hostname','ACD-Login-1'),('boot_id','unknown'),
                            ('pid_namespace','host'),('extra','unexpected')):
            identity=dict(LOGIN2);identity[field]=value
            bad.append(('10.120.48.32',json.dumps(identity)))
        for host,identity in bad:
            with self.subTest(host=host,identity=identity),self.assertRaises(ValueError):
                route.ssh_argv('never-executed',login_host=host,login_identity_json=identity)

    def test_alternate_command_executes_only_on_exact_host_boot_and_namespace(self):
        command='read-only-fixture'
        argv=route.ssh_argv(command,login_host='10.120.48.32',login_identity_json=json.dumps(LOGIN2))
        self.assertEqual(argv[-2],'jxu957@10.120.48.32')
        self.assertIn('-oStrictHostKeyChecking=yes',argv)
        self.assertIn('-oUserKnownHostsFile='+frozen_default.KNOWN_HOSTS,argv)
        code=shlex.split(argv[-1])[3]
        for mutation in ({},{'hostname':'ACD-Login-1'},{'boot_id':'another-boot'},
                         {'pid_namespace':'pid:[1]'}):
            actual=dict(LOGIN2,**mutation)
            with self.subTest(mutation=mutation),patch('socket.gethostname',return_value=actual['hostname']),\
                 patch('os.readlink',return_value=actual['pid_namespace']),\
                 patch('builtins.open',mock_open(read_data=actual['boot_id']+'\n')),\
                 patch('os.execv') as execute,patch('sys.stderr',new_callable=io.StringIO) as error:
                if mutation:
                    with self.assertRaises(SystemExit) as caught:exec(code,{})
                    self.assertEqual(caught.exception.code,86);execute.assert_not_called()
                else:
                    exec(code,{});execute.assert_called_once_with('/bin/bash',['bash','-c',command])

    def test_frozen_installer_route_is_private_and_dependency_installer_inherits_it(self):
        original=helper.remote_install_code;old_code=original()
        private=types.SimpleNamespace(remote_install_code=original,HPC_TOOL=helper.HPC_TOOL,
            MODULE_COMMIT=helper.MODULE_COMMIT,MODULE_SHA=helper.MODULE_SHA,MODULE_BYTES=helper.MODULE_BYTES)
        consumers.bind_private_installer_route(private,route.route_record('10.120.48.32',json.dumps(LOGIN2)))
        data=dict(hpc_tool=Path('/data/user/jxu957/experiments/fixture/dependencies.py'),
                  commit='a'*40,sha256='b'*64,raw=b'fixture')
        for code in (private.remote_install_code(),future.dependency_install_code(private,data)):
            self.assertIn("os.getuid()==14090 and socket.gethostname()=='ACD-Login-2'",code)
            self.assertNotIn("socket.gethostname()=='ACD-Login-1'",code)
            compile(code,'<private-installer>','exec')
        self.assertIs(helper.remote_install_code,original)
        self.assertEqual(helper.remote_install_code(),old_code)
        self.assertEqual(frozen_default.PEER,'jxu957@10.120.48.31')

    def test_private_loader_does_not_import_or_replace_reproduction_packages(self):
        import sys
        before={name:value for name,value in sys.modules.items() if name.startswith('reproduction')}
        root=Path(consumers.__file__).resolve().parents[3]
        private=consumers.route_module(root)
        self.assertEqual(private.ssh_argv('fixture'),route.ssh_argv('fixture'))
        self.assertEqual({name:value for name,value in sys.modules.items() if name.startswith('reproduction')},before)


if __name__=='__main__':unittest.main()
