import os
import stat
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from . import management_session as session


class SessionTests(unittest.TestCase):
    def test_default_and_explicit_private_master_preserve_the_exact_inner_command(self):
        argv = ['ssh', '-T', '-o', 'ProxyCommand=none', 'mgmt-host', 'retained18 exact command']
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(session.apply(argv), argv)
        socket = SimpleNamespace(st_uid=os.getuid(), st_mode=stat.S_IFSOCK | 0o600)
        parent = SimpleNamespace(st_uid=os.getuid(), st_mode=stat.S_IFDIR | 0o700)
        with patch.dict(os.environ, Q7B_MGMT_CONTROL_PATH='/tmp/q7b-mgmt-source-70188b8e50fd/socket'), \
             patch('pathlib.Path.is_symlink', return_value=False), \
             patch('pathlib.Path.stat', side_effect=[parent, socket]), \
             patch.object(session.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(session.apply(argv), ['ssh', '-S', '/tmp/q7b-mgmt-source-70188b8e50fd/socket', *argv[1:]])

    def test_wrong_task_or_dead_master_cannot_silently_open_a_new_mutation_connection(self):
        argv = ['ssh', 'mgmt-host', 'exact remote command']
        with patch.dict(os.environ, Q7B_MGMT_CONTROL_PATH='/tmp/some-other-session/socket'):
            with self.assertRaises(ValueError):session.apply(argv)
        socket = SimpleNamespace(st_uid=os.getuid(), st_mode=stat.S_IFSOCK | 0o600)
        parent = SimpleNamespace(st_uid=os.getuid(), st_mode=stat.S_IFDIR | 0o700)
        with patch.dict(os.environ, Q7B_MGMT_CONTROL_PATH='/tmp/q7b-mgmt-source-70188b8e50fd/socket'), \
             patch('pathlib.Path.is_symlink', return_value=False), \
             patch('pathlib.Path.stat', side_effect=[parent, socket]), \
             patch.object(session.subprocess, 'run', return_value=SimpleNamespace(returncode=255)):
            with self.assertRaises(RuntimeError):session.apply(argv)
