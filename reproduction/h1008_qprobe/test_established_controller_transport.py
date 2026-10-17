"""Small local fixtures: no network, subprocesses or controller execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import established_controller_transport as transport
from .management_transport import connection_options, remote, local_connection_path


class EstablishedControllerTests(unittest.TestCase):
    def connection(self):
        return dict(transport_mode=transport.MODE, controller=str(transport.CONTROLLER),
            controller_sha256=transport.CONTROLLER_SHA256, control_path=None, borrowed_socket=False)

    def test_explicit_mode_preserves_legacy_and_rejects_mixed_or_foreign_route(self):
        self.assertEqual(connection_options(self.connection()), {'established_controller':True})
        self.assertEqual(connection_options({'control_path':'owned-socket','compact_handshake':True}),
            {'control_path':'owned-socket','compact_handshake':True})
        self.assertEqual(connection_options({'transport_mode':'own_ordinary_alias_foreground_master','control_path':'owned-socket'}),
            {'control_path':'owned-socket','compact_handshake':False})
        for key,value in (('control_path','socket'),('controller','/other/control.py'),
                          ('controller_sha256','wrong'),('borrowed_socket',True)):
            wrong=self.connection();wrong[key]=value
            with self.assertRaises(ValueError):connection_options(wrong)
        with self.assertRaises(ValueError): connection_options({'transport_mode':'unknown'})
        with self.assertRaises(ValueError):remote('', 'unused', target='h100-10', established_controller=True, control_path='socket')
        with self.assertRaises(ValueError):transport.remote('', 'unused', target='h100-8')

    def test_routes_exact_stdin_once_and_existing_evidence_never_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence=Path(directory)/'receipt.json'
            route=Mock(return_value=['established','route'])
            worker=SimpleNamespace(returncode=0,stdout=b'{"result":"ok"}',stderr=b'')
            with patch.object(transport,'_controller',return_value=SimpleNamespace(route=route)), \
                    patch.object(transport.subprocess,'run',return_value=worker) as run:
                self.assertEqual(remote('original script',evidence,target='h100-10',established_controller=True),'{"result":"ok"}')
                route.assert_called_once_with('h100-10','/usr/bin/python3 -')
                run.assert_called_once_with(['established','route'],input=b'original script',capture_output=True,timeout=240)
                with self.assertRaises(FileExistsError):remote('original script',evidence,target='h100-10',established_controller=True)
                self.assertEqual(run.call_count,1)
            receipt=json.loads(evidence.read_text())
            self.assertEqual(receipt['controller_sha256'],transport.CONTROLLER_SHA256)
            self.assertIsNone(receipt['control_path'])

    def test_timeout_writes_failure_without_retry_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence=Path(directory)/'receipt.json'
            error=transport.subprocess.TimeoutExpired('fixture',240,output=b'partial',stderr=b'bounded')
            with patch.object(transport,'_controller',return_value=SimpleNamespace(route=lambda *args:['route'])), \
                    patch.object(transport.subprocess,'run',side_effect=error) as run:
                with self.assertRaises(RuntimeError):transport.remote('script',evidence)
                self.assertEqual(run.call_count,1)
            receipt=json.loads(evidence.read_text())
            self.assertEqual(receipt['error'],'BOUNDED_ESTABLISHED_ROUTE_TIMEOUT')
            self.assertEqual(receipt['stdout'],'partial')

    def test_changed_controller_is_rejected_before_execution(self):
        with patch.object(Path,'read_bytes',return_value=b'not the pinned controller'):
            with self.assertRaises(ValueError):transport._controller()

    def test_collectors_use_new_canonical_receipt_and_preserve_old_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(local_connection_path(root).name,'MANAGEMENT_DIRECT_MASTER_CURRENT.json')
            current=root/'work/search-g1-prefix1024-diversity512-20261016/ESTABLISHED_CONTROLLER_CURRENT.json'
            current.parent.mkdir(parents=True);current.write_text('{}')
            self.assertEqual(local_connection_path(root),current)


if __name__=='__main__':unittest.main()
