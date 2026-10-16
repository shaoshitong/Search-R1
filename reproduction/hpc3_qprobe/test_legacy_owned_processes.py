"""Real Linux child cleanup, forcing the actual RHEL8 missing-pidfd route."""
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from reproduction.a800 import owned_processes as m


@unittest.skipUnless(Path('/proc/self/stat').exists(),'Actual Linux process identity required')
class LegacyOwnedTests(unittest.TestCase):
    def test_actual_direct_child_cleanup_preserves_foreign_session(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            owner=m.OwnedProcesses(r/'ledger.json','r')
            p=owner.start('owned',[sys.executable,'-c','import time;time.sleep(60)'],dict(os.environ),r/'own.log',r)
            try:
                with patch.object(m,'signal_owned',side_effect=OSError(errno.ENOSYS,'old kernel')):owner.stop_all()
                p.wait(timeout=5);self.assertEqual(owner.jobs['owned']['state'],'STOPPED')
                self.assertIsNone(foreign.poll())
            finally:
                owner.stop_all();p.wait(timeout=5);foreign.terminate();foreign.wait(timeout=5)

    def test_recovered_ledger_cannot_signal_without_direct_child_handle(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);owner=m.OwnedProcesses(r/'ledger.json','r')
            p=owner.start('owned',[sys.executable,'-c','import time;time.sleep(60)'],dict(os.environ),r/'own.log',r)
            restored=m.OwnedProcesses(r/'ledger.json','r')
            try:
                with patch.object(m,'signal_owned',side_effect=OSError(errno.ENOSYS,'old kernel')),patch.object(m.os,'killpg') as kill:
                    with self.assertRaises(RuntimeError):restored.stop_all()
                    kill.assert_not_called()
                self.assertIsNone(p.poll())
            finally:
                with patch.object(m,'signal_owned',side_effect=OSError(errno.ENOSYS,'old kernel')):owner.stop_all()
                p.wait(timeout=5)

    def test_changed_start_ticks_cannot_signal_real_child(self):
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        try:
            record=m.identity(p.pid);record['start_ticks']+=1
            with patch.object(m.os,'killpg') as kill:
                with self.assertRaises(AssertionError):m.signal_owned_child_group(record,p,signal.SIGTERM)
                kill.assert_not_called()
            self.assertIsNone(p.poll())
        finally:p.terminate();p.wait(timeout=5)

    def test_reaped_child_cannot_pin_signal(self):
        p=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        record=m.identity(p.pid);p.wait(timeout=5)
        with patch.object(m.os,'killpg') as kill:
            with self.assertRaises(AssertionError):m.signal_owned_child_group(record,p,signal.SIGTERM)
            kill.assert_not_called()


if __name__=='__main__':unittest.main()
