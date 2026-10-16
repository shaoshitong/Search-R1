import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from reproduction.a800 import owned_processes as m
from reproduction.a800 import run_pipeline as pipeline

@unittest.skipUnless(Path('/proc/self/stat').exists() and hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),
                     'Modern Linux pidfd tests; the separate legacy suite tests missing-pidfd execution')
class OwnedProcessTests(unittest.TestCase):
    def test_actual_owned_cleanup_preserves_foreign_session(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
            owned=m.OwnedProcesses(root/'ledger.json','r')
            try:
                process=owned.start('owned',[sys.executable,'-c','import time;time.sleep(120)'],dict(os.environ),root/'job.log',root)
                recovered=m.OwnedProcesses(root/'ledger.json','r')
                self.assertTrue(recovered.recovery_only)
                with self.assertRaises(AssertionError):recovered.start('duplicate',[],{},root/'unused',root)
                recovered.stop_all();process.wait(timeout=5)
                self.assertIsNone(foreign.poll());self.assertEqual(recovered.jobs['owned']['state'],'STOPPED')
            finally:
                owned.stop_all();foreign.terminate();foreign.wait(timeout=5)
    def test_fast_exit_child_is_registered_before_leader_reap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);owned=m.OwnedProcesses(root/'ledger.json','r')
            code='import os,time;child=os.fork();time.sleep(.02) if child else time.sleep(120);os._exit(0)'
            process=owned.start('forked',[sys.executable,'-c',code],dict(os.environ),root/'fork.log',root)
            try:
                import time;time.sleep(.1)
                live=owned.observe('forked');self.assertTrue(live)
                process.poll();owned.stop_all()
                self.assertFalse(owned.observe('forked'))
            finally:owned.stop_all();process.wait(timeout=5)
    def test_pid_reuse_cannot_signal(self):
        r=dict(pid=1,start_ticks=2,sid=1,uid=0,command='old')
        with patch.object(m.os,'pidfd_open',return_value=8),patch.object(m.os,'close'),patch.object(m,'identity',return_value=dict(r,start_ticks=9)),patch.object(m.signal,'pidfd_send_signal') as send:
            with self.assertRaises(AssertionError):m.signal_owned(r,signal.SIGTERM)
            send.assert_not_called()
    def test_ray_title_change_is_not_pid_reuse(self):
        r=dict(pid=1,start_ticks=2,sid=1,uid=0,command='ray::IDLE')
        with patch.object(m,'identity',return_value=dict(r,command='ray::main_task')):self.assertTrue(m.same(r))

    def test_sigterm_during_spawn_defers_until_child_is_registered(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();owned=m.OwnedProcesses(root/'ledger.json','r')
            p=object.__new__(pipeline.Pipeline);p.base=root;p.repo=root;p.env=dict(os.environ);p.owned=owned;p.sample_error=None
            original=m.subprocess.Popen;children=[];old_handler=signal.signal(signal.SIGTERM,pipeline.request_interrupt)
            def interrupted_spawn(*args,**kwargs):
                child=original(*args,**kwargs);children.append(child);os.kill(os.getpid(),signal.SIGTERM);return child
            try:
                pipeline._interrupt_requested=None
                with patch.object(m.subprocess,'Popen',side_effect=interrupted_spawn):
                    with self.assertRaises(AssertionError):p.start('interrupted',[sys.executable,'-c','import time;time.sleep(120)'],[])
                self.assertIn('interrupted',json.loads((root/'ledger.json').read_text())['jobs'])
                owned.stop_all();children[0].wait(timeout=5)
            finally:
                signal.signal(signal.SIGTERM,old_handler);pipeline._interrupt_requested=None;owned.stop_all()
                for child in children:child.wait(timeout=5)

if __name__=='__main__':unittest.main()
