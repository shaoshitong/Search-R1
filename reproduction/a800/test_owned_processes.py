"""PID-safe cleanup including Python builds without native pidfd wrappers."""
import errno
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.a800 import owned_processes as owned


class PidfdTests(unittest.TestCase):
    def test_native_wrappers_preferred(self):
        with patch.object(owned.os,'pidfd_open',create=True,return_value=83) as opening, \
             patch.object(owned.signal,'pidfd_send_signal',create=True) as sending, \
             patch.object(owned,'_linux_pidfd_syscall',side_effect=AssertionError('Unexpected fallback')):
            self.assertEqual(owned._pidfd_open(123),83)
            owned._pidfd_send_signal(83,signal.SIGTERM)
        opening.assert_called_once_with(123,0)
        sending.assert_called_once_with(83,signal.SIGTERM,None,0)

    def test_missing_wrappers_use_correct_syscall_arguments(self):
        with patch.object(owned.os,'pidfd_open',None,create=True), \
             patch.object(owned.signal,'pidfd_send_signal',None,create=True), \
             patch.object(owned,'_linux_pidfd_syscall',side_effect=[83,0]) as syscall:
            self.assertEqual(owned._pidfd_open(123),83)
            self.assertEqual(owned._pidfd_send_signal(83,signal.SIGTERM),0)
        opening,sending=syscall.call_args_list
        self.assertEqual(opening.args[0],434)
        self.assertEqual([item.value for item in opening.args[1:]],[123,0])
        self.assertEqual(sending.args[0],424)
        self.assertEqual([item.value for item in sending.args[1:]],[83,signal.SIGTERM,None,0])

    def test_unsupported_platform_or_abi_fails_before_syscall(self):
        for system,machine in [('Darwin','x86_64'),('Linux','arm64'),('Linux','riscv64')]:
            with self.subTest(system=system,machine=machine), \
                 patch.object(owned.platform,'system',return_value=system), \
                 patch.object(owned.platform,'machine',return_value=machine), \
                 patch.object(owned.ctypes,'CDLL',side_effect=AssertionError('Unsafe syscall')):
                with self.assertRaises(OSError) as error:
                    owned._linux_pidfd_syscall(434)
                self.assertEqual(error.exception.errno,errno.ENOSYS)

    def test_kernel_failure_preserves_errno(self):
        with patch.object(owned.platform,'system',return_value='Linux'), \
             patch.object(owned.platform,'machine',return_value='x86_64'), \
             patch.object(owned.ctypes,'CDLL') as library, \
             patch.object(owned.ctypes,'get_errno',return_value=errno.ENOSYS):
            library.return_value.syscall.return_value=-1
            with self.assertRaises(OSError) as error:
                owned._linux_pidfd_syscall(434)
        self.assertEqual(error.exception.errno,errno.ENOSYS)

    def test_identity_checked_after_open_and_fd_closed_on_refusal(self):
        events=[]
        record=dict(pid=123,start_ticks=456,sid=123,uid=1000)
        with patch.object(owned,'_pidfd_open',side_effect=lambda pid:events.append(('open',pid)) or 83), \
             patch.object(owned,'same',side_effect=lambda value:events.append(('identity',value)) or False), \
             patch.object(owned,'_pidfd_send_signal',side_effect=lambda *args:events.append(('signal',args))), \
             patch.object(owned.os,'close',side_effect=lambda fd:events.append(('close',fd))):
            with self.assertRaises(AssertionError):owned.signal_owned(record,signal.SIGTERM)
        self.assertEqual(events,[('open',123),('identity',record),('close',83)])

    def test_every_process_birth_identity_field_is_checked(self):
        record=dict(pid=123,start_ticks=456,sid=123,uid=1000)
        with patch.object(owned,'identity',return_value=record):
            self.assertTrue(owned.same(record))
        for key in record:
            with self.subTest(key=key), \
                 patch.object(owned,'identity',return_value=dict(record,**{key:record[key]+1})):
                self.assertFalse(owned.same(record))

    def test_signal_failure_closes_open_fd(self):
        with patch.object(owned,'_pidfd_open',return_value=83), \
             patch.object(owned,'same',return_value=True), \
             patch.object(owned,'_pidfd_send_signal',side_effect=ProcessLookupError(errno.ESRCH,'Gone')), \
             patch.object(owned.os,'close') as closing:
            with self.assertRaises(ProcessLookupError):owned.signal_owned(dict(pid=123),signal.SIGTERM)
        closing.assert_called_once_with(83)

    def test_unavailable_pidfd_without_direct_child_handle_fails_closed(self):
        with patch.object(owned,'_signal_with_pidfd',side_effect=OSError(errno.ENOSYS,'No pidfds')), \
             patch.object(owned.os,'kill',side_effect=AssertionError('Ledger PID cannot be signaled')):
            with self.assertRaises(OSError) as error:
                owned.signal_owned(dict(pid=123),signal.SIGTERM)
        self.assertEqual(error.exception.errno,errno.ENOSYS)

    def test_cleanup_accepts_leader_exiting_just_before_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=owned.OwnedProcesses(Path(directory)/'ledger.json','own-exit-race')
            record=dict(pid=123,start_ticks=456,sid=123,uid=os.getuid(),state='S')
            owner.jobs['child']=dict(leader=record,state='STARTED')
            with patch.object(owner,'observe',return_value=[]), \
                 patch.object(owned,'same',return_value=True), \
                 patch.object(owned,'identity',return_value=record), \
                 patch.object(owned,'signal_owned',side_effect=ProcessLookupError(errno.ESRCH,'Exited')):
                owner.stop('child')
            self.assertEqual(owner.jobs['child']['state'],'STOPPED')

    @unittest.skipUnless(platform.system()=='Linux' and platform.machine() in ('x86_64','aarch64'),
                         'Real pidfd fallback requires a supported Linux ABI')
    def test_actual_owned_child_fallback_and_stale_identity(self):
        # Check kernel support before spawning a fixture. A Python wrapper being
        # absent is different from a kernel being unable to provide pidfds.
        with patch.object(owned.os,'pidfd_open',None,create=True), \
             patch.object(owned.signal,'pidfd_send_signal',None,create=True):
            fd=None
            try:
                fd=owned._pidfd_open(os.getpid())
                owned._pidfd_send_signal(fd,0)
            except OSError as error:
                if error.errno==errno.ENOSYS:
                    self.skipTest('Running kernel does not implement pidfds; fail-closed path tested separately')
                raise
            finally:
                if fd is not None:os.close(fd)
        child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.buffer.read()'],
                               stdin=subprocess.PIPE,start_new_session=True)
        record=owned.identity(child.pid)
        try:
            with patch.object(owned.os,'pidfd_open',None,create=True), \
                 patch.object(owned.signal,'pidfd_send_signal',None,create=True):
                # A real pidfd exists, but the forged birth identity cannot be
                # signaled. This proves refusal leaves the same child alive.
                stale=dict(record,start_ticks=record['start_ticks']+1)
                with self.assertRaises(AssertionError):owned.signal_owned(stale,signal.SIGTERM)
                self.assertIsNone(child.poll())
                owned.signal_owned(record,0)
                self.assertIsNone(child.poll())
                owned.signal_owned(record,signal.SIGTERM)
                self.assertEqual(child.wait(timeout=5),-signal.SIGTERM)
        finally:
            # Closing the child's unique pipe lets it exit cooperatively even
            # if a signaling test fails. Fixture cleanup never needs PID kill.
            child.stdin.close()
            child.wait(timeout=5)


@unittest.skipUnless(platform.system()=='Linux' and platform.machine() in ('x86_64','aarch64'),
                     'Direct-child fallback requires supported Linux glibc')
class DirectChildFallbackTests(unittest.TestCase):
    def setUp(self):
        owned._require_waitable_children()
        self.child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.buffer.read()'],
                                    stdin=subprocess.PIPE,start_new_session=True)
        self.record=owned.identity(self.child.pid)
        self.fallback=patch.object(owned,'_signal_with_pidfd',side_effect=OSError(errno.ENOSYS,'No pidfds'))
        self.fallback.start()

    def tearDown(self):
        self.fallback.stop()
        self.child.stdin.close()
        self.child.wait(timeout=5)

    def test_actual_child_stale_birth_rejected_then_fd_free_signal_succeeds(self):
        with self.assertRaises(AssertionError):
            owned.signal_owned(dict(self.record,start_ticks=self.record['start_ticks']+1),
                               signal.SIGTERM,process=self.child)
        self.assertIsNone(self.child.poll())
        owned.signal_owned(self.record,0,process=self.child)
        self.assertIsNone(self.child.poll())
        owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
        self.assertEqual(self.child.wait(timeout=5),-signal.SIGTERM)

    def test_no_signal_after_child_has_been_reaped(self):
        self.child.stdin.close();self.child.wait(timeout=5)
        with patch.object(owned.os,'kill') as killing:
            with self.assertRaises(ProcessLookupError):
                owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
        killing.assert_not_called()

    def test_exited_unreaped_child_is_reaped_under_lock_without_signal(self):
        self.child.stdin.close();deadline=time.monotonic()+5
        while owned.identity(self.child.pid)['state']!='Z' and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertEqual(owned.identity(self.child.pid)['state'],'Z')
        with patch.object(owned.os,'kill') as killing:
            with self.assertRaises(ProcessLookupError):
                owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
        killing.assert_not_called();self.assertEqual(self.child.returncode,0)

    def test_busy_wait_lock_cannot_race_other_reaper(self):
        self.child._waitpid_lock.acquire()
        try:
            with patch.object(owned.os,'kill') as killing:
                with self.assertRaises(RuntimeError):
                    owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
                killing.assert_not_called()
        finally:self.child._waitpid_lock.release()

    def test_wrong_parent_identity_or_nonleader_rejected_without_signal(self):
        wrong_parent=dict(self.record,ppid=os.getpid()+1)
        with patch.object(owned,'identity',return_value=wrong_parent), \
             patch.object(owned.os,'kill') as killing:
            with self.assertRaises(AssertionError):
                owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
            killing.assert_not_called()
        with patch.object(owned.os,'kill') as killing:
            with self.assertRaises(RuntimeError):
                owned.signal_owned(dict(self.record,sid=self.record['sid']+1),
                                   signal.SIGTERM,process=self.child)
            killing.assert_not_called()

    def test_custom_sigchld_or_kernel_auto_reaping_refused(self):
        with patch.object(owned.signal,'getsignal',return_value=signal.SIG_IGN), \
             patch.object(owned.os,'kill') as killing:
            with self.assertRaises(RuntimeError):
                owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
            killing.assert_not_called()
        original_cdll=owned.ctypes.CDLL
        real=original_cdll(None,use_errno=True)
        def unsafe_action(sig,new,old):
            old._obj.flags=2
            return 0
        class AutoReapingLibrary:
            gnu_get_libc_version=real.gnu_get_libc_version
            sigaction=staticmethod(unsafe_action)
        with patch.object(owned.ctypes,'CDLL',return_value=AutoReapingLibrary()), \
             patch.object(owned.os,'kill') as killing:
            with self.assertRaises(RuntimeError):
                owned.signal_owned(self.record,signal.SIGTERM,process=self.child)
            killing.assert_not_called()

    def test_owned_process_stop_retains_authority_but_recovered_ledger_does_not(self):
        with tempfile.TemporaryDirectory(prefix='owned-child-test-') as directory:
            root=Path(directory);ledger=root/'ledger.json';log=root/'child.log'
            owner=owned.OwnedProcesses(ledger,'own-child-test')
            code="import signal,sys,time; signal.signal(signal.SIGINT,lambda *_:sys.exit(0)); print('ready',flush=True); time.sleep(30)"
            child=owner.start('child',[sys.executable,'-c',code],dict(os.environ),log,root)
            try:
                deadline=time.monotonic()+5
                while 'ready' not in log.read_text() and time.monotonic()<deadline:time.sleep(.01)
                self.assertIn('ready',log.read_text())
                self.assertIs(owner.handles['child'],child)
                recovered=owned.OwnedProcesses(ledger,'own-child-test')
                self.assertEqual(recovered.handles,{})
                with self.assertRaises(OSError) as error:recovered.stop('child')
                self.assertEqual(error.exception.errno,errno.ENOSYS)
                self.assertIsNone(child.poll())
                owner.stop('child')
                self.assertEqual(owner.jobs['child']['state'],'STOPPED')
                self.assertEqual(child.wait(timeout=5),0)
            finally:
                if child.poll() is None:
                    owned.signal_owned(owner.jobs['child']['leader'],signal.SIGTERM,process=child)
                child.wait(timeout=5)


if __name__=='__main__':unittest.main()
