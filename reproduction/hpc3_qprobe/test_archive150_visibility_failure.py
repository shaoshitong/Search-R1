"""Failure admission attacks and real Linux unique-task retirement lifecycle."""
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import archive150_visibility_failure as recovery


class RefusalTests(unittest.TestCase):
    def test_python36_and_protected_receipts_reject_shared_write_symlinks_and_wrong_owner(self):
        ast.parse(Path(recovery.__file__).read_text(),feature_version=(3,6))
        with tempfile.TemporaryDirectory() as tmp,patch.object(recovery.storage,'GID',os.getgid()):
            path=Path(tmp).resolve()/'receipt.json';recovery.protected_publish(path,dict(actual=True))
            self.assertEqual(path.stat().st_mode&0o777,0o640)
            self.assertEqual(recovery.protected_document(path,os.getuid())[0],dict(actual=True))
            with self.assertRaises(ValueError):recovery.protected_document(path,os.getuid()+1)
            path.chmod(0o660)
            with self.assertRaises(ValueError):recovery.protected_document(path,os.getuid())
            path.chmod(0o640);link=path.with_name('link');link.symlink_to(path)
            with self.assertRaises(ValueError):recovery.protected_document(link,os.getuid())
            with self.assertRaises(FileExistsError):recovery.protected_publish(path,dict(actual=False))

    def test_wrong_birth_after_seize_cannot_interrupt_or_signal_a_reused_pid(self):
        identity=Mock(side_effect=[{},ValueError('target birth changed')])
        with patch.object(recovery.sys,'platform','linux'),patch.object(recovery,'target_identity',identity),\
             patch.object(recovery,'ptrace') as ptrace,patch.object(recovery.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'birth changed'):
                recovery.bound_sigterm(dict(recovery.TARGET,uid=os.getuid()),[],Path('/fixture'),lambda:None)
            self.assertEqual(ptrace.call_args_list[0].args,(recovery.SEIZE,recovery.TARGET['pid'],0))
            self.assertEqual(ptrace.call_count,1);kill.assert_not_called()

    def test_admission_change_after_stop_sends_zero_signals(self):
        stop=(128<<16)|(signal.SIGTRAP<<8)|0x7f
        before=Mock(side_effect=ValueError('payload appeared'))
        with patch.object(recovery.sys,'platform','linux'),patch.object(recovery,'target_identity'),\
             patch.object(recovery,'ptrace') as ptrace,patch.object(recovery.os,'kill') as kill,\
             patch.object(recovery.os,'waitpid',return_value=(recovery.TARGET['pid'],stop)):
            with self.assertRaisesRegex(ValueError,'payload appeared'):
                recovery.bound_sigterm(dict(recovery.TARGET,uid=os.getuid()),[],Path('/fixture'),before)
            kill.assert_not_called();self.assertEqual(ptrace.call_count,2)

    def test_terminal_wait_or_other_stop_cannot_be_used_as_signal_target(self):
        for status in (0,signal.SIGKILL,(signal.SIGSTOP<<8)|0x7f):
            with self.subTest(status=status),patch.object(recovery.sys,'platform','linux'),\
                 patch.object(recovery,'target_identity'),patch.object(recovery,'ptrace'),\
                 patch.object(recovery.os,'kill') as kill,patch.object(recovery.os,'waitpid',return_value=(recovery.TARGET['pid'],status)):
                with self.assertRaisesRegex(ValueError,'PTRACE_EVENT_STOP'):
                    recovery.bound_sigterm(dict(recovery.TARGET,uid=os.getuid()),[],Path('/fixture'),lambda:None)
                kill.assert_not_called()

    def test_exact_actual_failed_du_and_zero_payload_required_before_any_signal(self):
        request=dict(archive_child=recovery.TARGET,archive_token='token',excluded_paths=[],source=dict(commit=recovery.OLD))
        proof=dict(path=str(recovery.REQUEST),sha256=recovery.REQUEST_SHA)
        claim=dict(controller=recovery.FOLLOWER,step=150,source=request['source'],token='token')
        runner=dict(actual_process=recovery.RUNNER);child=dict(actual_process=recovery.TARGET,supervisor=recovery.RUNNER)
        du=dict(actual_parent=recovery.TARGET,actual_du_wait_exit_code=1,actual_du=dict(pid=123),
                stderr="du: cannot read directory '"+recovery.NEW_DENIAL+"': Permission denied\n")
        old=SimpleNamespace(request_guard=lambda *a,**kw:(request,proof),consume_du=lambda *a:(du,dict(sha256=recovery.DU_SHA)),
                            du_command=lambda *a:[],MEASURE=Path('/fixture'),live=lambda _:False)
        def document(path,*_):
            if path.name=='submit-claim.json':return claim,dict(sha256=recovery.CLAIM_SHA)
            return (runner if path.name=='runner-binding.json' else child),{}
        with patch.object(recovery,'old_module',return_value=old),patch.object(recovery.storage,'document',side_effect=document),\
             patch.object(recovery,'exists',return_value=False) as exists:
            self.assertFalse(recovery.preflight()['failed_stdout_is_usage'])
            old.live=lambda _:True
            with self.assertRaisesRegex(ValueError,'fully waited'):recovery.preflight()
            old.live=lambda _:False;du['actual_du_wait_exit_code']=0
            with self.assertRaisesRegex(ValueError,'fully waited'):recovery.preflight()
            du['actual_du_wait_exit_code']=1
            for bad in (recovery.PAYLOAD,recovery.REQUEST.parent/'mou-control',recovery.ARCHIVE/'task-result.json',
                        recovery.ARCHIVE/'storage-owner-partition-accepted.json'):
                exists.side_effect=lambda path:Path(path)==bad
                with self.subTest(path=bad),self.assertRaisesRegex(ValueError,'refuse retirement'):recovery.preflight()

    def test_stale_wrong_host_source_nonce_or_claim_report_refuses(self):
        facts=dict(request=dict(sha256='a'*64));source=dict(commit='b'*40);host=dict(boot='actual')
        report=dict(status='UNKNOWN_CROSS_OWNER_PROCESS_VISIBILITY_HIDEPID2',facts=facts,source=source,host=host,
            actual_reporter=dict(uid=recovery.storage.MOU),foreign_process_declared_dead=False,
            signals_sent=0,training_actions=0,permission_changes=0,
            foreign_proc_errors={name:dict(errno=2,kind='FileNotFoundError') for name in ('stat','status')},created_unix=100)
        with patch.object(recovery,'protected_document',return_value=(report,{})),patch.object(recovery,'host',return_value=host),\
             patch.object(recovery.time,'time',return_value=101):
            recovery.visibility_report(facts,source)
            for key,bad in (('host',{}),('source',dict(commit='c'*40)),('facts',{}),('created_unix',-100)):
                original=report[key];report[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):recovery.visibility_report(facts,source)
                report[key]=original


RUNNER_CODE=r'''import json,os,subprocess,sys
from pathlib import Path
from reproduction.hpc3_qprobe import owner_partition_storage as s
command=[sys.executable,'-c','import time;time.sleep(120)']
child=subprocess.Popen(command)
print(json.dumps(dict(target=s.birth(s.process(child.pid)),command=command,cwd=str(Path.cwd()))),flush=True)
sys.stdin.buffer.read()
code=child.wait()
Path(sys.argv[1]).write_text(json.dumps(dict(actual_parent=s.birth(s.process(os.getpid())),actual_child_pid=child.pid,actual_wait=code)))
'''


@unittest.skipUnless(sys.platform=='linux' and Path('/proc/self/stat').exists(),'real Linux ptrace lifecycle required')
class ActualLinuxRetirement(unittest.TestCase):
    def exercise(self,mode):
        with tempfile.TemporaryDirectory() as tmp:
            result=Path(tmp)/'actual-parent-held.json'
            runner=subprocess.Popen([sys.executable,'-c',RUNNER_CODE,str(result)],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            try:
                info=json.loads(runner.stdout.readline());target=info['target']
                script='''import json,sys\nfrom pathlib import Path\nfrom reproduction.hpc3_qprobe import archive150_visibility_failure as m\nv=json.loads(sys.argv[1]);mode=sys.argv[2]\noriginal=m.target_identity;calls=[];original_ptrace=m.ptrace\ndef trace(*args):\n calls.append(args);return original_ptrace(*args)\nm.ptrace=trace\ndef identity(*args,**kwargs):\n value=original(*args,**kwargs)\n if mode=="wrong-after-seize" and kwargs["tracer"]:raise ValueError("simulated changed birth after SEIZE")\n return value\nm.target_identity=identity\ndef before():\n if mode=="refuse-after-stop":raise ValueError("actual stopped task admission refused")\ntry:\n r=m.bound_sigterm(v["target"],v["command"],Path(v["cwd"]),before)\n print(json.dumps(dict(result=r,calls=calls)),flush=True)\nexcept Exception as e:\n print(json.dumps(dict(error=repr(e),calls=calls)),flush=True);sys.exit(1)\n'''
                traced=subprocess.run([sys.executable,'-c',script,json.dumps(info),mode],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
                self.assertEqual(traced.returncode,0 if mode=='success' else 1,traced.stderr)
                trace=json.loads(traced.stdout)
                if mode=='success':
                    self.assertEqual(trace['result']['signals_queued'],1)
                    self.assertEqual(trace['result']['process_group_signals'],0)
                else:
                    # Exiting the dedicated tracer auto-detaches, including an
                    # exception after actual PTRACE_INTERRUPT/ptrace-stop.
                    deadline=time.monotonic()+5
                    while True:
                        now=recovery.storage.process(target['pid'])
                        status=(Path('/proc')/str(target['pid'])/'status').read_text()
                        if now['state'] not in ('t','T') and '\nTracerPid:\t0\n' in status:break
                        self.assertLess(time.monotonic(),deadline);time.sleep(.01)
                    self.assertEqual(recovery.storage.birth(now),target)
                    if mode=='wrong-after-seize':self.assertEqual(len(trace['calls']),1)
                    # Test runner is the genuine parent, and terminates only
                    # its original still-owned subprocess during cleanup.
                    os.kill(target['pid'],signal.SIGTERM)
                runner.stdin.close();runner.wait(timeout=10)
                held=json.loads(result.read_text());self.assertEqual(held['actual_wait'],-signal.SIGTERM)
                self.assertEqual(held['actual_child_pid'],target['pid'])
                self.assertEqual(held['actual_parent']['pid'],target['sid'])
                self.assertFalse(recovery.storage.live(target))
            finally:
                if runner.poll() is None:
                    runner.stdin.close();runner.wait(timeout=130)
                runner.stdout.close();runner.stderr.close()

    def test_exact_bound_task_terminates_and_original_parent_really_waits_minus15(self):self.exercise('success')
    def test_wrong_birth_after_seize_exits_without_interrupt_and_auto_detaches(self):self.exercise('wrong-after-seize')
    def test_actual_ptrace_stop_exception_auto_detaches_and_resumes_original_task(self):self.exercise('refuse-after-stop')


if __name__=='__main__':unittest.main()
