"""Once-only launch and actual output/exit evidence, without external work."""
import copy
from contextlib import ExitStack, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from reproduction.hpc3_qprobe import independent_goal_control719225 as control


class IndependentFinalControlTests(unittest.TestCase):
    def test_missing200_creates_no_scope_claim_or_process(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()/'absent-on-purpose'
            with patch.object(control,'DIRECTORY',directory),patch.object(control,'caps'),\
                    patch.object(control.reader,'early_history',side_effect=ValueError('missing actual200')),\
                    patch.object(control.subprocess,'Popen') as launch,\
                    patch.object(control.ctl,'source') as source,patch.object(control.ctl,'publish') as publish:
                with self.assertRaisesRegex(ValueError,'missing actual200'):control.start('a'*40)
                self.assertFalse(directory.exists());launch.assert_not_called();source.assert_not_called();publish.assert_not_called()

    def test_existing_unknown_scope_is_inspected_never_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()
            with patch.object(control,'DIRECTORY',directory),patch.object(control,'caps'),\
                    patch.object(control,'inspect',return_value={'status':'UNKNOWN_FINAL_CONSUMER_NO_REPLAY'}) as inspect,\
                    patch.object(control,'ready') as ready,patch.object(control.subprocess,'Popen') as launch:
                self.assertEqual(control.start('a'*40)['status'],'UNKNOWN_FINAL_CONSUMER_NO_REPLAY')
                inspect.assert_called_once_with('a'*40);ready.assert_not_called();launch.assert_not_called()

    def fixture(self,directory):
        commit='a'*40;source=dict(commit=commit,publication_sha256='b'*64)
        process=dict(pid=100,start_ticks=9,sid=100,uid=14090)
        child=dict(pid=102,start_ticks=10,sid=100,uid=14090)
        claim=dict(source=source,created_unix=1)
        proof=dict(status=control.reader.STATUS,run_id=control.reader.RUN,source=source,
            completed_outer=200,world_size=4,actual_native_two_readbacks_reconsumed=True,
            update201_performed=False,goal_complete=False,actor_model_loaded=False,
            receipt_or_cache_or_claim_writes=0,training_or_helper_or_native_job_submissions=0,
            generated_trajectories=0,new_actor_head_or_scheduler_updates=0)
        for name,value in [('claim',claim),('process',process),('child',child)]:
            (directory/(name+'.json')).write_text(json.dumps(value))
        (directory/'consumer.stdout').write_text(json.dumps(proof));(directory/'consumer.stderr').write_bytes(b'')
        outputs={name:dict(path=str(directory/('consumer.'+name)),
            bytes=(directory/('consumer.'+name)).stat().st_size,
            sha256=control.ctl.digest(directory/('consumer.'+name))) for name in ('stdout','stderr')}
        result=dict(source=source,claim_sha256=control.ctl.digest(directory/'claim.json'),process=process,child=child,
            started_unix=2,completed_unix=3,actual_child_exit_code=0,supervisor_exit_code=0,error=None,**outputs)
        (directory/'result.json').write_text(json.dumps(result))
        return commit,result

    def test_real_output_bytes_exit_and_child_terminal_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,result=self.fixture(directory)
            reader=lambda p:json.loads(Path(p).read_text())
            with patch.object(control,'DIRECTORY',directory),patch.object(control.ctl,'read',side_effect=reader),\
                    patch.object(control.ctl,'source',return_value=(directory,result['source'])),\
                    patch.object(control.ctl,'live',return_value=False) as live:
                accepted=control.consume(commit)
                self.assertEqual(accepted['actual_child_exit_code'],0)
                self.assertFalse(accepted['goal_complete'])
                bad=copy.deepcopy(result);bad['actual_child_exit_code']=1
                (directory/'result.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'held final consumer wait'):control.consume(commit)
                (directory/'result.json').write_text(json.dumps(result))
                live.return_value=True
                with self.assertRaisesRegex(ValueError,'held final consumer wait'):control.consume(commit)
                live.return_value=False
                # The JSON still says PASS; changed physical stdout must fail.
                p=directory/'consumer.stdout';p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'output changed'):control.consume(commit)

    def test_summary_pass_with_new_work_or_logical201_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,_=self.fixture(directory)
            proof=json.loads((directory/'consumer.stdout').read_text())
            for key,value in [('update201_performed',True),('actor_model_loaded',True),
                    ('training_or_helper_or_native_job_submissions',1),('new_actor_head_or_scheduler_updates',1)]:
                bad=copy.deepcopy(proof);bad[key]=value
                with self.assertRaisesRegex(ValueError,'stdout is not the full pure200'):control.accept_output(bad,commit)

    def test_missing_or_null_kernel_birth_does_not_count_as_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,result=self.fixture(directory)
            reader=lambda p:json.loads(Path(p).read_text())
            with patch.object(control,'DIRECTORY',directory),patch.object(control.ctl,'read',side_effect=reader),\
                    patch.object(control.ctl,'source',return_value=(directory,result['source'])),\
                    patch.object(control.ctl,'live',return_value=False):
                for target in ('process','child'):
                    for missing in (False,True):
                        bad=copy.deepcopy(result)
                        if missing:bad[target].pop('start_ticks')
                        else:bad[target]['start_ticks']=None
                        (directory/(target+'.json')).write_text(json.dumps(bad[target]))
                        (directory/'result.json').write_text(json.dumps(bad))
                        with self.assertRaisesRegex(ValueError,'held final consumer wait'):control.consume(commit)
                        (directory/(target+'.json')).write_text(json.dumps(result[target]))


class SupervisorExecutionTests(unittest.TestCase):
    """Synthetic protocol fixtures with real local child output and wait().

    Kernel/source/recovery admission is simulated for portability. These tests
    prove holder execution and failure bookkeeping, never remote200 success.
    """
    def execute(self, directory, *, failure=None, recovered=False):
        commit='a'*40; recovery='b'*40; followthrough='c'*40
        source=dict(commit=commit,publication_sha256='d'*64)
        own=dict(pid=os.getpid(),start_ticks=100,sid=os.getpid(),uid=os.getuid())
        route={} if not recovered else dict(recovery_commit=recovery,
            recovery_followthrough_commit=followthrough)
        proof=dict(status=control.reader.STATUS,run_id=control.reader.RUN,source=source,
            completed_outer=200,world_size=4,actual_native_two_readbacks_reconsumed=True,
            update201_performed=False,goal_complete=False,actor_model_loaded=False,
            receipt_or_cache_or_claim_writes=0,training_or_helper_or_native_job_submissions=0,
            generated_trajectories=0,new_actor_head_or_scheduler_updates=0)
        admission={}
        if recovered:
            binding=dict(producer_final_cpu_recovery=dict(commit=recovery),
                followthrough_source=dict(commit=followthrough),
                followthrough_result_sha256='1'*64,goal_sha256='2'*64,
                consumer_binding_sha256='3'*64,raw_result_sha256='4'*64,
                raw_claim_sha256='5'*64)
            admission['recovery_binding']=binding
            proof.update(route,producer_final_cpu_recovery=binding['producer_final_cpu_recovery'],
                original_producer_accounting_remains_failed=True,original_EA_status_remains_failed=True,
                producer_accounting=dict(state='FAILED',exit_code='1:0'),
                actual_recovery_full_CPU_consumer=dict(source=binding['followthrough_source'],
                    result=dict(sha256='1'*64),goal=dict(sha256='2'*64),binding=dict(sha256='3'*64)),
                independent_all_normal_return_raw=dict(result=dict(sha256='4'*64),claim=dict(sha256='5'*64)))
        claim=dict(source=source,cwd=str(directory),created_unix=0,admission=admission,**route)
        (directory/'claim.json').write_text(json.dumps(claim))
        (directory/'process.json').write_text(json.dumps(own))
        if failure=='invalid_output':proof['update201_performed']=True
        stdout='not JSON' if failure=='malformed_output' else json.dumps(proof)
        stderr='measured child diagnostic' if failure=='stderr' else ''
        child_exit=7 if failure=='child_nonzero' else 0
        # The failed binding branch must still wait while its child is alive.
        delay=.05 if failure=='child_binding' else 0
        program=('import sys,time;time.sleep('+repr(delay)+');'
            'sys.stdout.write('+repr(stdout)+');sys.stdout.flush();'
            'sys.stderr.write('+repr(stderr)+');sys.stderr.flush();sys.exit('+str(child_exit)+')')
        actual_popen=subprocess.Popen; launched=[]
        expected=[sys.executable,'-m',control.CONSUMER,'--commit',commit]
        if recovered:expected+=['--recovery-commit',recovery,'--recovery-followthrough-commit',followthrough]
        def launch(command, **kwargs):
            self.assertEqual(command,expected)
            self.assertEqual(kwargs['cwd'],str(directory))
            self.assertEqual(set(kwargs),{'cwd','stdout','stderr'})
            child=actual_popen([sys.executable,'-c',program],**kwargs)
            child.wait=Mock(wraps=child.wait)
            launched.append(child)
            return child
        def identity(pid):
            if pid==own['pid']:return own
            self.assertEqual(pid,launched[0].pid)
            return dict(pid=pid,start_ticks=101,sid=own['sid']+(failure=='child_binding'),uid=own['uid'])
        preserved=Mock(side_effect=[None,ValueError('terminal recovery bytes changed')]
            if failure=='terminal_recovery' else None)
        with ExitStack() as stack:
            for target,name,value in ((control,'DIRECTORY',directory),(control,'caps',Mock()),
                    (control.ctl,'OWNER',os.getuid()),(control.ctl,'source',Mock(return_value=(directory,source))),
                    (control.ctl,'process',Mock(side_effect=identity)),
                    (control.ctl,'live',Mock(side_effect=lambda row:launched[0].poll() is None)),
                    (control,'preserved_recovery',preserved),(control.subprocess,'Popen',Mock(side_effect=launch))):
                stack.enter_context(patch.object(target,name,value))
            stack.enter_context(redirect_stderr(io.StringIO()))
            code=control.supervise(commit,**route)
        self.assertEqual(len(launched),1)
        child=launched[0];child.wait.assert_called_once_with()
        self.assertEqual(child.poll(),child_exit)
        result=json.loads((directory/'result.json').read_text())
        self.assertEqual(result['actual_child_exit_code'],child_exit)
        self.assertEqual(result['supervisor_exit_code'],code)
        self.assertEqual(result['source'],source)
        self.assertEqual(result['claim_sha256'],control.ctl.digest(directory/'claim.json'))
        self.assertEqual(result['process'],own)
        self.assertEqual(result['child']['pid'],child.pid)
        self.assertFalse(result['goal_complete'])
        for name,content in [('stdout',stdout),('stderr',stderr)]:
            path=directory/('consumer.'+name)
            self.assertEqual(path.read_text(),content)
            self.assertEqual(result[name],dict(path=str(path),bytes=path.stat().st_size,
                sha256=control.ctl.digest(path)))
        return code,result,preserved

    def test_actual_local_child_return0_normal_and_explicit_recovery_routes(self):
        for recovered in (False,True):
            with self.subTest(recovered=recovered),tempfile.TemporaryDirectory() as folder:
                code,result,preserved=self.execute(Path(folder).resolve(),recovered=recovered)
                self.assertEqual(code,0);self.assertIsNone(result['error'])
                self.assertEqual(preserved.call_count,2)

    def test_actual_child_nonzero_stderr_and_bad_stdout_publish_failed_results(self):
        cases={'child_nonzero':'did not return0','stderr':'stderr is nonempty',
            'invalid_output':'stdout is not the full pure200','malformed_output':'JSONDecodeError'}
        for failure,message in cases.items():
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as folder:
                code,result,preserved=self.execute(Path(folder).resolve(),failure=failure)
                self.assertEqual(code,1);self.assertIn(message,result['error'])
                self.assertEqual(preserved.call_count,1)

    def test_binding_failure_after_spawn_still_holds_child_until_actual_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            code,result,preserved=self.execute(Path(folder).resolve(),failure='child_binding')
            self.assertEqual(code,1);self.assertEqual(result['actual_child_exit_code'],0)
            self.assertIn('child identity differs',result['error'])
            self.assertEqual(preserved.call_count,1)

    def test_terminal_recovery_change_cannot_turn_child0_into_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            code,result,preserved=self.execute(Path(folder).resolve(),failure='terminal_recovery',recovered=True)
            self.assertEqual(code,1);self.assertEqual(result['actual_child_exit_code'],0)
            self.assertIn('terminal recovery bytes changed',result['error'])
            self.assertEqual(preserved.call_count,2)


if __name__=='__main__':unittest.main()
