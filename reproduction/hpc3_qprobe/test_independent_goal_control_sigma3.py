"""Portable protocol fixtures and real local child waits; no remote200 claim."""
from contextlib import ExitStack,redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as control


def documents(path):return json.loads(Path(path).read_bytes())


def output_fixture(commit):
    source=dict(commit=commit,publication_sha256='a'*64)
    follow_source=dict(commit=control.reader.FOLLOWTHROUGH)
    raw_source=dict(commit=control.reader.RAW)
    admission=dict(producer_binding=dict(run_id=control.reader.RUN),followthrough_source=follow_source,
        followthrough_result_sha256='b'*64,goal_sha256='c'*64,followthrough_binding_sha256='d'*64,
        raw_source=raw_source,raw_result_sha256='e'*64,raw_claim_sha256='f'*64)
    proof=dict(status=control.reader.STATUS,run_id=control.reader.RUN,producer_job=control.reader.JOB,
        producer_commit=control.reader.PRODUCER,source=source,followthrough_commit=control.reader.FOLLOWTHROUGH,
        completed_outer=200,world_size=4,current_history=dict(first=1,last=200,fresh_official_actor=True),
        actual_native_two_readbacks_reconsumed=True,update201_performed=False,goal_complete=False,
        actor_model_loaded=False,receipt_or_cache_or_claim_writes=0,training_or_helper_or_native_job_submissions=0,
        generated_trajectories=0,new_actor_head_or_scheduler_updates=0,producer_binding=admission['producer_binding'],
        actual_followthrough_full_CPU_consumer=dict(source=follow_source,result=dict(sha256='b'*64),
            goal=dict(sha256='c'*64),binding=dict(sha256='d'*64)),
        independent_all_normal_return_raw=dict(source=raw_source,result=dict(sha256='e'*64),claim=dict(sha256='f'*64)))
    return source,admission,proof


class ReadinessTests(unittest.TestCase):
    def test_terminal_small_receipts_admit_CPU_only_and_changed_goal_binding_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'control';base=root.parent/'sigma3';native_base=root.parent/'native'
            follow=root/'followthrough'/control.reader.JOB;raw=root/'observations/raw'
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
                return control.ctl.digest(path)
            publications={}
            for commit in (control.reader.FOLLOWTHROUGH,control.reader.RAW):
                path=root/'preparation'/('source-'+commit+'-verification.json')
                digest=write(path,dict(commit=commit,status='PUBLISHED_SOURCE_FULL_SHA_PASS',files={'tiny.py':'a'*64}))
                publications[commit]=dict(commit=commit,publication_path=str(path),publication_sha256=digest)
            process=dict(pid=100,start_ticks=1000,sid=100,uid=os.getuid())
            producer=dict(run_id=control.reader.RUN,producer_job=control.reader.JOB)
            settings_sha=write(base/'preparation/sigma3-experiment.json',dict(settings='actual bound bytes'))
            goal=dict(status=control.reader.GOAL_STATUS,run_id=control.reader.RUN,completed_outer=200,update201_performed=False)
            native=dict(run_id=control.reader.RUN,producer_commit=control.reader.PRODUCER,world_size=4,
                actual_completed_outer=200,actual_cuda_restore_accepted=True,update201_performed=False,fit_called=False,
                audit_id='audit',slurm_job_id='800001',actual_actor_optimizer_write_delta=0,
                actual_actor_scheduler_advance_delta=0,actual_head_update_delta=0,actual_head_optimizer_write_delta=0,generated_trajectories=0)
            native_sha=write(native_base/'logs/final-restore-readback.json',native)
            accepted_sha=write(native_base/'logs/final-restore-audit-accepted.json',dict(accepted=True,
                actual_cuda_restore_accepted=True,owned_cleanup=True,readback_sha256=native_sha,audit_id='audit'))
            goal['actual_native_restore_accepted_sha256']=accepted_sha
            goal_path=base/'logs/qprobe-final-goal-acceptance.json';goal_sha=write(goal_path,goal)
            accounting=dict(job='800001',state='COMPLETED',exit_code='0:0',raw='actual native accounting')
            write(follow/'process.json',dict(actual_process=process,commit=control.reader.FOLLOWTHROUGH))
            write(follow/'claim.json',dict(source=publications[control.reader.FOLLOWTHROUGH],
                binding=dict(producer,sigma3_settings_sha256=settings_sha)))
            binding=dict(actual_process=process,helper_commit=control.reader.FOLLOWTHROUGH,
                full_cpu_consumer_returned=True,goal_sha256=goal_sha,audit_base=str(native_base),actual_native_audit_accounting=accounting)
            write(follow/'final-consumer-binding.json',binding)
            write(follow/'result.json',dict(exit_code=0,error=None,actual_process=process,result=dict(
                phase='ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS',full_objective_accepted=True,
                actual_final_native_restore_accepted=True,final_goal_path=str(goal_path),final_goal_sha256=goal_sha)))
            raw_process=dict(pid=200,start_ticks=2000,sid=200,uid=os.getuid())
            write(raw/'process.json',raw_process)
            claim_sha=write(raw/'claim.json',dict(source=publications[control.reader.RAW]))
            write(raw/'result.json',dict(status=control.RAW_STATUS,source=publications[control.reader.RAW],
                claim_sha256=claim_sha,process=raw_process,successful_formal_outers=list(range(1,201)),normal_return_attempts=201))
            with patch.object(control.ctl,'ROOT',root),patch.object(control.ctl,'live',return_value=False),\
                 patch.object(control.reader,'ROOT',root),patch.object(control.reader,'BASE',base),\
                 patch.object(control.reader,'OWNER',os.getuid()),patch.object(control.reader.shared,'OWNER',os.getuid()),\
                 patch.object(control.reader,'FOLLOWTHROUGH_DIRECTORY',follow),patch.object(control.reader,'RAW_DIRECTORY',raw),\
                 patch.object(control.reader,'early_history',return_value=({},{})),\
                 patch.object(control.reader,'producer_acceptance',return_value=({},producer)),\
                 patch.object(control.reader,'completed',return_value=accounting),patch.object(control.reader,'load_fixed') as heavy:
                self.assertEqual(control.ready(control.reader.FOLLOWTHROUGH)['native_job'],'800001')
                heavy.assert_not_called()
                binding['goal_sha256']='f'*64;write(follow/'final-consumer-binding.json',binding)
                with self.assertRaisesRegex(ValueError,'actual terminal CPU return'):control.ready(control.reader.FOLLOWTHROUGH)

    def test_missing200_creates_no_scope_claim_or_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();base=root/'run';(base/'logs').mkdir(parents=True)
            (base/'logs/actual-updates.jsonl').write_text('{"step":1}\n')
            directory=root/'new-final-scope';before=sorted(str(p) for p in root.rglob('*'))
            with patch.object(control,'DIRECTORY',directory),patch.object(control,'caps'),\
                 patch.object(control.reader,'BASE',base),patch.object(control.reader.shared,'OWNER',os.getuid()),\
                 patch.object(control.ctl,'source') as source,patch.object(control.subprocess,'Popen') as launch,\
                 patch.object(control.reader,'producer_acceptance') as accounting:
                with self.assertRaisesRegex(ValueError,'fresh1..200'):
                    control.start('a'*40,followthrough_commit=control.reader.FOLLOWTHROUGH)
                source.assert_not_called();launch.assert_not_called();accounting.assert_not_called()
            self.assertEqual(before,sorted(str(p) for p in root.rglob('*')))

    def test_existing_unknown_scope_never_replays(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d).resolve();commit='a'*40
            (directory/'claim.json').write_text(json.dumps(dict(source=dict(commit=commit),followthrough_commit=control.reader.FOLLOWTHROUGH)))
            with patch.object(control,'DIRECTORY',directory),patch.object(control,'caps'),\
                 patch.object(control.ctl,'owner'),patch.object(control.ctl,'OWNER',os.getuid()),\
                 patch.object(control.ctl,'read',side_effect=documents),patch.object(control,'ready') as ready,\
                 patch.object(control.subprocess,'Popen') as launch:
                self.assertEqual(control.start(commit,followthrough_commit=control.reader.FOLLOWTHROUGH)['status'],'UNKNOWN_NO_REPLAY')
                ready.assert_not_called();launch.assert_not_called()

    def test_stdout_cannot_change_producer_goal_source_or_no201(self):
        commit='a'*40;_,admission,proof=output_fixture(commit)
        control.accept_output(proof,commit,control.reader.FOLLOWTHROUGH,admission)
        for key,value in [('producer_job','719225'),('update201_performed',True),('actor_model_loaded',True)]:
            bad=dict(proof,**{key:value})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'full pure sigma3'):
                control.accept_output(bad,commit,control.reader.FOLLOWTHROUGH,admission)
        bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer']['goal']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'admitted followthrough/raw'):
            control.accept_output(bad,commit,control.reader.FOLLOWTHROUGH,admission)


class HeldExecutionTests(unittest.TestCase):
    """Kernel/source admission is simulated; child stdout and wait are real."""
    def execute(self,directory,failure=None):
        commit='a'*40;source,admission,proof=output_fixture(commit)
        own=dict(pid=os.getpid(),start_ticks=100,sid=os.getpid(),uid=os.getuid())
        claim=dict(source=source,cwd=str(directory),created_unix=0,followthrough_commit=control.reader.FOLLOWTHROUGH,admission=admission)
        (directory/'claim.json').write_text(json.dumps(claim));(directory/'process.json').write_text(json.dumps(own))
        if failure=='bad_output':proof['update201_performed']=True
        stdout=json.dumps(proof);stderr='real child diagnostic' if failure=='stderr' else ''
        expected_exit=7 if failure=='nonzero' else 0
        program='import sys,time;time.sleep(.03);sys.stdout.write('+repr(stdout)+');sys.stderr.write('+repr(stderr)+');sys.exit('+str(expected_exit)+')'
        real_popen=subprocess.Popen;children=[]
        def launch(command,**kwargs):
            self.assertEqual(command,[sys.executable,'-m',control.CONSUMER,'--commit',commit,
                '--followthrough-commit',control.reader.FOLLOWTHROUGH])
            child=real_popen([sys.executable,'-c',program],**kwargs)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        def process(pid):
            if pid==own['pid']:return own
            self.assertEqual(pid,children[0].pid)
            return dict(pid=pid,start_ticks=101,sid=own['sid']+(failure=='birth'),uid=own['uid'])
        with ExitStack() as stack:
            for target,key,value in ((control,'DIRECTORY',directory),(control,'caps',Mock()),
                    (control.ctl,'OWNER',os.getuid()),(control.reader,'OWNER',os.getuid()),
                    (control.ctl,'source',Mock(return_value=(directory,source))),
                    (control.ctl,'read',Mock(side_effect=documents)),(control.ctl,'process',Mock(side_effect=process)),
                    (control.ctl,'live',Mock(side_effect=lambda birth:children[0].poll() is None)),
                    (control.subprocess,'Popen',Mock(side_effect=launch))):
                stack.enter_context(patch.object(target,key,value))
            with redirect_stderr(io.StringIO()):
                code=control.supervise(commit,followthrough_commit=control.reader.FOLLOWTHROUGH)
        self.assertEqual(len(children),1);children[0].wait.assert_called_once_with()
        self.assertEqual(children[0].poll(),expected_exit)
        result=documents(directory/'result.json')
        self.assertEqual(result['actual_child_exit_code'],expected_exit)
        self.assertEqual(result['supervisor_exit_code'],code)
        self.assertEqual(result['source'],source);self.assertFalse(result['goal_complete'])
        self.assertEqual(result['claim_sha256'],control.ctl.digest(directory/'claim.json'))
        for name,content in [('stdout',stdout),('stderr',stderr)]:
            path=directory/('consumer.'+name)
            self.assertEqual(path.read_text(),content)
            self.assertEqual(result[name],dict(path=str(path),bytes=path.stat().st_size,sha256=control.ctl.digest(path)))
        return commit,source,code,result

    def test_actual_local_wait0_and_terminal_consumer_require_both_births_dead(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d).resolve();commit,source,code,result=self.execute(directory)
            self.assertEqual(code,0);self.assertIsNone(result['error'])
            with patch.object(control,'DIRECTORY',directory),patch.object(control.ctl,'source',return_value=(directory,source)),\
                 patch.object(control.ctl,'read',side_effect=documents),patch.object(control.reader,'OWNER',os.getuid()),\
                 patch.object(control.ctl,'live',return_value=False):
                self.assertEqual(control.consume(commit,followthrough_commit=control.reader.FOLLOWTHROUGH)['status'],control.HELD_STATUS)
            for active in (result['process'],result['child']):
                with patch.object(control,'DIRECTORY',directory),patch.object(control.ctl,'source',return_value=(directory,source)),\
                     patch.object(control.ctl,'read',side_effect=documents),patch.object(control.reader,'OWNER',os.getuid()),\
                     patch.object(control.ctl,'live',side_effect=lambda row:row==active):
                    with self.assertRaisesRegex(ValueError,'held sigma3 wait/source/birth'):
                        control.consume(commit,followthrough_commit=control.reader.FOLLOWTHROUGH)

    def test_nonzero_stderr_wrong_stdout_and_post_spawn_birth_fail_closed(self):
        for failure,message in [('nonzero','did not return0'),('stderr','stderr is nonempty'),
                                ('bad_output','full pure sigma3'),('birth','child identity differs')]:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                _,_,code,result=self.execute(Path(d).resolve(),failure)
                self.assertEqual(code,1);self.assertIn(message,result['error'])


if __name__=='__main__':unittest.main()
