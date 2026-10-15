"""CPU fixtures: outage waiting, immutable predecessor routing, no replay."""
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import slurm_readonly as queries
from reproduction.hpc3_qprobe import raw_audit_followthrough_sigma3 as raw
from reproduction.hpc3_qprobe import followthrough_sigma3 as follow
from reproduction.hpc3_qprobe import sigma3_service_recovery as handoff
from reproduction.hpc3_qprobe import recover_sigma3_raw49 as recovered
from reproduction.hpc3_qprobe import final_control as control
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as final_reader
from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as shared
from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as holder


def unavailable():
    return queries.SlurmReadUnavailable(dict(argv=['sacct','-j','721485'],returncode=1,
        stderr='Munge encode failed; Connection refused',state='UNKNOWN'))


class ServiceOutage(unittest.TestCase):
    def test_failed_query_is_unknown_and_retains_real_error(self):
        value=SimpleNamespace(returncode=1,stdout='',stderr='Munge socket missing')
        with patch.object(queries.subprocess,'run',return_value=value):
            with self.assertRaises(queries.SlurmReadUnavailable) as caught:queries.query(['sacct','-j','721485'])
        self.assertEqual(caught.exception.evidence['state'],'UNKNOWN')
        self.assertEqual(caught.exception.evidence['returncode'],1)
        self.assertEqual(caught.exception.evidence['stderr'],'Munge socket missing')

    def test_timeout_is_unknown_and_submissions_cannot_be_retried(self):
        with patch.object(queries.subprocess,'run',side_effect=subprocess.TimeoutExpired(['sacct'],1)):
            with self.assertRaises(queries.SlurmReadUnavailable):queries.query(['sacct'])
        with patch.object(queries.subprocess,'run') as run:
            for argv in (['sbatch','train.sh'],['scancel','721485'],['scontrol','update','JobId=721485']):
                with self.subTest(argv=argv),self.assertRaises(ValueError):queries.query(argv)
            run.assert_not_called()

    def test_successful_query_with_wrong_owner_is_not_retried_as_unknown(self):
        with patch.object(follow,'slurm_query',return_value='721485|RUNNING|0:0|mou176|14090|jxu957|formal-user|emergency_acd\n'):
            with self.assertRaises(ValueError) as caught:follow.accounting()
        self.assertNotIsInstance(caught.exception,queries.SlurmReadUnavailable)

    def test_raw_outage_wait_keeps49_and_never_reexecutes_prefix(self):
        supervisor=dict(pid=10,start_ticks=20,sid=10,uid=14090)
        records=[{'task':dict(step=i,complete_outer=True)} for i in range(1,50)]
        source={'commit':'a'*40};claim=dict(source=source,parts=['fixture'])
        with patch.object(raw,'validate',return_value=(Path('/fixture'),source)),\
             patch.object(raw.ctl,'read',return_value=claim),patch.object(raw.ctl,'process',return_value=supervisor),\
             patch.object(raw.ctl,'publish') as publish,patch.object(raw,'plan',side_effect=[unavailable(),['fixture']]),\
             patch.object(raw,'actual_tasks',return_value=[r['task'] for r in records]),\
             patch.object(raw.ctl,'producer',side_effect=[unavailable(),({}, {}, '|RUNNING|')]),\
             patch.object(raw,'state') as state,patch.object(raw,'execute_task') as execute,\
             patch.object(raw.time,'sleep',side_effect=[None,None,KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):raw.worker('a'*40,initial_records=records)
        execute.assert_not_called()
        waits=[c.args[0] for c in state.call_args_list if c.args[0]['phase']=='UNKNOWN_ACCOUNTING_WAIT']
        self.assertEqual(len(waits),2);self.assertTrue(all(v['completed_attempt_audits']==49 for v in waits))
        self.assertFalse(any(c.args[0].name=='result.json' for c in publish.call_args_list))

    def test_follower_waits_then_continues_same_supervisor(self):
        birth=dict(pid=10,start_ticks=20,sid=10,uid=14090)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'process.json').write_text('{}')
            claim=dict(source={'commit':'a'*40},binding={'bound':True},cwd=str(Path.cwd().resolve()))
            def read(path):return {'actual_process':birth} if Path(path).name=='process.json' else claim
            with patch.object(follow,'DIRECTORY',directory),patch.object(follow,'read',side_effect=read),\
                 patch.object(follow.control,'process',return_value=birth),patch.object(follow.control,'source'),\
                 patch.object(follow.control,'publish') as publish,patch.object(follow,'fixed_binding',return_value={'bound':True}),\
                 patch.object(follow,'one_tick',side_effect=[unavailable(),{'full_objective_accepted':True}]) as tick,\
                 patch.object(follow,'publish_state') as state,patch.object(follow.time,'sleep'):
                self.assertEqual(follow.supervise('a'*40,initial_verified_archives={5:'old-sha'}),0)
            self.assertEqual(state.call_args_list[0].args[0]['phase'],'UNKNOWN_ACCOUNTING_WAIT')
            self.assertEqual(tick.call_args_list[1].args[1],{5:'old-sha'})
            self.assertEqual(publish.call_args_list[-1].args[1]['actual_process'],birth)

    def test_wrong_host_rejected_before_any_death_inference(self):
        with patch.object(handoff.socket,'gethostname',return_value='ACD-Login-2'),\
             patch.object(Path,'read_text',return_value=handoff.HOST['boot_id']),\
             patch.object(handoff.os,'readlink',return_value=handoff.HOST['pid_namespace']),\
             patch.object(handoff.ctl,'live') as live:
            with self.assertRaisesRegex(ValueError,'HOST_IDENTITY'):handoff.host_guard()
        live.assert_not_called()

    def test_same_size_predecessor_corruption_changes_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(handoff.ctl,'OWNER',os.getuid()):
            folder=Path(tmp).resolve();p=folder/'receipt.json';p.write_text('aaaa')
            before=handoff.snapshot(folder);p.write_text('bbbb');after=handoff.snapshot(folder)
        self.assertEqual(before['receipt.json']['bytes'],after['receipt.json']['bytes'])
        self.assertNotEqual(before['receipt.json']['sha256'],after['receipt.json']['sha256'])

    def test_original49_is_consumed_with_original_supervisor_not_new_source(self):
        task={'attempt_index':49,'attempt_id':'b'*32};old=Mock();old.consume_task.return_value={'original':True}
        service=SimpleNamespace(old_raw=lambda:(old,{}))
        path=recovered.task_directory(task)
        with patch.object(recovered,'service',return_value=service),patch.object(recovered,'core') as core:
            self.assertEqual(recovered.consume_task(task,path,{'pid':999}),{'original':True})
            old.consume_task.assert_called_once_with(task,path,handoff.RAW_BIRTH);core.assert_not_called()
            with self.assertRaises(ValueError):recovered.consume_task(task,recovered.DIRECTORY/'attempts'/task['attempt_id'],{})

    def test_suffix_receipt_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';(old/'accepted').mkdir(parents=True);(new/'accepted').mkdir(parents=True)
            for i in range(1,50):(old/'accepted'/('%06d.json'%i)).write_text('{}')
            (new/'accepted'/'000051.json').write_text('{}')
            with patch.object(recovered.recovery,'RAW_ORIGINAL',old),patch.object(recovered,'DIRECTORY',new):
                with self.assertRaisesRegex(ValueError,'overlap or have a gap'):recovered.accepted_paths()

    def test_existing_raw_recovery_claim_never_spawns_again(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(recovered,'DIRECTORY',Path(tmp)),\
             patch.object(recovered,'validate',return_value=(Path('/fixture'),{})),\
             patch.object(recovered,'service',return_value=SimpleNamespace(prepare=lambda commit:{})),\
             patch.object(recovered,'inspect',return_value={'status':'UNKNOWN_NO_REPLAY'}),\
             patch.object(recovered.subprocess,'Popen') as popen:
            self.assertEqual(recovered.start('a'*40)['status'],'UNKNOWN_NO_REPLAY')
        popen.assert_not_called()

    def test_two_real_source_roots_bind_private_raw_and_adjacent_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();local=Path(recovered.__file__).parent
            for commit in ('a'*40,'b'*40):
                repo=root/'source'/commit;folder=repo/'reproduction/hpc3_qprobe';folder.mkdir(parents=True)
                for name in ('final_control.py','recover_sigma3_raw49.py','raw_audit_followthrough_sigma3.py','sigma3_service_recovery.py'):
                    value=(local/name).read_text()
                    if name=='final_control.py':
                        value=value.replace("ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')",'ROOT=Path('+repr(str(root))+')')
                        value=value.replace('OWNER=14090','OWNER='+str(os.getuid()))
                    (folder/name).write_text(value)
                for name in ('verl/version/version','verl/single_controller/version/version'):
                    version=repo/name;version.parent.mkdir(parents=True,exist_ok=True);version.write_text('fixture\n')
                files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in repo.rglob('*') if p.is_file()}
                (root/'preparation').mkdir(exist_ok=True)
                (root/'preparation'/('source-'+commit+'-verification.json')).write_text(json.dumps(
                    dict(commit=commit,status='PUBLISHED_SOURCE_FULL_SHA_PASS',files=files)))
            env={k:'1' for k in control.CAPS};env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
            with patch.object(shared,'ROOT',root),patch.object(shared,'OWNER',os.getuid()),patch.dict(os.environ,env):
                loaded,source=shared.load_fixed('a'*40,'recover_sigma3_raw49','ctl')
                other,_=shared.load_fixed('b'*40,'raw_audit_followthrough_sigma3','ctl')
                # Ordinary imported runtime belongs to another reader source.
                with patch.object(loaded.runtime,'ctl',other.ctl):
                    repository,verified=loaded.validate('a'*40)
                    self.assertEqual(verified,source)
                    self.assertEqual(repository,root/'source'/('a'*40))
                    self.assertEqual(loaded.core().validate('a'*40)[1],source)
                    self.assertIs(loaded.service().ctl,loaded.ctl)
                    self.assertEqual(loaded.service().ctl.source('a'*40)[1],source)
                    with self.assertRaises(ValueError):other.ctl.source('a'*40)

    def test_existing_native_job_query_failure_then_recovery_never_submits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory=root/'final-audit';directory.mkdir()
            claim={'producer':{},'audit_base':'/fixture'};p=directory/'submit-claim.json';p.write_text(json.dumps(claim))
            (directory/'submission.json').write_text(json.dumps(dict(job='777',status='ACTUAL_SLURM_SUBMITTED',
                claim_sha256=control.digest(p))))
            with patch.object(control,'OWNER',os.getuid()),patch.object(control,'final_producer',return_value=({}, {}, '',None)),\
                 patch.object(control,'control_root',return_value=root),\
                 patch.object(control,'slurm_query',side_effect=[unavailable(),'777|RUNNING|0:0|14090|jxu957|formal-user|emergency_acd\n']),\
                 patch.object(control.subprocess,'Popen') as popen:
                with self.assertRaises(queries.SlurmReadUnavailable):control.inspect_audit('/fixture')
                self.assertEqual(control.inspect_audit('/fixture')['job'],'777')
            popen.assert_not_called();self.assertEqual(json.loads(p.read_text()),claim)

    def test_final_recovery_route_is_explicit_and_keeps_archive5_pin(self):
        commit='c'*40;old=final_reader.ARCHIVE5_RECOVERY_SOURCE
        with self.assertRaises(ValueError):final_reader.followthrough_route(commit,old)
        with self.assertRaises(ValueError):final_reader.followthrough_route(commit,'d'*40,commit)
        module,directory=final_reader.followthrough_route(commit,old,commit)
        self.assertEqual(module,'recover_sigma3_followthrough_service')
        self.assertEqual(directory,final_reader.SERVICE_RECOVERY_DIRECTORY/'followthrough-control')
        self.assertNotEqual(holder.scope(old,commit),holder.scope(old))
        self.assertEqual(holder.recovery_args(old,commit),['--recovery-commit',old,'--observer-recovery-commit',commit])


if __name__=='__main__':unittest.main()
