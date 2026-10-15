"""Terminal handoff, actual CPU identities and no-replay recovery boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import recovery_followthrough719225 as recovery
from reproduction.hpc3_qprobe import final_control as ctl


def birth(pid, sid=None):
    return dict(pid=pid, sid=pid if sid is None else sid, uid=14090, start_ticks=pid+100)


class RecoveryFollowthroughTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.patches = [patch.object(recovery, 'ROOT', self.root),
            patch.object(recovery, 'ORIGINAL', self.root/'followthrough/719225'),
            patch.object(recovery, 'DIRECTORY', self.root/'followthrough/recovery719225-finalcpu-20261015')]
        for item in self.patches:
            item.start()
        self.pub = dict(commit=recovery.EA, publication_path='source-proof', publication_sha256='a'*64)
        self.fixed = dict(job='719225', producer_commit='1843', run_id=recovery.reader.RUN)
        self.old = birth(100); self.own = birth(200, 199)
        self.actual = dict(job='719225', state='FAILED', exit_code='1:0',
            raw='719225|FAILED|1:0|jxu957|14090|jxu957|formal-user|emergency_acd\n')

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def ea(self):
        claim = dict(schema_version=1, source=self.pub, binding=self.fixed,
            command=[str(recovery.BASE/'env/bin/python'), '-m',
                'reproduction.hpc3_qprobe.followthrough719225', 'supervise', '--commit', recovery.EA],
            cwd=str(self.root/'source'/recovery.EA), created_unix=1,
            training_submission_permitted=False, actor_model_reload_permitted=False,
            direct_gpu_access_permitted=False, native_final_restore_only_submission_permitted=True)
        error = repr(ValueError(recovery.ERROR))
        documents = {'claim.json':claim, 'process.json':dict(actual_process=self.old, commit=recovery.EA,
            control=str(recovery.ORIGINAL)), 'started.json':dict(actual_process=self.old),
            'result.json':dict(exit_code=1, error=error, actual_process=self.old, checked_unix=3,
                training_submission_performed=False, actor_model_loaded=False,
                direct_gpu_device_access_performed=False, native_final_restore_only_submission_permitted=True),
            'state.json':dict(phase='ACTUAL_CPU_FOLLOWTHROUGH_STOPPED_FOR_DIAGNOSIS', error=error,
                previous_actual_state=None, full_objective_accepted=False),
            'producer-terminal-failure.json':dict(self.actual, checked_unix=2)}
        path = str(self.root/'source'/recovery.EA/'reproduction/hpc3_qprobe/followthrough719225.py')
        output = ('Traceback (most recent call last):\n  File "'+path+'", line 242, in supervise\n'
                  '    result=one_tick(commit,verified)\n  File "'+path+'", line 147, in one_tick\n'
                  '    raise ValueError(...)\nValueError: '+recovery.ERROR+'\n')
        return documents, output

    def validate_ea(self, documents, output, alive=lambda row:False):
        return recovery.validate_ea(documents, output, self.pub, self.fixed, self.actual, alive=alive)

    def test_original_ea_failure_is_dead_self_reported_not_invented_parent_wait(self):
        documents, output = self.ea(); proof = self.validate_ea(documents, output)
        self.assertFalse(proof['external_parent_wait_receipt_available'])
        self.assertEqual(proof['self_reported_exit_code'], 1)
        documents['state.json']['previous_actual_state'] = dict(accounting=dict(state='RUNNING'))
        self.validate_ea(documents, output)

    def test_ea_live_different_failure_wrong_source_marker_and_traceback_reject(self):
        documents, output = self.ea()
        with self.assertRaisesRegex(ValueError, 'birth is live'):
            self.validate_ea(documents, output, alive=lambda row:True)
        cases = [('result.json', 'exit_code', True), ('result.json', 'error', "ValueError('other')"),
            ('claim.json', 'source', dict(commit='b'*40)), ('started.json', 'actual_process', birth(101)),
            ('producer-terminal-failure.json', 'raw', self.actual['raw'].replace('14090', '14091'))]
        for name, key, value in cases:
            with self.subTest(name=name, key=key):
                bad = copy.deepcopy(documents); bad[name][key] = value
                with self.assertRaises(ValueError):
                    self.validate_ea(bad, output)
        for bad in (output.replace('line 147', 'line 148'), output.replace('in one_tick', 'in inspect'),
                    output+'another exception\n', output+output):
            with self.assertRaises(ValueError):
                self.validate_ea(documents, bad)

    def archive(self, final=False):
        step = None if final else 50; role = 'detached_cpu_final_audit_submission' if final else 'detached_cpu_pair_archive'
        directory = self.root/'final-control'/recovery.reader.RUN/('audit-controller' if final else 'archive-50')
        directory.mkdir(parents=True, exist_ok=True)
        claim = dict(schema_version=1, role=role, token='c'*32, step=step, producer=self.fixed,
            source=self.pub, controller=self.old, created_unix=1, training_permitted=False,
            direct_gpu_device_access_permitted=False, native_gpu_audit_submission_permitted=final)
        claim['command'] = ctl.command(self.root/'source'/recovery.EA,
            'audit-worker' if final else 'archive-worker', directory/'submit-claim.json', claim['token'])
        (directory/'submit-claim.json').write_text(json.dumps(claim))
        sha = ctl.digest(directory/'submit-claim.json'); runner = birth(300); child = birth(301, 300)
        common = dict(token=claim['token'], claim_sha256=sha)
        runner_record = dict(common, actual_process=runner, controller=self.old)
        task_binding = dict(common, actual_process=child, supervisor=runner)
        task = dict(common, actual_process=child, completed_unix=2,
            status='CPU_FINAL_SUBMIT_CONTROLLER_RETURNED' if final else 'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',
            training_performed=False, direct_gpu_device_access_performed=False,
            native_gpu_audit_submission_attempted=final, actor_model_loaded=False)
        (directory/'task-result.json').write_text(json.dumps(task))
        result = dict(common, actual_task_birth=child, actual_supervisor=runner, role=role,
            actual_task_exit_code=0, finished_unix=3,
            status='ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0' if final else 'ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0',
            task_result_sha256=ctl.digest(directory/'task-result.json'))
        (directory/'worker-result.json').write_text(json.dumps(result))
        return [claim, runner_record, task_binding, result, task], step

    def validate_archive(self, values, step, alive=lambda row:False, final=False):
        return recovery.validate_archive_documents(*values, step=step, binding=self.fixed,
            publication=self.pub, allowed_controllers=[self.old], alive=alive, final_submit=final)

    def test_actual_held_archive0_requires_both_dead_and_waits_same_live_runner(self):
        values, step = self.archive(); self.assertTrue(self.validate_archive(values, step)['ready'])
        state = self.validate_archive(values, step, alive=lambda row:row == values[1]['actual_process'])
        self.assertFalse(state['ready']); self.assertTrue(state['runner_alive'])
        with self.assertRaises(ValueError):
            self.validate_archive(values, step, alive=lambda row:row == values[2]['actual_process'])

    def test_missing_bindings_dead_unknown_and_failed_archives_never_pass(self):
        values, step = self.archive()
        for index in (1, 2, 4):
            bad = copy.deepcopy(values); bad[index] = None
            with self.subTest(missing=index), self.assertRaises(ValueError):
                self.validate_archive(bad, step)
        bad = copy.deepcopy(values); bad[3] = None
        with self.assertRaisesRegex(ValueError, 'dead without held outcome'):
            self.validate_archive(bad, step)
        self.assertFalse(self.validate_archive(bad, step, alive=lambda row:row == values[1]['actual_process'])['ready'])
        for index, key, value in ((0, 'controller', birth(999)), (0, 'source', dict(commit='d'*40)),
            (1, 'token', 'e'*32), (2, 'supervisor', birth(400)), (3, 'actual_task_birth', birth(402, 300)),
            (3, 'actual_task_exit_code', False), (3, 'actual_task_exit_code', 1),
            (3, 'task_result_sha256', '0'*64), (4, 'actor_model_loaded', True)):
            bad = copy.deepcopy(values); bad[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                self.validate_archive(bad, step)

    def test_final_cpu_submit_uses_same_full_identity_chain_and_distinct_scope(self):
        values, step = self.archive(final=True)
        self.assertTrue(self.validate_archive(values, step, final=True)['ready'])
        for index in (1, 2):
            bad = copy.deepcopy(values); bad[index] = None
            with self.assertRaises(ValueError):
                self.validate_archive(bad, step, final=True)
        bad = copy.deepcopy(values); bad[0]['native_gpu_audit_submission_permitted'] = False
        with self.assertRaises(ValueError):
            self.validate_archive(bad, step, final=True)

    def test_healthy_or_missing200_refuses_before_source_scope_or_process(self):
        for message in ('healthy producer', 'actual current6..200 absent'):
            with patch.object(recovery, 'caps'), patch.object(ctl, 'final_producer', side_effect=ValueError(message)), \
                    patch.object(ctl, 'source') as source, patch.object(recovery.subprocess, 'Popen') as process:
                with self.assertRaisesRegex(ValueError, message):
                    recovery.start('f'*40, 'b'*40)
                source.assert_not_called(); process.assert_not_called()
                self.assertFalse(recovery.DIRECTORY.exists())

    def test_existing_once_scope_is_inspected_without_admission_or_launch(self):
        recovery.DIRECTORY.mkdir(parents=True)
        with patch.object(recovery, 'caps'), patch.object(recovery, 'inspect', return_value={'status':'UNKNOWN'}) as inspect, \
                patch.object(recovery, 'admission') as admission, patch.object(recovery.subprocess, 'Popen') as process:
            self.assertEqual(recovery.start('f'*40, 'b'*40), {'status':'UNKNOWN'})
            inspect.assert_called_once_with('f'*40, 'b'*40); admission.assert_not_called(); process.assert_not_called()

    def test_orphan_payload_refuses_without_archive_or_native_submission(self):
        payload = self.root/'archives'/recovery.reader.RUN/'global_step_5'; payload.mkdir(parents=True)
        with patch.object(recovery, 'preserved'), patch.object(ctl, 'launch_archive') as archive, \
                patch.object(ctl, 'start_audit') as native:
            with self.assertRaisesRegex(ValueError, 'orphan archive payload'):
                recovery.one_tick(dict(source=dict(commit='f'*40)), self.own, {}, {})
            archive.assert_not_called(); native.assert_not_called()

    def test_archive_transport_tampering_refuses_before_next_archive_or_native(self):
        directory = self.root/'final-control'/recovery.reader.RUN/'archive-5'; directory.mkdir(parents=True)
        payload = self.root/'archives'/recovery.reader.RUN/'global_step_5'; payload.mkdir(parents=True)
        path = payload/'complete.json'; path.write_text('changed transport')
        observed = dict(status='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0', actual_task_exit_code=0,
            transport_path=str(path), transport_sha256='0'*64)
        with patch.object(recovery, 'preserved'), patch.object(recovery, 'archive_state', return_value=dict(ready=True)), \
                patch.object(ctl, 'inspect_archive', return_value=observed), patch.object(ctl, 'launch_archive') as launch, \
                patch.object(ctl, 'start_audit') as native, patch.object(recovery.reader, 'OWNER', path.stat().st_uid):
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                recovery.one_tick(dict(source=dict(commit='f'*40)), self.own, {}, {})
            launch.assert_not_called(); native.assert_not_called()

    def test_intermediate_pass_never_claims_full_goal(self):
        claim = dict(source=self.pub, admission=dict(producer_final_cpu_recovery={'commit':'b'*40},
            original_ea=dict(fingerprint='a'*64)))
        value = dict(status=recovery.STATUS, source=self.pub, actual_process=self.own,
            producer_final_cpu_recovery={'commit':'b'*40}, original_ea_fingerprint='a'*64,
            goal_complete=False, full_objective_accepted=False, independent_teacher_raw_final_consumer_still_required=True,
            result=dict(phase=recovery.STATUS, native4_goal_reader_pass=True, full_objective_accepted=False, goal_complete=False),
            training_submission_performed=False, actor_model_loaded=False, direct_gpu_device_access_performed=False,
            update201_performed=False)
        recovery.accept_output(value, claim, self.own)
        for key in ('goal_complete', 'full_objective_accepted', 'actor_model_loaded', 'update201_performed'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.accept_output(dict(value, **{key:True}), claim, self.own)

    def test_native_job_source_readback_sha_and_no_work_cannot_be_substituted(self):
        readback = dict(slurm_job_id='800000', run_id=recovery.reader.RUN,
            producer_commit=recovery.reader.PRODUCER, audit_commit=recovery.EA,
            world_size=4, actual_completed_outer=200, update201_performed=False, fit_called=False,
            actual_actor_optimizer_write_delta=0, actual_actor_scheduler_advance_delta=0,
            actual_head_update_delta=0, actual_head_optimizer_write_delta=0, generated_trajectories=0, audit_id='id')
        accepted = dict(accepted=True, actual_cuda_restore_accepted=True, owned_cleanup=True,
            readback_sha256='a'*64, audit_id='id')
        goal = dict(actual_native_restore_accepted_sha256='b'*64)
        values = [goal, readback, accepted, {'sha256':'a'*64}, {'sha256':'b'*64}, {'job':'800000'}]
        recovery.native_goal_binding(*values)
        cases = ((1, 'slurm_job_id', '800001'), (1, 'audit_commit', '0'*40),
            (1, 'world_size', 8), (1, 'actual_actor_optimizer_write_delta', 1),
            (1, 'generated_trajectories', False), (1, 'fit_called', True),
            (2, 'readback_sha256', '0'*64), (2, 'owned_cleanup', False),
            (0, 'actual_native_restore_accepted_sha256', '0'*64))
        for index, key, value in cases:
            bad = copy.deepcopy(values); bad[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                recovery.native_goal_binding(*bad)

    def test_live_parent_or_child_cannot_consume_a_published_wait0(self):
        parent = birth(500); child = birth(501, 500); commit = recovery.EA; stage1 = 'b'*40
        claim = dict(source=self.pub, recovery_commit=stage1, created_unix=1)
        result = dict(source=self.pub, claim_sha256='a'*64, process=parent, child=child,
            actual_child_exit_code=0, supervisor_exit_code=0, error=None, started_unix=2, completed_unix=3)
        documents = {'claim.json':claim, 'process.json':parent, 'child.json':child, 'result.json':result}
        for live_birth in (parent, child):
            with patch.object(recovery, 'caps'), patch.object(ctl, 'source', return_value=(self.root, self.pub)), \
                    patch.object(ctl, 'read', side_effect=lambda path:documents[Path(path).name]), \
                    patch.object(ctl, 'digest', return_value='a'*64), \
                    patch.object(ctl, 'live', side_effect=lambda row:row == live_birth), \
                    patch.object(recovery, 'preserved') as preserved:
                with self.assertRaisesRegex(ValueError, 'held recovery wait'):
                    recovery.consume(commit, stage1)
                preserved.assert_not_called()

    def test_pending_or_unknown_native_intent_never_calls_goal_or_resubmits(self):
        observed = dict(status='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0', actual_task_exit_code=0)
        for step in ctl.STEPS:
            (self.root/'final-control'/recovery.reader.RUN/('archive-'+str(step))).mkdir(parents=True)
            payload = self.root/'archives'/recovery.reader.RUN/('global_step_'+str(step)); payload.mkdir(parents=True)
            (payload/'complete.json').write_text('{}')
        (self.root/'final-control'/recovery.reader.RUN/'audit-controller').mkdir()
        def archive(base, step, full=False):
            if full:
                return dict(fresh_full_sha={'actual_full_sha':True})
            path = self.root/'archives'/recovery.reader.RUN/('global_step_'+str(step))/'complete.json'
            return dict(observed, transport_path=str(path), transport_sha256=ctl.digest(path))
        cpu = dict(status='ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0', actual_task_exit_code=0,
            runner_alive=False, task_alive=False)
        claim = dict(source=self.pub, recovery_commit='b'*40)
        with patch.object(recovery, 'preserved'), patch.object(recovery.reader, 'OWNER', self.root.stat().st_uid), \
                patch.object(recovery, 'archive_state', return_value=dict(ready=True, worker_result_sha256='a'*64)), \
                patch.object(ctl, 'inspect_archive', side_effect=archive), \
                patch.object(recovery, 'submit_state', return_value=dict(ready=True)), \
                patch.object(ctl, 'start_audit') as submit, \
                patch.object(recovery.normal, 'accounting', return_value=dict(job='800000', state='PENDING', exit_code='0:0')), \
                patch('reproduction.hpc3_qprobe.verify_goal.verify') as verify:
            with patch.object(ctl, 'inspect_audit', return_value=dict(status='ACTUAL_SLURM_SUBMITTED',
                    job='800000', audit_base=str(self.root/'native'), cpu_controller=cpu)):
                state = recovery.one_tick(claim, self.own, {}, {})
                self.assertEqual(state['phase'], 'ACTUAL_RECOVERY_FINAL_NATIVE_RESTORE_WAIT')
            with patch.object(ctl, 'inspect_audit', return_value=dict(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY',
                    job=None, cpu_controller=cpu)):
                with self.assertRaisesRegex(ValueError, 'submission unknown or failed'):
                    recovery.one_tick(claim, self.own, {}, {})
            submit.assert_not_called(); verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
