"""CPU fixtures for the companion's waiting, transport and once-only ordering."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('future_companion_fixture', HERE/'archive_post_callback_future_hk.py')
future = importlib.util.module_from_spec(spec); spec.loader.exec_module(future)
spec = importlib.util.spec_from_file_location('frozen_helper_fixture', HERE/'archive_post_callback5_hk.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
spec = importlib.util.spec_from_file_location('future_observer_fixture', HERE/'observe_post_callback_future_hk.py')
observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)


def phase(step):
    return dict(callbacks_complete=True, reward_refresh_pending=False, probe_round=min(step//50,3)*50,
        post_callback_rng_captured=True, final51713_pending_at_boundary=step == 200,
        sampling_rng_stage='post_QA5' if step == 5 else ('post_fixed512_pre_final51713' if step == 200 else
            'post_fixed512_and_grounding_refit'))


def fixture_helper(root):
    def write(path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: json.dump(value, stream)
    return SimpleNamespace(BASE=root/'base', RUN=old.RUN, JOB=old.JOB, TRAINING=old.TRAINING,
        STEP=5, CORE=root/'core5', CONTROL=root/'control', SOURCE_SNAPSHOT=root/'snapshot',
        MODULE_NAME=old.MODULE_NAME, MODULE_SHA=old.MODULE_SHA, MODULE_BYTES=old.MODULE_BYTES,
        MODULE_COMMIT=old.MODULE_COMMIT, HPC_TOOL=old.HPC_TOOL, HK_TOOL=old.HK_TOOL,
        validate_envelope=old.validate_envelope, remote_install_code=old.remote_install_code,
        remote_command=lambda code: code, redact=old.redact, write_json=write,
        canonical=lambda path: Path(path), fsync_directory=lambda path: None,
        sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        read_json=lambda path: json.loads(Path(path).read_text()), ready_core=Mock(return_value={'actual_ACK': True}))


class CompanionTests(unittest.TestCase):
    def test_envelope_requires_real_frozen_boundary_and_published_dependency_bytes(self):
        raw = (HERE.parents[1]/'easy2560_joint4/post_callback_archive.py').read_bytes()
        dependency = b'public dependency module\n'
        envelope = dict(password='fixture-only-password', module_source=raw.decode(),
            module_sha256=old.MODULE_SHA, published_commit=old.MODULE_COMMIT,
            dependencies_source=dependency.decode(), dependencies_commit='c'*40,
            dependencies_sha256=hashlib.sha256(dependency).hexdigest())
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name))
            secret, same, data = future.validate_envelope(helper, envelope)
            self.assertEqual(same, raw); self.assertEqual(data['raw'], dependency)
            for key, bad in [('published_commit','d'*40), ('dependencies_sha256','f'*64), ('dependencies_commit','HEAD')]:
                changed = dict(envelope); changed[key] = bad
                with self.assertRaises(ValueError): future.validate_envelope(helper, changed)

    def test_step_configuration_only_changes_private_four_fields(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); before = dict(vars(helper))
            future.configure_step(helper, 50)
            self.assertEqual(helper.STEP, 50)
            self.assertTrue(str(helper.SOURCE_SNAPSHOT).endswith('global_step_50/source-snapshot.json'))
            self.assertEqual({k:v for k,v in vars(helper).items() if k not in {'STEP','CORE','CONTROL','SOURCE_SNAPSHOT'}},
                {k:v for k,v in before.items() if k not in {'STEP','CORE','CONTROL','SOURCE_SNAPSHOT'}})
            with self.assertRaises(ValueError): future.configure_step(helper, 5)

    def test_real_rolling_filename_reaches_ready_and_wrong_round_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); helper.STEP = 50
            folder = helper.BASE/'preparation/rolling-checkpoints/global_step_50'; (folder/'boundary').mkdir(parents=True)
            manifest = dict(run_id=helper.RUN, source_commit=helper.TRAINING, completed_outer=50,
                callbacks_complete=True, probe_round=50)
            (folder/'boundary/manifest.json').write_text(json.dumps(manifest))
            (folder/'full-manifest.json').write_text('{}')
            code = future.boundary_observation_code(helper); output = io.StringIO()
            with contextlib.redirect_stdout(output): exec(compile(code, '<read-only-boundary-fixture>', 'exec'), {})
            self.assertTrue(json.loads(output.getvalue())['ready'])
            manifest['probe_round'] = 0; (folder/'boundary/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()): exec(code, {})

    def test_no_source_access_while_waiting_then_actual_ack(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); helper.STEP = 50; helper.CORE.mkdir()
            def arrived(_):
                for relative in ('source-snapshot.json','source-acceptance.json','archive-receipt.json','checkpoint/complete.json'):
                    path = helper.CORE/relative; path.parent.mkdir(exist_ok=True); path.write_text('{}')
            with patch.object(future, 'controller_observation', return_value={'state':'UNKNOWN'}), patch.object(future,'progress') as event:
                before, _ = future.wait_for_core(helper, [], sleep=arrived)
            self.assertEqual(before, {'actual_ACK':True}); event.assert_called_once()
            self.assertFalse(event.call_args.kwargs['source_ssh_called'])

    def test_dead_controller_without_core_is_dependency_terminal(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name))
            with patch.object(future,'controller_observation',return_value={'state':'TERMINAL','records':{'failure.json':{}}}), \
                    patch.object(future,'progress') as event:
                with self.assertRaisesRegex(RuntimeError, 'DEPENDENCY_TERMINAL'): future.wait_for_core(helper, [], sleep=Mock())
            helper.ready_core.assert_not_called(); self.assertFalse(event.call_args.kwargs['training_failure_inferred'])

    def test_completed_controller_is_valid_after_core200_actual_ack(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); helper.CORE.mkdir()
            for relative in ('source-snapshot.json','source-acceptance.json','archive-receipt.json','checkpoint/complete.json'):
                path = helper.CORE/relative; path.parent.mkdir(exist_ok=True); path.write_text('{}')
            with patch.object(future,'controller_observation',return_value={'state':'TERMINAL','records':{'result.json':{}}}):
                self.assertEqual(future.wait_for_core(helper, [50,100,150])[0], {'actual_ACK':True})

    def test_readonly_transport_has_bounded_retry_and_unknown_is_not_terminal(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); helper.STEP = 50
            returned = [subprocess.TimeoutExpired('ssh',180), SimpleNamespace(returncode=255,stderr='transport',stdout=''),
                SimpleNamespace(returncode=0,stderr='',stdout='{"ready":true}')]
            with patch.object(future.subprocess,'run',side_effect=returned) as called:
                value = future.observe_source(helper, lambda code:['ssh',code], {'SSHPASS':'fixture'}, sleep=Mock())
            self.assertTrue(value['ready']); self.assertEqual(called.call_count,3)
            with patch.object(future.subprocess,'run',return_value=returned[1]) as called:
                value = future.observe_source(helper, lambda code:['ssh',code], {}, attempts=2, sleep=Mock())
            self.assertTrue(value['transport_unknown']); self.assertFalse(value['ready']); self.assertEqual(called.call_count,2)

    def test_readonly_identity_and_bad_json_never_retry(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name))
            for row in (SimpleNamespace(returncode=86,stderr='wrong host',stdout=''),
                        SimpleNamespace(returncode=0,stderr='',stdout='not-json')):
                with patch.object(future.subprocess,'run',return_value=row) as called:
                    with self.assertRaises((RuntimeError,ValueError)): future.observe_source(helper,lambda code:[],{},sleep=Mock())
                self.assertEqual(called.call_count,1)

    def test_boundary_missing_and_controller_dead_is_dependency_terminal(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name))
            with patch.object(future,'observe_source',return_value={'ready':False}), \
                    patch.object(future,'controller_observation',return_value={'state':'TERMINAL'}), \
                    patch.object(future,'progress') as event:
                with self.assertRaisesRegex(RuntimeError,'DEPENDENCY_TERMINAL'):
                    future.wait_for_boundary(helper,Mock(),{},[],sleep=Mock())
            self.assertFalse(event.call_args.kwargs['training_failure_inferred'])

    def test_boundary_transport_unknown_waits_and_ready_can_follow_completed_controller(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); sleeper = Mock()
            with patch.object(future,'observe_source',side_effect=[{'ready':False,'transport_unknown':True},{'ready':True}]), \
                    patch.object(future,'controller_observation',return_value={'state':'TERMINAL'}) as controller, \
                    patch.object(future,'progress'):
                self.assertTrue(future.wait_for_boundary(helper,Mock(),{},[],sleep=sleeper)['ready'])
            sleeper.assert_called_once(); controller.assert_called_once()

    def test_source_commands_compile_and_dependency_install_is_exact(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name)); future.configure_step(helper,150)
            # Bind the real frozen generator to its original globals, then only
            # the new dependency's public identity is substituted in its output.
            data = dict(raw=b'# new public source\n', commit='b'*40, sha256='c'*64,
                hpc_tool=helper.BASE/'preparation/post-callback-tools'/('b'*40)/'post_callback_dependencies.py')
            for module in (None,data):
                for operation in ('snapshot','emit'):
                    code = future.source_operation(helper,operation,module); ast.parse(code)
                    self.assertIn('150',code)
                    if operation=='snapshot': self.assertIn('module.snapshot(base,150,',code)
            code = future.dependency_install_code(helper,data); ast.parse(code)
            self.assertIn('len(raw)==20', code); self.assertIn(repr(data['sha256']),code)
            self.assertNotIn(repr(helper.MODULE_SHA),code)

    def test_source_loader_really_loads_both_modules_against_original_core_without_torch(self):
        checkout = HERE.parents[2]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); helper = fixture_helper(root)
            for relative in ('reproduction/easy2560_joint4/__init__.py',
                             'reproduction/easy2560_joint4/checkpoint_archive.py',
                             'reproduction/easy2560_joint4/control.py','reproduction/easy2560_joint4/model7b.py'):
                target = helper.BASE/'repo'/relative; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(checkout/relative,target)
            helper.HPC_TOOL = helper.BASE/'preparation/tools/b87/post_callback_archive.py'
            helper.HPC_TOOL.parent.mkdir(parents=True)
            shutil.copyfile(checkout/'reproduction/easy2560_joint4/post_callback_archive.py',helper.HPC_TOOL)
            dependency_raw = (checkout/'reproduction/easy2560_joint4/post_callback_dependencies.py').read_bytes()
            target = helper.BASE/'preparation/tools/current/post_callback_dependencies.py'; target.parent.mkdir(parents=True)
            target.write_bytes(dependency_raw)
            data = dict(raw=dependency_raw,sha256=hashlib.sha256(dependency_raw).hexdigest(),hpc_tool=target)
            code = future.source_loader(helper,data)
            code += ('assert module.STATUS=="EASY2560_POST_CALLBACK_DEPENDENCIES_SHA_FSYNC_PASS"\n'
                'assert Path(module.boundary.__file__).resolve()==Path(%r)\n'
                'assert Path(module.core.__file__).resolve()==Path(%r)\n'
                'assert "torch" not in sys.modules\nprint("DUAL_SOURCE_LOADER_CPU_PASS")\n' %
                (str(helper.HPC_TOOL),str(helper.BASE/'repo/reproduction/easy2560_joint4/checkpoint_archive.py')))
            result = subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'DUAL_SOURCE_LOADER_CPU_PASS')

    def test_closure_must_be_fresh_and_complete_not_just_receipt_presence(self):
        with tempfile.TemporaryDirectory() as name:
            helper = fixture_helper(Path(name))
            accepted = dict(status='PASS', all_manifest_dependencies_covered=True, fresh_transferred_dependency_sha_checked=True,
                covered_core_sha_ack_verified=True, core_covered_payload_rehashed=False,
                binding=dict(core=dict(base=str(helper.BASE),run_id=helper.RUN,source_commit=helper.TRAINING,step=5,core_archive=str(helper.CORE))))
            dependencies = SimpleNamespace(STATUS='PASS', inspect=Mock(return_value=accepted))
            self.assertEqual(future.inspect_closure(helper,dependencies), accepted)
            for key in ('all_manifest_dependencies_covered','fresh_transferred_dependency_sha_checked','covered_core_sha_ack_verified'):
                wrong = copy.deepcopy(accepted); wrong[key] = False; dependencies.inspect.return_value = wrong
                with self.assertRaisesRegex(ValueError,'CLOSURE'): future.inspect_closure(helper,dependencies)

    def test_actual_dependency_inspector_nested_core_binding_and_observer(self):
        from reproduction.easy2560_joint4 import post_callback_dependencies as dependencies
        from reproduction.easy2560_joint4 import test_post_callback_dependencies as dependency_fixtures
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            base, archive, _, saved, _ = dependency_fixtures.DependencyArchiveTests().prepared(
                root, step=5, covered=True, restored=True)
            stream = io.BytesIO(); dependencies.emit(saved['snapshot'], stream); stream.seek(0)
            stage = dependencies.receive(archive, stream)['stage']; dependencies.accept(archive, stage)
            helper = SimpleNamespace(BASE=base, RUN='b'*32, TRAINING='c'*40, STEP=5, CORE=archive)
            # This is the real inspector and real accepted receipts, including
            # all provenance and payload SHA checks, not a fabricated schema.
            actual = future.inspect_closure(helper, dependencies)
            self.assertEqual(actual['binding']['core']['core_archive'], str(archive))
            self.assertNotIn('base', actual['binding'])
            self.assertIn('boundary_source_snapshot_sha256', actual['binding'])
            control = root/'control'; control.mkdir()
            identity = dict(base=str(base), run_id=helper.RUN, source_job='fixture-job',
                            training_commit=helper.TRAINING, birth={'pid':123})
            for filename in ('claim.json', 'launch-receipt.json'):
                (control/filename).write_text(json.dumps(identity))
            runner = SimpleNamespace(process_observation=lambda birth:dict(state='LIVE'))
            with patch.multiple(observer, ROOT=control, HK=root/'HK', BASE=str(base),
                                RUN=helper.RUN, JOB='fixture-job', TRAINING=helper.TRAINING), \
                    patch.object(observer.socket, 'gethostname', return_value='xielab3'), \
                    patch.object(observer.os, 'getuid', return_value=1002), \
                    patch.object(observer, 'load_runner', return_value=runner):
                observed = observer.observe()
                self.assertTrue(observed['milestones']['5']['boundary_accepted'])
                self.assertTrue(observed['milestones']['5']['dependencies_accepted'])
                self.assertEqual(observed['state'], 'RUNNING')
                receipt_path = archive/'post-callback-dependencies/dependency-archive-receipt.json'
                receipt = json.loads(receipt_path.read_text())
                receipt['binding']['core']['source_commit'] = 'f'*40
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'MILESTONE_ARCHIVE_IDENTITY_DIFFERS'):
                    observer.observe()

    def test_emit_must_exit_zero_before_accept_writer_never_retries(self):
        for rc in (0, 7):
            with self.subTest(rc=rc), tempfile.TemporaryDirectory() as name:
                helper = fixture_helper(Path(name)); helper.CONTROL.mkdir()
                helper.checked_json = Mock(return_value=dict(file_count=8,phase=phase(5)))
                events = []
                child = Mock(stdout=io.BytesIO(b'fixture stream'))
                child.wait.side_effect = lambda **kwargs: events.append('wait') or rc
                child.poll.return_value = rc
                accepted = dict(status='PASS',actual_cuda_restore_accepted=False)
                module = SimpleNamespace(STATUS='PASS', phase=phase,
                    receive=Mock(return_value={'stage':name+'/partial-preserved'}),
                    accept=Mock(side_effect=lambda *args: events.append('accept') or accepted))
                with patch.object(future.subprocess,'Popen',return_value=child) as start:
                    if rc:
                        with self.assertRaisesRegex(ValueError,'EMIT_EXIT_NONZERO_7'):
                            future.transfer_once(helper,lambda code:['ssh',code],{},module,'boundary')
                        module.accept.assert_not_called(); self.assertEqual(events,['wait'])
                    else:
                        self.assertEqual(future.transfer_once(helper,lambda code:['ssh',code],{},module,'boundary'),accepted)
                        self.assertEqual(events,['wait','accept'])
                self.assertEqual(start.call_count,1)
                self.assertEqual(json.loads((helper.CONTROL/'boundary-transfer.json').read_text())['emit_returncode'],rc)

    def _companion_fixture(self, reuse=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); helper = fixture_helper(root)
            helper.bootstrap = Mock(return_value=lambda code:[])
            helper.install_immutable = Mock()
            boundary = SimpleNamespace(STATUS='BOUNDARY', phase=phase)
            helper.load_module = Mock(return_value=boundary)
            data = dict(raw=b'dependencies',commit='c'*40,sha256='d'*64)
            if reuse:
                accepted = helper.CORE/'post-callback-dependencies/dependency-archive-receipt.json'
                accepted.parent.mkdir(parents=True); accepted.write_text('{"status":"fixture accepted"}')
            calls = []
            def transferred(instance, ssh, env, module, what, dependencies=None):
                calls.append((instance.STEP,what)); return dict(status='PASS',actual_cuda_restore_accepted=False)
            with patch.multiple(future, ROOT=root/'future', HK=root/'hk'), \
                    patch.object(future.socket,'gethostname',return_value='xielab3'), \
                    patch.object(future.os,'getuid',return_value=1002), \
                    patch.object(future,'current_birth',return_value={'pid':123,'uid':1002,'start_ticks':7,'boot_id':'fixture'}), \
                    patch.object(future,'frozen_helper',return_value=helper), \
                    patch.object(future,'validate_envelope',return_value=('fixture-secret',b'boundary',data)), \
                    patch.object(future,'install_dependency'), patch.object(future,'load_dependencies',return_value=object()), \
                    patch.object(future,'verify_completed_five',return_value={'step':5,'writer_replayed':False}), \
                    patch.object(future,'install_source_tools',return_value=[]), \
                    patch.object(future,'wait_for_core',return_value=({'actual_ACK':True},{'state':'LIVE'})), \
                    patch.object(future,'observe_source',return_value={'ready':True}), \
                    patch.object(future,'transfer_once',side_effect=transferred), \
                    patch.object(future,'inspect_closure',return_value={'closed':True}), patch.object(future,'progress'):
                result = future.run({'password':'fixture-secret'})
                self.assertEqual(result['completed_steps'],[50,100,150,200])
                expected = ([] if reuse else [(5,'dependencies')])+[(step,what) for step in future.STEPS for what in ('boundary','dependencies')]
                self.assertEqual(calls,expected)
                result5 = json.loads((root/'future/qualification-global_step_5/result.json').read_text())
                self.assertEqual(result5['accepted_dependency_archive_reused_read_only'],reuse)
                with self.assertRaisesRegex(ValueError,'ONCE_CLAIM_EXISTS_NO_REPLAY'): future.run({})
                self.assertEqual(len(calls),8 if reuse else 9)

    def test_companion_qualifies_only_new_cp5_dependencies_then_future_both_in_order(self):
        self._companion_fixture()

    def test_accepted_cp5_dependency_child_is_inspected_reused_and_never_retransferred(self):
        self._companion_fixture(reuse=True)

    def test_observer_loads_published_runner_without_file_global_for_python_c(self):
        with tempfile.TemporaryDirectory() as name:
            hk = Path(name).resolve(); commit = 'c'*40; folder = hk/'post-callback-future-launch-tools'/commit; folder.mkdir(parents=True)
            path = folder/'archive_post_callback_future_hk.py'; shutil.copyfile(HERE/path.name,path)
            launch = dict(launcher_commit=commit,tools=str(folder),public_code={path.name:dict(path=str(path),
                bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())})
            # The function no longer uses __file__, matching python -c observation.
            with patch.object(observer,'HK',hk), patch.dict(observer.__dict__,{'__file__':None}):
                self.assertEqual(observer.load_runner(launch).STEPS,(50,100,150,200))
                launch['public_code'][path.name]['sha256'] = 'f'*64
                with self.assertRaisesRegex(ValueError,'SOURCE_DIFFERS'): observer.load_runner(launch)


if __name__ == '__main__': unittest.main()
