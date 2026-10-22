import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

from reproduction.operations.easy7b_20261021 import fa3_archive_consumers_hk as m
from reproduction.easy2560_joint4 import archive_login_route as route

BASE=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022')
TRAINING='a'*40
CONTROLLER='b'*40
JOB='746999'


def fake_future(root):
    locations={name:root/name for name in ('bundle','core','companion')};calls=[]
    boundary=root/'boundary.py';boundary.write_text('boundary')
    prefix=root/'reproduction/easy2560_joint4';prefix.mkdir(parents=True)
    (prefix/'post_callback_dependencies.py').write_text('dependencies')
    helper=types.SimpleNamespace(HK_TOOL=boundary,MODULE_COMMIT='c'*40,MODULE_SHA='d'*64,
        write_json=m.publish,canonical=Path,redact=lambda value,password:str(value).replace(password,'[REDACTED]'))
    helper.bootstrap=lambda:Mock();helper.install_immutable=lambda *args:None
    helper.remote_install_code=lambda:"assert os.getuid()==14090 and socket.gethostname()=='ACD-Login-1'\n"
    helper.load_module=lambda:types.SimpleNamespace(phase=lambda step:{'step':step,'pre_final51713':step==200})
    helper.ready_core=lambda:{'step':helper.STEP,'actual_ack':True}
    future=types.SimpleNamespace(frozen_helper=lambda:helper,controller_observation=lambda current:{'state':'LIVE','records':{}},
        install_dependency=lambda *args:None,load_dependencies=lambda *args:object(),
        verify_completed_five=lambda *args:{'step':5,'skipped':True,'writer_replayed':False})
    future.validate_envelope=lambda h,e:('secret',b'boundary',{'commit':CONTROLLER,'sha256':'d'*64,'raw':b'dependencies'})
    def configure(h,step):
        h.STEP=step;h.CONTROL=future.ROOT/('global_step_'+str(step));h.CORE=root/'checkpoints'/str(step)
    future.configure_step=configure
    future.wait_for_core=lambda h,*args:(h.ready_core(),{'state':'LIVE'})
    future.wait_for_boundary=lambda h,*args:calls.append(('boundary-ready',h.STEP))
    future.install_source_tools=lambda h,*args:calls.append(('tools',h.STEP)) or []
    future.progress=lambda *args,**kwargs:None
    future.transfer_once=lambda h,ssh,env,module,name,*args:calls.append((name,h.STEP)) or {'accepted':True}
    future.inspect_closure=lambda *args:{'covered':True}
    return future,helper,locations,calls


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        # The real module is pure CPU transport construction. Temporary source
        # fixtures otherwise omit published code files unrelated to these tests.
        self.route_patch=patch.object(m,'route_module',return_value=route)
        self.route_patch.start();self.addCleanup(self.route_patch.stop)

    def test_future_only50_to200_and_no_cp5_writer_or_pending_job_source_install(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();future,helper,locations,calls=fake_future(root)
            with patch.object(m,'source',return_value=(root,{})),patch.object(m,'future_module',return_value=future),\
                 patch.object(m,'paths',return_value=locations),patch.object(m.socket,'gethostname',return_value='xielab3'),\
                 patch.object(m.os,'getuid',return_value=1002),patch.object(m,'birth',return_value={'pid':1}):
                result=m.run_companion(BASE,TRAINING,JOB,'secret',CONTROLLER,poll_seconds=1)
            self.assertEqual(result['completed_steps'],[50,100,150,200]);self.assertFalse(result['historical_CP5_replayed'])
            self.assertEqual([step for kind,step in calls if kind=='tools'],[50])
            self.assertEqual([step for kind,step in calls if kind=='boundary'],[50,100,150,200])
            self.assertEqual([step for kind,step in calls if kind=='dependencies'],[50,100,150,200])
            self.assertFalse((locations['companion']/'global_step_5').exists())
            self.assertTrue((locations['companion']/'historical-CP5-read-only.json').is_file())
            self.assertEqual(helper.BASE,BASE);self.assertEqual(helper.JOB,JOB);self.assertEqual(helper.TRAINING,TRAINING)
            for path in root.rglob('*.json'):self.assertNotIn('secret',path.read_text())

    def test_dependency_failure_preserved_and_no_writer_replayed(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();future,helper,locations,calls=fake_future(root)
            future.wait_for_core=Mock(side_effect=RuntimeError('DEPENDENCY_TERMINAL actual core failed'))
            with patch.object(m,'source',return_value=(root,{})),patch.object(m,'future_module',return_value=future),\
                 patch.object(m,'paths',return_value=locations),patch.object(m.socket,'gethostname',return_value='xielab3'),\
                 patch.object(m.os,'getuid',return_value=1002),patch.object(m,'birth',return_value={'pid':1}):
                with self.assertRaisesRegex(RuntimeError,'FAILED_INSPECT'):m.run_companion(BASE,TRAINING,JOB,'secret',CONTROLLER)
                with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):m.run_companion(BASE,TRAINING,JOB,'secret',CONTROLLER)
            failure=json.loads((locations['companion']/'failure.json').read_text())
            self.assertEqual(failure['status'],'DEPENDENCY_TERMINAL');self.assertFalse(failure['training_failure_inferred'])
            self.assertEqual(calls,[])

    def test_only_valid_completed_core200_is_nonterminal_pending_boundary(self):
        future=types.SimpleNamespace(controller_observation=lambda helper:state,configure_step=lambda h,s:setattr(h,'STEP',s))
        helper=types.SimpleNamespace(MODULE_COMMIT='c'*40,ready_core=Mock(return_value={'real_ack':True}))
        m.configure(future,helper,BASE,TRAINING,JOB)
        result=dict(status='EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS',completed_steps=list(m.STEPS),
                    base=str(BASE),source_job=JOB,source_commit=TRAINING)
        state={'state':'TERMINAL','records':{'result.json':result}}
        helper.STEP=200;self.assertEqual(future.controller_observation(helper)['state'],'COMPLETE')
        helper.ready_core.assert_called_once()
        state['records']['failure.json']={'error':'actual failure'}
        self.assertEqual(future.controller_observation(helper)['state'],'TERMINAL')
        del state['records']['failure.json'];helper.STEP=150
        self.assertEqual(future.controller_observation(helper)['state'],'TERMINAL')
        helper.STEP=200;result['source_commit']='f'*40
        self.assertEqual(future.controller_observation(helper)['state'],'TERMINAL')

    def test_launch_separates_runtime_and_consumer_commits_and_first_step50(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();locations={key:root/key for key in ('bundle','core','companion')}
            with patch.object(m,'source',return_value=(root,{})),patch.object(m,'paths',return_value=locations),\
                 patch.object(m.socket,'gethostname',return_value='xielab3'),patch.object(m.os,'getuid',return_value=1002),\
                 patch.object(m,'_detach',side_effect=lambda *args:{'birth':{'pid':123}}) as detached:
                result=m.launch(BASE,TRAINING,JOB,'secret',CONTROLLER)
                with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED'):m.launch(BASE,TRAINING,JOB,'secret',CONTROLLER)
            commands=[call.args[0] for call in detached.call_args_list]
            self.assertEqual(detached.call_count,2);self.assertIn('--first-step',commands[0]);self.assertEqual(commands[0][-1],'50')
            self.assertEqual(commands[0][commands[0].index('--commit')+1],TRAINING)
            self.assertEqual(commands[1][commands[1].index('--controller-commit')+1],CONTROLLER)
            self.assertEqual(result['first_step'],50);self.assertFalse(result['historical_CP5_replayed'])
            for path in root.rglob('*.json'):self.assertNotIn('secret',path.read_text())

    def test_detach_fsyncs_public_receipt_before_delivering_ram_password(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();receipt=root/'launch.json';writes=[]
            child=types.SimpleNamespace(pid=1,stdin=types.SimpleNamespace(write=lambda data:writes.append((receipt.exists(),data)),close=lambda:None))
            with patch.object(m.subprocess,'Popen',return_value=child),patch.object(m,'birth',return_value={'pid':1}):
                m._detach(['python3','-B','public.py'],root,{},'secret',receipt,root/'stdout.log',{'public':True})
            self.assertTrue(writes[0][0]);self.assertEqual(json.loads(writes[0][1]),{'password':'secret'})
            self.assertNotIn('secret',receipt.read_text())

    def test_explicit_login2_pin_reaches_both_new_children_and_public_claims(self):
        identity=dict(hostname='ACD-Login-2',boot_id='3c637b25-17cf-404c-971b-30bc4c4e9fbb',
                      pid_namespace='pid:[4026531836]')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();locations={key:root/key for key in ('bundle','core','companion')}
            with patch.object(m,'source',return_value=(root,{})),patch.object(m,'paths',return_value=locations),\
                 patch.object(m.socket,'gethostname',return_value='xielab3'),patch.object(m.os,'getuid',return_value=1002),\
                 patch.object(m,'_detach',return_value={'birth':{'pid':123}}) as detached:
                result=m.launch(BASE,TRAINING,JOB,'secret',CONTROLLER,
                    login_host='10.120.48.32',login_identity_json=json.dumps(identity))
            for call in detached.call_args_list:
                command=call.args[0]
                self.assertEqual(command[command.index('--login-host')+1],'10.120.48.32')
                self.assertEqual(json.loads(command[command.index('--login-identity-json')+1]),identity)
                self.assertNotIn('SSHPASS',call.args[2])
            self.assertEqual(result['source_ssh_route']['hpc_identity'],identity)
            self.assertFalse(result['source_ssh_route']['automatic_fallback'])
            claim=json.loads((locations['bundle']/'launch-claim.json').read_text())
            self.assertEqual(claim['source_ssh_route'],result['source_ssh_route'])

    def test_companion_uses_same_login2_for_every_source_phase_without_mutating_frozen_transport(self):
        identity=dict(hostname='ACD-Login-2',boot_id='3c637b25-17cf-404c-971b-30bc4c4e9fbb',
                      pid_namespace='pid:[4026531836]')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();future,helper,locations,calls=fake_future(root)
            frozen_transport=Mock();helper.bootstrap=Mock(return_value=frozen_transport)
            original_installer=helper.remote_install_code;actual=[]
            def seen(h,transport,*args):
                actual.append(transport('read-only-fixture-command'));return []
            future.install_source_tools=seen
            future.wait_for_boundary=lambda h,transport,*args:seen(h,transport)
            future.transfer_once=lambda h,transport,*args:seen(h,transport) or {'accepted':True}
            with patch.object(m,'source',return_value=(root,{})),patch.object(m,'future_module',return_value=future),\
                 patch.object(m,'paths',return_value=locations),patch.object(m.socket,'gethostname',return_value='xielab3'),\
                 patch.object(m.os,'getuid',return_value=1002),patch.object(m,'birth',return_value={'pid':1}):
                result=m.run_companion(BASE,TRAINING,JOB,'secret',CONTROLLER,
                    login_host='10.120.48.32',login_identity_json=json.dumps(identity))
            self.assertEqual(len(actual),13) # tool install, 4 observations, 8 transfers
            for argv in actual:
                self.assertEqual(argv[-2],'jxu957@10.120.48.32')
                self.assertIn(identity['boot_id'],argv[-1])
            helper.bootstrap.assert_called_once();frozen_transport.assert_not_called()
            self.assertIn("socket.gethostname()=='ACD-Login-2'",helper.remote_install_code())
            self.assertIn("socket.gethostname()=='ACD-Login-1'",original_installer())
            self.assertEqual(result['completed_steps'],[50,100,150,200])

    def test_missing_alternate_pin_rejected_before_launch_claim_or_child(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve()
            with patch.object(m,'source',return_value=(root,{})),\
                 patch.object(m.socket,'gethostname',return_value='xielab3'),patch.object(m.os,'getuid',return_value=1002),\
                 patch.object(m,'paths') as paths,patch.object(m,'_detach') as detached:
                with self.assertRaisesRegex(ValueError,'REQUIRES_EXPLICIT_IDENTITY'):
                    m.launch(BASE,TRAINING,JOB,'secret',CONTROLLER,login_host='10.120.48.32')
            paths.assert_not_called();detached.assert_not_called();self.assertEqual(list(root.iterdir()),[])

    def test_old_job_rejected_without_touching_old_consumers(self):
        with self.assertRaisesRegex(ValueError,'NEW_JOB_REQUIRED'):m.arguments(BASE,TRAINING,'745371')


if __name__=='__main__':unittest.main()
