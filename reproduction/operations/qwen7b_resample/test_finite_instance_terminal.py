"""CPU fixtures: real small NAS metadata/lock files and fake native processes."""
import contextlib
import copy
import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import finite_instance_terminal as observer
from . import finite_instance_terminal_control as control


def put(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FiniteInstanceTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);view = observer.View(root / 'nas');view.root.mkdir()
            selection = dict(schema_version=1,run_id=observer.RUN_ID,key='source:gpu2-tp1',node='h100-5',
                unit='sst-q7b-g5-fixture-gpu2-tp1-r1-'+observer.RUN_ID[:12]+'.service',instance='r1',slot='gpu2-tp1',
                directory='retrieval-generation-fixture-gpu2-tp1-r1',source_bundle_sha256='f'*64,
                owner_unit='fixture-owner.service',registry_sha256=observer.REGISTRY_SHA,renewal_plan_sha256=observer.RENEWAL_SHA,
                model_indices=[2],tensor_parallel_size=1)
            selection['submit_intent']=str(observer.BASE/'preparation'/selection['directory']/'submit-intent.json')
            unit=selection['unit'];group='/system.slice/'+unit;pid=9001;child=9002
            owner=dict(pid=9010,birth='1122',unit=selection['owner_unit'])
            ready=observer.BASE/'owner/owner-ready.json';put(view.path(ready),dict(owner=owner))
            ready_sha=view.metadata(ready)['sha256']
            evidence=observer.BASE/'launch/evidence.json'
            put(view.path(evidence),dict(main_pid=pid,unit=unit,cgroup=group,node=selection['node']))
            intent=dict(unit=unit,manifest=dict(unit=unit,role='slot',instance='r1',slot=selection['slot'],source_bundle_sha256='f'*64),
                context=dict(owner_ready_sha256=ready_sha))
            put(view.path(selection['submit_intent']),intent)
            job=observer.BASE/'pool/jobs'/('job-source-slot-gpu2-tp1-%d'%pid)/'attempt-fixture'
            lease=observer.BASE/'pool/gpu-slot-leases'/observer.RUN_ID/'9010-1122/gpu2-tp1/GPU-2.lock'
            view.path(lease).parent.mkdir(parents=True);view.path(lease).write_bytes(b'original model lease')
            primary=observer.BASE/'pool/gpu-locks/GPU-2.lock';view.path(primary).parent.mkdir(parents=True);view.path(primary).write_bytes(b'borrowed primary must stay unchanged')
            allocation=dict(run_id=observer.RUN_ID,node=selection['node'],native_unit=unit,native_cgroup=group,controller_pid=pid,
                source_owner=owner,model_indices=[2],tensor_parallel_size=1,allocation_evidence=str(evidence),
                allocation_evidence_sha256=view.metadata(evidence)['sha256'],source_owner_ready=str(ready),source_owner_ready_sha256=ready_sha,
                slot_lease_paths={'GPU-2':str(lease)},local_relay_process=dict(pid=9011),job_id='source-slot-gpu2-tp1-'+str(pid))
            put(view.path(job/'allocation.json'),allocation);put(view.path(observer.BASE/'config.json'),dict(run_id=observer.RUN_ID))
            proc=root/'proc';(proc/'sys/kernel/random').mkdir(parents=True)
            (proc/'sys/kernel/random/boot_id').write_text('11111111-2222-3333-4444-555555555555')
            live={str(pid):'2222',str(child):'3333'};state=['active'];clock=[1000.0]
            def process(number):
                if str(number)not in live:return None
                return dict(pid=number,birth=live[str(number)],state='R',cgroup='0::'+group+'\n')
            def native(unit_name):
                self.assertEqual(unit_name,unit)
                props=dict(Id=unit,LoadState='loaded',MainPID=str(pid)if state[0]=='active'else'0',ActiveState=state[0],
                    ControlGroup=group if state[0]=='active'else'',Result='success',ExecMainCode='0'if state[0]=='active'else'1',
                    ExecMainStatus='0',InvocationID='a'*32)
                if state[0]=='purged':props.update(LoadState='not-found',ActiveState='inactive',ExecMainCode='0',InvocationID='0'*32)
                if state[0]=='failed':props.update(Result='exit-code',ExecMainStatus='1')
                return dict(returncode=0,properties=props,stderr='',epoch=clock[0])
            def finish(purged=False):
                counts=dict(status='POOL_BATCH_BOUNDARY_TIME_LIMIT',run_id=observer.RUN_ID,job_id=allocation['job_id'],
                    completed_own_batches=3,questions=384,trajectories=1920,global_goal_complete=False)
                put(view.path(job/'sampler-exit.json'),counts)
                put(view.path(job/'complete.json'),dict(counts,controller_status='OWN_RELAY_CONSUMER_FINISHED',
                    sampler_returncode=0,own_children_reaped=True,foreign_processes_signalled=False,upstream_retriever_signalled=False))
                state[0]='purged'if purged else'inactive';live.clear()
            with patch.object(observer,'PROC',proc),patch.object(observer,'CGROUP',root/'cgroup'), \
                 patch.object(observer,'query',side_effect=native),patch.object(observer,'process',side_effect=process), \
                 patch.object(observer,'members',side_effect=lambda _:dict(live)),patch.object(observer.os,'geteuid',return_value=0), \
                 patch.object(observer.socket,'gethostname',return_value='h100-5'),patch.object(observer.time,'time',side_effect=lambda:clock[0]), \
                 patch.object(observer.time,'sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)), \
                 patch.object(observer,'journal',return_value=dict(available=False,required_for_pass=False)):
                yield dict(view=view,selection=selection,job=job,lease=lease,primary=primary,finish=finish,state=state,live=live,
                    capture=observer.root_for(selection)/'capture.json',proof=observer.root_for(selection)/'complete.json')

    def capture(self,f):
        value=observer.operate('capture',f['selection'],f['view'],'b'*64)
        self.assertEqual(value['status'],'FINITE_INSTANCE_CAPTURED_ACTIVE')
        record=f['view'].metadata(f['capture']);binding=record['value']['binding']
        self.assertEqual(binding['original_owned_births'],{'9001':'2222','9002':'3333'})
        self.assertEqual(binding['borrowed_processes'],[9010,9011])
        return record

    def test_active_capture_and_collect_pending_leave_original_files_untouched(self):
        with self.fixture()as f:
            files=[f['job']/'allocation.json',f['selection']['submit_intent'],f['lease'],f['primary']]
            before={str(p):f['view'].path(p).read_bytes()for p in files}
            self.capture(f)
            paths=set(f['view'].path(observer.root_for(f['selection'])).iterdir())
            with patch.object(observer,'release')as release:
                value=observer.operate('collect',f['selection'],f['view'],'b'*64)
            self.assertEqual(value['status'],'PENDING_MODEL_ACTIVE');release.assert_not_called()
            self.assertEqual(paths,set(f['view'].path(observer.root_for(f['selection'])).iterdir()))
            self.assertFalse(f['view'].path(f['proof']).exists())
            self.assertEqual(before,{str(p):f['view'].path(p).read_bytes()for p in files})

    def test_actual_application_complete_two_releases_and_exact_cache(self):
        with self.fixture()as f:
            self.capture(f);f['finish']()
            value=observer.operate('collect',f['selection'],f['view'],'b'*64)
            self.assertEqual(value['status'],observer.STATUS);self.assertEqual(value['native_exit_status'],0)
            proof=f['view'].metadata(f['proof'])['value'];observations=proof['release_observations']
            self.assertEqual(len(observations),2);self.assertGreaterEqual(observations[1]['epoch']-observations[0]['epoch'],3)
            self.assertTrue(all(observer.released(row)for row in observations))
            self.assertEqual(proof['binding']['controller_birth'],'2222')
            self.assertFalse(proof['signals_sent']);self.assertFalse(proof['primary_gpu_locks_modified'])
            cached=observer.operate('collect',f['selection'],f['view'],'b'*64)
            self.assertTrue(cached['cached_exact_binding_validated']);self.assertTrue(observer.released(cached['fresh_release_observation']))
            self.assertEqual(f['view'].path(f['primary']).read_bytes(),b'borrowed primary must stay unchanged')

    def test_purged_native_remains_unknown_without_journal_gate(self):
        with self.fixture()as f:
            self.capture(f);f['finish'](purged=True)
            value=observer.operate('collect',f['selection'],f['view'],'b'*64)
            self.assertEqual(value['status'],observer.STATUS);self.assertIsNone(value['native_exit_status'])
            proof=f['view'].metadata(f['proof'])['value']
            self.assertEqual(proof['native']['properties']['LoadState'],'not-found')
            self.assertIsNone(proof['native_exit_status']);self.assertFalse(proof['same_invocation_journal']['required_for_pass'])

    def test_wrong_instance_birth_or_release_cache_is_rejected(self):
        with self.fixture()as f:
            self.capture(f);f['finish']()
            self.assertEqual(observer.operate('collect',f['selection'],f['view'],'b'*64)['status'],observer.STATUS)
            good=f['view'].metadata(f['proof'])['value']
            wrong=copy.deepcopy(good);wrong['unit']='retired-consumer10-v5.service'
            wrong_birth=copy.deepcopy(good);wrong_birth['binding']['controller_birth']='99999'
            wrong_release=copy.deepcopy(good);wrong_release['release_observations'][0]['slot_leases_acquired']=[]
            wrong_native=copy.deepcopy(good);wrong_native['native']['properties']['Id']='retired-consumer10-v5.service'
            for value in (wrong,wrong_birth,wrong_release,wrong_native):
                with self.subTest(error=value['unit']):
                    put(f['view'].path(f['proof']),value)
                    result=observer.operate('collect',f['selection'],f['view'],'b'*64)
                    self.assertEqual(result['status'],'ROOT_ATTENTION_FINITE_INSTANCE_EVIDENCE_INVALID')

    def test_failed_incomplete_and_still_held_slot_never_pass(self):
        with self.fixture()as f:
            self.capture(f);f['state'][0]='failed';f['live'].clear()
            self.assertEqual(observer.operate('collect',f['selection'],f['view'],'b'*64)['status'],'ROOT_ATTENTION_NATIVE_MODEL_FAILED')
            f['state'][0]='inactive'
            self.assertEqual(observer.operate('collect',f['selection'],f['view'],'b'*64)['status'],'ROOT_ATTENTION_FINITE_INSTANCE_EVIDENCE_INVALID')
            f['finish']()
            with patch.object(observer.fcntl,'lockf',side_effect=BlockingIOError(errno.EAGAIN,'original slot still held')):
                result=observer.operate('collect',f['selection'],f['view'],'b'*64)
            self.assertEqual(result['status'],'PENDING_MODEL_RESOURCE_RELEASE');self.assertFalse(f['view'].path(f['proof']).exists())

    def test_control_selection_and_bundle_are_exact_small_observer_only_sources(self):
        import ast
        identity,files,manifest=control.bundle();ast.parse(control.REMOTE)
        self.assertEqual(len(identity),64);self.assertLess(sum(map(len,files.values())),64*1024)
        for name,raw in files.items():self.assertEqual(manifest['files'][name],dict(bytes=len(raw),sha256=observer.digest(raw)))
        with tempfile.TemporaryDirectory()as folder:
            with self.fixture()as f:
                selected=f['selection'];model=dict(key=selected['key'],role='model',node=selected['node'],unit=selected['unit'],
                    directory=selected['directory'],source_bundle_sha256=selected['source_bundle_sha256'],slot=selected['slot'])
                registry=dict(registry=[model,dict(key='source:owner',role='owner',node='h100-5',unit=selected['owner_unit'])])
                renewal=dict(rows=[dict(key=selected['key'],owner_registry_key='source:owner',model_specification=dict(indices=[2],tp=1),
                    current_r1=dict(unit=selected['unit'],directory=selected['directory'],frozen_source_bundle_sha256=selected['source_bundle_sha256'],submit_intent=selected['submit_intent']))])
                r=Path(folder)/'registry.json';p=Path(folder)/'plan.json';put(r,registry);put(p,renewal)
                with patch.object(observer,'REGISTRY_SHA',observer.digest(r.read_bytes())),patch.object(observer,'RENEWAL_SHA',observer.digest(p.read_bytes())):
                    actual=control.selection(selected['key'],r,p)
                    self.assertEqual(actual['unit'],selected['unit'])
                    with self.assertRaises(observer.EvidenceError):control.selection('source:owner',r,p)
                    with p.open('a')as stream:stream.write(' ')
                    with self.assertRaisesRegex(observer.EvidenceError,'SHA differs'):control.selection(selected['key'],r,p)


if __name__=='__main__':unittest.main()
