"""CPU-only ownership, capacity, admission-release and borrowed cleanup checks."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('relay_tp1_under_test', DIRECTORY / 'relay_tp1_worker.py')
worker = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):
    SPEC.loader.exec_module(worker)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(base, slot='gpu1'):
    owner = dict(worker.OWNERS[worker.SLOTS[slot]])
    owner.update(cgroup='0::/system.slice/' + owner['unit'] + '\n', netns_inode=33,
                 allocation_path=str(base / 'owner-allocation.json'), allocation_sha256='a' * 64, deadline_epoch=10000)
    relay_owner = dict(worker.OWNERS[1], cgroup='0::/system.slice/' + worker.V5_UNIT + '\n',
                       netns_inode=33, deadline_epoch=10000)
    return dict(schema_version=1, mode='NODE10_FINITE_TP1_CHILD_LEASE', run_id=worker.RUN_ID,
        slot=slot, unit='sst-q7b-g5-consumer10-tp1-' + slot + '-v1-' + worker.RUN_ID[:12] + '.service',
        base=str(base), model_indices=[worker.SLOTS[slot]], model_cuda_uuids=['GPU-test'], tensor_parallel_size=1,
        model_limits=dict(worker.LIMITS), lease_owner=owner, relay_owner=relay_owner,
        relay=dict(pid=worker.RELAY_PID, birth=worker.RELAY_BIRTH, cgroup=relay_owner['cgroup'], netns_inode=33),
        source=dict(source_unit=worker.SOURCE_UNIT, source_main_pid=1490214, source_main_birth='6807965',
                    source_pid=1490465, source_birth='6808057', source_sampling_deadline_epoch=11000),
        source_context=str(base / 'source-context.json'), source_context_sha256='b' * 64,
        primary_gpu_lock=dict(fd=90, device=1, inode=2), model_deadline_epoch=8000, captured_epoch=1000,
        reuse_local_relay_ready=str(base / 'relay-ready.json'), reuse_local_relay_sha256='c' * 64,
        upstream_ready=str(base / 'upstream-ready.json'))


class ConfigurationTests(unittest.TestCase):
    def test_four_finite_slots_have_correct_primary_and_relay_owners(self):
        with tempfile.TemporaryDirectory() as folder:
            for slot in worker.SLOTS:
                value = configuration(Path(folder), slot)
                worker.validate_configuration(value, slot)
                self.assertEqual(value['relay_owner']['pid'], 2371322)
                self.assertEqual(value['lease_owner']['pid'], 2452973 if slot == 'gpu5' else 2371322)
                self.assertNotEqual(value['unit'], value['lease_owner']['unit'])

    def test_rejects_topology_owner_birth_source_and_deadline_drift(self):
        changes = [lambda v: v.update(slot='gpu0'), lambda v: v.update(model_indices=[0]),
            lambda v: v.update(tensor_parallel_size=4), lambda v: v['model_limits'].update(seqs=32),
            lambda v: v['lease_owner'].update(pid=999), lambda v: v['lease_owner'].update(birth='reused'),
            lambda v: v['relay'].update(birth='reused'), lambda v: v['relay_owner'].update(unit='other.service'),
            lambda v: v['source'].update(source_main_birth='reused'),
            lambda v: v.update(model_deadline_epoch=9401), lambda v: v.update(captured_epoch=-7000),
            lambda v: v.update(unit=worker.V5_UNIT)]
        with tempfile.TemporaryDirectory() as folder:
            for number, change in enumerate(changes):
                with self.subTest(case=number):
                    value = configuration(Path(folder));change(value)
                    with self.assertRaises(ValueError):worker.validate_configuration(value, 'gpu1')

    def test_per_card_child_lease_never_uses_or_opens_the_primary_path(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            paths = []
            for slot,index in worker.SLOTS.items():
                value = configuration(base,slot);value['model_cuda_uuids']=['GPU-'+str(index)]
                paths.append(worker.lease_path(base,value))
            self.assertEqual(len(set(paths)),4)
            for slot, path in zip(worker.SLOTS, paths):
                self.assertIn('gpu-subleases', path.parts)
                self.assertNotIn('gpu-locks', path.parts)
                self.assertIn(str(worker.OWNERS[worker.SLOTS[slot]]['pid']), str(path))

    def test_borrowed_dependency_observer_cannot_stop_or_reap(self):
        value = configuration(Path('/tmp/cpu-only'))
        relay = SimpleNamespace(pid=2371352, proof=value['relay'], poll=lambda: None)
        observer = worker.BorrowedDependencies('/tmp/cpu-only', value, relay)
        with patch.object(worker, 'dependencies_live', return_value=True):self.assertIsNone(observer.poll())
        with patch.object(worker, 'dependencies_live', return_value=False):self.assertEqual(observer.poll(), 1)
        for name in ('terminate', 'kill', 'wait', 'send_signal', 'stop'):self.assertFalse(hasattr(observer, name))

    def test_peer_startup_proof_is_not_a_runtime_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);value = configuration(base)
            value['peer_startup_proof'] = dict(pid=999, birth='dead-peer')
            write(base / 'source-context.json', value['source']);value['source_context_sha256'] = sha(base / 'source-context.json')
            with patch.object(worker, 'alive', return_value=True) as alive, \
                 patch.object(worker, 'held_primary', return_value=value['primary_gpu_lock']), \
                 patch.object(worker.time, 'time', return_value=2000):
                self.assertTrue(worker.dependencies_live(base, value))
            self.assertEqual([call.args[0]['pid'] for call in alive.call_args_list], [2371322, 2371322, 2371352])


class PrimaryLockTests(unittest.TestCase):
    def test_fd_inode_and_actual_posix_owner_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);path = base / 'pool/gpu-locks/GPU-test.lock';path.parent.mkdir(parents=True)
            path.write_bytes(b'primary remains unchanged');info = path.stat()
            owner = dict(pid=2371322)
            expected = dict(fd=90, device=info.st_dev, inode=info.st_ino)
            real_stat, real_read = Path.stat, Path.read_text
            def stat(path, *args, **kwargs):
                if str(path) == '/proc/2371322/fd/90':return info
                return real_stat(path, *args, **kwargs)
            for actual_pid in (2371322, 999):
                def read(path, *args, **kwargs):
                    if str(path) == '/proc/2371322/fdinfo/90':
                        return 'lock:\t1: POSIX ADVISORY WRITE %d %x:%x:%d 0 EOF\n' % (
                            actual_pid, os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
                    return real_read(path, *args, **kwargs)
                with patch.object(Path, 'stat', stat), patch.object(Path, 'read_text', read):
                    if actual_pid == 2371322:self.assertEqual(worker.held_primary(base, owner, 'GPU-test', expected), expected)
                    else:
                        with self.assertRaises(ValueError):worker.held_primary(base, owner, 'GPU-test', expected)
            self.assertEqual(path.read_bytes(), b'primary remains unchanged')


class CaptureContextTests(unittest.TestCase):
    def test_actual_relay_and_source_context_field_shapes_bind_gpu5_primary_separately(self):
        """Replay the existing ready/source-context schema, without SSH or CUDA."""
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);value=configuration(base,'gpu5');owner=value['lease_owner'];relay_owner=value['relay_owner']
            owner['deadline_epoch']=9000
            relay=dict(pid=2371352,birth='1958624426',cgroup=relay_owner['cgroup'],netns_inode=33)
            relay_attempt=base/'pool/jobs/job-relay10-2371322/attempt-original'
            root=base/'preparation/relay-consumer10-control-v1'
            # The seven fields are emitted by the original start-upstream action.
            source=dict(source_unit=worker.SOURCE_UNIT,source_main_pid=1490214,source_main_birth='6807965',
                        source_pid=1490465,source_birth='6808057',source_netns_inode=4026542878,
                        source_sampling_deadline_epoch=11000)
            write(root/'source-context.json',source)
            upstream=dict(role='upstream',source_pid=1490465,source_birth='6808057',source_netns_inode=4026542878)
            write(root/'upstream-ready.json',upstream)
            ready=dict(status='RELAY_LISTENING',role='downstream',listen=dict(host='127.0.0.1',port=28831),
                listen_url='http://127.0.0.1:28831',process=relay,listener_netns_inode=33,
                source_pid=1490465,source_birth='6808057',source_netns_inode=4026542878,
                upstream_proof=dict(sha256=sha(root/'upstream-ready.json')),
                code_sha256=sha(DIRECTORY/'retrieval_relay.py'))
            write(relay_attempt/'relay-ready.json',ready)
            allocation=dict(model_indices=[4,5,6,7],model_cuda_uuids=['GPU-4','GPU-test','GPU-6','GPU-7'])
            real_iter,real_symlink,real_read=Path.iterdir,Path.is_symlink,Path.read_text
            def paths(path):
                return iter([Path('/proc/2371352/fd/8')]) if str(path)=='/proc/2371352/fd' else real_iter(path)
            def symlink(path):return True if str(path)=='/proc/2371352/fd/8' else real_symlink(path)
            def text(path,*a,**k):
                if str(path)=='/proc/net/tcp':return 'header\n 0: 0100007F:709F 00000000:0000 0A 0:0 00:0 0 0 0 777\n'
                return real_read(path,*a,**k)
            with patch.object(worker,'owner_snapshot',side_effect=[(owner,allocation,base/'parallel'),(relay_owner,{},relay_attempt)]), \
                 patch.object(worker,'process',return_value=relay),patch.object(worker,'held_primary',return_value=value['primary_gpu_lock']), \
                 patch.object(worker.shared,'load_pool',return_value=SimpleNamespace(nvml=lambda:{'GPU-test':dict(index=5,uuid='GPU-test',free_bytes=23*1024**3)})), \
                 patch.object(worker.time,'time',return_value=1000),patch('os.stat',return_value=SimpleNamespace(st_ino=33)), \
                 patch.object(Path,'iterdir',paths),patch.object(Path,'is_symlink',symlink),patch.object(Path,'read_text',text), \
                 patch('os.readlink',return_value='socket:[777]'):
                captured=worker.capture_context(base,'gpu5')
            self.assertEqual(captured['lease_owner']['pid'],2452973)
            self.assertEqual(captured['relay_owner']['pid'],2371322)
            self.assertEqual(captured['relay']['pid'],2371352)
            self.assertEqual(captured['source'],source)
            self.assertEqual(captured['model_deadline_epoch'],8400)
            self.assertEqual(captured['model_cuda_uuids'],['GPU-test'])


class AdmissionTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, slot='gpu1', probe_uuid='GPU-test'):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);value = configuration(base, slot)
            write(base / 'owner-allocation.json', dict(original_primary=True))
            value['lease_owner']['allocation_sha256'] = sha(base / 'owner-allocation.json')
            write(base / 'relay-ready.json', {});write(base / 'upstream-ready.json', {})
            context_path = base / 'context.json';write(context_path, value)
            evidence_path = base / 'evidence.json';group = '/system.slice/' + value['unit']
            detail = 'Id=%s\nMainPID=314\nActiveState=active\nControlGroup=%s\n' % (value['unit'], group)
            write(evidence_path, dict(node='h100-10', main_pid=314, unit=value['unit'], cgroup=group,
                unit_show=detail, resource_limits=dict(runtime_max_seconds=14400)))
            args = argparse.Namespace(base=str(base), allocation_evidence=str(evidence_path), tp=1,
                model_gpus=str(worker.SLOTS[slot]), model_max_seqs=64, model_batch_tokens=8192,
                model_kv_block_cap=4096, model_reserve_gib=1.0, batch_reserve_seconds=1200,
                reuse_local_relay_ready=str(base / 'relay-ready.json'), reuse_local_relay_sha256=value['reuse_local_relay_sha256'],
                reuse_local_relay_unit=worker.V5_UNIT, upstream_ready=str(base / 'upstream-ready.json'))
            opened = []
            def lock(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600);opened.append(fd);return fd
            pool = SimpleNamespace(read=lambda p: read_json(p), sha=sha, try_lock=lock,
                nvml=lambda: {'GPU-test':dict(index=worker.SLOTS[slot],uuid='GPU-test',free_bytes=23*1024**3)})
            real_stat, real_read = os.stat, Path.read_text
            def stat(path, *args, **kwargs):
                if str(path) in ('/proc/self/ns/mnt', '/proc/1/ns/mnt', '/proc/self/ns/net', '/proc/1/ns/net'):
                    return SimpleNamespace(st_ino=2 if str(path) == '/proc/self/ns/mnt' else 1)
                return real_stat(path, *args, **kwargs)
            def text(path, *args, **kwargs):
                if str(path) == '/proc/self/cgroup':return '0::' + group + '\n'
                return real_read(path, *args, **kwargs)
            with patch.dict(os.environ,Q7B_NODE10_TP1_CONTEXT=str(context_path),Q7B_NODE10_TP1_CONTEXT_SHA256=sha(context_path)), \
                 patch('socket.gethostname', return_value='h100-10'),patch('os.getpid',return_value=314), \
                 patch('os.stat',side_effect=stat),patch.object(Path,'read_text',text), \
                 patch.object(worker,'borrowed_relay',return_value=object()), \
                 patch.object(worker,'dependencies_live',return_value=True),patch.object(worker.time,'time',return_value=2000), \
                 patch.object(worker.shared,'foreign_contexts',return_value=[dict(pid=111,foreign=True)]), \
                 patch.object(worker.subprocess,'check_output',return_value=json.dumps({'actual_cuda_runtime_uuids':[probe_uuid]})):
                yield SimpleNamespace(base=base,value=value,args=args,pool=pool,opened=opened)
            for fd in opened:
                try:os.close(fd)
                except OSError:pass

    def test_admission_holds_only_own_child_lease_and_preserves_foreign(self):
        for slot in ('gpu1','gpu5'):
            with self.subTest(slot=slot),self.fixture(slot) as fixture:
                binding, locks = worker.admit(fixture.pool,fixture.args)
                self.assertEqual(binding['model_indices'],[worker.SLOTS[slot]])
                self.assertEqual(binding['tensor_parallel_size'],1)
                self.assertEqual(binding['lease_owner']['pid'],2452973 if slot=='gpu5' else 2371322)
                self.assertEqual(binding['source_model_deadline_epoch'],8000)
                self.assertEqual(binding['foreign_initial_contexts'],[dict(pid=111,foreign=True)])
                self.assertFalse(binding['primary_owner_owned']);self.assertFalse(binding['local_relay_owned'])
                self.assertEqual(len(locks),1)
                self.assertEqual(Path(binding['gpu_sublease_paths']['GPU-test']),worker.lease_path(fixture.base,fixture.value))
                self.assertFalse((fixture.base/'pool/gpu-locks').exists())

    def test_failed_actual_uuid_check_closes_only_its_own_fd(self):
        with self.fixture(probe_uuid='GPU-wrong') as fixture:
            with self.assertRaisesRegex(ValueError,'Actual CUDA UUID'):worker.admit(fixture.pool,fixture.args)
            self.assertEqual(len(fixture.opened),1)
            with self.assertRaises(OSError):os.fstat(fixture.opened[0])

    def test_dead_owner_or_bad_arguments_prevent_admission(self):
        with self.fixture() as fixture:
            fixture.args.model_kv_block_cap=8192
            with self.assertRaises(ValueError):worker.admit(fixture.pool,fixture.args)
            self.assertEqual(fixture.opened,[])
        with self.fixture() as fixture,patch.object(worker,'dependencies_live',return_value=False):
            with self.assertRaisesRegex(ValueError,'Dependency identity'):worker.admit(fixture.pool,fixture.args)
            with self.assertRaises(OSError):os.fstat(fixture.opened[0])


def read_json(path):
    return json.loads(Path(path).read_text())


class ConsumerCleanupTests(unittest.TestCase):
    def test_real_controller_finally_reaps_only_spawned_sampler_not_dependencies(self):
        consumer=worker.consumer
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);evidence=base/'evidence.json';upstream=base/'upstream.json';ready_path=base/'ready.json'
            write(evidence,dict(captured_epoch=1000,resource_limits=dict(runtime_max_seconds=14400)))
            write(upstream,dict(source_pid=1490465));write(ready_path,{})
            binding=dict(job_id='relay10-tp1-314',model_cuda_visible_devices='1',source_model_deadline_epoch=8000)
            args=argparse.Namespace(base=str(base),handoff_marker=None,allocation_evidence=str(evidence),upstream_ready=str(upstream),
                run_seconds=7000,batch_reserve_seconds=1200,reuse_local_relay_ready=str(ready_path),
                pool_worker=str(DIRECTORY/'pool_worker.py'),relay_script=str(DIRECTORY/'retrieval_relay.py'),
                tp=1,model_max_seqs=64,model_batch_tokens=8192,model_kv_block_cap=4096,model_reserve_gib=1.0)
            health=dict(ready=True,pid=1490465,run_id=worker.RUN_ID,gpu_shards=4,logical_flat_shards=64,
                        encoder_dtype='float16',index_dtype='float16',original_global_ids=True,endpoint='http://127.0.0.1:28831/retrieve')
            borrowed=SimpleNamespace(pid=2371352,proof=dict(pid=2371352,process_owned=False),poll=lambda:None)
            stopped=[];fd=os.open(base/'own-child-lease.lock',os.O_CREAT|os.O_RDWR,0o600)
            pool=SimpleNamespace(RUN_ID=worker.RUN_ID,URL=health['endpoint'],configuration=lambda p:None,
                handoff=lambda *a:{},publish=write,read=read_json,sha=sha,stop_group=lambda p:stopped.append(p.pid))
            class Sampler:
                pid=444;returncode=0
                def __init__(self,*a,**kw):write(Path(kw['env']['Q7B_POOL_INTERNAL'])/'sampler-exit.json',dict(new_complete_batches=1))
                def poll(self):return 0
            with patch.object(consumer,'admit',return_value=(binding,[fd])),patch.object(consumer,'borrowed_relay',return_value=borrowed), \
                 patch.object(consumer.subprocess,'Popen',side_effect=Sampler),patch.object(consumer.time,'time',return_value=2000), \
                 patch.object(consumer.time,'sleep'),patch.object(consumer.urllib.request,'urlopen',side_effect=lambda *a,**k:contextlib.closing(io.StringIO(json.dumps(health)))), \
                 patch('builtins.print'):
                consumer.controller(pool,args)
            self.assertEqual(stopped,[444])
            self.assertNotIn(2371352,stopped);self.assertNotIn(2371322,stopped);self.assertNotIn(2452973,stopped)
            with self.assertRaises(OSError):os.fstat(fd)


if __name__=='__main__':unittest.main()
