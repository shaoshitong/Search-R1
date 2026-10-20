"""CPU source-sublease admission coverage using a temporary proof tree."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('source_sublease_under_test', DIRECTORY / 'source_sublease_worker.py')
WORKER = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path):
    SPEC.loader.exec_module(WORKER)


class SourceSubleaseTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, wrong_owner=False, tp=4, wrong_peer_owner=False):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            read = lambda p: json.loads(Path(p).read_text())
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            run = '70188b8e50fd4b45bc420ce58014f46e'
            parent_unit = 'sst-q7b-g5-sharedvram-v6-' + run[:12] + '.service'
            own_unit = 'sst-q7b-g5-source5-sublease-cpu.service'
            parent_group = '0::/system.slice/' + parent_unit + '\n'
            own_group = '/system.slice/' + own_unit
            retrieval = [0, 1, 2, 5]
            model = [2] if tp == 1 else retrieval
            uuids = ['GPU-%d' % i for i in model]
            allocation_path = base / 'parent-allocation.json'
            write(allocation_path, dict(node='h100-5', controller_pid=1490214, native_unit=parent_unit,
                native_cgroup='/system.slice/' + parent_unit, model_indices=[3, 4, 6, 7],
                model_cuda_uuids=['GPU-%d' % i for i in (3, 4, 6, 7)],
                retrieval_indices=retrieval, retrieval_cuda_uuids=['GPU-%d' % i for i in retrieval]))
            proc, ns, fd_stats, original = {}, {}, {}, {}
            parent = dict(pid=1490214, birth='111', unit=parent_unit, cgroup=parent_group,
                netns_inode=901, mntns_inode=902, allocation_path=str(allocation_path), allocation_sha256=sha(allocation_path))
            retriever = dict(pid=1490465, birth='222', cgroup=parent_group, netns_inode=901)
            for proof in (parent, retriever):
                fields = ['S'] + ['0'] * 18 + [proof['birth']]
                proc['/proc/%d/stat' % proof['pid']] = '%d (cpu fixture) ' % proof['pid'] + ' '.join(fields)
                proc['/proc/%d/cgroup' % proof['pid']] = proof['cgroup']
                ns['/proc/%d/ns/net' % proof['pid']] = 901
            ns.update({'/proc/self/ns/net': 901, '/proc/self/ns/mnt': 902, '/proc/1490214/ns/mnt': 902})
            proc['/proc/self/cgroup'] = '0::' + own_group + '\n'
            lock_proofs = {}
            for number, uuid in enumerate(uuids, start=90):
                primary = base / 'pool/gpu-locks' / (uuid + '.lock')
                primary.parent.mkdir(parents=True, exist_ok=True)
                primary.write_bytes(b'original parent lease content')
                original[primary] = primary.read_bytes()
                info = primary.stat()
                lock_proofs[uuid] = dict(fd=number, device=info.st_dev, inode=info.st_ino)
                fd_stats['/proc/1490214/fd/%d' % number] = info
                proc['/proc/1490214/fdinfo/%d' % number] = (
                    'pos:\t0\nlock:\t1: POSIX ADVISORY WRITE %d %x:%x:%d 0 EOF\n' %
                    (999 if wrong_owner else 1490214, os.major(info.st_dev), os.minor(info.st_dev), info.st_ino))
            path = base / 'context.json'
            value = dict(schema_version=1 if tp == 4 else 2, mode='SOURCE5_SAME_RUN_GPU_SUBLEASE', run_id=run,
                model_indices=model, model_cuda_uuids=uuids, parent=parent, retriever=retriever,
                parent_gpu_locks=lock_proofs)
            if tp == 1:
                peer_unit = 'sst-q7b-g5-source5-retrieval-extra-cpu.service'
                peer_group = '0::/system.slice/' + peer_unit + '\n'
                peer_context_path = base / 'peer-context.json'
                write(peer_context_path, dict(run_id=run, parent=parent))
                root_child = base / 'pool/gpu-subleases' / run / '1490214-111' / 'GPU-2.lock'
                root_child.parent.mkdir(parents=True, exist_ok=True)
                root_child.write_bytes(b'original peer root-child lease content')
                original[root_child] = root_child.read_bytes()
                info = root_child.stat()
                fd_stats['/proc/2013773/fd/94'] = info
                proc['/proc/2013773/fdinfo/94'] = (
                    'lock:\t1: POSIX ADVISORY WRITE %d %x:%x:%d 0 EOF\n' %
                    (999 if wrong_peer_owner else 2013773, os.major(info.st_dev), os.minor(info.st_dev), info.st_ino))
                proc['/proc/2013773/stat'] = '2013773 (cpu peer) ' + ' '.join(['S'] + ['0'] * 18 + ['333'])
                proc['/proc/2013773/cgroup'] = peer_group
                ns.update({'/proc/2013773/ns/net': 901, '/proc/2013773/ns/mnt': 902})
                peer_allocation_path = base / 'peer-allocation.json'
                write(peer_allocation_path, dict(node='h100-5', controller_pid=2013773, native_unit=peer_unit,
                    native_cgroup='/system.slice/' + peer_unit, source_parent=parent,
                    source_context=str(peer_context_path), source_context_sha256=sha(peer_context_path),
                    model_indices=retrieval, model_cuda_uuids=['GPU-%d' % i for i in retrieval],
                    gpu_sublease_paths={'GPU-2': str(root_child)}))
                value.update(tensor_parallel_size=1, peer_sublease=dict(pid=2013773, birth='333', unit=peer_unit,
                    cgroup=peer_group, netns_inode=901, mntns_inode=902,
                    allocation_path=str(peer_allocation_path), allocation_sha256=sha(peer_allocation_path),
                    gpu_locks={'GPU-2': dict(fd=94, device=info.st_dev, inode=info.st_ino)}))
            write(path, value)
            evidence = base / 'evidence.json'
            write(evidence, dict(node='h100-5', main_pid=123, unit=own_unit, cgroup=own_group,
                unit_show='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (own_unit, own_group),
                original_cuda_visible_devices=','.join(map(str, model)), resource_limits=dict(runtime_max_seconds=28800)))
            upstream = base / 'upstream.json'
            write(upstream, dict(role='upstream', source_pid=1490465, listen_url='http://127.0.0.1:28831'))
            args = argparse.Namespace(base=str(base), model_gpus=','.join(map(str, model)), tp=tp, allocation_evidence=str(evidence),
                reuse_local_relay_ready=str(path), reuse_local_relay_sha256=sha(path),
                reuse_local_relay_unit=parent_unit, upstream_ready=str(upstream))
            lease_paths = []

            def lock(path):
                lease_paths.append(Path(path))
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd

            pool = SimpleNamespace(RUN_ID=run, read=read, sha=sha, try_lock=lock,
                nvml=lambda: {'GPU-%d' % i: dict(index=i, uuid='GPU-%d' % i, free_bytes=8 * 1024 ** 3)
                              for i in range(8)})
            real_stat, real_read_text = os.stat, Path.read_text

            def stat(path, *a, **kw):
                key = str(path)
                if key in ns:
                    return SimpleNamespace(st_ino=ns[key])
                if key in fd_stats:
                    return fd_stats[key]
                return real_stat(path, *a, **kw)

            def text(path, *a, **kw):
                return proc[str(path)] if str(path) in proc else real_read_text(path, *a, **kw)

            with patch.dict(os.environ, Q7B_SOURCE_SUBLEASE_CONTEXT=str(path)), \
                 patch('socket.gethostname', return_value='h100-5'), patch('os.getpid', return_value=123), \
                 patch('os.stat', side_effect=stat), patch.object(Path, 'read_text', text), \
                 patch.object(WORKER.shared, 'foreign_contexts', return_value=[]), \
                 patch.object(WORKER.subprocess, 'check_output', return_value=json.dumps(
                     dict(actual_cuda_runtime_uuids=uuids))) as probe:
                yield SimpleNamespace(base=base, args=args, pool=pool, proc=proc, original=original,
                    lease_paths=lease_paths, probe=probe, uuids=uuids)

    def test_admission_holds_only_child_leases_and_observes_both_sources(self):
        with self.fixture() as f:
            binding, locks = WORKER.admit(f.pool, f.args)
            try:
                self.assertEqual(binding['job_id'], 'source5-model-123')
                self.assertEqual(binding['model_indices'], [0, 1, 2, 5])
                self.assertEqual(binding['tensor_parallel_size'], 4)
                self.assertEqual(binding['actual_cuda_runtime_uuids'], f.uuids)
                self.assertFalse(binding['primary_gpu_locks_modified'])
                self.assertFalse(binding['source_parent_owned'])
                self.assertFalse(binding['shared_retriever_owned_by_consumer'])
                self.assertEqual(len(locks), 4)
                self.assertTrue(all('/gpu-subleases/' in str(p) and '/1490214-111/' in str(p) for p in f.lease_paths))
                for path, original in f.original.items():
                    self.assertEqual(path.read_bytes(), original)
                source = WORKER.borrowed_relay(f.pool, f.args)
                self.assertEqual(source.pid, 1490465)
                self.assertIsNone(source.poll())
                self.assertFalse(hasattr(source, 'terminate'))
                f.proc['/proc/1490214/stat'] = f.proc['/proc/1490214/stat'].rsplit(' ', 1)[0] + ' 112'
                self.assertEqual(source.poll(), 1, 'A live retriever cannot mask a changed parent birth')
            finally:
                for fd in locks:
                    os.close(fd)

    def test_open_parent_fd_without_its_write_lock_is_rejected(self):
        with self.fixture(wrong_owner=True) as f:
            with self.assertRaisesRegex(ValueError, 'parent POSIX WRITE'):
                WORKER.admit(f.pool, f.args)
            self.assertEqual(f.lease_paths, [])
            f.probe.assert_not_called()

    def test_tp1_holds_nested_lease_and_peer_exit_is_not_a_runtime_dependency(self):
        with self.fixture(tp=1) as f:
            binding, locks = WORKER.admit(f.pool, f.args)
            try:
                self.assertEqual(binding['model_indices'], [2])
                self.assertEqual(binding['tensor_parallel_size'], 1)
                self.assertEqual(binding['actual_cuda_runtime_uuids'], ['GPU-2'])
                self.assertEqual(len(locks), 1)
                self.assertIn('/1490214-111/co-share-with-2013773-333/GPU-2.lock', str(f.lease_paths[0]))
                self.assertEqual(binding['peer_sublease_startup_proof']['pid'], 2013773)
                self.assertFalse(binding['peer_is_runtime_dependency'])
                for path, original in f.original.items():
                    self.assertEqual(path.read_bytes(), original)
                source = WORKER.borrowed_relay(f.pool, f.args)
                f.proc['/proc/2013773/stat'] = f.proc['/proc/2013773/stat'].replace(') S ', ') Z ')
                self.assertIsNone(source.poll(), 'The peer authorizes startup but is not a resource dependency')
            finally:
                for fd in locks:
                    os.close(fd)

    def test_tp1_rejects_wrong_peer_lock_owner_before_acquiring_nested_lease(self):
        with self.fixture(tp=1, wrong_peer_owner=True) as f:
            with self.assertRaisesRegex(ValueError, 'peer POSIX WRITE'):
                WORKER.admit(f.pool, f.args)
            self.assertEqual(f.lease_paths, [])
            f.probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
