"""Tiny real filesystem copies and refusal/recovery cases; no remote or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.recursive_bran_a.ablation150_storage_v1 import test_inspect as fixtures
from . import archive

SOURCE, save = fixtures.SOURCE, fixtures.save
IMPLEMENTATION = '9' * 40


class ArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StorageChecks()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.base = self.fixture.root, self.fixture.base
        self.group = self.fixture.receipt['group']
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name).resolve() / 'archive'
        self.destination.mkdir()
        self.ready = dict(path='fixture-storage-ready.json', sha256='a' * 64,
                          retention_until=None, retention_date_known=False)
        headroom = patch.object(archive.meta, 'HEADROOM', 1024)
        headroom.start()
        self.addCleanup(headroom.stop)
        progress = patch.object(archive, 'progress_record')
        progress.start()
        self.addCleanup(progress.stop)
        for name, value in (('owner_present', False), ('open_writers', [])):
            patcher = patch.object(archive.meta, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        entry = json.loads((self.base / 'receipts/current-entry.json').read_text())
        self.output = Path(entry['output'])
        rows = []
        for step in range(1, 151):
            metrics = {}
            if step in archive.meta.KEYS:
                metrics = {'val/qa_count/all': 512, 'val/qa_count/nq': 256,
                    'val/qa_count/hotpotqa': 256, 'val/em/all': 96/512,
                    'val/em/nq': 54/256, 'val/em/hotpotqa': 42/256}
            manifest = self.base / f'checkpoints/training_state/global_step_{step}/post-evaluation/manifest.json'
            rows.append(dict(completed_outer=step, checkpoint=dict(path=str(manifest),
                sha256=archive.sha(manifest) if step in archive.meta.KEYS else 'a'*64,
                native_only=step not in archive.meta.KEYS), metrics=metrics,
                optimizer_calls_per_rank={str(rank):step*20 for rank in range(8)},
                **{key:self.fixture.receipt[key] for key in
                   ('group', 'run_id', 'source_commit', 'config_sha256')}))
        (self.output / 'completed-outers.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def copy(self):
        return archive.copy_closed(self.root, self.group, SOURCE, self.destination, self.ready, IMPLEMENTATION)

    def accepted(self):
        result = self.copy()
        path = Path(result['complete_receipt'])
        complete = json.loads(path.read_text())
        plan = json.loads(Path(complete['inventory']).read_text())
        return result, path, complete, plan

    def test_copy_all_canonical_and_hf_full_sha_no_original_mutation(self):
        originals = {str(path): path.read_bytes() for path in self.base.rglob('*') if path.is_file()}
        result, complete, value, plan = self.accepted()
        self.assertEqual(result['status'], 'COMPLETE_FULL_SHA_ACCEPTED_COPY_ONLY')
        self.assertEqual(sum(row['kind'] == 'CANONICAL_POST_NATIVE' for row in plan['members']), 36)
        for row in plan['members']:
            self.assertEqual(archive.sha(row['path']), row['sha256'])
            self.assertEqual(archive.sha(row['archive_path']), row['sha256'])
        for path, raw in originals.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        self.assertFalse(value['original_payloads_reclaimed'])
        verified = archive.verify_complete(complete, archive.sha(complete), self.root, self.destination, self.ready)
        self.assertTrue(verified['current_destination_full_sha_verified'])

    def test_active_or_incomplete_group_refuses_before_copy(self):
        with patch.object(archive.meta, 'owner_present', return_value=True):
            with self.assertRaisesRegex(ValueError, 'PID still present'):
                self.copy()
        path = self.output / 'completed-outers.jsonl'
        path.write_text(''.join(path.read_text().splitlines(keepends=True)[:-1]))
        with self.assertRaisesRegex(ValueError, 'missing complete native150'):
            self.copy()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_source_mismatch_and_destination_mismatch_never_reclaim(self):
        row = self.base / 'checkpoints/training_state/global_step_5/post-evaluation/rank-00.pt'
        row.write_bytes(b'mutated source')
        with self.assertRaisesRegex(ValueError, 'worker size changed|source SHA mismatch'):
            self.copy()
        self.assertTrue(row.exists())

    def test_changed_destination_fails_fresh_verification_originals_preserved(self):
        result, path, complete, plan = self.accepted()
        row = next(value for value in plan['members'] if value['kind'] == 'CANONICAL_POST_NATIVE')
        Path(row['archive_path']).write_bytes(b'wrong archived bytes')
        with self.assertRaisesRegex(ValueError, 'destination (fresh SHA|size) mismatch'):
            archive.verify_complete(path, archive.sha(path), self.root, self.destination, self.ready)
        self.assertTrue(Path(row['path']).exists())

    def test_interrupted_partial_resumes_matching_prefix(self):
        snapshot = archive.closure(self.root, self.group, SOURCE)
        plan = archive.inventory(snapshot, self.root, self.destination, IMPLEMENTATION)
        row = next(value for value in plan['members'] if value['kind'] == 'CANONICAL_POST_NATIVE')
        destination = Path(row['archive_path'])
        destination.parent.mkdir(parents=True)
        partial = destination.with_name(destination.name+'.partial')
        partial.write_bytes(Path(row['path']).read_bytes()[:4])
        proof = archive.copy_member(row, self.root, self.destination, lambda value: None)
        self.assertEqual(proof['destination_sha256'], row['sha256'])
        self.assertFalse(partial.exists())
        self.assertEqual(destination.read_bytes(), Path(row['path']).read_bytes())

    def test_wrong_partial_prefix_is_not_overwritten(self):
        snapshot = archive.closure(self.root, self.group, SOURCE)
        row = archive.inventory(snapshot, self.root, self.destination, IMPLEMENTATION)['members'][0]
        destination = Path(row['archive_path'])
        destination.parent.mkdir(parents=True)
        partial = destination.with_name(destination.name+'.partial')
        partial.write_bytes(b'bad!')
        with self.assertRaisesRegex(ValueError, 'partial prefix mismatch'):
            archive.copy_member(row, self.root, self.destination, lambda value: None)
        self.assertEqual(partial.read_bytes(), b'bad!')

    def test_exact_group_and_symlink_namespace_refuse(self):
        with self.assertRaisesRegex(ValueError, 'only named seven'):
            archive.closure(self.root, 'A-original', SOURCE)
        snapshot = archive.closure(self.root, self.group, SOURCE)
        row = archive.inventory(snapshot, self.root, self.destination, IMPLEMENTATION)['members'][0]
        dest = Path(row['archive_path'])
        dest.parent.mkdir(parents=True)
        dest.symlink_to(row['path'])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            archive.copy_member(row, self.root, self.destination, lambda value: None)

    def test_reclaim_requires_prior_complete_v2_and_real_low_space(self):
        result, path, complete, plan = self.accepted()
        with self.assertRaisesRegex(ValueError, 'fully pruned'):
            archive.reclaim(path, archive.sha(path), self.root, self.destination, self.ready,
                archive.closed.measured_burst(self.root, SOURCE)['next_key_burst_bytes'])
        self.assertTrue(all(Path(row['path']).exists() for row in plan['members']))
        self.finish_prior_pruning()
        # The actual filesystem is not low. Explicit reclaim makes no deletion.
        burst = archive.closed.measured_burst(self.root, SOURCE)['next_key_burst_bytes']
        actual = archive.reclaim(path, archive.sha(path), self.root, self.destination, self.ready, burst)
        self.assertEqual(actual['deleted_files_this_call'], 0)
        self.assertTrue(all(Path(row['path']).exists() for row in plan['members']))

    def finish_prior_pruning(self):
        snapshot = archive.closure(self.root, self.group, SOURCE)
        plan = archive.closed.plan_for(snapshot, self.root)
        directory_path = self.base / 'receipts/closed-storage-v2'
        plan_sha = save(directory_path / 'planned.json', plan)
        rows = [(state, member) for state in plan['candidate_states'] for member in state['files']]
        for index, (state, member) in enumerate(rows):
            binding = archive.closed.file_binding(plan_sha, index, state, member)
            save(directory_path / 'files' / f'{index:04d}-intent.json', binding)
            Path(member['path']).unlink()
            save(directory_path / 'files' / f'{index:04d}-actual.json',
                 dict(binding_sha256=archive.closed.digest(binding), action='UNLINKED_THIS_EXECUTION'))
        save(directory_path / 'actual.json', dict(protocol=archive.closed.PROTOCOL,
            plan_sha256=plan_sha, status='COMPLETE_SUPERSEDED_PAYLOADS_PRUNED', group=self.group,
            current_absent_files=len(rows), total_planned_files=len(rows)))

    def test_low_space_reclaim_only_native_and_restart_after_durable_intent(self):
        result, path, complete, plan = self.accepted()
        self.finish_prior_pruning()
        burst = archive.closed.measured_burst(self.root, SOURCE)['next_key_burst_bytes']
        unlink = archive.closed.unlink_bound
        count = [0]
        def interrupt(row, root):
            unlink(row, root)
            count[0] += 1
            if count[0] == 1:
                raise OSError('fixture process lost after unlink')
        with patch.object(archive.closed, 'free_bytes', return_value=0), \
                patch.object(archive.closed, 'unlink_bound', side_effect=interrupt):
            with self.assertRaisesRegex(OSError, 'process lost'):
                archive.reclaim(path, archive.sha(path), self.root, self.destination, self.ready, burst)
        with patch.object(archive.closed, 'free_bytes', return_value=0), \
                patch.object(archive.closed, 'measured_burst', return_value={'next_key_burst_bytes':burst}):
            actual = archive.reclaim(path, archive.sha(path), self.root, self.destination, self.ready, burst)
        self.assertEqual(actual['status'], 'ALL_CANONICAL_NATIVE_PAYLOADS_RELOCATED')
        self.assertEqual(actual['previous_or_recovery_absent'], 1)
        self.assertEqual(actual['deleted_files_this_call'], 35)
        self.assertTrue(all(not Path(row['path']).exists() for row in plan['members']
                            if row['kind'] == 'CANONICAL_POST_NATIVE'))
        self.assertTrue(all(Path(row['path']).exists() for row in plan['members']
                            if row['kind'] != 'CANONICAL_POST_NATIVE'))
        verified = archive.verify_complete(path, archive.sha(path), self.root, self.destination, self.ready)
        self.assertTrue(verified['current_destination_full_sha_verified'])
        fallback = archive.observed_burst(self.root, SOURCE, self.destination)
        self.assertEqual(fallback['next_key_burst_bytes'], burst)
        self.assertIn('ARCHIVED', fallback['size_evidence'])

    def test_storage_contract_two_node_object_and_actual_mount(self):
        probe = self.destination / 'probe.bin'
        probe.write_bytes(b'x' * 64)
        checksum = archive.sha(probe)
        readiness = dict(protocol=archive.READY_PROTOCOL, ready=True, study_root=str(self.root),
            archive_root=str(self.destination), retention_until=None,
            mount=dict(mountpoint=str(archive.MOUNT), fstype='nfs4', source=archive.MOUNT_SOURCE),
            probe=dict(path=str(probe), bytes=64, sha256=checksum, local_sha256=checksum,
                server_sha256=checksum, server_read_rc=0, write_rc=0))
        path = self.root / 'storage-ready.json'
        save(path, readiness)
        with patch.object(archive.meta, 'mounted_capacity', return_value=readiness['mount']):
            actual = archive.storage_ready(path, archive.sha(path), self.root, self.destination)
            self.assertFalse(actual['retention_date_known'])
            readiness['probe']['server_sha256'] = 'a' * 64
            save(path, readiness)
            with self.assertRaisesRegex(ValueError, 'two-node'):
                archive.storage_ready(path, archive.sha(path), self.root, self.destination)

    def test_metadata_poll_is_explicitly_not_fresh_destination_sha(self):
        result, path, complete, plan = self.accepted()
        with patch.object(archive, 'stable_sha', side_effect=AssertionError('no routine full stream')):
            status = archive.verify_complete(path, archive.sha(path), self.root, self.destination,
                                             self.ready, full_sha=False)
        self.assertFalse(status['current_destination_full_sha_verified'])
        self.assertIn('PRIOR_ACCEPTED', status['destination_sha_mode'])

    def test_maintenance_skips_active_group_and_never_runs_gpu(self):
        study = json.loads((self.root / 'study-progress.json').read_text())
        study['current'] = self.group
        save(self.root / 'study-progress.json', study)
        with patch.object(archive.closed, 'once', return_value={'status':'SPACE_SUFFICIENT_NO_PRUNING'}) as pruning, \
                patch.object(archive, 'copy_closed') as copying:
            result = archive.maintenance_once(self.root, SOURCE, self.destination, self.ready, IMPLEMENTATION)
        pruning.assert_called_once_with(self.root, SOURCE, apply=False)
        copying.assert_not_called()
        self.assertEqual(result['groups'], [])
        self.assertFalse(result['gpu_action_executed'])


if __name__ == '__main__':
    unittest.main()
