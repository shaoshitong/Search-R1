"""Tiny CPU filesystem fixtures only; never touch experiment outputs."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from release.recursive_bran_a.ablation150_storage_v1 import test_inspect as fixtures
from . import execute as storage

SOURCE = fixtures.SOURCE
save = fixtures.save


class ClosedStorageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StorageChecks()
        self.fixture.setUp()
        self.root, self.base, self.receipt = self.fixture.root, self.fixture.base, self.fixture.receipt
        output = self.base / 'output/attempt-fixture'
        records = []
        for step in range(1, 151):
            record = dict(completed_outer=step, optimizer_calls_per_rank={str(i): step*20 for i in range(8)},
                **{key: self.receipt[key] for key in ('group', 'run_id', 'source_commit', 'config_sha256')},
                metrics={})
            if step in storage.meta.KEYS:
                manifest = self.base / f'checkpoints/training_state/global_step_{step}/post-evaluation/manifest.json'
                record['checkpoint'] = dict(path=str(manifest),
                    sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), native_only=False)
                record['metrics'] = {'val/qa_count/all': 512, 'val/qa_count/nq': 256,
                    'val/qa_count/hotpotqa': 256, 'val/em/all': .125, 'val/em/nq': .125, 'val/em/hotpotqa': .125}
            records.append(record)
        (output / 'completed-outers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def tearDown(self):
        self.fixture.tearDown()

    def call(self, apply=True, free=1, active=False, writers=None):
        with patch.object(storage.meta, 'owner_present', return_value=active), \
             patch.object(storage.meta, 'open_writers', return_value=writers or []), \
             patch.object(storage, 'free_bytes', return_value=free):
            return storage.once(self.root, SOURCE, apply=apply)

    def payload(self, step=5, phase='pre', name='rank-00.pt'):
        directory = 'native-rolling' if phase == 'rolling' else 'training_state'
        path = self.base / f'checkpoints/{directory}/global_step_{step}'
        if phase == 'post':
            path /= 'post-evaluation'
        return path / name

    def test_low_space_gate_and_default_dry_run_never_delete(self):
        result = self.call(free=10**15)
        self.assertEqual(result['status'], 'SPACE_SUFFICIENT_NO_PRUNING')
        self.assertFalse((self.base / 'receipts/closed-storage-v2').exists())
        result = self.call(apply=False)
        self.assertEqual(result['groups'][0]['status'], 'CLOSED_GROUP_DRY_RUN')
        self.assertTrue(self.payload().exists())
        self.assertFalse((self.base / 'receipts/closed-storage-v2').exists())

    def test_active_owner_and_writer_refused(self):
        for active, writers in ((True, []), (False, [dict(pid=1, path=str(self.base / 'logs/open.log'))])):
            result = self.call(active=active, writers=writers)
            self.assertEqual(result['groups'][0]['status'], 'ATTENTION_NO_FURTHER_PRUNING')
            self.assertTrue(self.payload().exists())
            self.assertFalse((self.base / 'receipts/closed-storage-v2').exists())

    def test_exact_payload_allowlist_and_canonical_receipts(self):
        extra = self.payload(name='important.pt')
        extra.write_bytes(b'not a pruning candidate')
        log = self.base / 'logs/required.log'
        log.parent.mkdir()
        log.write_text('retained')
        result = self.call()
        row = result['groups'][0]
        self.assertEqual(row['status'], 'COMPLETE_SUPERSEDED_PAYLOADS_PRUNED')
        self.assertEqual(row['deleted_files_this_call'], 54)
        self.assertTrue(extra.exists() and log.exists())
        for step in storage.meta.KEYS:
            self.assertTrue(self.payload(step, 'post').exists())
            self.assertFalse(self.payload(step, 'pre').exists())
            self.assertTrue(self.payload(step, 'pre', 'manifest.json').exists())
        directory = self.base / 'receipts/closed-storage-v2'
        proof = json.loads((directory / 'canonical-verified.json').read_text())
        self.assertEqual(len(proof['members']), 48)
        actual = json.loads((directory / 'complete.json').read_text())
        self.assertTrue(all(t['historical_payload_status'] == 'PRUNED' for t in actual['tombstones']))
        again = self.call()
        self.assertEqual(again['groups'][0]['deleted_files_this_call'], 0)
        self.assertEqual(again['groups'][0]['canonical_sha_evidence'],
                         'REUSED_PREVIOUS_FULL_SHA_WITH_FRESH_UNCHANGED_METADATA')

    def test_canonical_sha_mismatch_refuses_before_unlink(self):
        path = self.payload(50, 'post')
        old = path.read_bytes()
        path.write_bytes(b'X'*len(old))
        result = self.call()
        self.assertIn('canonical payload SHA mismatch', result['groups'][0]['error'])
        self.assertTrue(self.payload().exists())
        self.assertFalse((self.base / 'receipts/closed-storage-v2/files').exists())

    def test_partial_receipt_resume_accepts_only_absent_after_own_intent(self):
        original = storage.unlink_bound
        def crash_after_unlink(row, root):
            original(row, root)
            raise RuntimeError('simulated interrupted receipt write')
        with patch.object(storage, 'unlink_bound', side_effect=crash_after_unlink):
            first = self.call()
        self.assertIn('simulated interrupted', first['groups'][0]['error'])
        directory = self.base / 'receipts/closed-storage-v2'
        self.assertTrue((directory / 'files/0000-intent.json').exists())
        self.assertFalse((directory / 'files/0000-actual.json').exists())
        second = self.call()
        self.assertEqual(second['groups'][0]['status'], 'COMPLETE_SUPERSEDED_PAYLOADS_PRUNED')
        actual = json.loads((directory / 'files/0000-actual.json').read_text())
        self.assertEqual(actual['action'], 'OBSERVED_ABSENT_AFTER_DURABLE_INTENT')
        self.assertFalse(actual['actual_unlink_observed_this_call'])

    def test_unexplained_absence_after_plan_refuses_before_any_more_pruning(self):
        with patch.object(storage, 'stream_sha', side_effect=RuntimeError('simulated before hash')):
            self.call()
        self.payload(100, 'pre').unlink()
        result = self.call()
        self.assertIn('no own durable deletion intent', result['groups'][0]['error'])
        self.assertTrue(self.payload(5, 'pre').exists())

    def test_forged_candidate_path_and_receipt_symlink_refuse(self):
        manifest = self.payload(name='manifest.json')
        value = json.loads(manifest.read_text())
        value['worker_states'][0]['path'] = str(self.payload(5, 'post'))
        save(manifest, value)
        result = self.call()
        self.assertIn('candidate worker path', result['groups'][0]['error'])
        self.assertTrue(self.payload().exists())
        directory = self.base / 'receipts/closed-storage-v2'
        external = self.root / 'not-receipt-storage'
        external.mkdir()
        directory.symlink_to(external, target_is_directory=True)
        result = self.call()
        self.assertIn('symlink', result['groups'][0]['error'])
        self.assertEqual(list(external.iterdir()), [])

    def test_missing_formal512_or_failed_wait_refuses(self):
        records = self.base / 'output/attempt-fixture/completed-outers.jsonl'
        rows = [json.loads(line) for line in records.read_text().splitlines()]
        rows[49]['metrics']['val/qa_count/all'] = 511
        records.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result = self.call()
        self.assertIn('complete balanced dev512', result['groups'][0]['error'])
        self.assertTrue(self.payload().exists())


if __name__ == '__main__':
    unittest.main()
