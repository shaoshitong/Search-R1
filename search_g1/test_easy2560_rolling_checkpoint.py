"""Opaque CPU bytes test durability/retention, never claim a CUDA reload."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search_g1 import easy2560_rolling_checkpoint as rolling


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


class Fixture:
    def __init__(self, root):
        self.base = Path(root).resolve() / 'current'
        self.run = dict(easy2560_selection=True, training_world_size=4, run_id='run-actual-fixture',
                        source_commit='a' * 40,
                        checkpoint_dir=str(self.base / 'checkpoints/search-g1-common8'))
        put(self.base / 'logs/run-manifest.json', self.run)
        self.actor = Path(self.run['checkpoint_dir']) / 'actor'

    def checkpoint(self, step, boundary=True):
        cp = self.actor / ('global_step_%d' % step)
        workers = []
        # Existing native4 layout: three model shards, index, six state files,
        # eight HF accessories, complete.json = 19 members.
        for i in range(3):
            put(cp / ('model-%05d-of-00003.safetensors' % (i + 1)), b'opaque model tensor bytes')
        index = {'weight_map': {'weight%d' % i: 'model-%05d-of-00003.safetensors' % (i + 1) for i in range(3)}}
        put(cp / 'model.safetensors.index.json', index)
        for i in range(8):
            put(cp / ('tokenizer-config-%d.json' % i), {'accessory': i})
        for rank in range(4):
            path = put(cp / ('training_state/rank-%d.pt' % rank),
                       ('opaque rank%d Adam, scheduler and complete RNG state at %d' % (rank, step)).encode())
            workers.append(dict(rank=rank, run_id=self.run['run_id'], path=str(path),
                                sha256=rolling._sha(path), bytes=path.stat().st_size))
        driver = put(cp / 'training_state/driver.pt', b'opaque candidate stream, cursor, RNG and driver state')
        put(cp / 'training_state/manifest.json', dict(completed_outer=step, question_probe=None, worker_files=workers,
            driver_file=dict(path=str(driver), sha256=rolling._sha(driver), bytes=driver.stat().st_size)))
        hashes = {str(path.relative_to(cp)): rolling._sha(path) for path in cp.rglob('*') if path.is_file()}
        put(cp / 'complete.json', dict(step=step, run_id=self.run['run_id'], checkpoint=str(cp),
                                      finite_all_saved_tensors=True, files_sha256=hashes))
        if not boundary:
            return cp, None
        directory = self.base / ('preparation/rolling-checkpoints/global_step_%d/boundary' % step)
        workers = []
        for rank in range(4):
            path = put(directory / ('rank-%d.pt' % rank), b'opaque post-callback rank RNG')
            workers.append(dict(rank=rank, path=str(path), sha256=rolling._sha(path)))
        driver = put(directory / 'driver.pt', b'opaque post-callback driver state, RNG and data cursor')
        manifest = put(directory / 'manifest.json', dict(schema_version=1, run_id=self.run['run_id'],
            completed_outer=step, worker_files=workers,
            driver_file=dict(path=str(driver), sha256=rolling._sha(driver)), probe_version='after-callback',
            probe_round=min((step // 50) * 50, 150), callbacks_complete=True, reward_refresh_pending=False))
        return cp, manifest

    def publish(self, step, **kwargs):
        cp, boundary = self.checkpoint(step)
        return rolling.publish_checkpoint(self.base, cp, boundary_manifest=boundary, **kwargs)


class RollingCheckpointTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Fixture(temporary.name)

    def test_latest_seals_full_native_state_and_post_callback_boundary_without_new_cp_members(self):
        f = self.fixture()
        result = f.publish(6)
        self.assertEqual(result['checkpoint_file_count'], 19)
        self.assertEqual(result['boundary_file_count'], 6)
        pointer = rolling.latest_checkpoint(f.base)
        manifest = rolling.verify_checkpoint_manifest(f.base, pointer['manifest_path'])
        self.assertEqual(pointer['step'], 6)
        self.assertEqual(manifest['boundary']['probe_version'], 'after-callback')
        self.assertEqual(len(list(p for p in Path(pointer['checkpoint']).rglob('*') if p.is_file())), 19)
        self.assertFalse(manifest['actual_cuda_restore_accepted'])

    def test_shared_mount_client_device_number_does_not_reject_same_bytes(self):
        f = self.fixture(); result = f.publish(6)
        pointer = result['latest']; path = Path(pointer['manifest_path'])
        manifest = json.loads(path.read_text())
        for row in list(manifest['files'].values()) + list(manifest['boundary_files'].values()):
            row['device'] += 10000  # GPU and login mounts expose different st_dev.
        if manifest['boundary_manifest'] is not None:
            manifest['boundary_manifest'] = manifest['boundary_files']['manifest.json']
        put(path, manifest)
        pointer['manifest_sha256'] = rolling._sha(path)
        put(f.base/'preparation/rolling-checkpoints/latest.json', pointer)
        self.assertEqual(rolling.latest_checkpoint(f.base, verify=True)['step'], 6)

    def test_successful_next_save_retires_old_payloads_only_after_pointer_points_to_next(self):
        f = self.fixture()
        f.publish(6)
        log = put(f.base / 'logs/keep.log', b'keep actual log')
        raw = put(f.base / 'logs/raw/keep.jsonl', b'keep raw evidence')
        selection = put(f.base / 'logs/selection/keep.json', b'keep selection')
        cp, boundary = f.checkpoint(7)
        original = rolling._retire_files
        seen = []
        def retire(directory, rows):
            seen.append(rolling.latest_checkpoint(f.base)['step'])
            original(directory, rows)
        with patch.object(rolling, '_retire_files', side_effect=retire):
            result = rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(seen, [7])
        self.assertEqual(result['retired_steps'], [6])
        self.assertFalse((f.actor / 'global_step_6').exists())
        self.assertTrue(cp.is_dir())
        audit = f.base / 'preparation/rolling-checkpoints/global_step_6'
        self.assertTrue((audit / 'full-manifest.json').is_file())
        self.assertTrue((audit / 'boundary/manifest.json').is_file())
        self.assertFalse(list((audit / 'boundary').glob('*.pt')))
        self.assertTrue(all(path.is_file() for path in (log, raw, selection)))

    def test_incomplete_next_state_keeps_previous_recoverable_save(self):
        f = self.fixture()
        old = f.publish(6)['latest']
        cp, boundary = f.checkpoint(7)
        (cp / 'training_state/rank-3.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'FULL_MODEL_ADAM_RNG_DRIVER_REQUIRED|PRODUCER_SHA_DIFFERS'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base), old)
        self.assertTrue((f.actor / 'global_step_6').is_dir())

    def test_boundary_sha_failure_keeps_previous_save_and_pointer(self):
        f = self.fixture()
        old = f.publish(6)['latest']
        cp, boundary = f.checkpoint(7)
        (boundary.parent / 'driver.pt').write_bytes(b'changed after boundary was saved')
        with self.assertRaisesRegex(ValueError, 'STATE_SHA_DIFFERS'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base), old)
        self.assertTrue((f.actor / 'global_step_6').is_dir())

    def test_fsync_failure_never_advances_latest(self):
        f = self.fixture()
        old = f.publish(6)['latest']
        cp, boundary = f.checkpoint(7)
        with patch.object(rolling.os, 'fsync', side_effect=OSError('fixture disk failure')):
            with self.assertRaises(OSError):
                rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base), old)
        self.assertTrue((f.actor / 'global_step_6').is_dir())

    def test_latest_directory_fsync_failure_keeps_both_complete_recovery_points(self):
        f = self.fixture()
        f.publish(6)
        cp, boundary = f.checkpoint(7)
        metadata = f.base / 'preparation/rolling-checkpoints'
        original = rolling._fsync_directory
        def fail_after_latest_replace(directory):
            if Path(directory) == metadata:
                raise OSError('fixture latest-directory fsync failure after replace')
            return original(directory)
        with patch.object(rolling, '_fsync_directory', side_effect=fail_after_latest_replace):
            with self.assertRaisesRegex(OSError, 'after replace'):
                rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        # The rename happened, while its directory durability was not confirmed.
        # Either head is safe after a crash: neither full save was retired.
        self.assertEqual(rolling.latest_checkpoint(f.base)['step'], 7)
        self.assertTrue((f.actor / 'global_step_6').is_dir())
        rolling.verify_checkpoint_manifest(f.base, metadata / 'global_step_6/full-manifest.json')
        rolling.verify_checkpoint_manifest(f.base, metadata / 'global_step_7/full-manifest.json')

    def test_pre_callback_or_pending_refit_boundary_is_never_published(self):
        for overrides in ({'callbacks_complete': False}, {'reward_refresh_pending': True}):
            with self.subTest(overrides=overrides):
                f = self.fixture()
                old = f.publish(6)['latest']
                cp, boundary = f.checkpoint(7)
                value = json.loads(boundary.read_bytes())
                value.update(overrides)
                put(boundary, value)
                with self.assertRaisesRegex(ValueError, 'POST_CALLBACK_BOUNDARY_REQUIRED'):
                    rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
                self.assertEqual(rolling.latest_checkpoint(f.base), old)
                self.assertTrue((f.actor / 'global_step_6').is_dir())

    def test_pointer_replace_failure_keeps_previous_save_and_retry_is_idempotent(self):
        f = self.fixture()
        old = f.publish(6)['latest']
        cp, boundary = f.checkpoint(7)
        original = rolling.os.replace
        def fail_latest(source, destination):
            if str(destination).endswith('/latest.json'):
                raise OSError('fixture atomic pointer failure')
            return original(source, destination)
        with patch.object(rolling.os, 'replace', side_effect=fail_latest):
            with self.assertRaises(OSError):
                rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base), old)
        self.assertTrue((f.actor / 'global_step_6').is_dir())
        result = rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(result['latest']['step'], 7)
        self.assertEqual(result['retired_steps'], [6])

    def test_milestones_are_always_kept_even_if_caller_passes_empty_milestone_set(self):
        f = self.fixture()
        for step in (5, 6, 50, 51, 100, 101, 150, 151, 200):
            f.publish(step, milestone_steps=())
        self.assertEqual(sorted(int(path.name.split('_')[-1]) for path in f.actor.iterdir()), [5, 50, 100, 150, 200])
        self.assertEqual(rolling.latest_checkpoint(f.base)['step'], 200)

    def test_unmanaged_checkpoint_and_other_base_remain_untouched(self):
        f = self.fixture()
        unmanaged, _ = f.checkpoint(4)
        other = Fixture(f.base.parent / 'other')
        other.publish(6)
        f.publish(6)
        f.publish(7)
        self.assertTrue(unmanaged.is_dir())
        self.assertEqual(rolling.latest_checkpoint(other.base)['step'], 6)
        self.assertTrue((other.actor / 'global_step_6').is_dir())

    def test_prune_refuses_corrupted_latest_before_deleting_any_old_checkpoint(self):
        f = self.fixture()
        f.publish(6)
        f.publish(7, prune=False)
        path = f.actor / 'global_step_7/training_state/driver.pt'
        before = path.stat()
        original = path.read_bytes()
        path.write_bytes(b'x' * len(original))
        # Even a same-size/mtime mutation must fail the independent full SHA.
        import os
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'FILE_SHA_CHANGED'):
            rolling.prune_checkpoints(f.base)
        self.assertTrue((f.actor / 'global_step_6').is_dir())

    def test_changed_old_members_are_not_broadly_removed(self):
        f = self.fixture()
        f.publish(6)
        extra = put(f.actor / 'global_step_6/unrecorded.pt', b'not sealed by manager')
        result = f.publish(7)
        self.assertEqual(result['retired_steps'], [])
        self.assertIn('RETIRE_MEMBERS_CHANGED', result['retention_errors'][0]['error'])
        self.assertTrue(extra.is_file())
        self.assertEqual(rolling.latest_checkpoint(f.base)['step'], 7)

    def test_retention_io_failure_does_not_invalidate_new_save(self):
        f = self.fixture()
        f.publish(6)
        cp, boundary = f.checkpoint(7)
        with patch.object(rolling, '_retire_files', side_effect=OSError('fixture unlink failure')):
            result = rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base)['step'], 7)
        self.assertEqual(result['retired_steps'], [])
        self.assertIn('fixture unlink failure', result['retention_errors'][0]['error'])
        self.assertTrue((f.actor / 'global_step_6').is_dir())
        retried = rolling.prune_checkpoints(f.base)
        self.assertEqual(retried['retired_steps'], [6])

    def test_symlink_escape_is_rejected_without_changing_old_save(self):
        f = self.fixture()
        old = f.publish(6)['latest']
        cp, boundary = f.checkpoint(7)
        outside = put(f.base.parent / 'foreign-tensor.pt', b'foreign protected bytes')
        path = cp / 'training_state/rank-0.pt'
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'NONREGULAR_OR_SYMLINK_FILE'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(outside.read_bytes(), b'foreign protected bytes')
        self.assertEqual(rolling.latest_checkpoint(f.base), old)

    def test_foreign_boundary_and_wrong_run_are_rejected(self):
        f = self.fixture()
        cp, boundary = f.checkpoint(6)
        with self.assertRaisesRegex(ValueError, 'RUN_BINDING_DIFFERS'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary, run_id='other-run')
        value = json.loads(boundary.read_bytes())
        value['run_id'] = 'foreign-run'
        put(boundary, value)
        with self.assertRaisesRegex(ValueError, 'BOUNDARY_RUN_OR_OUTER_DIFFERS'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertIsNone(rolling.latest_checkpoint(f.base))

    def test_backward_latest_and_outer201_are_rejected(self):
        f = self.fixture()
        f.publish(7)
        cp, boundary = f.checkpoint(6)
        with self.assertRaisesRegex(ValueError, 'LATEST_MUST_NOT_MOVE_BACKWARDS'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        cp, boundary = f.checkpoint(201)
        with self.assertRaisesRegex(ValueError, 'INVALID_OUTER'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base)['step'], 7)

    def test_state_proofs_cannot_point_to_another_checkpoint(self):
        f = self.fixture()
        f.publish(6)
        cp, boundary = f.checkpoint(7)
        path = cp / 'training_state/manifest.json'
        value = json.loads(path.read_bytes())
        value['driver_file']['path'] = str(f.actor / 'global_step_6/training_state/driver.pt')
        put(path, value)
        complete = json.loads((cp / 'complete.json').read_bytes())
        complete['files_sha256']['training_state/manifest.json'] = rolling._sha(path)
        put(cp / 'complete.json', complete)
        with self.assertRaisesRegex(ValueError, 'STATE_PATH_DIFFERS'):
            rolling.publish_checkpoint(f.base, cp, boundary_manifest=boundary)
        self.assertEqual(rolling.latest_checkpoint(f.base)['step'], 6)

    def test_latest_pointer_manifest_tampering_is_rejected(self):
        f = self.fixture()
        result = f.publish(6)
        path = Path(result['latest']['manifest_path'])
        value = json.loads(path.read_bytes())
        value['step'] = 5
        put(path, value)
        with self.assertRaisesRegex(ValueError, 'LATEST_MANIFEST_SHA_DIFFERS'):
            rolling.latest_checkpoint(f.base)


if __name__ == '__main__':
    unittest.main()
