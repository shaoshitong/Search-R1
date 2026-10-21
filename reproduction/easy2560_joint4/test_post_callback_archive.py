"""Real temporary-file fixtures for the independent eight-file CPU archive."""
import copy
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from reproduction.easy2560_joint4 import checkpoint_archive as core
from reproduction.easy2560_joint4 import post_callback_archive as boundary
from search_g1 import easy2560_rolling_checkpoint as rolling


class BoundaryArchiveTests(unittest.TestCase):
    def fixture(self, root, step=5):
        root = root.resolve()
        base = root/'source'; checkpoint = base/'checkpoints/search-g1-common8/actor'/('global_step_%d' % step)
        checkpoint.mkdir(parents=True)
        run_id = 'b'*32; commit = 'c'*40
        run = dict(run_id=run_id, source_commit=commit, easy2560_selection=True, question_probe=None,
                   training_world_size=4, checkpoint_dir=str(base/'checkpoints/search-g1-common8'))
        core.publish(base/'logs/run-manifest.json', run)
        core.publish(checkpoint/'model.safetensors.index.json', {'weight_map':{'weight':'model-00001-of-00001.safetensors'}})
        (checkpoint/'model-00001-of-00001.safetensors').write_bytes(b'fixture core model, never in boundary stream')
        workers = []
        for rank in range(4):
            path = checkpoint/('training_state/rank-%d.pt' % rank)
            path.parent.mkdir(exist_ok=True); path.write_bytes(('core Adam and RNG rank%d' % rank).encode())
            workers.append(dict(boundary.record(path), rank=rank, run_id=run_id))
        driver = checkpoint/'training_state/driver.pt'; driver.write_bytes(b'core pre-callback driver')
        core.publish(checkpoint/'training_state/manifest.json', dict(completed_outer=step, worker_files=workers,
            driver_file=boundary.record(driver), question_probe=None))
        hashes = {str(path.relative_to(checkpoint)):core.sha(path) for path in checkpoint.rglob('*') if path.is_file()}
        core.publish(checkpoint/'complete.json', dict(run_id=run_id, step=step, finite_all_saved_tensors=True,
            validated_tensor_count=339, checkpoint=str(checkpoint), files_sha256=hashes))

        directory = base/'preparation/rolling-checkpoints'/('global_step_%d' % step)/'boundary'
        directory.mkdir(parents=True)
        workers = []
        for rank in range(4):
            path = directory/('rank-%d.pt' % rank); path.write_bytes(('post-callback RNG rank%d' % rank).encode())
            workers.append(dict(boundary.record(path), rank=rank, run_id=run_id))
        driver = directory/'driver.pt'; driver.write_bytes(b'post-callback driver RNG and candidate cursor')
        version = run_id+':'+str(boundary.phase(step)['probe_round'])+':fixture'
        reward = directory/'reward-probe-manifest.json'
        core.publish(reward, dict(run_id=run_id, version=version, round=boundary.phase(step)['probe_round']))
        core.publish(directory/'manifest.json', dict(schema_version=1, role='easy2560_completed_callback_boundary',
            run_id=run_id, source_commit=commit, completed_outer=step, callbacks_complete=True,
            reward_refresh_pending=False, worker_files=workers, driver_file=boundary.record(driver),
            reward_probe_manifest=boundary.record(reward), probe_version=version,
            probe_round=boundary.phase(step)['probe_round']))
        rolling.publish_checkpoint(base, checkpoint, boundary_manifest=directory/'manifest.json', prune=False)

        archive = root/'HK/checkpoints'/run_id/('global_step_%d' % step)
        files = {'checkpoint/'+str(path.relative_to(checkpoint)):boundary.record(path)
                 for path in checkpoint.rglob('*') if path.is_file()}
        source = dict(schema_version=1, role='easy2560_full_checkpoint_source_snapshot', base=str(base),
            run_id=run_id, source_commit=commit, step=step, training_world_size=4, question_probe_saved=False,
            files=files, total_bytes=sum(row['bytes'] for row in files.values()))
        source_path = base/'preparation/checkpoint-archives'/('global_step_%d' % step)/'source-snapshot.json'
        core.publish(source_path, source)
        received = dict(status='EASY2560_ARCHIVE_RECEIVED_FRESH_SHA_FSYNC_PASS', base=str(base), run_id=run_id,
            source_commit=commit, step=step, snapshot_sha256=core.sha(source_path), file_count=len(files),
            total_bytes=source['total_bytes'], fresh_destination_sha_verified=True, destination_fsynced=True,
            actual_cuda_restore_accepted=False, checked_unix=1)
        accepted = dict(received, status=core.STATUS, source_snapshot=str(source_path),
            fresh_source_after_archive_sha_verified=True, checked_unix=2)
        core.publish(source_path.parent/'complete.json', accepted)
        core.publish(archive/'source-snapshot.json', source)
        core.publish(archive/'archive-receipt.json', received)
        core.publish(archive/'source-acceptance.json', accepted)
        for name, row in files.items():
            target = archive/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row['path'], target)
        return base, archive

    def prepared(self, root, step=5):
        base, archive = self.fixture(root, step)
        result = boundary.snapshot(base, step, archive)
        path = Path(result['snapshot'])
        return base, archive, path, json.loads(path.read_text())

    def tar(self, value, *, omitted=(), extra=None, changed=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w|') as archive:
            raw = core.encoded(value); header = tarfile.TarInfo('source-snapshot.json'); header.size = len(raw)
            archive.addfile(header, io.BytesIO(raw))
            for name, row in sorted(value['files'].items()):
                if name in omitted: continue
                raw = Path(row['path']).read_bytes()
                if changed == name: raw = bytes([raw[0] ^ 1])+raw[1:]
                header = tarfile.TarInfo(name); header.size = len(raw); archive.addfile(header, io.BytesIO(raw))
            if extra:
                header = tarfile.TarInfo(extra); header.size = 1; archive.addfile(header, io.BytesIO(b'x'))
        output.seek(0)
        return output

    def test_source_and_HK_complete_eight_file_roundtrip_without_core_changes(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, path, value = self.prepared(Path(td))
            original = {p.relative_to(archive):p.read_bytes() for p in archive.rglob('*') if p.is_file()}
            output = io.BytesIO(); boundary.emit(path, output); output.seek(0)
            with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode='r:') as emitted:
                self.assertEqual(set(emitted.getnames()), boundary.NAMES | {'source-snapshot.json'})
                self.assertEqual(len(emitted.getnames()), 9)
            transferred = boundary.receive(archive, output)
            self.assertFalse((archive/'post-callback-boundary').exists())
            result = boundary.accept(archive, transferred['stage'])
            self.assertEqual(result['status'], boundary.STATUS)
            self.assertEqual(result['file_count'], 8)
            self.assertTrue(result['phase']['post_callback_rng_captured'])
            self.assertEqual(result['phase']['sampling_rng_stage'], 'post_QA5')
            self.assertFalse(result['actual_cuda_restore_accepted'])
            self.assertFalse(result['actual_model_optimizer_rng_dataequal_accepted'])
            self.assertEqual(result['binding']['core_checkpoint_complete_sha256'], value['core_checkpoint_complete_sha256'])
            for name, raw in original.items(): self.assertEqual((archive/name).read_bytes(), raw)
            for name, row in value['files'].items():
                self.assertEqual(core.sha(archive/'post-callback-boundary'/name), row['sha256'])
            with self.assertRaisesRegex(ValueError, 'DESTINATION_EXISTS'):
                boundary.receive(archive, self.tar(value))

    def test_source_does_not_open_the_other_hosts_target_directory(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive = self.fixture(Path(td))
            nonexistent = Path('/HK-only-not-mounted/checkpoints')/archive.parent.name/archive.name
            result = boundary.snapshot(base, 5, nonexistent)
            self.assertEqual(json.loads(Path(result['snapshot']).read_text())['core_archive'], str(nonexistent))
            self.assertFalse(nonexistent.exists())

    def test_200_is_pre_final51713_and_refit_milestones_have_distinct_phase(self):
        for step in boundary.STEPS:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as td:
                _, archive, path, _ = self.prepared(Path(td), step)
                output = io.BytesIO(); boundary.emit(path, output); output.seek(0)
                stage = boundary.receive(archive, output)['stage']
                result = boundary.accept(archive, stage)
                self.assertEqual(result['phase']['probe_round'], min(step//50,3)*50)
                self.assertEqual(result['phase']['final51713_pending_at_boundary'], step == 200)
                if step == 200:
                    self.assertEqual(result['phase']['sampling_rng_stage'], 'post_fixed512_pre_final51713')
                elif step > 5:
                    self.assertEqual(result['phase']['sampling_rng_stage'], 'post_fixed512_and_grounding_refit')

    def test_source_rejects_unaccepted_core_before_writing_new_archive_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive = self.fixture(Path(td))
            acceptance = base/'preparation/checkpoint-archives/global_step_5/complete.json'
            value = json.loads(acceptance.read_text()); value['fresh_source_after_archive_sha_verified'] = False
            acceptance.write_bytes(core.encoded(value))
            with self.assertRaisesRegex(ValueError, 'ACCEPTED_SOURCE_CORE'):
                boundary.snapshot(base, 5, archive)
            self.assertFalse((base/'preparation/post-callback-archives/global_step_5/source-snapshot.json').exists())

    def test_wrong_step_and_path_escape_are_rejected_before_member_copy(self):
        for mutation in ('step', 'path'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                _, archive, _, value = self.prepared(Path(td))
                value = copy.deepcopy(value)
                if mutation == 'step': value['step'] = 50
                else: value['files']['boundary/driver.pt']['path'] = '/outside/driver.pt'
                # Reject the descriptor before reading any data member.
                raw = io.BytesIO()
                with tarfile.open(fileobj=raw, mode='w|') as stream:
                    data = core.encoded(value); header = tarfile.TarInfo('source-snapshot.json'); header.size = len(data)
                    stream.addfile(header, io.BytesIO(data))
                raw.seek(0)
                with self.assertRaises(ValueError): boundary.receive(archive, raw)
                self.assertFalse((archive/'post-callback-boundary').exists())

    def test_missing_unlisted_model_and_corrupt_members_never_publish(self):
        for option in ({'omitted':{'boundary/rank-3.pt'}}, {'extra':'model.safetensors'},
                       {'extra':'../escape'}, {'changed':'boundary/driver.pt'}):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as td:
                _, archive, _, value = self.prepared(Path(td))
                with self.assertRaises(ValueError): boundary.receive(archive, self.tar(value, **option))
                self.assertFalse((archive/'post-callback-boundary').exists())
                self.assertTrue(list(archive.glob('.post-callback-boundary.partial-*')))

    def test_wrong_core_complete_or_unaccepted_target_cannot_receive(self):
        for document_name in ('checkpoint/complete.json', 'source-acceptance.json'):
            with self.subTest(document=document_name), tempfile.TemporaryDirectory() as td:
                _, archive, _, value = self.prepared(Path(td))
                path = archive/document_name; data = json.loads(path.read_text()); data['step'] = 50
                path.write_bytes(core.encoded(data))
                with self.assertRaises(ValueError): boundary.receive(archive, self.tar(value))
                self.assertFalse((archive/'post-callback-boundary').exists())

    def test_post_receive_tamper_and_extra_staging_file_are_rejected_on_accept(self):
        for mutation in ('payload', 'extra'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                _, archive, _, value = self.prepared(Path(td))
                stage = Path(boundary.receive(archive, self.tar(value))['stage'])
                if mutation == 'payload': (stage/'boundary/rank-0.pt').write_bytes(b'tampered')
                else: (stage/'unexpected.txt').write_text('unexpected')
                with self.assertRaises(ValueError): boundary.accept(archive, stage)
                self.assertTrue(stage.exists())
                self.assertFalse((archive/'post-callback-boundary').exists())

    def test_source_emit_rejects_changed_state_before_writing_a_stream(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, path, value = self.prepared(Path(td))
            Path(value['files']['boundary/driver.pt']['path']).write_bytes(b'changed RNG')
            output = io.BytesIO()
            with self.assertRaisesRegex(ValueError, 'SOURCE_MEMBER_SHA'):
                boundary.emit(path, output)
            self.assertEqual(output.getvalue(), b'')

    def test_source_snapshot_rejects_a_missing_rank_without_weakening_the_rolling_seal(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive = self.fixture(Path(td))
            rank = base/'preparation/rolling-checkpoints/global_step_5/boundary/rank-3.pt'
            rank.unlink()
            with self.assertRaisesRegex(ValueError, 'BOUNDARY_MEMBERS_CHANGED'):
                boundary.snapshot(base, 5, archive)
            self.assertFalse((base/'preparation/post-callback-archives/global_step_5/source-snapshot.json').exists())

    def test_accept_does_not_replace_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, value = self.prepared(Path(td))
            stage = Path(boundary.receive(archive, self.tar(value))['stage'])
            destination = archive/'post-callback-boundary'; destination.mkdir()
            marker = destination/'preserved.txt'; marker.write_text('existing result')
            with self.assertRaisesRegex(ValueError, 'DESTINATION_EXISTS'):
                boundary.accept(archive, stage)
            self.assertEqual(marker.read_text(), 'existing result')
            self.assertTrue(stage.exists())


if __name__ == '__main__': unittest.main()
