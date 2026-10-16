"""Actual local byte files only: no torch/CUDA/state reload acceptance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('checkpoint_archive', Path(__file__).with_name('checkpoint_archive.py'))
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class Fixture:
    def __init__(self, root, step=5):
        self.root = Path(root).resolve()  # macOS /var aliases /private/var.
        self.personal = self.root/'experiments'
        self.base = self.personal/'qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016'
        self.model = self.personal/'official-model'
        self.data = self.personal/'official-data'
        self.archives = self.root/'backup/checkpoints'
        self.run_id, self.commit, self.step = 'a'*32, 'b'*40, step
        self.cp = self.base/'checkpoints/search-g1-common8/actor'/('global_step_' + str(step))
        self.pair = self.base/'artifacts/question-probe/completed-pairs'/('global_step_' + str(step))
        for path in (self.model, self.data, self.cp/'training_state', self.pair/'boundary-worker-rng',
                     self.base/'data', self.base/'assets/prefix20', self.base/'artifacts/reward-heads'):
            path.mkdir(parents=True, exist_ok=True)
        self.model_shas = {}
        for name in archive.MODEL_SHA256:
            path = self.model/name
            path.write_bytes(b'constructed frozen model dependency ' + name.encode())
            self.model_shas[name] = archive.sha(path)
        self.asset_specs = {}
        for name in archive.ASSETS:
            path = self.base/'assets/prefix20'/name
            path.write_bytes(b'constructed initial probe asset ' + name.encode())
            self.asset_specs[name] = (path.stat().st_size, archive.sha(path))
        inputs = {}
        for name in archive.DATA_NAMES:
            path = (self.data if name in archive.LINKED_DATA_NAMES else self.base/'data')/name
            path.write_bytes(b'constructed canonical data ' + name.encode())
            if name in archive.LINKED_DATA_NAMES:
                (self.base/'data'/name).symlink_to(path)
            inputs[name] = archive.sha(path)
        config = dict(probe_architecture=archive.ARCHITECTURE, candidate_questions=5120, selected_questions=512,
            exploration_questions=0, responses_per_question=5, head_learning_rate=1e-6,
            head_batch_size=256, head_epochs_per_outer=1, replay_window_outer=0,
            model_dir=str(self.model), head_safetensors=str(self.base/'assets/prefix20/head.safetensors'),
            probe_config=str(self.base/'assets/prefix20/probe-config.json'),
            eligible_pool_path=str(self.base/'assets/prefix20/eligible-question-pool.json'))
        self.run = dict(run_id=self.run_id, source_commit=self.commit, training_world_size=4,
            target_updates=200, data_selection_mode='prefix20_probe', initial_model=str(self.model),
            reference_model=str(self.model), model_revision=archive.REVISION,
            initial_model_sha256=self.model_shas, question_probe=config, input_sha256=inputs)
        names = ['complete.json', 'config.json', 'model.safetensors.index.json', 'weights.safetensors',
                 'training_state/manifest.json', 'training_state/driver.pt', 'training_state/question-probe-state.pt']
        names += ['training_state/rank-%d.pt' % r for r in range(4)]
        names += ['training_state/prefix20-rank-%d.pt' % r for r in range(4)]
        for name in names:
            (self.cp/name).write_bytes(b'constructed paired bytes ' + name.encode())
        (self.cp/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight': 'weights.safetensors'})))
        self.write_json(self.pair/'run-manifest.json', self.run)
        (self.pair/'boundary-driver.pt').write_bytes(b'constructed driver, selector and data/RNG state')
        for rank in range(4):
            (self.pair/'boundary-worker-rng'/('rank-%d.pt' % rank)).write_bytes(b'constructed boundary RNG ' + bytes([rank]))
        rewards = []
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            reward = self.base/'artifacts/reward-heads'/name
            reward.write_bytes(b'constructed reward dependency ' + name.encode())
            rewards.append(reward)
        round_id = min(step//50, 3)*50
        self.reward = dict(run_id=self.run_id, round=round_id, version=self.run_id+':'+str(round_id)+':fixture',
            commit=str(rewards[0]), boundary=str(rewards[1]), encoder=str(self.model),
            head_sha256={p.name: archive.sha(p) for p in rewards}, encoder_sha256=self.model_shas)
        self.write_json(self.pair/'reward-probe-manifest.json', self.reward)
        ids = ['%032x' % i for i in range(1, step+1)]
        self.receipt = dict(schema_version=1, status=archive.PAIR_STATUS, kind='question_probe_pair',
            step=step, run_id=self.run_id, checkpoint=str(self.cp), source_base=str(self.base), world_size=4,
            actor_snapshot_id=self.run_id+':'+str(step), question_probe_state_path=str(self.cp/'training_state/question-probe-state.pt'),
            head_path=str(self.cp/'training_state/prefix20-rank-0.pt'), head_version='prefix20-epoch68-outer-%04d' % step,
            files=[self.file_record(self.cp/name, relative_path=name) for name in sorted(names)],
            source_run_manifest=self.file_record(self.pair/'run-manifest.json'),
            boundary_driver=self.file_record(self.pair/'boundary-driver.pt'),
            reward_probe_manifest=self.file_record(self.pair/'reward-probe-manifest.json'),
            boundary_worker_files=[dict(self.file_record(self.pair/'boundary-worker-rng'/('rank-%d.pt' % rank)), rank=rank) for rank in range(4)],
            ranks=[dict(rank=rank, completed_outer=step, scheduler_last_epoch=step, scheduler_step_count=step+1,
                actual_optimizer_writes=step*20, last_attempt_index=step, used_attempt_ids=ids,
                optimizer_tensors_finite=True, policy_rejections=0) for rank in range(4)],
            attempt_progress=dict(completed_outer=step, attempt_index=step, cumulative_writes=step*20,
                aborted_attempts=0, full_budget_outer=step),
            historical_artifacts=dict(checkpoints=[], development={}, reward_archives=[],
                raw_training_prefix_verified=False, scope='constructed fixture'),
            reward_files=[self.file_record(p) for p in rewards]+[self.file_record(self.model/name) for name in sorted(self.model_shas)],
            reward_probe_version=self.reward['version'], reward_probe_round=round_id, feature_spec_sha='f'*64,
            driver_state=dict(completed_outer=step, source_consumed=step*5120),
            rng_boundary_stage='post_development_and_probe_adoption', scope='constructed fixture only',
            checked_unix=1, hf_saved_tensor_count=1, actual_cuda_restore_accepted=False)
        self.write_json(self.pair/'source-receipt.json', self.receipt)
        self.patches = [patch.object(archive, 'PERSONAL_ROOT', self.personal), patch.object(archive, 'MODEL', self.model),
            patch.object(archive, 'DATA', self.data), patch.object(archive, 'ARCHIVES', self.archives),
            patch.object(archive, 'MODEL_SHA256', self.model_shas), patch.object(archive, 'ASSETS', self.asset_specs),
            patch.object(archive.os, 'getuid', return_value=14090)]
        for item in self.patches:
            item.start()

    @staticmethod
    def write_json(path, value):
        Path(path).write_bytes(archive.document_bytes(value))

    @staticmethod
    def file_record(path, **extra):
        info = path.stat()
        return dict(path=str(path), inode=info.st_ino, bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                    full_read_sha256=archive.sha(path), **extra)

    def close(self):
        for item in reversed(self.patches):
            item.stop()

    @property
    def kwargs(self):
        return dict(base=self.base, commit=self.commit, run_id=self.run_id, step=self.step)

    def snapshot(self):
        return archive.snapshot(self.base, self.step, commit=self.commit, run_id=self.run_id)

    def mirror(self, snap):
        target = self.archives/self.run_id/('global_step_' + str(self.step))
        target.mkdir(parents=True)
        shutil.copyfile(snap['snapshot_file'], target/'source-snapshot.json')
        for row in snap['snapshot']['files']:
            path = target/row['payload_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row['source_path'], path)
        return target

    def archive(self, snap, target):
        with patch.object(archive.socket, 'gethostname', return_value='xielab3'):
            return archive.archive_readback(target, target/'source-snapshot.json', snap['snapshot_sha256'], **self.kwargs)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.fixture.close)

    def test_full_source_archive_source_chain_preserves_bytes_and_never_claims_cuda(self):
        f = self.fixture
        snap = f.snapshot()
        archive.validate_snapshot(snap['snapshot'], **f.kwargs)
        self.assertEqual(len(snap['snapshot']['dependencies']), 8+3+9)
        target = f.mirror(snap)
        proof = f.archive(snap, target)
        result = archive.accept_archive(f.base, snap['snapshot_file'], snap['snapshot_sha256'], proof,
            commit=f.commit, run_id=f.run_id, step=f.step)
        self.assertEqual(result['status'], archive.STATUS)
        self.assertFalse(result['actual_cuda_restore_accepted'])
        complete = archive._read_json(result['complete_file'])
        self.assertTrue(complete['source_readback']['all_source_files_fresh_sha_pass'])
        self.assertEqual(complete['hpc3_archive_copies'], 0)
        self.assertEqual(complete['hpc3_restore_targets_created'], 0)
        self.assertEqual(archive.sha(target/'archive-receipt.json'), proof['archive_receipt_sha256'])
        for row in snap['snapshot']['files']:
            self.assertTrue(Path(row['source_path']).is_file())
            self.assertEqual(archive.sha(target/row['payload_path']), row['sha256'])
        with self.assertRaises(FileExistsError):
            archive.accept_archive(f.base, snap['snapshot_file'], snap['snapshot_sha256'], proof,
                commit=f.commit, run_id=f.run_id, step=f.step)

    def test_binding_rejects_wrong_run_commit_step_topology_and_noncanonical_base(self):
        f = self.fixture
        snap = f.snapshot()['snapshot']
        for fields in (dict(run_id='c'*32), dict(commit='b'*7), dict(step=201),
                       dict(base=f.personal/'another-task'), dict(base=str(f.base)+'/../'+f.base.name)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                archive.validate_snapshot(snap, **dict(f.kwargs, **fields))
        changed = copy.deepcopy(snap)
        changed['world_size'] = 8
        with self.assertRaisesRegex(ValueError, 'BINDING'):
            archive.validate_snapshot(changed, **f.kwargs)

    def test_snapshot_rejects_missing_extra_duplicate_escaping_or_misbound_file_records(self):
        f = self.fixture
        snapshot = f.snapshot()['snapshot']
        mutations = [lambda s: s['files'].pop(),
            lambda s: s['files'].append(copy.deepcopy(s['files'][0])),
            lambda s: s['files'][0].update(payload_path='payload/../../outside'),
            lambda s: s['files'][0].update(source_path=str(f.personal/'unrelated-task/file')),
            lambda s: s['files'][0].update(bytes=True),
            lambda s: s['files'][0].update(sha256='f'*64),
            lambda s: s.update(bytes=s['bytes']+1),
            lambda s: s['dependencies'].pop('frozen_model/tokenizer.json')]
        for mutate in mutations:
            changed = copy.deepcopy(snapshot)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                archive.validate_snapshot(changed, **f.kwargs)

    def test_missing_actor_probe_driver_or_boundary_rank_is_rejected_before_claim(self):
        f = self.fixture
        for name in ('training_state/rank-3.pt', 'training_state/prefix20-rank-3.pt', 'training_state/driver.pt'):
            changed = copy.deepcopy(f.receipt)
            changed['files'] = [r for r in changed['files'] if r['relative_path'] != name]
            f.write_json(f.pair/'source-receipt.json', changed)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'FILE_SET_INCOMPLETE'):
                f.snapshot()
        changed = copy.deepcopy(f.receipt)
        changed['boundary_worker_files'].pop()
        f.write_json(f.pair/'source-receipt.json', changed)
        with self.assertRaisesRegex(ValueError, 'BOUNDARY_RNG'):
            f.snapshot()
        self.assertFalse((f.base/'preparation/checkpoint-archives/global_step_5/claim.json').exists())

    def test_checkpoint_extra_file_or_dangling_symlink_is_rejected(self):
        f = self.fixture
        path = f.cp/'unexpected.pt'
        path.write_bytes(b'not in producer receipt')
        with self.assertRaisesRegex(ValueError, 'EXACT_PAYLOAD_TREE'):
            f.snapshot()
        path.unlink()
        path.symlink_to(f.root/'does-not-exist')
        with self.assertRaisesRegex(ValueError, 'UNSAFE_TREE'):
            f.snapshot()

    def test_official_data_links_are_exact_and_model_dependencies_cannot_be_symlinks(self):
        f = self.fixture
        bound = f.base/'data/train.parquet'
        bound.unlink()
        bound.symlink_to(f.data/'validation.parquet')
        with self.assertRaisesRegex(ValueError, 'DATA_LINK'):
            f.snapshot()
        bound.unlink()
        bound.symlink_to(f.data/'train.parquet')
        model = f.model/'tokenizer.json'
        content = model.read_bytes()
        model.unlink()
        other = f.root/'same-tokenizer-bytes'
        other.write_bytes(content)
        model.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'SYMLINK_COMPONENT'):
            f.snapshot()

    def test_fresh_source_readback_detects_same_size_same_mtime_tampering(self):
        f = self.fixture
        snap = f.snapshot()
        path = f.cp/'training_state/prefix20-rank-3.pt'
        old = path.stat()
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        self.assertEqual(path.stat().st_ino, old.st_ino)
        with self.assertRaisesRegex(ValueError, 'FULL_SHA'):
            archive.source_readback(snap['snapshot_file'], snap['snapshot_sha256'], **f.kwargs)

    def test_source_readback_rechecks_original_data_links_after_archive_copy(self):
        f = self.fixture
        snap = f.snapshot()
        proof = f.archive(snap, f.mirror(snap))
        link = f.base/'data/train.parquet'
        link.unlink()
        link.symlink_to(f.data/'validation.parquet')
        with self.assertRaisesRegex(ValueError, 'OFFICIAL_DATA_LINK'):
            archive.accept_archive(f.base, snap['snapshot_file'], snap['snapshot_sha256'], proof,
                commit=f.commit, run_id=f.run_id, step=f.step)
        self.assertFalse((Path(snap['snapshot_file']).parent/'complete.json').exists())

    def test_archive_rejects_wrong_host_and_payload_tamper_before_receipt(self):
        f = self.fixture
        snap = f.snapshot()
        target = f.mirror(snap)
        with patch.object(archive.socket, 'gethostname', return_value='not-xielab3'):
            with self.assertRaisesRegex(ValueError, 'HKUST3_OBSERVER'):
                archive.archive_readback(target, target/'source-snapshot.json', snap['snapshot_sha256'], **f.kwargs)
        row = snap['snapshot']['files'][-1]
        (target/row['payload_path']).write_bytes(b'corrupt destination')
        with self.assertRaisesRegex(ValueError, 'FILE_BYTES|FULL_SHA'):
            f.archive(snap, target)
        self.assertFalse((target/'archive-receipt.json').exists())

    def test_archive_rejects_extra_empty_directory_and_symlink_directory(self):
        f = self.fixture
        snap = f.snapshot()
        target = f.mirror(snap)
        extra = target/'payload/extra-empty-directory'
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, 'EXACT_PAYLOAD_TREE'):
            f.archive(snap, target)
        extra.rmdir()
        extra.symlink_to(f.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'UNSAFE_TREE_DIRECTORY'):
            f.archive(snap, target)

    def test_accept_rejects_rewritten_archive_claims_even_with_recomputed_document_hash(self):
        f = self.fixture
        snap = f.snapshot()
        proof = f.archive(snap, f.mirror(snap))
        changes = [lambda p: p.update(actual_cuda_restore_accepted=True),
            lambda p: p.update(run_id='c'*32),
            lambda p: p.update(hpc3_restore_targets_created=1),
            lambda p: p['files'][0].update(archive_path='/backup/elsewhere'),
            lambda p: p['files'][0]['archive_identity'].update(bytes=999),
            lambda p: p.update(all_archive_files_fsync_completed=False)]
        for mutate in changes:
            changed = copy.deepcopy(proof)
            mutate(changed['archive_receipt'])
            changed['archive_receipt_sha256'] = archive.document_sha(changed['archive_receipt'])
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                archive.validate_archive_proof(changed, snap['snapshot'], snap['snapshot_sha256'])

    def test_archive_fsync_failure_cannot_publish_acceptance(self):
        f = self.fixture
        snap = f.snapshot()
        target = f.mirror(snap)
        with patch.object(archive.os, 'fsync', side_effect=OSError('constructed storage fsync failure')):
            with self.assertRaisesRegex(OSError, 'fsync failure'):
                f.archive(snap, target)
        self.assertFalse((target/'archive-receipt.json').exists())

    def test_duplicate_json_key_is_not_silently_accepted(self):
        path = self.fixture.root/'duplicate.json'
        path.write_text('{"run_id": "a", "run_id": "b"}')
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON'):
            archive._read_json(path)

    def test_all_five_milestones_are_supported_but_update201_is_never_admitted(self):
        for step in archive.STEPS:
            with self.subTest(step=step):
                base = archive.binding(self.fixture.base, self.fixture.commit, self.fixture.run_id, step)
                self.assertEqual(base, self.fixture.base)
        for step in (0, 4, 201, True):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, 'MILESTONE'):
                archive.binding(self.fixture.base, self.fixture.commit, self.fixture.run_id, step)

    def test_all_five_milestones_have_real_source_archive_and_final_readback_bytes(self):
        for step in archive.STEPS:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as temporary:
                f = Fixture(temporary, step=step)
                try:
                    snap = f.snapshot()
                    proof = f.archive(snap, f.mirror(snap))
                    result = archive.accept_archive(f.base, snap['snapshot_file'], snap['snapshot_sha256'], proof,
                        commit=f.commit, run_id=f.run_id, step=step)
                    self.assertEqual(result['step'], step)
                    self.assertFalse(result['actual_cuda_restore_accepted'])
                finally:
                    f.close()

    def test_unknown_producer_record_or_missing_reward_dependency_is_not_archived(self):
        f = self.fixture
        changed = copy.deepcopy(f.receipt)
        changed['unrecognized_payload'] = f.file_record(f.base/'data/audit.json')
        f.write_json(f.pair/'source-receipt.json', changed)
        with self.assertRaisesRegex(ValueError, 'PRODUCER_RECEIPT_SCHEMA'):
            f.snapshot()
        changed = copy.deepcopy(f.receipt)
        changed['reward_files'].pop()
        f.write_json(f.pair/'source-receipt.json', changed)
        with self.assertRaisesRegex(ValueError, 'REWARD_HEAD_OR_ENCODER_CLOSURE'):
            f.snapshot()

    def test_hkust3_dependency_hardlinks_are_full_read_and_fsynced_per_archive_path(self):
        f = self.fixture
        snap = f.snapshot()
        target = f.mirror(snap)
        shared = f.archives/f.run_id/'immutable-dependencies'
        shared.mkdir()
        for row in snap['snapshot']['files']:
            if row['source_path'].startswith(str(f.model)+'/'):
                copy_path = target/row['payload_path']
                shared_file = shared/Path(row['source_path']).name
                copy_path.rename(shared_file)
                os.link(shared_file, copy_path)
                self.assertEqual(copy_path.stat().st_ino, shared_file.stat().st_ino)
        proof = f.archive(snap, target)
        self.assertTrue(proof['archive_receipt']['all_archive_files_fresh_sha_pass'])
        self.assertTrue(proof['archive_receipt']['all_archive_files_fsync_completed'])


if __name__ == '__main__':
    unittest.main()
