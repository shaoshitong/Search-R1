import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import checkpoint_archive as archive
from reproduction.easy2560_joint4.control import BASE, launch_command, sha
from reproduction.easy2560_joint4.prepare_training import selection_settings
from reproduction.easy2560_joint4.native_ray_resources import raise_self_soft


class LaunchTests(unittest.TestCase):
    def test_native_four_batch_geometry_without_prefix20_head(self):
        settings = selection_settings('/remote/old-run', 32)
        self.assertEqual(settings['selection_candidate_questions'], 2560)
        self.assertEqual(settings['selection_selected_questions'], 512)
        self.assertTrue(settings['selection_fixed_candidate_budget'])
        self.assertEqual(settings['actor_micro_batch_per_rank'], 32)
        self.assertEqual(settings['actor_logical_micro_batch_per_rank'], 4)
        self.assertIs(settings['all_correct_virtual_zero'], False)
        self.assertNotIn('question_probe', settings)
        command = launch_command(BASE, 'a' * 40, 32)
        for option in ('--gres=gpu:4', '--account=jxu957', '--qos=formal-user', '--partition=emergency_acd'):
            self.assertIn(option, command)
        self.assertEqual(command[-1], '32')
        self.assertIn('/reproduction/easy2560_joint4/train.sh', command[-4])

    def test_physical_fallback_preserves_logical_four(self):
        for physical in (32, 16, 8, 4):
            settings = selection_settings('/remote/old-run', physical)
            self.assertEqual(settings['actor_logical_micro_batch_per_rank'], 4)
            self.assertEqual((32 // physical) * physical, 32)

    def test_nproc_raises_only_soft_and_uses_available_hard(self):
        state = [512, 4096]
        def setlimit(_, value): state[:] = value
        proof = raise_self_soft(getlimit=lambda _: tuple(state), setlimit=setlimit)
        self.assertEqual(state, [4096, 4096])
        self.assertTrue(proof['soft_raised'])
        self.assertTrue(proof['hard_unchanged'])

    def test_easy_workers_retain_the_existing_process_jit_isolation(self):
        from search_g1 import process_jit_cache as jit
        with patch.dict(os.environ, SEARCH_G1_BASE=BASE), patch.object(
                jit, 'process_identity', return_value={'identity': 'actual'}), patch.object(
                jit, 'isolate', return_value={'isolated': True}) as isolate:
            self.assertEqual(jit.configure(), {'isolated': True})
            isolate.assert_called_once_with(Path(BASE), {'identity': 'actual'})


class ArchiveTests(unittest.TestCase):
    def test_executed_controller_code_accepts_only_complete_jsonl_records(self):
        from reproduction.easy2560_joint4.archive_controller import checkpoint_status_code
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cp = root / 'checkpoints/search-g1-common8/actor/global_step_50'
            cp.mkdir(parents=True); (root / 'logs').mkdir()
            (cp / 'complete.json').write_text(json.dumps(dict(run_id='run', files_sha256={
                'training_state/rank-%d.pt' % r: 'hash' for r in range(4)})))
            (root / 'logs/native-allocation-admission.json').write_text(json.dumps(dict(slurm_job_id='123')))
            record = json.dumps(dict(step=50, run_id='run')).encode()
            ledger = root / 'logs/actual-updates.jsonl'
            for raw, expected in ((record + b'\n', True),
                                  (record + b'\n' + b'{"step":51', True),
                                  (record, False)):
                ledger.write_bytes(raw); output = io.StringIO()
                result = __import__('subprocess').CompletedProcess([], 0, 'RUNNING\n', '')
                with patch('subprocess.run', return_value=result), redirect_stdout(output):
                    exec(checkpoint_status_code(root, 50, '123'), {})
                self.assertEqual(json.loads(output.getvalue())['ready'], expected)

    def test_complete_checkpoint_does_not_query_purged_slurm_job(self):
        from reproduction.easy2560_joint4.archive_controller import checkpoint_status_code
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cp = root / 'checkpoints/search-g1-common8/actor/global_step_200'
            cp.mkdir(parents=True); (root / 'logs').mkdir()
            (cp / 'complete.json').write_text(json.dumps(dict(run_id='run', files_sha256={
                'training_state/rank-%d.pt' % r: 'hash' for r in range(4)})))
            (root / 'logs/native-allocation-admission.json').write_text(json.dumps(dict(slurm_job_id='123')))
            ledger = root / 'logs/actual-updates.jsonl'
            ledger.write_text(json.dumps(dict(step=200, run_id='run')) + '\n')
            output = io.StringIO()
            with patch('subprocess.run', side_effect=AssertionError('purged squeue must not run')) as query, redirect_stdout(output):
                exec(checkpoint_status_code(root, 200, '123'), {})
            query.assert_not_called()
            self.assertTrue(json.loads(output.getvalue())['ready'])

    def test_missing_checkpoint_keeps_failed_queue_lookup_unknown(self):
        from reproduction.easy2560_joint4.archive_controller import checkpoint_status_code
        result = __import__('subprocess').CompletedProcess([], 1, '', 'Invalid job id specified')
        with tempfile.TemporaryDirectory() as td:
            output = io.StringIO()
            with patch('subprocess.run', return_value=result), redirect_stdout(output):
                exec(checkpoint_status_code(Path(td), 200, '123'), {})
            state = json.loads(output.getvalue())
            self.assertFalse(state['ready'])
            self.assertEqual(state['queue_returncode'], 1)
            self.assertIn('Invalid job', state['queue_error'])

    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        files = {}
        # The third HF shard prevents a regression to the old two-shard allowlist.
        for name in ('checkpoint/model-00001-of-00003.safetensors',
                     'checkpoint/model-00002-of-00003.safetensors',
                     'checkpoint/model-00003-of-00003.safetensors',
                     *('checkpoint/training_state/rank-%d.pt' % r for r in range(4)),
                     'checkpoint/training_state/driver.pt'):
            path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name * 3).encode())
            files[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        snap = dict(schema_version=1, base=BASE, source_commit='a' * 40,
                    run_id='b' * 32, step=50, files=files,
                    total_bytes=sum(r['bytes'] for r in files.values()), actual_cuda_restore_accepted=False)
        path = source / 'snapshot.json'; path.write_bytes(archive.encoded(snap))
        return path, snap

    def test_full_source_stream_destination_and_source_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, snap = self.fixture(root)
            output = io.BytesIO(); archive.emit(path, output); output.seek(0)
            receipt = archive.receive(root / 'archive', output)
            accepted = archive.accept(path, receipt)
            self.assertEqual(accepted['status'], archive.STATUS)
            self.assertTrue(accepted['fresh_source_after_archive_sha_verified'])
            self.assertFalse(accepted['actual_cuda_restore_accepted'])
            self.assertEqual(receipt['file_count'], 8)
            self.assertTrue((root / 'archive/checkpoint/model-00003-of-00003.safetensors').is_file())

    def test_fresh_source_mutation_is_rejected_after_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, snap = self.fixture(root)
            output = io.BytesIO(); archive.emit(path, output); output.seek(0)
            receipt = archive.receive(root / 'archive', output)
            Path(next(iter(snap['files'].values()))['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'FRESH_SOURCE_SHA_DIFFERS'):
                archive.accept(path, receipt)

    def test_stream_never_accepts_missing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path, snap = self.fixture(root)
            output = io.BytesIO(); archive.emit(path, output)
            # Truncation is inside the final state payload, not only tar padding.
            raw = output.getvalue()[:1024]
            with self.assertRaises((ValueError, __import__('tarfile').ReadError)):
                archive.receive(root / 'archive', io.BytesIO(raw))


class GroundingReuseTests(unittest.TestCase):
    def test_original_fitted_bytes_new_run_version_and_provenance(self):
        from reproduction.easy2560_joint4 import adopt_reward_zero as reuse
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); old = root / 'old'; base = root / 'new'
            folder = old / 'artifacts/round_0-attempt-original'
            for directory in ('heads', 'dumps'): (folder / directory).mkdir(parents=True)
            for name in ('commit_probe.pkl', 'boundary_probe.pkl', 'calibration-report.json'):
                (folder / 'heads' / name).write_bytes(name.encode())
            (folder / 'dumps/calibration-label-proof.json').write_bytes(b'proof')
            shared = dict(initial_model='/remote/official', initial_model_sha256={'config.json': 'a' * 64},
                          model_revision='official-revision', calibration_question_ids=['q1', 'q2'])
            previous = dict(shared, run_id='old-run')
            (old / 'logs').mkdir()
            (old / 'logs/run-manifest.json').write_text(json.dumps(previous))
            original = dict(round=0, run_id='old-run', version='old-run:0:original',
                            encoder='/remote/official', encoder_sha256={'config.json': 'a' * 64},
                            commit=str(folder / 'heads/commit_probe.pkl'),
                            boundary=str(folder / 'heads/boundary_probe.pkl'),
                            label_proof=dict(question_ids=['q1', 'q2']),
                            calibration_report=dict(configured_cv_folds=2, require_all_cv_folds=True),
                            head_sha256={n: sha(folder / 'heads' / n) for n in ('commit_probe.pkl', 'boundary_probe.pkl')},
                            calibration_report_sha256=sha(folder / 'heads/calibration-report.json'),
                            dump_sha256={'calibration-label-proof.json': sha(folder / 'dumps/calibration-label-proof.json')},
                            published_unix=1)
            archive = folder / 'version-manifest.json'; archive.write_text(json.dumps(original))
            (old / 'logs/formal-inline-initial-probe.json').write_text(json.dumps(dict(
                archive=str(archive.relative_to(old)), archive_sha256=sha(archive))))
            run = dict(shared, run_id='new-run', source_commit='b' * 40,
                       easy2560_selection=True, initial_reward_probe_source_base=str(old))
            with patch.object(reuse, 'own_base', side_effect=Path):
                receipt = reuse.adopt(base, run)
                target = Path(receipt['target_archive']); live = json.loads(target.read_text())
                self.assertNotEqual(live['version'], original['version'])
                self.assertEqual(live['run_id'], 'new-run')
                self.assertFalse(live['new_calibration_generated'])
                checked = reuse.verify_adopted(base, run, live, target)
                self.assertEqual(checked, receipt)
                self.assertEqual((folder / 'heads/commit_probe.pkl').stat().st_ino,
                                 (target.parent / 'heads/commit_probe.pkl').stat().st_ino)
                (folder / 'heads/commit_probe.pkl').write_bytes(b'corruption')
                with self.assertRaisesRegex(ValueError, 'FITTED_BYTES_CHANGED'):
                    reuse.verify_adopted(base, run, live, target)


if __name__ == '__main__': unittest.main()
