"""CPU temporary-file fixtures; no SSH, Torch, model load, or training."""
import copy
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import post_callback_archive as boundary
from reproduction.easy2560_joint4 import post_callback_dependencies as deps
from reproduction.easy2560_joint4 import test_post_callback_archive as boundary_tests
from search_g1 import easy2560_rolling_checkpoint as rolling

core = boundary.core


def write(path, value):
    path.write_bytes(core.encoded(value))


class DependencyArchiveTests(unittest.TestCase):
    def fixture(self, root, step=50, covered=False, older_encoder=False, omit_encoder=False, restored=False):
        base, archive = boundary_tests.BoundaryArchiveTests().fixture(root, step)
        round_number = boundary.phase(step)['probe_round']
        folder = base/'artifacts'/('round_%d-attempt-' % round_number+'d'*32)
        (folder/'heads').mkdir(parents=True); (folder/'dumps').mkdir()
        payloads = {'heads/commit_probe.pkl': b'actual fitted commit bytes',
                    'heads/boundary_probe.pkl': b'actual fitted boundary bytes',
                    'heads/calibration-report.json': core.encoded({'both_folds':True}),
                    'dumps/nq-hotpot-common_0.npz': b'preserved calibration features, not a model',
                    'dumps/nq-hotpot-common_0.jsonl': b'{"calibration":true}\n',
                    'dumps/calibration-label-proof.json': core.encoded({'question_count':300})}
        for name, raw in payloads.items(): (folder/name).write_bytes(raw)
        checkpoint = base/'checkpoints/search-g1-common8/actor'/('global_step_%d' % step)
        encoder = checkpoint
        model_name = 'model-00001-of-00001.safetensors'
        if older_encoder:
            encoder = base/'checkpoints/search-g1-common8/actor/global_step_150'
            encoder.mkdir(parents=True, exist_ok=True)
            (encoder/model_name).write_bytes(b'earlier exact round150 model')
        probe = dict(run_id='b'*32, version='b'*32+':%d:fixture' % round_number, round=round_number,
            encoder=str(encoder), encoder_sha256={model_name:core.sha(encoder/model_name)},
            commit=str(folder/'heads/commit_probe.pkl'), boundary=str(folder/'heads/boundary_probe.pkl'),
            head_sha256={name:core.sha(folder/'heads'/name) for name in ('commit_probe.pkl','boundary_probe.pkl')},
            calibration_report_sha256=core.sha(folder/'heads/calibration-report.json'),
            dump_sha256={path.name:core.sha(path) for path in (folder/'dumps').iterdir()})
        write(folder/'version-manifest.json', probe)
        if restored:
            origin = root.resolve()/'origin'; original = origin/'artifacts'/('round_%d-attempt-' % round_number+'a'*32)
            shutil.copytree(folder, original)
            old = dict(probe, commit=str(original/'heads/commit_probe.pkl'), boundary=str(original/'heads/boundary_probe.pkl'))
            write(original/'version-manifest.json', old)
            for path in (folder/'dumps').iterdir(): path.unlink()
            restore_path = base/'preparation/restore-source-receipt.json'
            write(restore_path, dict(kind='easy2560_native4', status='EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS',
                run_id=probe['run_id'], source_base=str(origin), reward_probe_manifest=dict(
                    path=str(original/'version-manifest.json'), full_read_sha256=core.sha(original/'version-manifest.json'))))
            run_path = base/'logs/run-manifest.json'; run = json.loads(run_path.read_text())
            run['full_state_restore'] = dict(kind='easy2560_native4', source_receipt=str(restore_path),
                source_receipt_sha256=core.sha(restore_path), source_base=str(origin))
            write(run_path, run)
        rolling = base/'preparation/rolling-checkpoints'/('global_step_%d' % step)
        write(rolling/'boundary/reward-probe-manifest.json', probe)
        manifest = json.loads((rolling/'boundary/manifest.json').read_text())
        manifest['reward_probe_manifest'] = boundary.record(rolling/'boundary/reward-probe-manifest.json')
        manifest['probe_version'] = probe['version']; write(rolling/'boundary/manifest.json', manifest)
        seal = json.loads((rolling/'full-manifest.json').read_text())
        seal['boundary_files'] = {p.name:deps._record(p) for p in (rolling/'boundary').iterdir()}
        seal['boundary_manifest'] = seal['boundary_files']['manifest.json']; seal['boundary'] = manifest
        write(rolling/'full-manifest.json', seal)
        source_path = base/'preparation/checkpoint-archives'/('global_step_%d' % step)/'source-snapshot.json'
        source = json.loads(source_path.read_text())
        source['files']['metadata/run-manifest.json'] = boundary.record(base/'logs/run-manifest.json')
        if omit_encoder:
            source['files'].pop('checkpoint/'+model_name)
        if older_encoder and not omit_encoder:
            source['files']['encoders/round-150/'+model_name] = boundary.record(encoder/model_name)
        if covered:
            for path in folder.rglob('*'):
                if path.is_file(): source['files']['grounding/'+folder.name+'/'+str(path.relative_to(folder))] = boundary.record(path)
        source['total_bytes'] = sum(r['bytes'] for r in source['files'].values())
        write(source_path, source); write(archive/'source-snapshot.json', source)
        for name, row in source['files'].items():
            target = archive/name; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): shutil.copyfile(row['path'], target)
        received = json.loads((archive/'archive-receipt.json').read_text())
        received.update(snapshot_sha256=core.sha(source_path), file_count=len(source['files']), total_bytes=source['total_bytes'])
        write(archive/'archive-receipt.json', received)
        accepted = dict(received, status=core.STATUS, source_snapshot=str(source_path),
                        fresh_source_after_archive_sha_verified=True, checked_unix=2)
        write(source_path.parent/'complete.json', accepted); write(archive/'source-acceptance.json', accepted)
        descriptor = boundary.snapshot(base, step, archive)
        stream = io.BytesIO(); boundary.emit(descriptor['snapshot'], stream); stream.seek(0)
        transferred = boundary.receive(archive, stream); boundary.accept(archive, transferred['stage'])
        return base, archive, probe

    def prepared(self, root, **kwargs):
        base, archive, probe = self.fixture(root, **kwargs)
        saved = deps.snapshot(base, kwargs.get('step', 50), archive)
        return base, archive, probe, saved, json.loads(Path(saved['snapshot']).read_text())

    def tar(self, value, *, omit=None, extra=None, changed=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w|') as archive:
            raw = core.encoded(value); head = tarfile.TarInfo('source-snapshot.json'); head.size = len(raw)
            archive.addfile(head, io.BytesIO(raw))
            for name, row in sorted(value['dependencies'].items()):
                if row['storage'] != 'transfer' or name == omit: continue
                raw = Path(row['path']).read_bytes()
                if changed == name: raw = bytes([raw[0]^1])+raw[1:]
                head = tarfile.TarInfo(name); head.size = len(raw); archive.addfile(head, io.BytesIO(raw))
            if extra:
                head = tarfile.TarInfo(extra); head.size = 1; archive.addfile(head, io.BytesIO(b'x'))
        output.seek(0); return output

    def test_new_round50_dependencies_roundtrip_and_core_boundary_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, probe, saved, value = self.prepared(Path(td))
            before = {str(p.relative_to(archive)):p.read_bytes() for p in archive.rglob('*') if p.is_file()}
            self.assertEqual(saved['file_count'], 7); self.assertEqual(saved['transfer_file_count'], 7)
            stream = io.BytesIO(); deps.emit(saved['snapshot'], stream); stream.seek(0)
            with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode='r:') as tar:
                self.assertFalse(any('safetensors' in x for x in tar.getnames()))
            stage = deps.receive(archive, stream)['stage']; receipt = deps.accept(archive, stage)
            self.assertEqual(receipt['status'], deps.STATUS); self.assertFalse(receipt['encoder_weights_copied'])
            self.assertEqual(receipt['reward_version'], probe['version'])
            result = deps.inspect(archive)
            self.assertTrue(result['all_manifest_dependencies_covered']); self.assertTrue(result['fresh_transferred_dependency_sha_checked'])
            self.assertFalse(result['core_covered_payload_rehashed'])
            self.assertEqual(result['encoder_file_count'], 1); self.assertEqual(len(result['mappings']), 8)
            self.assertFalse(result['actual_model_optimizer_rng_dataequal_accepted'])
            for name, raw in before.items(): self.assertEqual((archive/name).read_bytes(), raw)
            self.assertEqual(json.loads((Path(probe['commit']).parent.parent/'version-manifest.json').read_text()), probe)

    def test_cp5_covered_dependencies_transfer_only_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, saved, _ = self.prepared(Path(td), step=5, covered=True)
            self.assertEqual(saved['transfer_file_count'], 0); self.assertEqual(saved['transfer_bytes'], 0)
            stream = io.BytesIO(); deps.emit(saved['snapshot'], stream); stream.seek(0)
            with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode='r:') as tar:
                self.assertEqual(tar.getnames(), ['source-snapshot.json'])
            stage = deps.receive(archive, stream)['stage']; deps.accept(archive, stage)
            self.assertEqual(deps.inspect(archive)['transfer_file_count'], 0)

    def test_actual_restored_shape_closes_missing_dumps_via_bound_single_hop(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, probe, saved, value = self.prepared(Path(td), step=5, covered=True, restored=True)
            self.assertEqual(value['origin']['kind'], 'sha_bound_easy_restore_single_hop')
            self.assertEqual(saved['file_count'], 10); self.assertEqual(saved['transfer_file_count'], 6)
            for name in probe['dump_sha256']:
                row = value['dependencies']['grounding/dumps/'+name]
                self.assertIn('/origin/artifacts/', row['path']); self.assertEqual(row['storage'], 'transfer')
            stream = io.BytesIO(); deps.emit(saved['snapshot'], stream); stream.seek(0)
            stage = deps.receive(archive, stream)['stage']; deps.accept(archive, stage)
            result = deps.inspect(archive)
            self.assertTrue(result['covered_core_sha_ack_verified']); self.assertFalse(result['core_covered_payload_rehashed'])
            self.assertEqual(result['reward_version'], probe['version']); self.assertEqual(result['file_count'], 10)
            self.assertFalse(any((Path(probe['commit']).parent.parent/'dumps').iterdir()))

    def test_restore_origin_receipt_sha_change_cannot_be_used_as_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, _ = self.fixture(Path(td), step=5, covered=True, restored=True)
            receipt = base/'preparation/restore-source-receipt.json'
            value = json.loads(receipt.read_text()); value['source_base'] = '/different/source'; write(receipt, value)
            with self.assertRaisesRegex(ValueError, 'ORIGIN_RECEIPT_SHA_OR_IDENTITY_DIFFERS'):
                deps.snapshot(base, 5, archive)

    def test_default_paths_never_hash_covered_model_tensor_bodies(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, probe = self.fixture(Path(td), step=5, covered=True)
            model = next(iter(probe['encoder_sha256']))
            forbidden = {(Path(probe['encoder'])/model).resolve(), (archive/'checkpoint'/model).resolve()}
            original = deps._record
            def guarded(path, *args, **kwargs):
                self.assertNotIn(Path(path).resolve(), forbidden, 'Default pipeline reread a covered model tensor')
                return original(path, *args, **kwargs)
            with patch.object(deps, '_record', side_effect=guarded):
                saved = deps.snapshot(base, 5, archive)
                stream = io.BytesIO(); deps.emit(saved['snapshot'], stream); stream.seek(0)
                stage = deps.receive(archive, stream)['stage']; receipt = deps.accept(archive, stage)
                result = deps.inspect(archive)
            self.assertTrue(receipt['fresh_transferred_dependency_sha_checked'])
            self.assertTrue(result['covered_core_sha_ack_verified']); self.assertFalse(result['core_covered_payload_rehashed'])

    def test_cp200_round150_encoder_maps_to_existing_core150_alias(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, _, value = self.prepared(Path(td), step=200, covered=True, older_encoder=True)
            self.assertEqual(value['reward_round'], 150)
            row = next(iter(value['encoder_files'].values()))
            self.assertTrue(row['core_member'].startswith('encoders/round-150/'))
            stage = deps.receive(archive, self.tar(value))['stage']; deps.accept(archive, stage)
            self.assertEqual(deps.inspect(archive)['reward_round'], 150)

    def test_uncovered_encoder_rejects_without_bigmodel_transfer(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, _ = self.fixture(Path(td), omit_encoder=True)
            with self.assertRaisesRegex(ValueError, 'ENCODER_NOT_COVERED'):
                deps.snapshot(base, 50, archive)
            self.assertFalse((archive/deps.CHILD).exists())
            with self.assertRaisesRegex(ValueError, 'SOURCE_ALREADY_ATTEMPTED'):
                deps.snapshot(base, 50, archive)

    def test_missing_corrupt_and_unlisted_dump_never_publish(self):
        dump = 'grounding/dumps/nq-hotpot-common_0.npz'
        for kwargs in ({'omit':dump}, {'changed':dump}, {'extra':'../escape'}, {'extra':'grounding/dumps/not-listed.npz'}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as td:
                _, archive, _, _, value = self.prepared(Path(td))
                with self.assertRaises(ValueError): deps.receive(archive, self.tar(value, **kwargs))
                self.assertFalse((archive/deps.CHILD).exists())
                self.assertTrue(list(archive.glob('.post-callback-dependencies.partial-*')))

    def test_source_member_change_is_rejected_before_stream(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, _, saved, value = self.prepared(Path(td))
            Path(value['dependencies']['grounding/heads/commit_probe.pkl']['path']).write_bytes(b'changed')
            output = io.BytesIO()
            with self.assertRaisesRegex(ValueError, 'SOURCE_IDENTITY_CHANGED'): deps.emit(saved['snapshot'], output)
            self.assertEqual(output.getvalue(), b'')

    def test_default_core_mapping_uses_ack_and_explicit_deep_checks_encoder(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, _, value = self.prepared(Path(td))
            row = next(iter(value['encoder_files'].values())); path = archive/row['core_member']
            raw = path.read_bytes(); path.write_bytes(bytes([raw[0]^1])+raw[1:])
            stage = deps.receive(archive, self.tar(value))['stage']; deps.accept(archive, stage)
            self.assertTrue(deps.inspect(archive)['covered_core_sha_ack_verified'])
            with self.assertRaisesRegex(ValueError, 'FRESH_DESTINATION_SHA_DIFFERS'): deps.inspect(archive, deep=True)

    def test_no_writer_can_be_automatically_replayed(self):
        with tempfile.TemporaryDirectory() as td:
            base, archive, _, saved, _ = self.prepared(Path(td))
            with self.assertRaisesRegex(ValueError, 'SOURCE_ALREADY_ATTEMPTED'): deps.snapshot(base, 50, archive)
            stream = io.BytesIO(); deps.emit(saved['snapshot'], stream)
            with self.assertRaisesRegex(ValueError, 'EMIT_ALREADY_ATTEMPTED'): deps.emit(saved['snapshot'], io.BytesIO())
            stream.seek(0); stage = deps.receive(archive, stream)['stage']
            with self.assertRaisesRegex(ValueError, 'RECEIVE_ALREADY_ATTEMPTED'): deps.receive(archive, io.BytesIO())
            deps.accept(archive, stage)
            with self.assertRaisesRegex(ValueError, 'ACCEPT_ALREADY_ATTEMPTED'): deps.accept(archive, stage)

    def test_stage_tamper_blocks_accept_and_keeps_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, _, value = self.prepared(Path(td))
            stage = Path(deps.receive(archive, self.tar(value))['stage'])
            (stage/'grounding/dumps/nq-hotpot-common_0.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError): deps.accept(archive, stage)
            self.assertTrue(stage.exists()); self.assertFalse((archive/deps.CHILD).exists())
            self.assertTrue((archive/'post-callback-boundary/boundary-archive-receipt.json').is_file())

    def test_existing_accepted_boundary_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, _, value = self.prepared(Path(td))
            (archive/'post-callback-boundary/boundary-archive-receipt.json').unlink()
            with self.assertRaises(FileNotFoundError): deps.receive(archive, self.tar(value))
            self.assertFalse((archive/deps.CHILD).exists())

    def test_json_and_head_limits_and_manifest_dump_scope_are_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            _, archive, _, _, value = self.prepared(Path(td))
            changed = copy.deepcopy(value); changed['dependencies']['grounding/heads/commit_probe.pkl']['bytes'] = deps.MAX_HEAD+1
            with self.assertRaisesRegex(ValueError, 'DEPENDENCY_PATH_SIZE_SHA_DIFFERS'): deps.receive(archive, self.tar(changed))


if __name__ == '__main__': unittest.main()
