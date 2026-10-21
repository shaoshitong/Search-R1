"""CPU-only private runtime copying and restore process environment checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8.model_identity import SEVEN
from reproduction.easy2560_joint4 import final_restore_only as restore
from reproduction.easy2560_joint4 import flash3_runtime


class FinalRestoreFlash3Tests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'; base = root/'target'
        (source/'data').mkdir(parents=True)
        for name in ('calibration-300.parquet', 'audit.json'):
            (source/'data'/name).write_bytes(name.encode())
        checkpoint = source/'checkpoints/global_step_200'; checkpoint.mkdir(parents=True)
        (checkpoint/'model.safetensors').write_bytes(b'untouched source checkpoint fixture')
        heads = source/'round150/heads'; heads.mkdir(parents=True)
        for name in ('commit_probe.pkl', 'boundary_probe.pkl', 'calibration-report.json'):
            (heads/name).write_bytes(name.encode())
        live = dict(run_id='r', round=150, version='r:150:saved', encoder='/remote/cp150',
            commit=str(heads/'commit_probe.pkl'), boundary=str(heads/'boundary_probe.pkl'),
            head_sha256={name:restore.sha(heads/name) for name in ('commit_probe.pkl', 'boundary_probe.pkl')},
            calibration_report_sha256=restore.sha(heads/'calibration-report.json'))
        manifest = source/'live.json'; manifest.write_text(json.dumps(live))
        receipt_path = source/'receipt.json'; receipt_path.write_text('{}')
        receipt = dict(source_job='producer', source_commit='a'*40,
            reward_probe_manifest={'path':str(manifest)}, reward_probe_round=150,
            reward_probe_version=live['version'], attempt_progress={'cumulative_writes':4000},
            driver_state={'source_consumed':512000})
        runtime = source/'private-fa3'; runtime.mkdir()
        names = dict(interface='flash_attn_interface.py', extension='flash_attn_3_cuda.fixture.so',
                     cudart='libcudart.so.12')
        sources = {}; binding = {}
        for role, name in names.items():
            path = runtime/name; path.write_bytes(('private CPU fixture '+role).encode())
            sources[role] = (str(path), restore.sha(path), path.stat().st_size)
            binding.update({role+'_path':str(path), role+'_sha256':restore.sha(path)})
        parent = dict(run_id='r', budget={'model_id':SEVEN[0], 'model_revision':SEVEN[1]},
            model_revision=SEVEN[1], initial_model='/remote/official', question_probe=None,
            full_state_restore={'completed_outer':12}, easy2560_selection=True, profile='common8',
            execution_site='hpc3', training_world_size=4, data_selection_mode='mixed_first',
            actor_logical_micro_batch_per_rank=4,
            infra_optimizations=dict(selected_policy_logits=False, exact_retrieval_cache=True,
                                     flash_attention_3_actor=True), infra_flash_attention_3=binding)
        return base, source, checkpoint, receipt_path, receipt, parent, sources

    def prepare(self, fixture):
        base, source, checkpoint, receipt_path, receipt, parent, sources = fixture
        def bootstrap(*args):
            for name in ('repo', 'logs', 'preparation', 'data'):
                (base/name).mkdir(parents=True)
            (base/'repo/entry.py').write_text('pass\n')
        with patch('reproduction.easy2560_joint4.control.own_base', side_effect=Path), \
             patch('reproduction.easy2560_joint4.control.bootstrap', side_effect=bootstrap), \
             patch.object(restore, 'source', return_value=(receipt_path,receipt,checkpoint,source,parent)), \
             patch('reproduction.common8.source_files.runtime_files', return_value=[base/'repo/entry.py']), \
             patch.object(flash3_runtime, 'SOURCES', sources):
            return restore.prepare(base, receipt_path, 'b'*40)

    def test_prepare_fresh_private_files_and_rebinds_only_target_paths(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(Path(td).resolve())
            base, source, checkpoint, _, _, parent, sources = fixture
            before = copy.deepcopy(parent)
            result = self.prepare(fixture)
            run = restore.read(base/'logs/run-manifest.json')
            self.assertEqual(parent, before)
            self.assertEqual(run['infra_optimizations'], parent['infra_optimizations'])
            self.assertEqual(run['full_state_restore']['completed_outer'], 200)
            self.assertEqual(run['full_state_restore']['source_checkpoint'], str(checkpoint))
            self.assertEqual(result['tensor_copies'], 0)
            self.assertFalse(result['actual_cuda_restore_accepted'])
            self.assertFalse(list(base.rglob('*.safetensors')))
            for role, (path, sha, size) in sources.items():
                copied = Path(run['infra_flash_attention_3'][role+'_path'])
                self.assertEqual(copied.parent, base/'preparation/flash3-runtime')
                self.assertNotEqual(copied, Path(path))
                self.assertFalse(copied.is_symlink())
                self.assertEqual((restore.sha(copied), copied.stat().st_size), (sha, size))
                self.assertEqual(run['infra_flash_attention_3'][role+'_sha256'], sha)
                self.assertEqual(restore.sha(path), sha)
            ready = restore.read(base/'preparation/flash3-runtime/ready.json')
            self.assertEqual(ready['status'], 'PINNED_PRIVATE_FA3_RUNTIME_SHA_PASS')

    def test_source_bound_runtime_sha_mismatch_cannot_publish_target_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(Path(td).resolve()); fixture[5]['infra_flash_attention_3']['extension_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SAVED_FLASH3_RUNTIME_SHA_DIFFERS'):
                self.prepare(fixture)
            self.assertFalse((fixture[0]/'logs/run-manifest.json').exists())

    def test_corrupt_private_runtime_is_rejected_without_rebinding(self):
        with tempfile.TemporaryDirectory() as td:
            base, _, _, _, _, run, sources = self.fixture(Path(td).resolve())
            (base/'preparation').mkdir(parents=True)
            before = copy.deepcopy(run)
            with patch('reproduction.easy2560_joint4.control.own_base', side_effect=Path), \
                 patch.object(flash3_runtime, 'SOURCES', sources):
                private = flash3_runtime.prepare(base)
                cudart = Path(private['cudart_path']); body = cudart.read_bytes()
                cudart.write_bytes(bytes([body[0] ^ 1])+body[1:])
                with self.assertRaisesRegex(ValueError, 'private FA3 payload changed'):
                    restore.prepare_flash3_runtime(base, run)
            self.assertEqual(run, before)

    def test_fa2_does_not_copy_runtime_or_require_preload(self):
        run = dict(infra_optimizations={'selected_policy_logits':False, 'exact_retrieval_cache':False})
        before = copy.deepcopy(run)
        with patch.object(flash3_runtime, 'prepare', side_effect=AssertionError('FA2 must not copy FA3')), \
             patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(restore.prepare_flash3_runtime(Path('/unused/new-target'), run))
            self.assertNotIn('LD_PRELOAD', restore.ray_env_vars(run))
        self.assertEqual(run, before)

    def test_ray_driver_and_actor_environment_requires_same_bound_preload(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, _, _, _, run, _ = self.fixture(Path(td).resolve())
            cudart = run['infra_flash_attention_3']['cudart_path']
            with patch.dict(os.environ, {'LD_PRELOAD':cudart}, clear=True):
                env_vars = restore.ray_env_vars(run)
                self.assertEqual(env_vars['LD_PRELOAD'], cudart)
                self.assertEqual(env_vars['NCCL_DEBUG'], 'WARN')
            for value in ('', '/foreign/libcudart.so.12'):
                with self.subTest(value=value), patch.dict(os.environ, {'LD_PRELOAD':value}, clear=True), \
                     self.assertRaisesRegex(ValueError, 'PRIVATE_FLASH3_CUDART_NOT_PRELOADED'):
                    restore.ray_env_vars(run)

    def test_restore_shell_preloads_private_library_before_python_exec_only_for_fa3(self):
        entry = Path(restore.__file__).with_suffix('.sh')
        for use_fa3 in (False, True):
            with self.subTest(fa3=use_fa3), tempfile.TemporaryDirectory() as td:
                root = Path(td); base = root/'target'; (base/'env/bin').mkdir(parents=True)
                (base/'repo').mkdir()
                # This shell shim records exec's environment; it loads no Torch/CUDA.
                python = base/'env/bin/python'
                python.write_text('#!/bin/sh\nprintf "%s" "${LD_PRELOAD-}" > "$CAPTURE"\n'
                                  'printf "%s\\n" "$@" > "$ARGUMENTS"\n')
                python.chmod(0o755)
                if use_fa3:
                    private = base/'preparation/flash3-runtime'; private.mkdir(parents=True)
                    (private/'ready.json').write_text('{}')
                env = dict(os.environ, CAPTURE=str(root/'environment'), ARGUMENTS=str(root/'arguments'))
                env.pop('LD_PRELOAD', None)
                subprocess.run(['bash', str(entry), str(base)], env=env, check=True, capture_output=True)
                self.assertEqual((root/'environment').read_text(),
                    str(base/'preparation/flash3-runtime/libcudart.so.12') if use_fa3 else '')
                self.assertEqual((root/'arguments').read_text().splitlines(),
                    ['-B', '-m', 'reproduction.easy2560_joint4.final_restore_only', 'execute', '--base', str(base)])


if __name__ == '__main__':
    unittest.main()
