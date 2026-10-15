"""CPU-only runner fixtures; no official weights, encoder, GPU or new evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from reproduction.a800_common import qprobe_qwen_block_warmup as runner


def artifact(base):
    shapes = runner.block_shapes(4, 2, 1, 8)
    tensors = {name: (torch.arange(__import__('math').prod(shape)).reshape(shape) / 100).to(torch.bfloat16)
               for name, shape in shapes.items()}
    tensors['unused'] = torch.tensor([1.], dtype=torch.bfloat16)
    header, buffers, offset = {}, [], 0
    index = {}
    for key, value in tensors.items():
        name = runner.PREFIX + key if key != 'unused' else 'model.layers.0.input_layernorm.weight'
        raw = value.view(torch.uint8).numpy().tobytes()
        header[name] = dict(dtype='BF16', shape=list(value.shape), data_offsets=[offset, offset + len(raw)])
        buffers.append(raw); offset += len(raw)
        index[name] = 'part.safetensors'
    raw = json.dumps(header).encode()
    path = base / 'part.safetensors'
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(buffers))
    (base / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=index)))
    hashes = {name: runner.sha(base/name) for name in ('part.safetensors', 'model.safetensors.index.json')}
    return shapes, tensors, hashes


def args():
    return SimpleNamespace(source_commit='c' * 40, source_run_base=Path('/data/source'),
        source_publication=Path('/data/publication.json'), teacher_identity_sha256='d' * 64,
        resume_token_feature_output=Path(runner.qualified.RESUME_OUTPUT), micro_batch_size=32, head_seed=1729)


class QwenBlockSourceTests(unittest.TestCase):
    def test_header_binding_and_safeopen_load_only_the_twelve_official_lastblock_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            shapes, tensors, hashes = artifact(base)
            proof = runner.block_index_binding(base, hashes, expected_shapes=shapes)
            self.assertEqual(proof['tensor_count'], 12)
            self.assertEqual(set(proof['tensors']), set(shapes))
            calls = []
            class Handle:
                def __enter__(self): return self
                def __exit__(self, *_args): pass
                def get_tensor(self, name):
                    calls.append(name)
                    return tensors[name[len(runner.PREFIX):]]
            def opener(path, **kwargs):
                self.assertEqual(kwargs, dict(framework='pt', device='cpu'))
                self.assertEqual(Path(path), base/'part.safetensors')
                return Handle()
            loaded = runner.load_block_weights(base, proof, opener=opener)
            self.assertEqual(set(loaded), set(shapes))
            self.assertEqual(set(calls), {runner.PREFIX + key for key in shapes})
            self.assertFalse(any('layers.0.' in name for name in calls))
            for name, value in loaded.items():
                self.assertEqual(runner.tensor_sha(value), proof['tensors'][name]['source_sha256'])
            raw = bytearray((base/'part.safetensors').read_bytes()); raw[-1] ^= 1
            (base/'part.safetensors').write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'SHARD_CHANGED_AFTER_PREPARATION'):
                runner.load_block_weights(base, proof, opener=opener)

    def test_complete_tensor_keys_shapes_dtypes_and_byte_counts_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            shapes, _, hashes = artifact(base)
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_OR_EXTRA'):
                runner.block_index_binding(base, hashes, expected_shapes=dict(list(shapes.items())[:-1]))
            changed = dict(shapes); changed['input_layernorm.weight'] = [5]
            with self.assertRaisesRegex(ValueError, 'SHAPE_OR_DTYPE'):
                runner.block_index_binding(base, hashes, expected_shapes=changed)
            (base/'model.safetensors.index.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'INDEX_SHA_CHANGED'):
                runner.block_index_binding(base, hashes, expected_shapes=shapes)

    def test_actual_copy_proof_requires_all_official_hashes_and_exact_float32_cast(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            shapes, _, hashes = artifact(base)
            expected = runner.block_index_binding(base, hashes, expected_shapes=shapes)
            actual = dict(source_layer=35, tensor_count=12, all_copied_parameters_trainable=True,
                tensors={key: dict(item, copied_fp32_sha256='a' * 64, exact_cast_equal=True)
                         for key, item in expected['tensors'].items()})
            runner.verify_copied_block(actual, expected)
            changed = copy.deepcopy(actual); changed['tensors']['mlp.up_proj.weight']['source_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'MATCH_OFFICIAL_SOURCE'):
                runner.verify_copied_block(changed, expected)
            changed = copy.deepcopy(actual); changed['tensors']['mlp.up_proj.weight']['exact_cast_equal'] = False
            with self.assertRaisesRegex(ValueError, 'NOT_EXACT'):
                runner.verify_copied_block(changed, expected)

    def test_task_contract_fixes_native_block10_and_raw_unnormalized_recent8_formal_replay(self):
        config = runner.task_config(args())
        self.assertEqual(config['epochs'], 10)
        self.assertEqual(config['block_index'], 35)
        self.assertEqual(config['micro_batch_size'], 32)
        self.assertEqual(config['classification_report'], 'binary_accuracy')
        self.assertFalse(config['replay_contract']['teacher_warmup_in_replay'])
        self.assertFalse(config['replay_contract']['normalize_weights'])
        self.assertEqual(config['replay_contract']['ages_included'], list(range(8)))
        self.assertEqual(config['qualified_token_warmup_ready_sha256'], runner.QUALIFIED_READY_SHA)

    def test_cpu_checks_require_actual_tests_with_no_skips_or_cuda(self):
        summary = dict(module='search_g1.test_qprobe_qwen_block_model', tests_run=5, failures=0,
            errors=0, skipped=0, passed=True, cuda_visible_devices='', cuda_initialized=False)
        def result(value):
            return SimpleNamespace(returncode=0, stdout='QWEN_BLOCK_CPU_CHECKS=' + json.dumps(value) + '\n', stderr='')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(runner.subprocess, 'run', return_value=result(summary)) as launch:
                checked = runner.cpu_checks(output)
            self.assertTrue(checked['passed'])
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(checked['log_sha256'], runner.sha(output/'cpu-checks.log'))
            for field, value in (('skipped', 1), ('tests_run', 0), ('cuda_initialized', True), ('passed', False)):
                bad = dict(summary, **{field: value})
                with self.subTest(field=field), patch.object(runner.subprocess, 'run', return_value=result(bad)):
                    with self.assertRaisesRegex(ValueError, 'DID_NOT_PASS'):
                        runner.cpu_checks(output)

    def test_progress_is_append_only_and_latest_snapshot_tracks_actual_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch('builtins.print'):
                runner.progress(output, dict(stage='one', head_adam_steps=0))
                runner.progress(output, dict(stage='two', head_adam_steps=1))
            rows = [json.loads(line) for line in (output/'progress.jsonl').read_text().splitlines()]
            self.assertEqual([row['head_adam_steps'] for row in rows], [0, 1])
            self.assertEqual(json.loads((output/'progress.json').read_text())['stage'], 'two')

    def test_partial_state_preserves_actual_writes_and_does_not_reset_adam(self):
        class Partial:
            update_count = 17
            fitted = False
            def _adam_steps(self): return 17
            def state_dict(self): return dict(update_count=17, fitted=False, optimizer={'step': 17}, private_rng=torch.arange(3))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            runner.preserve_partial(output, Partial(), {'official_block_source': 'fixed'})
            state = torch.load(output/'partial-head.pt', map_location='cpu', weights_only=False)
            receipt = json.loads((output/'partial-head.json').read_text())
            self.assertEqual(state['head']['optimizer']['step'], 17)
            self.assertEqual(receipt['head_adam_steps'], 17)
            self.assertFalse(receipt['rollback_performed'])
            self.assertFalse(receipt['automatic_retry'])

    def test_histories_require_every_real_minibatch_and_epoch_with_accuracy(self):
        metrics = dict(fit_epochs=10, training_groups=5120, actual_head_adam_updates=200, expected_head_adam_updates=200,
            epoch_history=[dict(epoch=i, actual_head_adam_updates=i*20) for i in range(11)],
            minibatch_history=[dict(head_update_ordinal=i, head_adam_steps_before=i-1, head_adam_steps_after=i,
                epoch=(i-1)//20+1, minibatch_index=(i-1)%20, batch_size=256, binary_accuracy=.5) for i in range(1, 201)])
        runner.verify_histories(metrics)
        changed = copy.deepcopy(metrics); changed['minibatch_history'][2]['head_adam_steps_after'] = 2
        with self.assertRaisesRegex(ValueError, 'MINIBATCH_ADAM_HISTORY'):
            runner.verify_histories(changed)
        changed = copy.deepcopy(metrics); del changed['minibatch_history'][2]['binary_accuracy']
        with self.assertRaisesRegex(ValueError, 'MINIBATCH_ADAM_HISTORY'):
            runner.verify_histories(changed)


if __name__ == '__main__': unittest.main()
