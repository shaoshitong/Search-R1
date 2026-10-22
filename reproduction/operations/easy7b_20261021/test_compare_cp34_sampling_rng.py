"""Real small CPU torch.save fixtures; never access a GPU, bridge or remote."""
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

from reproduction.operations.easy7b_20261021 import compare_cp34_sampling_rng as compare


def rng_fixture(rank):
    return dict(torch_cpu=torch.Generator().manual_seed(12+rank).get_state(),
        torch_cuda=torch.tensor([1, 2, rank, 4], dtype=torch.uint8),
        # Deliberately different actor/global/rollout namespaces within each file.
        actor_cuda_saved=torch.tensor([7, 8, rank, 9], dtype=torch.uint8),
        rollout_cuda_saved=torch.tensor([20, 21, rank, 22], dtype=torch.uint8),
        numpy=np.random.RandomState(24+rank).get_state(), python=random.Random(30+rank).getstate(),
        cuda_device=rank, request_generators={}, vllm_version='0.6.3',
        boundary='idle synchronous generation engine')


class CompareSamplingRngTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.old = Path(self.temp.name).resolve()/'old'; self.new = Path(self.temp.name).resolve()/'new'
        self.patch = patch.multiple(compare, OLD_BASE=self.old, NEW_BASE=self.new)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.paths = {}
        for rank in range(4):
            for role, base, suffix, commit in (
                (compare.ROLES[0], self.old, 'preparation/rolling-checkpoints/global_step_34/boundary', compare.OLD_COMMIT),
                (compare.ROLES[1], self.old, 'artifacts/mixed-candidate-pools/'+compare.OLD_ATTEMPT+'-pool-0/0-worker-rng', compare.OLD_COMMIT),
                (compare.ROLES[2], self.new, 'artifacts/mixed-candidate-pools/'+compare.NEW_ATTEMPT+'-pool-0/0-worker-rng', compare.NEW_COMMIT)):
                path = base/suffix/('rank-'+str(rank)+'.pt'); path.parent.mkdir(parents=True, exist_ok=True)
                value = dict(run_id=compare.RUN, source_commit=commit, rank=rank, world_size=4,
                    completed_outer=34, actual_optimizer_writes=680, rng=rng_fixture(rank))
                torch.save(value, path); self.paths[(rank, role)] = path

    def alter(self, rank, role, change):
        path = self.paths[(rank, role)]
        value = torch.load(path, map_location='cpu', weights_only=False)
        change(value); torch.save(value, path)

    def test_real_three_way_content_comparison_preserves_namespace_and_never_opens_large_state(self):
        for base in (self.old, self.new):
            for relative in ('checkpoints/search-g1-common8/actor/global_step_34/training_state/rank-0.pt',
                             'artifacts/mixed-candidate-pools/raw/0.pt'):
                path = base/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'not a sidecar')
        opened = []; original_open = compare.os.open; original_load = torch.load
        def guarded(path, *args, **kwargs):
            self.assertIn(Path(path), self.paths.values()); opened.append(Path(path))
            return original_open(path, *args, **kwargs)
        def cpu_load(source, *args, **kwargs):
            self.assertEqual(kwargs['map_location'], 'cpu')
            self.assertIsInstance(source, io.BytesIO)
            return original_load(source, *args, **kwargs)
        with patch.object(compare.os, 'open', guarded), patch.object(torch, 'load', cpu_load), \
             patch.object(torch.cuda, 'get_rng_state', side_effect=AssertionError('CUDA access')), \
             patch.object(torch.cuda, 'is_initialized', side_effect=AssertionError('CUDA access')):
            result = compare.compare()
        self.assertEqual(len(opened), 12)
        self.assertEqual(result['status'], 'RNG_BOUNDARY_COMPARISON_COMPLETE')
        for row in result['ranks']:
            self.assertTrue(all(value['all_three'] == 'EQUAL' for value in row['same_field_comparisons'].values()))
            self.assertEqual(row['next_rollout_private_rng']['all_three'], 'EQUAL')
            fields = row['files'][compare.ROLES[0]]['fields']
            self.assertNotEqual(fields['torch_cuda']['sha256'], fields['rollout_cuda_saved']['sha256'])
            self.assertEqual(fields['torch_cuda']['structure']['device'], 'cpu')
        self.assertFalse(result['model_or_adam_files_read']); self.assertFalse(result['training_modified'])
        json.dumps(result, allow_nan=False)

    def test_private_state_difference_is_detected_even_with_global_cuda_equal(self):
        self.alter(2, compare.ROLES[2], lambda value: value['rng']['rollout_cuda_saved'].__setitem__(0, 99))
        row = compare.compare()['ranks'][2]
        self.assertEqual(row['same_field_comparisons']['torch_cuda']['all_three'], 'EQUAL')
        self.assertEqual(row['next_rollout_private_rng']['all_three'], 'DIFFERENT')
        self.assertEqual(row['next_rollout_private_rng']['pairs'][compare.ROLES[0]+'__vs__'+compare.ROLES[1]], 'EQUAL')

    def test_missing_sidecar_or_field_is_unknown_without_fabricating_equality(self):
        self.paths[(0, compare.ROLES[1])].unlink()
        self.alter(1, compare.ROLES[2], lambda value: value['rng'].pop('rollout_cuda_saved'))
        result = compare.compare()
        self.assertEqual(result['status'], 'RNG_BOUNDARY_COMPARISON_UNKNOWN')
        self.assertEqual(result['ranks'][0]['files'][compare.ROLES[1]]['status'], 'UNKNOWN')
        self.assertEqual(result['ranks'][0]['next_rollout_private_rng']['all_three'], 'UNKNOWN')
        self.assertEqual(result['ranks'][1]['same_field_comparisons']['rollout_cuda_saved']['all_three'], 'UNKNOWN')

    def test_request_generators_numpy_python_and_tensor_dtype_are_content_hashed(self):
        self.alter(0, compare.ROLES[2], lambda value: value['rng']['request_generators'].update(
            request=dict(device='cuda:0', state=torch.tensor([1, 2, 3], dtype=torch.uint8))))
        self.alter(1, compare.ROLES[2], lambda value: value['rng'].__setitem__('numpy', np.random.RandomState(99).get_state()))
        self.alter(2, compare.ROLES[2], lambda value: value['rng'].__setitem__('python', random.Random(99).getstate()))
        result = compare.compare()
        for rank, field in ((0, 'request_generators'), (1, 'numpy'), (2, 'python')):
            self.assertEqual(result['ranks'][rank]['same_field_comparisons'][field]['all_three'], 'DIFFERENT')
        self.assertNotEqual(compare.fingerprint(torch.tensor([1], dtype=torch.uint8))['sha256'],
                            compare.fingerprint(torch.tensor([1], dtype=torch.int64))['sha256'])

    def test_wrong_identity_oversized_or_optimizer_file_is_not_admitted(self):
        self.alter(0, compare.ROLES[1], lambda value: value.__setitem__('completed_outer', 35))
        self.alter(1, compare.ROLES[2], lambda value: value.__setitem__('optimizer', {}))
        self.paths[(2, compare.ROLES[1])].write_bytes(b'x'*(compare.MAX_SMALL+1))
        result = compare.compare()
        for rank, role in ((0, compare.ROLES[1]), (1, compare.ROLES[2]), (2, compare.ROLES[1])):
            self.assertEqual(result['ranks'][rank]['files'][role]['status'], 'UNKNOWN')
            self.assertEqual(result['ranks'][rank]['files'][role]['reason'], 'SIDECAR_OBSERVATION_ERROR')

    def test_argument_traversal_rejected_before_any_read_and_no_state_mutation(self):
        before = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
        with patch.object(compare, '_read', side_effect=AssertionError('unexpected file open')):
            with redirect_stdout(io.StringIO()) as stream:
                rc = compare.main(['--old-attempt', '../0'])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(stream.getvalue())['status'], 'RNG_COMPARISON_ARGUMENT_ERROR')
        compare.compare()
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(after[0], before[1][0]); self.assertTrue(np.array_equal(after[1], before[1][1]))
        self.assertEqual(after[2:], before[1][2:]); self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))


if __name__ == '__main__': unittest.main()
