"""CPU schema regression checks; no real milestone or GPU PASS is fabricated."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from .actor import _state_binding, learning_rate, optimizer_layout
from .checkpoint_cpu_once import (digest, hash_file, qwen_shapes, validate_actor_state,
                                  validate_pool_state, validate_probe_state)


def actor_state_fixture():
    """Tiny actual CPU Adam history used only to exercise state schema checks."""
    config = dict(run_id='new-run', official_model_dir='/mnt/official', grounding_probe_version='grounding0', seed=1729)
    model = torch.nn.Linear(2, 1).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed: min(completed / 57, 1.))
    for _outer in range(5):
        for _step in range(20):
            optimizer.zero_grad(); model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        scheduler.step()
    binding = _state_binding(config)
    saved = dict(run_id=config['run_id'], completed_outer=5, rank=0, binding=binding,
        actual_adam_steps=100, scheduler_completed_outer=5, optimizer=optimizer.state_dict(),
        optimizer_parameter_layout=optimizer_layout(optimizer), scheduler=scheduler.state_dict(),
        model_master_dtype='torch.float32', optimizer_layout='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world8',
        rng=dict(python=random.Random(9).getstate(), numpy=np.random.RandomState(9).get_state(),
                 torch_cpu=torch.Generator(device='cpu').manual_seed(9).get_state(),
                 torch_cuda=torch.tensor([1, 2], dtype=torch.uint8)))
    # CUDA bytes above are a schema-only fixture; no actual GPU state application
    # or a full HF/actor savepoint audit is claimed by this test.
    return config, dict(binding=binding), saved


class MilestoneCPUContracts(unittest.TestCase):
    def test_real_cpu_adam_scheduler_state_and_bad_rank_layout(self):
        config, actor, saved = actor_state_fixture()
        summary = validate_actor_state(saved, actor, config, 5, 0)
        self.assertEqual(summary['actual_adam_steps'], 100)
        self.assertEqual(summary['next_learning_rate'], learning_rate(5))
        self.assertFalse(summary['rng']['cuda_rng_applied'])
        changed = copy.deepcopy(saved); changed['rank'] = 1
        with self.assertRaisesRegex(ValueError, 'another run/rank'):
            validate_actor_state(changed, actor, config, 5, 0)
        changed = copy.deepcopy(saved); changed['optimizer_parameter_layout'][0][0]['shape'] = [999]
        with self.assertRaisesRegex(ValueError, 'shape/layout'):
            validate_actor_state(changed, actor, config, 5, 0)
        changed = copy.deepcopy(saved); changed['scheduler']['last_epoch'] = 4
        with self.assertRaisesRegex(ValueError, 'LambdaLR'):
            validate_actor_state(changed, actor, config, 5, 0)

    def test_mismatched_actual_adam_step_and_rng_rejected(self):
        config, actor, saved = actor_state_fixture()
        changed = copy.deepcopy(saved)
        next(iter(changed['optimizer']['state'].values()))['step'] = torch.tensor(99.)
        with self.assertRaisesRegex(ValueError, 'Adam counter'):
            validate_actor_state(changed, actor, config, 5, 0)
        changed = copy.deepcopy(saved); changed['rng']['torch_cuda'] = torch.tensor([], dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, 'CUDA RNG'):
            validate_actor_state(changed, actor, config, 5, 0)

    def test_real_cycle_boundary_cursor_equation_and_permutation(self):
        size, outer = 168510, 33
        state = dict(seed=1729, cycle=1, cursor=450, next_outer=34,
                     consumed_candidates=5120*outer, candidate_questions=5120, order=list(range(size)))
        binding = dict(eligible_questions=size, sha256='source')
        source = dict(eligible_unique_questions=size, records=[None]*size, canonical_rows=169028,
                      excluded_diagnostic_questions=512, canonical_train_sha256='train')
        reservation = dict(outer=outer, pool_sha256='source', after=state, after_sha256=digest(state),
                           indices=list(range(5120)), cycles=[0]*4670+[1]*450)
        config = dict(seed=1729, canonical_train_sha256='train')
        summary = validate_pool_state(state, reservation, binding, source, config, outer)
        self.assertEqual((summary['cycle'], summary['cursor']), (1, 450))
        changed = dict(state, cursor=0)
        with self.assertRaisesRegex(ValueError, 'cycle/cursor'):
            validate_pool_state(changed, reservation, binding, source, config, outer)
        bad = dict(state, order=[0]*size)
        with self.assertRaisesRegex(ValueError, 'permutation'):
            validate_pool_state(bad, reservation, binding, source, config, outer)

    def test_independent_whole_file_hash_detects_changed_producer_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small-proof.json'
            payload = json.dumps(dict(actual=20)).encode(); path.write_bytes(payload)
            producer = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            proof = hash_file(path, producer)
            self.assertTrue(proof['independent_full_sha'])
            path.write_bytes(json.dumps(dict(actual=19)).encode())
            with self.assertRaisesRegex(ValueError, 'producer'):
                hash_file(path, producer)

    def test_official_architecture_and_unsupported_probe_cannot_be_passed_by_counts(self):
        cfg = dict(model_type='qwen2', hidden_size=2048, num_hidden_layers=36, intermediate_size=11008,
                   vocab_size=151936, num_attention_heads=16, num_key_value_heads=2, tie_word_embeddings=False)
        shapes = qwen_shapes(cfg)
        self.assertEqual(len(shapes), 435)
        self.assertEqual(shapes['model.layers.35.self_attn.k_proj.weight'], (256, 2048))
        with self.assertRaisesRegex(ValueError, 'architecture'):
            qwen_shapes(dict(cfg, num_hidden_layers=24))
        fake = dict(run_id='new', completed_outer=5, actual_adam_steps_per_rank=[50]*8,
                    learning_rate=1e-6, actor_version='old-actor', binding={}, binding_sha256=digest({}))
        with self.assertRaisesRegex(ValueError, 'scientific binding'):
            validate_probe_state(fake, dict(binding_sha256=digest({})), dict(run_id='new'), 5, 'old-actor')


if __name__ == '__main__': unittest.main()
