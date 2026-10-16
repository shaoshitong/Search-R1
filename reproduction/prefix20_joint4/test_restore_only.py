"""CPU copy/data/guard tests only; these cannot establish actual CUDA recovery."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.prefix20_joint4.restore_only_assets import copy_verified, record, regular_under, sha, verify_target_copy, ROLE
from search_g1.prefix20_restore_only import deny_training, guard_probe_operation, check_pending_membership, install_mutation_guards
from reproduction.prefix20_joint4.restore_only import validate_actor_readbacks


class TargetCopyTests(unittest.TestCase):
    def test_copy_has_distinct_object_and_fresh_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, target = root / 'source', root / 'target/member'
            source.write_bytes(b'checkpoint CPU fixture')
            result = copy_verified(source, target, sha(source), source.stat().st_size)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertNotEqual(target.stat().st_ino, source.stat().st_ino)
            self.assertEqual(result['full_read_sha256'], sha(source))
            with self.assertRaisesRegex(ValueError, 'MEMBER_EXISTS'):
                copy_verified(source, target, sha(source))

    def test_changed_source_and_target_ancestor_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'source'
            source.write_text('before')
            digest = sha(source)
            source.write_text('after')
            with self.assertRaisesRegex(ValueError, 'SOURCE_SHA'):
                copy_verified(source, root / 'fresh', digest)
            real = root / 'real'
            real.mkdir()
            (root / 'link').symlink_to(real)
            with self.assertRaisesRegex(ValueError, 'PARENT_IS_SYMLINK'):
                copy_verified(source, root / 'link/member', sha(source))

    def fixture(self, root):
        target, original = root / 'target', root / 'original'
        target.mkdir()
        original.mkdir()
        (target / 'logs').mkdir()
        (target / 'preparation').mkdir()
        paths = {}
        for name in ('actor', 'boundary', 'driver', 'source-run', 'reward-manifest', 'reward-head',
                     'model/config', 'head', 'probe-config', 'pool'):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            paths[name] = path
        paths['reward-manifest'].write_text(json.dumps(dict(commit=str(paths['reward-head']),
            boundary=str(paths['reward-head']), encoder=str(target / 'model'))))
        pair = dict(source_base=str(target), checkpoint=str(target / 'checkpoint'),
            files=[record(paths['actor'])], boundary_worker_files=[record(paths['boundary'])],
            boundary_driver=record(paths['driver']), source_run_manifest=record(paths['source-run']),
            reward_probe_manifest=record(paths['reward-manifest']), reward_files=[record(paths['reward-head'])])
        pair_path = target / 'pair.json'
        pair_path.write_text(json.dumps(pair))
        run = dict(initial_model=str(target / 'model'), reference_model=str(target / 'model'),
            full_state_restore=dict(source_receipt=str(pair_path), source_checkpoint=str(target / 'checkpoint')),
            question_probe=dict(model_dir=str(target / 'model'), head_safetensors=str(paths['head']),
                                probe_config=str(paths['probe-config']), eligible_pool_path=str(paths['pool'])))
        run_path = target / 'logs/run-manifest.json'
        run_path.write_text(json.dumps(run))
        proof = dict(role=ROLE, target_base=str(target), step=200, world_size=4,
            target_pair_receipt=str(pair_path), files=[record(path) for path in paths.values()] + [record(pair_path), record(run_path)])
        proof_path = target / 'preparation/copy.json'
        proof_path.write_text(json.dumps(proof))
        return target, original, proof_path, pair_path, run_path, proof

    def test_target_seal_rejects_tamper_or_original_load_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            target, original, proof_path, pair_path, run_path, proof = self.fixture(Path(directory).resolve())
            self.assertEqual(verify_target_copy(target, proof_path)['world_size'], 4)
            run = json.loads(run_path.read_text())
            run['full_state_restore']['source_checkpoint'] = str(original)
            run_path.write_text(json.dumps(run))
            proof['files'][-1] = record(run_path)
            proof_path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, 'LOAD_PATH_ESCAPES'):
                verify_target_copy(target, proof_path)
            (target / 'actor').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'MEMBER_CHANGED'):
                verify_target_copy(target, proof_path)

    def test_regular_target_rejects_source_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'original'
            source.write_text('state')
            target = root / 'target'
            target.mkdir()
            (target / 'linked').symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'OUTSIDE_TARGET_OR_SYMLINK'):
                regular_under(target / 'linked', target)


class NoMutationTests(unittest.TestCase):
    def test_real_cpu_adam_load_remains_allowed_but_future_steps_are_blocked(self):
        import torch
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        original = torch.optim.Adam([parameter], lr=1e-6)
        parameter.grad = torch.tensor([.1])
        original.step()
        saved = copy.deepcopy(original.state_dict())
        target = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.]))], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(target, lambda _:1.)
        saved_scheduler = copy.deepcopy(scheduler.state_dict())
        worker = SimpleNamespace(_is_actor=True, _is_rollout=False, actor_optimizer=target, actor_lr_scheduler=scheduler)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'logs').mkdir()
            (root / 'logs/run-manifest.json').write_text(json.dumps(dict(restore_only_role=ROLE)))
            with patch.dict(os.environ, SEARCH_G1_BASE=str(root)):
                install_mutation_guards(worker)
                target.load_state_dict(saved)
                scheduler.load_state_dict(saved_scheduler)
                from search_g1.full_state_restore import equal_tree
                equal_tree(scheduler.state_dict(), saved_scheduler)
                self.assertEqual(worker._restore_only_forbidden_calls, {})
                with self.assertRaisesRegex(RuntimeError, 'actor_optimizer_step'):
                    target.step()
                with self.assertRaisesRegex(RuntimeError, 'actor_scheduler_step'):
                    scheduler.step()
        self.assertEqual(worker._restore_only_forbidden_calls, {'actor_optimizer_step':1, 'actor_scheduler_step':1})

    def test_final_counters_use_source_actor_writes_not_assumed_4000(self):
        rows = [dict(rank=i, actual_cuda_uuid='GPU-'+str(i), actual_optimizer_writes=4371,
            actual_counters=dict(completed_outer=200), scheduler_last_epoch=200, scheduler_step_count=201,
            process_identity=dict(pid=100+i, start_ticks=1000+i), forbidden_call_counts={},
            model_equal=True, optimizer_equal=True, scheduler_equal=True, boundary_rng_equal=True) for i in range(4)]
        allocation = dict(actual_cuda_runtime_uuids=['GPU-'+str(i) for i in range(4)])
        validate_actor_readbacks(rows, allocation, 4371)
        with self.assertRaisesRegex(ValueError, 'STATE_OR_COUNTER'):
            validate_actor_readbacks(rows, allocation, 4000)
        rows[2]['actual_counters']['completed_outer'] = 201
        with self.assertRaisesRegex(ValueError, 'STATE_OR_COUNTER'):
            validate_actor_readbacks(rows, allocation, 4371)

    def test_explicit_role_denies_training_and_probe_prediction_or_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'logs').mkdir()
            (root / 'logs/run-manifest.json').write_text(json.dumps(dict(restore_only_role=ROLE)))
            worker = SimpleNamespace(_restore_only_forbidden_calls={})
            with patch.dict(os.environ, SEARCH_G1_BASE=str(root)):
                with self.assertRaisesRegex(RuntimeError, 'TRAINING_ENTRY'):
                    deny_training()
                for operation in ('predict', 'update', 'unknown'):
                    with self.assertRaisesRegex(RuntimeError, 'PROBE_OPERATION'):
                        guard_probe_operation(worker, operation)
                for operation in ('init', 'load_state_dict', 'state_dict', 'readback'):
                    guard_probe_operation(worker, operation)
            self.assertEqual(worker._restore_only_forbidden_calls, {'probe_predict':1, 'probe_update':1, 'probe_unknown':1})

    def test_independent_entry_does_not_call_train_or_generate_or_save(self):
        entry = Path(__file__).with_name('restore_only.py')
        tree = ast.parse(entry.read_text())
        forbidden = {'fit', '_validate', 'score', 'predict', 'update', 'update_actor', 'generate_sequences',
                     'generate', 'save_checkpoint', '_save_checkpoint', 'begin', 'collect_pool', 'one'}
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called.intersection(forbidden))
        # Existing training binder must retain its final200 rejection.
        import search_g1.qprobe_checkpoint as paired
        self.assertIn("receipt['step']<200", Path(paired.__file__).read_text())


class PendingMembershipTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        import pandas as pd
        import torch
        from search_g1.qprobe_features import original_question_and_id
        rows = [dict(prompt=[dict(role='user', content='question '+str(i))], data_source='nq', extra_info=dict(index=i)) for i in range(3)]
        frame = pd.DataFrame(rows)
        order = [2, 0, 1]
        indices = order[1:]
        pairs = [original_question_and_id(frame.iloc[i]) for i in indices]
        class Pending:
            def __len__(self):return 2
        pending = Pending()
        pending.batch = dict(qprobe_source_index=torch.tensor(indices), pool_index=torch.tensor(indices),
                            canonical_source_index=torch.tensor([10+i for i in indices]), pool_cycle=torch.tensor([2, 2]))
        pending.non_tensor_batch = dict(candidate_original_question=np.array([row[0] for row in pairs], dtype=object),
                                        candidate_stable_id=np.array([row[1] for row in pairs], dtype=object))
        state = dict(candidate_order=order, candidate_cursor=3, epoch=2, consumed_source_rows=8,
                     pending_source=pending, pending_offset=1)
        stream = SimpleNamespace(state=lambda:state, loader=SimpleNamespace(dataset=SimpleNamespace(
            dataframe=frame, prompt_key='prompt', indices=[10, 11, 12])), binding=dict(ordered_membership_sha256='binding'))
        return stream, state

    def test_partial_pending_membership_is_checked_without_advancing(self):
        stream, state = self.fixture()
        before = (state['candidate_cursor'], state['pending_offset'], state['consumed_source_rows'])
        result = check_pending_membership(stream)
        self.assertEqual(result['pending_rows_verified'], 2)
        self.assertEqual(before, (state['candidate_cursor'], state['pending_offset'], state['consumed_source_rows']))

    def test_changed_pending_canonical_index_or_question_is_rejected(self):
        stream, state = self.fixture()
        state['pending_source'].batch['canonical_source_index'][0] = 999
        with self.assertRaisesRegex(ValueError, 'INDEX_MEMBERSHIP'):
            check_pending_membership(stream)
        stream, state = self.fixture()
        state['pending_source'].non_tensor_batch['candidate_original_question'][0] = 'wrong question'
        with self.assertRaisesRegex(ValueError, 'CANONICAL_QUESTION'):
            check_pending_membership(stream)

    def test_consumption_identity_and_empty_pending_are_strict(self):
        stream, state = self.fixture()
        state['consumed_source_rows'] += 1
        with self.assertRaisesRegex(ValueError, 'CONSUMPTION_IDENTITY'):
            check_pending_membership(stream)
        state.update(pending_source=None, pending_offset=0, consumed_source_rows=9)
        self.assertEqual(check_pending_membership(stream)['pending_rows_verified'], 0)


if __name__ == '__main__':
    unittest.main()
