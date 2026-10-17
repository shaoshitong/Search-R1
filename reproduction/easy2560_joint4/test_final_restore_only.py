"""CPU mechanics and negative cases; no CUDA acceptance is claimed by tests."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import final_restore_only as module


class FinalRestoreOnlyTests(unittest.TestCase):
    def test_preparation_reuses_remote_tensor_and_copies_only_saved_grounding_heads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(); base = root/'target'; old = root/'source'
            (old/'data').mkdir(parents=True)
            for name in ('calibration-300.parquet', 'audit.json'): (old/'data'/name).write_bytes(name.encode())
            cp = old/'checkpoints/global_step_200'; cp.mkdir(parents=True)
            tensor = cp/'rank-0.pt'; tensor.write_bytes(b'CPU synthetic bytes must stay remote source')
            heads = old/'round150/heads'; heads.mkdir(parents=True)
            for name in ('commit_probe.pkl', 'boundary_probe.pkl', 'calibration-report.json'):
                (heads/name).write_bytes(name.encode())
            live = dict(run_id='r', round=150, version='r:150:saved', encoder='/remote/cp150',
                commit=str(heads/'commit_probe.pkl'), boundary=str(heads/'boundary_probe.pkl'),
                head_sha256={name:module.sha(heads/name) for name in ('commit_probe.pkl', 'boundary_probe.pkl')},
                calibration_report_sha256=module.sha(heads/'calibration-report.json'))
            manifest = old/'live.json'; manifest.write_text(json.dumps(live))
            path = old/'receipt.json'; path.write_text('{}')
            receipt = dict(source_job='producer', source_commit='a'*40,
                reward_probe_manifest={'path':str(manifest)}, reward_probe_round=150,
                reward_probe_version=live['version'], attempt_progress={'cumulative_writes':2345},
                driver_state={'source_consumed':512000})
            parent = dict(run_id='r', budget={'original':True}, initial_model='/remote/official',
                full_state_restore={'completed_outer':5}, question_probe=None)
            def bootstrap(*args):
                for name in ('repo', 'logs', 'preparation', 'data'): (base/name).mkdir(parents=True)
                (base/'repo/entry.py').write_text('pass\n')
            with patch('reproduction.easy2560_joint4.control.own_base', side_effect=Path), patch(
                    'reproduction.easy2560_joint4.control.bootstrap', side_effect=bootstrap), patch.object(
                    module, 'source', return_value=(path,receipt,cp,old,parent)), patch(
                    'reproduction.common8.source_files.runtime_files', return_value=[base/'repo/entry.py']):
                result = module.prepare(base, path, 'b'*40)
            run = module.read(base/'logs/run-manifest.json')
            self.assertEqual(run['restore_only_role'], module.ROLE)
            self.assertEqual(run['full_state_restore']['completed_outer'], 200)
            self.assertEqual(run['full_state_restore']['source_checkpoint'], str(cp))
            self.assertIsNone(run['question_probe']); self.assertEqual(run['budget'], parent['budget'])
            self.assertFalse(result['actual_cuda_restore_accepted']); self.assertEqual(result['tensor_copies'], 0)
            self.assertFalse(list(base.rglob('*.pt')))
            self.assertEqual(tensor.read_bytes(), b'CPU synthetic bytes must stay remote source')
            target = module.read(base/'artifacts/live/manifest.json')
            self.assertEqual(Path(target['commit']).read_bytes(), (heads/'commit_probe.pkl').read_bytes())
            self.assertEqual(target['version'], live['version'])

    def test_real_cpu_adam_scheduler_load_allowed_and_future_mutations_denied(self):
        import torch
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.]))], lr=1e-6)
        parameter = optimizer.param_groups[0]['params'][0]; parameter.grad = torch.tensor([.2]); optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        target = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.]))], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(target, lambda _:1.)
        schedule = copy.deepcopy(scheduler.state_dict())
        worker = SimpleNamespace(actor_optimizer=target, actor_lr_scheduler=scheduler,
            rollout=SimpleNamespace(inference_engine=SimpleNamespace(generate=lambda:None,
                llm_engine=SimpleNamespace(add_request=lambda:None))))
        module.install_guards(worker); target.load_state_dict(saved); scheduler.load_state_dict(schedule)
        from search_g1.full_state_restore import equal_tree
        equal_tree(target.state_dict(), saved); equal_tree(scheduler.state_dict(), schedule)
        for operation in (target.step, scheduler.step, worker.rollout.inference_engine.generate):
            with self.assertRaisesRegex(RuntimeError, 'MUTATION_FORBIDDEN'): operation()
        self.assertEqual(sum(worker._easy_final_forbidden_calls.values()), 3)

    def actor_fixture(self):
        rows = [dict(rank=i, actual_cuda_uuid='GPU-'+str(i), actual_optimizer_writes=2345,
            actual_counters={'completed_outer':200}, scheduler_last_epoch=200, scheduler_step_count=201,
            process_identity={'pid':100+i,'start_ticks':1000+i}, forbidden_call_counts={},
            model_equal=True, optimizer_equal=True, scheduler_equal=True, saved_rng_equal=True,
            model_hf_exact_tensors_checked_on_rank0=435 if i==0 else 0) for i in range(4)]
        return rows, dict(actual_cuda_runtime_uuids=['GPU-'+str(i) for i in range(4)]), dict(
            attempt_progress={'cumulative_writes':2345}, hf_saved_tensor_count=435)

    def test_all4_source_counters_and_actual435_model_readbacks_required(self):
        rows, allocation, receipt = self.actor_fixture(); module.validate_actors(rows, allocation, receipt)
        for field, value in (('saved_rng_equal',False), ('actual_optimizer_writes',4000), ('scheduler_last_epoch',201)):
            altered = copy.deepcopy(rows); altered[2][field] = value
            with self.assertRaises(ValueError): module.validate_actors(altered, allocation, receipt)
        rows[0]['model_hf_exact_tensors_checked_on_rank0'] = 434
        with self.assertRaisesRegex(ValueError, 'ALL435'): module.validate_actors(rows, allocation, receipt)

    def test_source_no201_covers_all_ten_ledgers_and_rejects201(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'logs').mkdir(); cp = root/'checkpoints/global_step_200'; cp.mkdir(parents=True)
            (root/'logs/actual-updates.jsonl').write_text('{"step":200}\n')
            for name in ['actual-attempts.jsonl'] + [role+'-rank-'+str(rank)+'.jsonl'
                    for role in ('optimizer','scheduler') for rank in range(4)]: (root/'logs'/name).write_text('{}\n')
            before = module.source_no201(root, cp); self.assertEqual(len(before['source_ledgers']), 10)
            (root/'logs/optimizer-rank-2.jsonl').write_text('changed\n')
            self.assertNotEqual(before['source_ledgers'], module.source_no201(root,cp)['source_ledgers'])
            (cp.parent/'global_step_201').mkdir()
            with self.assertRaisesRegex(ValueError,'NO201'): module.source_no201(root,cp)

    def test_entry_has_no_train_generation_update_or_save_calls(self):
        path = Path(module.__file__); tree = ast.parse(path.read_text())
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        forbidden = {'fit', '_validate', 'generate', 'generate_sequences', 'update_actor', 'save_checkpoint',
                     '_save_checkpoint', 'collect_pool', 'one', 'from_config'}
        self.assertFalse(called & forbidden)
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any('prefix20_restore_only' in name or 'prefix20_runtime' in name for name in imports))
        subprocess.run(['bash', '-n', str(path.with_suffix('.sh'))], check=True)

    def test_target_training_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'logs').mkdir(); module.no_training(root)
            (root/'logs/optimizer-rank-0.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'TRAINING_OR_201'): module.no_training(root)

    def test_partial_pending_membership_is_canonical_without_consuming_a_row(self):
        import torch
        class Pending:
            batch = {key:torch.tensor(value) for key, value in dict(qprobe_source_index=[2,0],
                pool_index=[2,0], canonical_source_index=[12,10], pool_cycle=[0,0]).items()}
            non_tensor_batch = dict(candidate_original_question=['q2','q0'], candidate_stable_id=['id2','id0'])
            def __len__(self): return 2
        pending = Pending()
        class ILoc:
            def __getitem__(self, key): return {'index':key}
        state = dict(candidate_order=[2,0,1], candidate_cursor=2, epoch=0, pending_source=pending,
                     pending_offset=1, consumed_source_rows=1)
        stream = SimpleNamespace(state=lambda:state, binding={'ordered_membership_sha256':'actual-pool'},
            loader=SimpleNamespace(dataset=SimpleNamespace(dataframe=SimpleNamespace(iloc=ILoc()),
                                                          prompt_key='prompt', indices=[10,11,12])))
        with patch('search_g1.qprobe_features.original_question_and_id',
                   side_effect=lambda row,_:('q'+str(row['index']),'id'+str(row['index']))):
            result = module.pending_membership(stream)
            self.assertEqual(result['pending_rows_verified'],2); self.assertEqual(state['pending_offset'],1)
            pending.batch['canonical_source_index'][1] = 11
            with self.assertRaisesRegex(ValueError,'CANONICAL_MEMBERSHIP'): module.pending_membership(stream)


if __name__ == '__main__': unittest.main()
