"""Exercise the actual CPU identity function without importing GPU packages.

prepare_identity itself is pure Python. Load its exact AST to avoid importing
the surrounding module's Torch-only probability/audit readers on this Mac.
Only the native allocation reader is replaced by the small bound run fixture.
"""
import ast
import hashlib
from pathlib import Path
import unittest
import uuid

from reproduction.common8.topology import world_size


class Parents:
    def __init__(self, count):
        self.non_tensor_batch = {'index':list(range(count))}
    def __len__(self):
        return len(self.non_tensor_batch['index'])


class IdentityTests(unittest.TestCase):
    def prepare(self, run, count, outer=1, offset=0):
        source=Path(__file__).resolve().parents[2]/'reproduction/common8/behavior_runtime.py'
        tree=ast.parse(source.read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='prepare_identity')
        scope=dict(_native_run=lambda:(Path('/native-fixture'),run,'751077'),
                   world_size=world_size,uuid=uuid,
                   document_sha256=lambda value:hashlib.sha256(str(value).encode()).hexdigest())
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
        return scope['prepare_identity'](Parents(count),outer,repeats=5,candidate_prefix=True,candidate_offset=offset,
                                        attempt_id='a'*32,attempt_index=1)

    def run_config(self):
        return dict(run_id='dyn-test',source_sha256={},profile='common8',execution_site='hpc3',
                    training_world_size=4,execution_backend='slurm',dynrpo7b_selection=True,
                    data_selection_mode='dynrpo7b',actor_logical_micro_batch_per_rank=2,
                    question_probe={'probe_architecture':'qwen7b_L15_packed_group_variance_v1'},
                    policy_safety_mode='inline_first_five')

    def test_full_pools_and_original_g5(self):
        run=self.run_config()
        ordinary=self.prepare(run,5120)
        periodic=self.prepare(run,2560,outer=10)
        self.assertEqual(len(set(ordinary['trajectory_ids'])),5120*5)
        self.assertEqual(len(set(periodic['trajectory_ids'])),2560*5)
        self.assertTrue(ordinary['parent_group_ids'][4992].startswith('4992-'))
        self.assertEqual([value.rsplit('-',1)[1] for value in ordinary['trajectory_ids'][4992*5:4993*5]],
                         ['0','1','2','3','4'])
        with self.assertRaises(AssertionError):self.prepare(run,5120,outer=10)
        with self.assertRaises(AssertionError):self.prepare(run,128,offset=4992)

    def test_old_prefix_binding_remains_specific(self):
        run=self.run_config()
        legacy=dict(run,data_selection_mode='prefix20_probe',
                    question_probe={'probe_architecture':'qwen2_prefix20_trajectory_em_v1'})
        self.assertEqual(len(self.prepare(legacy,5120)['trajectory_ids']),25600)
        with self.assertRaises(AssertionError):self.prepare(legacy,2560,outer=10)
        with self.assertRaises(AssertionError):self.prepare(dict(run,dynrpo7b_selection=False),5120)
        with self.assertRaises(AssertionError):self.prepare(dict(run,question_probe=legacy['question_probe']),5120)


if __name__=='__main__':
    unittest.main()
