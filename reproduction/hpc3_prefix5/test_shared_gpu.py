"""Four physical cards, four train ranks, colocated retrieval and grounding."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .runtime import shared_devices
from .control import SBATCH
from reproduction.prefix_selection200 import gpu_layout, a800_runtime


class SharedGPUPlacementTests(unittest.TestCase):
    def fixture(self):
        return shared_devices([1,3,5,7],{i:'GPU-%d'%i for i in [1,3,5,7]})

    def test_four_noncontiguous_physical_cards_are_shared_not_eight(self):
        c=self.fixture();ids,uuids=gpu_layout.placement(c)
        self.assertEqual(ids,[1,3,5,7]);self.assertEqual(len(uuids),4)
        self.assertEqual(c['physical_gpu_indices'],c['auxiliary_gpu_indices'])
        self.assertEqual([gpu_layout.grounding_device(c,r) for r in range(4)],ids)
        self.assertIn('#SBATCH --gres=gpu:4\n',SBATCH)
        self.assertNotIn('--gres=gpu:8',SBATCH)

    def test_extra_or_duplicate_allocation_is_rejected(self):
        for ids in ([0,1,2,3,4,5,6,7],[0,1,2,2],[]):
            with self.assertRaises(ValueError):shared_devices(ids,{i:'GPU-%d'%i for i in ids})
        with self.assertRaises(ValueError):shared_devices([0,1,2,3],dict.fromkeys(range(4),'GPU-same'))

    def test_partial_alias_or_wrong_grounding_mapping_is_rejected(self):
        for change in (dict(auxiliary_gpu_indices=[1,3,5,6]),
                       dict(auxiliary_gpu_uuids=['GPU-other']*4),
                       dict(grounding_physical_gpus=[7]*4),
                       dict(grounding_refit_visible_device=0),dict(gpu_resource_layout='unknown')):
            c=self.fixture();c.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):gpu_layout.placement(c)

    def test_shared_admission_checks_each_unique_card_and_owns_no_foreign_process(self):
        c=self.fixture();c.update(execution_backend='slurm_hpc3',run_id='cpu',source_commit='a'*40)
        inventory={i:dict(index=i,uuid='GPU-%d'%i,memory_MiB=0) for i in [1,3,5,7]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'logs').mkdir()
            with patch.object(a800_runtime,'PREFIX',str(root)),patch.dict('os.environ',SLURM_JOB_ID='728815'), \
                 patch.object(a800_runtime,'gpu_inventory',return_value=inventory), \
                 patch.object(a800_runtime,'compute_processes',return_value=[]):
                receipt=a800_runtime.native_allocation(root,c)
                self.assertEqual(receipt['unique_physical_gpu_count'],4)
                self.assertEqual(len(receipt['actual_inventory']),4)
                self.assertFalse(receipt['foreign_processes_stopped'])
                with patch.object(a800_runtime,'compute_processes',return_value=[dict(uuid='GPU-3',pid=123)]):
                    with self.assertRaisesRegex(RuntimeError,'occupied'):a800_runtime.native_allocation(root,c)

    def test_legacy_separate_layout_remains_explicitly_distinct(self):
        c=self.fixture();c.pop('gpu_resource_layout')
        with self.assertRaises(ValueError):gpu_layout.placement(c)
        c.update(auxiliary_gpu_indices=[0,2,4,6],auxiliary_gpu_uuids=['GPU-%d'%i for i in [0,2,4,6]])
        self.assertEqual(len(gpu_layout.placement(c)[1]),8)


if __name__=='__main__':unittest.main()
