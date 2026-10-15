"""Native retrieval must not query devices outside a four-GPU allocation."""
import os
import unittest
from unittest import mock

from reproduction.common8.retriever import native_gpu_binding


class RetrieverTopology(unittest.TestCase):
    def allocation(self, count):
        return dict(visible_devices=[str(i) for i in range(count)],
                    actual_cuda_runtime_uuids=['GPU-'+str(i) for i in range(count)])

    def test_four_allocated_devices_do_not_query_eight(self):
        allocation=self.allocation(4)
        def query(count=8):
            if count > 4:
                raise RuntimeError('CUDA invalid device ordinal outside the allocation')
            return allocation['actual_cuda_runtime_uuids']
        with mock.patch.dict(os.environ, {'SEARCH_G1_TRAINING_WORLD_SIZE':'4'}), \
                mock.patch('reproduction.common8.native_allocation.cuda_uuids',side_effect=query) as reader:
            self.assertEqual(native_gpu_binding(dict(profile='common8',training_world_size=4),
                allocation,'0,1,2,3'),(4,allocation['actual_cuda_runtime_uuids']))
        reader.assert_called_once_with(4)

    def test_legacy_eight_card_allocation_still_checks_all_eight(self):
        allocation=self.allocation(8)
        with mock.patch.dict(os.environ, {'SEARCH_G1_TRAINING_WORLD_SIZE':'8'}), \
                mock.patch('reproduction.common8.native_allocation.cuda_uuids',
                           return_value=allocation['actual_cuda_runtime_uuids']) as reader:
            self.assertEqual(native_gpu_binding(dict(profile='common8',training_world_size=8),
                allocation,','.join(allocation['visible_devices']))[0],8)
        reader.assert_called_once_with(8)

    def test_wrong_visible_assignment_is_rejected_before_cuda_query(self):
        with mock.patch.dict(os.environ, {'SEARCH_G1_TRAINING_WORLD_SIZE':'4'}), \
                mock.patch('reproduction.common8.native_allocation.cuda_uuids') as reader:
            with self.assertRaises(AssertionError):
                native_gpu_binding(dict(profile='common8',training_world_size=4),self.allocation(4),'0,1,2,4')
        reader.assert_not_called()

    def test_wrong_or_duplicate_cuda_identity_is_rejected(self):
        for actual in [['GPU-0']*4,['GPU-0','GPU-1','GPU-2','foreign']]:
            with mock.patch.dict(os.environ, {'SEARCH_G1_TRAINING_WORLD_SIZE':'4'}), \
                    mock.patch('reproduction.common8.native_allocation.cuda_uuids',return_value=actual):
                with self.assertRaises(AssertionError):
                    native_gpu_binding(dict(profile='common8',training_world_size=4),self.allocation(4),'0,1,2,3')


if __name__=='__main__':unittest.main()
