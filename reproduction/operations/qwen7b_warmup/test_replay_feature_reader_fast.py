"""Small bit-identity and corruption checks for the optimized feature decoder."""
import unittest
from pathlib import Path
import tempfile
import torch
from . import test_replay_feature_reader as fixture_module
from .replay_feature_reader import ReplayFeatureReader
from .replay_feature_reader_fast import FastReplayFeatureReader

class FastReaderTests(unittest.TestCase):
    def test_exact_own_delta_padding_duplicate_ids_and_independent_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            ref, expected = fixture_module.ReplayReaderTests().fixture(Path(folder))
            readers=[cls([ref],26,synthetic_fixture=True,fixture_dimension=2) for cls in (ReplayFeatureReader,FastReplayFeatureReader)]
            try:
                for ids in (['t0','t1','t2','t0'], ['t2','t1','t0']):
                    old,new=[r.get(ids,'cpu') for r in readers]
                    for key in ('features','attention_mask','position_ids'):
                        self.assertTrue(torch.equal(old[key],new[key]),key)
                    for key in ('trajectory_ids','qids','capture_layer','capture_kind','model_revision'):
                        self.assertEqual(old[key],new[key],key)
            finally:
                for reader in readers:reader.close()

    def test_disk_corruption_after_cached_read_still_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            ref,_=fixture_module.ReplayReaderTests().fixture(Path(folder))
            reader=FastReplayFeatureReader([ref],26,synthetic_fixture=True,fixture_dimension=2)
            try:
                reader.get(['t0','t1'],'cpu')
                path=Path(ref['path']).parent/'policy.bin';raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
                with self.assertRaises(ValueError):reader.get(['t0'],'cpu')
            finally:reader.close()

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
