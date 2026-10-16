import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search_g1.process_jit_cache import isolate,configure


class ProcessCacheTests(unittest.TestCase):
    def test_concurrent_ranks_and_reused_pids_have_disjoint_cache_roots(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=True):
            root=Path(directory).resolve()
            first=dict(host='ACD1-26',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid=123,start_ticks=100,uid=os.getuid())
            proofs=[]
            for pid,tick in [(123,100),(124,100),(123,200)]:
                identity=dict(first,pid=pid,start_ticks=tick);proofs.append(isolate(root,identity))
            paths=[p['paths']['TRITON_CACHE_DIR'] for p in proofs]
            self.assertEqual(len(set(paths)),3)
            self.assertEqual(isolate(root,first),proofs[0])
            marker=Path(paths[0])/'triton_.cubin';marker.write_bytes(b'complete kernel')
            self.assertFalse((Path(paths[1])/'triton_.cubin').exists())
            self.assertEqual(marker.read_bytes(),b'complete kernel')

    def test_other_tasks_remain_unchanged_and_unsafe_cache_is_rejected(self):
        with patch.dict(os.environ,{'SEARCH_G1_BASE':'/data/user/jxu957/experiments/topd-20261008','TRITON_CACHE_DIR':'existing'}):
            self.assertIsNone(configure());self.assertEqual(os.environ['TRITON_CACHE_DIR'],'existing')
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=True):
            root=Path(directory).resolve();(root/'cache').symlink_to(root,target_is_directory=True)
            identity=dict(host='ACD1-26',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid=123,start_ticks=100,uid=os.getuid())
            with self.assertRaisesRegex(ValueError,'NOT_OWNED'):isolate(root,identity)
            identity['host']='../foreign'
            with self.assertRaisesRegex(ValueError,'IDENTITY_INVALID'):isolate(root,identity)


if __name__=='__main__':unittest.main()
