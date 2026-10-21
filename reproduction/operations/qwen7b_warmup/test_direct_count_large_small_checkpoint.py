"""One bounded synthetic CPU qualification, no official weights or real data."""
import copy
import gc
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from . import direct_count_large_small_checkpoint as a
from . import probe_ablation_train as u


class SmallCheckpointTest(unittest.TestCase):
    def test_small_fullstate_fourhash_and_scope_rejection(self):
        torch.set_num_threads(2)
        class Head(nn.Module):
            def __init__(self):
                super().__init__();self.attention=nn.Linear(6,1,bias=False);self.projection=nn.Linear(6,4);self.classifier=nn.Sequential(nn.Linear(12,128),nn.GELU(),nn.Linear(128,6))
        raw=Head();ema=copy.deepcopy(raw);opt=torch.optim.Adam(raw.parameters(),lr=1e-4)
        rng=u.rng_state(torch.device('cpu'))
        archive=[dict(rank=i,model_sha256='a'*64,ema_sha256='b'*64,optimizer_sha256='c'*64,steps=10464,rng=copy.deepcopy(rng))for i in range(4)]
        def state(cursor):
            components=dict(raw=u.model_hash(raw),ema=u.model_hash(ema),adam=u.optimizer_hash(opt),rng=u.rng_hash(rng))
            return dict(kind='LARGE_FROZEN_E1_HEAD_ONLY_FULL_STATE',head=dict(raw=copy.deepcopy(raw.state_dict()),ema=copy.deepcopy(ema.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),steps=cursor,accepted_hashes={k:components[k]for k in ('raw','ema','adam')},new_parameters_initialized_once=True,inherited_legacy_optimizer_reset=False),rng=copy.deepcopy(rng),cursor=cursor,epoch=0,config={},cache={},source_all_four_rank_states=copy.deepcopy(archive),frozen_legacy_hashes=dict(raw='a'*64,buffers='d'*64),initial_head_hashes={},accepted_hashes=components,data_cursor={})
        zero=state(0)
        sum(p.square().sum()for p in raw.parameters()).backward();opt.step()
        with torch.no_grad():
            for dst,src in zip(ema.parameters(),raw.parameters()):dst.mul_(.99).add_(src,alpha=.01)
        one=state(1)
        for value in (zero,one):
            a.geometry(value,a.shapes(6,4));self.assertEqual(a.component_hashes(value),value['accepted_hashes']);self.assertTrue(a.restore_small(value,6,4))
        self.assertEqual(a.archive_signature(zero),a.archive_signature(one))
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'small.pt';torch.save(one,p);bounds=a.zip_guard(p)
            self.assertLess(bounds['file_bytes'],1024**2)
            meta=torch.load(p,map_location='meta',mmap=True,weights_only=False);a.geometry(meta,a.shapes(6,4))
            loaded=torch.load(p,map_location='cpu',mmap=True,weights_only=False);self.assertEqual(a.component_hashes(loaded),one['accepted_hashes'])
            with self.assertRaises(ValueError):a.zip_guard(p,limit=32)
            # Shared-storage mmap must close before NFS removes the test file.
            del loaded,meta
            gc.collect()
        wrong=copy.deepcopy(one);wrong['raw']={}
        with self.assertRaises(ValueError):a.geometry(wrong,a.shapes(6,4))
        wrong=copy.deepcopy(one);wrong['head']['raw']['projection.weight']=torch.zeros(4,7)
        with self.assertRaises(ValueError):a.geometry(wrong,a.shapes(6,4))
        wrong=copy.deepcopy(one);wrong['head']['optimizer']['state'][0]['step'].fill_(0)
        with self.assertRaises(ValueError):a.component_hashes(wrong)
        wrong=copy.deepcopy(one);wrong['source_all_four_rank_states'][0]['optimizer']={}
        with self.assertRaises(ValueError):a.geometry(wrong,a.shapes(6,4))


if __name__=='__main__':unittest.main()
