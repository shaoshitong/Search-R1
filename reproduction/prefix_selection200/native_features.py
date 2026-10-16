"""Read exact sampling-actor hidden mmap rows, never load/forward an encoder."""
import hashlib
import json
from pathlib import Path

from .actor_hidden import QMAX, PREFIX, WIDTH, DIM
from .contract import WORLD


class NativeFeatures:
    def __init__(self, directory):
        import numpy as np
        from .probe import prefix_row
        self.path = Path(directory)
        p = self.path/'scan-complete.json'
        self.manifest = json.loads(p.read_text())
        self.manifest_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        m = self.manifest
        if (not m.get('actor_hidden_native') or m.get('independent_encoder_forward') is not False
                or m['hidden_feature_actor_version'] != m['actor_version']
                or m['head_max_tokens'] != WIDTH or m['max_policy_prefix_tokens'] != PREFIX
                or len(m['actor_hidden_shards']) != WORLD):
            raise ValueError('Scoring must read actual current-actor native hidden')
        rows = [json.loads(line) for line in (self.path/'index.jsonl').open()]
        self.index = {r['trajectory_id']:prefix_row(r) for r in rows}
        if len(self.index) != m['count'] or m['count'] not in (2560,25600):
            raise ValueError('Native features require all complete unique G5 identities')
        self.positions = np.load(self.path/'packed_policy_positions.npy',mmap_mode='r')
        self.loaded = {}
        for rank, item in enumerate(m['actor_hidden_shards']):
            path = Path(item['path'])
            if (not str(path).startswith(('/mnt/', '/data/'))
                    or path.stat().st_size != item['bytes']):
                raise ValueError('Native hidden shard path/size differs')
            self.loaded[rank] = np.load(path,mmap_mode='r')
            if self.loaded[rank].shape != (item['count'],WIDTH,DIM) or self.loaded[rank].dtype != np.uint16:
                raise ValueError('Native BF16-bit shard shape differs')
        for row in self.index.values():
            if (row['feature_actor_version'] != m['actor_version'] or row['feature_layer'] != 27
                    or row['feature_boundary'] != 'post_block_hidden_plus_residual'
                    or not 0 <= row['feature_rank'] < WORLD
                    or not 0 <= row['feature_row'] < self.loaded[row['feature_rank']].shape[0]
                    or not 0 <= row['policy_length'] <= PREFIX):
                raise ValueError('Captured row belongs to another actor or layer')

    def get_batch(self, ids, device):
        import numpy as np
        import torch
        from reproduction.prefix20_online.model import expected_policy_positions
        rows = [self.index[tid] for tid in ids]
        features = torch.empty((len(ids),WIDTH,DIM),dtype=torch.bfloat16)
        question = torch.zeros((len(ids),WIDTH),dtype=torch.bool)
        policy = torch.zeros_like(question)
        positions = torch.zeros((len(ids),WIDTH),dtype=torch.long)
        for j,row in enumerate(rows):
            raw=np.asarray(self.loaded[row['feature_rank']][row['feature_row']]).copy()
            features[j].copy_(torch.from_numpy(raw).view(torch.bfloat16))
            qlen,plen=row['question_length'],row['policy_length']
            question[j,:qlen]=True;policy[j,QMAX:QMAX+plen]=True
            positions[j,:qlen]=torch.arange(qlen)
            off=row['policy_offset'];pos=self.positions[off:off+plen].tolist()
            if pos != (expected_policy_positions(row['segment_bounds']) if plen else []):
                raise ValueError('Native feature own-token positions changed')
            positions[j,QMAX:QMAX+plen]=torch.tensor(pos,dtype=torch.long)
        mask=question|policy
        features.masked_fill_(~mask.unsqueeze(-1),0)
        batch=dict(features=features,attention_mask=mask,position_ids=positions,
            question_mask=question,policy_mask=policy,
            question_lengths=torch.tensor([r['question_length'] for r in rows],dtype=torch.long),
            policy_lengths=torch.tensor([r['policy_length'] for r in rows],dtype=torch.long),
            policy_segments=[r['segment_bounds'] for r in rows],trajectory_ids=list(ids),
            qids=[str(r['qid']) for r in rows])
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

    def close(self):
        self.loaded.clear()
