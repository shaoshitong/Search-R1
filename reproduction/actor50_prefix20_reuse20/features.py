"""Fixed prepared Prefix20 rows. The head sees features/masks/positions only."""
from pathlib import Path
import json
import numpy as np
import torch

from reproduction.prefix_selection200.probe import candidate_groups
from .data import QMAX,PREFIX,WIDTH,DIM,read,sha


class Features20:
    def __init__(self,path):
        self.path=Path(path);self.manifest=read(self.path/'features-complete.json')
        meta=read(self.path/'prepared.json');assert sha(self.path/'index.jsonl')==meta['index_sha256']
        self.index={r['trajectory_id']:r for r in map(json.loads,(self.path/'index.jsonl').open())}
        assert len(self.index)==2560
        self.groups=candidate_groups(self.index,512)
        self.loaded={rank:np.load(item['path'],mmap_mode='r',allow_pickle=False) for rank,item in enumerate(self.manifest['shards'])}
        for rank,array in self.loaded.items():assert array.shape==(640,WIDTH,DIM) and array.dtype==np.uint16

    def get_batch(self,ids,device):
        rows=[self.index[tid] for tid in ids]
        features=torch.empty((len(rows),WIDTH,DIM),dtype=torch.bfloat16)
        question=torch.zeros((len(rows),WIDTH),dtype=torch.bool);policy=torch.zeros_like(question)
        positions=torch.zeros((len(rows),WIDTH),dtype=torch.long)
        for i,row in enumerate(rows):
            raw=np.asarray(self.loaded[row['feature_rank']][row['feature_row']]).copy()
            features[i].copy_(torch.from_numpy(raw).view(torch.bfloat16))
            qlen,plen=row['question_length'],row['policy_length']
            question[i,:qlen]=True;policy[i,QMAX:QMAX+plen]=True
            positions[i,:qlen]=torch.arange(qlen)
            positions[i,QMAX:QMAX+plen]=torch.tensor(row['policy_positions'],dtype=torch.long)
        mask=question|policy;features.masked_fill_(~mask.unsqueeze(-1),0)
        result=dict(features=features,attention_mask=mask,position_ids=positions,
            question_mask=question,policy_mask=policy,question_lengths=torch.tensor([r['question_length'] for r in rows]),
            policy_lengths=torch.tensor([r['policy_length'] for r in rows]),policy_segments=[r['segment_bounds'] for r in rows],
            trajectory_ids=list(ids),qids=[str(r['qid']) for r in rows])
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in result.items()}

    def close(self):self.loaded.clear()
