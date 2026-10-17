"""Read-only composition of four actual actor0050 native-hidden G5 datasets."""
import json
from pathlib import Path

from .native_features import NativeFeatures
from .probe import join_own_em, concatenate, batch_subset
from reproduction.prefix20_group_joint100.train_epoch import batch_to_device
from reproduction.prefix20_online.launch import sha


class CombinedFeatures:
    def __init__(self,dataset_manifest,selected_blocks=None):
        self.path=Path(dataset_manifest);self.manifest=json.loads(self.path.read_text());self.manifest_sha=sha(self.path)
        assert self.manifest['questions']==20480 and self.manifest['trajectories']==102400
        self.selected_blocks=list(range(len(self.manifest['blocks'])))if selected_blocks is None else list(selected_blocks)
        self.parts=[];self.owner={};self.index={}
        for block,source_block in enumerate(self.selected_blocks):
            entry=self.manifest['blocks'][source_block]
            part=NativeFeatures(entry['scan_path'],validate_reads=False);assert part.manifest_sha==entry['scan_sha256']
            assert part.manifest['actor_version']==self.manifest['actor_version']
            own=Path(entry['own_em']);assert sha(own)==entry['own_em_sha256']
            records=[json.loads(line)for line in own.open()]
            part.index=join_own_em(part.index,records,self.manifest['actor_version'],part.manifest_sha,entry['selection_sha256'])
            for tid,row in part.index.items():
                assert tid not in self.index
                self.index[tid]=dict(row,candidate_ordinal=block*5120+row['candidate_ordinal'])
                self.owner[tid]=block
            self.parts.append(part)
        assert len(self.index)==len(self.selected_blocks)*25600

    def get_batch(self,ids,device):
        chunks=[];ordered=[]
        for block,part in enumerate(self.parts):
            selected=[tid for tid in ids if self.owner[tid]==block]
            if selected:chunks.append(part.get_batch(selected,'cpu'));ordered.extend(selected)
        positions={tid:i for i,tid in enumerate(ordered)}
        batch=chunks[0]if len(chunks)==1 else concatenate(chunks)
        if ordered!=ids:batch=batch_subset(batch,[positions[tid]for tid in ids])
        return batch_to_device(batch,device)

    def close(self):
        for part in self.parts:part.close()


class QuestionSubsetFeatures:
    """Question-disjoint read-only view; all five siblings stay in one split."""
    def __init__(self,parent,groups,name,seed):
        import hashlib
        self.parent=parent;self.manifest=parent.manifest;self.selected_blocks=parent.selected_blocks
        self.binding=dict(source_manifest_sha256=parent.manifest_sha,split=name,seed=seed,
            questions=len(groups),question_ids=[g['qid']for g in groups])
        self.manifest_sha=hashlib.sha256(json.dumps(self.binding,sort_keys=True).encode()).hexdigest()
        self.index={tid:dict(parent.index[tid],candidate_ordinal=i)for i,g in enumerate(groups)for tid in g['trajectory_ids']}
    def get_batch(self,ids,device):return self.parent.get_batch(ids,device)
    def close(self):pass  # Parent lifetime is owned by the single trainer.


def split_question_features(parent,train_questions,seed):
    import random
    from .probe import candidate_groups
    groups=candidate_groups(parent.index,len(parent.index)//5)
    order=list(range(len(groups)));random.Random(seed).shuffle(order)
    selected=set(order[:train_questions])
    train=[g for i,g in enumerate(groups)if i in selected]
    test=[g for i,g in enumerate(groups)if i not in selected]
    return QuestionSubsetFeatures(parent,train,'train',seed),QuestionSubsetFeatures(parent,test,'test',seed)
