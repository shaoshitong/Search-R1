"""Use the exported eligible membership; never train on its fixed diagnostics."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from search_g1.qprobe_runtime import QProbeCandidateStream

POOL_SHA='ae13710254e7ae77cf5531b62a755ec2000bc3fba20051bbafaff1f6714aa8fc'
TRAIN_SHA='5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1'


def eligible_indices(dataframe,prompt_key,records):
    """Verify exported canonical offsets against stable IDs and both text hashes."""
    from search_g1.qprobe_features import original_question_and_id
    from reproduction.prefix20_online.pool import question_hash,normalized_hash
    lookup={}
    for i,(_,row) in enumerate(dataframe.iterrows()):
        question,qid=original_question_and_id(row,prompt_key)
        if qid in lookup:raise ValueError('PREFIX20_SOURCE_STABLE_ID_REPEATS')
        lookup[qid]=(i,question_hash(question),normalized_hash(question))
    ids=[r['qid'] for r in records];normalized=[r['normalized_question_hash'] for r in records]
    if len(set(ids))!=len(ids) or len(set(normalized))!=len(normalized):
        raise ValueError('PREFIX20_EXPORTED_ELIGIBLE_POOL_REPEATS')
    indices=[]
    for row in records:
        actual=lookup.get(row['qid'])
        index=row['source_index']
        if type(index) is not int or not 0<=index<len(dataframe):
            raise ValueError('PREFIX20_CANONICAL_SOURCE_INDEX_INVALID')
        if actual is None or actual!=(index,row['question_hash'],row['normalized_question_hash']):
            raise ValueError('PREFIX20_ELIGIBLE_QUESTION_ABSENT_OR_CHANGED')
        indices.append(index)
    return indices


class EligibleDataset:
    def __init__(self,source,indices):
        self.source=source;self.indices=indices;self.prompt_key=source.prompt_key
        self.dataframe=source.dataframe.iloc[indices]
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):return self.source[self.indices[index]]


class Prefix20CandidateStream(QProbeCandidateStream):
    def __init__(self,loader,eligible_pool_path,restored_state=None):
        from reproduction.prefix20_joint4.assets import sha
        path=Path(eligible_pool_path)
        if not path.is_absolute() or not path.is_file() or sha(path)!=POOL_SHA:
            raise ValueError('PREFIX20_EXPORTED_ELIGIBLE_POOL_SHA_DIFFERS')
        pool=json.loads(path.read_text())
        if (pool['role']!='canonical_nonoverlap_question_pool' or pool['canonical_train_sha256']!=TRAIN_SHA or
                pool['canonical_rows']!=169028 or pool['eligible_unique_questions']!=168510 or
                len(pool['records'])!=168510 or pool['excluded_diagnostic_questions']!=512):
            raise ValueError('PREFIX20_EXPORTED_ELIGIBLE_POOL_CONTRACT_DIFFERS')
        source=loader.dataset
        if len(source.dataframe)!=pool['canonical_rows']:
            raise ValueError('PREFIX20_CANONICAL_DATASET_WAS_FILTERED_OR_REINDEXED')
        self._initialize(loader,pool['records'],restored_state)

    def _initialize(self,loader,records,restored_state=None):
        """A verified membership initializes a fresh sampler, never epoch68 cursor."""
        from torch.utils.data import RandomSampler
        source=loader.dataset
        indices=eligible_indices(source.dataframe,source.prompt_key,records)
        eligible=EligibleDataset(source,indices)
        proxy=SimpleNamespace(dataset=eligible,sampler=RandomSampler(eligible),collate_fn=loader.collate_fn)
        mapping=[dict(pool_index=i,canonical_source_index=index,qid=records[i]['qid'])
                 for i,index in enumerate(indices)]
        raw=json.dumps(mapping,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        self.binding=dict(stream_kind='prefix20_exported_eligible_v1',canonical_train_sha256=TRAIN_SHA,
            eligible_pool_sha256=POOL_SHA,ordered_membership_sha256=hashlib.sha256(raw).hexdigest(),
            eligible_questions=len(records),cycle_policy='permutation_once_defer_current_pool_at_boundary')
        if restored_state is not None and restored_state.get('prefix20_pool_binding')!=self.binding:
            raise ValueError('PREFIX20_RESTORED_POOL_BINDING_DIFFERS')
        super().__init__(proxy,restored_state=restored_state)
        self._pool_seen=None

    def state(self):
        if self._pool_seen is not None:raise ValueError('PREFIX20_CANNOT_SNAPSHOT_PARTIAL_CANDIDATE_POOL')
        return dict(super().state(),prefix20_pool_binding=dict(self.binding))

    def _fill(self):
        source=self._source
        if source.cursor==len(source.order):
            shuffled=list(iter(self.loader.sampler));seen=self._pool_seen or set()
            source.order=[i for i in shuffled if i not in seen]+[i for i in shuffled if i in seen]
            source.cursor=0;source.epoch+=1
        super()._fill()
        import torch
        pending=source.pending
        indices=pending.batch['qprobe_source_index'].tolist()
        pending.batch['pool_index']=torch.tensor(indices,dtype=torch.long)
        pending.batch['canonical_source_index']=torch.tensor([self.loader.dataset.indices[i] for i in indices],dtype=torch.long)
        pending.batch['pool_cycle']=torch.full((len(indices),),source.epoch,dtype=torch.long)

    def one(self):
        if self._pool_seen is None:raise ValueError('PREFIX20_CANDIDATES_REQUIRE_EXPLICIT_POOL_TRANSACTION')
        import torch
        row=super().one();index=int(row.batch['qprobe_source_index'][0])
        if index in self._pool_seen:raise ValueError('PREFIX20_QUESTION_REPEATED_INSIDE_CANDIDATE_POOL')
        row.batch['pool_index']=torch.tensor([index],dtype=torch.long)
        row.batch['canonical_source_index']=torch.tensor([self.loader.dataset.indices[index]],dtype=torch.long)
        row.batch['pool_cycle']=torch.tensor([self.epoch],dtype=torch.long)
        self._pool_seen.add(index)
        return row

    def collect_pool(self,count):
        from search_g1.qprobe_runtime import _collect_pool
        if self._pool_seen is not None or type(count) is not int or not 0<count<=len(self.loader.dataset):
            raise ValueError('PREFIX20_INVALID_OR_NESTED_CANDIDATE_POOL')
        self._pool_seen=set();before=self.consumed
        try:
            result=_collect_pool(self,count)
            if self.consumed-before!=count or len(self._pool_seen)!=count:
                raise ValueError('PREFIX20_CANDIDATE_ROWS_WERE_DROPPED_OR_REPEATED')
            return result
        finally:self._pool_seen=None

    def materialize(self,candidate_parents,ordinals):
        import torch
        for ordinal in ordinals:
            pool_index=int(candidate_parents.batch['qprobe_source_index'][ordinal])
            if (int(candidate_parents.batch['pool_index'][ordinal])!=pool_index or
                    int(candidate_parents.batch['canonical_source_index'][ordinal])!=self.loader.dataset.indices[pool_index]):
                raise ValueError('PREFIX20_SELECTED_POOL_CANONICAL_INDEX_DIFFERS')
        result=super().materialize(candidate_parents,ordinals)
        for name in ('pool_index','canonical_source_index','pool_cycle'):
            result.batch[name]=torch.tensor([int(candidate_parents.batch[name][i]) for i in ordinals],dtype=torch.long)
        return result
