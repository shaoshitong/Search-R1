"""Label-blind whole-G5 plans shared by all20k arms; compatible mixup pairs."""
from collections import defaultdict
import json
import random


def layout_key(group,index):
    return tuple((index[t]['question_length'],index[t]['policy_length'],
                  json.dumps(index[t]['segment_bounds'],sort_keys=True))for t in group['trajectory_ids'])


def plan_epoch(keys,seed,epoch):
    count=len(keys)
    if count not in (5120,20480):raise ValueError('Use5120 or20480 complete G5 questions')
    steps=count*100//5120
    rng=random.Random(seed+epoch*100003);buckets=defaultdict(list)
    for i,key in enumerate(keys):buckets[key].append(i)
    pairs=[];singles=[]
    for values in buckets.values():
        rng.shuffle(values)
        while len(values)>=2:pairs.append((values.pop(),values.pop()))
        singles.extend(values)
    rng.shuffle(pairs);rng.shuffle(singles)
    def take_single():
        if not singles:
            a,b=pairs.pop();singles.extend((a,b))
        return singles.pop()
    result=[];extras_start=0
    for step in range(steps):
        n=52 if(step+1)%5==0 else 51;counts=[6]*8
        for j in range(n-48):counts[(extras_start+j)%8]+=1
        extras_start=(extras_start+n-48)%8
        question_indices=[];rank_pairs=[]
        for size in counts:
            local=[];paired=[]
            while len(local)+1<size:
                if pairs:
                    a,b=pairs.pop();paired.append((len(local),len(local)+1));local.extend((a,b))
                else:local.extend((take_single(),take_single()))
            if len(local)<size:local.append(take_single())
            question_indices.extend(local);rank_pairs.append(paired)
        result.append(dict(question_indices=question_indices,global_questions=n,rank_questions=counts,mix_pairs=rank_pairs))
    flattened=[i for row in result for i in row['question_indices']]
    assert len(flattened)==len(set(flattened))==count and not pairs and not singles
    assert all(sum(row['rank_questions'][r]for row in result)==count//8 for r in range(8))
    return result
