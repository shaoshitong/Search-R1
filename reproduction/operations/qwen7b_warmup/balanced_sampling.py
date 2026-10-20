"""Deterministic 1:1 whole-G5 sampling without replacement within an epoch."""
import hashlib,json,random


def plan(groups,seed,epoch,budget=67198):
    if budget<4 or budget%2:raise ValueError('Balanced question budget must be even and at least four')
    if len({g['qid'] for g in groups})!=len(groups):raise ValueError('Duplicate training question')
    positive=[];negative=[]
    for i,g in enumerate(groups):
        y=g['labels']
        if len(y)!=5 or any(v not in (0,1) for v in y):raise ValueError('Complete own-EM G5 required')
        (positive if 0<sum(y)<5 else negative).append(i)
    half=budget//2
    if min(len(positive),len(negative))<half:raise ValueError('Insufficient distinct questions for balanced epoch')
    rng=random.Random(seed+epoch*100003)
    rng.shuffle(positive);rng.shuffle(negative)
    ids=[x for pair in zip(positive[:half],negative[:half]) for x in pair]
    batches=[ids[i:i+51] for i in range(0,len(ids),51)]
    if len(batches[-1])<4:raise ValueError('Budget would leave fewer than four complete G5 groups')
    for batch in batches:rng.shuffle(batch)
    assert len(set(ids))==budget
    return batches


def proof(groups,batches):
    ids=[i for b in batches for i in b];ks=[sum(groups[i]['labels']) for i in ids]
    return dict(questions=len(ids),mixed=sum(0<k<5 for k in ks),nonmixed=sum(k in (0,5) for k in ks),all_wrong=ks.count(0),all_correct=ks.count(5),updates=len(batches),sha256=hashlib.sha256(json.dumps(batches,separators=(',',':')).encode()).hexdigest(),unique_questions=len(set(ids)),sampling='1:1 Mixed/nonMixed; without replacement within epoch; negatives naturally sampled')
