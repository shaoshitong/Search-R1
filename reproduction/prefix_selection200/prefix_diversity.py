"""Outcome-free, probe-free deterministic selection of five of ten prefixes."""
import hashlib
import json
import math

MODE='ten_native_prefixes_farthest_first_five_v1'


def enabled(config):
    return config.get('prefix_diversity',{}).get('mode')==MODE and not config.get('diagnostic_only')


def draws(config):return 10 if enabled(config) else 5


def distance(left,right):
    """Equal mean of normalized positional mismatch and token-bigram Jaccard.

    Only actual authored tokens enter; masked padding and question tokens do
    not. Exact sequence equality is handled separately to guarantee maximum
    distinct-prefix coverage before duplicates can fill a shortfall.
    """
    a,b=tuple(left),tuple(right)
    if a==b:return 0.
    n=max(len(a),len(b));pos=(sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b)))/max(1,n)
    def grams(x):return set(zip(x,x[1:])) if len(x)>1 else {(v,)for v in x}
    x,y=grams(a),grams(b);union=x|y
    return .5*pos+.5*(1-len(x&y)/len(union) if union else 0.)


def select(candidates):
    """Return original input indices; retain full native sample identities."""
    if len(candidates)!=10:raise ValueError('Exactly ten actual prefixes required')
    tokens=[tuple(row['policy_token_ids'])for row in candidates]
    if any(any(type(t)is not int for t in row)for row in tokens):raise ValueError('Actual integer token IDs required')
    matrix=[[distance(a,b)for b in tokens]for a in tokens]
    representatives=[];seen=set()
    for i,value in enumerate(tokens):
        if value not in seen:seen.add(value);representatives.append(i)
    first=max(representatives,key=lambda i:(math.fsum(matrix[i][j]for j in representatives),-i))
    chosen=[first]
    while len(chosen)<min(5,len(representatives)):
        remaining=[i for i in representatives if i not in chosen]
        chosen.append(max(remaining,key=lambda i:(min(matrix[i][j]for j in chosen),math.fsum(matrix[i][j]for j in chosen),-i)))
    while len(chosen)<5:
        remaining=[i for i in range(10)if i not in chosen]
        chosen.append(max(remaining,key=lambda i:(min(matrix[i][j]for j in chosen),math.fsum(matrix[i][j]for j in chosen),-i)))
    # Give the downstream five slots stable original-draw order, never relabel
    # the native trajectory ID or change its retained tokens/log probabilities.
    chosen.sort()
    def mean(ids):return math.fsum(matrix[i][j]for n,i in enumerate(ids)for j in ids[n+1:])/(len(ids)*(len(ids)-1)/2)
    proof=dict(mode=MODE,selected_input_indices=chosen,input_count=10,retained_count=5,
        unique_input_prefixes=len(seen),unique_retained_prefixes=len({tokens[i]for i in chosen}),
        mean_pairwise_distance_before=mean(list(range(10))),mean_pairwise_distance_after=mean(chosen),
        probe_outputs_used=False,outcome_labels_used=False,
        candidates=[dict(trajectory_id=row['trajectory_id'],sampled_response_index=row['response_index'],
            policy_token_ids=list(tokens[i]),prefix_sha256=row['prefix_sha256'],selected=i in chosen)
            for i,row in enumerate(candidates)])
    if proof['unique_retained_prefixes']!=min(5,len(seen)):raise AssertionError('Distinct-prefix coverage lost')
    proof['selection_sha256']=hashlib.sha256(json.dumps(proof,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return chosen,proof
