"""Bound RPC payloads without changing the evaluator's native micro4 grouping."""
import json
import os


def chunked_call(original,owner,items,batch=32):
    if type(batch) is not int or batch<4 or batch%4:
        raise ValueError('Grounding RPC boundaries must preserve native micro4 groups')
    if not items:return original(owner,items)
    result=[]
    for start in range(0,len(items),batch):
        subset=items[start:start+batch]
        rows=original(owner,subset)
        if len(rows)!=len(subset):raise ValueError('Required evaluator omitted a chunk outcome')
        result.extend(rows)
        print('SEARCH_G1_GROUNDING_RPC_PROGRESS='+json.dumps(dict(rank=int(os.environ.get('RANK','-1')),
            completed_items=len(result),total_items=len(items),rpc_items=len(subset),
            evaluator_version=getattr(owner,'last_probe_version',None))),flush=True)
    assert len(result)==len(items)
    return result
