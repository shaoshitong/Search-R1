"""Read SHA-committed score ledgers; transfer only aggregate scalars."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssh_retained18

REMOTE = r'''
import hashlib,json,math,os,time
from pathlib import Path
from collections import Counter,defaultdict
os.nice(19)
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
out=root/'runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73/output/final-eval-fast'
manifest=json.loads((out/'asset-manifest.json').read_text())
identity={k:manifest[k] for k in ('config_sha256','model_sha256','dataset_sha256')}
expected=manifest['expected_source_counts'];sources=defaultdict(Counter)
seen=set();cost=Counter();ranks=Counter();proofs=[];started=time.time()
for done in sorted((out/'scores/batches').glob('*.done.json')):
    receipt=json.loads(done.read_text());p=done.with_name(done.name.replace('.done.json','.json'))
    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
    v=json.loads(raw);assert v['identity']==identity and v['prefix_count']==0
    assert len(v['rows'])==receipt['count']
    indices=[r['index'] for r in v['rows']]
    assert indices==list(range(indices[0],indices[0]+len(indices)))
    for r in v['rows']:
        i=r['index'];assert 0<=i<51713 and i not in seen;seen.add(i)
        assert r['source'] in expected
        for k in ('f1','em','answer_format','format_rate','generated_tokens','search_calls'):
            z=r[k];assert type(z) in (int,float) and math.isfinite(z) and z>=0
            if k not in ('generated_tokens','search_calls'):assert z<=1
            sources[r['source']][k]+=z
        sources[r['source']]['count']+=1
    for k,z in v['cost'].items():
        if type(z) in (int,float):
            assert math.isfinite(z) and z>=0;cost[k]+=z
    ranks[str(v['rank'])]+=len(indices)
    proofs.append(dict(name=p.name,bytes=len(raw),sha256=receipt['sha256'],count=receipt['count']))
summary={}
for s,need in expected.items():
    c=sources[s];n=c['count'];assert n<=need
    summary[s]=dict(count=n,expected=need,source_complete=n==need,
        f1=c['f1']/n if n else None,em=c['em']/n if n else None,
        generated_tokens=c['generated_tokens'],search_calls=c['search_calls'])
print(json.dumps(dict(schema='tree_fast_final_committed_scalar_progress_v1',epoch=time.time(),
    observed_start_epoch=started,identity=identity,evaluated_count=len(seen),target51713_complete=len(seen)==51713,
    partial_source_metrics=summary,rank_rows=dict(ranks),cost_of_committed_batches_only=dict(cost),
    cost_scope='Batch process CPU/wall sums overlap across ranks; neither full service wall nor exclusive GPU billing. In-flight batch cost excluded.',
    proof_count=len(proofs),batch_proofs=proofs,raw_rows_transferred=False)))
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a=p.parse_args();output=Path(a.output)
    assert not output.exists()
    os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
    started=time.time()
    r=ssh_retained18.run('python3 -B -c '+shlex.quote(REMOTE), timeout=90)
    result=dict(returncode=r.returncode,stderr=r.stderr,local_wall_seconds=time.time()-started)
    if r.returncode==0:result['remote']=json.loads(r.stdout)
    else:result['stdout']=r.stdout
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    data=result.get('remote',{})
    print(json.dumps(dict(returncode=r.returncode,evaluated_count=data.get('evaluated_count'),
                         partial_source_metrics=data.get('partial_source_metrics'))))

if __name__=='__main__':main()
