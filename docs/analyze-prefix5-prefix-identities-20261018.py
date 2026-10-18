"""Read-only remote metadata audit of equal predicted probabilities and Prefix5.

Compares stored token/position hashes, not regenerated text or tensor payloads.
Raw trajectory data stays remote; returns only per-outer aggregate evidence.
"""
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_bytes())


def main():
    base=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2')
    original=base.with_name('dynrpo-prefix5-thread5-20261017')
    out=base/'analysis/probe-mixed-diagnostics-20261018-v1/prefix-identity-summary.json'
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if out.exists():
        assert read(out)['source_sha256']==source_sha
        print(out.read_text());return
    results=[]
    for n in range(10,111,10):
        score=base/'outers'/f'outer-{n:06d}'/'probe-score'
        manifest=read(score/'manifest.json')
        raw=(score/'index.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest['index_sha256']
        index={r['trajectory_id']:r for r in json.loads(raw)}
        scores=(score/'candidate-scores.jsonl').read_bytes()
        assert hashlib.sha256(scores).hexdigest()==manifest['candidate_scores_sha256']
        groups=[json.loads(line) for line in scores.splitlines() if line]
        own=read(score.parent/'full-pool/complete.json')['own_em']
        rawem=Path(own['path']).read_bytes()
        assert hashlib.sha256(rawem).hexdigest()==own['sha256']
        em={r['trajectory_id']:r['strict_em'] for r in map(json.loads,rawem.splitlines())}
        same_prefix=same_context_positions=same_p=mixed_same_prefix=same_p_different_prefix=0
        uq=[]
        for group in groups:
            rows=[index[t] for t in group['trajectory_ids']]
            tokens={r['policy_prefix_sha256'] for r in rows}
            signatures={(r['policy_prefix_sha256'],r['position_ids_sha256'],
                         json.dumps(r['segment_bounds'],sort_keys=True)) for r in rows}
            equal_token=len(tokens)==1
            equal_p=max(group['trajectory_probabilities'])-min(group['trajectory_probabilities'])<1e-10
            mixed=0<sum(em[t] for t in group['trajectory_ids'])<5
            same_prefix+=equal_token;same_context_positions+=len(signatures)==1
            same_p+=equal_p;mixed_same_prefix+=equal_token and mixed
            same_p_different_prefix+=equal_p and not equal_token
            uq.append(len(tokens))
        results.append(dict(outer=n,groups=len(groups),same_prefix5_groups=same_prefix,
                            same_prefix5_and_policy_positions_groups=same_context_positions,
                            same_five_p_groups=same_p,
                            same_p_but_different_prefix5_groups=same_p_different_prefix,
                            true_mixed_among_same_prefix5=mixed_same_prefix,
                            mean_unique_prefix5_per_question=sum(uq)/len(uq),
                            index_sha256=manifest['index_sha256'],
                            scores_sha256=manifest['candidate_scores_sha256'],
                            own_em_sha256=own['sha256']))
    result=dict(status='SAVED_PREFIX5_IDENTITY_AND_P_EQUALITY_VERIFIED',epoch=time.time(),
                source_sha256=source_sha,rows=results,
                scope='Stored policy-token hashes and policy positions; no hidden tensor or model read. Same-prefix equality is expected for deterministic causal probe inference.',
                all_row_data_remote=True,training_modified=False)
    with out.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
