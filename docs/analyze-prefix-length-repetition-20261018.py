"""Remote CPU-only Prefix5/10/20 duplication from already completed rollouts.

Uses original sampled policy token IDs, excluding observations. Does not sample,
load model/tensor payloads, alter Prefix5 training, or start a resume job.
"""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

BASE=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2')
ORIGINAL=BASE.with_name('dynrpo-prefix5-thread5-20261017')
OUT=BASE/'analysis/prefix5-10-20-repetition-20261018'


def read(path):return json.loads(path.read_bytes())


def publish(name,value):
    path=OUT/name;temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


class HashedReader:
    def __init__(self,handle):self.handle=handle;self.digest=hashlib.sha256();self.count=0
    def read(self,n=-1):
        data=self.handle.read(n);self.digest.update(data);self.count+=len(data);return data


def summarize(groups,length):
    eligible={qid:rows for qid,rows in groups.items() if all(r['length']>=length for r in rows)}
    full_same=any_same=pair_same=mixed_n=mixed_same=unique=0
    histogram=collections.Counter()
    for rows in eligible.values():
        prefixes=[tuple(r['prefix'][:length]) for r in rows];counts=collections.Counter(prefixes)
        full_same+=len(counts)==1;any_same+=len(counts)<5
        pair_same+=sum(n*(n-1)//2 for n in counts.values());unique+=len(counts);histogram[len(counts)]+=1
        mixed=0<sum(r['em'] for r in rows)<5;mixed_n+=mixed;mixed_same+=mixed and len(counts)==1
    n=len(eligible)
    return dict(prefix_tokens=length,total_groups=len(groups),all_five_reach_length_groups=n,
                insufficient_length_groups=len(groups)-n,full_length_coverage=n/len(groups),
                all_five_identical_groups=full_same,all_five_identical_fraction=full_same/n if n else None,
                any_pair_identical_groups=any_same,any_pair_identical_fraction=any_same/n if n else None,
                identical_pairs=pair_same,total_pairs=n*10,pair_identical_fraction=pair_same/(10*n) if n else None,
                mean_distinct_prefixes=unique/n if n else None,distinct_prefix_histogram=dict(histogram),
                true_mixed_eligible_groups=mixed_n,true_mixed_all_five_identical_groups=mixed_same,
                true_mixed_all_five_identical_fraction=mixed_same/mixed_n if mixed_n else None)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(OUT/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (OUT/'summary.json').exists():
        assert read(OUT/'summary.json')['source_sha256']==source_sha
        print(json.dumps(dict(status='EXISTING_COMPLETE',output=str(OUT/'summary.json'))));return
    started=time.time();results=[]
    # Initial selected512 is reported separately; all periodic pools have2560.
    for n in [1]+list(range(10,111,10)):
        root=(ORIGINAL if n==1 else BASE)/'outers'/f'outer-{n:06d}'
        phase=root/('full' if n==1 else 'full-pool');receipt=read(phase/'complete.json')
        manifest=read(root/'probe-score/manifest.json');indexraw=(root/'probe-score/index.json').read_bytes()
        assert hashlib.sha256(indexraw).hexdigest()==manifest['index_sha256']
        index={r['trajectory_id']:r for r in json.loads(indexraw)}
        groups=collections.defaultdict(list);seen=set();sources=[];rows_read=0
        for shard in receipt['shards']:
            path=Path(shard['path'])
            with path.open('rb') as handle:
                hashed=HashedReader(handle)
                with gzip.open(hashed,'rt') as stream:
                    for line in stream:
                        row=json.loads(line);tid=row['trajectory_id'];assert tid not in seen;seen.add(tid)
                        assert row['outer']==n and row['actor_version']==manifest['actor_version']
                        ids=row['response_token_ids'];mask=row['response_info_mask'];assert len(ids)==len(mask)
                        assert set(mask)<=set([0,1]);policy=[token for token,keep in zip(ids,mask) if keep]
                        expected=index[tid]['policy_prefix_sha256']
                        canonical=json.dumps(policy[:5],sort_keys=True,separators=(',',':')).encode()
                        assert hashlib.sha256(canonical).hexdigest()==expected
                        # Independently recover the retained policy stream from action receipts.
                        turns=[token for turn in row['behavior_turns']
                               for token in turn['raw_generated_token_ids'][:turn['kept_token_count']]]
                        assert turns==policy
                        groups[str(row['qid'])].append(dict(prefix=policy[:20],length=len(policy),em=row['strict_em']))
                        rows_read+=1
                assert hashed.digest.hexdigest()==shard['sha256'] and hashed.count==shard['bytes']
            sources.append(dict(path=str(path),sha256=shard['sha256'],bytes=shard['bytes']))
            publish('progress.json',dict(stage='STREAMING_SAVED_ROLLOUTS',outer=n,shards_complete=len(sources),rows_read=rows_read,completed_outers=[r['outer'] for r in results]))
        assert all(len(rows)==5 for rows in groups.values())
        assert len(groups)==(512 if n==1 else 2560)
        item=dict(outer=n,actor_version=manifest['actor_version'],
                  scope='selected512_selection_biased' if n==1 else 'complete2560_candidate_pool',
                  source_shards=sources,original_prefix5_binding_verified_for_every_trajectory=True,
                  policy_mask_and_retained_action_stream_equal=True,
                  by_length=[summarize(groups,length) for length in [5,10,20]])
        # Same questions for all three lengths: removes changing denominator effects.
        shared={qid:rows for qid,rows in groups.items() if all(r['length']>=20 for r in rows)}
        item['same_questions_all_reach20']=[summarize(shared,length) for length in [5,10,20]]
        results.append(item);publish('partial-summary.json',dict(rows=results))
    result=dict(status='SAVED_PREFIX5_10_20_REPETITION_ANALYSIS_COMPLETE',epoch=time.time(),
                elapsed_seconds=time.time()-started,source_sha256=source_sha,rows=results,
                token_definition='Original sampled policy tokens with response_info_mask1; exclude retrieval observations, no text retokenization, no new rollout',
                primary_definition='All five trajectory prefixes exactly equal, conditional on all five reaching length L',
                same_question_comparison='Use same_questions_all_reach20 to compare5/10/20 with a common denominator',
                limits='Token repetition only. Does not measure Prefix10/20 probe prediction, fitting quality, hidden equality, or eventual training benefit.',
                training_modified=False,gpu_used=False,all_raw_data_remain_remote=True)
    publish('summary.json',result);publish('progress.json',dict(stage='COMPLETE',elapsed_seconds=time.time()-started))
    print(json.dumps(dict(status=result['status'],path=str(OUT/'summary.json'),elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
