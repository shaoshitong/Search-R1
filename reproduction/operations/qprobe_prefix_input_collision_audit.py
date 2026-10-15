"""Existing prefix metadata only: repeated inputs and conflicting fixed labels.

No tokens, tensors, model, GPU, inference or generation is read. These are
descriptive ceilings on this fixed cached dataset, not achievable heldout
generalization scores. Never use them to stop or alter an active fit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(cache):
    manifest=json.loads((cache/'manifest.json').read_text())
    assert sha(cache/'manifest.json')=='0d1a004c5677888dad4646c17525577bfe57acf1a413efd97ba8d36a59170298'
    assert sha(cache/'index.jsonl')==manifest['files']['index.jsonl']['sha256']
    rows=[json.loads(line) for line in (cache/'index.jsonl').read_text().splitlines()]
    result={}
    for split in ('train','test'):
        selected=[r for r in rows if r['split']==split]
        n=len(selected);positive=sum(r['strict_em'] for r in selected);negative=n-positive
        assert 0<positive<n
        summary={}
        for name,key in (('actual_token_position_prefix',lambda r:r['prefix_sha256']),
                         ('exact_cached_features_and_positions',lambda r:(r['feature_sha256'],r['position_ids_sha256'],r['length']))):
            groups=defaultdict(Counter)
            for row in selected:groups[key(row)][row['strict_em']]+=1
            conflicting=[g for g in groups.values() if g[0] and g[1]]
            bound=sum(max(g[0],g[1]) for g in groups.values())/n
            ba_bound=sum(max(.5*g[1]/positive,.5*g[0]/negative) for g in groups.values())
            min_bce=0.
            for g in groups.values():
                total=g[0]+g[1]
                for count in g.values():
                    if count:min_bce-=count/n*math.log(count/total)
            summary[name]=dict(unique_inputs=len(groups),repeated_rows=n-len(groups),
                conflicting_input_groups=len(conflicting),rows_with_conflicting_same_input_labels=sum(sum(g.values()) for g in conflicting),
                fixed_dataset_oracle_natural_accuracy_upper_bound=bound,
                fixed_dataset_oracle_balanced_accuracy_upper_bound=ba_bound,
                fixed_dataset_natural_BCE_lower_bound=min_bce)
        by_question=defaultdict(set)
        for row in selected:by_question[row['qid']].add(row['prefix_sha256'])
        summary['distinct_prefixes_per_question_histogram']=dict(sorted(Counter(len(v) for v in by_question.values()).items()))
        result[split]=dict(trajectories=n,questions=len(by_question),positive=positive,negative=negative,**summary)
    return dict(role='existing_prefix_hash_label_collision_metadata_only',manifest_sha256=sha(cache/'manifest.json'),
        scope='fixed cache descriptive oracle bounds; not generalization estimates; no extra model evaluation',
        raw_tokens_exported=False,model_or_tensor_read=False,metrics=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args();assert args.cache.is_absolute() and str(args.cache).startswith('/data/')
    print(json.dumps(audit(args.cache),allow_nan=False))
