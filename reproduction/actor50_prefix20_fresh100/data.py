"""New non-overlapping question plans and native20 feature adoption."""
import json
from pathlib import Path
import numpy as np

from reproduction.actor50_prefix20_reuse20.data import read,sha,publish,canonical
from reproduction.actor50_probe100.pipeline import question_bundles
from reproduction.prefix_selection200.probe import candidate_groups,join_own_em


def fresh_plan(base,c):
    import hashlib
    eligible,test=question_bundles(c)
    original=Path(c['original_prefix20_base']);old=set();old_hashes=set()
    for cycle in range(1,11):
        for row in map(json.loads,(original/'data'/('cycle-%04d'%cycle)/'index.jsonl').open()):
            old.add(str(row['qid']));old_hashes.add(row['question_hash'])
    assert len(old)==5120 and {str(r['qid']) for r in eligible[:5120]}==old
    normalized_old={r['normalized_question_hash'] for r in eligible[:5120]}
    excluded_test={str(r['qid']) for r in test};selected=[];seen=set(old);texts=set(normalized_old)
    for row in eligible[5120:]:
        qid=str(row['qid']);text=row['normalized_question_hash']
        if qid in seen or qid in excluded_test or text in texts or row['question_hash'] in old_hashes:continue
        selected.append(row);seen.add(qid);texts.add(text)
        if len(selected)==40*512:break
    assert len(selected)==20480 and not {str(r['qid']) for r in selected}&old
    ordered=[{k:r[k] for k in ('qid','question_hash','normalized_question_hash','source_index')} for r in selected]
    proof=dict(old_first20_epoch_train_questions_excluded=5120,new_distinct_questions=20480,
        fixed_test_excluded=True,normalized_question_repetition_rejected=True,
        ordered_new_identity_sha256=hashlib.sha256(canonical(ordered)).hexdigest(),
        source_canonical_sha256=c['canonical_train_sha256'],first_new_cycle=11,last_new_cycle=50)
    target=Path(base)/'preparation/fresh-plan-proof.json'
    if target.exists():assert read(target)==proof
    else:publish(target,proof)
    return selected,proof


def adopt(base,cycle,prefix,full):
    base=Path(base);c=read(base/'preparation/config.json');scan=Path(prefix['scan_path'])
    assert sha(scan/'scan-complete.json')==prefix['scan_sha256']
    manifest=read(scan/'scan-complete.json')
    assert manifest['actor_version']==c['actor_version'] and manifest['max_policy_prefix_tokens']==20 and manifest['head_max_tokens']==293
    assert sha(scan/'index.jsonl')==manifest['files']['index.jsonl']['sha256']
    raw={r['trajectory_id']:r for r in map(json.loads,(scan/'index.jsonl').open())}
    assert sha(full['own_em']['path'])==full['own_em']['sha256']
    labels=[json.loads(line) for line in Path(full['own_em']['path']).open()]
    joined=join_own_em(raw,labels,c['actor_version'],prefix['scan_sha256'],full['selection_sha256'])
    groups=candidate_groups(joined,512);positions=np.load(scan/'packed_policy_positions.npy',mmap_mode='r',allow_pickle=False)
    rows=[]
    for group in groups:
        for tid in group['trajectory_ids']:
            row=joined[tid];out={k:row[k] for k in ('trajectory_id','qid','question_hash','candidate_ordinal','response_index','feature_rank','feature_row','prefix_sha256','strict_em')}
            plen=row['prefix_length'];offset=row['policy_offset'];assert 0<=plen<=20
            out.update(actor_version=c['actor_version'],question_length=row['initial_context']['request_length'],
                policy_length=plen,policy_positions=positions[offset:offset+plen].tolist(),segment_bounds=row['segment_bounds'])
            rows.append(out)
    target=base/'data'/('cycle-%04d'%cycle);target.mkdir(parents=True,exist_ok=False)
    with (target/'index.jsonl').open('x') as f:
        for row in rows:f.write(canonical(row).decode()+'\n')
    for rank,shard in enumerate(manifest['actor_hidden_shards']):
        assert shard['count']==640 and Path(shard['path']).stat().st_size==shard['bytes']
    publish(target/'prepared.json',dict(role='fresh_native_prefix20_G5_training_data',cycle=cycle,index_sha256=sha(target/'index.jsonl'),
        actor_version=c['actor_version'],questions=512,trajectories=2560,source_scan_sha256=prefix['scan_sha256'],
        own_EM_sha256=full['own_em']['sha256'],new_rollout=True,reused_previous_training_data=False,
        native_actor_layer27_all20_positions=True,new_encoder_forward=False))
    publish(target/'features-complete.json',dict(dataset='cycle-%04d'%cycle,shards=manifest['actor_hidden_shards'],
        actor_version=c['actor_version'],prefix_tokens=20))
    publish(base/'cycles'/('cycle-%04d.json'%cycle),dict(cycle=cycle,questions=512,trajectories=2560,
        prefix_scan_sha256=prefix['scan_sha256'],full_complete_sha256=sha(base/'outers'/('outer-%06d'%cycle)/'full/complete.json'),
        labels_sha256=full['own_em']['sha256'],actual_mixed=full['actual_selected_mixed_questions'],fresh_G5=True))
    return rows
