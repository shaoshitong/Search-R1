"""Reuse authenticated completed G5 traces and their original strict EM labels."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time

QMAX=273
PREFIX=20
WIDTH=QMAX+PREFIX
DIM=2048


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(path):
    out=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):out.update(block)
    return out.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_bytes(canonical(value));temp.replace(path)


def retained(row,limit=PREFIX):
    """First actual kept policy tokens, with original causal requests per turn."""
    ids=[];positions=[];segments=[]
    for turn in row['behavior_turns']:
        take=min(turn['kept_token_count'],limit-len(ids))
        if take<=0:
            if len(ids)==limit:break
            continue
        request=turn['actual_request_token_ids'];tokens=turn['raw_generated_token_ids'][:take]
        assert len(tokens)==take and all(type(t) is int for t in request+tokens)
        segments.append(dict(turn_index=turn['turn_index'],policy_start=len(ids),policy_length=take,
                             position_start=len(request),request_length=len(request),input_ids=request+tokens))
        ids.extend(tokens);positions.extend(range(len(request),len(request)+take))
    flat=[token for token,policy in zip(row['response_token_ids'],row['response_info_mask']) if policy]
    assert ids==flat[:limit] and len(ids)==min(limit,len(flat))
    return ids,positions,segments


def repetition(rows):
    groups=collections.defaultdict(list)
    for row in rows:groups[str(row['qid'])].append(row)
    result={}
    for length in (5,10,20):
        full=0;same=0;mixed=0;same_mixed=0;all_dist=collections.Counter()
        for siblings in groups.values():
            assert len(siblings)==5 and {r['response_index'] for r in siblings}==set(range(5))
            all_dist[len({tuple(r['policy_ids'][:length]) for r in siblings})]+=1
            if any(r['policy_length']<length for r in siblings):continue
            full+=1;eq=len({tuple(r['policy_ids'][:length]) for r in siblings})==1;same+=eq
            truth=0<sum(r['strict_em'] for r in siblings)<5;mixed+=truth;same_mixed+=truth and eq
        result[str(length)]=dict(full_length_G5_groups=full,short_G5_groups=len(groups)-full,
            full_length_all5_same=same,fraction_all5_same=same/full if full else None,
            full_length_trueMixed=mixed,full_length_trueMixed_all5_same=same_mixed,
            all_groups_unique_retained_prefix_count_distribution=dict(all_dist))
    return result


def prepare(base):
    base=Path(base);c=read(base/'preparation/config.json');source=Path(c['source_prefix5_base'])
    assert read(source/'training/cycle-0010/complete.json')['completed_epochs']==20
    assert read(source/'logs/cleanup-status.json')['owned_births_all_gone']
    output=[];train_qids=set();test_qids=set()
    for number in range(11):
        name='test' if number==0 else 'cycle-%04d'%number
        bundle=read(source/'fixed-test/complete.json' if number==0 else source/'cycles'/('cycle-%04d.json'%number))
        scan=Path(bundle['scan_path']);old_manifest=read(scan/'scan-complete.json')
        assert sha(scan/'scan-complete.json')==bundle['scan_sha256']
        assert sha(scan/'index.jsonl')==old_manifest['files']['index.jsonl']['sha256']
        original={r['trajectory_id']:r for r in map(json.loads,(scan/'index.jsonl').open())}
        assert len(original)==2560 and bundle['actor_version']==c['actor_version']
        assert sha(bundle['own_em'])==bundle['own_em_sha256']
        labels={r['trajectory_id']:r for r in map(json.loads,Path(bundle['own_em']).open())}
        assert set(labels)==set(original)
        full_path=scan.parents[1]/'full/complete.json';full=read(full_path)
        assert sha(full_path)==bundle['full_complete_sha256']
        target=base/'data'/name;target.mkdir(parents=True,exist_ok=False)
        requests={rank:[] for rank in range(4)};rows=[];seen=set()
        for shard in full['shards']:
            path=Path(shard['path']);assert sha(path)==shard['sha256'] and path.stat().st_size==shard['bytes']
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt') as stream:
                for line in stream:
                    record=json.loads(line);tid=record['trajectory_id'];old=original[tid];label=labels[tid]
                    assert tid not in seen and record['actor_version']==c['actor_version']
                    assert record['strict_em']==label['strict_em'] and type(record['strict_em']) is int and record['strict_em'] in (0,1)
                    assert record['prefix_sha256']==old['prefix_sha256']==label['prefix_sha256']
                    assert record['prefix_scan_sha256']==bundle['scan_sha256']
                    ids,positions,segments=retained(record)
                    if segments:
                        assert segments[0]['turn_index']==0
                        assert segments[0]['input_ids'][:segments[0]['request_length']]==record['initial_prompt_token_ids']
                    assert hashlib.sha256(canonical(ids[:5])).hexdigest()==old['policy_prefix_sha256']
                    assert hashlib.sha256(canonical(positions[:5])).hexdigest()==old['position_ids_sha256']
                    qlen=len(record['initial_prompt_token_ids']);assert 1<=qlen<=QMAX
                    assert qlen==old['initial_context']['request_length']
                    row={k:old[k] for k in ('trajectory_id','qid','question_hash','candidate_ordinal','response_index','feature_rank','feature_row','prefix_sha256')}
                    row.update(actor_version=c['actor_version'],question_length=qlen,policy_length=len(ids),
                        policy_ids=ids,policy_positions=positions,strict_em=record['strict_em'],
                        segment_bounds=[{k:s[k] for k in ('turn_index','policy_start','policy_length','position_start')} for s in segments])
                    rows.append(row);seen.add(tid)
                    requests[old['feature_rank']].append(dict(trajectory_id=tid,feature_row=old['feature_row'],segments=segments))
            publish(base/'logs/status.json',dict(stage='REUSING_STORED_TRACES',dataset=name,read_trajectories=len(rows),actor_updates=0,checked_unix=time.time()))
        assert seen==set(original) and len(rows)==2560
        qids={str(r['qid']) for r in rows};assert len(qids)==512
        if number:
            assert not qids&train_qids;train_qids|=qids
        else:test_qids=qids
        assert not train_qids&test_qids
        # Native row addresses stay unchanged. Preserve question+first5 bit-for-bit.
        rows.sort(key=lambda r:(r['candidate_ordinal'],r['response_index']))
        with (target/'index.jsonl').open('x') as stream:
            for row in rows:stream.write(canonical(row).decode()+'\n')
        for rank in range(4):
            assert len(requests[rank])==640 and {r['feature_row'] for r in requests[rank]}==set(range(640))
            with (target/('rank-%d-requests.jsonl'%rank)).open('x') as stream:
                for row in requests[rank]:stream.write(canonical(row).decode()+'\n')
        value=dict(role='same_frozen_actor50_existing_G5_prefix20_inputs',dataset=name,cycle=number,
            actor_version=c['actor_version'],questions=512,trajectories=2560,
            source_prefix5_bundle=bundle,source_full_complete_sha256=sha(full_path),source_full_shards=full['shards'],
            source_native_feature_shards=old_manifest['actor_hidden_shards'],index_sha256=sha(target/'index.jsonl'),
            request_sha256={str(rank):sha(target/('rank-%d-requests.jsonl'%rank)) for rank in range(4)},
            repetition=repetition(rows),new_rollouts=0,strict_EM_recomputed=False,original_strict_EM_reused=True,
            first5_and_question_features='original_native_BF16_rows_preserved',
            extra15_features='frozen_actor0050_causal_teacher_forcing_of_stored_tokens',checked_unix=time.time())
        publish(target/'prepared.json',value);output.append(value)
    assert len(train_qids)==5120 and len(test_qids)==512 and not train_qids&test_qids
    result=dict(role='all10_train_G5_plus_fixed512_test_reused',datasets=output,
        train_questions=5120,test_questions=512,total_trajectories=28160,
        frozen_actor=c['actor_version'],new_rollouts=0,training_test_disjoint=True,checked_unix=time.time())
    publish(base/'preparation/data-prepared.json',result);return result
