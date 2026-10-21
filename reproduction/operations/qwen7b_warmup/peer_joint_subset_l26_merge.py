"""Accept the four finished audited122 delta feature shards, never alter30016.

The producer already rehashes every packed tensor and validates the real reader
before its complete receipt. This independent metadata join freshly verifies
all manifest/index SHAs, sizes, masks/positions and exact own G10 identity/EM.
It explicitly does not claim a second full tensor bulk rehash.
"""
import argparse,fcntl,json,time
from pathlib import Path
from . import replay_probe_layers as r
from . import peer_newseed_ingest as raw
from . import peer_joint_subset_l26 as e
from .replay_feature_reader import ReplayFeatureReader


def projection(rows,expected):
    if len(rows)!=expected:raise ValueError('Exact accepted train subset count required')
    qs=set();ts=set();out=[]
    for g in rows:
        if g['qid']in qs or len(g['trajectory_ids'])!=10 or len(set(g['trajectory_ids']))!=10 or ts.intersection(g['trajectory_ids']):raise ValueError('Real own ten IDs/qids must be unique')
        qs.add(g['qid']);ts.update(g['trajectory_ids']);out.append({k:g[k]for k in ('qid','question_hash','trajectory_ids')})
    return out,[out[i::4]for i in range(4)]


def qualify_first(base):
    from .replay_bank_capture import verify_completed_chunk
    out=base/'first-four-chunks-independent-audit.json'
    if out.exists():return r.read(out)
    started=time.monotonic();prep=r.read(base/'preparation.json');lookup={r.read(v['path'])['shard']:v for v in prep['selections']};items=[]
    for i in range(4):
        ref=lookup[i];sel=r.read(r.checked_file(ref)['path']);ids=[v['trajectory_id']for v in sel['trajectories'][:320]];p=base/('shard-%02d'%i)/'capture/chunk-00000';done=verify_completed_chunk(p,ids)
        if done['source_selection']!=ref or done['batching']!=prep['batching']or done['actor_updates']or set(done['feature_manifests'])!={'26'}:raise ValueError('First complete chunk provenance differs')
        manifest=done['feature_manifests']['26'];reader=ReplayFeatureReader([manifest],26)
        try:
            if set(reader.records)!=set(ids):raise ValueError('Actual first chunk ownIDs differ')
            reader.get([ids[0],ids[len(ids)//2],ids[-1]],'cpu')
        finally:reader.close()
        m=r.read(manifest['path']);items.append(dict(shard=i,chunk=r.fingerprints(p/'complete.json'),manifest=manifest,trajectories=len(ids),files=len(m['files']),payload_bytes=sum(v['bytes']for v in m['files'].values()),all_payload_SHA_fresh=True,reader_three_exact_rows_reconstructed=True))
    v=dict(status='ALL_FOUR_ACTUAL_FIRST_LAYER26_CHUNKS_FRESH_SHA_AND_READER_PASSED',source_publication=r.fingerprints(base/'source-publication.json'),preparation=r.fingerprints(base/'preparation.json'),shards=items,trajectories=sum(x['trajectories']for x in items),actor_updates=0,probe_updates=0,elapsed_seconds=time.monotonic()-started,epoch=time.time());raw.immutable(out,v);return v

def join_groups(identity,truth,expected=None):
    projected,_=projection(identity['groups'],expected)
    lookup={g['qid']:g for g in truth}
    if len(lookup)!=len(truth):raise ValueError('Exact accepted raw delta ownEM groups required')
    result=[]
    for wanted in projected:
        g=lookup[wanted['qid']]
        if g['question_hash']!=wanted['question_hash'] or g['trajectory_ids']!=wanted['trajectory_ids']:raise ValueError('Pre-label identity/ordered ten own IDs differ')
        if len(g['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in g['labels']):raise ValueError('Ten actual own binary strictEM values required')
        result.append(dict(wanted,labels=g['labels']))
    return result


def merge(base,cohort=None):
    base=r.remote(base);prep0=r.read(base/'preparation.json');c=cohort or dict(target=str(base),questions=prep0['questions'],tag='JOINT_TRAIN_SUBSET',shard_status='JOINT_TRAIN_SUBSET_REAL_G10_L26_SHARD_COMPLETE')
    expected=c['questions']
    base=r.remote(base)
    if base!=r.remote(c['target']):raise ValueError('Only this independent accepted122 feature bank target')
    with (base/'bank-acceptance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prep_ref=r.fingerprints(base/'preparation.json');prep=r.read(prep_ref['path'])
        if prep['status']!='ACCEPTED_TRAIN_SUBSET_GEOMETRY_FOUR_SHARDS_PREPARED' or prep['blocked_shards'] or prep['questions']!=expected or prep['trajectories']!=expected*10:raise ValueError('All original geometry must have passed, no truncation')
        identity=r.read(r.checked_file(prep['identity_plan'])['path']);projected,parts=projection(identity['groups'],expected)
        out=base/'bank'
        if (out/'complete.json').exists():
            done=r.read(out/'complete.json')
            if done['preparation']!=prep_ref:raise ValueError('Existing bank source differs')
            for item in done['outputs'].values():r.checked_file(item)
            return done
        # No output bank is written until all four real producer completions exist.
        completes=[]
        for i in range(4):
            p=base/('shard-%02d'%i)/'capture/complete.json'
            if not p.exists():raise ValueError('Actual shard%d completion absent; no incomplete bank accepted'%i)
            completes.append(r.fingerprints(p))
        qualify_first(base)
        first_ref=r.fingerprints(base/'first-four-chunks-independent-audit.json');first=r.read(first_ref['path'])
        if first['status']!='ALL_FOUR_ACTUAL_FIRST_LAYER26_CHUNKS_FRESH_SHA_AND_READER_PASSED' or first['preparation']!=prep_ref:raise ValueError('Actual independent first chunk qualifier missing')
        manifests=[];sources=[];allids=set();validator=object.__new__(ReplayFeatureReader);validator.dimension=r.DIM;validator.layer=26;packed_bytes=0
        for i,item in enumerate(completes):
            per_shard=len(parts[i]);chunks_per_shard=(per_shard*10+319)//320
            done=r.read(r.checked_file(item)['path']);selection=r.read(r.checked_file(done['source_selection'])['path'])
            declarations=[dict(qid=g['qid'],trajectory_id=t) for g in parts[i] for t in g['trajectory_ids']]
            if (done['status']!=c['shard_status'] or done['actor_updates']!=0 or done['probe_updates']!=0 or done['new_responses_generated']!=0 or done['questions']!=per_shard or done['trajectories']!=per_shard*10 or done['batching']!=prep['batching'] or selection['trajectories']!=declarations or selection['source_ingest']!=prep['raw_complete'] or selection['question_plan']!=prep['identity_plan'] or done['compute_layers']!=[20,23,26] or done['persist_layers']!=[26]):raise ValueError('Actual immutable own-source/count/backend differs')
            if len(done['chunks'])!=chunks_per_shard or len(done['feature_manifests_by_capture_layer']['26'])!=chunks_per_shard:raise ValueError('All122 complete320-trajectory chunks required')
            expected_ids={v['trajectory_id']:v['qid'] for v in declarations};ownids=set()
            for ref in done['feature_manifests_by_capture_layer']['26']:
                m=r.read(r.checked_file(ref)['path'])
                if (m['status']!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' or m['layout']!='lossless_question_base_xor_policy_v1' or m['capture_layer']!=26 or m['model_revision']!=r.REVISION or m['prefix_tokens']!=20 or m['dimension']!=3584 or m['question_max_tokens']!=273 or m['actor_optimizer_updates']!=0):raise ValueError('Qualified lossless feature geometry differs')
                for f in m['files'].values():
                    if Path(f['path']).stat().st_size!=f['bytes']:raise ValueError('Producer-qualified payload size changed')
                    packed_bytes+=f['bytes']
                index=r.checked_file(m['files']['index.jsonl']);count=0
                with Path(index['path']).open() as stream:
                    for line in stream:
                        row=json.loads(line);validator.validate_row(row,m);tid=row['trajectory_id']
                        if tid in ownids or tid in allids or expected_ids.get(tid)!=row['qid']:raise ValueError('Wrong/repeated authentic feature ID/question')
                        ownids.add(tid);count+=1
                if not 0<count<=320 or count!=m['count']:raise ValueError('Actual index count differs')
                manifests.append(ref)
            if ownids!=set(expected_ids):raise ValueError('Whole authentic G10 feature coverage differs')
            allids.update(ownids);sources.append(dict(shard=i,complete=item,selection=done['source_selection'],features=done['features']))
            raw.atomic(base/'bank-acceptance-progress.json',dict(status='ACTUAL_DELTA_G10_METADATA_AND_OWN_ID_JOIN_ADVANCING',accepted_shards=i+1,trajectories=len(allids),actor_updates=0,time=time.time()))
        if len(allids)!=expected*10:raise ValueError('Exact15616 accepted realG10 required')
        accepted=r.read(r.checked_file(prep['raw_complete'])['path']);truth=r.read(r.checked_file(accepted['outputs']['groups'])['path'])['train'];groups=join_groups(identity,truth,expected)
        if {t for g in groups for t in g['trajectory_ids']}!=allids:raise ValueError('OwnEM truth and feature identities differ')
        out.mkdir(exist_ok=True);raw.immutable(out/'groups.json',{'train':groups});raw.immutable(out/'features-layer26.json',manifests);raw.immutable(out/'sources.json',sources)
        done=dict(status=c['tag']+'_REAL_G10_LAYER26_FEATURE_BANK_ACCEPTED',preparation=prep_ref,identity_plan=prep['identity_plan'],raw_complete=prep['raw_complete'],first_chunk_independent_audit=first_ref,outputs={n:r.fingerprints(out/n) for n in ('groups.json','features-layer26.json','sources.json')},questions=expected,trajectories=expected*10,domains=prep['domains'],manifests=len(manifests),packed_bytes=packed_bytes,capture_layer=26,compute_layers=[20,23,26],prefix_tokens=20,question_max_tokens=273,writer_layout='lossless_question_base_xor_policy_v1',all_own_IDs_strictEM_and_metadata_SHA_verified=True,producer_full_payload_SHA_and_reader_qualifications_preserved=True,payload_sizes_and_index_SHA_freshly_verified=True,payload_tensor_SHA_rehashed_in_this_metadata_merge=False,labels_consulted_for_identity_selection=False,existing30016_or_active_training_changed=False,probe_training_submitted=False,actor_updates=0,final_pool_read=False,time=time.time())
        raw.immutable(out/'complete.json',done);return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();print(json.dumps(merge(a.base),sort_keys=True))
