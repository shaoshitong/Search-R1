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
from . import peer_audited122_l26_replay as e
from .replay_feature_reader import ReplayFeatureReader


def join_groups(identity,truth,expected=e.QUESTIONS):
    projected,_=e.split_identity(identity['groups'],expected)
    lookup={g['qid']:g for g in truth}
    if len(lookup)!=len(truth) or len(truth)!=expected:raise ValueError('Exact accepted raw delta ownEM groups required')
    result=[]
    for wanted in projected:
        g=lookup[wanted['qid']]
        if g['question_hash']!=wanted['question_hash'] or g['trajectory_ids']!=wanted['trajectory_ids']:raise ValueError('Pre-label identity/ordered ten own IDs differ')
        if len(g['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in g['labels']):raise ValueError('Ten actual own binary strictEM values required')
        result.append(dict(wanted,labels=g['labels']))
    return result


def merge(base):
    base=r.remote(base)
    if base!=e.TARGET:raise ValueError('Only this independent accepted122 feature bank target')
    with (base/'bank-acceptance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prep_ref=r.fingerprints(base/'preparation.json');prep=r.read(prep_ref['path'])
        if prep['status']!='AUDITED122_G10_PREFIX20_GEOMETRY_FOUR_SHARDS_PREPARED' or prep['blocked_shards'] or prep['questions']!=15616 or prep['trajectories']!=156160:raise ValueError('All original geometry must have passed, no truncation')
        identity=r.read(r.checked_file(prep['identity_plan'])['path']);projected,parts=e.split_identity(identity['groups'])
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
        first_ref=r.fingerprints(base/'first-four-chunks-independent-audit.json');first=r.read(first_ref['path'])
        if first['status']!='ALL_FOUR_ACTUAL_FIRST_LAYER26_CHUNKS_FRESH_SHA_AND_READER_PASSED' or first['preparation']!=prep_ref:raise ValueError('Actual independent first chunk qualifier missing')
        manifests=[];sources=[];allids=set();validator=object.__new__(ReplayFeatureReader);validator.dimension=r.DIM;validator.layer=26;packed_bytes=0
        for i,item in enumerate(completes):
            done=r.read(r.checked_file(item)['path']);selection=r.read(r.checked_file(done['source_selection'])['path'])
            declarations=[dict(qid=g['qid'],trajectory_id=t) for g in parts[i] for t in g['trajectory_ids']]
            if (done['status']!=e.STATUS or done['actor_updates']!=0 or done['probe_updates']!=0 or done['new_responses_generated']!=0 or done['questions']!=3904 or done['trajectories']!=39040 or done['batching']!=prep['batching'] or selection['trajectories']!=declarations or selection['source_ingest']!=prep['raw_complete'] or selection['question_plan']!=prep['identity_plan'] or done['compute_layers']!=[20,23,26] or done['persist_layers']!=[26]):raise ValueError('Actual immutable own-source/count/backend differs')
            if len(done['chunks'])!=122 or len(done['feature_manifests_by_capture_layer']['26'])!=122:raise ValueError('All122 complete320-trajectory chunks required')
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
                if count!=320 or count!=m['count']:raise ValueError('Actual index count differs')
                manifests.append(ref)
            if ownids!=set(expected_ids):raise ValueError('Whole authentic G10 feature coverage differs')
            allids.update(ownids);sources.append(dict(shard=i,complete=item,selection=done['source_selection'],features=done['features']))
            raw.atomic(base/'bank-acceptance-progress.json',dict(status='ACTUAL_DELTA_G10_METADATA_AND_OWN_ID_JOIN_ADVANCING',accepted_shards=i+1,trajectories=len(allids),actor_updates=0,time=time.time()))
        if len(allids)!=156160:raise ValueError('Exact15616 accepted realG10 required')
        accepted=r.read(r.checked_file(prep['raw_complete'])['path']);truth=r.read(r.checked_file(accepted['outputs']['groups'])['path'])['train'];groups=join_groups(identity,truth)
        if {t for g in groups for t in g['trajectory_ids']}!=allids:raise ValueError('OwnEM truth and feature identities differ')
        out.mkdir(exist_ok=True);raw.immutable(out/'groups.json',{'train':groups});raw.immutable(out/'features-layer26.json',manifests);raw.immutable(out/'sources.json',sources)
        done=dict(status='AUDITED122_REAL_G10_LAYER26_FEATURE_BANK_ACCEPTED',preparation=prep_ref,identity_plan=prep['identity_plan'],raw_complete=prep['raw_complete'],first_chunk_independent_audit=first_ref,outputs={n:r.fingerprints(out/n) for n in ('groups.json','features-layer26.json','sources.json')},questions=15616,trajectories=156160,domains=prep['domains'],manifests=len(manifests),packed_bytes=packed_bytes,capture_layer=26,compute_layers=[20,23,26],prefix_tokens=20,question_max_tokens=273,writer_layout='lossless_question_base_xor_policy_v1',all_own_IDs_strictEM_and_metadata_SHA_verified=True,producer_full_payload_SHA_and_reader_qualifications_preserved=True,payload_sizes_and_index_SHA_freshly_verified=True,payload_tensor_SHA_rehashed_in_this_metadata_merge=False,labels_consulted_for_identity_selection=False,existing30016_or_active_training_changed=False,probe_training_submitted=False,actor_updates=0,final_pool_read=False,time=time.time())
        raw.immutable(out/'complete.json',done);return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(e.TARGET));a=p.parse_args();print(json.dumps(merge(a.base),sort_keys=True))
