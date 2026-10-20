"""Metadata/ownEM acceptance after all four additional feature shards complete.

No GPU or training is submitted. Producer full-payload SHA+reader qualifications
are retained, not replaced by a claim that this metadata join rehashes tensors.
"""
import argparse,json,time
from pathlib import Path
from . import replay_probe_layers as r
from . import peer_newseed_ingest as raw
from . import peer_expanded_l26_replay as e
from .replay_feature_reader import ReplayFeatureReader


def join_groups(identity,truth,existing=8192,added=21824):
    e.split_new(identity,existing=existing,added=added);lookup={g['qid']:g for g in truth}
    if len(lookup)!=len(truth):raise ValueError('Canonical raw ownEM groups repeated')
    result=[]
    for wanted in identity['groups']:
        g=lookup[wanted['qid']]
        if g['question_hash']!=wanted['question_hash'] or g['trajectory_ids']!=wanted['trajectory_ids']:
            raise ValueError('Actual old5+new5 identities differ from pre-label plan')
        if len(g['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in g['labels']):
            raise ValueError('Ten actual own strictEM values required')
        result.append(dict(qid=g['qid'],question_hash=g['question_hash'],trajectory_ids=g['trajectory_ids'],labels=g['labels']))
    return result


def merge(base):
    base=r.remote(base)
    if base!=e.TARGET:raise ValueError('Exact isolated additional-feature target required')
    prep_ref=r.fingerprints(base/'preparation.json');prep=r.read(prep_ref['path']);identity=r.read(r.checked_file(prep['identity_plan'])['path']);parts=e.split_new(identity)
    out=base/'bank';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        done=r.read(out/'complete.json')
        if done['preparation']!=prep_ref:raise ValueError('Existing merged input has different source')
        for item in done['outputs'].values():r.checked_file(item)
        return done
    # Require every actual terminal producer completion BEFORE writing any bank.
    descriptors=[('prior',i,item) for i,item in enumerate(prep['prior_feature_completes'])]
    for i in range(4):
        p=base/('shard-%02d'%i)/'capture/complete.json'
        if not p.exists():raise ValueError('All four actual additional completions required; shard%d incomplete'%i)
        descriptors.append(('additional',i,r.fingerprints(p)))
    manifests=[];sources=[];allids=set();validator=object.__new__(ReplayFeatureReader);validator.dimension=r.DIM;validator.layer=26
    for kind,i,item in descriptors:
        done=r.read(r.checked_file(item)['path']);selection=r.read(r.checked_file(done['source_selection'])['path'])
        expected=identity['groups'][:8192][i::4] if kind=='prior' else parts[i]
        declarations=[dict(qid=g['qid'],trajectory_id=t) for g in expected for t in g['trajectory_ids']]
        if (done['status']!=(e.old.STATUS if kind=='prior' else e.STATUS) or done['actor_updates']!=0 or
            done['questions']!=len(expected) or done['trajectories']!=10*len(expected) or done['batching']!=prep['batching'] or
            selection['trajectories']!=declarations or selection['source_ingest']!=prep['raw_complete'] or
            selection['question_plan']!=(prep['prior_identity_plan'] if kind=='prior' else prep['identity_plan'])):
            raise ValueError('Actual own source/identity/backend differs')
        expected_ids={v['trajectory_id']:v['qid'] for v in declarations};ownids=set()
        for ref in done['feature_manifests_by_capture_layer']['26']:
            m=r.read(r.checked_file(ref)['path'])
            if (m['status']!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' or m['capture_layer']!=26 or m['model_revision']!=r.REVISION or
                m['prefix_tokens']!=20 or m['dimension']!=r.DIM or m['question_max_tokens']!=273):raise ValueError('Feature geometry/backend differs')
            for f in m['files'].values():
                if Path(f['path']).stat().st_size!=f['bytes']:raise ValueError('Producer-qualified payload size changed')
            index=r.checked_file(m['files']['index.jsonl']);count=0
            with Path(index['path']).open() as stream:
                for line in stream:
                    row=json.loads(line);validator.validate_row(row,m);tid=row['trajectory_id']
                    if tid in ownids or tid in allids or expected_ids.get(tid)!=row['qid']:raise ValueError('Wrong/repeated actual feature identity')
                    ownids.add(tid);count+=1
            if count!=m['count']:raise ValueError('Actual manifest index count differs')
            manifests.append(ref)
        if ownids!=set(expected_ids):raise ValueError('Whole own G10 feature coverage differs')
        allids.update(ownids);sources.append(dict(kind=kind,shard=i,complete=item,selection=done['source_selection']))
    if len(allids)!=300160:raise ValueError('30016 actual G10 questions required')
    accepted=r.read(r.checked_file(prep['raw_complete'])['path']);truth=r.read(r.checked_file(accepted['outputs']['groups'])['path'])['train'];groups=join_groups(identity,truth)
    raw.immutable(out/'groups.json',{'train':groups});raw.immutable(out/'features-layer26.json',manifests)
    raw.immutable(out/'sources.json',sources)
    done=dict(status='EXPANDED_REAL_G10_LAYER26_REUSED8192_PLUS_ADDITIONAL21824_ACCEPTED',preparation=prep_ref,identity_plan=prep['identity_plan'],raw_complete=prep['raw_complete'],
        outputs={name:r.fingerprints(out/name) for name in ('groups.json','features-layer26.json','sources.json')},questions=30016,trajectories=300160,prior_questions_reused=8192,new_questions_captured=21824,
        capture_layer=26,prefix_tokens=20,question_max_tokens=273,onepass_B32_updates=938,
        all_own_IDs_strictEM_and_metadata_SHA_verified=True,producer_full_payload_SHA_and_reader_qualifications_preserved=True,
        payload_sizes_and_index_SHA_freshly_verified=True,payload_tensor_SHA_rehashed_in_this_metadata_merge=False,
        labels_consulted_for_identity_selection=False,probe_training_submitted=False,actor_updates=0,final_pool_read=False,time=time.time())
    raw.immutable(out/'complete.json',done);return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(e.TARGET));a=p.parse_args();print(json.dumps(merge(a.base),sort_keys=True))
