"""Accept every canonical question's authentic G5, never mix sampling events."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from .download_model import atomic, fingerprints, REVISION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    base=Path(args.base);config=json.loads((base/'config.json').read_text())
    if (base/'bank/complete.json').exists():
        print('EXISTING_COMPLETE_BANK_NOT_REGENERATED');return
    expected=json.loads((base/'inputs/manifest.json').read_text())
    groups, features, seen={},[],set()
    for shard in range(8):
        root=base/'sampling'/('shard-%02d'%shard)
        done=json.loads((root/'complete.json').read_text())
        assert done['run_id']==config['run_id'] and done['model_revision']==REVISION
        for path in done['batches']:
            path=Path(path);batch=json.loads(path.read_text())
            input_path=base/'inputs'/('shard-%02d'%shard)/(batch['batch']+'.json')
            source=json.loads(input_path.read_text())
            assert fingerprints(input_path)['sha256']==batch['input_sha256']
            inputs={r['qid']:r for r in source['rows']}
            prefix=batch['prefix'];full=batch['full']
            assert prefix['actor_version']==full['actor_version']=='official7b-'+REVISION
            labels={}
            item=full['own_em'];assert fingerprints(item['path'])['sha256']==item['sha256']
            for line in Path(item['path']).open():
                row=json.loads(line);assert row['trajectory_id'] not in labels and row['strict_em'] in (0,1)
                labels[row['trajectory_id']]=row['strict_em']
            own={}
            item=prefix['prefix'];assert fingerprints(item['path'])['sha256']==item['sha256']
            with gzip.open(item['path'],'rt') as handle:
                for line in handle:
                    row=json.loads(line);tid=row['trajectory_id'];qid=row['qid']
                    assert tid not in seen and tid in labels and qid in inputs
                    assert row['question_hash']==inputs[qid]['question_hash']
                    assert row['split']==inputs[qid]['split']
                    assert row['initial_prompt_token_ids']==inputs[qid]['initial_prompt_token_ids']
                    assert row['retained_policy_length']<=20 and row['feature_layer']==26
                    seen.add(tid);own.setdefault(qid,[]).append((row['response_index'],tid,labels[tid]))
            assert set(own)==set(inputs) and set(labels)=={v[1] for group in own.values() for v in group}
            for qid, rows in own.items():
                rows.sort();assert [v[0] for v in rows]==list(range(5)) and qid not in groups
                source=inputs[qid]
                groups[qid]=dict(qid=qid,split=source['split'],source_ordinal=source['source_ordinal'],
                    question_hash=source['question_hash'],trajectory_ids=[v[1] for v in rows],labels=[v[2] for v in rows])
            item=prefix['actor_hidden'];assert fingerprints(item['path'])['sha256']==item['sha256']
            features.append(item['path'])
    train=sorted((g for g in groups.values() if g['split']=='train'),key=lambda g:g['source_ordinal'])
    diagnostic=sorted((g for g in groups.values() if g['split']=='diagnostic'),key=lambda g:g['source_ordinal'])
    assert len(train)==169028 and len(diagnostic)==512 and len(seen)==847700
    assert hashlib.sha256(json.dumps([r['qid'] for r in train]).encode()).hexdigest()==expected['ordered_train_qid_sha256']
    root=base/'bank';root.mkdir(exist_ok=True)
    atomic(root/'groups.json',dict(train=train,diagnostic=diagnostic))
    atomic(root/'features.json',features)
    atomic(root/'complete.json',dict(status='FULL_CANONICAL_7B_G5_NATIVE_FEATURES_AND_OWN_EM_VERIFIED',
        train_questions=169028,train_trajectories=845140,diagnostic_questions=512,diagnostic_trajectories=2560,
        features=fingerprints(root/'features.json'),groups=fingerprints(root/'groups.json'),
        model_revision=REVISION,feature_layer=26,capture_prefix_tokens=20,fit_prefix_tokens=5,
        run_id=config['run_id'],actor_updates=0,epoch=time.time()))
    print(json.dumps(dict(status='BANK_ACCEPTED',questions=len(groups),trajectories=len(seen))))


if __name__=='__main__':main()
