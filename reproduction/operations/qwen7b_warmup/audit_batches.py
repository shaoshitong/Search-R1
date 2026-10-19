"""Read-only acceptance of sealed real batches; never imports a GPU model."""
import argparse
import gzip
import json
from pathlib import Path
import time
from reproduction.operations.qwen7b_warmup.download_model import atomic,fingerprints
from reproduction.operations.qwen7b_warmup.packed_features import FeatureReader,QMAX


def read(p):return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    base=Path(args.base);c=read(base/'config.json')
    model=read(Path(c['model_dir'])/'complete.json')
    assert model['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' and model['fresh_all_files_SHA256']
    paths=sorted((base/'sampling').glob('shard-*/batches/*/complete.json'));assert len(paths)>=2
    proofs=[];manifests=[];samples=[];histogram=[0]*6;feature_bytes=0;total=0
    for path in paths[:2]:
        batch=read(path);shard=path.parents[2].name
        input_path=base/'inputs'/shard/(batch['batch']+'.json')
        assert fingerprints(input_path)['sha256']==batch['input_sha256'] and batch['run_id']==c['run_id']
        inputs={r['qid']:r for r in read(input_path)['rows']};p=batch['prefix'];f=batch['full']
        assert p['actor_version']==f['actor_version']=='official7b-'+c['model_revision']
        for item in (p['prefix'],p['states'],p['actor_hidden'],f['full'],f['own_em']):
            assert fingerprints(item['path'])['sha256']==item['sha256']
        manifest=p['actor_hidden']['path'];proof=read(manifest);manifests.append(manifest)
        assert proof['prefix_tokens']==20 and proof['feature_layer']==26
        for item in proof['files'].values():
            actual=fingerprints(item['path']);assert actual['sha256']==item['sha256'] and actual['bytes']==item['bytes']
            feature_bytes+=actual['bytes']
        labels={}
        for line in Path(f['own_em']['path']).open():
            row=json.loads(line);assert row['trajectory_id'] not in labels and row['strict_em'] in (0,1)
            labels[row['trajectory_id']]=row['strict_em']
        groups={};rows=[]
        with gzip.open(p['prefix']['path'],'rt') as handle:
            for line in handle:
                row=json.loads(line);qid=row['qid'];assert row['split']==inputs[qid]['split']
                assert row['initial_prompt_token_ids']==inputs[qid]['initial_prompt_token_ids']
                assert row['feature_layer']==26 and row['retained_policy_length']<=20
                groups.setdefault(qid,[]).append(labels[row['trajectory_id']]);rows.append(row)
        assert set(groups)==set(inputs) and all(len(g)==5 for g in groups.values())
        for values in groups.values():histogram[sum(values)]+=1
        samples.extend(rows[::max(1,len(rows)//10)][:10]);total+=len(groups)
        proofs.append(dict(path=str(path),questions=len(groups),trajectories=len(rows),actor_updates=0))
    reader=FeatureReader(manifests)
    try:
        for row in samples:
            data=reader.get([row['trajectory_id']],'cpu');qlen=len(row['initial_prompt_token_ids']);plen=row['retained_policy_length']
            assert int(data['attention_mask'][0,:QMAX].sum())==qlen
            assert int(data['attention_mask'][0,QMAX:].sum())==plen
            assert int(data['attention_mask'][0,QMAX:QMAX+5].sum())==min(5,plen)
            assert data['position_ids'][0,QMAX:QMAX+plen].tolist()==row['policy_position_ids']
    finally:reader.close()
    result=dict(status='REAL_SEALED_G5_BATCHES_FILES_AND_FEATURE_READ_VERIFIED',batches=proofs,
        questions=total,trajectories=5*total,own_EM_K_histogram=histogram,
        mixed_fraction=sum(histogram[1:5])/total,feature_layer=26,head_layer=27,
        fit_prefix=5,capture_prefix=20,real_rows_independently_reconstructed=len(samples),
        feature_bytes=feature_bytes,projected_full_bank_feature_bytes=feature_bytes/total*169540,
        projection_is_sample_based=True,actor_updates=0,epoch=time.time())
    atomic(base/'preparation/real-batch-acceptance.json',result);print(json.dumps(result))


if __name__=='__main__':main()
