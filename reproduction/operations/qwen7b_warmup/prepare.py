"""Remote-only canonical full-dataset partition and 7B tokenizer scan."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid
from .download_model import atomic, fingerprints, REVISION, MODEL

BASE = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--prefix', type=int, choices=(5,), default=5)
    args = parser.parse_args()
    base = Path(args.base)
    assert str(base) == BASE
    if (base/'config.json').exists():
        c = json.loads((base/'config.json').read_text())
        assert c['prefix_tokens'] == args.prefix and c['train_questions'] == 169028 and c['diagnostic_questions'] == 512
        print(json.dumps(dict(status='EXISTING_PLAN_NO_REPARTITION',run_id=c['run_id'])))
        return
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    from reproduction.prefix20_online.pool import identity
    from reproduction.prefix20_online.rollout import prompt_tokens
    previous = json.loads(Path('/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/preparation/prepared-config.json').read_text())
    model = base/'assets/qwen25-7b-instruct-a09a35458c70'
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    paths = {'train':Path(previous['canonical_train']),'diagnostic':Path(previous['validation_questions'])}
    rows, identities, hashes = {}, {}, {}
    qmax = 0
    for split, path in paths.items():
        data = pq.read_table(path).to_pylist()
        assert len(data) == (169028 if split == 'train' else 512)
        rows[split] = []
        for ordinal, row in enumerate(data):
            own = identity(row)
            assert own['qid'] not in identities, 'Canonical train/dev upstream identity overlap'
            identities[own['qid']] = split
            ids = prompt_tokens(tokenizer,row['prompt'])
            assert 0 < len(ids) <= 2048
            qmax = max(qmax,len(ids))
            rows[split].append(dict(row,**own,split=split,source_ordinal=ordinal,
                candidate_ordinal=ordinal+(169028 if split == 'diagnostic' else 0),
                initial_prompt_token_ids=ids))
        hashes[split] = fingerprints(path)['sha256']
    config = dict(base=str(base),run_id=uuid.uuid4().hex,model=MODEL,model_revision=REVISION,
        model_dir=str(model),warmup_only=True,actor_updates_authorized=False,actor_optimizer_updates=0,
        train_questions=169028,train_trajectories=845140,diagnostic_questions=512,diagnostic_trajectories=2560,
        sampling_shards=8,gpus_per_sampling_job=2,generation_engines_per_job=1,retriever_gpus_per_job=1,
        prefix_tokens=args.prefix,capture_prefix_tokens=20,question_max_tokens=qmax,feature_layer=26,native_head_layer=27,
        feature_dimension=3584,feature_depth_rule='human_explicit_layer26_features_plus_layer27_block',
        fit_epochs=20,fit_world=4,fit_lr=5e-6,fit_ema=.99,fit_group_weight=1.,
        fit_sampling='whole_G5_three_class_balance_all_majority_once_minority_with_replacement',
        fit_global_groups=51,fit_micro_groups=2,seed=1729,input_sha256=hashes,
        all_rollout_data_generated_once_before_fit=True,epoch_RNG_reset=False,
        warmup_source='thread01a1488e-807d-7ca3-b024-96b8fe97486c current native question+prefix recipe',
        common_manifest_sha256=previous['common_manifest_sha256'],
        source_commit=json.loads((base/'preparation/source-publication.json').read_text())['commit'],
        **{k:previous[k] for k in ('canonical_train','validation_questions','retrieval_index','retrieval_corpus','retrieval_encoder','retrieval_data_cache')})
    manifests=[]
    for shard in range(8):
        directory = base/'inputs'/('shard-%02d'%shard)
        directory.mkdir(parents=True,exist_ok=True)
        batch_ordinal = 0
        counts = {}
        for split in ('train','diagnostic'):
            own = rows[split][shard::8]
            counts[split] = len(own)
            for offset in range(0,len(own),128):
                path = directory/('batch-%06d.json'%batch_ordinal)
                batch=dict(shard=shard,split=split,rows=own[offset:offset+128],model_revision=REVISION)
                atomic(path,batch)
                manifests.append(dict(path=str(path),sha256=fingerprints(path)['sha256'],questions=len(batch['rows']),split=split,shard=shard))
                batch_ordinal += 1
        atomic(directory/'partition.json',dict(shard=shard,counts=counts,batches=batch_ordinal))
    assert sum(r['questions'] for r in manifests if r['split']=='train')==169028
    assert sum(r['questions'] for r in manifests if r['split']=='diagnostic')==512
    atomic(base/'inputs/manifest.json',dict(batches=manifests,source_sha256=hashes,
        ordered_train_qid_sha256=hashlib.sha256(json.dumps([r['qid'] for r in rows['train']]).encode()).hexdigest(),
        questions=169540,trajectories=847700,question_max_tokens=qmax,model_revision=REVISION,
        all_canonical_rows_retained=True,final_QA_used=False,epoch=time.time()))
    atomic(base/'config.json',config)
    print(json.dumps(dict(status='FULL_DATA_G5_EIGHT_SHARD_PLAN_READY',question_max_tokens=qmax,
        train_questions=169028,train_trajectories=845140,diagnostic_questions=512,batches=len(manifests),prefix=args.prefix)))


if __name__ == '__main__':
    main()
