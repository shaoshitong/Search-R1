import pathlib,json,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
p=b/'online-mixed-layer-prefix-20261022-2gpu-v1/source-publication.json';f=b/'online-mixed-layer-prefix-20261022-v1/fixture-complete.json'
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
x=json.loads(f.read_text());print(json.dumps(dict(source_publication=json.loads(p.read_text()),source_publication_ref=proof(p),fixture_ref=proof(f),fixture={k:x[k]for k in('status','training_questions','validation_questions','fixed_G5_sha256','fixed_train_plan_sha256','token_shards','plans')})))
