import pathlib,json,hashlib
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-performance-20261022-v2/fullstep-timing.json')
raw=p.read_bytes();x=json.loads(raw)
assert x['status']=='INDEPENDENT_CP128_SAME_RANDOM_BATCH_FULLSTEP_PROFILE_COMPLETE' and x['final_state_restored_to_CP128'] is True and x['formal_training_update_count']==0
print(json.dumps(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
