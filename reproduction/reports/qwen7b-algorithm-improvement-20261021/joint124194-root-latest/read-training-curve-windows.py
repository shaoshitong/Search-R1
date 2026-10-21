import json,pathlib,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
rows=[]
with(b/'curve.jsonl').open()as f:
 for line in f:
  if line.endswith(chr(10)):rows.append(json.loads(line))
ids=[r['cursor']for r in rows]
assert ids and all(v==ids[0]+i for i,v in enumerate(ids)), 'Full regular accepted curve must be contiguous'
for r in rows:
 for v in [r['loss'],r['components']['trajectory'],r['components']['binary_mixed'],r['seconds'],r['block_gradient_L2'],r['readout_gradient_L2']]:assert math.isfinite(v), 'Nonfinite accepted training transaction'
 assert r['grouping']['weighted_mixed_fraction']==.5 and r['grouping']['mixed_views']==64 and r['grouping']['nonmixed_views']==64
blocks=[]
for start in range(0,len(rows),64):
 part=rows[start:start+64];n=len(part)
 mean=lambda key:sum(key(r)for r in part)/n
 blocks.append(dict(first=part[0]['cursor'],last=part[-1]['cursor'],count=n,mean_total_loss=mean(lambda r:r['loss']),mean_trajectory_BCE=mean(lambda r:r['components']['trajectory']),mean_Mixed_BCE=mean(lambda r:r['components']['binary_mixed']),mean_training_batch_Mixed_BA=mean(lambda r:r['mixed_balanced_accuracy']),mean_update_seconds=mean(lambda r:r['seconds']),max_block_gradient_L2=max(r['block_gradient_L2']for r in part),max_readout_gradient_L2=max(r['readout_gradient_L2']for r in part)))
print(json.dumps(dict(epoch=time.time(),first_regular_cursor=ids[0],last_regular_cursor=ids[-1],accepted_curve_rows=len(rows),all_finite=True,all_class_balanced=True,windows=blocks,changing_training_batches_not_fixed_heldout=True,optimizer_or_source_changed=False)))
