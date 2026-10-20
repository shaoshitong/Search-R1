import pathlib,json,time,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');assert hashlib.sha256((b/'config.json').read_bytes()).hexdigest()=='16b2fd34e3e8907307b3866298b33216478fd77c2dc1b9ca80638e0929800130';rows=[]
with (b/'curve.jsonl').open() as f:
 for line in f:
  if line.endswith('\n'):rows.append(json.loads(line))
assert [x['cursor'] for x in rows]==list(range(1,len(rows)+1));points=[]
for k in [1,32,64,128,192,256]:
 if k>len(rows):continue
 row=rows[k-1];components=row['components'];assert abs(row['loss']-(components['trajectory']+components['three_event']))<2e-6;window=rows[max(0,k-16):k];point={'update':k,'Adam_steps':row['steps'],'actual_batch_loss':row['loss'],'trajectory_BCE':components['trajectory'],'three_event_NLL':components['three_event'],'binary_Mixed_BCE_diagnostic_not_optimized':components['binary_mixed'],'batch_seconds':row['seconds'],'trailing_window_updates':len(window),'trailing_window_mean':{name:sum(x['components'][key] for x in window)/len(window) for name,key in [('trajectory_BCE','trajectory'),('three_event_NLL','three_event'),('binary_Mixed_BCE','binary_mixed')]}};points.append(point)
print(json.dumps({'status':'ACTUAL_NO_INTERRUPT_PEER8192_LEARNINGCURVE_SNAPSHOT','complete_updates_observed':len(rows),'points':points,'same_batch_across_points_claim':False,'loss_fall_is_generalization_success_claim':False,'actor_updates':0,'no_extra_evaluation':True,'observed_epoch':time.time()}))
