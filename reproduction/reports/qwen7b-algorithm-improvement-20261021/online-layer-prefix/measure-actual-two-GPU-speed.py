import pathlib,json,statistics,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out=[]
for layer,prefix in ((27,20),(21,20),(27,80)):
 p=b/('layer%d-prefix%d'%(layer,prefix))/'curve.jsonl'
 if not p.exists():continue
 rows=[json.loads(x)for x in p.read_text().splitlines()if x];intervals=[y['time']-x['time']for x,y in zip(rows,rows[1:])if y['epoch']==x['epoch']and y['cursor']==x['cursor']+1]
 out.append(dict(layer=layer,prefix=prefix,observed_updates=len(rows),last_cursor=rows[-1]['cursor'],global_batch_questions=128,world_size=2,gradient_accumulation=8,same_epoch_intervals=len(intervals),seconds_per128_questions_median=statistics.median(intervals),seconds_per128_questions_mean=sum(intervals)/len(intervals),questions_per_minute_same_epoch=128*60*len(intervals)/sum(intervals),first_to_last_update_seconds=rows[-1]['time']-rows[0]['time'],end_to_end_interval_includes_inter_epoch_save_and_evaluation=True))
print(json.dumps(dict(time=time.time(),variants=out)))
