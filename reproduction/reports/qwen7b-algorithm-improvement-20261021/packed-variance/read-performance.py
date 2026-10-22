import pathlib,json,time,statistics
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-M3to10-P20-variance-8192-20261022-v1');curves=[json.loads(s)for s in(b/'curve.jsonl').read_text().splitlines()if s.strip()];out=dict(time=time.time(),epochs=[],only_rank0_logged=True,global_questions_per_update=128,loss_not_used_for_success=True)
for e in sorted({z['epoch']for z in curves}):
 rows=[z for z in curves if z['epoch']==e];intervals=[z['time']-p['time']for p,z in zip(rows,rows[1:])if z['cursor']==p['cursor']+1];a=dict(epoch=e,updates=len(rows),first_cursor=rows[0]['cursor'],last_cursor=rows[-1]['cursor'],rank0_mean_prefill_seconds=statistics.mean(z['prefill_seconds']for z in rows),actor_rows_equal_group_rows=all(z['actor_rows']==z['question_groups']and z['prefill_rows_equal_groups']for z in rows))
 if intervals:a.update(actual_mean_seconds_per_global_update=statistics.mean(intervals),actual_median_seconds_per_global_update=statistics.median(intervals),global_questions_per_second=128/statistics.mean(intervals),intervals=len(intervals),excludes_checkpoint_and_epoch_evaluation=True)
 out['epochs'].append(a)
print(json.dumps(out))
