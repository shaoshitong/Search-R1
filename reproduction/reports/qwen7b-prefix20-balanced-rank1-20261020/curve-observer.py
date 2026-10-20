import pathlib,json,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020');rows=[]
with (b/'probe/curve.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r['epoch']==2:rows.append(r)
assert rows and len({r['batch'] for r in rows})==len(rows)
for r in rows:
 assert r['steps']==10464+r['batch'] and r['actor_updates']==0
 assert all(math.isfinite(r[k]) for k in ['trajectory_bce','group_bce','pairwise_loss','total_loss'])
 assert abs(r['total_loss']-(r['trajectory_bce']+r['group_bce']+r['pairwise_loss']))<1e-5
for a,c in zip(rows,rows[1:]):assert c['batch']==a['batch']+1
def summary(part):
 return dict(first_batch=part[0]['batch'],last_batch=part[-1]['batch'],updates=len(part),means={k:sum(r[k] for r in part)/len(part) for k in ['trajectory_bce','group_bce','pairwise_loss','pairwise_accuracy','mean_p_range','mean_rank_step_seconds']})
print(json.dumps(dict(time=time.time(),epoch=2,first_batch=rows[0]['batch'],last_batch=rows[-1]['batch'],accepted_steps=rows[-1]['steps'],curve_contiguous=True,finite_losses=True,rank_weight_one_total_confirmed=True,first50=summary(rows[:50]),last50=summary(rows[-50:]),warning='Different shuffled minibatches, not sealed fixed-EMA evaluation; not a generalization result')))
