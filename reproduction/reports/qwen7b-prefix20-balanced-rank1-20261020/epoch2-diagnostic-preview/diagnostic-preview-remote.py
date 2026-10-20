import pathlib,json,gzip,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020');receipt=json.loads((b/'probe/latest.json').read_text());assert receipt['epoch']==2 and receipt['cursor']==1318 and receipt['steps']==11782
truth={g['qid']:g for g in json.loads((b/'bank/groups.json').read_text())['diagnostic']};seen=set();classes={k:dict(n=0,group_ce=0.,q_sum=0.,predicted_mixed=0,trajectory_ce=0.,trajectory_correct=0,saturated_wrong=0,trajectory_tp=0,trajectory_fp=0,trajectory_fn=0,trajectory_tn=0) for k in ['all_wrong','mixed','all_correct']}
for rank in range(4):
 with gzip.open(str(b/'probe/predictions'/('epoch-0002-diagnostic-rank%d.jsonl.gz'%rank)),'rt') as f:
  for line in f:
   r=json.loads(line);qid=r['qid'];assert qid in truth and qid not in seen;seen.add(qid);assert r['epoch']==2 and r['split']=='diagnostic' and r['weight_kind']=='EMA0.99';assert r['trajectory_ids']==truth[qid]['trajectory_ids'] and r['strict_em']==truth[qid]['labels']
   k=sum(r['strict_em']);name='all_wrong' if k==0 else 'all_correct' if k==5 else 'mixed';target=name=='mixed';q=r['predicted_mixed'];assert 0<q<1
   c=classes[name];c['n']+=1;c['group_ce']+=-math.log(q) if target else -math.log1p(-q);c['q_sum']+=q;c['predicted_mixed']+=q>=.5
   for p,y in zip(r['p_j'],r['strict_em']):
    assert math.isfinite(p) and 0<=p<=1
    correct=(p>=.5)==bool(y);c['trajectory_correct']+=correct
    c['trajectory_tp']+=p>=.5 and bool(y);c['trajectory_fp']+=p>=.5 and not bool(y);c['trajectory_fn']+=p<.5 and bool(y);c['trajectory_tn']+=p<.5 and not bool(y)
    if (y and p==0) or (not y and p==1):c['saturated_wrong']+=1
    else:c['trajectory_ce']+=-math.log(p) if y else -math.log1p(-p)
assert seen==set(truth) and len(seen)==512;total=sum(c['group_ce'] for c in classes.values());out={}
for name,c in classes.items():
 n=c['n'];out[name]=dict(questions=n,mixed_bce_rounded_probabilities=c['group_ce']/n,mean_predicted_mixed=c['q_sum']/n,predicted_mixed_fraction=c['predicted_mixed']/n,mixed_bce_contribution=c['group_ce']/512,mixed_bce_share=c['group_ce']/total,trajectory_bce_rounded_probabilities=None if c['saturated_wrong'] else c['trajectory_ce']/(5*n),trajectory_accuracy=c['trajectory_correct']/(5*n),trajectory_confusion={k:c['trajectory_'+k] for k in ['tp','fp','fn','tn']},saturated_wrong_trajectory_probabilities=c['saturated_wrong'])
print(json.dumps(dict(status='COMPLETE_DIAGNOSTIC_PREDICTIONS_PREVIEW_FULL_EPOCH_METRICS_PENDING',epoch=2,steps=11782,checkpoint_sha256=receipt['sha256'],weight_kind='EMA0.99',questions=512,group_bce_rounded_probabilities=total/512,classes=out,training_updated=False,final_epoch_metric_claimed=False,time=time.time())))
