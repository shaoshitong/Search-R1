"""Frozen-arm metrics from authentic scoring logits; no optimizer or encoder."""
import json
from pathlib import Path


def scored_metrics(scored,own_em):
    from . import probe as p
    scored=Path(scored);m=p.read(scored/'manifest.json')
    with Path(own_em).open() as stream:
        rows=[json.loads(line)for line in stream]
    tids={r['trajectory_id']for r in rows}
    original={r['trajectory_id']:r for r in p.read(scored/'index.json')}
    if len(tids)!=len(rows) or not tids<=original.keys():raise ValueError('Only actual unique labeled scoring rows')
    index=p.join_own_em({t:original[t]for t in tids},rows,m['actor_version'],m['prefix_scan_sha256'],m['selection_sha256'])
    predictions=[]
    with (scored/'candidate-scores.jsonl').open() as stream:
        for line in stream:
            group=json.loads(line);ids=group['trajectory_ids']
            if not set(ids)&tids:continue
            if not set(ids)<=tids:raise ValueError('Do not split a genuine G5')
            predictions.extend((tid,float(z),index[tid]['strict_em'])for tid,z in zip(ids,group['trajectory_logits']))
    if len(predictions)!=len(index):raise ValueError('Missing authentic scoring logits')
    metrics,_=p.evaluate_predictions(predictions,index,len(index)//5,group_weight=1.)
    return metrics,index,m


def report(args):
    from . import probe as p
    from .ab_policy import FROZEN,validate_receipt
    config=p.read(Path(args.state_dir).parent/'preparation/launch-config.json')
    if config.get('selection_probe_experiment')!=FROZEN:raise ValueError('Frozen report only')
    head=p.read(Path(args.state_dir)/'latest.json');validate_receipt(config,head,args.outer)
    train,index,m=scored_metrics(args.scored_features,args.own_em)
    splits={'train':train}
    if args.frozen_test_score:
        test,tindex,tm=scored_metrics(args.frozen_test_score,args.test_own_em)
        if tm['actor_version']!=m['actor_version'] or tm['run_id']!=m['run_id']:
            raise ValueError('Paired diagnostic must use the same sampling actor')
        if ({r['qid']for r in index.values()}&{r['qid']for r in tindex.values()}
                or {r['question_hash']for r in index.values()}&{r['question_hash']for r in tindex.values()}):
            raise ValueError('Training/diagnostic question overlap')
        splits['test']=test
    metrics=dict(status='PROBE_FROZEN_NO_OPTIMIZER_UPDATE',run_id=config['run_id'],outer=args.outer,
        new_adam_steps_per_rank=0,actual_adam_steps_per_rank=[0]*4,last_update_outer=0,
        before_update=dict(frozen=splits),after_update=dict(frozen=splits),
        metrics_distribution='original_unresampled_current_outer_groups',
        frozen_weights_unchanged=True,scoring_logits_reused=True,no_encoder_or_optimizer=True,
        actor_version=m['actor_version'])
    p.once(Path(args.output),metrics)
    return dict(status=metrics['status'],output=args.output,train=train,test=splits.get('test'))
