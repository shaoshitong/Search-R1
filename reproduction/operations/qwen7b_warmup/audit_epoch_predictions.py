"""Read sealed EMA predictions only; never load weights or modify training."""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def histogram(rows,truth):
    result=[0]*6
    for r in rows:result[sum(truth[r['qid']]['labels'])]+=1
    return result


def accept(base,epoch):
    b=Path(base);config=read(b/'config.json');receipt=read(b/'probe'/('epoch-%04d.json'%epoch))
    assert receipt['epoch']==epoch and receipt['completed_epoch'] and receipt['cursor']==0 and receipt['independent_cpu_reload']
    metric=receipt['metrics'];bank=read(b/'bank/groups.json');truth={r['qid']:r for r in bank['diagnostic']};assert len(truth)==512
    classes={name:dict(n=0,q_sum=0.,ce_sum=0.,rounded_wrong_saturation=0,predicted_mixed=0,all_five_p_identical=0,p_range_sum=0.) for name in ['all_wrong','mixed','all_correct']};seen=set();files=[]
    for rank in range(4):
        path=b/'probe/predictions'/('epoch-%04d-diagnostic-rank%d.jsonl.gz'%(epoch,rank));files.append(dict(path=str(path),sha256=sha(path)))
        with gzip.open(str(path),'rt') as f:
            for line in f:
                r=json.loads(line);qid=r['qid'];assert qid not in seen and qid in truth;seen.add(qid)
                assert r['epoch']==epoch and r['split']=='diagnostic' and r['weight_kind']=='EMA0.99'
                assert r['trajectory_ids']==truth[qid]['trajectory_ids'] and r['strict_em']==truth[qid]['labels']
                p=r['p_j'];q=r['predicted_mixed'];assert len(p)==5 and all(math.isfinite(x) and 0<=x<=1 for x in p) and math.isfinite(q) and 0<=q<=1
                ar=aw=1.
                for x in p:ar*=x;aw*=1-x
                assert abs(q-(1-ar-aw))<2e-6
                k=sum(r['strict_em']);name='all_wrong' if k==0 else 'all_correct' if k==5 else 'mixed';target=name=='mixed';c=classes[name]
                assert r['true_mixed']==target
                c['n']+=1;c['q_sum']+=q;c['predicted_mixed']+=q>=.5;c['all_five_p_identical']+=len(set(p))==1;c['p_range_sum']+=max(p)-min(p)
                if (q==0 and target) or (q==1 and not target):c['rounded_wrong_saturation']+=1
                else:c['ce_sum']+=(-math.log(q) if target else -math.log1p(-q))
    assert seen==set(truth)
    probability_ce=None if any(c['rounded_wrong_saturation'] for c in classes.values()) else sum(c['ce_sum'] for c in classes.values())/512
    if probability_ce is not None:assert abs(probability_ce-metric['diagnostic']['group_bce'])<1e-4,'Rounded-probability BCE disagrees with sealed stable-logit metric'
    for name,c in classes.items():
        n=c['n'];assert n
        classes[name]=dict(questions=n,mixed_bce_from_rounded_probability=None if c['rounded_wrong_saturation'] else c['ce_sum']/n,mean_predicted_q=c['q_sum']/n,predicted_mixed_fraction=c['predicted_mixed']/n,all_five_p_identical=c['all_five_p_identical'],mean_p_range=c['p_range_sum']/n,rounded_wrong_saturation=c['rounded_wrong_saturation'])
    selected_metric=metric['selection2048'];path=Path(selected_metric['predictions']['path']);assert sha(path)==selected_metric['predictions']['sha256'];pred=[]
    with gzip.open(str(path),'rt') as f:
        for line in f:pred.append(json.loads(line))
    held=Path(config['selection_base']);proof=read(held/'bank/complete.json');assert sha(held/'bank/groups.json')==proof['groups_sha256']
    groups=read(held/'bank/groups.json')['selection_validation'];truth={g['qid']:g for g in groups};assert len(pred)==len(truth)==2048 and len({r['qid'] for r in pred})==2048 and {r['qid'] for r in pred}==set(truth)
    for r in pred:
        assert set(r)=={'qid','trajectory_ids','p_j','predicted_mixed'} and r['trajectory_ids']==truth[r['qid']]['trajectory_ids']
        ar=aw=1.
        assert len(r['p_j'])==5
        for x in r['p_j']:assert math.isfinite(x) and 0<=x<=1;ar*=x;aw*=1-x
        assert abs(1-ar-aw-r['predicted_mixed'])<1e-12
    selected=sorted(pred,key=lambda r:(-r['predicted_mixed'],r['qid']))[:512]
    pool_hist=histogram(pred,truth);top_hist=histogram(selected,truth)
    assert pool_hist==selected_metric['pool']['K_histogram'] and top_hist==selected_metric['selected512']['K_histogram']
    assert sum(top_hist[1:5])==selected_metric['selected512']['true_mixed']
    return dict(status='SEALED_DIAGNOSTIC_AND_INDEPENDENT_SELECTION_RECOMPUTED',base=str(b),epoch=epoch,steps=receipt['steps'],checkpoint_sha256=receipt['sha256'],diagnostic_stable_logit_metrics=metric['diagnostic'],diagnostic_classwise=classes,diagnostic_rounded_probability_bce=probability_ce,diagnostic_prediction_files=files,selection_prediction_sha256=sha(path),pool_K_histogram=pool_hist,selected_K_histogram=top_hist,pool_mixed=sum(pool_hist[1:5]),selected_mixed=sum(top_hist[1:5]),selected_mixed_fraction=sum(top_hist[1:5])/512,ranked_without_truth=True,metrics_source='sealed epoch receipt, never mutable metrics-latest',training_updated=False,time=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--epoch',type=int,required=True);a=p.parse_args();print(json.dumps(accept(a.base,a.epoch)),flush=True)
