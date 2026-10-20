"""Independent streaming acceptance of full natural train/diagnostic predictions.

Only reads sealed epoch files and writes small audit receipts. No GPU/model,
training data regeneration, scheduling, or active trainer changes.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.audit-tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def accept_split(base,epoch,split,groups,metrics):
    expected={row['qid']:row for row in groups};assert len(expected)==len(groups)
    assert len(groups)==(169028 if split=='train' else 512)
    seen=set();files=[];conf=[0,0,0,0];three=[[0]*3 for _ in range(3)]
    right=0;max_error=0.;predicted_sum=0.;k_hist=[0]*6
    def category(k):return 0 if k==0 else 2 if k==5 else 1
    for rank in range(4):
        path=base/'probe/predictions'/('epoch-%04d-%s-rank%d.jsonl.gz'%(epoch,split,rank))
        assert path.is_file(),'Missing sealed prediction file: '+str(path)
        rows=0
        with gzip.open(str(path),'rt') as stream:
            for line in stream:
                row=json.loads(line);qid=row['qid']
                assert qid in expected and qid not in seen;seen.add(qid);reference=expected[qid]
                assert row['epoch']==epoch and row['split']==split and row['weight_kind']=='EMA0.99'
                assert row['trajectory_ids']==reference['trajectory_ids']
                assert row['strict_em']==reference['labels'] and len(row['p_j'])==5
                probabilities=row['p_j'];assert all(math.isfinite(x) and 0<=x<=1 for x in probabilities)
                all_right=all_wrong=1.
                for p in probabilities:all_right*=p;all_wrong*=1-p
                mixed=row['predicted_mixed'];assert math.isfinite(mixed) and 0<=mixed<=1
                error=abs(mixed-(1-all_right-all_wrong));assert error<2e-6;max_error=max(max_error,error)
                count=sum(reference['labels']);k_hist[count]+=1;true=0<count<5
                assert row['true_mixed']==true;prediction=mixed>=.5;predicted_sum+=mixed
                conf[0 if prediction and true else 1 if prediction else 2 if true else 3]+=1
                predicted_count=sum(p>=.5 for p in probabilities)
                three[category(count)][category(predicted_count)]+=1
                right+=sum((p>=.5)==bool(y) for p,y in zip(probabilities,reference['labels']));rows+=1
        assert rows==len(groups[rank::4]),'Per-rank natural partition differs'
        files.append(dict(path=str(path),rows=rows,bytes=path.stat().st_size,sha256=sha256(path)))
    assert len(seen)==len(groups)
    tp,fp,fn,tn=conf;accuracy=right/(len(groups)*5);ba=.5*(tp/(tp+fn)+tn/(tn+fp))
    assert metrics['groups']==len(groups) and metrics['trajectories']==len(groups)*5
    assert metrics['weight_kind']=='EMA0.99' and metrics['resampled'] is False
    assert dict(zip(('tp','fp','fn','tn'),conf))==metrics['confusion']
    assert three==metrics['three_class_confusion']
    for actual,key in [(accuracy,'trajectory_accuracy'),(ba,'mixed_balanced_accuracy'),
                       ((tp+tn)/len(groups),'mixed_accuracy'),(predicted_sum/len(groups),'mean_predicted_mixed'),
                       (sum(k_hist[1:5])/len(groups),'true_mixed_fraction')]:
        assert math.isclose(actual,metrics[key],rel_tol=0,abs_tol=2e-7),(key,actual,metrics[key])
    for key in ('trajectory_bce','group_bce','total_loss'):assert math.isfinite(metrics[key]) and metrics[key]>=0
    assert math.isclose(metrics['total_loss'],metrics['trajectory_bce']+metrics['group_bce'],rel_tol=0,abs_tol=1e-10)
    return dict(groups=len(groups),trajectories=len(groups)*5,confusion=dict(zip(('tp','fp','fn','tn'),conf)),
        three_class_confusion=three,strict_EM_K_histogram=k_hist,trajectory_accuracy=accuracy,
        mixed_balanced_accuracy=ba,max_probability_formula_error=max_error,files=files)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--epoch',required=True,type=int);args=parser.parse_args();base=args.base;epoch=args.epoch
    assert 0<=epoch<=20
    bank=read(base/'bank/complete.json');assert bank['status']=='FULL_CANONICAL_7B_G5_NATIVE_FEATURES_AND_OWN_EM_VERIFIED'
    path=base/'bank/groups.json';assert sha256(path)==bank['groups']['sha256'];groups=read(path)
    if epoch==0:
        metric_path=base/'probe/epoch-0000-metrics.json';metrics=read(metric_path)
    else:
        metric_path=base/'probe'/('epoch-%04d.json'%epoch);receipt=read(metric_path)
        assert receipt['completed_epoch'] is True and receipt['epoch']==epoch and receipt['actor_updates']==0
        metrics=receipt['metrics']
    report=dict(status='FULL_NATURAL_G5_EPOCH_PREDICTIONS_AND_METRICS_VERIFIED',epoch=epoch,
        bank_sha256=sha256(base/'bank/complete.json'),metric_file=str(metric_path),metric_file_sha256=sha256(metric_path),
        splits={split:accept_split(base,epoch,split,groups[split],metrics[split]) for split in ('train','diagnostic')},
        actor_updates=0,weight_kind='EMA0.99',auditor_source_sha256=sha256(Path(__file__)),epoch_time=time.time())
    output=base/'prediction-audit';output.mkdir(exist_ok=True)
    atomic(output/('epoch-%04d.json'%epoch),report)
    print('RESULT_JSON='+json.dumps(report),flush=True)


if __name__=='__main__':main()
