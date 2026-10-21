"""Existing accepted E0/CP1408 probabilities only; never forwards or fits."""
import gzip,hashlib,json,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
B=A/'joint124194-current-snapshot-eval-v1'

def proof(p):
    p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def calculate(rows):
    result={}
    for name,condition in [('allwrong',lambda k:k==0),('Mixed',lambda k:0<k<5),('allcorrect',lambda k:k==5)]:
        chosen=[r for r in rows if condition(sum(r['strict_em']))];n=len(chosen);ps=[x for r in chosen for x in r['p_j']]
        ranges=[max(r['p_j'])-min(r['p_j'])for r in chosen]
        gaps=[sum(p for p,y in zip(r['p_j'],r['strict_em'])if y)/sum(r['strict_em'])-sum(p for p,y in zip(r['p_j'],r['strict_em'])if not y)/(5-sum(r['strict_em']))for r in chosen]if name=='Mixed'else[]
        result[name]=dict(questions=n,trajectories=5*n,mean_p=sum(ps)/len(ps)if ps else None,mean_within_question_p_range=sum(ranges)/n if n else None,p_gt_point5_count=sum(p>.5 for p in ps),p_gt_point5_fraction=sum(p>.5 for p in ps)/len(ps)if ps else None,range_lt_1e_minus4_questions=sum(x<1e-4 for x in ranges),range_lt_1e_minus4_fraction=sum(x<1e-4 for x in ranges)/n if n else None,Mixed_group_mean_p_correct_minus_p_wrong=sum(gaps)/len(gaps)if gaps else None,mean_q=sum(r['predicted_mixed']for r in chosen)/n if n else None)
    return result

def run():
    started=time.monotonic();paired_path=B/'paired-E0-CP1408-independent-audit.json';assert proof(paired_path)['sha256']=='2038471b69022fc7613e7a59bcb4684b45039f8887fcd218779bef0b62dcf564'
    paired=json.loads(paired_path.read_text());results={};identity=None
    for label in('raw','EMA'):
        results[label]={}
        for node,key in [('E0','E0_predictions'),('CP1408','CP1408_predictions')]:
            ref=paired['results'][label][key];assert proof(ref['path'])==ref
            with gzip.open(ref['path'],'rt')as f:rows=[json.loads(x)for x in f]
            own=[(r['qid'],r['trajectory_ids'],r['strict_em'])for r in rows];assert len(rows)==2048
            if identity is None:identity=own
            else:assert own==identity
            results[label][node]=calculate(rows)
    out=dict(status='EXISTING_E0_CP1408_SAME_G5_PROBABILITY_CLASS_STATS_COMPLETE',source_paired_audit=proof(paired_path),results=results,p_threshold_strict_greater_than=.5,approximate_identical_range_strict_less_than=1e-4,Mixed_gap_weighting='mean over real Mixed questions of per-question mean p(correct) minus mean p(wrong)',GPU_forward=False,model_tensor_loaded=False,training_changed=False,external_or_final_new_predictions=False,elapsed_seconds=time.monotonic()-started,epoch=time.time())
    path=B/'E0-CP1408-probability-class-stats.json'
    with path.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(receipt=proof(path),**out)))

if __name__=='__main__':run()
