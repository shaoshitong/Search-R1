"""Independent CPU audit of sealed G5 predictions; no model/tensor loads.

Run on HPC3. Only aggregate confusion/selection metrics leave shared storage.
The final checkpoint's runner-side reload proof is reported, not re-certified.
"""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def checked(item):
    path=Path(item['path'])
    if sha(path)!=item['sha256']:
        raise ValueError('Immutable input SHA differs: '+str(path))
    return path


def metrics(rows):
    labels=[0<sum(r['strict_em'])<5 for r in rows]
    scores=[r['predicted_mixed'] for r in rows]
    called=[v>=.5 for v in scores]
    tp=sum(a and b for a,b in zip(called,labels))
    fn=sum(not a and b for a,b in zip(called,labels))
    fp=sum(a and not b for a,b in zip(called,labels))
    tn=sum(not a and not b for a,b in zip(called,labels))
    if not tp+fn or not tn+fp:
        raise ValueError('Both real Mixed and nonMixed classes required')
    ordered=sorted(zip(scores,labels));rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum+=(i+1+j)*.5*sum(y for _,y in ordered[i:j]);i=j
    pos,neg=tp+fn,tn+fp
    hist=[sum(sum(r['strict_em'])==k for r in rows) for k in range(6)]
    classes={}
    for name,lo,hi in [('allwrong',0,0),('mixed',1,4),('allcorrect',5,5)]:
        own=[r for r in rows if lo<=sum(r['strict_em'])<=hi]
        classes[name]=dict(questions=len(own),predicted_mixed=sum(r['predicted_mixed']>=.5 for r in own),
            mean_q=sum(r['predicted_mixed'] for r in own)/len(own) if own else None)
    return dict(questions=len(rows),threshold=.5,confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
        mixed_balanced_accuracy=.5*(tp/pos+tn/neg),mixed_recall=tp/pos,nonmixed_recall=tn/neg,
        mixed_fraction=pos/len(rows),mixed_roc_auc=(rank_sum-pos*(pos+1)/2)/(pos*neg),
        k_histogram=hist,classwise=classes)


def audit(base):
    base=Path(base).resolve()
    if not str(base).startswith('/data/user/'):
        raise ValueError('Prediction data stays on remote shared storage')
    path=base/'complete.json'
    if not path.exists():
        return dict(status='NOT_COMPLETE_NO_METRIC_ACCEPTANCE',base=str(base))
    result=read(path);cpath=base/'config.json';c=read(cpath)
    if sha(cpath)!=result['config_sha256'] or result['cursor']!=c['update_budget']:
        raise ValueError('Sealed completion config/cursor mismatch')
    proof=read(base/'next-update-replay.json')
    if proof['status']!='SINGLE_GPU_NEXT_UPDATE_REPLAY_PASSED' or proof['actual']!=proof['expected']:
        raise ValueError('Actual first-update replay did not pass')
    checkpoint=result['checkpoint']
    if checkpoint!=read(base/'latest.json') or checkpoint['cursor']!=result['cursor']:
        raise ValueError('Latest immutable checkpoint not paired with sealed completion')
    banks={
        'diagnostic512':read(checked(c['bank']))['diagnostic'],
        'selection2048':read(checked(c['selection_bank']))['selection_validation']}
    out={}
    for split,truth_rows in banks.items():
        expected=512 if split=='diagnostic512' else 2048
        truth={r['qid']:r for r in truth_rows}
        if len(truth)!=expected:raise ValueError('Fixed question set differs')
        sealed=result['metrics'][split];p=checked(sealed['predictions'])
        with gzip.open(str(p),'rt') as f:rows=[json.loads(line) for line in f]
        if len(rows)!=expected or len({r['qid'] for r in rows})!=expected or {r['qid'] for r in rows}!=set(truth):
            raise ValueError('Sealed predictions do not cover the fixed question set')
        formula_error=0.
        for row in rows:
            own=truth[row['qid']]
            if row['trajectory_ids']!=own['trajectory_ids'] or row['strict_em']!=own['labels']:
                raise ValueError('Prediction IDs/own strict EM disagree with source bank')
            p_j=row['p_j'];q=row['predicted_mixed']
            if len(p_j)!=5 or any(not math.isfinite(v) or not 0<=v<=1 for v in p_j+[q]):
                raise ValueError('Invalid probability vector')
            ar=aw=1.
            for value in p_j:ar*=value;aw*=1-value
            formula_error=max(formula_error,abs(1-ar-aw-q))
        if formula_error>2e-6:raise ValueError('Stored score is not the declared G5 formula')
        pool=metrics(rows)
        selected=metrics(sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512])
        for actual,reference in [(pool,sealed['pool']),(selected,sealed['selected512'])]:
            for key in ('confusion','k_histogram'):
                if actual[key]!=reference[key]:raise ValueError('Stored '+key+' differs')
            for key in ('mixed_balanced_accuracy','mixed_recall','nonmixed_recall','mixed_roc_auc','mixed_fraction'):
                if abs(actual[key]-reference[key])>1e-12:raise ValueError('Stored '+key+' differs')
        out[split]=dict(pool=pool,selected512=selected,maximum_probability_formula_error=formula_error,
            prediction_sha256=sealed['predictions']['sha256'])
    return dict(status='INDEPENDENT_SEALED_OWN_EM_SHA_G5_METRICS_RECOMPUTED',base=str(base),
        config_sha256=sha(cpath),cursor=result['cursor'],steps=result['steps'],checkpoint=checkpoint,
        checkpoint_tensor_reload_performed_here=False,first_update_actual_replay_passed=True,
        metrics=out,training_modified=False,time=time.time())



result=audit('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-loss128-v2/event')
result['audit_source_commit']='3c2c2f7c3a78b74060a07c8cf73f6172521b8516'
result['audit_source_sha256']='80d789bcfab5ec3e470cd68391c1c5550ff6c18acd14173ebcf55e64ac15e58a'
print(json.dumps(result))
