"""Stdlib-only aggregate paired E0/current snapshot audit; no tensor loading."""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
BASE=Path(A)/'joint124194-current-snapshot-eval-v1';MAIN=Path(A)/'joint-balanced-native-E4-124194-v1'
E0_SHA='be5ff3311fa5bbfe23bf2052fd2645c695232acee2ddcd1936497f46d52ba5d7'

def proof(path):
    p=Path(path);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def checked(ref):
    actual=proof(ref['path']);assert actual['sha256']==ref['sha256'] and ('bytes' not in ref or actual['bytes']==ref['bytes']);return Path(ref['path'])

def rows(ref):
    with gzip.open(checked(ref),'rt') as f:data=[json.loads(x) for x in f]
    assert len(data)==2048 and len({r['qid'] for r in data})==2048
    for r in data:
        assert len(r['trajectory_ids'])==len(set(r['trajectory_ids']))==len(r['strict_em'])==len(r['p_j'])==5
        assert all(type(y)is int and y in(0,1) for y in r['strict_em']) and all(math.isfinite(x)and 0<=x<=1 for x in r['p_j'])
        q=1-math.prod(r['p_j'])-math.prod(1-x for x in r['p_j']);assert abs(q-r['predicted_mixed'])<3e-6
    return data

def auc(data):
    vals=sorted((r['predicted_mixed'],int(0<sum(r['strict_em'])<5))for r in data);n=sum(y for _,y in vals);neg=len(vals)-n
    if not n or not neg:return None
    rank=0.;i=0
    while i<len(vals):
        j=i+1
        while j<len(vals) and vals[j][0]==vals[i][0]:j+=1
        rank+=sum(y for _,y in vals[i:j])*((i+1+j)/2.);i=j
    return (rank-n*(n+1)/2)/(n*neg)

def stats(data):
    pool=[0]*6;selected=[0]*6;classes={name:dict(n=0,predicted_Mixed=0,trajectory_bce=0.,groupMixed_bce=0.)for name in('allwrong','Mixed','allcorrect')}
    for r in data:
        k=sum(r['strict_em']);pool[k]+=1;name='allwrong'if k==0 else 'allcorrect'if k==5 else'Mixed';v=classes[name];v['n']+=1;v['predicted_Mixed']+=int(r['predicted_mixed']>=.5)
        p=[min(1-1e-15,max(1e-15,x))for x in r['p_j']];q=min(1-1e-15,max(1e-15,r['predicted_mixed']));y=int(0<k<5)
        v['trajectory_bce']+=sum(-math.log(x)if yy else-math.log1p(-x)for x,yy in zip(p,r['strict_em']))/5
        v['groupMixed_bce']+=-math.log(q)if y else-math.log1p(-q)
    for r in sorted(data,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]:selected[sum(r['strict_em'])]+=1
    for name,v in classes.items():
        n=v['n'];v['predicted_Mixed_rate']=v['predicted_Mixed']/n if n else None
        v['nonMixed_prediction_count']=n-v['predicted_Mixed'];v['nonMixed_prediction_rate']=(n-v['predicted_Mixed'])/n if n else None
        for key in('trajectory_bce','groupMixed_bce'):v[key]=v[key]/n if n else None
        v['total_bce']=v['trajectory_bce']+v['groupMixed_bce']if n else None
    tp=classes['Mixed']['predicted_Mixed'];positive=classes['Mixed']['n'];negative=len(data)-positive;fp=classes['allwrong']['predicted_Mixed']+classes['allcorrect']['predicted_Mixed'];tpr=tp/positive;tnr=1-fp/negative
    return dict(questions=len(data),Mixed=sum(pool[1:5]),K_histogram=pool,selected512_K_histogram=selected,selected512_Mixed=sum(selected[1:5]),selected512_Mixed_fraction=sum(selected[1:5])/512,balanced_accuracy=(tpr+tnr)/2,Mixed_TPR=tpr,nonMixed_TNR=tnr,AUC=auc(data),classwise=classes)

def run():
    start=time.monotonic();c=json.loads((BASE/'complete.json').read_text());e0path=MAIN/'evaluations/epoch-00/complete.json';assert proof(e0path)['sha256']==E0_SHA;e0=json.loads(e0path.read_text())
    assert c['snapshot_cursor']==1408 and c['training_six_components_unchanged']and c['six_components_before']==c['six_components_after']and c['actor_updates']==c['probe_updates']==0 and c['final4096_scored']is False
    assert c['validation_proof']['fixed_G5_sha256']==e0['validation_proof']['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
    result={};identity=None
    for label in('raw','EMA'):
        em=e0['metrics'][label]['fixed_selection2048'];cm=c['metrics'][label]['fixed_selection2048'];before=rows(em['predictions']);after=rows(cm['predictions'])
        ids=lambda rs:[(r['qid'],r['trajectory_ids'],r['strict_em'])for r in rs]
        assert ids(before)==ids(after)
        if identity is None:identity=ids(before)
        else:assert identity==ids(before)
        bs=stats(before);cs=stats(after)
        for s,m in((bs,em),(cs,cm)):
            loss=m['diagnostic_losses'];s['native_logit_diagnostic_losses']={k:loss[k]for k in('trajectory_bce','group_bce')};s['native_logit_diagnostic_losses']['total_bce']=loss['trajectory_bce']+loss['group_bce']
            assert abs(s['balanced_accuracy']-m['pool']['mixed_balanced_accuracy'])<1e-12
            assert abs(s['AUC']-m['pool']['mixed_roc_auc'])<1e-12
            assert s['selected512_K_histogram']==m['selected512']['k_histogram']
        delta=dict(BA_pp=100*(cs['balanced_accuracy']-bs['balanced_accuracy']),Mixed_TPR_pp=100*(cs['Mixed_TPR']-bs['Mixed_TPR']),nonMixed_TNR_pp=100*(cs['nonMixed_TNR']-bs['nonMixed_TNR']),AUC=cs['AUC']-bs['AUC'],Top512_Mixed_count=cs['selected512_Mixed']-bs['selected512_Mixed'],Top512_Mixed_pp=100*(cs['selected512_Mixed_fraction']-bs['selected512_Mixed_fraction']))
        for name in('allwrong','allcorrect','Mixed'):delta[name+'_predicted_Mixed_pp']=100*(cs['classwise'][name]['predicted_Mixed_rate']-bs['classwise'][name]['predicted_Mixed_rate'])
        result[label]=dict(E0=bs,CP1408=cs,delta_from_E0=delta,E0_predictions=em['predictions'],CP1408_predictions=cm['predictions'])
    out=dict(status='CP1408_E0_SAME_FIXED2048_OWN_IDS_LABELS_Q_AND_AGGREGATES_INDEPENDENTLY_PASSED',snapshot_cursor=1408,complete=proof(BASE/'complete.json'),E0_complete=proof(e0path),results=result,threshold=.5,top_k=512,trajectory_and_binaryMixed_weights=[1,1],three_event_rank_weights=0,probability_class_losses_clipping_epsilon=1e-15,exact_native_total_uses_saved_logit_loss=True,final_labels_or_predictions_read=False,model_tensor_loaded=False,fit_or_parameter_update=False,elapsed_seconds=time.monotonic()-start,epoch=time.time())
    dest=BASE/'paired-E0-CP1408-independent-audit.json';raw=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if dest.exists():raise ValueError('Do not repeat accepted audit; fresh read its receipt instead')
    with dest.open('xb')as f:f.write(raw)
    print(json.dumps(dict(audit=proof(dest),**out)))

if __name__=='__main__':run()
