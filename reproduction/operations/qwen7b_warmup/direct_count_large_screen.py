"""Independent CPU-only large-head diagnostic screen; no fitting/model loads.

Reads only this diagnostic's training-pool own-EM metadata, saved q/K rows and
frozen E1 logits. It never opens any external or final prediction pool.
"""
import argparse,array,gzip,hashlib,json,math,sys
from pathlib import Path

EPOCHS=(0,1,2,4,8,20)
MODES=('raw','EMA')
SEED=9137
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
    p=Path(p);raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
    p=Path(ref['path']);actual=proof(p)
    if any(actual[k]!=ref[k]for k in ('path','bytes','sha256')):raise ValueError('Frozen CPU input SHA/bytes differ')
    return p
def immutable(p,v):
    p=Path(p);raw=canonical(v)+b'\n';p.parent.mkdir(exist_ok=True,parents=True)
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('Frozen screen metadata differs')
    else:p.write_bytes(raw)


def fixed_halves(qids):
    if len(qids)!=2560 or len(set(qids))!=2560:raise ValueError('Exactly2560 distinct NQ holdout qids required')
    ranked=sorted(qids,key=lambda q:(digest(['large-count-NQ-halves',SEED,q]),q))
    return [ranked[:1280],ranked[1280:]]


def auc(scores,labels):
    positive=sum(labels);negative=len(labels)-positive
    if not positive or not negative:return None
    ranked=sorted(zip(scores,labels),key=lambda x:x[0]);total=0.;i=0
    while i<len(ranked):
        j=i+1
        while j<len(ranked)and ranked[j][0]==ranked[i][0]:j+=1
        total+=sum(y for _,y in ranked[i:j])*(i+1+j)/2;i=j
    return (total-positive*(positive+1)/2)/(positive*negative)


def metrics(rows):
    if not rows:return dict(questions=0,Mixed_BA=None)
    if any(type(r['K'])is not int or not 0<=r['K']<=5 or not math.isfinite(r['q'])or not 0<=r['q']<=1 for r in rows):
        raise ValueError('Finite own-G5 q/K required')
    ys=[0<r['K']<5 for r in rows];pred=[r['q']>=.5 for r in rows];p=sum(ys);n=len(ys)-p
    tp=sum(y and v for y,v in zip(ys,pred));tn=sum(not y and not v for y,v in zip(ys,pred))
    tpr=tp/p if p else None;tnr=tn/n if n else None
    top=sorted(rows,key=lambda r:(-r['q'],r['qid']))[:len(rows)//4]
    def fpr(k):
        group=[r for r in rows if r['K']==k]
        return sum(r['q']>=.5 for r in group)/len(group)if group else None
    return dict(questions=len(rows),K_histogram=[sum(r['K']==k for r in rows)for k in range(6)],
        Mixed_BA=(tpr+tnr)/2 if p and n else None,Mixed_recall=tpr,nonMixed_recall=tnr,
        allwrong_FPR=fpr(0),allcorrect_FPR=fpr(5),Mixed_AUC=auc([r['q']for r in rows],ys),
        predicted_Mixed_fraction=sum(pred)/len(rows),
        Top25percent=dict(questions=len(top),Mixed_count=sum(0<r['K']<5 for r in top),
            Mixed_fraction=sum(0<r['K']<5 for r in top)/len(top)if top else None,
            K_histogram=[sum(r['K']==k for r in top)for k in range(6)]))


def qualifies(candidate,baseline):
    """Strict whole-NQ improvement; both fixed halves noninferior on both axes."""
    a,b=candidate['NQ'],baseline['NQ']
    gate=(a['Mixed_BA']is not None and b['Mixed_BA']is not None
        and a['Mixed_BA']>b['Mixed_BA']
        and a['Top25percent']['Mixed_count']>b['Top25percent']['Mixed_count'])
    for half in ('half0','half1'):
        a,b=candidate[half],baseline[half]
        gate=gate and a['Mixed_BA']is not None and b['Mixed_BA']is not None and a['Mixed_BA']>=b['Mixed_BA'] and a['Top25percent']['Mixed_count']>=b['Top25percent']['Mixed_count']
    return bool(gate)


def select_candidate(candidates,baseline):
    expected={(e,m)for e in EPOCHS for m in MODES}
    if {(r['epoch'],r['model'])for r in candidates}!=expected or len(candidates)!=12:
        raise ValueError('All12 predeclared raw/EMA nodes must finish before selecting one')
    eligible=[r for r in candidates if qualifies(r,baseline)]
    eligible.sort(key=lambda r:(-r['NQ']['Top25percent']['Mixed_count'],-r['NQ']['Mixed_BA'],r['epoch'],0 if r['model']=='EMA'else 1))
    return eligible[0]if eligible else None


def freeze(base,expected_config_sha):
    base=Path(base);c=read(base/'config.json')
    if proof(base/'config.json')['sha256']!=expected_config_sha:raise ValueError('Exact prepared config required')
    selection=read(checked(c['inputs']['internal_holdout_selection']))
    rows=selection['groups']
    if any('labels'in r or 'strict_em'in r for r in rows):raise ValueError('Only identity-only selection metadata may freeze halves')
    nq=[r['qid']for r in rows if r['domain']in ('nq','NQ')]
    halves=fixed_halves(nq)
    result=dict(role='LABEL_BLIND_INTERNAL_NQ_HALVES_BEFORE_CAPACITY_TRAINING',config=proof(base/'config.json'),
        selection=c['inputs']['internal_holdout_selection'],seed=SEED,
        rule='SHA256 canonical [large-count-NQ-halves,9137,qid], tie qid; first1280/last1280',
        halves=halves,half_SHA256=[digest(h)for h in halves],labels_read=False,scores_read=False,
        candidate_epochs=list(EPOCHS),candidate_models=list(MODES),threshold=.5,
        primary_NQ_gate='BA and Top640 strictly exceed same-cache frozen-E1 product-q; both fixed1280 halves BA and Top320 not lower',
        tie_order=['NQ Top640 descending','NQ BA descending','earlier epoch','EMA before raw'],
        external_requires_separate_loaded_candidate_same5_online_micro8_qualification=True,
        head_fit_micro320_not_claimed_online_micro8_bitexact=True,
        no_eligible_candidate_means_no_external=True)
    p=base/'screen-protocol/frozen-NQ-halves.json';immutable(p,result)
    return dict(status='LARGE_CAPACITY_SCREEN_AND_NQ_HALVES_FROZEN_NO_GPU_OR_LABELS',file=proof(p),half_SHA256=result['half_SHA256'],half_questions=[1280,1280],candidate_nodes=12)


def audit(base,expected_config_sha):
    base=Path(base);c=read(base/'config.json')
    if proof(base/'config.json')['sha256']!=expected_config_sha:raise ValueError('Exact prepared config required')
    frozen=read(base/'screen-protocol/frozen-NQ-halves.json');halves=frozen['halves'];sets=[set(h)for h in halves]
    if frozen['config']!=proof(base/'config.json')or frozen['selection']!=c['inputs']['internal_holdout_selection']:raise ValueError('Frozen halves source differs')
    cohort=read(checked(c['cohort']));qids=cohort['holdout_qids'];fit=cohort['fit_qids']
    fixed={g['qid']:g for g in read(checked(c['inputs']['internal_holdout_groups']))['internal_holdout']}
    groups={g['qid']:g for g in read(checked(c['inputs']['merged_groups']))['train']}
    cache=read(base/'cache/complete.json')
    if cache['trajectory_ids_sha256']!=digest([t for q in fit+qids for t in groups[q]['trajectory_ids']]):raise ValueError('Frozen logit trajectory order differs')
    if sys.byteorder!='little':raise ValueError('Declared f32 cache uses little-endian host')
    z=array.array('f');z.frombytes(checked(cache['files']['logits']).read_bytes())
    if len(z)!=(len(fit)+len(qids))*10:raise ValueError('Frozen logits count differs')
    baseline=[]
    def sigmoid(x):return 1/(1+math.exp(-x))if x>=0 else math.exp(x)/(1+math.exp(x))
    for i,q in enumerate(qids):
        own=groups[q]['trajectory_ids'][5:]
        if fixed[q]['trajectory_ids']!=own or fixed[q]['labels']!=groups[q]['labels'][5:]:raise ValueError('Own committed new5 labels/IDs differ')
        values=[float(z[(len(fit)+i)*10+j])for j in range(5,10)]
        if not all(math.isfinite(v)for v in values):raise ValueError('Nonfinite frozen E1 logit')
        ps=[sigmoid(v)for v in values]
        score=1-math.prod(ps)-math.prod(1-p for p in ps)
        baseline.append(dict(qid=q,q=min(1.,max(0.,score)),K=sum(fixed[q]['labels']),domain=fixed[q]['domain']))
    def split_metrics(rows):
        nq=[r for r in rows if r['qid']in sets[0]or r['qid']in sets[1]]
        return dict(NQ=metrics(nq),half0=metrics([r for r in rows if r['qid']in sets[0]]),half1=metrics([r for r in rows if r['qid']in sets[1]]),
            HotpotQA=metrics([r for r in rows if r['qid']not in sets[0]and r['qid']not in sets[1]]),pooled=metrics(rows))
    reference=split_metrics(baseline);candidates=[];sources=[]
    for epoch in EPOCHS:
        done=read(base/'evaluations'/('epoch-%02d'%epoch)/'complete.json');checked(done['checkpoint'])
        if not done['training_components_unchanged']:raise ValueError('Head evaluation changed state')
        for model in MODES:
            ref=done['predictions'][model+'-internal_holdout_new5'];rows=[]
            with gzip.open(checked(ref),'rt')as f:
                for line in f:rows.append(json.loads(line))
            if len(rows)!=5120 or {r['qid']for r in rows}!=set(qids)or len({r['qid']for r in rows})!=5120:raise ValueError('Exact holdout coverage required')
            if any(r['K']!=sum(fixed[r['qid']]['labels'])for r in rows):raise ValueError('Own new5 strict-EM count differs')
            value=dict(epoch=epoch,model=model,**split_metrics(rows))
            fit_ref=done['predictions'][model+'-fit_primaryG5']
            with gzip.open(checked(fit_ref),'rt')as f:fit_rows=[json.loads(line)for line in f]
            if len(fit_rows)!=30016 or {r['qid']for r in fit_rows}!=set(fit):raise ValueError('Fit-primary row coverage differs')
            value['fit_primaryG5_qK_statistics']=metrics(fit_rows)
            value['screen_pass']=qualifies(value,reference);candidates.append(value);sources.extend([ref,fit_ref])
    selected=select_candidate(candidates,reference)
    result=dict(status='LARGE_CAPACITY_INTERNAL_SCREEN_COMPLETE',config=proof(base/'config.json'),halves=proof(base/'screen-protocol/frozen-NQ-halves.json'),
        frozen_E1_baseline=reference,baseline_arithmetic='stable float64 sigmoid of exact cached float32 Z; product formula float64',
        candidate_nodes=candidates,selected_internal_candidate=selected,one_external_candidate_may_be_prepared=selected is not None,
        external_scoring_permitted=False,external_requires_actual_model_load_and_online_micro8_same5_qualification=True,
        head_micro320_online_micro8_bitexact_claim=False,prediction_sources=sources,
        final_pool_read=False,external_payload_opened=False,parameters_fitted=False,
        own_G5_labels_joined_from_frozen_refs=True,per_prediction_IDs_inherited_from_qualified_scoring_source=True)
    p=base/'screen-protocol/independent-screen-complete.json';immutable(p,result)
    return dict(status=result['status'],file=proof(p),selected_internal_candidate=selected,external_scoring_permitted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('freeze','audit'),required=True);p.add_argument('--base',required=True);p.add_argument('--config-sha256',required=True);args=p.parse_args()
    print(json.dumps((freeze if args.action=='freeze'else audit)(args.base,args.config_sha256)),flush=True)
if __name__=='__main__':main()
