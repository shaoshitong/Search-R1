"""Read-only source-bound domain decomposition of already accepted predictions.

No fitting, new model score, per-domain reselection, tensors, or final access.
The ONLY selection is each model's existing global(-q,qid) Top512.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
OUT=A/'existing-predictions-domain-cursor512-v1'
EXPECTED={'nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle'}


def proof(path):
    p=Path(path); data=p.read_bytes()
    return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def checked(ref):
    p=Path(ref['path'])
    if not p.is_absolute() or B not in p.parents or '..' in p.parts:
        raise ValueError('Only explicitly bound remote experiment assets allowed')
    actual=proof(p)
    if actual['sha256']!=ref['sha256'] or ('bytes' in ref and actual['bytes']!=ref['bytes']):
        raise ValueError('Bound source SHA/bytes differ: '+str(p))
    return p


def read(path):return json.loads(Path(path).read_text())
def pin(path,sha):return checked(dict(path=str(path),sha256=sha))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rows(path):
    with gzip.open(str(path),'rt') as f:return [json.loads(line) for line in f if line.strip()]

def qscore(ps):
    pos=neg=1.
    for p in ps:
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Finite probabilities required')
        pos*=p;neg*=1-p
    return 1-pos-neg


def auc(rs):
    pairs=sorted((r['score'],0<sum(r['strict_em'])<5) for r in rs)
    npos=sum(y for _,y in pairs);nneg=len(pairs)-npos
    if not npos or not nneg:return None
    total=0.;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        total+=(i+1+j)/2*sum(y for _,y in pairs[i:j]);i=j
    return (total-npos*(npos+1)/2)/(npos*nneg)


def metrics(rs):
    ks=[sum(r['strict_em']) for r in rs];pred=[r['score']>=.5 for r in rs]
    tp=sum(p and 0<k<5 for p,k in zip(pred,ks));fn=sum(not p and 0<k<5 for p,k in zip(pred,ks))
    fp=sum(p and k in (0,5) for p,k in zip(pred,ks));tn=sum(not p and k in (0,5) for p,k in zip(pred,ks))
    tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
    def fpr(k):
        return sum(p for p,v in zip(pred,ks) if v==k)/ks.count(k) if ks.count(k) else None
    return dict(questions=len(rs),mixed_questions=tp+fn,mixed_fraction=(tp+fn)/len(rs) if rs else None,
        K_histogram=[ks.count(k) for k in range(6)],confusion=dict(tp=tp,fn=fn,tn=tn,fp=fp),
        BA=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,TPR=tpr,TNR=tnr,
        allwrong_FPR=fpr(0),allcorrect_FPR=fpr(5),AUC=auc(rs),threshold=.5)


def domain_summary(rs,domains):
    if len(rs)!=len({r['qid'] for r in rs}) or set(r['qid'] for r in rs)!=set(domains):
        raise ValueError('All source-bound question domains required exactly once')
    # Select ONCE globally, never independently within a domain.
    selected=sorted(rs,key=lambda r:(-r['score'],str(r['qid'])))[:512]
    names=sorted(EXPECTED|set(domains.values()));result={}
    for name in names:
        own=[r for r in rs if domains[r['qid']]==name]
        top=[r for r in selected if domains[r['qid']]==name]
        result[name]=dict(pool=metrics(own),original_global_Top512_members=metrics(top),
            covered=bool(own),per_domain_reselection=False)
    return dict(pool=metrics(rs),global_Top512=metrics(selected),domains=result,
        global_Top512_ordered_qids_sha256=digest([r['qid'] for r in selected]),
        per_domain_reselection=False,absent_domain_metric=None)


def own_predictions(rs,truth):
    if len(rs)!=len(truth) or len({r['qid'] for r in rs})!=len(truth):raise ValueError('Prediction membership differs')
    output=[];allids=set();maxerr=0.
    for r in rs:
        g=truth[r['qid']]
        if (r['trajectory_ids']!=g['trajectory_ids'] or r['strict_em']!=g['labels'] or len(r['p_j'])!=5
                or len(set(r['trajectory_ids']))!=5 or allids.intersection(r['trajectory_ids'])):
            raise ValueError('Own original fixed G5 IDs/strict EM differ')
        if any(type(y) is not int or y not in (0,1) for y in g['labels']):raise ValueError('True binary EM required')
        allids.update(r['trajectory_ids']);q=r['predicted_mixed'];err=abs(q-qscore(r['p_j']));maxerr=max(maxerr,err)
        if not math.isfinite(q) or not 0<=q<=1 or err>2e-6:raise ValueError('Finite G5 q formula differs')
        output.append(dict(qid=r['qid'],strict_em=r['strict_em'],score=q))
    return output,maxerr


def join_domains(groups,metadata):
    by={r['qid']:r for r in metadata};domains={}
    if len(by)!=len(metadata):raise ValueError('Domain metadata qids repeat')
    for g in groups:
        m=by[g['qid']]
        if m['question_hash']!=g['question_hash']:raise ValueError('Canonical question hash differs')
        domain=m.get('data_source')
        if not isinstance(domain,str) or not domain:raise ValueError('Actual source domain field missing')
        domains[g['qid']]=domain.lower()
    return domains


def canonical_membership(internal,fit_qids,held_qids,external,source_config,source_manifest):
    """Project ONLY identity inputs; no targets/rewards/EM columns are read."""
    import pyarrow.parquet as pq
    import pyarrow as pa
    pa.set_cpu_count(2)
    c=read(source_config);manifest=read(source_manifest);path=Path(c['canonical_train'])
    expected=c['input_sha256']['train']
    assert manifest['source_sha256']['train']==expected and manifest['all_canonical_rows_retained'] is True
    st=path.stat();stamp=(st.st_size,st.st_mtime_ns)
    parquet=pq.ParquetFile(str(path));assert parquet.metadata.num_rows==169028
    cols=set(parquet.schema_arrow.names)
    # Canonical producer identity joins prompt message contents and uses
    # extra_info.index, falling back to a normalized prompt hash only if absent.
    assert {'prompt','data_source','extra_info'}<=cols
    import unicodedata
    qids=[];hashes=set();by={};domain=Counter()
    for batch in parquet.iter_batches(batch_size=2048,columns=['prompt','data_source','extra_info.index']):
        for row in batch.to_pylist():
            prompt='\n'.join(m['content'] for m in row['prompt']);qh=hashlib.sha256(prompt.encode()).hexdigest()
            extra=row.get('extra_info',{});source_id=extra.get('index') if isinstance(extra,dict) else None
            if source_id is None:source_id=hashlib.sha256(' '.join(unicodedata.normalize('NFKC',prompt).split()).encode()).hexdigest()
            qid=json.dumps([str(row['data_source']),str(source_id)])
            if qid in by:raise ValueError('Canonical warmup qids repeat')
            by[qid]=qh;hashes.add(qh);qids.append(qid);domain[str(row['data_source'])]+=1
    assert len(by)==169028 and hashlib.sha256(json.dumps(qids).encode()).hexdigest()==manifest['ordered_train_qid_sha256']
    assert stamp==(path.stat().st_size,path.stat().st_mtime_ns)
    def overlap(groups):
        out=dict(questions=len(groups),qid_overlap=sum(g['qid'] in by for g in groups),
            question_hash_overlap=sum(g['question_hash'] in hashes for g in groups),
            matching_qid_and_hash=sum(by.get(g['qid'])==g['question_hash'] for g in groups))
        return out
    own={g['qid']:g for g in internal}
    result=dict(canonical_questions=169028,source_config=proof(source_config),source_input_manifest=proof(source_manifest),
        canonical_parquet=dict(path=str(path),bytes=stamp[0],whole_file_sha256_inherited_from_accepted_producer=expected,
            this_audit_fresh_full_file_SHA=False,source_stat_pre_post_identical=True),
        columns_read=['prompt','data_source','extra_info.index'],gold_reward_EM_columns_read=False,
        producer_identity_rule='qid=json([str(data_source),str(extra_info.index)]); hash=sha256(newline-joined prompt contents)',
        ordered_warmup_qid_sha256=manifest['ordered_train_qid_sha256'],domain_counts=dict(domain),
        internal8192=overlap(internal),internal_fit6144=overlap([own[q] for q in fit_qids]),
        internal_held2048=overlap([own[q] for q in held_qids]),external_fixedHF2048=overlap(external))
    assert result['internal8192']['matching_qid_and_hash']==8192
    assert result['internal_fit6144']['matching_qid_and_hash']==6144 and result['internal_held2048']['matching_qid_and_hash']==2048
    assert result['external_fixedHF2048']['qid_overlap']==result['external_fixedHF2048']['question_hash_overlap']==0
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT/'complete.json');a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(100,100))
    start=time.time();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise ValueError('Preserve existing terminal decomposition')
    pins={
        'paired_manifest':pin(A/'peer-expanded-append-stability-1a0f5b4/cursor-000512-v2/manifest.json','aff8cd750ab734159cc95e6f0a2e064e6ca8d9f81394898ed0cd9f2f879a6015'),
        'append_audit':pin(A/'peer-expanded-append-only-train-v2/evaluations/cursor-000512/independent-audit.json','fef550807527ff089e9a305d159bc82862e0738d738f89302052f3ccb72affeb'),
        'append_config':pin(A/'peer-expanded-append-only-train-v2/config.json','d56668bdd2e9839abe042c0d4f720225cb0cafd62f22924867dbfda883b17a80'),
        'external_questions':pin(B/'selection-2048-20261020/inputs/questions.json','9c26795ae1fd4095b3d3af46f2bbefb162897e40dd4ded8e8f94bf41eb0ecc0a'),
        'internal_audit':pin(A/'prefix-confidence-incremental-v1/fit/independent-internal-audit.json','f77c10b3868ad895c799574b06cd47191e4968fad8cf90f0d141860249b6cdd6'),
        'internal_config':pin(A/'prefix-confidence-incremental-v1/fit-config-v2.json','3ec15b790d5b0db67b88fdc22e74ea692eb216e34e0955853f4c273a30392371'),
        'data_config':pin(A/'prefix-confidence-incremental-v1/config.json','457fd8deba5242eece98239b6bae8812b6c96e2141c6454fe99242c8a593d827')}
    audit=read(pins['append_audit']);assert audit['cursor']==512 and audit['actual_new_batch_grouping_all_checked']
    ia=read(pins['internal_audit']);assert ia['external_opening_permitted'] is False
    c=read(pins['append_config']);e=next(e for e in c['evaluations'] if e['name']=='selection2048FreshG5HF')
    bank=read(checked(e['bank']))[e['split']];fixed=read(checked(e['fixed_g5']));fixed=fixed.get('fixed_G5',fixed)
    assert fixed['summary']['labels_consulted'] is False
    by={g['qid']:g for g in bank};truth={}
    for g in fixed['groups']:
        source=by[g['qid']];ys=dict(zip(source['trajectory_ids'],source['labels']))
        truth[g['qid']]=dict(qid=g['qid'],question_hash=source['question_hash'],trajectory_ids=g['trajectory_ids'],labels=[ys[t] for t in g['trajectory_ids']])
    assert len(truth)==2048
    external_domains=join_domains(list(truth.values()),read(pins['external_questions'])['rows'])
    manifest=read(pins['paired_manifest']);models={};predrefs={};qerrors={}
    for name,ref in manifest['predictions'].items():
        actual=checked(ref);rs,err=own_predictions(rows(actual),truth);key='append512' if name=='candidate' else name
        models[key]=domain_summary(rs,external_domains);predrefs[key]=proof(actual);qerrors[key]=err
    a.output.with_name('progress.json').write_text(json.dumps(dict(stage='EXTERNAL_EXISTING_THREE_MODELS_OWN_IDS_Q_DOMAIN_COMPLETE',models=list(models),questions=2048,time=time.time()))+'\n')
    cfg=read(pins['internal_config']);dc=read(pins['data_config']);groups=read(checked(cfg['groups']));split=read(checked(cfg['split']))
    ordered=sorted((g['qid'] for g in groups),key=lambda q:hashlib.sha256(('prefix-confidence-qid-9137:'+q).encode()).hexdigest())
    assert split['held_qids']==ordered[:2048] and split['fit_qids']==ordered[2048:]
    ids=read(checked(dc['identity_plan']));assert [g['qid'] for g in ids['groups']]==[g['qid'] for g in groups]
    events=read(checked(dc['source_events']));event=next(e for e in events['events'] if e['event']=='newseed20261021');want={g['qid']:g for g in groups};metadata=[];inputrefs=[]
    for batch in event['batches']:
        path=checked(batch['input']);inputrefs.append(proof(path))
        for r in read(path)['rows']:
            if r['qid'] in want:metadata.append(dict(qid=r['qid'],question_hash=r['question_hash'],data_source=r['data_source']))
    domains=join_domains(groups,metadata);assert Counter(domains.values())==ids['domain_scope']['question_counts']
    source=rows(checked(cfg['E4_predictions']));own={g['qid']:g for g in groups};internal=[];held=set(split['held_qids'])
    assert len(source)==8192 and len({r['qid'] for r in source})==8192
    for r in source:
        g=own[r['qid']]
        assert r['trajectory_ids']==g['trajectory_ids'] and r['strict_em']==g['labels'] and len(r['p_j'])==10
        if r['qid'] in held:internal.append(dict(qid=r['qid'],strict_em=g['labels'][5:],score=qscore(r['p_j'][5:])))
    helddomains={q:domains[q] for q in held};assert len(internal)==2048
    membership=canonical_membership(groups,split['fit_qids'],split['held_qids'],list(truth.values()),B/'config.json',B/'inputs/manifest.json')
    result=dict(status='EXISTING_PREDICTIONS_DOMAIN_ID_OWN_EM_Q_DECOMPOSITION_COMPLETE',
        role='descriptive_existing_predictions_only',warmup_membership=membership,source_files={k:proof(v) for k,v in pins.items()},
        external_fixed_G5_bank=e['bank'],external_fixed_G5_plan=e['fixed_g5'],external_predictions=predrefs,
        external_domain_metadata_counts=dict(Counter(external_domains.values())),external_models=models,q_formula_max_errors=qerrors,
        internal_confidence=dict(scope='heldout_newseed5_same_native_warmup_seen_questions_NQ_training_cohort',
            source_predictions=cfg['E4_predictions'],groups=cfg['groups'],split=cfg['split'],
            domain_metadata_input_refs_count=len(inputrefs),domain_metadata_input_refs_sha256=digest(inputrefs),
            all8192_domain_counts=dict(Counter(domains.values())),held2048_domain_counts=dict(Counter(helddomains.values())),
            E4=domain_summary(internal,helddomains)),
        NQ_to_other_domain_shift_explanation_supported=bool(set(external_domains.values())-set(helddomains.values())),
        no_per_domain_top512_reselection=True,parameters_fitted=False,threshold_fitted=False,
        new_model_scores_created=False,final_pool_read=False,GPU_used=False,model_tensor_read=False,
        real_scalar_calibrator_state_read=False,elapsed_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,time=time.time())
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');temp.replace(a.output)
    print(json.dumps(dict(receipt=proof(a.output),result=result)),flush=True)

if __name__=='__main__':main()
