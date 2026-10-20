"""Remote CPU diagnostic: direct Mixed logistic heads on frozen L26 Prefix20.

Finite precommitted2x3 grid; grouped inner80/20; winner weights/threshold freeze
BEFORE any viewed2048 evaluation inputs/labels/predictions are opened. No GPU,
actor/block update, production substitution, final pool or package installation.
"""
import argparse,fcntl,gc,gzip,hashlib,json,math,mmap,os,sys,threading,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from . import g10_grouping as grouping
from . import replay_probe_layers as replay
from .replay_feature_reader import ReplayFeatureReader
from .peer_g10_learnability import checked,read,ref,digest,causal_signature
from .evaluate_frozen_validation import fixed_groups

STATUS='REMOTE_CPU_L26_PREFIX20_DIRECT_MIXED_LINEAR_HEAD_DIAGNOSTIC_COMPLETE'
RSS_LIMIT=3*1024**3
GRID=(.001,.01,.1)
METHODS=('prefix_mean','prefix_last')
LIMIT_SECONDS=26*60


def split_question(qid,seed=1729):
    return int(digest(['peer8192-direct-head-inner80',seed,str(qid)]),16)%5==0


def sigmoid(z):
    z=np.asarray(z,dtype=np.float64);return np.exp(-np.logaddexp(0.,-z))


def metric(scores,K,qids,threshold,top_k=None):
    scores=np.asarray(scores,dtype=np.float64);K=np.asarray(K);y=(K>0)&(K<5);p=scores>=threshold
    if len(scores)!=len(K) or len(qids)!=len(K) or not np.isfinite(scores).all() or not (y.any() and (~y).any()):raise ValueError('Finite labeled two-class own G5 required')
    tp=int((p&y).sum());tn=int((~p&~y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());pos=int(y.sum());neg=len(y)-pos
    order=np.argsort(scores,kind='stable');a=scores[order];yy=y[order];rank=0.;i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[j]==a[i]:j+=1
        rank+=float(yy[i:j].sum())*(i+1+j)/2;i=j
    result=dict(questions_or_equal_weight_views=len(y),K_histogram=np.bincount(K,minlength=6).tolist(),Mixed=int(pos),nonMixed=int(neg),
        balanced_accuracy=.5*(tp/pos+tn/neg),Mixed_recall=tp/pos,nonMixed_recall=tn/neg,nonMixed_as_Mixed_FPR=fp/neg,Mixed_as_nonMixed_FNR=fn/pos,
        allwrong_FPR=float(p[K==0].mean()) if (K==0).any() else None,allcorrect_FPR=float(p[K==5].mean()) if (K==5).any() else None,
        AUC=(rank-pos*(pos+1)/2)/(pos*neg),confusion=dict(TP=tp,FN=fn,TN=tn,FP=fp),decision_score_threshold=float(threshold))
    if top_k:
        ids=sorted(range(len(y)),key=lambda i:(-float(scores[i]),str(qids[i])))[:top_k];h=np.bincount(K[ids],minlength=6)
        result['Top%d'%top_k]=dict(questions=len(ids),K_histogram=h.tolist(),Mixed=int(h[1:5].sum()),Mixed_fraction=float(h[1:5].sum()/len(ids)),allwrong=int(h[0]),allcorrect=int(h[5]),selected_qids_sha256=digest([qids[i] for i in ids]),labels_used_for_selection=False)
    return result


def choose_threshold(scores,K):
    """Internal-only maximum BA; tied scores move together. Deterministic ties."""
    scores=np.asarray(scores,dtype=np.float64);K=np.asarray(K);y=(K>0)&(K<5);pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:raise ValueError('Both classes required for internal threshold selection')
    order=np.argsort(-scores,kind='stable');a=scores[order];yy=y[order];tp=fp=0;best=(.5,-abs(float(np.nextafter(a[0],np.inf))),float(np.nextafter(a[0],np.inf)))
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[j]==a[i]:j+=1
        tp+=int(yy[i:j].sum());fp+=(j-i)-int(yy[i:j].sum());t=float(a[i]);candidate=(.5*(tp/pos+1-fp/neg),-abs(t),t)
        if candidate>best:best=candidate
        i=j
    return best[2]


def pool_policy(payload,length,dimension):
    if len(payload)!=length*dimension*2:raise ValueError('Truncated own policy span')
    if not length:return dict(prefix_mean=np.zeros(dimension,dtype=np.float32),prefix_last=np.zeros(dimension,dtype=np.float32))
    bits=np.frombuffer(payload,dtype='<u2').astype('<u4');bits<<=16;values=bits.view('<f4').reshape(length,dimension)
    if not np.isfinite(values).all():raise ValueError('Nonfinite actual Prefix20 representation')
    return dict(prefix_mean=values.mean(axis=0,dtype=np.float32),prefix_last=values[-1].copy())


def group_vector(trajectories,indices,lengths):
    x=trajectories[np.asarray(indices)];lens=lengths[np.asarray(indices)].astype(np.float32)/20
    return np.concatenate((x.mean(axis=0),x.var(axis=0),np.asarray([lens.mean(),lens.var(),(lens==0).mean()],dtype=np.float32)))


def fit_logistic(X,y,l2,progress,max_iter=80):
    import torch
    torch.manual_seed(1729);torch.use_deterministic_algorithms(True)
    x=torch.from_numpy(np.ascontiguousarray(X,dtype=np.float32));target=torch.from_numpy(np.asarray(y,dtype=np.float32))
    w=torch.zeros(x.shape[1],requires_grad=True);bias=torch.tensor(float(np.log((float(target.sum())+.5)/(len(target)-float(target.sum())+.5))),requires_grad=True)
    optimizer=torch.optim.LBFGS([w,bias],lr=1.,max_iter=max_iter,history_size=5,tolerance_grad=1e-6,tolerance_change=1e-8,line_search_fn='strong_wolfe')
    calls=0;first=None
    def closure():
        nonlocal calls,first
        optimizer.zero_grad();logits=x.mv(w)+bias;bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target);loss=bce+.5*l2*w.square().sum();loss.backward();calls+=1
        if first is None:first=float(loss.detach())
        if calls%5==1:progress(calls,float(loss.detach()))
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        logits=x.mv(w)+bias;bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,target));loss=bce+.5*l2*float(w.square().sum())
    return w.detach().numpy().copy(),float(bias.detach()),dict(closure_calls=calls,iterations=int(optimizer.state[w].get('n_iter',0)),maximum_iterations=max_iter,initial_regularized_loss=first,final_regularized_loss=loss,unregularized_training_BCE=bce,gradient_max=max(float(w.grad.abs().max()),float(bias.grad.abs())))


def read_train(config):
    train=read(checked(config['training_input']));plan=read(checked(config['training_plan']));identity=read(checked(train['identity_plan']));raw=read(checked(train['raw_complete']))
    if train['status']!='PEER8192_AUTHENTIC_RAW_G10_AND_FOUR_HF_FEATURE_SHARDS_ACCEPTED' or train['questions']!=8192 or train['trajectories']!=81920 or raw['strict_EM_mismatches']!=0 or raw['final_labels_or_rollouts_read'] is not False or train['actor_updates']!=0:raise ValueError('Actual accepted peer8192 own G10 inputs required')
    by={g['qid']:g for g in read(checked(raw['outputs']['groups']))['train']};groups=[];qhash=set();qids=set();tids=set()
    for g in identity['groups']:
        a=by[g['qid']]
        if (a['question_hash']!=g['question_hash'] or a['trajectory_ids']!=g['trajectory_ids'] or len(a['trajectory_ids'])!=10 or len(a['labels'])!=10 or len(set(a['trajectory_ids']))!=10 or any(type(y) is not int or y not in (0,1) for y in a['labels']) or g['qid'] in qids or a['question_hash'] in qhash or tids.intersection(a['trajectory_ids'])):raise ValueError('Training own EM/question/trajectory binding differs')
        groups.append(a);qhash.add(a['question_hash']);qids.add(g['qid']);tids.update(a['trajectory_ids'])
    if len(groups)!=8192 or digest([g['qid'] for g in groups])!=train['ordered_qid_sha256'] or plan['cohort_qids']!=[g['qid'] for g in groups] or len(plan['batches'])!=256 or any(len(b)!=32 for b in plan['batches']) or set(plan['batch_epochs'])!={0} or {q for b in plan['batches'] for q in b}!=qids:raise ValueError('Actual B32x256 epoch0 cohort/plan differs')
    prefixes=[];manifests=[]
    for item in train['feature_shard_completes']:
        done=read(checked(item));selected=read(checked(done['source_selection']))
        if done['source_runtime']!='huggingface_frozen_causal_prefill' or done['model_revision']!=replay.REVISION or done['actor_updates']!=0 or done['batching']['batch_size']!=32 or done['batching']['padded_token_budget']!=16384:raise ValueError('Actual accepted B32 frozen-HF feature source required')
        prefixes+=selected['prefix_files'];manifests+=done['feature_manifests_by_capture_layer']['26']
    if manifests!=train['feature_manifests']:raise ValueError('Feature manifests differ from actual accepted train entry')
    lookup={g['qid']:g for g in groups};views=[]
    for batch in plan['batches']:
        v=grouping.build_training_views([lookup[q] for q in batch],mode='random',views_per_question=4,seed=1729,epoch=0,on_unbalance='raise',role='train')['groups']
        for row in v:
            if (row['weight_numerator'],row['weight_denominator'])!=(1,4):raise ValueError('Four equal view weights required')
        views+=v
    if len(views)!=32768 or any(n!=4 for n in Counter(v['qid'] for v in views).values()):raise ValueError('Actual four unique G5 views per question required')
    return groups,views,manifests,prefixes,dict(training_input=config['training_input'],training_plan=config['training_plan'],identity_plan=train['identity_plan'],raw_complete=train['raw_complete'],raw_groups=raw['outputs']['groups'],fit_questions=8192,views=32768,view_weight='1/4',per_original_question_total_weight='1',question_ordered_sha256=train['ordered_qid_sha256'])


def validate_prefixes(prefixes,reader,groups):
    wanted={t:g['qid'] for g in groups for t in g['trajectory_ids']};seen=set();counts=Counter()
    if set(reader.records)!=set(wanted):raise ValueError('Actual indexed feature coverage differs')
    for item in prefixes:
        checked(item)
        with gzip.open(item['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line);tid=row['trajectory_id']
                if tid not in wanted or tid in seen or row['qid']!=wanted[tid] or any(k in row for k in ('strict_em','labels','ground_truth','response_token_ids')):raise ValueError('Wrong/unsealed own causal prefix or forbidden label payload')
                causal_signature(row);own=reader.records[tid][1]
                if (own['source_prefix_sha256']!=row['prefix_sha256'] or own['policy_length']!=len(row['policy_token_ids']) or own['policy_positions']!=row['policy_position_ids'] or own['prompt_sha256']!=digest(row['initial_prompt_token_ids']) or own['policy_token_sha256']!=digest(row['policy_token_ids'])):raise ValueError('Own token/prompt/positions/source SHA differ from feature index')
                seen.add(tid);counts['short_prefixes']+=own['policy_length']<20;counts['zero_prefixes']+=own['policy_length']==0
    if seen!=set(wanted):raise ValueError('Missing own causal input binding')
    return dict(trajectories=len(seen),prefix_sources=prefixes,own_source_digest_tokens_positions_and_causal_signature_checked=True,**counts)


def reduce_groups(groups,views,manifests,prefixes,out,methods,progress):
    # Constructor verifies every full manifest/payload SHA, including policy.bin,
    # before any reduction. Read only own policy spans thereafter; Q273 is not
    # reconstructed. The previous captured-bank acceptance remains independent.
    reader=ReplayFeatureReader(manifests,26);proof=reader.proof;binding=validate_prefixes(prefixes,reader,groups)
    qrow=defaultdict(list)
    for i,v in enumerate(views):qrow[v['qid']].append((i,v))
    size=2*reader.dimension+3;Xs={m:np.lib.format.open_memmap(out/(m+'.npy'),mode='w+',dtype=np.float32,shape=(len(views),size)) for m in methods}
    streams={}
    for ordinal,g in enumerate(groups):
        reduced={m:[] for m in methods};lengths=[]
        for tid in g['trajectory_ids']:
            root,row=reader.records[tid];n=row['policy_length'];lengths.append(n)
            if root not in streams:
                if len(streams)>=16:streams.pop(next(iter(streams))).close()
                streams[root]=open(Path(root)/'policy.bin','rb')
            b=reader.span(streams[root],row['policy_offset'],n*reader.dimension*2);pooled=pool_policy(b,n,reader.dimension)
            for m in methods:reduced[m].append(pooled[m])
        arrays={m:np.stack(reduced[m]) for m in methods};lens=np.asarray(lengths)
        for index,v in qrow[g['qid']]:
            ix=[g['trajectory_ids'].index(t) for t in v['trajectory_ids']]
            for m in methods:Xs[m][index]=group_vector(arrays[m],ix,lens)
        if ordinal%128==0:
            for x in Xs.values():evict(x)
            progress('REDUCING_ACTUAL_POLICY_SPANS',questions=ordinal+1,expected_questions=len(groups))
    for x in Xs.values():evict(x)
    for stream in streams.values():stream.close()
    reader.close();return Xs,dict(feature_payload_and_manifest_SHA_fresh=True,policy_only_spans_read_for_pooling=True,question_latent_reconstruction=False,row_retained_feature_SHA_reconstructed_in_this_diagnostic=False,prefix_binding=binding,reader=proof,dimension=size)


def external_groups(config,train_groups):
    done=read(checked(config['validation_complete']))
    if (done['status']!='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE' or done['source_complete']['sha256']!='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5' or done['fixed_G5_sha256']!='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7' or done['questions']!=2048 or done['actor_updates']!=0 or done['new_G5_selected'] is not False or done['batching']['batch_size']!=32 or done['batching']['padded_token_budget']!=16384):raise ValueError('Exact fresh fixedG5 same-HF validation required')
    accepted=read(checked(done['source_complete']));groups=read(checked(done['source_groups']))['validation'];fixed=read(checked(done['source_fixed_G5']));own=fixed_groups(groups,fixed);trainqid={g['qid'] for g in train_groups};trainhash={g['question_hash'] for g in train_groups}
    if len(own)!=2048 or trainqid.intersection(g['qid'] for g in own) or trainhash.intersection(g['question_hash'] for g in groups) or accepted['fitting_on_validation_allowed'] is not False:raise ValueError('Training/validation question identity leakage')
    selection=read(checked(done['source_selection']));manifests=read(checked(done['features']))
    if manifests!=done['feature_manifests_by_capture_layer']['26']:raise ValueError('Fixed-G5 feature descriptor list changed')
    return own,manifests,selection['prefix_files'],done


def evict(array):
    array.flush();array._mmap.madvise(mmap.MADV_DONTNEED)


def take_rows(array,mask):
    indices=np.flatnonzero(mask);result=np.empty((len(indices),array.shape[1]),dtype=np.float32)
    for start in range(0,len(indices),512):
        result[start:start+512]=array[indices[start:start+512]];evict(array)
    return result


def standardizer(X):
    total=np.zeros(X.shape[1],dtype=np.float64);square=total.copy()
    for start in range(0,len(X),512):
        chunk=X[start:start+512].astype(np.float64);total+=chunk.sum(axis=0);square+=np.einsum('ij,ij->j',chunk,chunk)
    mean=total/len(X);scale=np.sqrt(np.maximum(0,square/len(X)-mean**2));scale[scale<1e-5]=1.
    return mean.astype(np.float32),scale.astype(np.float32)


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    raise RuntimeError('Actual Linux RSS required')


def run(config,out):
    login=config.get('execution_mode')=='bounded_login_cpu2'
    if sys.platform=='darwin' or os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','NoDevFiles','-1') or int(os.environ.get('SLURM_GPUS_ON_NODE','0'))!=0 or (not login and not os.environ.get('SLURM_JOB_ID')):raise ValueError('Explicit authorized remote CPU-only mode required')
    if login:
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2])

    out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():raise ValueError('Completed diagnostic preserved')
        if (out/'progress.json').exists():raise ValueError('Partial diagnostic preserved; use new isolated published attempt')
        import torch
        torch.set_num_threads(2 if login else min(12,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))));torch.set_num_interop_threads(1)
        watchdog_stop=threading.Event();peak_rss=[0]
        def watch():
            while not watchdog_stop.wait(.25):
                current=rss_bytes();peak_rss[0]=max(peak_rss[0],current)
                if current>RSS_LIMIT or time.monotonic()-start>LIMIT_SECONDS:
                    replay.atomic(out/'failure.json',dict(error_type='OwnCPUResourceCeiling',RSS_bytes=current,RSS_limit=RSS_LIMIT,elapsed_seconds=time.monotonic()-start,time=time.time()));os._exit(70)
        threading.Thread(target=watch,daemon=True).start()
        def progress(stage,**values):
            if time.monotonic()-start>LIMIT_SECONDS:raise TimeoutError('Precommitted26-minute diagnostic compute bound')
            if rss_bytes()>RSS_LIMIT:raise MemoryError('Own-process3GiB RSS ceiling')
            replay.atomic(out/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-start,time=time.time(),**values))
        progress('TRAIN_INPUT_ACCEPTANCE');groups,views,manifests,prefixes,binding=read_train(config);(out/'train').mkdir();Xs,proof=reduce_groups(groups,views,manifests,prefixes,out/'train',METHODS,progress);replay.atomic(out/'train/input-audit.json',proof)
        inner=np.asarray([split_question(v['qid']) for v in views]);K=np.asarray([sum(v['labels']) for v in views],dtype=np.int64);y=((K>0)&(K<5)).astype(np.float32);qids=[v['qid'] for v in views]
        if not inner.any() or inner.all() or len({v['qid'] for i,v in enumerate(views) if inner[i]}&{v['qid'] for i,v in enumerate(views) if not inner[i]}):raise ValueError('Whole-question internal isolation failed')
        replay.atomic(out/'internal-split.json',dict(seed=1729,split='sha256_namespace_plus_qid_mod5_equals0_is_internal_validation',train_questions=len({q for q,z in zip(qids,inner) if not z}),validation_questions=len({q for q,z in zip(qids,inner) if z}),train_ordered_qid_sha256=digest([g['qid'] for g in groups if not split_question(g['qid'])]),validation_ordered_qid_sha256=digest([g['qid'] for g in groups if split_question(g['qid'])]),labels_used_to_split=False,all_views_of_qid_same_split=True))
        results=[];best=None
        for method in METHODS:
            progress('STANDARDIZE_TRAIN_PART_ONLY',pooling=method);X=take_rows(Xs[method],~inner);mean,scale=standardizer(X);X-=mean;X/=scale;Xv=take_rows(Xs[method],inner);Xv-=mean;Xv/=scale
            for l2 in GRID:
                def fit_progress(calls,loss):progress('CPU_LOGISTIC_FIT',pooling=method,L2=l2,closure_calls=calls,regularized_loss=loss)
                w,b,fit=fit_logistic(X,y[~inner],l2,fit_progress);s=Xv@w+b;threshold=choose_threshold(s,K[inner]);metrics=metric(s,K[inner],[q for q,z in zip(qids,inner) if z],threshold);item=dict(pooling=method,L2=l2,internal=metrics,fit=fit);results.append(item);replay.atomic(out/'internal-grid-results.json',results)
                key=(metrics['balanced_accuracy'],metrics['AUC'],int(method=='prefix_mean'),l2)
                if best is None or key>best[0]:best=(key,dict(pooling=method,L2=l2,threshold_logit=float(threshold),internal=metrics,fit=fit),w.copy(),b,mean.copy(),scale.copy())
            del X,Xv
        _,winner,w,b,mean,scale=best;np.savez(out/'winner-weights.npz',weight=w,bias=np.asarray(b),mean=mean,scale=scale)
        frozen=dict(status='INTERNAL_ONLY_HYPERPARAMETERS_AND_WEIGHTS_FROZEN_BEFORE_EXTERNAL_OPEN',winner=winner,weights=ref(out/'winner-weights.npz'),train_binding=binding,internal_split=ref(out/'internal-split.json'),grid_results=ref(out/'internal-grid-results.json'),methods=list(METHODS),L2_grid=list(GRID),external_inputs_opened=False,fit_uses_inner80_only_no_postselection_refit=True,time=time.time());replay.atomic(out/'frozen-before-external.json',frozen);freeze=ref(out/'frozen-before-external.json')
        # Only the single frozen winner is permitted to open the viewed external
        # pool. No setting is revised based on this one evaluation.
        progress('FROZEN_WINNER_EXTERNAL_ACCEPTANCE');egs,em,ep,ed=external_groups(config,groups);ev=[dict(qid=g['qid'],trajectory_ids=g['trajectory_ids']) for g in egs];(out/'external').mkdir();E,ea=reduce_groups(egs,ev,em,ep,out/'external',(winner['pooling'],),progress);replay.atomic(out/'external/input-audit.json',ea);Xe=np.asarray(E[winner['pooling']]);Xe=(Xe-mean)/scale;scores=Xe@w+b;EK=np.asarray([sum(g['labels']) for g in egs]);eq=[g['qid'] for g in egs];external=metric(scores,EK,eq,winner['threshold_logit'],512)
        predictions=out/'external/predictions.jsonl.gz'
        with gzip.open(predictions,'wt') as f:
            for i,g in enumerate(egs):f.write(json.dumps(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],strict_em=g['labels'],predicted_Mixed_logit=float(scores[i]),predicted_Mixed_probability=float(sigmoid(scores[i]))))+'\n')
        # Existing E4 is read only AFTER the winner is frozen, with own labels
        # and exact fixed IDs joined. Never tune its threshold on this pool.
        baseline=read(checked(config['E4_baseline_complete']));rows=[]
        with gzip.open(checked(config['E4_predictions']),'rt') as f:
            for line in f:rows.append(json.loads(line))
        by={r['qid']:r for r in rows}
        if len(by)!=2048 or set(by)!=set(eq):raise ValueError('Existing E4 fixed-pool question set differs')
        qs=[]
        for g in egs:
            r=by[g['qid']]
            if r['trajectory_ids']!=g['trajectory_ids'] or r['strict_em']!=g['labels']:raise ValueError('Existing E4 own labels/fixedIDs differ')
            qs.append(r['predicted_mixed'])
        e4=metric(np.asarray(qs),EK,eq,.5,512);checked(config['E4_baseline_complete']);checked(config['E4_predictions'])
        if ref(out/'frozen-before-external.json')!=freeze:raise ValueError('Frozen winner changed after external evaluation')
        progress('FINAL_SOURCE_ACCEPTANCE');checked(config['training_input']);checked(config['training_plan']);checked(config['validation_complete']);replay.atomic(out/'complete.json',dict(status=STATUS,source_config=ref(config['_path']),train_binding=binding,train_feature_and_prefix_audit=ref(out/'train/input-audit.json'),external_feature_and_prefix_audit=ref(out/'external/input-audit.json'),frozen_before_external=freeze,winner=winner,internal_grid_results=ref(out/'internal-grid-results.json'),external=external,same_HF_E4_baseline=e4,E4_source=config['E4_baseline_complete'],E4_predictions=config['E4_predictions'],external_predictions=ref(predictions),fixed_G5_sha256=ed['fixed_G5_sha256'],new_G5_selected=False,external_scored_models=1,external_used_for_hyperparameter_selection=False,external_is_viewed_recipe_validation_not_untouched_final=True,final_pool_read=False,diagnostic512_read=False,actor_updates=0,existing_probe_or_Adam_updates=0,production_replacement_authorized=False,CPU_threads=torch.get_num_threads(),execution_mode=config.get('execution_mode','CPU0GPU_Slurm'),peak_observed_RSS_bytes=peak_rss[0],RSS_ceiling_bytes=RSS_LIMIT,GPU_requested=0,job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,time=time.time()))
        watchdog_stop.set();return read(out/'complete.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();config=read(a.config);config['_path']=str(Path(a.config))
    try:result=run(config,a.output)
    except BaseException as e:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True);replay.atomic(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)[:1000],time=time.time()));raise
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
