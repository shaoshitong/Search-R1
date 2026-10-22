"""CPU-only final B128 online-raw curve vs frozen raw/EMA receipt summary.
No model/feature/rollout/checkpoint/prediction loads or scoring/updates.
"""
import hashlib,json,math,statistics,time
from pathlib import Path
BASE=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1')
CONFIG_SHA='e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837'
PRIOR=BASE.parent/'training-domain-role-priors-v1/complete.json'
PRIOR_SHA='e9b67c1dc80e8a8def53746a6fbfc78167330cc153cd9955eda2dd5f0571c8eb'
RANGES=((1676,2577),(2578,3898),(3899,5219),(5220,6540))

def read(path):
    raw=path.read_bytes();return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def dist(xs):
    x=sorted(xs);return dict(mean=statistics.mean(x),minimum=x[0],maximum=x[-1],median=statistics.median(x),p95=x[min(len(x)-1,math.ceil(.95*len(x))-1)])

def summary(rows):
    count={k:sum(int(r['confusion'][k])for r in rows)for k in('tp','tn','fp','fn')};tp,tn,fp,fn=[count[k]for k in('tp','tn','fp','fn')];total=tp+tn+fp+fn;tpr=tp/(tp+fn);tnr=tn/(tn+fp);views=sum(r['views']for r in rows)
    return dict(start_cursor=rows[0]['cursor'],end_cursor=rows[-1]['cursor'],updates=len(rows),question_occurrences=sum(r['questions']for r in rows),correlated_G5_views=views,binary_Mixed_BCE=sum(r['binary_Mixed_BCE']*r['views']for r in rows)/views,confusion=count,MixedBA=(tpr+tnr)/2,overall_accuracy=(tp+tn)/total,Mixed_TPR=tpr,nonMixed_TNR=tnr,nonMixed_FPR=fp/(fp+tn),Mixed_FNR=fn/(fn+tp),predicted_Mixed_fraction=(tp+fp)/total,gradients={k:dist([r[k]for r in rows])for k in('native_gradient_L2','group_head_gradient_L2','gradient_norm_before_clip')},gradient_clip_active_updates=sum(r['gradient_norm_before_clip']>r['gradient_clip']for r in rows),gradient_clip_active_fraction=sum(r['gradient_norm_before_clip']>r['gradient_clip']for r in rows)/len(rows),all_predicted_Mixed_updates=sum(r['confusion']['tn']==0 and r['confusion']['fn']==0 for r in rows),all_predicted_nonMixed_updates=sum(r['confusion']['tp']==0 and r['confusion']['fp']==0 for r in rows),raw_online_changing_model_and_sampled_views=True,fixed_checkpoint_training_evaluation=False,EMA_training_evaluation=False)

def evaluation(done):
    out={}
    for name in('raw','EMA'):
        v=done['metrics'][name]['fixed_selection2048'];p=v['pool'];top=v['selected512'];c=p['confusion'];n=sum(c.values())
        out[name]=dict(binaryMixedBCE=v['binaryMixedBCE'],MixedBA=p['mixed_balanced_accuracy'],overall_accuracy=(c['tp']+c['tn'])/n,Mixed_TPR=p['mixed_recall'],nonMixed_TNR=p['nonmixed_recall'],nonMixed_FPR=p['nonmixed_false_positive_rate'],Mixed_FNR=p['mixed_false_negative_rate'],predicted_Mixed_fraction=(c['tp']+c['fp'])/n,confusion=c,AUC=p['mixed_roc_auc'],Top512_Mixed=top['mixed_questions'],Top512_size=top['questions'],classwise=p['classwise'],questions=p['questions'],checkpoint_cursor=done['cursor'])
    return out

def choose(n,k):return math.factorial(n)//math.factorial(k)//math.factorial(n-k)if n>=k else 0

def sampler_mismatch(prior):
    hist={d:v['K10_histogram']for d,v in prior['domains'].items()};hist['pooled']=[sum(h[k]for h in hist.values())for k in range(11)];out={}
    for domain,h in hist.items():
        n=sum(h);m=sum(h[1:10]);positive_combs=[252-choose(k,5)-choose(10-k,5)for k in range(11)];pos_den=sum(nk*pk for nk,pk in zip(h,positive_combs));neg_den=252*n-pos_den;table=[]
        for k,nk in enumerate(h):
            p=positive_combs[k]/252;cur_pos=nk/m if 0<k<10 else 0.;natural_pos=nk*positive_combs[k]/pos_den;cur_neg=nk/n;natural_neg=nk*(252-positive_combs[k])/neg_den
            table.append(dict(K10=k,questions=nk,natural_pMix=p,current_P_K_given_positive_role=cur_pos,natural_class_conditional_P_K_given_positive_role=natural_pos,current_to_natural_positive_ratio=cur_pos/natural_pos if natural_pos else None,current_P_K_given_negative_role=cur_neg,natural_class_conditional_P_K_given_negative_role=natural_neg,current_to_natural_negative_ratio=cur_neg/natural_neg if natural_neg else None))
        out[domain]=dict(questions=n,eligible=m,uniform_natural_G5_Mixed=pos_den/(252*n),K5_negative_current_fraction=table[5]['current_P_K_given_negative_role'],K5_negative_natural_class_conditional_fraction=table[5]['natural_class_conditional_P_K_given_negative_role'],K5_negative_oversampling_factor=table[5]['current_to_natural_negative_ratio'],K10_table=table)
    return dict(domain_distributions=out,current_sampling='positive q uniform over K1..9; negative q uniform over all; then uniform conditional ownG5 subset, four draws with replacement',natural_class_balanced_target='q positive proportional pMix(q); q negative proportional1-pMix(q); then uniform class-conditional G5',current_hard_policy=dict(epoch0_hard_request_fraction=0.,later_per_role_hard_request_fraction=.5,hard_pool='previous epoch observed qids ranked by mean four-view training BCE; top ceil(observed_qids/4)',hard_draw='uniform with replacement; reject used-qid up to32 times then fallback uniform cycle',all_observed_large_pool_approximation=dict(hard_member_weight_vs_uniform=2.5,nonhard_weight_vs_uniform=.5,hard_to_nonhard_ratio=5.),general_approximate_hard_member_multiplier='.5+.5*P/H where P=role pool size,H=observed top-quarter size; finite-batch rejections/fallback alter this',actual_hard_members_sizes_or_K_domain_weights_observed=False,reason='Only sampler-state hashes are in curve; complete sampler payload lives in large checkpoints and is not read'),causal_failure_attribution_established=False)

def run():
    started=time.perf_counter();c,cr=read(BASE/'config.json');done,dr=read(BASE/'complete.json');prior,pr=read(PRIOR)
    if cr['sha256']!=CONFIG_SHA or pr['sha256']!=PRIOR_SHA or prior['config']!=cr:raise ValueError('Actual source config/prior binding differs')
    if done['cursor']!=6540 or done['status']!='OFFICIAL_DIRECT_MIXED_B128_FOUR_EPOCH_QUESTION_BUDGET_COMPLETE':raise ValueError('Real complete B128 four-epoch run required')
    path=BASE/'curve.jsonl';before=path.stat();raw=path.read_bytes();after=path.stat()
    if(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Curve changed during final read')
    all_rows=[json.loads(line)for line in raw.splitlines()if line];rows=[r for r in all_rows if 1676<=r['cursor']<=6540]
    if [r['cursor']for r in rows]!=list(range(1676,6541)):raise ValueError('Missing/repeated/nonordered accepted B128 cursor')
    for r in rows:
        if r['questions']!=128 or r['views']!=512 or r['mixed_views']!=256 or r['nonmixed_views']!=256 or r['native_steps']!=r['cursor']or r['head_steps']!=r['cursor']:raise ValueError('Actual update/balanced real-G5 clock differs')
        cf=r['confusion']
        if any(v<0 or int(v)!=v for v in cf.values())or cf['tp']+cf['fn']!=256 or cf['tn']+cf['fp']!=256:raise ValueError('Invalid real view confusion counts')
        if any(not math.isfinite(r[k])for k in('binary_Mixed_BCE','native_gradient_L2','group_head_gradient_L2','gradient_norm_before_clip')):raise ValueError('Nonfinite recorded training metric')
    epochs={}
    for number,(lo,hi)in enumerate(RANGES,1):
        part=[r for r in rows if lo<=r['cursor']<=hi];ev,ref=read(BASE/'evaluations'/('epoch-%02d'%number)/'complete.json')
        if ev['cursor']!=hi or ev['epoch']!=number or ev['config_sha256']!=CONFIG_SHA or ev['checkpoint']['cursor']!=hi or ev['checkpoint']['allocation_job_id']!='746266' or ev['models_two_Adam_RNG_native_buffers_unchanged']is not True:raise ValueError('Fixed evaluation/checkpoint/source identity differs')
        epochs['epoch%d'%number]=dict(interval=[lo,hi],online_raw=summary(part),online_raw_first100=summary(part[:100]),online_raw_last100=summary(part[-100:]),frozen_fixed2048=evaluation(ev),evaluation_complete=ref,checkpoint={k:ev['checkpoint'][k]for k in('path','bytes','sha256','cursor','native_steps','head_steps')},scope='E1 window is B128 tail only; its earlier1675 B32 updates are not in this curve'if number==1 else'full B128 question-visit epoch; class-hard sampling changes online train distribution')
    return dict(status='B128_COMPLETE_TRAIN_CURVE_AND_FIXED_EPOCH_RECEIPTS_CPU_SUMMARY',source_config=cr,main_complete=dr,curve=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),all_rows=len(all_rows),accepted_B128_rows=len(rows)),prior_metadata=pr,epochs=epochs,all_B128_online_raw=summary(rows),sampling_target_mismatch=sampler_mismatch(prior),domain_role_priors=prior['uniform_early_role_pool_marginals'],new_model_or_scoring_or_GPU=False,feature_rollout_checkpoint_prediction_payload_read=False,final4096_outcomes_read=False,training_files_changed=False,elapsed_seconds=time.perf_counter()-started,observed_epoch=time.time())

if __name__=='__main__':print(json.dumps(run(),sort_keys=True,allow_nan=False))
