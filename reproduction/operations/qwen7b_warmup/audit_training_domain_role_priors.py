"""Read-only domain/class-role priors of the existing real-G10 sampler.

Only source config, accepted groups/metadata and identity exclusions are read.
No model, feature, rollout, checkpoint, prediction or actual batch replay.
"""
import argparse,collections,hashlib,json,math,resource,time
from pathlib import Path
SAMPLER_SHA='5d0fbb1e582ae4972957e0fbea18da6d12b61b7f7aac2f7094ba3de81a03e7d3'
SAMPLER_FILE='reproduction/operations/qwen7b_warmup/balanced_hard_g10_sampler.py'

def proof(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def checked(ref):
    p=Path(ref['path']);a=proof(p)
    if not p.is_absolute()or a['sha256']!=ref['sha256']or('bytes'in ref and a['bytes']!=ref['bytes']):raise ValueError('Actual metadata SHA/size differs: '+str(p))
    return p

def read(ref):
    with checked(ref).open()as f:return json.load(f)

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def choose(n,k):return math.factorial(n)//math.factorial(k)//math.factorial(n-k)if n>=k else 0

def run(config,sha,output):
    start=time.perf_counter();config_ref=proof(config)
    if config_ref['sha256']!=sha:raise ValueError('Exact current B128 config required')
    c=read(config_ref);bank=read(c['complete_bank']);exclusions=read(c['exclusions'])
    if c['batch_questions']!=128 or c['architecture']['views_per_question']!=4 or c['fit_questions']!=169028:raise ValueError('This audit binds the actual B128/4view169028 configuration')
    if c['code_hashes'][SAMPLER_FILE]!=SAMPLER_SHA:raise ValueError('Sampler definition/source differs')
    if bank['status']!='FROZEN_FIVE_BANK_REAL_G10_169028_NATIVE_JOINT_INPUTS_ACCEPTED'or bank['groups']!=c['groups']:raise ValueError('Actual accepted complete/groups binding differs')
    forbidden={g['qid']for g in exclusions['groups']};blocked_hashes={g['question_hash']for g in exclusions['groups']}
    if len(forbidden)!=6656:raise ValueError('Expected only fixed6656 evaluation identity exclusion metadata')
    groups=read(c['groups'])['train'];stats={};qids=set();hashes=set();bad_qid=bad_hash=0
    for g in groups:
        qid=g['qid'];domain=json.loads(qid)[0]
        if domain not in('nq','hotpotqa')or qid in qids:raise ValueError('Canonical training qid/domain binding differs')
        labels=g['labels']
        if len(labels)!=10 or any(type(y)is not int or y not in(0,1)for y in labels):raise ValueError('Only authentic G10 own EM labels')
        if len(g['trajectory_ids'])!=10 or len(set(g['trajectory_ids']))!=10:raise ValueError('Own10 real trajectories required')
        qids.add(qid);hashes.add(g['question_hash']);bad_qid+=qid in forbidden;bad_hash+=g['question_hash']in blocked_hashes
        d=stats.setdefault(domain,dict(questions=0,K10_histogram=[0]*11,positive_eligible_questions=0,uniform_G5_Mixed_subsets_total=0))
        k=sum(labels);d['questions']+=1;d['K10_histogram'][k]+=1;d['positive_eligible_questions']+=0<k<10;d['uniform_G5_Mixed_subsets_total']+=252-choose(k,5)-choose(10-k,5)
    if bad_qid or bad_hash:raise ValueError('True evaluation qid/hash entered current training groups')
    if len(qids)!=169028 or digest(sorted(qids))!=bank['canonical_qid_sha256']:raise ValueError('Current G10 groups do not equal accepted full canonical169028 IDs')
    if {d:v['questions']for d,v in stats.items()}!=bank['domains']:raise ValueError('Canonical bank domain counts differ')
    total=len(groups);positive=sum(v['positive_eligible_questions']for v in stats.values());negative=total;half=c['batch_questions']//2
    ideal={};finite={}
    for domain,d in stats.items():
        n=d['questions'];m=d['positive_eligible_questions'];p_d_pos=m/positive;p_d_neg=n/total
        d.update(G10_heterogeneous_fraction=m/n,negative_eligible_questions=n,uniform_G10_then_uniform_real_G5_natural_Mixed_probability=d['uniform_G5_Mixed_subsets_total']/(252*n))
        ideal[domain]=dict(P_domain_given_Mixed_role=p_d_pos,P_domain_given_nonMixed_role=p_d_neg,P_Mixed_role_given_domain=p_d_pos/(p_d_pos+p_d_neg),P_nonMixed_role_given_domain=p_d_neg/(p_d_pos+p_d_neg),P_domain_in_balanced_stream=.5*(p_d_pos+p_d_neg),positive_question_draw_weight_per_question=.5/positive,negative_question_draw_weight_per_question=.5/negative)
        neg_excluding_expected_positive=(n-half*p_d_pos)/(total-half)
        finite[domain]=dict(P_domain_given_nonMixed_role=neg_excluding_expected_positive,P_Mixed_role_given_domain=p_d_pos/(p_d_pos+neg_excluding_expected_positive),negative_domain_probability_delta_from_pool_marginal=neg_excluding_expected_positive-p_d_neg)
    result=dict(status='CURRENT_B128_REAL_G10_DOMAIN_ROLE_PRIORS_METADATA_COMPLETE',config=config_ref,groups=c['groups'],complete_bank=c['complete_bank'],identity_exclusions=c['exclusions'],sampler_source_commit=c['sampler_source'],sampler_file_sha256=SAMPLER_SHA,canonical_qid_sha256=bank['canonical_qid_sha256'],canonical_domain_counts=bank['domains'],questions=total,positive_eligible=positive,negative_eligible=negative,K10_histogram=[sum(v['K10_histogram'][k]for v in stats.values())for k in range(11)],domains=stats,uniform_early_role_pool_marginals=ideal,first_fresh_uniform_batch_expected_distinct_question_correction=finite,formulas=dict(positive_pool='K10 in 1..9',negative_pool='all G10 questions; even heterogeneous G10 has >=5 own trajectories sharing a label',role_prior='P(Mixed-role)=P(nonMixed-role)=1/2; four conditional G5 views each weight1/4 per question',P_domain_given_Mixed_role='M_domain / sum(M_domain)',P_domain_given_nonMixed_role='N_domain / sum(N_domain)',P_Mixed_role_given_domain='(M_domain/M) / ((M_domain/M)+(N_domain/N))',natural_G5_Mixed='mean_q [1-(C(K10,5)+C(10-K10,5))/C(10,5)]',finite_batch='positive slots are selected first; hypothetical fresh uniform permutations: negative domain mass=(N_domain-(B/2)*M_domain/M)/(N-B/2)'),scope=dict(early_uniform_only=True,hard_pool_after_epoch0_not_modelled=True,uniform_shuffled_cycles_and_cross_role_duplicate_skips_mean_marginals_are_not_actual_batch_counts=True,actual_per_qid_batch_domain_observed=False,actual_batch_domain_unavailable_reason='Existing saved curves do not retain per-question sampled IDs; no checkpoint/sampler or RNG replay requested.',four_views_are_correlated_within_question=True,training_text_already_in_original_warmup=True,causal_effect_of_domain_prior_on_validation_not_established=True,validation_predictions_or_labels_read=False,final_outcomes_read=False,evaluation_identity_metadata_only=True,feature_rollout_tensor_checkpoint_read=False,GPU_used=False,training_job_changed=False),identity_overlap=dict(evaluation_qids=bad_qid,evaluation_hashes=bad_hash),uniform_natural_Mixed_overall=sum(v['uniform_G5_Mixed_subsets_total']for v in stats.values())/(252*total),elapsed_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,epoch=time.time())
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(receipt=proof(p),result=result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.config,a.config_sha256,a.output)))
