"""Train-only Prefix20 hard-example enrichment and class-matched random control.

No model loading/GPU/Slurm, no active runner import, no validation/final fitting.
The initial canonical169028 warmup ALREADY used these questions: this only adds
questions outside the CURRENT6528 controlled-fit cohort, never claims unseen
initial-warmup or unseen previous-finetuning examples. Source EM remains own G5.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

CANONICAL=169028
CORE=6528
CANDIDATES=32768
QUOTA=4092
BATCH=48
ROLES={'matched_fit','diagnostic512','selection2048','frozen_final'}
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def identity_digest(qids):
    # Existing warmup plan/EMA evaluator uses ordinary sorted-key compact JSON.
    # All native qids are ASCII; keep explicit encoding compatible nonetheless.
    return hashlib.sha256(json.dumps(qids,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def rank(seed,scope,qid):
    if type(seed) is not int:raise ValueError('Explicit integer seed required')
    return digest(['training-hard-enrichment-v1',seed,scope,qid])


def source_identities(groups):
    seen_qids=set();seen_tids=set();out=[]
    for g in groups:
        qid=g['qid'];qhash=g['question_hash'];tids=g['trajectory_ids']
        if (not isinstance(qid,str) or not qid or qid in seen_qids or
                not isinstance(qhash,str) or len(qhash)!=64 or any(c not in '0123456789abcdef' for c in qhash) or
                not isinstance(tids,list) or len(tids)!=5 or len(set(tids))!=5 or
                any(not isinstance(t,str) or not t for t in tids) or seen_tids.intersection(tids)):
            raise ValueError('Distinct actual question IDs/hash and five own globally distinct trajectory IDs required')
        seen_qids.add(qid);seen_tids.update(tids)
        out.append(dict(qid=qid,question_hash=qhash,trajectory_ids=list(tids)))
    return out


def own_group(source):
    identity=source_identities([source])[0];labels=source['labels']
    if not isinstance(labels,list) or len(labels)!=5 or any(type(v) is not int or v not in (0,1) for v in labels):
        raise ValueError('Five own strict binary integer EM labels required')
    if source.get('split','train')!='train':raise ValueError('Only authentic training groups may enter hard mining')
    return dict(identity,labels=list(labels),split='train')


def exclusion_sets(audits,required_scopes=True):
    """Open only small identity-only audits; source_refs are provenance, not data.

    Each audit source_ref carries role matched_fit/diagnostic512/selection2048/
    frozen_final, or audit.scopes declares those source names. Never open final
    source_refs here. Actual current-fit/diagnostic membership is checked below.
    """
    if not audits:raise ValueError('Explicit training-mining identity exclusions required')
    qids=set();hashes=set();mapping={};declared=set();duplicates=0
    for audit in audits:
        if audit.get('role')!='training_mining_exclusions' or not audit.get('source_refs'):
            raise ValueError('SHA-bound training_mining_exclusions identity audit with source_refs required')
        if any(k in audit for k in ('labels','scores','rollouts')):
            raise ValueError('Exclusion audit must remain identity-only')
        scopes=audit.get('scopes',{})
        declared.update(scopes.keys() if isinstance(scopes,dict) else scopes)
        for ref in audit['source_refs']:
            if not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str) or len(ref['sha256'])!=64:
                raise ValueError('Exclusion source provenance requires actual path/SHA')
            if ref.get('role') in ROLES:declared.add(ref['role'])
        for row in audit['groups']:
            if set(row)!={'qid','question_hash'}:raise ValueError('Exclusion rows contain only qid/question_hash')
            qid,qhash=row['qid'],row['question_hash']
            if not isinstance(qid,str) or not qid or not isinstance(qhash,str) or len(qhash)!=64:
                raise ValueError('Valid exclusion qid/question hash required')
            if qid in mapping and mapping[qid]!=qhash:raise ValueError('Conflicting exclusion identity hashes')
            if qid in mapping:duplicates+=1
            mapping[qid]=qhash;qids.add(qid);hashes.add(qhash)
    if required_scopes and not ROLES<=declared:
        raise ValueError('Exclusion provenance must explicitly cover '+','.join(sorted(ROLES-declared)))
    return qids,hashes,dict(questions=len(qids),question_hashes=len(hashes),duplicate_identity_rows=duplicates,
                          declared_scopes=sorted(declared),final_source_files_opened=False)


def core_groups(bank,plan,expected=CORE):
    lookup={g['qid']:g for g in bank['train']};qids=plan['cohort_qids']
    if len(lookup)!=len(bank['train']) or len(qids)!=expected or len(set(qids))!=expected:
        raise ValueError('Exact unique current controlled-fit cohort required')
    if set(qids)!={q for b in plan['batches'] for q in b}:
        raise ValueError('Current matched cohort/batch identity disagreement')
    groups=[own_group(lookup[q]) for q in qids];source_identities(groups)
    return groups


def prepare_candidates(bank,matched_plan,audits,*,source_bank,features,checkpoint,
        exclusion_refs,matched_plan_ref,seed=20261021,synthetic_fixture=False,
        candidate_count=CANDIDATES,expected_source=CANONICAL,expected_core=CORE):
    if not synthetic_fixture and (candidate_count,expected_source,expected_core)!=(CANDIDATES,CANONICAL,CORE):
        raise ValueError('Production uses full canonical169028/current6528/candidate32768 budgets')
    if len(bank['train'])!=expected_source:raise ValueError('Source train bank is the whole canonical eligible pool')
    if candidate_count<1:raise ValueError('Positive candidate budget required')
    core=core_groups(bank,matched_plan,expected_core)
    if len({g['question_hash'] for g in core})!=len(core):
        raise ValueError('Current core contains repeated question hashes; cannot promise unique full branches')
    excluded,excluded_hashes,exclusion=exclusion_sets(audits,required_scopes=not synthetic_fixture)
    for g in core+bank.get('diagnostic',[]):
        if g['qid'] not in excluded or g['question_hash'] not in excluded_hashes:
            raise ValueError('Current core or fixed diagnostic is missing from explicit exclusions')
    identities=source_identities(bank['train']);by={g['qid']:g for g in bank['train']}
    # Selection sees identities only. Whole169028 source is NOT an exclusion set.
    eligible=[g for g in identities if g['qid'] not in excluded and g['question_hash'] not in excluded_hashes]
    eligible.sort(key=lambda g:(rank(seed,'candidate-question-hash',g['qid']),g['qid']))
    unique=[];seen_hashes=set();hash_duplicates=0
    for g in eligible:
        if g['question_hash'] in seen_hashes:hash_duplicates+=1;continue
        seen_hashes.add(g['question_hash']);unique.append(g)
    if len(unique)<candidate_count:
        raise ValueError('Only%d eligible distinct-question identities, requested%d; no resampling/validation fallback'%(len(unique),candidate_count))
    chosen=unique[:candidate_count]
    # Join own EM only after identity-only question selection is frozen.
    selected=[own_group(by[g['qid']]) for g in chosen]
    summary=dict(questions=len(selected),trajectories=5*len(selected),
        question_selection_uses_labels=False,question_selection_uses_scores=False,
        selection_uses_labels=False,uses_scores=False,validation_or_final_questions_used=False,
        unseen_in_initial169028_warmup_claim=False,unseen_in_previous_finetuning_claim=False,
        unseen_initial_warmup=False,unseen_previous_ft=False,excluded_current_matched6528_fit=True,
        excluded_current_fit_questions=expected_core,eligible_source_questions=len(identities),
        eligible_after_exclusions=len(eligible),eligible_distinct_question_hashes=len(unique),
        duplicate_eligible_question_hashes_pruned=hash_duplicates,
        source_bank_train_is_not_an_exclusion_set=True,exclusion=exclusion,
        ordered_qid_sha256=identity_digest([g['qid'] for g in selected]),
        ordered_own_trajectory_ids_sha256=identity_digest([g['trajectory_ids'] for g in selected]),
        own_EM_joined_after_question_selection=True,source_actor_optimizer_updates=0,seed=seed)
    return dict(schema_version=1,role='training_hard_mining_candidates',groups=selected,
        summary=summary,source_bank=source_bank,features=features,checkpoint=checkpoint,
        source_checkpoint=checkpoint,
        exclusion_refs=exclusion_refs,source_matched_plan=matched_plan_ref,prefix_tokens=20,
        model_revision=REVISION,initial_warmup_already_used_all169028_questions=True)


def product(values):
    out=1.
    for value in values:out*=value
    return out


def validate_predictions(candidate_plan,rows,expected=CANDIDATES,tolerance=1e-6):
    required=dict(questions=expected,trajectories=5*expected,
        question_selection_uses_labels=False,question_selection_uses_scores=False,
        validation_or_final_questions_used=False,unseen_in_initial169028_warmup_claim=False,
        unseen_in_previous_finetuning_claim=False,excluded_current_matched6528_fit=True)
    if (candidate_plan.get('role')!='training_hard_mining_candidates' or
            candidate_plan.get('prefix_tokens')!=20 or candidate_plan.get('model_revision')!=REVISION or
            candidate_plan.get('source_checkpoint')!=candidate_plan.get('checkpoint') or
            any(candidate_plan['summary'].get(k)!=v for k,v in required.items())):
        raise ValueError('Training-only Prefix20 canonical actor and label-blind candidate plan required')
    groups=candidate_plan['groups'];own={g['qid']:own_group(g) for g in groups}
    if len(groups)!=expected or len(own)!=expected:raise ValueError('Complete declared candidate coverage required')
    source_identities(groups)
    if candidate_plan['summary']['ordered_qid_sha256']!=identity_digest([g['qid'] for g in groups]):
        raise ValueError('Candidate ordered qid identity SHA differs')
    if candidate_plan['summary']['ordered_own_trajectory_ids_sha256']!=identity_digest([g['trajectory_ids'] for g in groups]):
        raise ValueError('Candidate own trajectory identity SHA differs')
    result={};max_difference=0.
    for row in rows:
        qid=row['qid']
        if qid not in own or qid in result:raise ValueError('Extra/repeated scored question')
        g=own[qid]
        if (row['trajectory_ids']!=g['trajectory_ids'] or row['strict_em']!=g['labels'] or
                not isinstance(row['strict_em'],list) or
                any(type(y) is not int or y not in (0,1) for y in row['strict_em'])):
            raise ValueError('Score five-trajectory IDs/order or own EM differ from authentic candidate')
        ps=row['p_j'];q=row['predicted_mixed']
        if (not isinstance(ps,list) or len(ps)!=5 or
                any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in ps) or
                type(q) not in (int,float) or not math.isfinite(q) or not 0<=q<=1):
            raise ValueError('Finite own five probabilities and Mixed probability in[0,1] required')
        calculated=1-product(ps)-product([1-p for p in ps]);difference=abs(q-calculated)
        if difference>tolerance:raise ValueError('Predicted q differs from own five-p product formula')
        max_difference=max(max_difference,difference)
        result[qid]=dict(group=g,p_j=list(ps),q=float(q),K=sum(g['labels']))
    if set(result)!=set(own):raise ValueError('Incomplete scored candidate pool; do not select from partial32768')
    return result,dict(questions=len(result),trajectories=5*len(result),
        complete_own_ids_EM_probabilities_verified=True,q_formula='1-product(p_j)-product(1-p_j)',
        q_absolute_tolerance=tolerance,max_q_roundoff=max_difference,scoring_did_not_choose_questions=True)


class QuotaUnreachable(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__('Requested hard enrichment quota unreachable: '+json.dumps(details,sort_keys=True))


def _distribution(groups):return [sum(sum(g['labels'])==k for g in groups) for k in range(6)]


def select_enrichment(candidate_plan,predictions,core,*,seed=20261021,quota=QUOTA,
        expected_candidates=CANDIDATES,expected_core=CORE,synthetic_fixture=False):
    if not synthetic_fixture and (quota,expected_candidates,expected_core)!=(QUOTA,CANDIDATES,CORE):
        raise ValueError('Production quota4092/candidates32768/core6528 required')
    core=[own_group(g) for g in core];source_identities(core)
    if len(core)!=expected_core:raise ValueError('Exact current core budget required')
    scored,verified=validate_predictions(candidate_plan,predictions,expected_candidates)
    cq={g['qid'] for g in core};ch={g['question_hash'] for g in core}
    if cq&set(scored) or ch&{r['group']['question_hash'] for r in scored.values()}:
        raise ValueError('Core and supplementary candidates overlap by qid/question hash')
    k0=[r for r in scored.values() if r['K']==0];k5=[r for r in scored.values() if r['K']==5]
    mixed=[r for r in scored.values() if 0<r['K']<5]
    reachable=min(len(k0),len(k5),len(mixed)//2)
    if reachable<quota:
        raise QuotaUnreachable(dict(requested_per_stratum=quota,maximum_common_reachable_quota=reachable,
            actual_allwrong=len(k0),actual_allcorrect=len(k5),actual_Mixed=len(mixed),
            required_Mixed=2*quota,required_total_supplement=4*quota,
            no_synthetic_labels=True,no_resampling=True,no_validation_fallback=True))
    def hashed(rows,scope):return sorted(rows,key=lambda r:(rank(seed,scope,r['group']['qid']),r['group']['qid']))
    high0=sorted(k0,key=lambda r:(-r['q'],rank(seed,'hard-allwrong-tie',r['group']['qid'])))[:quota]
    high5=sorted(k5,key=lambda r:(-r['q'],rank(seed,'hard-allcorrect-tie',r['group']['qid'])))[:quota]
    lowmixed=sorted(mixed,key=lambda r:(r['q'],rank(seed,'hard-low-Mixed-tie',r['group']['qid'])))[:quota]
    lowids={r['group']['qid'] for r in lowmixed}
    anchor=hashed([r for r in mixed if r['group']['qid'] not in lowids],'hard-remaining-Mixed-anchor')[:quota]
    random_mixed=hashed(mixed,'random-control-Mixed')[:2*quota]
    strata=dict(targeted=dict(high_q_allwrong=high0,high_q_allcorrect=high5,low_q_Mixed=lowmixed,Mixed_hash_anchor=anchor),
        random_control=dict(random_allwrong=hashed(k0,'random-control-allwrong')[:quota],
            random_allcorrect=hashed(k5,'random-control-allcorrect')[:quota],
            random_Mixed_A=random_mixed[:quota],random_Mixed_B=random_mixed[quota:]))
    branches={}
    for name,parts in strata.items():
        rows=[r for part in parts.values() for r in part];supp=[r['group'] for r in rows]
        qids=[g['qid'] for g in supp];source_identities(supp)
        if len(set(qids))!=4*quota:raise ValueError('Hard/random strata must contain unique actual questions')
        combined=core+supp;source_identities(combined)
        if len({g['question_hash'] for g in combined})!=len(combined):raise ValueError('Duplicate actual question hash in full branch')
        combined.sort(key=lambda g:(rank(seed,'paired-training-epoch-order',g['qid']),g['qid']))
        total=expected_core+4*quota
        if not synthetic_fixture and (total!=22896 or total%BATCH):raise ValueError('Exact B48*477 one-pass budget required')
        batch=BATCH if not synthetic_fixture else 1
        batch_ids=[g['qid'] for g in combined]
        plan=dict(cohort_qids=batch_ids,batches=[batch_ids[i:i+batch] for i in range(0,total,batch)],
            batch_epochs=[0]*(total//batch),global_groups=batch,update_budget=total//batch,
            seed=seed,epochs=1,each_question_once=True,ordered_qid_sha256=identity_digest(batch_ids),
            role='training_hard_enrichment_fit_plan',source_checkpoint=candidate_plan['checkpoint'])
        strata_summary={}
        for stratum,items in parts.items():
            qs=[r['q'] for r in items];ys=[r['K'] for r in items]
            strata_summary[stratum]=dict(questions=len(items),K_histogram=[ys.count(k) for k in range(6)],
                q_min=min(qs),q_max=max(qs),q_mean=sum(qs)/len(qs),
                true_negative_called_Mixed=sum(r['K'] in (0,5) and r['q']>=.5 for r in items),
                true_Mixed_called_nonMixed=sum(0<r['K']<5 and r['q']<.5 for r in items),
                qid_sha256=identity_digest([r['group']['qid'] for r in items]))
        summary=dict(branch=name,core_questions=len(core),supplement_questions=len(supp),
            total_unique_questions=len(combined),total_unique_trajectories=5*len(combined),
            supplement_K_histogram=_distribution(supp),full_K_histogram=_distribution(combined),
            supplement_class_counts=dict(allwrong=quota,allcorrect=quota,Mixed=2*quota),
            strata=strata_summary,question_selection_uses_labels=True,
            question_selection_uses_scores=name=='targeted',validation_or_final_questions_used=False,
            high_low_are_rank_based_not_a_new_probability_threshold=True,
            strict_error_threshold_for_reporting=.5,quota_strata_probability_threshold_filter=False,
            unseen_in_initial169028_warmup_claim=False,unseen_in_previous_finetuning_claim=False,
            Mixed_internal_K_distribution_matched_claim=False,
            same_core_seed_unique_question_count_class_ratios_and_budget_as_other_branch=True,
            global_groups=batch,update_budget=total//batch,epochs=1,seed=seed)
        branches[name]=dict(groups=combined,supplement_groups=supp,plan=plan,summary=summary)
    return dict(role='training_only_hard_enrichment_and_class_matched_random_control',branches=branches,
        candidate_verification=verified,source_candidate_summary=candidate_plan['summary'],
        controls=dict(same_core=True,same_unique_count=True,same_macro_class_counts=True,same_seed=True,same_update_budget=True,
            Mixed_K1_K2_K3_K4_matched=False,initial_warmup_unseen_claim=False,previous_finetuning_unseen_claim=False))


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()


def checked(ref):
    p=Path(ref['path'])
    if not p.is_absolute() or p.resolve()!=p or not str(p).startswith(('/data/','/dataset/','/backup/','/mnt/')):
        raise ValueError('Absolute nonsymlink remote source path required')
    if file_sha(p)!=ref['sha256'] or ('bytes' in ref and p.stat().st_size!=ref['bytes']):
        raise ValueError('Actual source SHA/bytes mismatch: '+str(p))
    return p


def read(path):return json.loads(Path(path).read_text())


def once(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    raw=canonical(value)+b'\n'
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Refuse changing immutable CPU plan/output '+str(path))
    else:
        with path.open('xb') as f:f.write(raw)
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def verify_scoring_receipt(receipt,candidate_ref,candidate):
    if (receipt['status']!='E4_PREFIX20_TRAIN_ONLY_CANDIDATE_SCORING_COMPLETE' or
            receipt['questions']!=CANDIDATES or receipt['trajectories']!=5*CANDIDATES):
        raise ValueError('Complete actual E4 candidate scoring receipt required')
    b=receipt['binding'];cp=b['checkpoint']
    if (b['role']!='training_hard_mining_candidates' or b['actor_optimizer_updates']!=0 or
            b['probe_optimizer_updates']!=0 or b['hard_strata_selected'] is not False or
            cp['name']!='E4' or cp['steps']!=14418 or cp['epoch']!=4 or
            b['candidate_plan']['sha256']!=candidate_ref['sha256'] or
            cp['sha256']!=candidate['checkpoint']['sha256'] or
            b['source_bank']['sha256']!=candidate['source_bank']['sha256'] or
            b['features']['sha256']!=candidate['features']['sha256']):
        raise ValueError('E4/source/role/complete-score binding mismatch')
    for observed,wanted in ((b['candidate_plan'],candidate_ref),(cp,candidate['checkpoint']),
            (b['source_bank'],candidate['source_bank']),(b['features'],candidate['features'])):
        if any(observed.get(k)!=wanted[k] for k in ('path','sha256','bytes') if k in wanted):
            raise ValueError('Full scoring source path/bytes/SHA binding mismatch')
    acceptance=b['plan_acceptance']
    for key in ('questions','trajectories','ordered_qid_sha256','ordered_own_trajectory_ids_sha256'):
        if acceptance[key]!=candidate['summary'][key]:raise ValueError('Scoring acceptance differs from complete candidate identities')
    if b['exclusion_refs']!=candidate['exclusion_refs']:raise ValueError('Scoring exclusion identity provenance changed')
    return receipt['predictions']


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--action',choices=('prepare','select'),required=True);a=p.parse_args()
    config_path=Path(a.config);c=read(config_path);out=Path(a.output)
    if not out.is_absolute() or not str(out).startswith('/data/user/jxu957/experiments/'):
        raise ValueError('All real identities/own EM/predictions/plans remain on remote shared storage')
    bank=read(checked(c['source_bank']));matched=read(checked(c['matched_plan']));checked(c['features'])
    audits=[read(checked(ref)) for ref in c['exclusion_refs']]
    if a.action=='prepare':
        candidate=prepare_candidates(bank,matched,audits,source_bank=c['source_bank'],features=c['features'],
            checkpoint=c['checkpoint'],exclusion_refs=c['exclusion_refs'],matched_plan_ref=c['matched_plan'],seed=c['seed'])
        ref=once(out/'candidates.json',candidate)
        result=dict(status='TRAIN_ONLY32768_HASH_CANDIDATES_PREPARED_NOT_SCORED',candidate_plan=ref,summary=candidate['summary'])
        once(out/'prepare-complete.json',result);print(json.dumps(result),flush=True);return
    candidate=read(checked(c['candidate_plan']));receipt=read(checked(c['scoring_complete']))
    for key in ('source_bank','features','checkpoint'):
        if candidate[key]!=c[key]:raise ValueError('Selection configuration differs from frozen candidate '+key)
    if candidate['source_matched_plan']!=c['matched_plan'] or candidate['exclusion_refs']!=c['exclusion_refs']:
        raise ValueError('Selection core/exclusions differ from candidate preparation')
    prediction_ref=verify_scoring_receipt(receipt,c['candidate_plan'],candidate)
    path=checked(prediction_ref)
    with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
    core=core_groups(bank,matched)
    try:result=select_enrichment(candidate,rows,core,seed=c['seed'])
    except QuotaUnreachable as error:
        once(out/'quota-unreachable.json',dict(status='TRAIN_ONLY_QUOTA_UNREACHABLE_NOT_RESAMPLED',details=error.details,
            source_scores=prediction_ref,source_candidate=c['candidate_plan']));raise
    outputs={}
    for name,branch in result['branches'].items():
        root=out/name
        outputs[name]=dict(groups=once(root/'groups.json',dict(train=branch['groups'])),
            plan=once(root/'plan.json',branch['plan']),summary=branch['summary'])
    complete=dict(status='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE',outputs=outputs,
        candidate_verification=result['candidate_verification'],controls=result['controls'],
        source_candidates=c['candidate_plan'],source_scoring_receipt=c['scoring_complete'],source_predictions=prediction_ref,
        source_config=dict(path=str(config_path),bytes=config_path.stat().st_size,sha256=file_sha(config_path)),
        source_bank=c['source_bank'],features=c['features'],checkpoint=c['checkpoint'],source_matched_plan=c['matched_plan'],
        exclusion_refs=c['exclusion_refs'],GPU_used=False,actor_updates=0,probe_updates=0)
    once(out/'selection-complete.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
