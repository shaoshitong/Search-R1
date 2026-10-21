"""Export one SHA-bound audited delta against explicitly named prior snapshots.

No discovery of new source batches, rollout, model/GPU use, or prior snapshot
mutation. Counts/domains/prior membership are immutable plan data, not inferred
from changing producer state or hardcoded to a historical delta budget.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
from . import peer_newseed_export as e
from . import peer_newseed_delta as d
BASE=Path(e.RECORDED)
STATUS='FROZEN_AUDITED_DELTA_SUBSET_RAW_SNAPSHOT_FRESH_SHA_ACCEPTED'

def digest(value):return hashlib.sha256(e.canonical(value)).hexdigest()
def checked(ref):
    p=Path(ref['path']);e.regular(p.parent,p)
    if p.stat().st_size!=ref['bytes'] or e.sha(p)!=ref['sha256']:raise ValueError('Frozen source descriptor changed: '+str(p))
    return p

def validate_selection(v):
    if (v.get('subset')!='existing-audited' or v.get('source_base')!=str(BASE) or v.get('source_run_id')!=e.RUN_ID or v.get('seed')!=e.SEED or v.get('model_revision')!=e.REVISION or v.get('sampling_attempt_id')!=e.ATTEMPT):raise ValueError('Exact declared same-source audited frozen delta required')
    rows=v['selected_batches'];counts=[v.get(k) for k in ('batches','questions','trajectories')]
    if any(type(x) is not int or x<1 for x in counts) or len(rows)!=v['batches'] or len({x['input'] for x in rows})!=len(rows) or rows!=sorted(rows,key=lambda x:x['input']):raise ValueError('Exact distinct frozen batch order/count required')
    if sum(x['questions'] for x in rows)!=v['questions'] or v['trajectories']!=v['questions']*5 or sum(x['trajectories'] for x in rows)!=v['trajectories'] or sum(v['domain_counts'].values())!=v['questions'] or digest(rows)!=v['batch_records_sha256']:raise ValueError('Frozen record totals/domains/digest differ')
    if any(x['audit']['status']!='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND' for x in rows):raise ValueError('Every selected batch requires frozen independent fstat proof')
    prior=v.get('prior_snapshots')
    if not isinstance(prior,list) or not prior or len({p['complete']['path'] for p in prior})!=len(prior):raise ValueError('Explicit distinct prior snapshot descriptors required')
    expected_status='FROZEN_SOURCE_AUDITED%d_AFTER_ACCEPTED%d_METADATA_ONLY'%(v['batches'],sum(x['expected_batches'] for x in prior))
    if v.get('status')!=expected_status:raise ValueError('Frozen metadata-only status/count binding differs')
    for p in prior:
        if any(type(p.get(k)) is not int or p[k]<1 for k in ('expected_batches','expected_questions','expected_trajectories')) or p['expected_trajectories']!=p['expected_questions']*5:raise ValueError('Typed prior expected budget required')
    return v

def prior_membership(priors,allow_canonical_hash_repeats=False):
    old=dict(qids=set(),hashes=set(),tids=set(),inputs=set());proofs=[]
    for expected in priors:
        ref=expected['complete'];complete=checked(ref);v=e.read(complete);p=checked(v['plan']);membership=checked(v['canonical_membership']);plan=e.read(p);groups=e.read(membership)['groups']
        if (v['questions']!=expected['expected_questions'] or v['trajectories']!=expected['expected_trajectories'] or v['frozen_sealed_batches']!=expected['expected_batches'] or len(groups)!=v['questions'] or len(plan['selected_batches'])!=v['frozen_sealed_batches']):raise ValueError('Actual prior snapshot budget differs from immutable plan')
        inputs={x['input'] for x in plan['selected_batches']};ids={g['qid'] for g in groups};tids={t for g in groups for t in g['trajectory_ids']};hashes={g['question_hash'] for g in groups}
        if len(ids)!=len(groups) or len(tids)!=len(groups)*5 or any(len(g['trajectory_ids'])!=5 for g in groups) or old['inputs'].intersection(inputs) or old['qids'].intersection(ids) or old['tids'].intersection(tids) or (old['hashes'].intersection(hashes) and not allow_canonical_hash_repeats):raise ValueError('Prior snapshots overlap or real five identities are incomplete')
        for key,values in [('inputs',inputs),('qids',ids),('tids',tids),('hashes',hashes)]:old[key].update(values)
        proofs.append(dict(complete=e.fp(complete),plan=e.fp(p),membership=e.fp(membership),questions=len(groups),trajectories=len(tids),batches=len(inputs)))
    return old,proofs

def identity_guard(old,groups,seen,allow_canonical_hash_repeats=False):
    for g in groups:
        q,h,ts=g['qid'],g['question_hash'],g['trajectory_ids']
        if q in old['qids'] or (h in old['hashes'] and not allow_canonical_hash_repeats) or old['tids'].intersection(ts):raise ValueError('New frozen delta overlaps accepted prior identities')
        if q in seen['qids'] or (h in seen['hashes'] and not allow_canonical_hash_repeats) or seen['tids'].intersection(ts) or len(ts)!=5 or len(set(ts))!=5:raise ValueError('New delta repeats question/hash/authentic-five identity')
        if h in old['hashes']:seen['prior_hash_overlap_questions']=seen.get('prior_hash_overlap_questions',0)+1
        if h in seen['hashes']:seen['internal_duplicate_hash_questions']=seen.get('internal_duplicate_hash_questions',0)+1
        seen['qids'].add(q);seen['hashes'].add(h);seen['tids'].update(ts)

def verify_id_digests(ids,ordered_digest,sorted_proof_digest):
    if len(ids)!=len(set(ids)) or digest(ids)!=ordered_digest or digest(sorted(ids))!=sorted_proof_digest:raise ValueError('Actual ordered plan IDs or independently sorted proof IDs differ')

def freeze(base,selection_path,selection_sha,root):
    base,selection_path,root=map(Path,(base,selection_path,root));selref=e.fp(selection_path)
    if selref['sha256']!=selection_sha:raise ValueError('Exact published frozen selection file SHA required')
    selection=validate_selection(e.read(selection_path))
    if base!=BASE or base.resolve()!=base:raise ValueError('Canonical approved physical source only')
    config=checked(selection['source_config']);manifest=checked(selection['input_manifest'])
    if config!=base/'config.json' or manifest!=base/'inputs/manifest.json':raise ValueError('Canonical source binding differs')
    d.contract(e.read(config));declared=e.read(manifest)
    if declared['questions']!=169028 or len(declared['batches'])!=1322:raise ValueError('Original full input manifest differs')
    policy=selection.get('canonical_train_text_hash_policy')
    declared_policy=selection.get('training_text_hash_policy')
    if policy is None and declared_policy is not None:
        expected='retain canonical distinct qids/authentic trajectory IDs; train-only repeated text hashes are reported; evaluation qid/hash exclusion remains hard'
        if declared_policy!=expected:raise ValueError('Unknown frozen training text-hash policy')
        policy=dict(allow_same_hash_for_distinct_canonical_training_qids=True,qid_and_own_ids_unique=True,nontraining_qid_or_hash_overlap_forbidden=True,expected_prior_hash_overlap_questions=selection['prior_identity_overlap']['question_hashes'],expected_internal_duplicate_hash_questions=selection['within_selected_training_hash_duplicates'])
    allow_hashes=policy is not None
    if allow_hashes:
        if policy.get('allow_same_hash_for_distinct_canonical_training_qids')is not True or policy.get('qid_and_own_ids_unique')is not True or policy.get('nontraining_qid_or_hash_overlap_forbidden')is not True or any(type(policy.get(k))is not int or policy[k]<0 for k in ('expected_prior_hash_overlap_questions','expected_internal_duplicate_hash_questions')):raise ValueError('Explicit canonical train hash-repeat policy and exact counts required')
    old,prior_refs=prior_membership(selection['prior_snapshots'],allow_hashes);seen=dict(qids=set(),hashes=set(),tids=set());selected=[];domains={}
    for index,item in enumerate(selection['selected_batches']):
        if item['input'] in old['inputs']:raise ValueError('Accepted prior batch repeated in new delta')
        inp=checked(item['input_ref']);seal=e.regular(base,base/item['seal'])
        if inp!=base/item['input'] or e.sha(seal)!=item['seal_sha256'] or e.sha(inp)!=item['input_sha256']:raise ValueError('Frozen input/seal differs')
        cfg=checked(item['config']);d.contract(e.read(cfg));auditfile=checked(item['audit']['proof'])
        if item['audit'].get('independent_sidecar_audit_root'):
            from .peer_newseed_source_audit_dispatch import proof
            fresh=proof(base,item,selection['source_config']['sha256'])
        else:fresh=d.cached_audit(base,item['input'],selection['source_config']['sha256'],item['input_sha256'],item['questions'])
        if fresh['status']!='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND' or fresh['proof']['sha256']!=item['audit']['proof']['sha256']:raise ValueError('Frozen independent audit no longer matches source fstats/own IDs')
        rows=e.read(inp)['rows'];own=[];qids=[];hashes=[];counts={}
        for row in rows:
            if row['split']!='train':raise ValueError('Nontraining source question')
            ts=[e.RUN_ID+'-outer000001-c%04d-g%d'%(row['candidate_ordinal'],j) for j in range(5)];own.append(dict(qid=row['qid'],question_hash=row['question_hash'],trajectory_ids=ts));qids.append(row['qid']);hashes.append(row['question_hash']);domain=json.loads(row['qid'])[0];counts[domain]=counts.get(domain,0)+1;domains[domain]=domains.get(domain,0)+1
        if len(rows)!=item['questions'] or digest(qids)!=item['ordered_qids_sha256'] or digest(hashes)!=item['ordered_question_hashes_sha256'] or counts!=item['domain_counts'] or digest([t for g in own for t in g['trajectory_ids']])!=item['expected_own_G5_IDs_sha256']:raise ValueError('Actual frozen identity differs')
        verify_id_digests([t for g in own for t in g['trajectory_ids']],item['expected_own_G5_IDs_sha256'],fresh['own_trajectory_ids_sha256'])
        identity_guard(old,own,seen,allow_hashes);selected.append(dict(input=item['input'],input_sha256=item['input_sha256'],input_stat=e.fs(inp),seal=item['seal'],seal_sha256=item['seal_sha256'],seal_stat=e.fs(seal),questions=item['questions'],trajectories=item['trajectories'],**({'audit_dispatch':item['audit']}if item['audit'].get('independent_sidecar_audit_root')else{})))
        e.atomic(root/'subset-preflight-progress.json',dict(stage='FROZEN_SUBSET_INPUT_AND_CURRENT_PROOF_ADVANCING',batches=index+1,questions=len(seen['qids']),trajectories=len(seen['tids']),epoch=time.time()))
    if len(seen['qids'])!=selection['questions'] or len(seen['tids'])!=selection['trajectories'] or domains!=selection['domain_counts']:raise ValueError('Actual frozen totals/domains differ')
    hash_counts=dict(prior_hash_overlap_questions=seen.get('prior_hash_overlap_questions',0),internal_duplicate_hash_questions=seen.get('internal_duplicate_hash_questions',0))
    if allow_hashes and (hash_counts['prior_hash_overlap_questions']!=policy['expected_prior_hash_overlap_questions'] or hash_counts['internal_duplicate_hash_questions']!=policy['expected_internal_duplicate_hash_questions']):raise ValueError('Actual canonical train text-repeat counts differ from frozen plan')
    return dict(canonical_train_text_hash_policy=policy,canonical_train_text_hash_repeats=hash_counts,source_base=str(base),source_node='h100-10',source_run_id=e.RUN_ID,seed=e.SEED,revision=e.REVISION,source_config=e.fp(config),input_manifest=e.fp(manifest),selected_batches=selected,unsealed_inputs_excluded=[],source_input_batches=1322,source_all_inputs_sealed=False,questions=selection['questions'],trajectories=selection['trajectories'],cohort_seal_digest=digest(selected),discovery_verified_payloads=False,selection_rule='EXACT published audited subset; no global discovery or label/probe selection',subset_selection=selref,subset_batch_records_sha256=selection['batch_records_sha256'],subset_exporter_sha256=e.sha(__file__),previous_imported_snapshots=prior_refs,out_of_subset_not_inspected=True,prior_identity_overlap=dict(batches=0,qids=0,question_hashes=len(seen['hashes']&old['hashes']),trajectory_ids=0))

def run(base,output,snapshot,selection_path,selection_sha):
    root=Path(output)/snapshot;root.mkdir(parents=True,exist_ok=True);frozen=freeze(base,selection_path,selection_sha,root)
    if (root/'plan.json').exists():
        prior=e.read(root/'plan.json')
        for key in ('subset_selection','subset_batch_records_sha256','subset_exporter_sha256','previous_imported_snapshots','prior_identity_overlap','canonical_train_text_hash_policy','canonical_train_text_hash_repeats'):
            if prior.get(key)!=frozen[key]:raise ValueError('Existing subset plan binding differs')
    e.immutable(root/'plan.json',dict(frozen,snapshot_id=snapshot,batches_per_archive=16,exporter_sha256=e.sha(e.__file__),auditor_sha256=e.AUDIT_SHA));old_discover=e.discover
    try:
        e.discover=lambda actual:frozen if Path(actual)==Path(base) else (_ for _ in ()).throw(ValueError('Source differs'));done=e.export(Path(base),Path(output),snapshot,batches_per_archive=16)
    finally:e.discover=old_discover
    if done['questions']!=frozen['questions'] or done['trajectories']!=frozen['trajectories'] or done['frozen_sealed_batches']!=len(frozen['selected_batches']):raise ValueError('Exported subset totals differ')
    result=dict(status=STATUS,snapshot_id=snapshot,selection=e.fp(selection_path),subset_record_sha256=frozen['subset_batch_records_sha256'],source_complete=e.fp(root/'complete.json'),source_plan=e.fp(root/'plan.json'),questions=frozen['questions'],trajectories=frozen['trajectories'],batches=len(frozen['selected_batches']),archives=len(done['groups']),archive_bytes=sum(g['archive_bytes'] for g in done['groups']),prior_identity_overlap=frozen['prior_identity_overlap'],canonical_train_text_hash_policy=frozen['canonical_train_text_hash_policy'],canonical_train_text_hash_repeats=frozen['canonical_train_text_hash_repeats'],all_archive_and_member_SHA_fresh=True,no_global_discovery=True,producer_source_modified=False,GPU_started=False,epoch=time.time())
    accepted=root/'subset-complete.json'
    if accepted.exists():
        previous=e.read(accepted)
        for key in ('selection','subset_record_sha256','source_complete','source_plan','questions','trajectories','batches','archives','archive_bytes'):
            if previous[key]!=result[key]:raise ValueError('Existing subset acceptance changed')
        return previous
    e.immutable(accepted,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256',required=True);p.add_argument('--snapshot-id',required=True);p.add_argument('--base',type=Path,default=BASE);p.add_argument('--output-root',type=Path,default=Path(e.OUTPUT));a=p.parse_args()
    if sys.platform=='darwin' or a.base!=BASE or a.output_root!=Path(e.OUTPUT):p.error('Remote canonical source/independent output only')
    print(json.dumps(run(a.base,a.output_root,a.snapshot_id,a.selection,a.selection_sha256)),flush=True)
if __name__=='__main__':main()
