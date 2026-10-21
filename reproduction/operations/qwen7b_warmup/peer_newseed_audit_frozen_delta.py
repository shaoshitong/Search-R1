"""Audit and freeze one finite donor delta outside the producer directory."""
import argparse, fcntl, hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
from types import SimpleNamespace
from . import peer_newseed_delta as d
from . import peer_newseed_freeze_after466 as f

POLICY = 'retain canonical distinct qids/authentic trajectory IDs; train-only repeated text hashes are reported; evaluation qid/hash exclusion remains hard'


def atomic(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        stream.write(d.canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def reference_contract(value):
    rows=value['delta_batches_metadata'];priors=value['prior_snapshots']
    if (value['status']!='FRESH_SOURCE_SEALED_METADATA_MINUS_ACCEPTED971'
        or value['accepted_prior_batches']!=sum(p['expected_batches']for p in priors)
        or value['accepted_prior_questions']!=sum(p['expected_questions']for p in priors)
        or value['delta_records_sha256']!=d.digest(rows) or not rows
        or len(rows)!=value['delta_batches'] or sum(r['questions']for r in rows)!=value['delta_questions']
        or sum(r['trajectories']for r in rows)!=value['delta_new5_trajectories']
        or any(r['trajectories']!=5*r['questions']for r in rows)
        or len({r['input']for r in rows})!=len(rows)):
        raise ValueError('Finite observed donor delta membership/count/SHA differs')
    return sorted(rows,key=lambda r:r['input'])


def run(reference_path,reference_sha256,exclusions_path,exclusions_sha256,output):
    if sys.platform=='darwin':raise ValueError('Real donor payload audit is remote-only')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if not root.is_absolute()or root.is_symlink()or str(root).startswith(str(d.BASE)+'/'):
        raise ValueError('Independent absolute sidecar output required')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    ref,rref=d.load(Path(reference_path));ex,eref=d.load(Path(exclusions_path))
    if rref['sha256']!=reference_sha256 or eref['sha256']!=exclusions_sha256:raise ValueError('Input reference SHA differs')
    rows=reference_contract(ref)
    if (ex['role']!='joint_training_fixed_evaluation_identity_exclusions'
        or ex['scope_counts']!={'diagnostic':512,'selection':2048,'final':4096}
        or ex['final_labels_or_rollouts_read']is not False or len(ex['groups'])!=6656):
        raise ValueError('Frozen evaluation identity-only exclusions differ')
    excluded_qids={g['qid']for g in ex['groups']};excluded_hashes={g['question_hash']for g in ex['groups']}
    cfg,cref=f.checked(ref['source_config']);d.contract(cfg);_,mref=f.checked(ref['input_manifest'])
    old,oldrefs=f.prior_membership(ref['prior_snapshots'])
    if len(old['inputs'])!=ref['accepted_prior_batches']or len(old['qids'])!=ref['accepted_prior_questions']:
        raise ValueError('Accepted prior source identity counts differ')
    source=Path(__file__).with_name('peer_newseed_audit_reference.py')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=d.AUDITOR:raise ValueError('Pinned CPU auditor differs')
    spec=importlib.util.spec_from_file_location('pinned_source_auditor',source);aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)
    with(root/'controller.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(root/'complete.json').exists():
            value,_=d.load(root/'complete.json')
            if value['source_reference']!=rref or value['evaluation_exclusions']!=eref:raise ValueError('Existing completion has other inputs')
            return value
        owner=dict(pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),epoch=time.time(),source_reference=rref)
        atomic(root/'controller.json',owner)
        a=aud.Audit(SimpleNamespace(base=str(d.BASE),audit_dir=str(root/'audit'),path_map=[],max_new_batches=0))
        a.root_config=cfg;a.root_config_hash=cref['sha256']
        selected=[];members=[];seen={k:set()for k in('qids','hashes','tids')};domains={};prior_hash=within_hash=0;started=time.time()
        try:
            for item in rows:
                inp=d.regular(d.BASE,item['input']);batch,iref=d.load(inp);_,sref=d.load(d.regular(d.BASE,item['seal']))
                if iref['sha256']!=item['input_sha256']or sref['sha256']!=item['seal_sha256']or item['input']in old['inputs']:
                    raise ValueError('Frozen source input/seal changed or previously accepted')
                proof=a.batch(inp,batch,iref['sha256'])
                if not proof or proof.get('deferred')or proof['questions']!=item['questions']or proof['labels']!=item['trajectories']:
                    raise ValueError('Fresh audit did not complete the entire authentic G5')
                _,after=d.load(d.regular(d.BASE,item['seal']))
                if after!=sref:raise ValueError('Producer seal changed across owned audit')
                audit=f.sidecar_audit(d.BASE,item['input'],cref['sha256'],iref['sha256'],item['questions'],root/'audit')
                qids=[];hashes=[];tids=[];own_domains={}
                for q in batch['rows']:
                    qid,h=q['qid'],q['question_hash'];ts=[d.RUN+'-outer000001-c%04d-g%d'%(q['candidate_ordinal'],j)for j in range(5)]
                    if(q['split']!='train'or qid in old['qids']or qid in seen['qids']or old['tids'].intersection(ts)or seen['tids'].intersection(ts)
                       or qid in excluded_qids or h in excluded_hashes):raise ValueError('Authentic train identity overlap or evaluation leakage')
                    prior_hash+=int(h in old['hashes']);within_hash+=int(h in seen['hashes'])
                    seen['qids'].add(qid);seen['hashes'].add(h);seen['tids'].update(ts);qids.append(qid);hashes.append(h);tids+=ts
                    domain=json.loads(qid)[0];domains[domain]=domains.get(domain,0)+1;own_domains[domain]=own_domains.get(domain,0)+1
                    members.append(dict(qid=qid,question_hash=h,trajectory_ids=ts,domain=domain,source_input=item['input'],source_seal=sref,
                        sampling_protocol_source_run_id=d.RUN,execution_source='donor_H100_original_sealed_G5'))
                if d.digest(qids)!=item['ordered_qids_sha256']or d.digest(tids)!=item['expected_own_G5_IDs_sha256']or d.digest(sorted(tids))!=audit['own_trajectory_ids_sha256']:
                    raise ValueError('Frozen ordered IDs or independently sorted own-G5 IDs differ')
                selected.append(dict(item,audit=audit,input_ref=iref,domain_counts=own_domains))
                atomic(root/'progress.json',dict(status='FROZEN_DONOR_DELTA_FRESH_SHA_CAUSAL_OWN_G5_AUDIT_ADVANCING',checked_batches=len(selected),
                    checked_questions=len(seen['qids']),checked_trajectories=len(seen['tids']),expected_batches=len(rows),expected_questions=ref['delta_questions'],
                    epoch=time.time(),elapsed_seconds=time.time()-started,source_reference=rref,strict_EM_tokenizer_recomputed=False,producer_modified=False))
            n=len(selected);q=len(members);baseline=ref['accepted_prior_batches']
            plan=dict(status='FROZEN_SOURCE_AUDITED%d_AFTER_ACCEPTED%d_METADATA_ONLY'%(n,baseline),schema='FINITE_DONOR_DELTA_AUDITED_SUBSET_V1',subset='existing-audited',
                source_base=str(d.BASE),source_run_id=d.RUN,seed=d.SEED,source_seed=1729,model_revision=d.REVISION,sampling_attempt_id=d.ATTEMPT,
                source_config=cref,input_manifest=mref,batches=n,questions=q,trajectories=5*q,domain_counts=domains,selected_batches=selected,batch_records_sha256=d.digest(selected),
                prior_snapshots=ref['prior_snapshots'],accepted_identity_membership_refs=oldrefs,prior_identity_overlap=dict(batches=0,qids=0,question_hashes=prior_hash,trajectory_ids=0),
                training_text_hash_policy=POLICY,within_selected_training_hash_duplicates=within_hash,selected_unique_question_hashes=len(seen['hashes']),
                evaluation_exclusions=eref,evaluation_identity_overlap=dict(qids=0,question_hashes=0),reference_observation=rref,
                selection_rule='all complete source G5 in this finite after971 observation; no labels or scores; no later source growth',
                existing_audit_all_actual_file_fstat_matches=True,existing_audit_payload_SHA_inherited_only=False,fresh_payload_SHA_recomputed=True,
                strict_EM_tokenizer_recomputed=False,future_export_must_revalidate_source_fstats_inputs_and_fresh_archive_member_SHA=True,
                HPC3_tail_same_qid_tid_does_not_imply_same_payload=True,whole_G5_source_selection_deferred_to_parent=True,
                original_warmup_question_unseen_claim=False,payload_export_started=False,GPU_started=False,producer_source_modified=False,created_epoch=time.time())
            atomic(root/'audited-plan.json',plan);atomic(root/'whole-G5-identities.json',dict(role='DONOR_FROZEN_DELTA_WHOLE_G5_IDENTITY_ONLY',groups=members,
                source_run_id=d.RUN,source_reference=rref,plan=d.load(root/'audited-plan.json')[1],labels_or_rollout_payload_included=False))
            result=dict(status='FINITE_DONOR_DELTA_FRESH_SHA_CAUSAL_OWN_G5_AUDIT_AND_FROZEN_PLAN_COMPLETE',source_reference=rref,evaluation_exclusions=eref,
                plan=d.load(root/'audited-plan.json')[1],whole_G5_identities=d.load(root/'whole-G5-identities.json')[1],proof_root=str(root/'audit'),
                batches=n,questions=q,trajectories=5*q,domain_counts=domains,prior_training_text_hash_overlap_questions=prior_hash,within_training_text_hash_duplicates=within_hash,
                evaluation_identity_overlap=dict(qids=0,question_hashes=0),source_auditor_sha256=d.AUDITOR,bulk_payload_SHA_recomputed=True,
                strict_EM_tokenizer_recomputed=False,producer_modified=False,payload_exported=False,GPU_used=False,epoch=time.time(),elapsed_seconds=time.time()-started)
            atomic(root/'complete.json',result);return result
        except BaseException as exc:
            atomic(root/'failure.json',dict(error=type(exc).__name__,message=str(exc),checked_batches=len(selected),checked_questions=len(seen['qids']),epoch=time.time(),source_reference=rref));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--reference-sha256',required=True);p.add_argument('--exclusions',required=True);p.add_argument('--exclusions-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.reference,a.reference_sha256,a.exclusions,a.exclusions_sha256,a.output)))

if __name__=='__main__':main()
