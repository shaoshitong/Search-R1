"""Freeze an explicit subset of one finite, SHA-bound donor metadata observation.

Only seal/config/input and identity/proof metadata are read. No payload is read
or exported, no live producer file/process is modified. Missing independent
proofs must be produced separately before those members can be frozen.
"""
import copy, fcntl, hashlib, json, os, time
from pathlib import Path
from . import peer_newseed_delta as d


def descriptor(path):
    _, ref = d.load(Path(path)); return ref


def checked(ref):
    value, actual = d.load(Path(ref['path']))
    if any(actual[k] != ref[k] for k in ('path','bytes','sha256')):
        raise ValueError('Immutable metadata descriptor differs: '+ref['path'])
    return value, actual


def prior_membership(priors):
    old = {k:set() for k in ('inputs','qids','hashes','tids')}; refs=[]
    for expect in priors:
        complete, cr = checked(expect['complete'])
        plan, pr = checked(complete['plan']); member, mr = checked(complete['canonical_membership'])
        rows, groups = plan['selected_batches'], member['groups']
        if (len(rows)!=expect['expected_batches'] or len(groups)!=expect['expected_questions']
            or complete['questions']!=len(groups) or complete['trajectories']!=5*len(groups)):
            raise ValueError('Prior snapshot membership/count differs')
        now=dict(inputs={r['input'] for r in rows},qids={g['qid'] for g in groups},
                 hashes={g['question_hash'] for g in groups},tids={t for g in groups for t in g['trajectory_ids']})
        if len(now['inputs'])!=len(rows) or len(now['qids'])!=len(groups) or len(now['tids'])!=5*len(groups):
            raise ValueError('Prior identities repeat')
        for k in old:
            if old[k] & now[k]: raise ValueError('Prior snapshots overlap: '+k)
            old[k].update(now[k])
        refs.append(dict(complete=cr,plan=pr,membership=mr))
    return old,refs


def sidecar_audit(base,rel,config_sha,input_sha,questions,audit_root):
    """Same pinned auditor/file-stat contract as donor cache, alternate own root."""
    inp=Path(rel);dest=base/'sampling'/inp.parent.name/'batches'/inp.stem;outer=dest/'outers/outer-000001'
    files=dict(batch_seal=dest/'complete.json',config=dest/'config.json',questions=outer/'questions.json',
        prefix_seal=outer/'prefix/complete.json',prefix_scan=outer/'prefix/scan.json',prefix_rank=outer/'prefix/rank-0-complete.json',
        full_seal=outer/'full/complete.json',full_rank=outer/'full/rank-0-complete.json',selection=outer/'probe-score/selection.json',
        prefix=outer/'prefix/rank-0-prefix.jsonl.gz',full=outer/'full/rank-0-full.jsonl.gz',own_em=outer/'full/rank-0-own-em.jsonl')
    root=Path(audit_root)
    if not root.is_absolute() or str(root).startswith(str(base)+'/') or root.is_symlink():
        raise ValueError('Separate real absolute own audit root required')
    proof_path=root/'batches'/inp.parent.name/(inp.stem+'.json')
    proof,ref=d.load(d.regular(root,str(proof_path.relative_to(root))))
    expected=dict(cache_version=1,auditor_sha256=d.AUDITOR,run_id=d.RUN,seed=d.SEED,root_config_sha256=config_sha,input_sha256=input_sha,
                  files={k:d.fstat(p) for k,p in files.items()})
    if proof.get('status')!='VERIFIED' or proof.get('binding')!=expected:
        raise ValueError('Independent sidecar proof is not currently fstat bound')
    counts=proof['proof']
    if any(counts.get(k)!=v for k,v in dict(questions=questions,trajectories=questions*5,labels=questions*5).items()):
        raise ValueError('Sidecar own-G5 counts differ')
    return dict(status='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND',proof=ref,verified_epoch=proof['verified_epoch'],
        own_trajectory_ids_sha256=counts['actual_trajectory_ids_sha256'],bulk_payload_SHA_recomputed=False,
        strict_EM_tokenizer_recomputed=False,independent_sidecar_audit_root=str(root))


def freeze(reference,priors,output,*,which='existing243',audit_root=None):
    if (reference['status']!='FRESH_SOURCE_SEALED_METADATA_MINUS_ACCEPTED466' or reference['accepted_prior_batches']!=466
        or reference['delta_batches']!=505 or reference['delta_questions']!=64546
        or d.digest(reference['delta_batches_metadata'])!=reference['delta_records_sha256']):
        raise ValueError('Exact finite parent505-after466 observation required')
    if which not in ('existing243','missing262'):raise ValueError('Explicit finite proof partition required')
    rows=[r for r in reference['delta_batches_metadata'] if (r['existing_audit']['status']=='VERIFIED')==(which=='existing243')]
    if len(rows)!=(243 if which=='existing243' else 262):raise ValueError('Parent partition membership differs')
    if which=='missing262' and not audit_root:raise ValueError('Missing262 requires completed independent sidecar proofs')
    output=Path(output)
    if not output.is_absolute() or str(output).startswith(str(d.BASE)+'/') or output.parent.is_symlink():raise ValueError('Independent absolute metadata destination required')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():
            v,ref=d.load(output)
            if v['reference_observation_sha256']!=d.digest(reference) or v['partition']!=which:raise ValueError('Existing plan is a different freeze')
            return dict(plan=ref,metadata_plan=v,existing_preserved=True)
        cfg,cref=checked(reference['source_config']);d.contract(cfg)
        manifest,mref=checked(reference['input_manifest'])
        if manifest['questions']!=169028 or len(manifest['batches'])!=1322:raise ValueError('Canonical source budget differs')
        old,oldrefs=prior_membership(priors)
        if len(old['inputs'])!=466 or len(old['qids'])!=59648 or len(old['tids'])!=298240:raise ValueError('Exact accepted466 identities required')
        selected=[];seen={k:set() for k in ('qids','hashes','tids')};domains={};prior_hash_overlap=0;within_hash_duplicates=0
        for r in rows:
            if r['input'] in old['inputs']:raise ValueError('Prior accepted input reselected')
            seal,sref=d.load(d.regular(d.BASE,r['seal']))
            batch,iref=d.load(d.regular(d.BASE,r['input']))
            if sref['sha256']!=r['seal_sha256'] or iref['sha256']!=r['input_sha256']:raise ValueError('Finite parent input/seal changed')
            inp=Path(r['input']);config,configref=d.load(d.regular(d.BASE,str(Path('sampling')/inp.parent.name/'batches'/inp.stem/'config.json')));d.contract(config)
            audit=(sidecar_audit(d.BASE,r['input'],cref['sha256'],iref['sha256'],r['questions'],audit_root) if audit_root else d.cached_audit(d.BASE,r['input'],cref['sha256'],iref['sha256'],r['questions']))
            if audit['status']!='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND':raise ValueError('No current independent proof: '+r['input'])
            if not audit_root and audit['proof']['sha256']!=r['existing_audit']['proof']['sha256']:raise ValueError('Parent selected proof changed')
            qids=[];hashes=[];tids=[];own_domains={}
            for q in batch['rows']:
                qid,h=q['qid'],q['question_hash'];ts=[d.RUN+'-outer000001-c%04d-g%d'%(q['candidate_ordinal'],j)for j in range(5)]
                if q['split']!='train' or qid in old['qids'] or old['tids'].intersection(ts):raise ValueError('New question/trajectory identity overlaps accepted466')
                prior_hash_overlap+=int(h in old['hashes']);within_hash_duplicates+=int(h in seen['hashes'])
                if qid in seen['qids'] or seen['tids'].intersection(ts):raise ValueError('New question/trajectory identities repeat')
                seen['qids'].add(qid);seen['hashes'].add(h);seen['tids'].update(ts);qids.append(qid);hashes.append(h);tids+=ts
                domain=json.loads(qid)[0];domains[domain]=domains.get(domain,0)+1;own_domains[domain]=own_domains.get(domain,0)+1
            if len(qids)!=r['questions'] or d.digest(sorted(tids))!=audit['own_trajectory_ids_sha256']:raise ValueError('Input-derived sorted own-G5 proof differs')
            selected.append(dict(input=r['input'],input_sha256=r['input_sha256'],seal=r['seal'],seal_sha256=r['seal_sha256'],
                questions=r['questions'],trajectories=r['trajectories'],input_ref=iref,config=configref,audit=audit,
                ordered_qids_sha256=d.digest(qids),ordered_question_hashes_sha256=d.digest(hashes),expected_own_G5_IDs_sha256=d.digest(tids),domain_counts=own_domains))
        selected.sort(key=lambda r:r['input']);n=len(selected);q=len(seen['qids'])
        v=dict(status='FROZEN_SOURCE_AUDITED%d_AFTER_ACCEPTED466_METADATA_ONLY'%n,schema='FINITE_PARENT505_AUDITED_SUBSET_V1',subset='existing-audited',
            source_base=str(d.BASE),source_run_id=d.RUN,seed=d.SEED,source_seed=1729,model_revision=d.REVISION,sampling_attempt_id=d.ATTEMPT,
            source_config=cref,input_manifest=mref,batches=n,questions=q,trajectories=5*q,domain_counts=domains,selected_batches=selected,
            batch_records_sha256=d.digest(selected),prior_snapshots=priors,accepted466_identity_membership_refs=oldrefs,
            prior_identity_overlap=dict(batches=0,qids=0,question_hashes=prior_hash_overlap,trajectory_ids=0),partition=which,
            training_text_hash_policy='retain canonical distinct qids/authentic trajectory IDs; train-only repeated text hashes are reported; evaluation qid/hash exclusion remains hard',
            within_selected_training_hash_duplicates=within_hash_duplicates,selected_unique_question_hashes=len(seen['hashes']),
            parent505_records_sha256=reference['delta_records_sha256'],reference_observation_sha256=d.digest(reference),
            selection_rule='Exact finite parent505 difference after accepted466; fixed existing243 vs missing262 proof partition; no labels/scores',
            existing_audit_all_actual_file_fstat_matches=True,existing_audit_payload_SHA_inherited_only=True,fresh_payload_SHA_recomputed=False,
            strict_EM_tokenizer_recomputed=False,future_export_must_revalidate_source_fstats_inputs_and_fresh_archive_member_SHA=True,
            original_warmup_question_unseen_claim=False,payload_export_started=False,GPU_started=False,producer_source_modified=False,created_epoch=time.time())
        tmp=output.with_suffix('.tmp')
        with tmp.open('xb')as f:f.write(d.canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,output);actual,ref=d.load(output)
        if actual!=v:raise ValueError('Plan atomic readback differs')
        return dict(plan=ref,metadata_plan=v,existing_preserved=False)
