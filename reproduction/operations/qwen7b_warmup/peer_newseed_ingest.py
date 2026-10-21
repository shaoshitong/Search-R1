"""CPU-only acceptance of received authentic old5 + newseed5 as raw G10.

A target receiver complete is mandatory. No model tensors/hidden features are
loaded or created. Final exclusions are identity-only and never dereferenced.
An accepted raw bank is NOT a feature bank, a fitted probe, or an untouched
question holdout: the original warmup saw all 169028 canonical train questions.
"""
import argparse
import fcntl
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from . import peer_newseed_receive as receiver
from . import peer_newseed_audit_reference as audit
from .peer_newseed_export import AUDIT_SHA, RUN_ID, SEED, ATTEMPT, SOURCE, RECORDED

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
REVISION=receiver.REVISION
STATUS='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'
SCOPES={'diagnostic512':512,'selection2048':2048,'frozen_final4096':4096}
PROTOCOL=('sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype')


def read(path): return json.loads(receiver.regular(path).read_bytes())
def canonical(value): return receiver.canonical(value)
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def ref(path):
    path=receiver.regular(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=receiver.sha(path))
def checked(item):
    path=receiver.regular(item['path']); receiver.verify_file(path,item); return path
def atomic(path,value): return receiver.atomic(path,value)
def immutable(path,value): return receiver.immutable(path,value)
def rows(path): return audit.rows(Path(path))


def identity_rows(values,expected=None):
    """Strict identity-only records. Never follow any source_refs."""
    seen={};hashes=set()
    for row in values:
        if set(row)!={'qid','question_hash'}: raise ValueError('Identity rows must contain only qid/question_hash')
        qid,qhash=row['qid'],row['question_hash']
        if (not isinstance(qid,str) or not qid or qid in seen or
                not isinstance(qhash,str) or len(qhash)!=64 or any(c not in '0123456789abcdef' for c in qhash)):
            raise ValueError('Invalid/duplicate canonical exclusion identity')
        seen[qid]=qhash;hashes.add(qhash)
    if expected is not None and len(seen)!=expected: raise ValueError('Exclusion scope count differs')
    return seen,hashes


def exclusion_sets(manifests,counts=SCOPES):
    if set(manifests)!=set(counts): raise ValueError('Exactly diagnostic512/selection2048/frozen_final4096 identities required')
    out={};allq={};allh=set()
    for role in sorted(counts):
        value=manifests[role]
        if value.get('role')!=role or value.get('identity_only') is not True:
            raise ValueError('Explicit role-bound identity-only exclusions required')
        own,hashes=identity_rows(value['groups'],counts[role])
        # Separate roles may alias text; report/exclude hashes, but qid ownership
        # must be unambiguous. No exclusions are applied to current fit6528.
        for qid,qhash in own.items():
            if qid in allq: raise ValueError('Evaluation identity scopes overlap')
            allq[qid]=qhash
        allh.update(hashes);out[role]=dict(questions=len(own),unique_question_hashes=len(hashes))
    return allq,allh,out


def final_identity_projection(value):
    """Accept only the actual frozen identity schema, never labels/source data."""
    if value.get('role')!='final' or value.get('frozen') is not True or value.get('labels_read') is not False:
        raise ValueError('Frozen identity-only final exclusion plan required')
    allowed={'qid','question_hash','canonical_question_sha256','pool'}
    result=[]
    for row in value['questions']:
        if not {'qid','question_hash'}<=set(row) or not set(row)<=allowed:
            raise ValueError('Final plan row contains non-identity fields')
        result.append(dict(qid=row['qid'],question_hash=row['question_hash']))
    identity_rows(result,4096)
    expected=hashlib.sha256(json.dumps([g['qid'] for g in result],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if value.get('ordered_qid_sha256')!=expected:raise ValueError('Frozen final identity plan SHA differs')
    return result


def own_group(group,g=5):
    tids=group['trajectory_ids'];labels=group['labels']
    if (not isinstance(group['qid'],str) or not group['qid'] or not isinstance(group['question_hash'],str) or
            len(group['question_hash'])!=64 or len(tids)!=g or len(set(tids))!=g or
            any(not isinstance(t,str) or not t for t in tids) or len(labels)!=g or
            any(type(y) is not int or y not in (0,1) for y in labels)):
        raise ValueError('Authentic own IDs and exact binary own-EM labels required')
    return group


def qualify_membership(old,new,exclusions,counts=SCOPES):
    excluded,excluded_hashes,scopes=exclusion_sets(exclusions,counts)
    original={g['qid']:g for g in old}
    if len(original)!=len(old): raise ValueError('Canonical original train qids repeat')
    accepted=[];seen=set();rejected={role:0 for role in counts};hash_excluded=0;seen_tids=set()
    role_qids={role:{g['qid'] for g in value['groups']} for role,value in exclusions.items()}
    for row in new:
        qid=row['qid']
        if qid in seen: raise ValueError('New snapshot canonical question repeats')
        seen.add(qid)
        source=original.get(qid)
        if source is None or source['question_hash']!=row['question_hash']:
            raise ValueError('New question/hash is not canonical original training data')
        if 'source_ordinal' in row and row['source_ordinal']!=source['source_ordinal']:
            raise ValueError('New canonical source ordinal differs')
        matched=[role for role,ids in role_qids.items() if qid in ids]
        if matched:
            if excluded[qid]!=row['question_hash']: raise ValueError('Excluded qid has conflicting canonical hash')
            for role in matched: rejected[role]+=1
            continue
        if row['question_hash'] in excluded_hashes: hash_excluded+=1;continue
        own_group(source);own_group(row)  # Only AFTER identity exclusions; no excluded label inspection.
        tids=source['trajectory_ids']+row['trajectory_ids']
        if len(set(tids))!=10 or seen_tids.intersection(tids): raise ValueError('Old/new actual trajectory namespaces collide')
        seen_tids.update(tids)
        accepted.append(dict(qid=qid,question_hash=row['question_hash'],source_ordinal=source['source_ordinal'],
            trajectory_ids=tids,labels=source['labels']+row['labels'],
            real_G5_events=[dict(event='original',trajectory_ids=source['trajectory_ids'],labels=source['labels']),
                dict(event='newseed20261021',trajectory_ids=row['trajectory_ids'],labels=row['labels'])]))
    accepted.sort(key=lambda g:g['source_ordinal'])
    return accepted,dict(snapshot_questions=len(new),accepted_questions=len(accepted),
        excluded_by_qid=rejected,additional_excluded_by_question_hash=hash_excluded,scopes=scopes,
        original_warmup_saw_all_169028_training_questions=True,new_question_holdout_claim=False,
        matched_fit6528_not_an_exclusion=True,final_source_refs_opened=False,
        accepted_unique_question_hashes=len({g['question_hash'] for g in accepted}),
        accepted_duplicate_hash_questions=len(accepted)-len({g['question_hash'] for g in accepted}))


def receiver_gate(root,*,fresh_files):
    """Check genuine target acceptance before any raw payload or original scan."""
    root=Path(root);done=read(root/'complete.json');catalog=read(root/'catalog.json');binding=read(root/'catalog-binding.json')
    receiver.validate_catalog(catalog,root.name,binding['prefix'])
    if (hashlib.sha256(binding['catalog_bytes'].encode()).hexdigest()!=binding['catalog_sha256'] or
            json.loads(binding['catalog_bytes'])!=catalog or done.get('status')!=receiver.STATUS or
            done.get('snapshot_id')!=root.name or done.get('target')!=str(root/'raw') or
            done.get('catalog_sha256')!=binding['catalog_sha256'] or done.get('catalog_commit')!=binding['catalog_commit'] or
            not done.get('all_archive_SHA256') or not done.get('fresh_all_file_SHA256') or
            done.get('feature_bank_created') is not False or done.get('actor_updates')!=0 or done.get('probe_updates')!=0 or
            done.get('groups')!=len(catalog['groups']) or done.get('questions')!=catalog['source_snapshot']['questions'] or
            done.get('trajectories')!=5*done['questions']):
        raise ValueError('Actual complete raw HPC3 acceptance required; an upload receipt is insufficient')
    expected_files=sum(len(g['files']) for g in catalog['groups'])
    if done['files']!=expected_files: raise ValueError('Receiver all-file count differs')
    for group in catalog['groups']:
        name=receiver.group_name(group['group']);proof=read(root/'accepted'/(name+'.json'))
        if (proof.get('status')!='TARGET_ARCHIVE_AND_ALL_MEMBER_FILES_FRESH_SHA_VERIFIED' or
                proof.get('catalog_sha256')!=binding['catalog_sha256'] or proof.get('archive_sha256')!=group['archive_sha256'] or
                proof.get('archive_bytes')!=group['archive_bytes'] or proof.get('files')!=len(group['files'])):
            raise ValueError('Target group fresh acceptance differs')
        if fresh_files:
            receiver.verify_file(root/'archives'/Path(group['archive']).name,receiver.archive_record(group))
            for item in group['files']: receiver.verify_file(root/'raw'/item['path'],item)
    return done,catalog,binding


def compare_protocol(old,new):
    if any(key not in old or key not in new or old[key]!=new[key] for key in PROTOCOL):
        raise ValueError('Old/new actual sampling and interaction protocols differ')
    return {key:old[key] for key in PROTOCOL}


def audit_trajectory(prefix,full,source,config,own_label,score,ledger,event):
    """Replay actual causal token/context/chunk positions; never infer a label."""
    tid=full['trajectory_id'];qid=source['qid'];run=config['run_id'];attempt=config.get('sampling_attempt_id')
    actor='official7b-'+REVISION;index=full['response_index']
    if type(index) is not int or not 0<=index<5: raise ValueError('Each real event has actual slots0..4')
    wanted='%s-outer000001-c%04d-g%d'%(run,source['candidate_ordinal'],index)
    for row in (prefix,full):
        for key,value in dict(trajectory_id=wanted,qid=qid,question_hash=source['question_hash'],run_id=run,outer=1,
                split='train',response_index=index,actor_version=actor,candidate_ordinal=source['candidate_ordinal'],
                initial_prompt_token_ids=source['initial_prompt_token_ids']).items():
            if row.get(key)!=value: raise ValueError('Actual own trajectory/prompt identity differs: '+key)
    claimed=prefix['prefix_sha256']
    # Native capture appends feature_* locators after sealing the causal prefix.
    # Qualified replay normalizes the same exact sealed body; never hash locators.
    material={key:value for key,value in prefix.items() if key!='prefix_sha256' and not key.startswith('feature_')}
    if digest(material)!=claimed or prefix.get('future_information_included') is not False:
        raise ValueError('Prefix SHA/future-information contract differs')
    if (full.get('prefix_sha256')!=claimed or full.get('policy_snapshot_id')!=actor or
            full.get('reward_model')!=source['reward_model'] or full.get('data_source')!=source['data_source'] or
            type(full.get('strict_em')) is not int or full['strict_em']!=own_label or own_label not in (0,1)):
        raise ValueError('Full own EM/actor/gold/prefix binding differs')
    prompt=audit.token_ids(source['initial_prompt_token_ids'],'prompt',nonempty=True)
    rebuilt=[];mask=[];logs=[];journal=[];retained=[];positions=[];segments=[];seeds=[]
    for turn in full['behavior_turns']:
        request=audit.token_ids(turn['actual_request_token_ids'],'request',nonempty=True)
        raw=audit.token_ids(turn['raw_generated_token_ids'],'raw')
        lp=audit.log_probs(turn['selected_logprobs'],'raw');kept=turn['kept_token_count']
        if (request!=prompt+rebuilt or type(kept) is not int or not 0<=kept<=len(raw)==len(lp) or
                turn['trajectory_id']!=tid or turn['policy_snapshot_id']!=actor or
                turn['prefix_cut'].get('kept_token_ids')!=raw[:kept] or turn['prefix_cut'].get('kept_token_count')!=kept):
            raise ValueError('Actual causal turn request/cut differs')
        chunkids=[];chunklp=[];chunks=turn['actual_sampling_chunks']
        if not chunks: raise ValueError('Actual native chunks required')
        for chunk in chunks:
            t,c=chunk['logical_turn_index'],chunk['native_chunk_index']
            expected_seed=audit.seed_for(run,tid,t,c,config['seed'],attempt)
            if (chunk['actual_request_token_ids']!=request+chunkids or t!=turn['turn_index'] or
                    chunk['request_id']!='%s-t%d-chunk%d'%(tid,t,c) or chunk['seed']!=expected_seed or
                    chunk.get('sampling_attempt_id')!=attempt or chunk.get('sampling',{}).get('seed')!=expected_seed):
                raise ValueError('Actual native chunk context/seed differs')
            ids=audit.token_ids(chunk['raw_generated_token_ids'],'chunk');pl=audit.log_probs(chunk['selected_logprobs'],'chunk')
            if len(ids)!=len(pl): raise ValueError('Actual native chunk logprobs differ')
            chunkids.extend(ids);chunklp.extend(pl);seeds.append(expected_seed)
        if chunkids!=raw or chunklp!=lp: raise ValueError('Native chunks do not reconstruct turn')
        journal.extend(chunks);take=min(kept,20-len(retained))
        if take:
            segments.append(dict(turn_index=turn['turn_index'],policy_start=len(retained),policy_length=take,
                request_length=len(request),position_start=len(request),context_tokens=request+raw[:take]))
            retained.extend(raw[:take]);positions.extend(range(len(request),len(request)+take))
        observation=audit.token_ids(turn['observation_token_ids'],'observation');correction=audit.token_ids(turn['correction_token_ids'],'correction')
        if observation and correction: raise ValueError('Both environment suffixes present')
        env=observation+correction;rebuilt.extend(raw[:kept]+env);mask.extend([1]*kept+[0]*len(env));logs.extend(lp[:kept]+[0.]*len(env))
    if (not full['behavior_turns'] or journal!=full['sampling_journal'] or rebuilt!=full['response_token_ids'] or
            mask!=full['response_info_mask'] or logs!=full['response_log_probs']):
        raise ValueError('Actual full response/mask/journal does not reconstruct')
    if (prefix['policy_token_ids']!=retained or prefix['policy_position_ids']!=positions or
            prefix['prefix_segments']!=segments or prefix['retained_policy_length']!=len(retained) or
            len(retained)>20 or (len(retained)<20 and prefix['terminal'] is not True)):
        raise ValueError('Prefix20 causal contexts/positions differ')
    if full['terminal_reason'] not in ('natural_eos','answer','final_action_budget'):
        raise ValueError('Missing actual full terminal')
    if prefix['terminal'] and prefix['terminal_reason']!=full['terminal_reason']:
        raise ValueError('Prefix natural terminal differs from continuation')
    if score(full)!=own_label: raise ValueError('Tokenizer strict EM recomputation differs')
    own=ledger.setdefault(qid,{})
    for seed in seeds:
        if seed in own: raise ValueError('Actual same-question original/new native seed overlap')
        own[seed]=dict(event=event,trajectory_id=tid)
    return dict(seed_chunks=len(seeds),policy_tokens=sum(mask),response_tokens=len(rebuilt),
        multi_segment_prefix=int(len(segments)>1),short_terminal_prefix=int(len(retained)<20))


def prepare(receiver_root,output,original=B,selection=None,final_plan=None):
    """Small CPU binding only. Absent target complete => NOT_READY, no output."""
    root,output,original=map(Path,(receiver_root,output,original))
    if not (root/'complete.json').is_file(): return dict(status='NOT_READY_TARGET_RAW_RECEIVER_COMPLETE_ABSENT',receiver=str(root),prepared=False)
    done,catalog,binding=receiver_gate(root,fresh_files=False)
    c=read(original/'config.json');bank=read(original/'bank/groups.json');proof=read(original/'bank/complete.json')
    if (proof['status']!='FULL_CANONICAL_7B_G5_NATIVE_FEATURES_AND_OWN_EM_VERIFIED' or proof['actor_updates']!=0 or
            proof['model_revision']!=REVISION or receiver.sha(original/'bank/groups.json')!=proof['groups']['sha256'] or
            c['model_revision']!=REVISION or c['actor_optimizer_updates']!=0): raise ValueError('Original canonical bank is not accepted frozen initial7B')
    selection=Path(selection or original/'selection-2048-20261020/bank/groups.json')
    final_plan=Path(final_plan or original/'algorithm-improvement-20261021/training-extraG5-matched6528-20261021/final-exclusion-plan.json')
    val=read(selection);selected_proof=read(selection.parent/'complete.json');final=read(final_plan)
    if receiver.sha(selection)!=selected_proof['groups_sha256']:raise ValueError('Accepted selection2048 bank SHA differs')
    final_ids=final_identity_projection(final)
    values={'diagnostic512':bank['diagnostic'],'selection2048':val['selection_validation'],'frozen_final4096':final_ids}
    sources={'diagnostic512':original/'bank/groups.json','selection2048':selection,'frozen_final4096':final_plan}
    exclusions={role:dict(role=role,identity_only=True,groups=[dict(qid=g['qid'],question_hash=g['question_hash']) for g in values[role]],
        source_ref=ref(sources[role]),source_refs_followed=False) for role in SCOPES}
    exclusion_sets(exclusions)
    if len(bank['train'])!=169028: raise ValueError('Entire original canonical training membership required')
    config=dict(receiver=str(root),output=str(output),model_revision=REVISION,original=str(original),
        original_config=ref(original/'config.json'),original_bank=ref(original/'bank/groups.json'),
        original_complete=ref(original/'bank/complete.json'),original_inputs=ref(original/'inputs/manifest.json'),
        canonical_train=ref(c['canonical_train']),actor_complete=ref(Path(c['model_dir'])/'complete.json'),
        receiver_complete=ref(root/'complete.json'),receiver_catalog=ref(root/'catalog.json'),
        receiver_binding=ref(root/'catalog-binding.json'),exclusions=exclusions,
        ingester_sha256=receiver.sha(__file__),pinned_peer_auditor_sha256=AUDIT_SHA,
        original_warmup_saw_all_169028_training_questions=True,actor_updates=0,probe_updates=0)
    if config['canonical_train']['sha256']!=c['input_sha256']['train']: raise ValueError('Canonical train source SHA differs')
    if output==root or output==original or str(output).startswith(str(root)+'/') or str(output).startswith(str(original/'bank')+'/'):
        raise ValueError('Independent ingest output required')
    receiver.secure_dir(output);immutable(output/'config.json',config)
    for role,value in exclusions.items(): immutable(output/'identity-exclusions'/(role+'.json'),value)
    result=dict(status='CPU_RAW_G10_INGEST_PREPARED_NOT_INGESTED',config=ref(output/'config.json'),
        snapshot_id=root.name,questions=done['questions'],trajectories=done['trajectories'],feature_bank_created=False,submitted=False)
    immutable(output/'preparation.json',result);return result


def _canonical_parquet(config,bank):
    import pyarrow.parquet as pq
    from reproduction.prefix20_online.pool import identity
    rows_=pq.read_table(checked(config['canonical_train'])).to_pylist()
    if len(rows_)!=169028 or len(bank)!=169028: raise ValueError('Canonical original train169028 required')
    old={g['qid']:g for g in bank};seen=set()
    for ordinal,row in enumerate(rows_):
        own=identity(row);qid=own['qid']
        if qid in seen or qid not in old or old[qid]['question_hash']!=own['question_hash'] or old[qid]['source_ordinal']!=ordinal:
            raise ValueError('Original B bank does not match fresh canonical train parquet identities')
        seen.add(qid)
    return len(seen)


def _tokenizer(config,old_config):
    proof=read(checked(config['actor_complete']))
    if proof.get('status')!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' or proof.get('revision')!=REVISION:
        raise ValueError('Accepted official7B tokenizer source required')
    model=Path(old_config['model_dir']);tokenizer_files={}
    for name,item in proof['files'].items():
        if name.endswith(('.json','.txt','.model')):
            if Path(name).name!=name: raise ValueError('Tokenizer source path escapes official model')
            path=model/name;receiver.verify_file(path,item);tokenizer_files[name]=ref(path)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    from reproduction.prefix20_online.rollout import strict_label
    score=lambda row:strict_label(tokenizer,row['response_token_ids'],row['response_info_mask'],row['reward_model']['ground_truth'])
    return score,dict(files=tokenizer_files,scorer=ref(inspect.getfile(strict_label)),model_tensor_loaded=False)


def ingest(config_path):
    c=read(config_path);output=Path(c['output']);root=Path(c['receiver']);original=Path(c['original'])
    if (sys.platform=='darwin' or not str(output).startswith('/data/user/') or
            output==root or output==original or str(output).startswith(str(root)+'/')):
        raise ValueError('Independent remote-only CPU ingest output required')
    if c['ingester_sha256']!=receiver.sha(__file__) or receiver.sha(audit.__file__)!=AUDIT_SHA:
        raise ValueError('Bound published ingest/deep-auditor source differs')
    for key in ('original_config','original_bank','original_complete','original_inputs','canonical_train','actor_complete',
            'receiver_complete','receiver_catalog','receiver_binding'):checked(c[key])
    receiver.secure_dir(output)
    with (output/'ingest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'complete.json').exists():
            done=read(output/'complete.json')
            for item in done['outputs'].values():checked(item)
            if done['config']!=ref(config_path):raise ValueError('Existing acceptance belongs to different config')
            return done
        started=time.monotonic()
        try:
            done,catalog,binding=receiver_gate(root,fresh_files=False)
            old_c=read(checked(c['original_config']));old_bank=read(checked(c['original_bank']));_canonical_parquet(c,old_bank['train'])
            old={g['qid']:g for g in old_bank['train']};new={};group_roots=[];source_refs=[];allnew=set()
            for group in catalog['groups']:
                prefix=root/'raw'/root.name/receiver.group_name(group['group']);cohort=read(prefix/'cohort.json')
                if (cohort.get('source_run_id')!=RUN_ID or cohort.get('seed')!=SEED or cohort.get('sampling_attempt_id')!=ATTEMPT or
                        cohort.get('revision')!=REVISION or cohort.get('prefix_tokens')!=20 or cohort.get('no_label_selection') is not True or
                        cohort.get('hidden_exported') is not False or cohort['questions']!=group['questions'] or cohort['trajectories']!=5*group['questions']):
                    raise ValueError('Received raw source cohort contract differs')
                group_roots.append((prefix,group));source_refs.append(ref(prefix/'cohort.json'))
                for g in cohort['groups']:
                    # Membership is identity-only here; excluded own labels are
                    # not validated/used before the exclusion boundary.
                    if g['qid'] in new or allnew.intersection(g['trajectory_ids']):raise ValueError('Frozen source cohort repeats identity')
                    new[g['qid']]=g;allnew.update(g['trajectory_ids'])
            if len(new)!=done['questions']:raise ValueError('Received cohort totals differ')
            joined,overlap=qualify_membership(old_bank['train'],list(new.values()),c['exclusions'])
            if not joined:raise ValueError('No eligible training-only raw G10 question remains')
            wanted={g['qid'] for g in joined};old_keep={q:old[q] for q in wanted};new_keep={q:new[q] for q in wanted};score,tokenizer_proof=_tokenizer(c,old_c);prompts={};old_batches=[];protocol=None;ledger={}
            counters=dict(trajectories=0,seed_chunks=0,policy_tokens=0,response_tokens=0,multi_segment_prefix=0,short_terminal_prefix=0)
            def process(source_base,inp,batch,conf,event,keep,*,recorded=None,metadata_root=None):
                nonlocal protocol
                dest=source_base/'sampling'/inp.parent.name/'batches'/inp.stem;outer=dest/'outers/outer-000001'
                seal=read(dest/'complete.json');prefix_receipt=seal['prefix'];full_receipt=seal['full']
                if seal.get('run_id')!=conf['run_id'] or seal.get('input_sha256')!=receiver.sha(inp) or seal.get('actor_optimizer_updates')!=0:
                    raise ValueError('Actual batch actor/input/run seal differs')
                ownprotocol=compare_protocol(prefix_receipt,full_receipt)
                if protocol is None:protocol=ownprotocol
                else:compare_protocol(protocol,ownprotocol)
                def artifact(item):
                    p=Path(item['path'])
                    if recorded:
                        p=next((source_base/str(p)[len(oldpath)+1:] for oldpath in recorded if str(p).startswith(oldpath+'/')),None)
                        if p is None:raise ValueError('Received raw ref escapes its frozen source namespace')
                    receiver.verify_file(p,item);return p
                prefix_path=artifact(prefix_receipt['prefix']);full_path=artifact(full_receipt['full']);em_path=artifact(full_receipt['own_em'])
                scan=outer/'prefix/scan.json';selection=outer/'probe-score/selection.json'
                scan_sha=receiver.sha(scan);selection_sha=receiver.sha(selection)
                stable_paths=(inp,dest/'complete.json',prefix_path,full_path,em_path,scan,selection)
                before={str(path):audit.stat(path) for path in stable_paths}
                old_scan_sha,old_selection_sha=scan_sha,selection_sha;attested=None
                if full_receipt.get('prefix_scan_sha256')!=scan_sha or full_receipt.get('selection_sha256')!=selection_sha:
                    if event!='newseed20261021' or metadata_root is None:raise ValueError('Full prefix-scan/selection seal differs')
                    from .peer_newseed_target_attestation import validate
                    attested=validate(metadata_root,str(inp.relative_to(source_base)),source_base,conf,batch,prefix_receipt,full_receipt,scan_sha,selection_sha)
                    old_scan_sha,old_selection_sha=attested['old_scan_sha'],attested['old_selection_sha']
                by={r['qid']:r for r in batch['rows']};expected={tid:(qid,label) for qid,g in keep.items() if qid in by for tid,label in zip(g['trajectory_ids'],g['labels'])}
                def own_records(path):
                    result={}
                    for row in rows(path):
                        tid=row['trajectory_id']
                        if tid not in expected:continue
                        if tid in result:raise ValueError('Actual selected own record repeats')
                        result[tid]=row
                    return result
                prefixes=own_records(prefix_path);labels=own_records(em_path);seen=set()
                if set(prefixes)!=set(expected) or set(labels)!=set(expected):raise ValueError('Actual own prefix/EM membership missing')
                for row in rows(full_path):
                    tid=row['trajectory_id']
                    if tid not in expected:continue
                    if tid in seen:raise ValueError('Actual full trajectory repeats')
                    seen.add(tid);qid,label=expected[tid];source=by[qid]
                    if qid not in prompts:prompts[qid]=source
                    elif any(prompts[qid].get(k)!=source.get(k) for k in ('qid','question_hash','initial_prompt_token_ids','reward_model','data_source','source_ordinal','candidate_ordinal')):
                        raise ValueError('Old/new actual question/token prompt/gold differs')
                    projected={key:row[key] for key in ('trajectory_id','qid','strict_em','actor_version','prefix_sha256','prefix_scan_sha256','selection_sha256')}
                    if labels[tid]!=projected or row['prefix_scan_sha256']!=old_scan_sha or row['selection_sha256']!=old_selection_sha:
                        raise ValueError('Actual own EM or scan/selection linkage differs')
                    result=audit_trajectory(prefixes[tid],row,source,conf,label,score,ledger,event)
                    counters['trajectories']+=1
                    for key,value in result.items():counters[key]+=value
                if seen!=set(expected):raise ValueError('Actual full trajectory membership missing')
                if before!={str(path):audit.stat(path) for path in stable_paths}:raise ValueError('Sealed raw source changed during acceptance')
                return dict(event=event,input=ref(inp),seal=ref(dest/'complete.json'),prefix=ref(prefix_path),full=ref(full_path),own_em=ref(em_path),questions=len({q for q,_ in expected.values()}),**({'metadata_resealed_attestation':attested['descriptor'],'original_receipt_chain_equal_current':False}if attested else{}))
            manifest=read(checked(c['original_inputs']))
            for item in manifest['batches']:
                if item.get('split')!='train':continue
                inp=Path(item['path']);batch=read(inp)
                if not wanted.intersection(r['qid'] for r in batch['rows']):continue
                if receiver.sha(inp)!=item['sha256']:raise ValueError('Original token-input source SHA differs')
                proof=process(original,inp,batch,old_c,'original',old_keep)
                old_batches.append(proof)
                atomic(output/'progress.json',dict(stage='ORIGINAL_RAW_G5_STRICT_EM_CAUSAL_SEED_AUDIT',**counters,epoch=time.time()))
            new_batches=[]
            for prefix,group in group_roots:
                source_base=prefix/'source';conf=read(source_base/'config.json')
                if conf.get('run_id')!=RUN_ID or conf.get('seed')!=SEED or conf.get('sampling_attempt_id')!=ATTEMPT or conf.get('model_revision')!=REVISION:
                    raise ValueError('Received source config differs')
                for inp in sorted((source_base/'inputs').glob('shard-*/batch*.json')):
                    batch=read(inp)
                    if not wanted.intersection(r['qid'] for r in batch['rows']):continue
                    proof=process(source_base,inp,batch,conf,'newseed20261021',new_keep,recorded=(SOURCE,RECORDED),metadata_root=prefix)
                    new_batches.append(proof)
                    atomic(output/'progress.json',dict(stage='NEWSEED_RAW_G5_STRICT_EM_CAUSAL_SEED_AUDIT',**counters,epoch=time.time()))
            if counters['trajectories']!=10*len(joined) or set(prompts)!=wanted:raise ValueError('Incomplete accepted raw G10 audit')
            hist=[0]*11;oldhist=[0]*6;newhist=[0]*6
            for group in joined:
                hist[sum(group['labels'])]+=1;oldhist[sum(group['labels'][:5])]+=1;newhist[sum(group['labels'][5:])]+=1
            immutable(output/'raw-g10/groups.json',dict(train=joined));immutable(output/'raw-g10/source-events.json',dict(events=[dict(event='original',batches=old_batches),dict(event='newseed20261021',batches=new_batches)],repacked=False,hidden_present=False))
            immutable(output/'raw-g10/overlap.json',overlap)
            result=dict(status=STATUS,role='train',config=ref(config_path),receiver_complete=c['receiver_complete'],receiver_catalog=c['receiver_catalog'],
                questions=len(joined),trajectories=10*len(joined),real_trajectories_per_question=10,outputs={name:ref(output/('raw-g10/'+name+'.json')) for name in ('groups','source-events','overlap')},
                receiver_fresh_all_archives_and_members_SHA_inherited=True,ingest_used_raw_payload_SHA_fresh=True,
                snapshot_id=root.name,source_run_id=RUN_ID,source_seed=SEED,source_attempt=ATTEMPT,original_run_id=old_c['run_id'],original_seed=old_c['seed'],
                source_cohorts=source_refs,tokenizer=tokenizer_proof,protocol=protocol,histogram_K0_to10=hist,old5_histogram_K0_to5=oldhist,new5_histogram_K0_to5=newhist,
                mixed_G10_questions=sum(hist[1:10]),own_strict_EM_recomputed=counters['trajectories'],strict_EM_mismatches=0,
                actual_same_question_old_new_seed_collisions=0,seed_namespace_sha256=digest({q:sorted(ledger[q]) for q in sorted(ledger)}),
                raw_context_audit=counters,model_tensor_loaded=False,feature_bank_created=False,hidden_created=False,actor_updates=0,probe_updates=0,
                original_warmup_saw_all_169028_training_questions=True,new_question_holdout_claim=False,final_labels_or_rollouts_read=False,
                final_identity_source_refs_followed=False,Mac_data_transit=False,elapsed_seconds=time.monotonic()-started,epoch=time.time())
            immutable(output/'complete.json',result);atomic(output/'progress.json',dict(stage=STATUS,questions=len(joined),trajectories=counters['trajectories'],epoch=time.time()));return result
        except BaseException as error:
            atomic(output/'failure.json',dict(status='RAW_G10_NOT_ACCEPTED',error_type=type(error).__name__,error=str(error)[:900],epoch=time.time()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=('prepare','status','ingest'),required=True)
    parser.add_argument('--receiver',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--original',type=Path,default=B)
    parser.add_argument('--selection',type=Path);parser.add_argument('--final-identities',type=Path);parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.action=='ingest':
        if args.config is None:parser.error('--config required')
        value=ingest(args.config)
    elif args.action=='prepare':
        if args.receiver is None or args.output is None:parser.error('--receiver and --output required')
        if sys.platform=='darwin':parser.error('Prepare must bind real remote storage, never raw data on Mac')
        value=prepare(args.receiver,args.output,args.original,args.selection,args.final_identities)
    else:
        if args.output is None:parser.error('--output required')
        value=read(args.output/'complete.json') if (args.output/'complete.json').is_file() else dict(status='NOT_ACCEPTED',prepared=(args.output/'config.json').is_file())
    print(json.dumps(value,sort_keys=True),flush=True)
if __name__=='__main__':main()
