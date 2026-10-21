"""Freeze only the exact100 already-audited remaining source batches.

One standalone small metadata plan is written outside the peer run. No payload
export, upload/download, raw rollout read, process change or GPU operation.
"""
import copy,fcntl,hashlib,json,os
from pathlib import Path
import time

OUTPUT=Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/metadata-plans/audited100-after366-20261021.json')
STATUS='FROZEN_SOURCE_AUDITED100_AFTER_ACCEPTED366_METADATA_ONLY'


def freeze(original_binding,accepted122_binding,reference,*,base_module,remaining_module,output=OUTPUT):
 d,m=base_module,remaining_module;output=Path(output)
 if output!=OUTPUT:raise ValueError('Exact independent NAS metadata-plan destination required')
 if (reference['status']!='READONLY_SOURCE_G5_REMAINING_AFTER_ACCEPTED366_NOT_MIGRATED'
     or reference['accepted_baseline']['batches']!=366 or reference['newly_sealed_complete_G5_metadata']!=dict(batches=128,questions=16384,trajectories=81920,domains={'hotpotqa':16256,'nq':128})
     or reference['new_existing_independent_audit_current_fstat']!=dict(batches=100,questions=12800,trajectories=64000,domains={'hotpotqa':12672,'nq':128})):
  raise ValueError('Exact finite observed128/available audited100 source reference required')
 if accepted122_binding['ingest_complete']!=reference['accepted_baseline']['accepted122_raw_ingest']:raise ValueError('Accepted366 target binding differs')
 output.parent.mkdir(exist_ok=True)
 if output.parent.resolve()!=output.parent:raise ValueError('Real independent metadata directory required')
 with output.with_suffix('.lock').open('a')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if output.exists():
   existing,ref=d.load(output)
   if existing['status']!=STATUS or existing['reference_observation_sha256']!=d.digest(reference):raise ValueError('Existing authoritative plan is a different freeze')
   return dict(status=STATUS,plan=ref,metadata_plan=existing,existing_preserved=True)
  new,complete_ref=d.load(m.NEW_SNAPSHOT/'complete.json');baseline,baseline_ref=d.load(m.NEW_SNAPSHOT/'plan.json')
  if complete_ref['sha256']!=m.SOURCE_COMPLETE_SHA or baseline_ref['sha256']!=m.SOURCE_PLAN_SHA:raise ValueError('Actual accepted122 source baseline changed')
  current=d.inventory(original_binding);p=current['plan'];remaining=m.subtract_accepted(p['selected_delta_batches'],baseline['selected_batches'])
  cutoff=reference['oldest_latest_new_seal_file_mtime']['latest_epoch'];observed=[];refs=[]
  for row in remaining:
   path=d.regular(d.BASE,row['seal'])
   if path.stat().st_mtime>cutoff:continue
   value,ref=d.load(path)
   if ref['sha256']!=row['seal_sha256']:raise ValueError('Reference source seal changed')
   refs.append(dict(input=row['input'],input_sha256=row['input_sha256'],seal=ref,questions=row['questions'],
    trajectories=row['trajectories'],domain_counts=row['domain_counts'],audit=row['audit'],seal_file_mtime=path.stat().st_mtime));observed.append(row)
  expected=reference['metadata_binding']['remaining_batch_records_sha256'];current_digest=d.digest(refs)
  prior_proof_cutoff=reference['oldest_latest_new_existing_proof_epoch']['latest_epoch']
  reconstructed=copy.deepcopy(refs);later_proofs=0
  for row in reconstructed:
   audit=row['audit']
   if audit.get('status')=='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND'and audit.get('verified_epoch',0)>prior_proof_cutoff:
    row['audit']={'status':'SEALED_NO_EXISTING_INDEPENDENT_BATCH_PROOF'};later_proofs+=1
  reconstructed_digest=d.digest(reconstructed)
  if len(refs)!=128 or (current_digest!=expected and reconstructed_digest!=expected):
   raise ValueError('Original128 selected metadata not reproduced: '+json.dumps(dict(count=len(refs),current=current_digest,normalized_later_proofs=reconstructed_digest,expected=expected,later_proofs=later_proofs)))
  selected=[r for r in observed if r['audit']['status']=='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND'and r['audit']['verified_epoch']<=prior_proof_cutoff]
  if m._counts(selected)!=reference['new_existing_independent_audit_current_fstat']:raise ValueError('Exact already-audited100 scope differs')
  if p['source_config']!=reference['metadata_binding']['source_config']or p['input_manifest']!=reference['metadata_binding']['input_manifest']:raise ValueError('Fixed source config/manifest changed')
  selected=sorted(selected,key=lambda r:r['input']);selected_sha=d.digest(selected)
  # Fresh identity metadata only: complete/plan/membership are separate frozen
  # snapshot metadata, and selected input JSON supplies no raw rollout payload.
  prior_refs=[];prior_q=set();prior_h=set();prior_t=set();prior_inputs=set()
  for complete in (d.load(Path(p['imported_snapshot_complete']['path']))[0],new):
   membership,mref=d.load(Path(complete['canonical_membership']['path']));prior_refs.append(mref)
   if any(mref[k]!=complete['canonical_membership'][k]for k in ('bytes','sha256')):raise ValueError('Prior snapshot identity membership changed')
   frozen,pr=d.load(Path(complete['plan']['path']))
   if any(pr[k]!=complete['plan'][k]for k in ('bytes','sha256')):raise ValueError('Prior snapshot frozen source plan changed')
   prior_inputs.update(r['input']for r in frozen['selected_batches'])
   for g in membership['groups']:
    if g['qid']in prior_q or prior_t.intersection(g['trajectory_ids']):raise ValueError('Prior accepted366 identities repeat')
    prior_q.add(g['qid']);prior_h.add(g['question_hash']);prior_t.update(g['trajectory_ids'])
  if len(prior_inputs)!=366 or len(prior_q)!=46848 or len(prior_t)!=234240:raise ValueError('Exact accepted366 full-identity scope differs')
  ownq=set();ownh=set();ownt=set();overlap=dict(batches=0,qids=0,question_hashes=0,trajectory_ids=0)
  for r in selected:
   batch,ref=d.load(d.regular(d.BASE,r['input']))
   if ref!=r['input_ref']:raise ValueError('Selected100 input metadata changed')
   overlap['batches']+=int(r['input']in prior_inputs)
   for row in batch['rows']:
    q,h=row['qid'],row['question_hash'];ts=[d.RUN+'-outer000001-c%04d-g%d'%(row['candidate_ordinal'],j)for j in range(5)]
    if q in ownq or ownt.intersection(ts):raise ValueError('Selected100 own identities repeat')
    overlap['qids']+=int(q in prior_q);overlap['question_hashes']+=int(h in prior_h);overlap['trajectory_ids']+=len(set(ts)&prior_t)
    ownq.add(q);ownh.add(h);ownt.update(ts)
  if any(overlap.values())or len(ownq)!=12800 or len(ownt)!=64000:raise ValueError('Audited100 overlaps accepted366 or own counts differ')
  plan=dict(status=STATUS,schema='FIXED_AUDITED_SOURCE_G5_SUBSET_AFTER_ACCEPTED366_V1',subset='existing-audited',source_base=str(d.BASE),
   source_run_id=d.RUN,seed=d.SEED,source_seed=1729,model_revision=d.REVISION,sampling_attempt_id=d.ATTEMPT,
   source_config=p['source_config'],input_manifest=p['input_manifest'],batches=100,questions=12800,trajectories=64000,
   domain_counts={'hotpotqa':12672,'nq':128},selected_batches=selected,batch_records_sha256=selected_sha,
   selection_rule='EXACT first finite128 remainder observation; only its100 pre-existing independently audited batches whose proof matches current all-file fstats; no labels/probe scores',
   reference_observation_sha256=d.digest(reference),reference_remaining128_records_sha256=reference['metadata_binding']['remaining_batch_records_sha256'],
   current_reference128_records_sha256=current_digest,original_reference128_reconstructed_sha256=reconstructed_digest,
   excluded_pending28_later_proofs=later_proofs,old_selected100_proof_cutoff_epoch=prior_proof_cutoff,
   original_selected100_payload_input_seal_and_audit_SHA_unchanged=True,
   metadata_change_explanation='Only later independent proofs on originally unselected pending28 may differ; replacing those new proof objects with their prior missing-proof marker must reproduce the exact old128 digest. Original selected100 proof/input/seal bindings must remain exact.',
   source_baseline_accepted366=reference['accepted_baseline'],accepted_baseline_batch_count=366,
   prior_snapshots=[dict(complete=p['imported_snapshot_complete'],expected_batches=244,expected_questions=31232,expected_trajectories=156160),dict(complete=complete_ref,expected_batches=122,expected_questions=15616,expected_trajectories=78080)],
   strict_difference_against_accepted366=True,selected_batch_paths_overlap_accepted366=0,
   accepted366_identity_membership_refs=prior_refs,accepted366_identity_overlap=overlap,selected_unique_question_hashes=len(ownh),
   pending28_excluded=True,pending28_questions=3584,source_input_batches=1322,source_input_questions=169028,
   existing_audit_all_actual_file_fstat_matches=True,existing_audit_payload_SHA_inherited_only=True,
   fresh_payload_SHA_recomputed=False,strict_EM_tokenizer_recomputed=False,
   future_export_must_revalidate_source_fstats_inputs_and_fresh_archive_member_SHA=True,
   original_warmup_question_unseen_claim=False,authority_plan_path=str(output),
   payload_export_started=False,payload_upload_started=False,payload_download_started=False,GPU_started=False,
   source_run_or_peer_process_modified=False,peer_thread_messaged=False,created_epoch=time.time())
  temp=output.with_suffix('.tmp')
  with temp.open('xb')as f:f.write(d.canonical(plan)+b'\n');f.flush();os.fsync(f.fileno())
  temp.replace(output);check,ref=d.load(output)
  if check!=plan:raise ValueError('Authoritative remote small plan readback differs')
  return dict(status=STATUS,plan=ref,metadata_plan=plan,existing_preserved=False)
