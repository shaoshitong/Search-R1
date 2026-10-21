"""Finite source-only inventory after the accepted244+122 raw migrations.

The existing source metadata auditor is explicitly injected, never changed.
No export/download, peer process operation, rollout/EM payload read, GPU, or
claim that these remaining source batches are already migrated is performed.
"""
import hashlib,json
from pathlib import Path
import time

NEW_SNAPSHOT=Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/audited122-delta-20261021-673b8cb6')
SOURCE_COMPLETE_SHA='bb20a9e0f9e9dfea5ebdaf2ce6000d60d0c9ac0cdad0636b0afa9cd16f2733c4'
SOURCE_PLAN_SHA='3465c82e0d930569013b8d054e714f20cdb9f35972cbe7570829d13878e729da'
RECORD_SHA='673b8cb6abe7108e1d03a98b518121dafcd1ec10b156ec78f37c5faacfb7cc21'

def subtract_accepted(current_delta,newly_accepted):
    now={x['input']:x for x in current_delta};accepted={x['input']:x for x in newly_accepted}
    if len(now)!=len(current_delta) or len(accepted)!=len(newly_accepted) or not set(accepted)<=set(now):
        raise ValueError('Distinct accepted source batch must still be sealed')
    for name,row in accepted.items():
        if any(row[k]!=now[name][k] for k in ('input_sha256','seal_sha256','questions','trajectories')):
            raise ValueError('Previously accepted122 source metadata changed')
    return [now[k]for k in sorted(set(now)-set(accepted))]

def _counts(rows):
    domains={}
    for row in rows:
        for name,count in row['domain_counts'].items():domains[name]=domains.get(name,0)+count
    return dict(batches=len(rows),questions=sum(r['questions']for r in rows),trajectories=sum(r['trajectories']for r in rows),domains=domains)

def _time_range(values):
    values=[v for v in values if isinstance(v,(int,float))]
    def fmt(v):return time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(v))
    return dict(oldest_epoch=min(values)if values else None,latest_epoch=max(values)if values else None,
        oldest_UTC=fmt(min(values))if values else None,latest_UTC=fmt(max(values))if values else None)

def inventory(original_import_binding,accepted122_binding,*,base_module):
    d=base_module;started=time.time()
    if (accepted122_binding['status']!='BOUNDED_AUDITED122_RAW_TRANSFER_AND_STRICT_EM_G10_ACCEPTED'
            or accepted122_binding['received_questions']!=15616 or accepted122_binding['received_newseed5_trajectories']!=78080
            or accepted122_binding['eligible_questions']!=15616 or accepted122_binding['eligible_authentic_G10_trajectories']!=156160
            or accepted122_binding['ingest_complete']['sha256']!='13f7e6a24699dd9858387210552a4d922ab521bd9a9c7c99648dab147f36048e'
            or accepted122_binding['target_independent_full_SHA_audit']['sha256']!='2cfe0417cdecac5d53d9c565c81cbb21d0fdf0e9ec894c67a2009d6482d9b28c'):
        raise ValueError('Actual accepted122 receiver/strictEM/independent-SHA binding required')
    newer,complete_ref=d.load(NEW_SNAPSHOT/'complete.json');plan,plan_ref=d.load(NEW_SNAPSHOT/'plan.json')
    if (complete_ref['sha256']!=SOURCE_COMPLETE_SHA or plan_ref['sha256']!=SOURCE_PLAN_SHA
            or newer['questions']!=15616 or newer['trajectories']!=78080 or newer['frozen_sealed_batches']!=122
            or plan['subset_batch_records_sha256']!=RECORD_SHA or plan['old244_identity_overlap']!={'batches':0,'qids':0,'question_hashes':0,'trajectory_ids':0}):
        raise ValueError('Exact newly accepted frozen122 source baseline differs')
    observed=d.inventory(original_import_binding);p=observed['plan']
    remaining=subtract_accepted(p['selected_delta_batches'],plan['selected_batches'])
    if p['imported_batches']!=244 or len(plan['selected_batches'])!=122:raise ValueError('Fixed366 accepted source batch baseline required')
    accepted={r['input']for r in plan['selected_batches']}
    if len(accepted)!=122:raise ValueError('New accepted source baseline repeats batches')
    # All output records stay aggregate. Only seal/proof/config/input metadata
    # was read by the inherited inventory; raw prefixes/full/EM remain unopened.
    times=[];epochs=[];proof_epochs=[];refs=[]
    for row in remaining:
        path=d.regular(d.BASE,row['seal']);value,ref=d.load(path)
        if ref['sha256']!=row['seal_sha256']:raise ValueError('Remaining source seal changed after inventory')
        times.append(path.stat().st_mtime);epochs.append(value.get('epoch'));proof_epochs.append(row['audit'].get('verified_epoch'))
        refs.append(dict(input=row['input'],input_sha256=row['input_sha256'],seal=ref,questions=row['questions'],
            trajectories=row['trajectories'],domain_counts=row['domain_counts'],audit=row['audit'],seal_file_mtime=times[-1]))
    audited=[r for r in remaining if r['audit']['status']=='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND']
    pending=[r for r in remaining if r not in audited]
    root_files={}
    for name in ('complete.json','failure.json'):
        path=d.BASE/name
        if path.exists():
            value,ref=d.load(d.regular(d.BASE,name));root_files[name]=dict(ref=ref,status=value.get('status'),error_type=value.get('error_type'),epoch=value.get('epoch'))
        else:root_files[name]=dict(exists=False)
    root_after,root_ref=d.load(d.regular(d.BASE,'config.json'));manifest_after,manifest_ref=d.load(d.regular(d.BASE,'inputs/manifest.json'))
    if root_ref!=p['source_config'] or manifest_ref!=p['input_manifest']:raise ValueError('Source config/manifest changed during finite observation')
    now_status,now_status_ref=d.load(d.regular(d.BASE,'audit/latest.json'))
    return dict(status='READONLY_SOURCE_G5_REMAINING_AFTER_ACCEPTED366_NOT_MIGRATED',started_epoch=started,epoch=time.time(),
        elapsed_seconds=time.time()-started,source_base=p['source_base'],source_run_id=p['source_run_id'],seed=p['seed'],model_revision=p['model_revision'],
        accepted_baseline=dict(batches=366,questions=46848,newseed5_trajectories=234240,
            first244_source_complete=p['imported_snapshot_complete'],accepted122_source_complete=complete_ref,
            accepted122_source_plan=plan_ref,accepted122_target_complete=accepted122_binding['receiver_complete'],accepted122_raw_ingest=accepted122_binding['ingest_complete']),
        current_sealed=dict(batches=p['current_sealed_batches'],questions=p['current_sealed_questions'],
            trajectories=5*p['current_sealed_questions'],existing_audit_current_fstat_bound_batches=p['current_audit_fstat_bound_batches']),
        newly_sealed_complete_G5_metadata=_counts(remaining),new_existing_independent_audit_current_fstat=_counts(audited),
        new_sealed_pending_current_independent_audit=_counts(pending),
        remaining_unsealed_input_batches=p['remaining_unsealed_batches'],all169028_sealed=p['all169028_complete'],
        source_input_batches=1322,source_input_questions=169028,source_failure_and_completion_markers=root_files,
        current_independent_audit_summary=p['current_audit_summary'],current_independent_audit_metadata=p['current_audit_latest'],
        second_actual_audit_status_metadata=now_status_ref,second_actual_audit_status={k:now_status.get(k)for k in ('status','epoch','verified_batches','questions','trajectories','failed_batches','sealed_pending_batches')},
        oldest_latest_new_seal_file_mtime=_time_range(times),oldest_latest_new_seal_epoch_if_present=_time_range(epochs),oldest_latest_new_existing_proof_epoch=_time_range(proof_epochs),
        metadata_binding=dict(source_config=p['source_config'],input_manifest=p['input_manifest'],
            inherited_full_sealed_inventory_plan_sha256=observed['plan_sha256'],remaining_batch_records_sha256=d.digest(refs),remaining_batch_records=len(refs)),
        existing_audit_is_only_cached_fstat_bound_not_fresh_payload_SHA=True,prefix_full_ownEM_payload_opened=False,
        strict_EM_recomputed=False,source_or_peer_process_modified=False,peer_thread_messaged=False,export_or_payload_migration_started=False,
        newly_sealed_data_already_on_HPC3_claim=False,GPU_started=False,Mac_payload_transit=False)
