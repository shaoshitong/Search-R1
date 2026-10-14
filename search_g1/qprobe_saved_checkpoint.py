"""Explicit recovery of real saved actor/head5 before the lost dev RNG boundary.

This source is not a completed-pair receipt. It restores the exact checkpoint
RNG, retains the already measured dev512 result, and records the discontinuity
to the unavailable post-evaluation RNG. No teacher or actor prefix is replayed.
"""
import copy
import json
import os
from pathlib import Path
import time

SAVED_KIND='question_probe_saved_checkpoint'
SAVED_STATUS='QPROBE_SAVED_CHECKPOINT5_PRE_DEVELOPMENT_FULL_SHA_CPU_PASS'
QPROBE_RESTORE_KINDS=('question_probe_pair',SAVED_KIND)
QPROBE_SOURCE_STATUSES=('QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',SAVED_STATUS)
_CACHE={}


def need(value,message):
    if not value:raise ValueError('QPROBE_SAVED_CHECKPOINT_'+message)


def read(path):return json.loads(Path(path).read_text())


def _rows(base,name,first,last,key):
    path=Path(base)/'logs'/name
    return [row for row in map(json.loads,path.read_text().splitlines()) if first<=row[key]<=last]


def snapshot_part(base,run,first,last):
    from search_g1.qprobe_checkpoint import _file_record
    from reproduction.common8.policy_calibration_plan import document_sha256
    updates=_rows(base,'actual-updates.jsonl',first,last,'step')
    attempts=_rows(base,'actual-attempts.jsonl',first,last,'logical_update')
    need([row['step'] for row in updates]==list(range(first,last+1)),'REAL_RAW_PREFIX_INCOMPLETE')
    need(all(row['run_id']==run['run_id'] for row in updates+attempts),'RAW_PREFIX_RUN_DIFFERS')
    need({row['attempt_id'] for row in attempts if row['complete_outer']}==
         {row['attempt_id'] for row in updates},'RAW_PREFIX_ACTOR_SUCCESSES_DIFFER')
    return dict(base=str(Path(base).resolve()),first=first,last=last,
        source_run_manifest=_file_record(Path(base)/'logs/run-manifest.json'),
        updates_prefix_sha256=document_sha256(updates),attempts_prefix_sha256=document_sha256(attempts),
        semantics='bounded original raw rows, verified in their original producer directory; no synthesized history')


def verify_part(part,run):
    from search_g1.full_state_restore import digest
    from search_g1.qprobe_checkpoint import _identity_record
    from reproduction.common8.policy_calibration_plan import document_sha256
    base=Path(part['base']).resolve();record=part['source_run_manifest']
    _identity_record(record);need(digest(record['path'])==record['full_read_sha256'] and
        digest(base/'logs/run-manifest.json')==record['full_read_sha256'],'PREFIX_MANIFEST_CHANGED')
    parent=read(record['path']);need(parent['run_id']==run['run_id'],'PREFIX_RUN_CHANGED')
    for key in ('budget','input_sha256','initial_model_sha256','training_world_size','actor_micro_batch_per_rank'):
        need(parent[key]==run[key],'PREFIX_SCIENTIFIC_BINDING_CHANGED_'+key)
    first,last=part['first'],part['last']
    need(type(first) is type(last) is int and 1<=first<=last<=200,'PREFIX_RANGE_INVALID')
    updates=_rows(base,'actual-updates.jsonl',first,last,'step')
    attempts=_rows(base,'actual-attempts.jsonl',first,last,'logical_update')
    need([row['step'] for row in updates]==list(range(first,last+1)) and
        document_sha256(updates)==part['updates_prefix_sha256'] and
        document_sha256(attempts)==part['attempts_prefix_sha256'],'ACTUAL_RAW_PREFIX_CHANGED')
    need(all(row['run_id']==run['run_id'] for row in updates+attempts),'PREFIX_RAW_RUN_CHANGED')
    return dict(base=base,run=parent,first=first,last=last)


def history_parts(base,run,last=200):
    """Read authentic original prefix parts plus the real current suffix."""
    from search_g1.full_state_restore import source,verified_runtime_prefix
    binding=run.get('full_state_restore')
    parts=[];offset=0
    if binding:
        need(binding.get('kind') in QPROBE_RESTORE_KINDS,'UNSUPPORTED_HISTORY_RESTORE_KIND')
        _,receipt,_,_,_=source(binding['source_receipt']);offset=binding['completed_outer']
        need(last>=offset,'HISTORY_BEFORE_SOURCE_OUTER')
        for part in receipt.get('historical_artifacts',{}).get('training_parts',[]):
            parts.append(verify_part(part,run))
        need(parts and [i for part in parts for i in range(part['first'],part['last']+1)]==
            list(range(1,offset+1)),'AUTHENTIC_PREFIX_PARTS_MISSING')
        prefix=verified_runtime_prefix(base,run)
        need(prefix['completed_outer']==offset,'ACTUAL_RESTORE_PREFIX_NOT_VERIFIED')
    if last>offset:
        part=snapshot_part(base,run,offset+1,last);parts.append(verify_part(part,run))
    need([i for part in parts for i in range(part['first'],part['last']+1)]==list(range(1,last+1)),
         'HISTORY_COVERAGE_OR_DUPLICATE')
    return parts


def history_records(base,run,outer):
    return [snapshot_part(part['base'],part['run'],part['first'],part['last'])
            for part in history_parts(base,run,last=outer)]


def build_source(receipt_path):
    """Publish a distinct source admission from the already verified failure."""
    from reproduction.hpc3_qprobe.formal_audit_failure718650 import BASE,CHECKPOINT,RUN,SOURCE,metadata
    from search_g1.qprobe_checkpoint import _file_record,digest
    from search_g1.full_state_restore import publish
    metadata();directory=BASE/'failed-job-718650';closed=read(directory/'terminal-and-files.json')
    cpu=read(directory/'saved-prefix-cpu-validation.json');run=read(BASE/'logs/run-manifest.json')
    need(closed['status']=='ACTUAL_FAILED_NATIVE4_CHECKPOINT5_AND_RAW_EVIDENCE_FULL_SHA_PRESERVED' and
         closed['cpu_state_validation_pending'] is False and
         cpu['status']=='ACTUAL_SAVED_NATIVE4_CHECKPOINT5_AND_CORRECTED_STRUCTURAL_PREFIX_CPU_PASS',
         'ACTUAL_FAILURE_CPU_CLOSURE_MISSING')
    need(cpu['run_id']==RUN and cpu['source_commit']==SOURCE and
         cpu['candidate_source_cursor']==25600 and cpu['completed_pair_available'] is False and
         cpu['actual_cuda_restore_accepted'] is False,'SAVED_BOUNDARY_IDENTITY_DIFFERS')
    from search_g1.qprobe_checkpoint import _identity_record
    files=[]
    for old in closed['files']:
        _identity_record(old)
        path=Path(old['path'])
        if path.is_relative_to(CHECKPOINT):
            files.append(dict(old,relative_path=str(path.relative_to(CHECKPOINT))))
    need(len(files)==21 and {row['relative_path'] for row in files}==
         {str(p.relative_to(CHECKPOINT)) for p in CHECKPOINT.rglob('*') if p.is_file()},'SAVED_FILE_SET_DIFFERS')
    by_name={row['relative_path']:row for row in files};complete=read(CHECKPOINT/'complete.json')
    need(all(by_name[name]['full_read_sha256']==sha for name,sha in complete['files_sha256'].items()),
         'SAVED_COMPLETE_SHA_DIFFERS')
    live_path=BASE/'artifacts/live/manifest.json';live=read(live_path)
    reward=[]
    for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
        row=_file_record(live[key],relative_path=key)
        need(row['full_read_sha256']==live['head_sha256'][name],'PINNED_REWARD_HEAD_DIFFERS');reward.append(row)
    # Official frozen encoder files were independently admitted and stay in
    # their immutable namespace; hash their small transport metadata here.
    for name,sha in live['encoder_sha256'].items():
        p=Path(live['encoder'])/name;s=p.stat()
        reward.append(dict(path=str(p.resolve()),relative_path='reward_encoder/'+name,inode=s.st_ino,
            bytes=s.st_size,mtime_ns=s.st_mtime_ns,full_read_sha256=sha))
        need(sha==run['initial_model_sha256'][name],'FROZEN_REWARD_ENCODER_DIFFERS')
    checkpoint_rows=[dict(step=5,**_file_record(CHECKPOINT/'complete.json'))]
    development={k:_file_record(BASE/'logs'/n) for k,n in (
        ('before','formal-inline-development-before.json'),('after5','formal-inline-development-after.json'))}
    state=by_name['training_state/question-probe-state.pt']
    receipt=dict(schema_version=1,status=SAVED_STATUS,kind=SAVED_KIND,step=5,world_size=4,run_id=RUN,
        checkpoint=str(CHECKPOINT),source_base=str(BASE),source_run_manifest=_file_record(BASE/'logs/run-manifest.json'),
        actor_snapshot_id=RUN+':5',ranks=cpu['ranks'],files=files,
        driver_file=by_name['training_state/driver.pt'],question_probe_state_path=state['path'],
        head_path=by_name['training_state/question-probe-state.pt.head.pt']['path'],
        feature_spec_sha=complete['question_probe']['feature_spec_sha'],head_version=cpu['head_version'],
        hf_saved_tensor_count=435,attempt_progress=dict(completed_outer=5,cumulative_writes=100,
            attempt_index=5,aborted_attempts=0),driver_state=dict(completed_outer=5,source_consumed=25600),
        reward_probe_manifest=_file_record(live_path),reward_files=reward,reward_probe_version=live['version'],
        reward_probe_round=0,historical_artifacts=dict(checkpoints=checkpoint_rows,development=development,
            reward_archives=[],training_parts=[snapshot_part(BASE,run,1,5)]),
        failure_evidence=_file_record(directory/'terminal-and-files.json'),
        saved_state_cpu_validation=_file_record(directory/'saved-prefix-cpu-validation.json'),
        corrected_structural_prefix=_file_record(directory/'corrected-reader-prefix.json'),
        completed_pair_available=False,actual_cuda_restore_accepted=False,
        sampling_rng_boundary='saved checkpoint before dev512-after5',post_evaluation_rng_available=False,
        bitwise_continuation_from_failed_post_evaluation=False,checked_unix=time.time())
    publish(receipt_path,receipt);read_source(receipt_path)
    return dict(path=str(receipt_path),sha256=digest(receipt_path),status=SAVED_STATUS,
                completed_pair_available=False,actual_cuda_restore_accepted=False)


def read_source(receipt_path):
    from search_g1.qprobe_checkpoint import _identity_record,digest
    p=Path(receipt_path).resolve();r=read(p);base=Path(r['source_base']).resolve();cp=Path(r['checkpoint']).resolve()
    need(r['schema_version']==1 and r['status']==SAVED_STATUS and r['kind']==SAVED_KIND and
        type(r['step']) is int and r['step']==5 and type(r['world_size']) is int and r['world_size']==4,
        'SOURCE_TYPE_OR_TOPOLOGY_DIFFERS')
    need(cp.is_relative_to(base) and r['actor_snapshot_id']==r['run_id']+':5' and
        r['completed_pair_available'] is False and r['actual_cuda_restore_accepted'] is False and
        r['post_evaluation_rng_available'] is False and
        r['sampling_rng_boundary']=='saved checkpoint before dev512-after5','SOURCE_BOUNDARY_RELABELED')
    need(len(r['files'])==21 and {x['relative_path'] for x in r['files']}==
        {str(f.relative_to(cp)) for f in cp.rglob('*') if f.is_file()},'SOURCE_FILE_SET_CHANGED')
    rows=r['files']+r['reward_files']+[r['source_run_manifest'],r['reward_probe_manifest'],
        r['failure_evidence'],r['saved_state_cpu_validation'],r['corrected_structural_prefix']]
    rows+=list(r['historical_artifacts']['development'].values())+r['historical_artifacts']['checkpoints']
    identities=tuple(_identity_record(row) for row in rows)
    key=(str(p),digest(p),identities)
    if key not in _CACHE:
        parent=read(r['source_run_manifest']['path'])
        need(parent['run_id']==r['run_id'] and parent['training_world_size']==4 and
             parent['actor_micro_batch_per_rank']==4 and parent.get('full_state_restore') is None,
             'ORIGINAL_FRESH_NATIVE4_PARENT_DIFFERS')
        need(sorted(row['rank'] for row in r['ranks'])==list(range(4)) and all(
            row['completed_outer']==row['scheduler_last_epoch']==5 and row['scheduler_step_count']==6 and
            row['actual_optimizer_writes']==100 and row['last_attempt_index']==5 and
            row['optimizer_tensors_finite'] is True and row['policy_rejections']==0 for row in r['ranks']),
            'SOURCE_RANK_STATES_DIFFERS')
        for row in rows:
            if row['bytes']<=16*2**20 or row['path']==r['head_path']:
                need(digest(row['path'])==row['full_read_sha256'],'SOURCE_SMALL_PROOF_SHA_CHANGED')
        verify_failure_proofs(r,parent)
        need(r['attempt_progress']==dict(completed_outer=5,cumulative_writes=100,attempt_index=5,aborted_attempts=0),
             'SOURCE_DRIVER_COUNTS_CHANGED')
        need(r['driver_state']==dict(completed_outer=5,source_consumed=25600),'SOURCE_CURSOR_CHANGED')
        parts=[verify_part(row,parent) for row in r['historical_artifacts']['training_parts']]
        need(len(parts)==1 and (parts[0]['first'],parts[0]['last'])==(1,5),'SOURCE_RAW_PREFIX_CHANGED')
        _CACHE[key]=parent
    return p,r,cp,base,_CACHE[key]


def verify_failure_proofs(receipt,parent):
    """Cross-bind the real failure, CPU readbacks and unchanged checkpoint set."""
    from reproduction.hpc3_qprobe.formal_audit_failure718650 import terminal
    r=receipt;closed=read(r['failure_evidence']['path'])
    cpu=read(r['saved_state_cpu_validation']['path'])
    corrected=read(r['corrected_structural_prefix']['path'])
    terminal(closed['actual_accounting'])
    need(closed['status']=='ACTUAL_FAILED_NATIVE4_CHECKPOINT5_AND_RAW_EVIDENCE_FULL_SHA_PRESERVED' and
         closed['cpu_state_validation_pending'] is False and closed['job']=='718650' and
         closed['run_id']==r['run_id']==cpu['run_id'] and
         closed['source_commit']==parent['source_commit']==cpu['source_commit']==corrected['original_source_commit'] and
         closed['base']==r['source_base'] and closed['checkpoint']==r['checkpoint'] and
         closed['actor_head_completed_outers']==5 and closed['checkpoint_file_count']==21 and
         closed['owned_components_stopped'] is True and closed['private_tmpfs_cleaned'] is True and
         closed['first_five_acceptance_complete'] is False and
         closed['completed_pair_available'] is False and closed['actual_cuda_restore_accepted'] is False and
         closed['post_evaluation_rng_boundary_available'] is False and
         closed['checkpoint_rng_boundary']=='pre-dev512-after5','FAILURE_SOURCE_PROOF_DIFFERS')
    need(cpu['status']=='ACTUAL_SAVED_NATIVE4_CHECKPOINT5_AND_CORRECTED_STRUCTURAL_PREFIX_CPU_PASS' and
         cpu==closed['saved_prefix_cpu_validation'] and cpu['ranks']==r['ranks'] and
         cpu['head_version']==r['head_version'] and cpu['head_replay_groups']==7680 and
         cpu['candidate_source_cursor']==25600 and
         all(cpu[key] is True for key in ('head_private_rng_cpu_readback_equal',
             'head_adam_model_replay_cpu_readback_equal','selector_rng_cpu_readback_equal',
             'driver_candidate_saved_state_equal','corrected_first_five_structural_reader_accepted')) and
         all(cpu[key] is False for key in ('actual_constructed_stream_restored','actual_cuda_restore_accepted',
             'completed_pair_available','post_evaluation_rng_boundary_available')) and
         cpu['actual_checkpoint_rng_boundary']=='pre-dev512-after5','CPU_FAILURE_READBACK_PROOF_DIFFERS')
    safety=corrected['safety']
    need(corrected['original_raw_evidence_modified'] is False and corrected['actual_cuda_restore_accepted'] is False and
         safety['structural_accepted'] is True and safety['actual_complete_outers']==5 and
         safety['actual_optimizer_writes_per_rank']==100 and safety['unique_global_monitor_count']==200,
         'CORRECTED_ORIGINAL_PREFIX_PROOF_DIFFERS')
    cp=Path(r['checkpoint']);by_name={row['relative_path']:row for row in r['files']}
    original={row['path']:row for row in closed['files'] if Path(row['path']).is_relative_to(cp)}
    need(len(original)==21 and all({k:v for k,v in original[row['path']].items() if k!='relative_path'}==
         {k:v for k,v in row.items() if k!='relative_path'}
         for row in r['files']),'PRESERVED_CHECKPOINT_FILES_DIFFER')
    complete=read(cp/'complete.json')
    need(complete['run_id']==r['run_id'] and complete['step']==5 and
         complete['validated_tensor_count']==r['hf_saved_tensor_count']==435 and
         complete['finite_all_saved_tensors'] is True and complete['actual_optimizer_writes_per_rank']==100 and
         all(by_name[name]['full_read_sha256']==sha for name,sha in complete['files_sha256'].items()) and
         set(complete['files_sha256'])==set(by_name)-{'complete.json'} and
         r['driver_file']==by_name['training_state/driver.pt'] and
         r['question_probe_state_path']==by_name['training_state/question-probe-state.pt']['path'] and
         r['head_path']==by_name['training_state/question-probe-state.pt.head.pt']['path'] and
         complete['question_probe']['feature_spec_sha']==r['feature_spec_sha'] and
         complete['question_probe']['head_version']==r['head_version'] and
         r['bitwise_continuation_from_failed_post_evaluation'] is False,
         'ORIGINAL_COMPLETE_OR_SAVED_STATE_BINDING_DIFFERS')


def restore_driver(trainer,receipt_path):
    import torch,numpy as np,random
    from search_g1.full_state_restore import equal_tree,bind_restored_probe,publish
    from search_g1.training_state import driver_rng,compact_stream_state
    _,r,cp,_,_=read_source(receipt_path);proof=r['driver_file']
    from search_g1.qprobe_checkpoint import digest
    need(digest(proof['path'])==proof['full_read_sha256'],'SAVED_DRIVER_SHA_CHANGED')
    saved=torch.load(proof['path'],map_location='cpu',weights_only=False)
    need(saved['completed_outer']==5 and saved['candidate_stream']['consumed_source_rows']==25600,
         'ACTUAL_DRIVER_CURSOR_DIFFERS')
    equal_tree(saved['attempt_progress'],r['attempt_progress'])
    base=Path(os.environ['SEARCH_G1_BASE']);run=read(base/'logs/run-manifest.json');live=read(base/'artifacts/live/manifest.json')
    bind_restored_probe(trainer,saved,run,live,r['run_id'])
    torch.set_rng_state(saved['rng']['torch_cpu']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    equal_tree(driver_rng(),saved['rng'])
    trainer._restored_candidate_state=compact_stream_state(saved['candidate_stream'])
    trainer._restored_attempt_progress=copy.deepcopy(saved['attempt_progress'])
    trainer._restored_qprobe_state=dict(path=r['question_probe_state_path'],actor_snapshot_id=r['actor_snapshot_id'],
        head_version=r['head_version'],feature_spec_sha=r['feature_spec_sha'],source_receipt_sha256=digest(receipt_path))
    row=dict(source_checkpoint=str(cp),actual_driver_rng_readback_equal=True,completed_outer=5,
        actual_candidate_source_cursor=25600,candidate_order_count=len(saved['candidate_stream']['candidate_order']),
        pending_offset=saved['candidate_stream']['pending_offset'],actual_probe_version=saved['probe_version'],actual_probe_round=0,
        attempt_progress=saved['attempt_progress'],source_driver_full_sha256=proof['full_read_sha256'],
        sampling_rng_boundary=r['sampling_rng_boundary'],bitwise_continuation_from_failed_post_evaluation=False,
        reward_alpha_fixed=True,historical_reward_diagnostics_restored=False,checked_unix=time.time())
    publish(base/'logs/full-state-restored-driver.json',row);return row


def record_runtime_restore(trainer,receipt_path):
    """Require actual constructed-stream/head/private RNG equal twice, before use."""
    import torch
    from search_g1.full_state_restore import source,equal_tree,publish,digest
    from search_g1.training_state import compact_stream_state
    _,r,cp,_,_=source(receipt_path)
    saved=torch.load(r['question_probe_state_path'],map_location='cpu',weights_only=False)
    raw_head=torch.load(r['head_path'],map_location='cpu',weights_only=True)
    driver_record=r['driver_file'] if r['kind']==SAVED_KIND else r['boundary_driver']
    original_driver=torch.load(driver_record['path'],map_location='cpu',weights_only=False)
    runtime=trainer.qprobe_runtime
    from search_g1.training_state import driver_rng
    for _ in range(2):
        need(runtime.completed_outer==runtime.head.completed_outer==r['step'] and runtime.version==r['head_version'],
             'ACTUAL_HEAD_OUTER_VERSION_DIFFERS')
        equal_tree(runtime.head.state_dict(),raw_head)
        equal_tree(runtime.selection_rng.bit_generator.state,saved['selection_rng'])
        equal_tree(compact_stream_state(trainer.mixed_stream.state()),saved['candidate_state'])
        equal_tree(driver_rng(),original_driver['rng'])
    row=dict(run_id=r['run_id'],completed_outer=r['step'],actual_head_adam_private_rng_replay_equal=True,
        actual_selector_rng_equal=True,actual_constructed_candidate_stream_equal=True,
        actual_postconstruction_driver_rng_equal=True,readback_passes=2,
        head_version=runtime.version,head_replay_groups=len(runtime.head.replay_labels),
        source_receipt_sha256=digest(receipt_path),candidate_source_cursor=trainer.mixed_stream.consumed,
        checked_unix=time.time())
    publish(Path(os.environ['SEARCH_G1_BASE'])/'logs/full-state-restored-question-probe.json',row)
    return row


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    need(a.receipt.is_absolute() and a.receipt.parent==Path(
        '/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014/failed-job-718650'),
         'EXPLICIT_FAILURE_SOURCE_NAMESPACE_REQUIRED')
    print(json.dumps(build_source(a.receipt)),flush=True)
