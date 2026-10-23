"""Prepare a new FA3 run while745371 trains, then submit after its owned stop.

This entry never stops a job, edits the source runtime, changes restoration
semantics, or applies the superseded FA2-closeness qualification gate. Source
receipts still freshly validate the full model/Adam/scheduler/RNG/data boundary;
the ordinary native4 production workers must prove their real CUDA restore.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from reproduction.easy2560_joint4 import control

SOURCE_BASE=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
SOURCE_JOB='745371'
SOURCE_COMMIT='e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
RUN_ID='232f20e112ac4a978c5aba52a0367195'
BASE='/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022'
PROFILE='fa3-exact-cache-v1'
MINIMUM_OUTER=12
AUTHORIZATION=dict(user_message_id='01a15f5b-7e6d-7fb3-ab3c-89aa8ec3d074',
    original_utc='2026-10-21T16:46:34.094Z',original_cst='2026-10-22 00:46:34.094 CST',
    scope='Switch at a complete checkpoint to a correct FlashAttention3 implementation; compare SDPA reference',
    superseded_gate='FA2 numerical closeness is no longer the sole criterion; historical rejections remain evidence')


def _need(condition,message):
    if not condition:raise ValueError('EASY7B_FA3_TRANSITION_'+message)


def _json(path):return json.loads(Path(path).read_bytes())


def _source():
    run=_json(SOURCE_BASE/'logs/run-manifest.json')
    _need((SOURCE_BASE/'job.id').read_text().strip()==SOURCE_JOB and run['run_id']==RUN_ID and
          run['source_commit']==SOURCE_COMMIT and run.get('training_world_size')==4 and
          run.get('actor_micro_batch_per_rank')==8 and run.get('actor_logical_micro_batch_per_rank')==4,
          'OWN_CURRENT_SOURCE_IDENTITY_DIFFERS')
    return run


def _target(base):
    base=control.own_base(base);_need(base!=SOURCE_BASE,'NEW_PRIVATE_BASE_REQUIRED')
    return base


def _consumers(base,commit,job=None):
    return dict(base=str(base),source_commit=commit,source_job=job,run_id=RUN_ID,
        historical_CP5='Keep the existing accepted old-base core + post-callback + dependency archive; do not replay',
        remaining_milestones=[50,100,150,200],core_controller_first_step=50,
        core_controller='New independent archive_controller --base/--commit/--job binding; never edit the old controller',
        post_callback_companion='New independent companion bound to this base/commit/job and the existing run_id',
        final_evaluation='Native51713 + seven source metrics after200; final native4 actual restore separately required',
        final_update=200,update201_allowed=False,archive_processes_started_here=False)


def prepare(base,commit):
    """All small assets/runtime are ready before the parent stops the source."""
    from search_g1.easy7b_infra import LEAN_PROFILES
    base=_target(base);source_run=_source();ready=base/'preparation/fa3-transition-prepared.json'
    if ready.exists():
        result=_json(ready)
        _need(result.get('status')=='READY_FOR_AUTHORIZED_BOUNDARY_TRANSITION' and result.get('commit')==commit and
              result.get('target_base')==str(base) and result.get('source_job')==SOURCE_JOB and
              result.get('source_base')==str(SOURCE_BASE),'EXISTING_PREPARATION_DIFFERS')
        return result
    _need(not any((base/name).exists() for name in ('job.id','logs/run-manifest.json',
          'preparation/training-submit-claim.json')),'TARGET_ALREADY_SUBMITTED')
    bootstrap=control.bootstrap(base,commit,predecessor=str(SOURCE_BASE))
    from reproduction.easy2560_joint4.retrieval_proxy import external_binding
    original=Path(source_run.get('external_retriever',{}).get('receipt_path') or
                  SOURCE_BASE/'preparation/external-retriever-receipt.json')
    original_binding=external_binding(original)
    _need(original_binding['service_job_id']=='745198','EXISTING_RETRIEVER745198_REQUIRED')
    external_path=base/'preparation/external-retriever-receipt.json';document=_json(original)
    if external_path.exists():_need(_json(external_path)==document,'EXTERNAL_RECEIPT_ALREADY_DIFFERS')
    else:control.publish(external_path,document)
    external=external_binding(external_path)
    from reproduction.easy2560_joint4.flash3_runtime import prepare as prepare_runtime
    runtime=prepare_runtime(base)
    result=dict(status='READY_FOR_AUTHORIZED_BOUNDARY_TRANSITION',commit=commit,profile=PROFILE,
        target_base=str(base),source_base=str(SOURCE_BASE),source_job=SOURCE_JOB,
        source_runtime_commit=SOURCE_COMMIT,run_id=RUN_ID,minimum_source_completed_outer=MINIMUM_OUTER,
        actual_source_checkpoint_not_chosen_until_source_terminal=True,
        environment=bootstrap['environment'],immutable_source_commit=commit,
        external_retriever_receipt=str(external_path),external_retriever_receipt_sha256=control.sha(external_path),
        external_retriever_service_job=external['service_job_id'],private_flash3_runtime=runtime,
        actor_micro_batch_per_rank=8,actor_logical_micro_batch_per_rank=4,training_world_size=4,
        authorization=AUTHORIZATION,historical_FA2_rejections_preserved=True,
        actor_FA3_only=PROFILE not in LEAN_PROFILES,grounding_SDPA_unchanged=True,
        rollout_backend_unchanged=PROFILE not in LEAN_PROFILES,
        compact_gqa=PROFILE in LEAN_PROFILES,
        performance_profile='lean_v1' if PROFILE in LEAN_PROFILES else 'legacy',
        archive_consumer_plan=_consumers(base,commit),source_job_stopped_here=False,
        training_submitted=False,actual_cuda_restore_accepted=False,checked_unix=time.time())
    control.publish(ready,result)
    return result


def _terminal_source():
    queue=subprocess.run(['squeue','-h','-j',SOURCE_JOB,'--format=%i|%u|%T'],
                         capture_output=True,text=True,timeout=30)
    _need(queue.returncode==0 and not queue.stdout.strip(),'SOURCE_JOB_NOT_CONFIRMED_TERMINAL')
    accounting=subprocess.run(['sacct','-n','-X','-j',SOURCE_JOB,'--format=JobIDRaw,User,State,ExitCode','-P'],
                              capture_output=True,text=True,timeout=30)
    _need(accounting.returncode==0,'SOURCE_ACCOUNTING_UNAVAILABLE')
    rows=[line.split('|') for line in accounting.stdout.splitlines() if line.strip()]
    rows=[row for row in rows if row[0]==SOURCE_JOB]
    terminal={'CANCELLED','COMPLETED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
    _need(len(rows)==1 and len(rows[0])>=4 and rows[0][1]=='jxu957' and
          rows[0][2].split()[0].split('+')[0] in terminal,'EXACT_OWN_SOURCE_JOB_TERMINAL_REQUIRED')
    return dict(source_job=SOURCE_JOB,queue_returncode=queue.returncode,queue_stdout=queue.stdout,
        accounting_returncode=accounting.returncode,accounting_stdout=accounting.stdout,
        observed_source_state=rows[0][2],observed_source_exit_code=rows[0][3],checked_unix=time.time())


def submit(base,commit):
    """Once-only CPU source audit and normal native4 submit, after an external stop."""
    base=_target(base);_source();prepared=_json(base/'preparation/fa3-transition-prepared.json')
    _need(prepared.get('status')=='READY_FOR_AUTHORIZED_BOUNDARY_TRANSITION' and prepared.get('commit')==commit and
          prepared.get('profile')==PROFILE and prepared.get('source_base')==str(SOURCE_BASE) and
          prepared.get('source_job')==SOURCE_JOB,'PREPARED_TARGET_IDENTITY_DIFFERS')
    claim=base/'preparation/fa3-transition-submit-intent.json'
    _need(not any(path.exists() for path in (claim,base/'job.id',base/'logs/run-manifest.json',
          base/'preparation/training-submit-claim.json')),'SUBMISSION_ALREADY_ATTEMPTED_INSPECT_NOT_REPLAY')
    terminal=_terminal_source()
    from search_g1.easy2560_rolling_checkpoint import latest_checkpoint
    latest=latest_checkpoint(SOURCE_BASE,verify=False)
    _need(latest is not None and MINIMUM_OUTER<=latest['step']<200 and latest['run_id']==RUN_ID and
          latest['source_commit']==SOURCE_COMMIT,'SEALED_SOURCE_CP12_OR_LATER_REQUIRED')
    control.publish(claim,dict(base=str(base),commit=commit,source_checkpoint=latest,source_terminal=terminal,
        authorization=AUTHORIZATION,automatic_retry=False,source_job_stopped_here=False,created_unix=time.time()))
    # Explicit CP avoids a redundant full91GB latest(verify=True) hash. The
    # publisher below freshly hashes all actual CP/rank/boundary/encoder files.
    from search_g1.easy2560_restore import create_receipt
    receipt=base/'preparation/fa3-full-state-source-receipt.json'
    audited=create_receipt(SOURCE_BASE,latest['checkpoint'],receipt);saved=_json(receipt)
    _need(saved['run_id']==RUN_ID and saved['source_job']==SOURCE_JOB and saved['step']==latest['step'] and
          saved['hf_saved_tensor_count']==339 and saved['world_size']==4 and saved['post_callback_rng_captured'] is True,
          'REAL_NATIVE4_POST_CALLBACK_RECEIPT_REQUIRED')
    source_summary=dict(source_step=saved['step'],next_outer=saved['step']+1,
        actual_optimizer_writes_per_rank=saved['actual_optimizer_writes_per_rank'],
        source_consumed=saved['driver_state']['source_consumed'],
        candidate_cursor=saved['driver_state']['candidate_cursor'],
        attempt_progress=saved['attempt_progress'],reward_probe_round=saved['reward_probe_round'],
        reward_probe_version=saved['reward_probe_version'],reward_probe_manifest=saved['reward_probe_manifest'],
        preserved_source_cursor_not_reset=True,source_model_optimizer_scheduler_rng_data_real_readback_pending=True)
    control.publish(base/'preparation/fa3-transition-source-ready.json',dict(status='FULL_SOURCE_CPU_AUDIT_READY',
        receipt=audited,source_summary=source_summary,source_terminal=terminal,checked_unix=time.time()))
    result=control.submit(base,commit,actor_micro=8,predecessor=str(SOURCE_BASE),restore_receipt=str(receipt),
        infra_profile=PROFILE,external_retriever_receipt=str(base/'preparation/external-retriever-receipt.json'))
    value=dict(status='FA3_NATIVE4_SUBMITTED_ACTUAL_RESTORE_PENDING',target_base=str(base),source_commit=commit,
        job=result['job'],source_summary=source_summary,restore_receipt=str(receipt),restore_receipt_sha256=control.sha(receipt),
        authorization=AUTHORIZATION,historical_FA2_rejections_preserved=True,
        actual_cuda_restore_accepted=False,automatic_retry=False,source_job_stopped_here=False,
        archive_consumer_plan=_consumers(base,commit,result['job']),submission=result,checked_unix=time.time())
    control.publish(base/'preparation/fa3-transition-submitted.json',value)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','submit'));parser.add_argument('--base',default=BASE)
    parser.add_argument('--commit',required=True);args=parser.parse_args(argv)
    print(json.dumps((prepare if args.mode=='prepare' else submit)(args.base,args.commit),sort_keys=True),flush=True)


if __name__=='__main__':main()
