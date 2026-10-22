"""Bounded read-only CP34 recovery followup:35..39 complete and40 saved.

No result from this observer starts, stops, submits or resumes any experiment.
Runtime and observer commits are explicit and distinct identities.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time

from reproduction.operations.easy7b_20261021 import watch_fa3_transition_local as shared
from reproduction.operations.easy7b_20261021 import observe_cp34_recovery_hpc as remote

SOURCE_BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022'
SOURCE_COMMIT = 'ebb6844db6a1ae739a06d5cc31d3510e7b4779b1'


def restoration(value, job, runtime_commit):
    run = shared.document(value,'logs/run-manifest.json'); binding=run.get('full_state_restore',{})
    workers=shared.document(value,'logs/full-state-restored-workers.json')
    driver=shared.document(value,'logs/full-state-restored-driver.json')
    stream=shared.document(value,'logs/easy2560-restored-stream-before-next-outer.json')
    rows=workers.get('ranks',[]); receipt_sha=binding.get('source_receipt_sha256')
    checkpoint=SOURCE_BASE+'/checkpoints/search-g1-common8/actor/global_step_34'
    if not (run.get('run_id')==remote.RUN and run.get('source_commit')==runtime_commit and
        run.get('training_world_size')==4 and run.get('actor_micro_batch_per_rank')==8 and
        run.get('actor_logical_micro_batch_per_rank')==4 and binding.get('kind')=='easy2560_native4' and
        binding.get('completed_outer')==34 and binding.get('actual_optimizer_writes_per_rank')==680 and
        binding.get('source_job')==remote.SOURCE_JOB and binding.get('source_commit')==SOURCE_COMMIT and
        binding.get('source_base')==SOURCE_BASE and binding.get('source_checkpoint')==checkpoint and
        binding.get('candidate_source_cursor')==87040 and isinstance(receipt_sha,str) and
        re.fullmatch('[0-9a-f]{64}',receipt_sha) and
        workers.get('run_id')==remote.RUN and workers.get('source_commit')==runtime_commit and
        workers.get('slurm_job_id')==job and workers.get('source_step')==34 and
        workers.get('actual_world_size')==4 and workers.get('actual_optimizer_writes_per_rank')==680 and
        workers.get('source_receipt_sha256')==receipt_sha and len(rows)==4 and
        sorted(row.get('rank',-1) for row in rows)==list(range(4)) and
        len({row.get('actual_cuda_uuid') for row in rows})==4 and all(row.get('actual_cuda_uuid') for row in rows)):
        return False
    for row in rows:
        individual=value.get('actual_restore_ranks',{}).get(str(row['rank']),{})
        if not (individual.get('present') is True and individual.get('value')==row and
            row.get('slurm_job_id')==job and row.get('run_id')==remote.RUN and
            row.get('source_receipt_sha256')==receipt_sha and row.get('source_job')==remote.SOURCE_JOB and
            row.get('source_commit')==SOURCE_COMMIT and row.get('source_step')==34 and
            row.get('source_checkpoint')==checkpoint and row.get('actual_optimizer_writes')==680 and
            row.get('scheduler_last_epoch')==34 and row.get('world_size')==4 and
            row.get('model_hf_exact_tensors_checked_on_rank0')==(339 if row['rank']==0 else 0) and
            all(row.get(key) is True for key in ('actual_model_collective_readback_equal',
                'actual_adam_scheduler_readback_equal','actual_rng_readback_equal'))):
            return False
    return bool(driver.get('run_id')==remote.RUN and driver.get('slurm_job_id')==job and
        driver.get('source_receipt_sha256')==receipt_sha and driver.get('source_step')==34 and
        driver.get('completed_outer')==34 and driver.get('source_job')==remote.SOURCE_JOB and
        driver.get('source_commit')==SOURCE_COMMIT and driver.get('source_checkpoint')==checkpoint and
        driver.get('actual_candidate_source_cursor')==87040 and driver.get('actual_driver_dataequal') is True and
        driver.get('attempt_progress',{}).get('completed_outer')==34 and
        driver.get('attempt_progress',{}).get('cumulative_writes')==680 and
        driver.get('actual_candidate_stream_readback_equal') is True and driver.get('actual_driver_rng_readback_equal') is True and
        stream.get('completed_outer')==34 and stream.get('candidate_source_cursor')==87040 and
        stream.get('current_world_size')==4 and stream.get('actual_candidate_stream_readback_equal') is True and
        stream.get('actual_driver_rng_readback_equal') is True)


def summarize(value, job, runtime_commit, completed=39, next_outer=40):
    progress=value.get('new_training_progress',{}); wanted=list(range(35,completed+1)); issues=[]
    state=shared.job_state(value,job); service_state=shared.job_state(value,'745198')
    updates=[r for r in value.get('post_restore_update_records',[]) if r.get('step') in wanted]
    attempts=[r for r in value.get('post_restore_attempt_records',[]) if r.get('logical_update') in wanted]
    updates_ok=([r.get('step') for r in updates]==wanted and all(r.get('run_id')==remote.RUN and
        r.get('update_status')=='complete' and r.get('actual_optimizer_writes')==20 and
        r.get('full_budget_complete') is True and r.get('opportunities_processed')==20 and
        r.get('opportunities_unattempted')==0 and r.get('optimizer_writes_before')==(r['step']-1)*20 and
        r.get('optimizer_writes_after')==r['step']*20 for r in updates))
    attempts_ok=([r.get('logical_update') for r in attempts]==wanted and all(r.get('run_id')==remote.RUN and
        r.get('slurm_job_id')==job and r.get('status')=='complete' and r.get('complete_outer') is True and
        r.get('all_ranks_agreed') is True and r.get('real_rank_count')==4 and
        r.get('full_budget_complete') is True and r.get('optimizer_writes_this_attempt')==20 and
        r.get('optimizer_rollbacks_this_attempt',0)==0 and r.get('scheduler_advances')==1 and
        r.get('optimizer_writes_after')==r['logical_update']*20 for r in attempts))
    rank_proofs={}; all_ranks=True
    for rank in range(4):
        rows=[r for r in value.get('post_restore_scheduler_records',{}).get(str(rank),[])
              if r.get('kind')=='COMPLETE' and r.get('logical_update') in wanted]
        scheduler=([r.get('logical_update') for r in rows]==wanted and all(r.get('run_id')==remote.RUN and
            r.get('slurm_job_id')==job and r.get('rank')==rank and r.get('actual_optimizer_writes')==20 and
            r.get('full_budget_complete') is True and r.get('scheduler_advances')==1 and
            r.get('scheduler_last_epoch_after')==r['logical_update'] and
            r.get('writes_before')==(r['logical_update']-1)*20 and r.get('writes_after')==r['logical_update']*20 for r in rows))
        budget=[r for r in value.get('post_restore_optimizer_records',{}).get(str(rank),[])
                if r.get('logical_update') in wanted]
        optimizer=([r.get('logical_update') for r in budget]==wanted and all(r.get('run_id')==remote.RUN and
            r.get('rank')==rank and r.get('optimizer_writes')==20 and
            r.get('cumulative_writes')==r['logical_update']*20 and r.get('full_budget_complete') is True and
            r.get('opportunities_processed')==20 and r.get('local_micro')==8 and r.get('local_logical_micro')==4 and
            r.get('local_mini')==32 and r.get('gradient_accumulation')==4 and
            r.get('actual_write_counts')==list(range((r['logical_update']-1)*20+1,r['logical_update']*20+1)) for r in budget))
        audit=value.get('actual_flash3_execution',{}).get(str(rank),{}).get('latest_actual_record',{})
        forwards=sum(n for n in audit.get('forward_success_returns',{}).values() if type(n) is int)
        backward=audit.get('kernel_backward_success_returns',0); outer=audit.get('logical_update')
        fa3=(audit.get('run_id')==remote.RUN and audit.get('rank')==rank and
            audit.get('actual_flash3_training_executed') is True and audit.get('cuda_synchronized') is True and
            type(outer) is int and outer>=completed and audit.get('actual_optimizer_writes')==outer*20 and
            forwards>0 and type(backward) is int and backward>0)
        rank_proofs[str(rank)]=dict(scheduler_through_target=scheduler,optimizer20_through_target=optimizer,actual_fa3_through_target=fa3,
            forward_returns=forwards,backward_returns=backward,outer=outer,writes=audit.get('actual_optimizer_writes'))
        all_ranks=all_ranks and scheduler and optimizer and fa3
    latest=progress.get('preparation/rolling-checkpoints/latest.json',{})
    seal=(latest.get('status')=='EASY2560_ROLLING_CHECKPOINT_READY' and latest.get('run_id')==remote.RUN and
          latest.get('source_commit')==runtime_commit and type(latest.get('step')) is int and latest['step']>=completed)
    saved=any(type(p.get('step')) is int and p['step']>=next_outer and
        type(p.get('completed_questions')) is int and p['completed_questions']>0 and
        type(p.get('saved_chunk_files')) is int and p['saved_chunk_files']>0 for p in progress.get('candidate_pools',[]))
    logs=[value.get('new_training_slurm_log',{}),progress.get('training_log',{}),progress.get('trainer_log',{})]
    errors=[line[-1500:] for log in logs for line in log.get('tail','').splitlines() if shared.ERROR.search(line)][-10:]
    errors += value.get('grounding_cache_release',{}).get('error_lines',[])
    if errors: issues.append('Actual error observed in current recovery log tail')
    if state in shared.TERMINAL: issues.append('Recovery job terminal: '+state)
    if service_state in shared.TERMINAL: issues.append('Retrieval service terminal: '+service_state)
    if value.get('new_job_record',{}).get('matches_requested_job') is not True: issues.append('Requested job and actual job.id differ or are unavailable')
    if value.get('post_restore_ledger_observation_errors') or progress.get('compact_summary_observation_errors'):
        issues.append('Actual ledger reader reported malformed evidence')
    for row in attempts:
        if row.get('status')!='complete' or row.get('optimizer_writes_this_attempt')!=20 or row.get('optimizer_rollbacks_this_attempt',0)!=0:
            issues.append('Observed guarded/partial/aborted recovery attempt')
    cache=value.get('grounding_cache_release',{}).get('latest') or {}
    cache_ok=(cache.get('role')=='probe_post_judge_cuda_cache_release' and cache.get('cuda_synchronized') is True and
        type(cache.get('released_reserved_bytes')) is int and isinstance(cache.get('before'),dict) and isinstance(cache.get('after'),dict))
    allocator=shared.document({'documents':{'allocator':value.get('allocator_resources',{})}},'allocator')
    allocator_ok=(allocator.get('run_id')==remote.RUN and allocator.get('source_commit')==runtime_commit and
        allocator.get('slurm_job_id')==job and allocator.get('torch_allocator_env',{}).get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True')
    restore_ok=restoration(value,job,runtime_commit)
    success=bool(state=='RUNNING' and service_state=='RUNNING' and restore_ok and updates_ok and attempts_ok and
        all_ranks and seal and saved and cache_ok and allocator_ok and not issues)
    metrics=progress.get('completed_outer_metrics',[]); last=metrics[-1] if metrics else {}
    return dict(epoch=value.get('epoch'),job=job,job_state=state,retrieval_service_state=service_state,
        source_completed_outer=34,source_writes_per_rank=680,source_candidate_cursor=87040,
        completed_outer=last.get('step'),reward=last.get('reward'),checkpoint_outer=latest.get('step'),
        required_completed_outers=wanted,actual_all4_restore_observed=restore_ok,
        complete_updates_observed=updates_ok,complete_attempts_observed=attempts_ok,all4_rank_proofs=rank_proofs,
        sealed_through_target=seal,next_outer_saved=saved,candidate_pools=progress.get('candidate_pools',[]),
        grounding_cache_release_observed=cache_ok,grounding_cache_release=cache,
        allocator_config_observed=allocator_ok,error_lines=errors,inspection_issues=issues,
        requested_followup_observed=success,overall_goal_complete=False,production_mutated=False)


def command(job,runtime_commit,operator_commit):
    code=('import json,sys\nsys.dont_write_bytecode=True\nsys.path.insert(0,'+
        repr(str(remote.BASE/'source'/operator_commit))+')\n'+
        'from reproduction.operations.easy7b_20261021.observe_cp34_recovery_hpc import observe\n'+
        'print(json.dumps(observe('+','.join(map(repr,(job,runtime_commit,operator_commit)))+'),sort_keys=True,allow_nan=False),flush=True)')
    return shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-commit',required=True); parser.add_argument('--operator-commit',required=True)
    parser.add_argument('--job',required=True); parser.add_argument('--completed-outer',type=int,default=39)
    parser.add_argument('--next-outer',type=int,default=40); parser.add_argument('--observations',type=int,default=180)
    parser.add_argument('--interval',type=int,default=60); parser.add_argument('--login-host',default='10.120.48.32')
    parser.add_argument('--login-identity-json')
    args=parser.parse_args(argv)
    try:
        remote.arguments(args.job,args.runtime_commit,args.operator_commit)
        host,pin=shared.resolve_login_route(args.login_host,args.login_identity_json)
        if not (39<=args.completed_outer<args.next_outer<=200 and 1<=args.observations<=180 and 10<=args.interval<=60):
            raise ValueError('Require target39..199, later next<=200, observations1..180, interval10..60')
    except ValueError as error: parser.error(str(error))
    workspace=Path(__file__).resolve().parents[3].parents[1]
    bridge=shared.load_bridge(workspace)
    output=workspace/'outputs/easy2560-qwen7b-20261021'/('cp34-recovery-watch-'+str(time.time_ns()))
    output.mkdir(parents=True); previous=None
    for index in range(args.observations):
        wrapper={}
        try:
            result=bridge.run('jxu957',command(args.job,args.runtime_commit,args.operator_commit),'',str(workspace/'work'),
                login_host=host,expected_host_identity=pin,observation_timeout=55)
            wrapper=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            if result.returncode: raise ValueError('Read-only transport returned '+str(result.returncode))
            summary=summarize(json.loads(result.stdout),args.job,args.runtime_commit,args.completed_outer,args.next_outer)
        except Exception as error:
            summary=dict(epoch=time.time(),observation_unknown=True,observation_error=str(error),
                inspection_issues=['Read-only observation failed; training failure is not inferred'],
                requested_followup_observed=False,training_failure_inferred=False,overall_goal_complete=False,production_mutated=False)
        summary.update(observation=index+1,runtime_commit=args.runtime_commit,operator_commit=args.operator_commit,
                       login_host=host,login_identity=pin)
        stable={k:v for k,v in summary.items() if k not in ('epoch','observation')}
        fingerprint=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
        changed=fingerprint!=previous
        if changed or summary.get('inspection_issues'):
            path=output/('observation-%03d.json' % (index+1)); shared.store(path,wrapper)
            summary['full_observation_path']=str(path)
        with (output/'summaries.jsonl').open('a') as stream:
            stream.write(json.dumps(summary,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        shared.store(output/'latest-summary.json',summary)
        if changed or summary.get('inspection_issues'): print(json.dumps(summary,sort_keys=True),flush=True)
        previous=fingerprint
        if summary.get('inspection_issues'): return 2
        if summary.get('requested_followup_observed'): return 0
        if index+1<args.observations: time.sleep(args.interval)
    return 3


if __name__=='__main__': raise SystemExit(main())
