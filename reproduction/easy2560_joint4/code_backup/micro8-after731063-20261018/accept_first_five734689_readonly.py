"""Read existing734689 native receipts for6..10 plus real11; never write or train."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro8-after731063-20261018')
JOB = '734689'
COMMIT = '4cc07f3103917f5ac6bb81f6b022d454809c1054'
RUN = 'dfe9c94084bb48b2a9bb7b60f9d882da'
CODE = BASE/'source/28e839d3649782598392d2c1424c9aa277d53733'
CHAIN_SHA = 'a503352c5113834be34735e3b8e7f8d0d071483d8edd6ebd7ae71ebd40166db9'


class Pending(Exception): pass


def need(ok, message):
    if not ok: raise ValueError('FIRST_FIVE734689_' + message)


def digest(value):
    raw = json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def context(row): return row.get('context') or row


def same(row, attempt):
    c = context(row)
    return (c.get('attempt_index'),c.get('attempt_id')) == (attempt['attempt_index'],attempt['attempt_id'])


def read_prefix(path, *, jsonl=False, next11=False):
    """Hash only the stable consumed prefix; a later append is not a failure."""
    if not path.is_file(): raise Pending('missing '+str(path))
    parts=[]; rows=[]; evidence=None; epoch_header=None; errors=[]; length=0
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        for line in stream:
            need(len(line)<=2*1024**2,'METADATA_LINE_TOO_LARGE')
            if not line.endswith(b'\n') and (jsonl or next11): break
            if jsonl:
                row=json.loads(line)
                if context(row).get('logical_update',row.get('step',0)) > 10: break
                rows.append(row)
            parts.append(line)
            length+=len(line)
            if next11:
                text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',line.decode(errors='replace'))
                if any(marker in text for marker in ('Traceback (most recent call last)','CUDA out of memory',
                    'RayTaskError(','SEARCH_G1_FATAL_SCHEDULER_EVIDENCE=')):
                    errors.append(text.strip()[-1000:])
                match=re.search(r'(\{"stage":\s*"mixed_candidate_fill".*)',text)
                if match:
                    candidate=json.JSONDecoder().raw_decode(match.group(1))[0]
                    if candidate.get('step') == 11 and candidate.get('candidates',0)>0:
                        evidence=dict(kind='native_candidate_progress',value=candidate)
                if re.search(r'epoch\s+\d+,\s+step\s+11(?:\D|$)',text):
                    epoch_header=text.strip()[-500:]
                if evidence: break
            need(length<=128*1024**2,'METADATA_PREFIX_LIMIT')
        raw=b''.join(parts); after=os.fstat(stream.fileno())
        need((before.st_dev,before.st_ino,before.st_uid)==(after.st_dev,after.st_ino,after.st_uid) and
             after.st_size>=len(raw),'PREFIX_FILE_REPLACED_OR_TRUNCATED')
        stream.seek(0);need(stream.read(len(raw))==raw,'CONSUMED_PREFIX_CHANGED')
    end=path.stat();need((end.st_dev,end.st_ino)==(before.st_dev,before.st_ino),'PREFIX_PATH_REPLACED')
    record=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),prefix_only=True)
    if next11: record.update(error_lines=errors[-5:],native11_epoch_header=epoch_header)
    return (rows if jsonl else evidence if next11 else raw),record


def terminal(update, budget, schedule, rank):
    step=update['step']; before=update['optimizer_writes_before']; after=update['optimizer_writes_after']
    writes=update['actual_optimizer_writes']; processed=update['opportunities_processed']
    need(type(writes) is int and 1<=writes<=processed<=20 and after-before==writes,'UPDATE_COUNTERS_DIFFER')
    need(update['full_budget_complete'] is (writes==processed==20) and
         update['opportunities_unattempted']==20-processed and
         update['update_status']==('complete' if writes==processed==20 else 'partial'),'UPDATE_BUDGET_FLAGS_DIFFER')
    need(same(budget,update) and same(schedule,update),'TERMINAL_ATTEMPT_DIFFERS')
    need((budget['run_id'],budget['rank'],budget['logical_update'],budget['kind'])==(RUN,rank,step,'OUTER_BUDGET'),
         'NATIVE_WORKER_TERMINAL_IDENTITY_DIFFERS')
    need(tuple(budget[k] for k in ('local_trajectories','local_mini','local_micro','local_logical_micro','gradient_accumulation'))
         ==(640,32,8,4,4),'NATIVE_PHYSICAL_OR_LOGICAL_GEOMETRY_DIFFERS')
    need((budget['optimizer_writes_before'],budget['optimizer_writes_after'],budget['cumulative_writes'],
          budget['optimizer_writes'],budget['opportunities_processed'])==(before,after,after,writes,processed),
         'NATIVE_BUDGET_COUNTS_DIFFER')
    need(budget['full_budget_complete'] is (writes==20 and processed==20),'NATIVE_FULL_BUDGET_FLAG_DIFFERS')
    need((schedule['kind'],schedule['run_id'],schedule['rank'],schedule['world_size'],schedule['slurm_job_id'],
          schedule['source_commit'],schedule['logical_update'])==('COMPLETE',RUN,rank,4,JOB,COMMIT,step),
         'NATIVE_SCHEDULER_IDENTITY_DIFFERS')
    expected=dict(completed_outer_before=step-1,completed_outer_after=step,scheduler_last_epoch_before=step-1,
        scheduler_last_epoch_after=step,scheduler_step_count_before=step,scheduler_step_count_after=step+1,
        writes_before=before,writes_after=after,actual_optimizer_writes=writes,opportunities_processed=processed,
        optimizer_opportunities=20,scheduler_advances=1,warmup_steps=57)
    need(all(type(schedule[k]) is int and schedule[k]==v for k,v in expected.items()),'NATIVE_SCHEDULER_COUNTS_DIFFER')
    need(schedule['batch_gate_sha256']==update['behavior_probability_gate_sha256'],'NATIVE_GATE_LINK_DIFFERS')
    for key,expected_lr in [('consumed_learning_rates',1e-6*(step-1)/57),('next_learning_rates',1e-6*step/57)]:
        need(schedule[key] and all(math.isclose(v,expected_lr,rel_tol=1e-12,abs_tol=1e-18) for v in schedule[key]),
             'NATIVE_WARMUP_DIFFERS')
    return dict(rank=rank,processed=processed,writes=writes,cumulative_writes=after,scheduler_outer=step,
                physical_micro=8,logical_micro=4,accumulation=4)


def inspect(data):
    run=data['run']; chain=data['restore_chain']
    need((run['run_id'],run['source_commit'],run['training_world_size'],run['actor_micro_batch_per_rank'],
          run['actor_logical_micro_batch_per_rank'],run['all_correct_virtual_zero'])==(RUN,COMMIT,4,8,4,False),
         'CURRENT_MANIFEST_DIFFERS')
    need((run['budget']['ppo_mini_batch_size'],run['budget']['ppo_micro_batch_size'],
          run['budget']['per_rank_actor_logical_micro_batch'])==(128,32,4),'CURRENT_GLOBAL_BATCH_DIFFERS')
    transition=run['full_state_restore']['physical_micro_transition']
    need((transition['source_physical_micro_per_rank'],transition['target_physical_micro_per_rank'],
          transition['target_accumulation_steps'],transition['global_mini_batch'],transition['logical_micro_per_rank'])
         ==(16,8,4,128,4),'BOUND_PHYSICAL_FALLBACK_DIFFERS')
    need(chain['current_job']==JOB and chain['current_source_commit']==COMMIT and chain['source_job']=='730226' and
         chain['completed_outer']==5 and chain['historical_optimizer_writes']==100 and
         chain['native4_worker_driver_restore_proofs_authenticated'] is True,'ACTUAL_CP5_RESTORE_PROOFS_REQUIRED')
    updates=data['updates']; need([u['step'] for u in updates]==list(range(6,6+len(updates))) and
                                all(u['run_id']==RUN for u in updates),'CURRENT_UPDATE_PREFIX_DIFFERS')
    result=dict(status='PENDING_FIRST_FIVE_OR_NATIVE11',accepted=False,job=JOB,source_commit=COMMIT,run_id=RUN,
        completed_outers=[u['step'] for u in updates],required_completed_outers=[6,7,8,9,10],
        original_checkpoint5_actual_restore=chain,native_next11=data.get('next11'),outers=[],attempts=[],
        source_reads=data.get('source_reads',[]),read_only=True,Tensor_or_model_reads=0,Torch_imports=0,
        GPU_allocations=0,training_updates=0,mutations=0,goal_complete=False,checked_unix=time.time())
    attempts=data['attempts']; prior_writes=100; completed=5; prior_index=chain['historical_attempt_index']
    successes={u['attempt_id']:u for u in updates}; pending=False
    for attempt in attempts:
        index=attempt['attempt_index'];aid=attempt['attempt_id'];step=attempt['logical_update']
        need(index==prior_index+1 and step==completed+1 and attempt['run_id']==RUN and attempt['slurm_job_id']==JOB and
             attempt['real_rank_count']==4 and attempt['all_ranks_agreed'] is True,'DRIVER_ATTEMPT_SEQUENCE_DIFFERS')
        writes=attempt['optimizer_writes_this_attempt'];processed=attempt['opportunities_processed']
        need(type(writes) is int and 0<=writes<=processed<=20 and attempt['optimizer_writes_before']==prior_writes and
             attempt['optimizer_writes_after']==prior_writes+writes,'DRIVER_ATTEMPT_COUNTS_DIFFER')
        statuses=attempt['opportunity_statuses']; retained=[v in ('written','written_with_post_guard') for v in statuses]
        need(len(statuses)==processed and sum(retained)==writes and attempt['complete_outer'] is (writes>0) and
             attempt['scheduler_advances']==int(writes>0) and attempt['completed_outer']==completed+int(writes>0),
             'DRIVER_OPPORTUNITY_DISPOSITION_DIFFERS')
        need(attempt['opportunities_unattempted']==20-processed and
             attempt['full_budget_complete'] is (writes==processed==20),'DRIVER_BUDGET_DISPOSITION_DIFFERS')
        skipped=statuses.count('skipped_guard');rollbacks=statuses.count('rolled_back_guard')
        need(writes+skipped+rollbacks==processed and attempt['optimizer_rollbacks_this_attempt']==rollbacks and
             attempt['optimizer_calls_this_attempt']==writes+rollbacks,'DRIVER_CALL_SKIP_ROLLBACK_DIFFERS')
        update=successes.get(aid)
        if writes and update is None: pending=True;break
        per_rank=[]
        for rank in range(4):
            worker=[r for r in data['optimizers'][str(rank)] if same(r,attempt)]
            schedule=[r for r in data['schedulers'][str(rank)] if same(r,attempt)]
            if not worker or not schedule or schedule[-1]['kind'] not in ('COMPLETE','ABORTED_ATTEMPT'):
                pending=True;break
            need(all((s['run_id'],s['rank'],s['world_size'],s['slurm_job_id'],s['source_commit'],s['logical_update'])==
                 (RUN,rank,4,JOB,COMMIT,step) for s in schedule),'NATIVE_SCHEDULER_ATTEMPT_IDENTITY_DIFFERS')
            need(all(r['run_id']==RUN and r['rank']==rank and context(r)['logical_update']==step for r in worker),
                 'NATIVE_WORKER_ATTEMPT_IDENTITY_DIFFERS')
            opportunities=[r for r in worker if r['kind']=='OPPORTUNITY']
            write_rows=[r for r in worker if r['kind']=='OPTIMIZER_WRITE']
            need([r['opportunity_index'] for r in opportunities]==list(range(processed)) and
                 [r['status'] for r in opportunities]==statuses,'NATIVE_OPPORTUNITY_ORDER_OR_STATUS_DIFFERS')
            cumulative=prior_writes
            for i,row in enumerate(opportunities):
                need(row['processed'] is True and context(row)['minibatch_index']==i and
                     row['optimizer_write_succeeded'] is retained[i] and row['optimizer_writes_before']==cumulative,
                     'NATIVE_OPPORTUNITY_WRITE_BINDING_DIFFERS')
                need(row['optimizer_write_rolled_back'] is (statuses[i]=='rolled_back_guard') and
                     row['optimizer_call_attempted'] is (retained[i] or statuses[i]=='rolled_back_guard'),
                     'NATIVE_SKIP_ROLLBACK_CALL_DISPOSITION_DIFFERS')
                cumulative+=int(retained[i]);need(row['optimizer_writes_after']==cumulative,'NATIVE_OPPORTUNITY_COUNTER_DIFFERS')
            need([r['write_count'] for r in write_rows]==list(range(prior_writes+1,prior_writes+writes+1)) and
                 [context(r)['minibatch_index'] for r in write_rows]==[i for i,v in enumerate(retained) if v],
                 'NATIVE_RETAINED_WRITE_SEQUENCE_DIFFERS')
            need(all(r['learning_rates'] and all(math.isclose(lr,1e-6*(step-1)/57,rel_tol=1e-12,abs_tol=1e-18)
                 for lr in r['learning_rates']) for r in write_rows),'NATIVE_WRITE_LR_DIFFERS')
            if writes:
                need([r['kind'] for r in schedule]==['BEGIN','COMMIT_PREPARED','COMPLETE'],'NATIVE_SCHEDULER_PHASE_DIFFERS')
                budgets=[r for r in worker if r['kind']=='OUTER_BUDGET']
                need(len(budgets)==1,'NATIVE_TERMINAL_BUDGET_REQUIRED')
                per_rank.append(terminal(update,budgets[0],schedule[-1],rank))
                need(schedule[-1]['source_map_sha256']==digest(run['source_sha256']) and
                     schedule[-1]['scheduler_contract_sha256']==digest(run['budget']),'NATIVE_CONFIG_SHA_DIFFERS')
                restored=data['restored_workers'][str(rank)]['process_identity']
                need((schedule[-1]['process_pid'],schedule[-1]['process_start_ticks'])==
                     (restored['pid'],restored['start_ticks']),'NATIVE_RESTORED_PROCESS_BIRTH_DIFFERS')
            else:
                need([r['kind'] for r in schedule]==['BEGIN','ABORTED_ATTEMPT'] and
                     schedule[-1]['scheduler_advances']==0 and schedule[-1]['writes_after']==prior_writes and
                     schedule[-1]['completed_outer_after']==completed,'ABORTED_SCHEDULER_CHANGED')
                aborted=[r for r in worker if r['kind']=='ABORTED_ATTEMPT']
                need(len(aborted)==1 and aborted[0]['optimizer_writes']==0 and
                     aborted[0]['cumulative_writes']==prior_writes and aborted[0]['opportunities_processed']==processed,
                     'ABORTED_WORKER_TERMINAL_DIFFERS')
        if pending:break
        result['attempts'].append(dict(logical_outer=step,attempt_index=index,attempt_id=aid,processed=processed,
            writes=writes,skips=skipped,rollbacks=rollbacks,unattempted=20-processed,complete_outer=writes>0))
        if writes:
            need(update['optimizer_writes_before']==prior_writes and update['optimizer_writes_after']==prior_writes+writes and
                 update['actual_optimizer_writes']==writes and update['opportunities_processed']==processed,
                 'COMPLETE_UPDATE_ATTEMPT_DIFFERS')
            result['outers'].append(dict(outer=step,attempt_index=index,attempt_id=aid,opportunities=20,processed=processed,
                writes=writes,skips=skipped,rollbacks=rollbacks,unattempted=20-processed,ranks=per_rank))
            completed+=1
        prior_writes+=writes;prior_index=index
    result.update(cumulative_retained_Adam_writes_per_rank=prior_writes,validated_completed_outer=completed,
                  partial_native_receipt_pending=pending)
    result['validated_attempt_totals']={k:sum(a[k] for a in result['attempts']) for k in
                                      ('processed','writes','skips','rollbacks','unattempted')}
    if data.get('error_lines') or data.get('pipeline_stage') in ('FAILED','OWNED_CLEANUP_FAILED'):
        result.update(status='ATTENTION_REQUIRED_NEW_SOURCE_ERROR',error_lines=data.get('error_lines',[]));return result
    next11=data.get('next11') or {}
    if completed==10 and len(result['outers'])==5 and not pending and next11.get('kind')=='native_candidate_progress' and \
            next11.get('value',{}).get('step')==11 and next11['value'].get('candidates',0)>0:
        full=all(row['processed']==20 and row['unattempted']==0 for row in result['outers'])
        result.update(status='RECOVERY_FIVE6_10_AND_NATIVE11_ACCEPTED' if full else 'ATTENTION_REQUIRED_PROCESSED_BUDGET_INCOMPLETE',
                      accepted=full,all_five_processed20=full)
    return result


def capture():
    need((BASE/'job.id').read_text().strip()==JOB,'ACTUAL_JOB_ID_DIFFERS')
    source_reads=[]
    def small(path):
        raw,record=read_prefix(path);source_reads.append(record);return json.loads(raw)
    binding=small(CODE/'reproduction/easy2560_joint4/code_backup/micro8-after731063-20261018/current-readonly-followup-commands.json')
    need((binding['job'],binding['source_commit'],binding['base'],binding['read_only'])==(JOB,COMMIT,str(BASE),True),
         'PUBLISHED_CURRENT_READONLY_BINDING_DIFFERS')
    run=small(BASE/'logs/run-manifest.json')
    helper=CODE/'reproduction/easy2560_joint4/checkpoint_evidence.py'
    need(hashlib.sha256(helper.read_bytes()).hexdigest()==CHAIN_SHA,'CPU_CHAIN_HELPER_SHA_DIFFERS')
    sys.path.insert(0,str(CODE))
    from reproduction.easy2560_joint4 import checkpoint_evidence
    need(Path(checkpoint_evidence.__file__)==helper,'CPU_CHAIN_HELPER_IMPORT_DIFFERS')
    try: chain=checkpoint_evidence.authenticated_restore_chain(BASE,run)['public']
    except FileNotFoundError as error: raise Pending('restore proof not yet complete: '+str(error.filename))
    restored=small(BASE/'logs/full-state-restored-workers.json')
    stage=small(BASE/'logs/pipeline-status.json')
    def ledger(path):
        rows,record=read_prefix(path,jsonl=True);source_reads.append(record);return rows
    updates=ledger(BASE/'logs/actual-updates.jsonl');attempts=ledger(BASE/'logs/actual-attempts.jsonl')
    optimizers={str(r):ledger(BASE/f'logs/optimizer-rank-{r}.jsonl') for r in range(4)}
    schedulers={str(r):ledger(BASE/f'logs/scheduler-rank-{r}.jsonl') for r in range(4)}
    next11,record=read_prefix(BASE/'logs/search-g1-common8-training.log',next11=True);source_reads.append(record)
    return dict(run=run,restore_chain=chain,restored_workers={str(r['rank']):r for r in restored['ranks']},
        updates=updates,attempts=attempts,optimizers=optimizers,schedulers=schedulers,next11=next11,
        source_reads=source_reads,pipeline_stage=stage.get('stage'),error_lines=record.get('error_lines',[]))


def main():
    try: result=inspect(capture())
    except (Pending,FileNotFoundError) as error:
        result=dict(status='PENDING_FIRST_FIVE_OR_NATIVE11',accepted=False,detail=str(error),job=JOB,goal_complete=False)
    except (ValueError,KeyError,TypeError,AssertionError) as error:
        result=dict(status='ATTENTION_REQUIRED_RECEIPT_OR_SCHEMA_DIFFERS',accepted=False,detail=str(error),job=JOB,goal_complete=False)
    print(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
