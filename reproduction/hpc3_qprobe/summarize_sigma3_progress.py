"""Pure CPU partial-progress summary of actual held RAW0cca sigma3 receipts.

No Torch, model, generation, fitting, writes or job actions. Reconsumes each
selected original held child0 and freshly hashes its scalar output/head bytes.
This does NOT re-read original rollout/private-trace tensors, replace final200
raw acceptance, or accept native restoration. AUC/concordance are means of
per-outer, within-policy metrics, never pooled across changing policies.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from reproduction.hpc3_qprobe import final_control as ctl
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader

BASE=reader.BASE
RUN=reader.RUN
PRODUCER=reader.PRODUCER
JOB=reader.JOB
RAW=reader.RAW
GROUPS={'all':512,'random_exploration':64,'predicted_diversity':448}
STD=tuple(math.sqrt(k*(5-k)/20) for k in range(6))
STATUS='ACTUAL_SIGMA3_PARTIAL_AUDITED_PREQUENTIAL_PROGRESS_SUMMARY'


def need(ok,message):ctl.need(ok,'SIGMA3_PROGRESS: '+message)


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def exact_count(fraction,n):
    need(finite(fraction) and 0<=fraction<=1,'invalid measured fraction')
    count=round(fraction*n)
    need(abs(fraction*n-count)<=1e-8,'mixed fraction is not an exact integer count')
    return count


def check_metrics(canonical):
    metrics=canonical['prequential'];distribution=canonical['K_distribution']
    need(set(distribution)=={str(k) for k in range(6)} and
         all(type(v) is int and v>=0 for v in distribution.values()) and sum(distribution.values())==512,
         'strict G5 K0..5 distribution differs')
    need(type(canonical['strict_correct']) is int and canonical['strict_correct']==
         sum(k*distribution[str(k)] for k in range(6)),'strict correct count differs')
    need(set(metrics)==set(GROUPS),'prequential cohort names differ')
    for name,n in GROUPS.items():
        value=metrics[name]
        need(type(value['questions']) is int and value['questions']==n,'fixed E64/top448 cohort size differs')
        for key in ('std_mae','std_mse','actual_sample_std_mean','count_nll','mean_mae','mixed_brier'):
            need(finite(value[key]) and value[key]>=0,'prequential metric invalid: '+key)
        need(value['actual_sample_std_mean']<=STD[2]+1e-10,'actual sigma exceeds G5 bound')
        for key in ('actual_mixed_fraction','all_wrong_fraction','all_correct_fraction'):
            exact_count(value[key],n)
        need(sum(exact_count(value[key],n) for key in ('actual_mixed_fraction','all_wrong_fraction','all_correct_fraction'))==n,
             'cohort mixed/allwrong/allcorrect do not partition all questions')
        need(finite(value['actual_em_mean']) and 0<=value['actual_em_mean']<=1,'strict EM mean invalid')
        for key in ('std_mixed_auc','std_pairwise_concordance'):
            if key in value:need(finite(value[key]) and 0<=value[key]<=1,'within-policy rank statistic invalid')
    actual=metrics['all'];random=metrics['random_exploration'];top=metrics['predicted_diversity']
    for key in ('actual_mixed_fraction','all_wrong_fraction','all_correct_fraction','actual_sample_std_mean',
                'std_mae','std_mse','actual_em_mean','count_nll','mean_mae','mixed_brier'):
        need(close(actual[key],(64*random[key]+448*top[key])/512),'prequential cohort weighted mean differs: '+key)
    mixed=512-distribution['0']-distribution['5']
    need(exact_count(actual['actual_mixed_fraction'],512)==mixed and
         exact_count(actual['all_wrong_fraction'],512)==distribution['0'] and
         exact_count(actual['all_correct_fraction'],512)==distribution['5'] and
         close(actual['actual_em_mean'],canonical['strict_correct']/2560) and
         close(actual['actual_sample_std_mean'],math.fsum(distribution[str(k)]*STD[k] for k in range(6))/512),
         'K distribution/mixed/strict EM/actual sample sigma disagree')
    return metrics


def head_event(task,canonical,retained):
    root=BASE/'artifacts/question-probe';folder='completed' if task['complete_outer'] else 'aborted'
    expected=root/folder/(task['attempt_id']+'.json');bound=canonical['head_file']
    need(bound['path']==str(expected),'head scalar path is outside the actual attempt')
    payload,proof=reader.read_file(expected,expected_sha=bound['sha256'],expected_bytes=bound['bytes'],capture=True)
    retained.remember(proof);event=json.loads(payload)
    need(event['outer']==task['step'] and event['attempt_id']==task['attempt_id'], 'head scalar attempt differs')
    if not task['complete_outer']:
        need(event['stage']=='question_probe_attempt_aborted' and event['head_updated'] is False and
             event['labels_observed'] is True and 'head_update' not in event,'guard unexpectedly fitted a head')
        return None,proof
    need(event['stage']=='question_probe_outer_committed' and event['prequential']==canonical['prequential'] and
         event['successful_actor_snapshot_id']==RUN+':'+str(task['step']), 'head/canonical prequential differs')
    update=event['head_update']
    need(update['completed_outer']==update['update_count']==task['step'] and
         update['label_policy_snapshot_id']==event['label_policy_snapshot_id'] and
         type(update['head_optimizer_updates']) is int and update['head_optimizer_updates']>0 and
         type(update['replay_groups']) is int and update['replay_groups']>=512,'postfit head counters differ')
    for key in ('replay_weighted_loss','sigma_ce','std_mse','count_ce','mean_mse'):
        need(finite(update[key]) and update[key]>=0,'nonfinite/negative postfit loss component')
    calculated=update['sigma_ce']+update['std_mse']+.1*update['count_ce']
    need(abs(calculated-update['replay_weighted_loss'])<=2e-6,'postfit CE3 + normalized sigma MSE + .1 CE6 formula differs')
    return dict(weighted_replay_loss=update['replay_weighted_loss'],sigma_ce3=update['sigma_ce'],
        normalized_sigma_mse=update['std_mse'],auxiliary_ce6=update['count_ce'],auxiliary_weight=.1,
        mean_mse_diagnostic_only_weight_zero=update['mean_mse'],formula_abs_error=abs(calculated-update['replay_weighted_loss']),
        replay_groups=update['replay_groups'],head_optimizer_updates_this_fit=update['head_optimizer_updates']),proof


def aggregate(outers):
    need(outers,'no successful audited outers to aggregate')
    groups={}
    for name in GROUPS:
        values=[o['prequential'][name] for o in outers];n=sum(v['questions'] for v in values)
        mixed=sum(exact_count(v['actual_mixed_fraction'],v['questions']) for v in values)
        group=dict(questions=n,mixed_questions=mixed,mixed_rate=mixed/n)
        for original,label in (('actual_sample_std_mean','actual_sigma_sample_ddof1_mean'),
                               ('std_mae','prequential_sigma_mae_raw_scale'),('std_mse','prequential_sigma_mse_raw_scale')):
            group[label]=math.fsum(v[original]*v['questions'] for v in values)/n
        for key in ('std_mixed_auc','std_pairwise_concordance'):
            available=[v[key] for v in values if key in v]
            group[key+'_within_policy_outer_mean']=math.fsum(available)/len(available) if available else None
            group[key+'_defined_outers']=len(available)
        groups[name]=group
    top=groups['predicted_diversity'];random=groups['random_exploration'];rate=random['mixed_rate']
    return dict(first_outer=outers[0]['outer'],last_outer=outers[-1]['outer'],successful_outers=len(outers),
        cohorts=groups,top_minus_random_mixed_pp=100*(top['mixed_rate']-rate),
        top_over_random_mixed_rate_ratio=top['mixed_rate']/rate if rate else None,
        ratio_defined=bool(rate),K_distribution={str(k):sum(o['K_distribution'][str(k)] for o in outers) for k in range(6)},
        ranking_scope='arithmetic mean of defined per-outer within-policy statistics; never cross-policy pooled AUC/concordance')


def accepted_prefix(raw,parts):
    folder=raw.DIRECTORY/'accepted';s=folder.lstat()
    need(folder.resolve()==folder and stat.S_ISDIR(s.st_mode) and s.st_uid==ctl.OWNER,'unsafe accepted receipt directory')
    with os.scandir(folder) as entries:
        paths=sorted(Path(e.path) for e in entries if e.name.endswith('.json'))
    need(paths and [p.name for p in paths]==['%06d.json'%i for i in range(1,len(paths)+1)],
         'accepted receipt namespace is not an exact contiguous attempt prefix')
    tasks=raw.actual_tasks(parts);need(len(tasks)>=len(paths),'accepted prefix extends beyond actual normal-return attempts')
    values=[]
    for i,path in enumerate(paths):
        record,proof=reader.document(path)
        need(record['task']==tasks[i],'accepted receipt is not the actual guarded attempt prefix')
        values.append((record,proof))
    successes=[v[0]['task']['step'] for v in values if v[0]['task']['complete_outer']]
    need(successes==list(range(1,len(successes)+1)) and successes,'successful audited prefix has a gap or is empty')
    return values,tasks


def summarize(commit,*,max_outer=None):
    ctl.owner();ctl.caps()
    need(sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','bytecode writes must be disabled')
    repository,source=ctl.source(commit)
    need(Path(__file__).resolve()==repository/'reproduction/hpc3_qprobe/summarize_sigma3_progress.py' and
         os.environ.get('QPROBE_SOURCE')==str(repository),'summary must load its own published immutable source')
    raw,raw_source=reader.load_fixed(RAW,'raw_audit_followthrough_sigma3','ctl')
    _,verified=raw.validate(RAW);need(verified==raw_source and raw.BASE==BASE,'fixed RAW0cca module source/scope differs')
    claim,claim_file=reader.document(raw.DIRECTORY/'claim.json');supervisor,process_file=reader.document(raw.DIRECTORY/'process.json')
    parts=claim['parts']
    need(claim['source']==raw_source and claim['training_control_permitted'] is False and claim['goal_complete'] is False and
         len(parts)==1 and parts[0]['first']==1 and parts[0]['last']==200 and
         raw.birth_ok(supervisor) and supervisor['sid']==supervisor['pid'],'RAW original once claim/birth differs')
    producer=parts[0]['producer']
    need(producer['base']==str(BASE) and producer['run_id']==RUN and producer['producer_job']==JOB and
         producer['producer_commit']==PRODUCER,'RAW original fresh producer identity differs')
    run,manifest=reader.document(BASE/'logs/run-manifest.json')
    need(run['run_id']==RUN and run['source_commit']==PRODUCER and run['training_world_size']==4 and not run.get('full_state_restore') and
         all(run['question_probe'].get(k)==v for k,v in reader.LOSS.items()),'actual sigma3 producer/loss identity differs')
    accepted,tasks=accepted_prefix(raw,parts)
    available=sum(r[0]['task']['complete_outer'] for r in accepted)
    need(max_outer is None or type(max_outer) is int and 1<=max_outer<=available,'max_outer exceeds the successful audited prefix')
    last=available if max_outer is None else max_outer
    # An explicit earlier outer stops at that success; default consumes all
    # currently accepted guards, including any guards for the next outer.
    count=len(accepted) if max_outer is None else next(i+1 for i,(r,_) in enumerate(accepted)
        if r['task']['complete_outer'] and r['task']['step']==last)
    selected=accepted[:count];retained=reader.SharedReads()
    for proof in (claim_file,process_file,manifest):retained.remember(proof)
    outers=[];proofs=[];guards=[]
    for record,accepted_file in selected:
        task=record['task'];folder=raw.DIRECTORY/'attempts'/task['attempt_id']
        consumed=raw.consume_task(task,folder,supervisor)
        need(consumed==record,'accepted receipt differs from actual held child/output')
        output,output_file=reader.document(folder/'audit.stdout')
        need(output_file['sha256']==consumed['output_sha256'],'actual held stdout changed after consumption')
        # Every original result/child/claim/output is small metadata here.
        result_file=reader.read_file(folder/'result.json',expected_sha=consumed['result_sha256'])[1]
        retained.remember(result_file)
        for path in (folder/'claim.json',folder/'child.json',folder/'audit.stderr'):
            retained.remember(reader.read_file(path)[1])
        for proof in (accepted_file,output_file):retained.remember(proof)
        canonical=output['canonical_proof'];metrics=check_metrics(canonical)
        loss,head_file=head_event(task,canonical,retained)
        item=dict(attempt_index=task['attempt_index'],attempt_id=task['attempt_id'],outer=task['step'],
            complete_outer=task['complete_outer'],accepted=accepted_file,result=result_file,
            output=output_file,head=head_file)
        proofs.append(item)
        if task['complete_outer']:
            outers.append(dict(outer=task['step'],attempt_id=task['attempt_id'],prequential=metrics,
                K_distribution=canonical['K_distribution'],postfit_loss=loss))
        else:guards.append(dict(outer=task['step'],attempt_id=task['attempt_id'],attempt_index=task['attempt_index']))
    need([o['outer'] for o in outers]==list(range(1,last+1)),'consumed successful outer prefix differs')
    # The original live producer may append later tasks. Only this immutable
    # bounded prefix must remain equal; a partial active ledger fails closed.
    need(raw.actual_tasks(parts)[:count]==[r[0]['task'] for r in selected],'actual guarded attempt prefix changed during summary')
    need([raw.consume_task(r[0]['task'],raw.DIRECTORY/'attempts'/r[0]['task']['attempt_id'],supervisor) for r in selected]==
         [r[0] for r in selected],'original held task/result graph changed during summary')
    retained.finish()
    return dict(status=STATUS,run_id=RUN,producer_job=JOB,producer_commit=PRODUCER,source=source,raw_source=raw_source,
        successful_outer_prefix=last,accepted_successful_outer_prefix_available=available,
        normal_return_attempts_consumed=count,fully_skipped_guard_attempts=len(guards),guards=guards,
        cumulative=aggregate(outers),recent10=aggregate(outers[-10:]),per_outer=outers,proofs=proofs,
        raw_original_claim=claim_file,raw_original_supervisor=process_file,
        goal_complete=False,actor_model_loaded=False,generated_trajectories=0,new_updates=0,evidence_writes=0,
        scope='partial audited successful prefix only; fresh scalar/head hashes; not fresh raw/private re-decode or final200/native4 acceptance',
        checked_unix=time.time())


def main():
    for name in ctl.CAPS:os.environ[name]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True);p.add_argument('--max-outer',type=int)
    a=p.parse_args();print(json.dumps(summarize(a.commit,max_outer=a.max_outer),allow_nan=False),flush=True)


if __name__=='__main__':main()
