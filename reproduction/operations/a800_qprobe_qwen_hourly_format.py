"""Format one saved native observer receipt against a prior delivered anchor.

Pure local scalar/hash projection: no network, model, tensor or anchor writes.
The supplied anchor declares an actual earlier delivery; this tool cannot prove
message delivery and never advances that anchor on observation alone.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


IDENTITY=('run_id','execution_id','source_commit','base')


def number(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None


def scalars(value,names):
    return {key:number(value.get(key)) for key in names}


def state(value):
    result={'status':value.get('status') if value.get('status') in ('observed','unknown','missing') else 'unknown'}
    if result['status']!='observed':
        reason=value.get('reason');result['reason']=reason if isinstance(reason,str) and re.fullmatch('[A-Za-z0-9_]{1,96}',reason) else 'not_available_in_saved_receipt'
    return result


def covered(section,anchor,completed,key):
    if section.get('status')!='observed':return None
    start=section.get('after_complete_step');rows=section.get(key)
    if type(start) is not int or type(completed) is not int or not start<=anchor<=completed or not isinstance(rows,list):return None
    if [row.get('step') for row in rows]!=list(range(start+1,completed+1)):return None
    return [row for row in rows if row['step']>anchor]


def format_summary(receipt,anchor):
    if receipt.get('role')!='single_native_qwen_primary_metadata_receipt':raise ValueError('NATIVE_HOURLY_WRONG_RECEIPT_ROLE')
    observation=receipt.get('observation') or {}
    if observation.get('role')!='qprobe_native_fresh_primary_metadata_readonly':raise ValueError('NATIVE_HOURLY_WRONG_OBSERVER_ROLE')
    for key in IDENTITY:
        if not receipt.get(key) or receipt[key]!=observation.get(key) or receipt[key]!=anchor.get(key):
            raise ValueError('NATIVE_HOURLY_CROSS_RUN_OR_EXECUTION_'+key)
    if (not all(re.fullmatch('[0-9a-f]{32}',receipt[k]) for k in ('run_id','execution_id')) or
            not re.fullmatch('[0-9a-f]{40}',receipt['source_commit']) or
            not re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',receipt['base'])):
        raise ValueError('NATIVE_HOURLY_IDENTITY_SCHEMA')
    if (anchor.get('delivered') is not True or type(anchor.get('completed_outer')) is not int or
            not 0<=anchor['completed_outer']<=200 or number(anchor.get('delivered_unix')) is None or anchor['delivered_unix']<=0):
        raise ValueError('NATIVE_HOURLY_REQUIRES_ACTUAL_DELIVERED_ANCHOR')
    step=anchor['completed_outer'];reward=observation.get('reward_window',{});counts=observation.get('opportunity_counts',{})
    completed=reward.get('recorded_complete_outers') if reward.get('status')=='observed' else None
    rows=covered(reward,step,completed,'new_complete_rows')
    if number(observation.get('observed_unix')) is None or observation['observed_unix']<anchor['delivered_unix']:rows=None
    window={'status':'unknown','reason':'receipt_does_not_cover_delivered_anchor','after_complete_step':step}
    if rows is not None:
        for row in rows:
            if (not all(number(row.get(k)) is not None for k in ('reward_mean','successful_adam','skip','processed')) or
                    row['processed']!=20 or row['successful_adam']<=0 or row['successful_adam']+row['skip']!=20):
                raise ValueError('NATIVE_HOURLY_INVALID_RETURNED_COMPLETE_ROW')
        values=[row['reward_mean'] for row in rows]
        window=dict(status='observed',after_complete_step=step,newly_complete_steps=[row['step'] for row in rows],
            complete_outers=len(rows),processed=sum(row['processed'] for row in rows),
            returned_adam=sum(row['successful_adam'] for row in rows),skip=sum(row['skip'] for row in rows),
            reward_mean=math.fsum(values)/len(values) if values else None,
            reward_min=min(values) if values else None,reward_max=max(values) if values else None,
            scope='complete returned actor outers since prior actual delivery; partial attempts excluded')
    cumulative=state(counts)
    count_names=('current_execution_processed','current_execution_successful_adam','current_execution_skip',
        'current_execution_complete_outer_successful_adam','current_execution_aborted_attempts',
        'current_execution_aborted_attempt_retained_adam','current_execution_closed_attempt_unattempted',
        'current_execution_optimizer_call_attempts','current_execution_optimizer_write_rollbacks')
    trusted=counts.get('status')=='observed' and counts.get('four_rank_complete_prefix_agreement') is True
    cumulative.update(scalars(counts,count_names) if trusted else {key:None for key in count_names})
    partial=counts.get('current_partial') if trusted else None
    cumulative['current_partial']=scalars(partial,('attempt_index','logical_update','processed','successful_adam',
        'processed_skip','opportunities_remaining_or_unattempted')) if isinstance(partial,dict) else None
    cumulative['scope']='current execution four replicas counted once; processed skips exclude unattempted; retained partial Adam is separate from returned complete outers'
    prequential=observation.get('selected_prequential_window',{})
    selected=covered(prequential,step,completed,'records') if rows is not None else None
    acquisition={'status':'unknown','reason':'selected_prefit_window_missing_or_incomplete','after_complete_step':step}
    if selected is not None and all(row.get('status')=='observed' and row.get('selected_questions')==512 and
            all(type(row.get(k)) is int and 0<=row[k]<=512 for k in ('true_mixed_count','predicted_mixed_count','prefit_correct_count')) for row in selected):
        questions=512*len(selected)
        acquisition=dict(status='observed',after_complete_step=step,complete_outers=len(selected),selected_questions=questions,
            true_mixed_count=sum(row['true_mixed_count'] for row in selected),
            true_mixed_fraction=sum(row['true_mixed_count'] for row in selected)/questions if questions else None,
            predicted_mixed_fraction=sum(row['predicted_mixed_count'] for row in selected)/questions if questions else None,
            prefit_binary_accuracy=sum(row['prefit_correct_count'] for row in selected)/questions if questions else None)
    latest_selected=prequential.get('latest',{})
    acquisition['latest']=dict(state(latest_selected),**scalars(latest_selected,('step','selected_questions','true_mixed_count',
        'true_mixed_fraction','predicted_mixed_count','predicted_mixed_fraction','prefit_binary_accuracy'))) if latest_selected.get('status')=='observed' and latest_selected.get('step')==completed else dict(status='unknown',reason='latest_selected_prefit_metric_missing')
    acquisition['scope']='selected512 labels and frozen pre-rollout/pre-fit binary prediction; not post-fit replay or warmup5120'
    head=observation.get('latest_head_commit',{});postfit=state(head)
    if head.get('status')=='observed' and head.get('outer')==completed:
        postfit.update(scalars(head,('outer','postfit_replay_binary_accuracy','raw_age_weighted_replay_loss','replay_mean_raw_weight')))
        postfit.update(scalars(head.get('recorded_head_update',{}),('replay_groups','head_optimizer_updates','head_adam_steps_after')))
        ages=head.get('replay_ages_included');postfit['replay_ages_included']=ages if isinstance(ages,list) and all(type(age) is int and 0<=age<8 for age in ages) else None
    else:postfit=dict(status='unknown',reason='postfit_head_not_same_latest_complete_outer')
    postfit.update(unweighted_replay_total_loss=None,unweighted_replay_total_loss_status='unknown_not_recorded',
        scope='post-fit recent8 replay; raw-age-weighted loss changes with replay and age mass, not a fixed-set fit improvement')
    latest_reward=reward.get('latest',{})
    scores={}
    for name,value in observation.get('scores',{}).get('scores',{}).items():
        if name in ('stage','validation','development_after5'):
            scores[name]=dict(state(value),**scalars(value,('step','weighted_em','weighted_f1','question_count')))
    gpu=observation.get('gpu_samples',{});gpu_summary=dict(state(gpu),**scalars(gpu,('sample_count','latest_sample_unix','maximum_scanned_sampling_gap_seconds')))
    gpu_summary['scan_from_start']=gpu.get('scan_from_start') if type(gpu.get('scan_from_start')) is bool else None
    gpu_summary['latest_devices']=[scalars(item,('index','used_mib','total_mib','util_percent')) for item in gpu.get('latest_devices',[])[:16]] if isinstance(gpu.get('latest_devices'),list) else None
    gpu_summary['complete_step_window_peaks']=dict(status='unknown',reason='not_computed_by_observer')
    gpu_summary['scope']='existing current-execution physical-node samples; sampling gaps retained; no complete-step peak inferred'
    runtime=observation.get('runtime_status',{});stage=runtime.get('pipeline_stage')
    return dict(role='native_qwen_hourly_scalar_summary',**{key:receipt[key] for key in IDENTITY},
        status='observed' if observation.get('status')=='observed' and window['status']=='observed' and acquisition['status']=='observed' and trusted else 'unknown',
        metadata_only=True,network_access=False,anchor_mutated=False,observed_unix=number(observation.get('observed_unix')),
        delivered_anchor=dict(completed_outer=step,delivered_unix=anchor['delivered_unix'],delivered=True,
            evidence_scope='caller-supplied prior actual-message delivery declaration; formatting does not prove or advance delivery'),
        recorded_complete_outers=completed,complete_return_window=window,cumulative_actor_counts=cumulative,
        latest_reward=scalars(latest_reward,('step','reward_mean')) if reward.get('status')=='observed' else dict(step=None,reward_mean=None),
        selected_prequential=acquisition,postfit_recent8_replay=postfit,
        warmup_scope='separate fixed5120 initialization; its200 HeadAdam never count as formal outers or formal replay',
        known_em=scores,gpu_samples=gpu_summary,
        pipeline_stage=stage if isinstance(stage,str) and re.fullmatch('[A-Z][A-Z0-9_]{0,95}',stage) else None)


def read(path):
    path=Path(path)
    if path.suffix!='.json' or path.is_symlink() or path.resolve()!=path or path.stat().st_size>2*1024**2:
        raise ValueError('NATIVE_HOURLY_INPUT_MUST_BE_SMALL_LOCAL_METADATA_JSON')
    raw=path.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',required=True,type=Path);parser.add_argument('--delivered-anchor',required=True,type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    receipt,receipt_sha=read(args.receipt);anchor,anchor_sha=read(args.delivered_anchor)
    result=format_summary(receipt,anchor);result.update(observer_receipt_sha256=receipt_sha,delivered_anchor_sha256=anchor_sha)
    raw=json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    if args.output:
        if args.output in (args.receipt,args.delivered_anchor) or args.output.suffix!='.json':raise ValueError('NATIVE_HOURLY_OUTPUT_CANNOT_REPLACE_INPUT_OR_ANCHOR')
        with args.output.open('x',encoding='utf-8') as stream:stream.write(raw)
    print(raw,end='')


if __name__=='__main__':main()
