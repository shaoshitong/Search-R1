"""CPU final-goal audit, beyond common8 structural completion.

Fail early unless all200 raw successful updates already exist. This never
launches training, generates trajectories, fits a head or allocates a GPU.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from search_g1.qprobe_features import sha_file


def need(value, message):
    if not value:raise ValueError('QPROBE_GOAL: '+message)


def file_proof(row):
    path=Path(row['path'])
    need(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and
         sha_file(path)==row.get('sha256',row.get('full_read_sha256')), 'actual file SHA/size differs')


def history_parts(base, run, *, last=200):
    """Read authenticated lineage without copying old rows into the new run."""
    base=Path(base).resolve()
    if run.get('full_state_restore'):
        from search_g1.qprobe_saved_checkpoint import history_parts as authenticated_parts
        parts=authenticated_parts(base,run,last=last)
    else:parts=[dict(base=base,run=run,first=1,last=last)]
    need(isinstance(parts,list) and parts,'actual bounded training history missing')
    cursor=1;seen=set()
    for part in parts:
        origin=Path(part['base'])
        first,end=part['first'],part['last'];parent=part['run']
        need(origin.is_absolute() and origin.resolve()==origin and origin not in seen and
             type(first) is int and type(end) is int and first==cursor and first<=end<=last,
             'training history has an overlap, gap, unsafe path or repeated producer')
        manifest=origin/'logs/run-manifest.json'
        need(manifest.is_file() and not manifest.is_symlink() and json.loads(manifest.read_text())==parent,
             'historical producer manifest differs from authenticated lineage')
        need(parent['run_id']==run['run_id'] and parent.get('training_world_size')==4 and
             parent['input_sha256']==run['input_sha256'] and
             parent['initial_model_sha256']==run['initial_model_sha256'] and parent['budget']==run['budget'],
             'historical producer scientific/data/topology binding differs')
        for key in ('candidate_questions','selected_questions','responses_per_question','exploration_questions','seed'):
            need(parent['question_probe'][key]==run['question_probe'][key],'historical acquisition differs: '+key)
        # Provenance remains the original source, not the current source map.
        need(isinstance(parent['source_commit'],str) and isinstance(parent['source_sha256'],dict) and
             parent['source_sha256'],'historical source provenance missing')
        part['base']=origin;seen.add(origin);cursor=end+1
    need(cursor==last+1 and parts[-1]['base']==base and parts[-1]['run']==run,
         'training history does not end at the actual final producer')
    return parts


def part_updates(parts):
    records=[];proofs=[]
    for part in parts:
        path=part['base']/'logs/actual-updates.jsonl'
        need(path.is_file() and not path.is_symlink(), 'actual completed200 receipt missing; no work may be launched by audit')
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        need(all(type(row['step']) is int and 1<=row['step']<=200 for row in rows),
             'actual producer contains update201 or malformed counters')
        need(rows and [row['step'] for row in rows]==list(range(part['first'],rows[-1]['step']+1)),
             'producer raw ledger contains copied prefix rows, omissions or duplicate updates')
        selected=[row for row in rows if part['first']<=row['step']<=part['last']]
        need([row['step'] for row in selected]==list(range(part['first'],part['last']+1)) and
             all(row['run_id']==part['run']['run_id'] for row in selected),
             'actual bounded raw history is incomplete, duplicated or reordered')
        records.extend(selected)
        proofs.append(dict(base=str(part['base']),source_commit=part['run']['source_commit'],
            source_manifest_sha256=sha_file(part['base']/'logs/run-manifest.json'),
            actual_updates_sha256=sha_file(path),first=part['first'],last=part['last']))
    need([row['step'] for row in records]==list(range(1,201)),
         'require actual bounded raw1..200 without reconstructed updates or update201')
    return records,proofs


def step_part(parts, step):
    owners=[part for part in parts if part['first']<=step<=part['last']]
    need(len(owners)==1,'fixed evaluation has no unique actual producer')
    return owners[0]


def verify_teacher(base, run):
    from search_g1.qprobe_warmup import TeacherCache, verify_frozen
    from search_g1.full_state_restore import equal_tree
    import torch
    settings=run['question_probe'];cache_root=Path(settings['teacher_warmup_cache']).resolve()
    identity=json.loads((cache_root/'identity.json').read_text())
    need((cache_root/'complete.json').is_file(), 'teacher complete receipt missing')
    need(identity['training_world_size']==4 and len(identity['questions'])==5120 and
         len({x['qid'] for x in identity['questions']})==5120, 'teacher pool/topology differs')
    from search_g1.qprobe_features import FeatureCache
    features=FeatureCache(settings['cache_dir'])
    need(identity['model_files_sha256']==run['initial_model_sha256'] and
         identity['train_parquet_sha256']==run['input_sha256']['train.parquet'] and
         identity['feature_spec_sha']==features.spec_sha and
         identity['data_manifest_sha256']==run['budget']['common_manifest_sha256'],
         'teacher model/data/features namespace differs')
    for name,sha in dict(identity['scorer_sources'],**identity['rollout_sources']).items():
        need(run['source_sha256'].get(name)==sha, 'teacher sampling/scoring source differs')
    # Already completed immutable cache: avoid creating a writer/lock or label.
    cache=TeacherCache.__new__(TeacherCache);cache.root=cache_root;cache.identity=identity
    cached,groups,chunks=cache.verify_complete()
    ready_path=base/'artifacts/question-probe/teacher-warmup-ready.json'
    ready=json.loads(ready_path.read_text())
    need(ready['fitted_questions']==5120 and ready['completed_training_outers']==0 and
         ready['head_metrics']['bootstrap_groups']==5120 and ready['cache_receipt']==cached,
         'all5120 were not fitted before formal outer1')
    verify_frozen(ready['before_workers'],ready['after_workers'],4)
    need(sorted(x['rank'] for x in ready['actor_rng_restored'])==list(range(4)) and
         all(x['rng_equal'] for x in ready['actor_rng_restored']), 'actual four-rank teacher RNG restoration absent')
    for rows in [ready['before_workers'],ready['after_workers']]+[
            rows for chunk in chunks for rows in (chunk['frozen_workers']['before'],chunk['frozen_workers']['after'])]:
        for row in rows:file_proof(row['rng_receipt'])
    file_proof(ready['head_checkpoint'])
    head=torch.load(ready['head_checkpoint']['path'],map_location='cpu',weights_only=True)
    need(head['completed_outer']==head['update_count']==0 and len(head['replay_labels'])==5120 and
         not head['completed_attempts'], 'initial head is not a fresh complete5120 fit')
    expected={x['qid']:x for x in groups}
    need(len({x['qid'] for x in head['replay_labels']})==5120, 'initial head duplicated/missing teacher labels')
    for row in head['replay_labels']:
        need(row['source_outer']==0 and row['strict_em']==expected[row['qid']]['strict_em'], 'head teacher label differs')
    driver_path=base/'artifacts/question-probe/teacher-warmup-driver-boundary/complete.json'
    driver=json.loads(driver_path.read_text())
    need(driver['run_id']==run['run_id'] and driver['source_commit']==run['source_commit'] and
         driver['run_manifest_sha256']==sha_file(base/'logs/run-manifest.json') and
         driver['head_state_equal'] and driver['head_scheduler'] is None, 'actual driver/head boundary proof differs')
    for key in ('driver_rng_equal','actual_candidate_stream_equal','selector_rng_equal'):need(driver[key] is True,key)
    for key in ('before','after'):file_proof(driver[key])
    equal_tree(torch.load(driver['before']['path'],map_location='cpu',weights_only=False),
               torch.load(driver['after']['path'],map_location='cpu',weights_only=False))
    return dict(questions=5120,trajectories=25600,bootstrap_head_optimizer_updates=ready['head_metrics']['head_optimizer_updates'],
                reused_questions=ready['reused_questions'],ready_sha256=sha_file(ready_path),driver_boundary_sha256=sha_file(driver_path))


def verify_selection(selection, rollout, seed):
    from search_g1.qprobe_selection import select_questions,prequential_metrics,count_predictions
    from search_g1.full_state_restore import equal_tree
    need(selection['attempted_candidates']==5120 and selection['selected_groups']==512 and
         selection['exploration_questions']==64 and selection['prediction_before_label'] is True and
         selection['post_rollout_em_filter'] is False, 'selection budget/prediction chronology differs')
    questions=selection['questions'];need(len(questions)==5120,'candidate pool incomplete')
    proxies=[SimpleNamespace(ordinal=q['ordinal'],qid=q['qid'],question_hash=q['question_hash'],
                             content_hash=q['content_sha256']) for q in questions]
    probabilities=[q['probabilities'] for q in questions]
    predictions=count_predictions(probabilities)
    for key,field in [('mean','predicted_mean'),('expected_std','predicted_std'),
                      ('expected_variance','predicted_variance'),('mixed_probability','predicted_mixed_probability')]:
        need(np.allclose(predictions[key],[q[field] for q in questions],atol=1e-12,rtol=0), 'saved count moments differ')
    rng=np.random.default_rng();rng.bit_generator.state=selection['selection_rng_before']
    chosen=select_questions(proxies,probabilities,selected_questions=512,exploration_questions=64,
                            rng=rng,tie_seed=str(seed)+':'+str(selection['step']))
    equal_tree(rng.bit_generator.state,selection['selection_rng_after'])
    need([(x.ordinal,x.selection_source) for x in chosen]==[
         (q['ordinal'],q['selection_source']) for q in questions if q['selected']], 'actual E64/top448 selection differs')
    need(rollout['no_post_rollout_refill'] is True and rollout['raw_generated_trajectories']==2560 and
         rollout['responses_per_group']==5 and len(rollout['groups'])==512, 'actual G5/no-refill budget differs')
    groups=rollout['groups'];need([(x['ordinal'],x['source']) for x in groups]==[
        (x.ordinal,x.selection_source) for x in chosen], 'rollout changed selected group order')
    need(len({tid for x in groups for tid in x['trajectory_ids']})==2560 and
         all(len(x['trajectory_ids'])==5 for x in groups), 'trajectory identities incomplete or duplicated')
    labels=[x['strict_correctness'] for x in groups]
    measured=prequential_metrics([probabilities[x.ordinal] for x in chosen],labels,[x.selection_source for x in chosen])
    equal_tree(measured,rollout['prequential'])
    return measured


def verify_evaluations(base, run, *, parts=None):
    from reproduction.common8.validation_identity import expected
    base=Path(base).resolve()
    parts=parts if parts is not None else history_parts(base,run)
    results={}
    def check(row,record,ordered_ids,sources,producer_run):
        need(row['run_id']==producer_run['run_id'] and row['question_ids']==ordered_ids and row['data_sources']==sources and
             row['validation_sha256']==record['sha256'] and len(row['em'])==len(ordered_ids)==len(row['f1']),
             'actual fixed evaluation ordered identity differs')
        need(all(type(x) in (int,float) and x in (0,1) for x in row['em']) and
             all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in row['f1']) and
             all(em<=f1 for em,f1 in zip(row['em'],row['f1'])), 'actual strict EM/F1 values invalid')
        need(row['correct']==sum(row['em']) and row['weighted_em']==sum(row['em'])/len(ordered_ids) and
             row['weighted_f1']==math.fsum(row['f1'])/len(ordered_ids) and row['counts']==record['source_counts'],
             'actual evaluation weighted scores/counts differ')
    for step in (5,50,100,150,200):
        part=step_part(parts,step);origin=part['base']
        validation,frame,ids=expected(origin,'validation')
        ledger=[json.loads(x) for x in (origin/'logs/validation-identities.jsonl').read_text().splitlines()]
        rows=[x for x in ledger if x['split']=='validation' and x['step']==step and x['final'] is False]
        need(len(rows)==1,'missing/duplicate exact fixed512 validation at'+str(step))
        need(len(ids)==512,'fixed validation does not contain512 native questions')
        row=rows[0];check(row,validation,ids,frame['data_source'].tolist(),part['run'])
        results[str(step)]=dict(correct=row['correct'],questions=512,weighted_em=row['weighted_em'],weighted_f1=row['weighted_f1'],
            producer_base=str(origin),producer_commit=part['run']['source_commit'],
            actual_ledger_sha256=sha_file(origin/'logs/validation-identities.jsonl'))
    final_part=step_part(parts,200)
    need(final_part['base']==base and final_part['run']==run,'final evaluation is not the current final producer')
    final_record,final_frame,final_ids=expected(final_part['base'],'final')
    final=json.loads((final_part['base']/'logs/final-identity.json').read_text())
    need(final['step']==200 and final['final'] is True and len(final_ids)==51713,'final native51713 missing')
    check(final,final_record,final_ids,final_frame['data_source'].tolist(),final_part['run'])
    em={};f1={}
    for source,count in final_record['source_counts'].items():
        indices=[i for i,s in enumerate(final['data_sources']) if s==source]
        need(len(indices)==count,'final source count differs')
        em[source]=sum(final['em'][i] for i in indices)/count
        f1[source]=math.fsum(final['f1'][i] for i in indices)/count
    return results,final,em,f1


def verify_formal_part(part, updates):
    """Independently read each producer's own gates, plan and four-rank rows."""
    from reproduction.common8.behavior_runtime import read_gate_receipt,verify_write_receipt_links
    from reproduction.common8.runtime import budget_rows,verify_scheduler_receipts
    from reproduction.common8.worker_evidence import read_console_form
    from reproduction.common8.formal_inline_safety import verify_plan
    from reproduction.common8.formal_attempt_audit import verify_attempt_history
    base,run,first,last=part['base'],part['run'],part['first'],part['last']
    need(first==(run.get('full_state_restore') or {}).get('completed_outer',0)+1,
         'history part does not begin at its actual restored boundary')
    gates={str(row['step']):read_gate_receipt(base,run,row['step'],update_record=row) for row in updates}
    verify_write_receipt_links(base,gates,first,last)
    budget_rows(read_console_form(base,run['run_id']),first,last,complete_records=updates,run=run)
    schedulers=verify_scheduler_receipts(base,run,first,last,gates)
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text())
    _,plan_sha=verify_plan(base,artifact,run)
    audit=verify_attempt_history(base,run,updates,schedulers,last=last,plan_sha256=plan_sha)
    need(audit['structural_accepted'] is True and audit['completed_outer_count']==last,
         'actual historical formal attempt audit failed')
    return dict(base=str(base),first=first,last=last,source_commit=run['source_commit'],
        prospective_plan_sha256=plan_sha,actual_attempt_audit=audit)


def verify_attempt_parts(parts):
    """Prove commits and selections against bounded, original raw attempts."""
    from search_g1.full_state_restore import equal_tree
    attempts=[];hashes=[];formal=[];seen=set();successes=set();completed=0
    for part in parts:
        base,run,first,last=part['base'],part['run'],part['first'],part['last']
        root=base/'artifacts/question-probe'
        path=base/'logs/actual-attempts.jsonl'
        need(path.is_file() and not path.is_symlink(),'actual bounded attempt ledger missing')
        all_attempts=[json.loads(x) for x in path.read_text().splitlines()]
        need(all(type(x['logical_update']) is int and first<=x['logical_update']<=200 for x in all_attempts),
             'actual attempt201 or malformed logical counter exists')
        selected=[x for x in all_attempts if first<=x['logical_update']<=last]
        events={}
        for event_path in (root/'completed').glob('*.json'):
            need(event_path.is_file() and not event_path.is_symlink(),'unsafe head commit evidence')
            event=json.loads(event_path.read_text())
            need(type(event['outer']) is int and first<=event['outer']<=200,
                 'head update201 or copied historical commit exists')
            if first<=event['outer']<=last:events[event_path.stem]=event_path
        successful={x['attempt_id'] for x in selected if x['complete_outer']}
        need(set(events)==successful,'head commits do not match bounded actual actor successes')
        for attempt in selected:
            attempt_id=attempt['attempt_id']
            need(isinstance(attempt_id,str) and re.fullmatch('[0-9a-f]{32}',attempt_id) and
                 attempt_id not in seen and attempt['run_id']==run['run_id'] and
                 type(attempt['attempt_index']) is int and attempt['attempt_index']==len(attempts)+1 and
                 attempt['logical_update']==completed+1 and type(attempt['complete_outer']) is bool,
                 'attempt identity, chronology or original run differs across history parts')
            seen.add(attempt_id)
            directory=root/'attempts'/attempt_id
            for name in ('selection.json','rollout-audit.json'):
                need((directory/name).is_file() and not (directory/name).is_symlink(),'unsafe actual selection/rollout evidence')
            selection=json.loads((directory/'selection.json').read_text())
            rollout=json.loads((directory/'rollout-audit.json').read_text())
            need(selection['step']==rollout['step']==attempt['logical_update'] and
                 selection['attempt_id']==rollout['attempt_id']==attempt_id and
                 selection['policy_snapshot_id']==rollout['policy_snapshot_id'],
                 'actor/generation selection identity differs')
            measured=verify_selection(selection,rollout,run['question_probe']['seed'])
            if attempt['complete_outer']:
                completed+=1;successes.add(attempt_id)
                event=json.loads(events[attempt_id].read_text())
                need(event['outer']==attempt['completed_outer']==completed and
                     event['successful_actor_snapshot_id']==run['run_id']+':'+str(completed) and
                     event['prediction_version']==selection['head_version'] and
                     event['label_policy_snapshot_id']==selection['policy_snapshot_id'] and
                     event['head_update']['update_count']==completed,
                     'head updated outside successful actor transaction')
                equal_tree(measured,event['prequential'])
            else:
                aborted=json.loads((root/'aborted'/(attempt_id+'.json')).read_text())
                need(attempt['completed_outer']==completed and aborted['head_updated'] is False and
                     attempt_id not in events,'failed actor updated head')
            attempts.append(attempt)
            hashes.append(dict(producer_base=str(base),producer_commit=run['source_commit'],attempt_id=attempt_id,
                selection_sha256=sha_file(directory/'selection.json'),rollout_sha256=sha_file(directory/'rollout-audit.json')))
        need(selected and selected[-1]['complete_outer'] is True and completed==last,
             'actual history part does not end at its complete boundary')
        updates=[json.loads(x) for x in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
        updates=[row for row in updates if first<=row['step']<=last]
        formal.append(verify_formal_part(part,updates))
        for rank in range(4):
            scheduler=[json.loads(x) for x in (base/'logs'/('scheduler-rank-'+str(rank)+'.jsonl')).read_text().splitlines()]
            optimizer=[json.loads(x) for x in (base/'logs'/('optimizer-rank-'+str(rank)+'.jsonl')).read_text().splitlines()]
            need(all(type(x.get('logical_update',0)) is int and 0<=x.get('logical_update',0)<=200 for x in scheduler) and
                 all(type((x.get('context') or {}).get('logical_update',0)) is int and
                     0<=(x.get('context') or {}).get('logical_update',0)<=200 for x in optimizer),
                 'actor/scheduler update201 exists in an actual producer')
    need(completed==len(successes)==200,'head commits do not match all200 actual actor successes')
    return attempts,hashes,formal


def verify(base, audit_base):
    base=Path(base).resolve();audit_base=Path(audit_base).resolve()
    need(os.getuid()==14090,'actual jxu957 final CPU owner required')
    run=json.loads((base/'logs/run-manifest.json').read_text())
    records_path=base/'logs/actual-updates.jsonl'
    need(records_path.is_file(),'actual completed200 receipt missing; no work may be launched by audit')
    current_updates=[json.loads(x) for x in records_path.read_text().splitlines()]
    current_first=(run.get('full_state_restore') or {}).get('completed_outer',0)+1
    need(all(type(row['step']) is int for row in current_updates) and
         [row['step'] for row in current_updates]==list(range(current_first,201)),
         'actual current completed200 suffix missing or contains copied prefix/update201; no source recovery may be loaded by audit')
    need(run['training_world_size']==4,'this audit requires the actual native4 topology')
    parts=history_parts(base,run);updates,history=part_updates(parts)
    full=json.loads((base/'logs/full-run-acceptance.json').read_text())
    need(full['structural_accepted'] is True and full['actual_outer_updates']==200 and full['run_id']==run['run_id'],
         'full structural/evaluation acceptance missing')
    initial=parts[0]
    need(initial['first']==1 and initial['run'].get('full_state_restore') is None,
         'teacher must come from the actual initial fresh producer')
    teacher=verify_teacher(initial['base'],initial['run'])
    teacher.update(producer_base=str(initial['base']),producer_commit=initial['run']['source_commit'])
    evaluations,final,source_em,source_f1=verify_evaluations(base,run,parts=parts)
    need(source_em==full['final_source_em'] and source_f1==full['final_source_f1'] and
         final['weighted_em']==full['final_weighted_em'] and final['weighted_f1']==full['final_weighted_f1'],
         'final structural report differs from actual canonical saved evaluation')
    attempts,hashes,formal_parts=verify_attempt_parts(parts)
    root=base/'artifacts/question-probe'
    from reproduction.hpc3_qprobe.final_restore_audit import verify_transport
    archive_proofs={}
    for step in (5,50,100,150,200):
        p=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/archives')/run['run_id']/('global_step_'+str(step))/'complete.json'
        transport=json.loads(p.read_text());need(transport['step']==step and transport['run_id']==run['run_id'],'milestone transport differs')
        need(transport['source_receipt']['path']==str(root/'completed-pairs'/('global_step_'+str(step))/'source-receipt.json'),
             'milestone transport does not originate in this actual full run')
        verify_transport(p,transport['target_receipt']['path'],expected_step=step)
        archive_proofs[str(step)]=dict(path=str(p),sha256=sha_file(p),files=len(transport['files']))
        if step==200:
            final_transport=transport
    accepted_path=audit_base/'logs/final-restore-audit-accepted.json';accepted=json.loads(accepted_path.read_text())
    native_path=audit_base/'logs/final-restore-readback.json';native=json.loads(native_path.read_text())
    need(accepted['accepted'] and accepted['actual_cuda_restore_accepted'] and accepted['owned_cleanup'] and
         accepted['readback_sha256']==sha_file(native_path) and native['run_id']==run['run_id'] and
         native['world_size']==4 and native['actual_completed_outer']==200 and not native['update201_performed'] and
         accepted['audit_id']==native['audit_id'] and native['producer_commit']==run['source_commit'] and
         accepted['source_receipt_sha256']==native['source_receipt_sha256']==final_transport['target_receipt']['sha256'] and
         accepted['transport_proof_sha256']==archive_proofs['200']['sha256'], 'actual native4 recovery/cleanup/target binding absent')
    for key in ('actual_actor_optimizer_write_delta','actual_actor_scheduler_advance_delta',
                'actual_head_update_delta','actual_head_optimizer_write_delta','generated_trajectories'):
        need(native[key]==0,'restore audit performed new work: '+key)
    state=subprocess.check_output(['sacct','-nP','-j',native['slurm_job_id'],'-X','--format=JobID,State,ExitCode,UID'],text=True,timeout=30).strip()
    need(state==native['slurm_job_id']+'|COMPLETED|0:0|14090','actual restore job has not completed successfully')
    pipeline=json.loads((base/'logs/pipeline-status.json').read_text())
    cleanup=json.loads((base/'logs/cleanup-status.json').read_text())
    need(pipeline['run_id']==run['run_id'] and pipeline['stage']=='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP' and
         cleanup['cleaned'] is True and cleanup['run_id']==run['run_id'],'training producer/final evaluation/owned cleanup not finished')
    recovery_path=base/'preparation/recovery-source-binding.json'
    if recovery_path.is_file():
        recovery=json.loads(recovery_path.read_text())
        if recovery.get('failed_job')=='718509':
            from reproduction.hpc3_qprobe.telemetry_failure718509 import cleanup_binding
            cleanup_binding(require_complete=True)
    need(len(source_em)==len(source_f1)==7 and sum(full['final_counts'].values())==51713,'canonical final51713/seven sources missing')
    result=dict(status='QPROBE_FULL200_TEACHER_HEAD_FIXED_VALIDATION_ARCHIVE_NATIVE4_GOAL_PASS',run_id=run['run_id'],
        completed_outer=200,update201_performed=False,teacher=teacher,actual_attempts=len(attempts),actual_head_successful_commits=200,
        attempt_evidence=hashes,archives=archive_proofs,fixed_validation=evaluations,final_native_questions=51713,
        final_weighted_em=full['final_weighted_em'],final_weighted_f1=full['final_weighted_f1'],
        final_macro_em=math.fsum(source_em.values())/7,final_macro_f1=math.fsum(source_f1.values())/7,
        actual_native_restore_accepted_sha256=sha_file(accepted_path),checked_unix=time.time())
    result.update(actual_training_parts=history,actual_formal_part_audits=formal_parts,
        historical_rows_republished_as_current=False,
        bitwise_continuation_of_failed_execution=False if any(
            (part['run'].get('full_state_restore') or {}).get('kind')=='question_probe_saved_checkpoint'
            for part in parts) else None)
    from reproduction.hpc3_qprobe.control import publish
    publish(base/'logs/qprobe-final-goal-acceptance.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--audit-base',required=True);a=p.parse_args()
    print(json.dumps(verify(a.base,a.audit_base)),flush=True)
