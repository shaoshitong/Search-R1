"""Real paired dev512 reporting at official0 and fresh5; accuracy is measured.

An explicit historical-baseline policy skips current-run official0 generation
and compares a source-bound scalar baseline with the real current fresh5 rows.
Fresh native4 Prefix20 may explicitly skip all initial evaluation and report
only its real fresh5 rows, without claiming a before score or comparison.
This does not add an accuracy/no-drop acceptance rule. The existing stage64
checks remain separate and retain their original canonical occurrences.
"""
import json,math,os,re,time
from pathlib import Path
from reproduction.common8.formal_contract import SAFETY_MODE,assert_formal_run,skip_initial_evaluation,SKIP_INITIAL_EVALUATION_POLICY
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.verify_data import digest

ROLE='formal_inline_development_evaluation'
HISTORICAL_ROLE='formal_inline_historical_development_baseline'
PATHS={0:'logs/formal-inline-development-before.json',5:'logs/formal-inline-development-after.json'}
SKIPPED_INITIAL='logs/formal-inline-development-initial-skipped.json'
SKIPPED_ROLE='formal_inline_initial_evaluation_skipped'


def record_skipped_initial(base,run):
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.formal_inline_safety import verify_plan
    assert skip_initial_evaluation(run)
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text());_,plan_sha=verify_plan(base,artifact,run)
    result=dict(schema_version=1,role=SKIPPED_ROLE,run_id=run['run_id'],step=0,
        policy=SKIP_INITIAL_EVALUATION_POLICY,policy_safety_mode=SAFETY_MODE,plan_sha256=plan_sha,
        source_map_sha256=document_sha256(run['source_sha256']),current_run_evaluation_performed=False,
        historical_comparison=False,paired_same_run=False,checked_unix=time.time(),
        semantics='Initial stage64 and dev512 evaluation explicitly skipped; start real training; fixed5/50/100/150/200 remain')
    assert not (base/PATHS[0]).exists(), 'Skipped initial evaluation cannot coexist with a before score'
    write_once(base/SKIPPED_INITIAL,result)
    print('SEARCH_G1_INITIAL_EVALUATION_POLICY='+SKIP_INITIAL_EVALUATION_POLICY,flush=True)
    return result


def historical_baseline(base,run):
    """Authenticate an explicitly selected historical scalar receipt, never rows."""
    from reproduction.common8.verify_data import MANIFEST_SHA
    base=Path(base);assert run.get('pretraining_baseline_policy')=='reuse_historical'
    assert_formal_run(run)
    baseline=run['historical_development_baseline']
    assert baseline['schema_version']==1 and baseline['role']=='historical_development_baseline'
    assert re.fullmatch('[0-9a-f]{32}',baseline['source_run_id']) and baseline['source_run_id']!=run['run_id']
    assert re.fullmatch('[0-9a-f]{40}',baseline['source_commit'])
    assert re.fullmatch('[0-9a-f]{64}',baseline['source_artifact_sha256'])
    assert str(baseline['source_native_job_id']).isdigit()
    relative=Path(baseline['source_artifact_relative'])
    assert not relative.is_absolute() and '..' not in relative.parts
    source=base/relative
    assert source.is_file() and not source.is_symlink() and source.resolve()==source and source.stat().st_uid==os.getuid()
    assert source.stat().st_size<=256*1024 and digest(source)==baseline['source_artifact_sha256']
    evidence=json.loads(source.read_text())
    fields=('schema_version','role','source_run_id','source_native_job_id','source_commit',
        'common_manifest_sha256','validation_sha256','model_revision','question_count',
        'source_counts','correct','weighted_em','weighted_f1')
    assert all(evidence[key]==baseline[key] for key in fields), 'Historical baseline scalar identity differs from source bytes'
    assert not any(key in baseline or key in evidence for key in ('rows','question_ids','data_sources','em','f1'))
    manifest=base/'data/manifest.json';assert digest(manifest)==MANIFEST_SHA==baseline['common_manifest_sha256']
    validation=json.loads(manifest.read_text())['splits']['validation']
    assert baseline['validation_sha256']==validation['sha256'] and digest(base/'data'/validation['path'])==validation['sha256']
    assert baseline['model_revision']==run['model_revision'] and run['budget']['common_manifest_sha256']==MANIFEST_SHA
    assert type(baseline['question_count']) is int and baseline['question_count']==validation['rows']==512
    assert baseline['source_counts']==validation['source_counts']=={'nq':256,'hotpotqa':256}
    assert type(baseline['correct']) is int and 0<=baseline['correct']<=512
    assert type(baseline['weighted_em']) in (int,float) and baseline['weighted_em']==baseline['correct']/512
    assert type(baseline['weighted_f1']) in (int,float) and math.isfinite(baseline['weighted_f1']) and baseline['weighted_em']<=baseline['weighted_f1']<=1
    result={key:baseline[key] for key in fields+('source_artifact_relative','source_artifact_sha256')}
    for key in ('valid_action_rate','executed_search_question_rate','executed_search_calls_per_question'):
        if key in baseline:
            assert type(baseline[key]) in (int,float) and math.isfinite(baseline[key]) and baseline[key]>=0
            assert evidence[key]==baseline[key]
            result[key]=baseline[key]
    return result


def record_historical_baseline(base,run):
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.formal_inline_safety import verify_plan
    baseline=historical_baseline(base,run)
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text());_,plan_sha=verify_plan(base,artifact,run)
    result=dict(baseline,role=HISTORICAL_ROLE,comparison_run_id=run['run_id'],step=0,
        policy_safety_mode=SAFETY_MODE,plan_sha256=plan_sha,source_map_sha256=document_sha256(run['source_sha256']),
        historical_comparison=True,current_run_evaluation_performed=False,paired_same_run=False,
        accuracy_health_gate=False,checked_unix=time.time(),
        semantics='Historical official-model dev512 scalar baseline; current-run pretraining generation was explicitly skipped; no per-question historical rows are asserted')
    write_once(base/PATHS[0],result)
    return result


def score_counts(rows):
    correct={source:sum(row['em'] for row in rows if row['data_source']==source)
             for source in ('nq','hotpotqa')}
    return dict(correct_count=sum(row['em'] for row in rows),f1_sum=math.fsum(row['f1'] for row in rows),
                source_correct=correct,source_em={source:value/256 for source,value in correct.items()})


def active_run():
    if os.environ.get('SEARCH_G1_PROFILE')!='common8':return None
    base=Path(os.environ['SEARCH_G1_BASE']).resolve()
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run.get('policy_safety_mode')!=SAFETY_MODE:return None
    assert_formal_run(run)
    return base,run


def configure_development(trainer):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn
    from reproduction.common8 import validation_identity as validation
    import copy
    base,run=active_run();assert int(trainer.global_steps) in (0,5)
    record,frame,_=validation.expected(base,'validation')
    dataset=copy.copy(trainer.val_dataset)
    dataset.dataframe=validation.bind_validation_dataframe(frame,base,'validation')
    trainer.val_dataloader=DataLoader(dataset,batch_size=64,shuffle=False,drop_last=False,collate_fn=collate_fn)
    validation._ACTIVE_SPLIT='validation';validation._ACTIVE_SOURCE_COUNTS=dict(record['source_counts'])
    print('SEARCH_G1_EVALUATION_SPLIT=validation',flush=True)


def record_development(step,rows,original_rewards):
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.formal_inline_safety import verify_plan
    assert type(step) is int and step in PATHS
    base,run=active_run();record,frame,ids=expected(base,'validation')
    assert step!=0 or not skip_initial_evaluation(run), 'Initial evaluation was explicitly skipped'
    result=aggregate(rows,ids,frame['data_source'].tolist())
    result.update(score_counts(rows))
    assert [row['em'] for row in rows]==list(original_rewards), 'Measured strict answers must match actual EM reward'
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text())
    _,plan_sha=verify_plan(base,artifact,run)
    result.update(schema_version=1,role=ROLE,run_id=run['run_id'],policy_safety_mode=SAFETY_MODE,step=step,
        plan_sha256=plan_sha,validation_sha256=record['sha256'],source_map_sha256=document_sha256(run['source_sha256']),
        checked_unix=time.time(),semantics=('real after5 dev512 EM/F1/actions; no before evaluation or comparison'
            if skip_initial_evaluation(run) else 'paired real dev512 EM/F1/actions; accuracy change is reported without a zero-drop gate'))
    write_once(base/PATHS[step],result)
    return result


def evaluate_development(trainer):
    """Same live trainer and weights; preserve normal stage/dev loader selection."""
    from reproduction.common8 import validation_identity as validation
    active=active_run();assert active is not None and int(trainer.global_steps) in PATHS
    base,run=active
    if int(trainer.global_steps)==0 and skip_initial_evaluation(run):
        return record_skipped_initial(base,run)
    if int(trainer.global_steps)==0 and run.get('pretraining_baseline_policy')=='reuse_historical':
        return record_historical_baseline(base,run)
    original_loader=trainer.val_dataloader
    original_split=validation._ACTIVE_SPLIT;original_sources=validation._ACTIVE_SOURCE_COUNTS
    try:
        return trainer._validate(inline_development=True)
    finally:
        trainer.val_dataloader=original_loader
        validation._ACTIVE_SPLIT=original_split;validation._ACTIVE_SOURCE_COUNTS=original_sources


def verify_pair(base,run):
    """Rebuild ordered scores/actions from actual scalar rows; do not gate gains."""
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    from reproduction.common8.formal_inline_safety import verify_plan
    base=Path(base).resolve();assert_formal_run(run)
    if run.get('full_state_restore'):
        from search_g1.full_state_restore import verify_restored_pair
        return verify_restored_pair(base,run)
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text());_,plan_sha=verify_plan(base,artifact,run)
    record,frame,ids=expected(base,'validation');proofs={};hashes={}
    skipped=skip_initial_evaluation(run)
    if skipped:
        target=base/SKIPPED_INITIAL
        assert target.is_file() and not target.is_symlink() and target.resolve()==target
        initial=json.loads(target.read_text())
        required=dict(schema_version=1,role=SKIPPED_ROLE,run_id=run['run_id'],step=0,
            policy=SKIP_INITIAL_EVALUATION_POLICY,policy_safety_mode=SAFETY_MODE,plan_sha256=plan_sha,
            source_map_sha256=document_sha256(run['source_sha256']),current_run_evaluation_performed=False,
            historical_comparison=False,paired_same_run=False)
        assert all(initial.get(key)==value for key,value in required.items())
        assert not any(key in initial for key in ('rows','question_ids','data_sources','em','f1','weighted_em','weighted_f1'))
        assert not (base/PATHS[0]).exists(), 'Skipped initial evaluation cannot coexist with a before score'
        hashes['skipped0']=digest(target)
    for step,path in PATHS.items():
        if step==0 and skipped:continue
        target=base/path
        assert target.is_file() and not target.is_symlink() and target.resolve()==target
        proof=json.loads(target.read_text())
        if step==0 and run.get('pretraining_baseline_policy')=='reuse_historical':
            baseline=historical_baseline(base,run)
            assert proof['schema_version']==1 and proof['role']==HISTORICAL_ROLE and proof['comparison_run_id']==run['run_id']
            assert proof['policy_safety_mode']==SAFETY_MODE and proof['step']==0 and proof['plan_sha256']==plan_sha
            assert proof['source_map_sha256']==document_sha256(run['source_sha256'])
            assert all(proof[key]==value for key,value in baseline.items() if key!='role')
            assert proof['historical_comparison'] is True and proof['current_run_evaluation_performed'] is False and proof['paired_same_run'] is False
            assert not any(key in proof for key in ('rows','question_ids','data_sources','em','f1'))
            proofs[step]=proof;hashes[str(step)]=digest(target)
            continue
        measured=aggregate(proof['rows'],ids,frame['data_source'].tolist())
        measured.update(score_counts(proof['rows']))
        assert proof['schema_version']==1 and proof['role']==ROLE and proof['run_id']==run['run_id']
        assert proof['policy_safety_mode']==SAFETY_MODE and proof['step']==step and proof['plan_sha256']==plan_sha
        assert proof['source_map_sha256']==document_sha256(run['source_sha256']) and proof['validation_sha256']==record['sha256']
        assert all(proof[name]==value for name,value in measured.items() if name not in ('role','semantics'))
        proofs[step]=proof;hashes[str(step)]=digest(target)
    after=proofs[5]
    if skipped:
        return dict(schema_version=1,role='formal_inline_after_only_development_report',run_id=run['run_id'],plan_sha256=plan_sha,
            pretraining_baseline_policy=SKIP_INITIAL_EVALUATION_POLICY,initial_evaluation=initial,
            before=None,after=after,file_sha256=hashes,changes={},paired_same_run=False,historical_comparison=False,
            current_run_before_evaluation_performed=False,current_run_after_evaluation_performed=True,
            accuracy_health_gate=False,semantics='Real current-run after512 scores only; initial evaluation explicitly skipped; no baseline or accuracy change is asserted')
    before=proofs[0]
    if run.get('pretraining_baseline_policy')=='reuse_historical':
        assert after['question_ids']==ids and after['data_sources']==frame['data_source'].tolist()
        change_keys=('weighted_em','weighted_f1')+tuple(key for key in ('valid_action_rate',
            'executed_search_question_rate','executed_search_calls_per_question') if key in before and key in after)
        return dict(schema_version=1,role='formal_inline_historical_development_comparison',run_id=run['run_id'],plan_sha256=plan_sha,
            before=before,after=after,file_sha256=hashes,historical_baseline_source_sha256=before['source_artifact_sha256'],
            changes={name:after[name]-before[name] for name in change_keys},paired_same_run=False,
            current_run_after_evaluation_performed=True,accuracy_health_gate=False,
            semantics='True current-run after512 scores compared with an authenticated historical official-model scalar baseline; this is not a same-run paired evaluation or a health gate')
    assert before['question_ids']==after['question_ids']==ids and before['data_sources']==after['data_sources']==frame['data_source'].tolist()
    return dict(schema_version=1,role='formal_inline_paired_development_report',run_id=run['run_id'],plan_sha256=plan_sha,
        before=before,after=after,file_sha256=hashes,
        changes={name:after[name]-before[name] for name in ('weighted_em','weighted_f1','valid_action_rate',
            'executed_search_question_rate','executed_search_calls_per_question')},
        accuracy_health_gate=False,semantics='Actual paired512 scores/actions, including any decline; no invented zero-drop requirement')
