"""CPU-only immutable matched6528 authentic G20 inputs; never launch a GPU job.

Complete actual original5+three real extraG5 train G20 and precommitted two-fresh-G5
validation MUST exist first. No checkpoint/model tensor is loaded here.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from . import probe_ablation_train as u
from .probe_g20_train import accepted_g20,validate_plan,frozen_validation
from .g20_grouping import validate as validate20,digest as digest20
from .deploy_g10_pilot_suite import fixed_existing_g5

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
TARGET=PARENT+'/algorithm-improvement-20261021/g20-matched6528-v1'
VARIANTS=('random','global-weighted-balanced')
TRAIN_STATUS='ACTUAL_ORIGINAL5_PLUS_THREE_INDEPENDENT_G5_G20_SHA_OWN_EM_SEEDS_VERIFIED'
VALIDATION_STATUS='VALIDATION_ACTUAL_TWO_FRESH_G5_G10_SHA_OWN_EM_VERIFIED'
VALIDATION_COMPLETE_SHA='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'
VALIDATION_CONSUMER_SHA=dict(groups='73ac10cc9bb6b734871399030bcb0f84736e47afff042fbc2ccaa2341f06c6a0',
    features='1748ce0dbc37f48e39ec5736783d0e54c3195bbcfaeed2e0fc28e5c8e8f1d0bd',
    fixed_G5_plan='b6e5b57fee270c8fa61c1f03f212b5baef1125d205c006475f65a0690ed4aecb')


def ref(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if u.read(path)!=value:raise ValueError('Different existing immutable matched-G20 input: '+str(path))
    else:u.atomic(path,value)


def validation_consumer(complete):
    if (complete['status']!=VALIDATION_STATUS or complete['role']!='validation' or
            complete['untouched_final'] is not False or complete['tuning_allowed'] is not True or
            complete['fitting_on_validation_allowed'] is not False or complete['questions']!=2048 or
            complete['trajectories']!=20480 or complete['real_fresh_events']!=2 or
            complete['real_trajectories_per_event']!=10240 or complete['actor_optimizer_updates']!=0 or
            complete['synthetic_EM'] is not False or complete['fixed_G5_was_chosen_before_own_EM'] is not True):
        raise ValueError('Complete actual recipe-selection validation2048 G10 required, never final/fitting')
    c=complete['consumer_inputs']
    if (c['group_split']!='validation' or c['features_format']!='flat_list_of_native_complete_paths' or
            c['selection_labels_consulted'] is not False):
        raise ValueError('Precommitted label-blind fixedG5 consumer inputs required')
    return copy.deepcopy(c)


def assert_old10_prefix(groups, old10):
    current={g['qid']:g for g in groups};old={g['qid']:g for g in old10}
    if len(current)!=6528 or set(current)!=set(old):raise ValueError('Same matched6528 original questions required')
    for q,g in current.items():
        other=old[q]
        if (g['question_hash']!=other['question_hash'] or
                g['trajectory_ids'][:10]!=other['trajectory_ids'] or
                g['labels'][:10]!=other['labels']):
            raise ValueError('G20 must preserve accepted G10 own IDs/EM as its exact prefix')


def shared_plan(groups, reference_plan, old_plan, old_plan_ref):
    qids=[g['qid'] for g in groups];expected=reference_plan['cohort_qids']
    if len(qids)!=6528 or len(set(qids))!=6528 or set(qids)!=set(expected):
        raise ValueError('Exact original matched6528 cohort required')
    batches=copy.deepcopy(old_plan['batches']);epochs=list(old_plan['batch_epochs'])
    used=[q for batch in batches for q in batch];lookup={g['qid']:g for g in groups}
    if (len(batches)!=136 or epochs!=[0]*136 or any(len(b)!=48 for b in batches) or
            len(used)!=6528 or len(set(used))!=6528 or set(used)!=set(qids) or
            any(sum(0<sum(lookup[q]['labels'])<20 for q in b)<24 for b in batches)):
        raise ValueError('Inherited G10 B48x136 order must remain complete and strictly balance-feasible')
    # Preserve every actual G10 question batch/order. Only the authentic source
    # pool and its descriptive metadata expand from10 to20 trajectories.
    return dict(cohort_qids=list(old_plan['cohort_qids']),batches=batches,batch_epochs=epochs,
        original_questions=6528,global_original_questions=48,updates=136,source_group_size=20,
        source_groups_sha256=digest20(validate20(groups)),seed=1729,epoch=0,
        inherited_G10_plan=copy.deepcopy(old_plan_ref),inherited_G10_batches_unchanged=True,
        inherited_G10_batch_epochs_unchanged=True,fresh_G20_label_reordering=False,
        reference_matched_plan_identity_sha256=u.digest(reference_plan),
        fit_real_trajectories=130560,views_per_question=4,all_questions_once=True,
        old_B51_128_equal_update_budget_claim=False,new_original_question_batch=48,new_update_budget=136)


def make_config(reference,name,hashes,commit,train_complete,validation_complete,plan,evaluations):
    if name not in VARIANTS or reference['source_checkpoint']['sha256']!=u.SOURCE_SHA:
        raise ValueError('Fixed E1 raw/EMA/fullAdam source and explicit grouping required')
    c={k:copy.deepcopy(reference[k]) for k in ('source_checkpoint','source_config','source_bank','model_config','model_dir')}
    grouping=dict(mode=name,seed=1729)
    c.update(g20_complete=copy.deepcopy(train_complete),validation_g10_complete=copy.deepcopy(validation_complete),
        plan=copy.deepcopy(plan),update_budget=136,global_questions=48,smaller_question_budget_acknowledged=True,
        micro_trajectories=5,save_every=16,max_virtual_views=192,max_inter_view_pairs=9216,
        grouping=grouping,recipe=dict(id='g20-matched6528-'+name,loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),
        evaluations=copy.deepcopy(evaluations),code_hashes=copy.deepcopy(hashes),source_commit=commit,
        fit_questions=6528,fit_real_trajectories=130560,source_group_size=20,supports_G20=True,source_world_size=4,current_world_size=1,
        source_adam_steps=10464,actor_updates=0,screening_only=True,optimizer_reset=False,block_reset=False,readout_reset=False,
        equal_budget_to_B51_128_baseline_claim=False,four_rank_bitexact_resume_claim=False,
        original_question_total_weight=1.,view_weight_numerator=1,view_weight_denominator=4,
        first_actual_update_replay_required=True,validation_features_only_in_evaluation=True)
    return c


def render_sbatch(name,base,code_root):
    import shlex
    if name not in VARIANTS:raise ValueError('Unknown matched-G20 variant')
    for item in (base,code_root):
        p=Path(item)
        if p.parent!=Path(TARGET) or any(x.isspace() for x in item) or any(x in item for x in '\r\n;`$'):
            raise ValueError('Independent safe matched-G20 child paths required')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-g20m6528-@@NAME@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE_Q@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.g20_qualification_entry --base @@BASE_Q@@
'''.replace('@@NAME@@',name).replace('@@BASE@@',base).replace('@@BASE_Q@@',shlex.quote(base)).replace('@@CODE_Q@@',shlex.quote(code_root))


def prepare(base,train_path,validation_path):
    if str(base)!=TARGET or base.resolve()!=base:raise ValueError('Only independent matched6528 suite path allowed')
    # All actual completion gates precede bank/label consumption and config creation.
    train=u.read(train_path);val=u.read(validation_path)
    if train['status']!=TRAIN_STATUS or train['questions']!=6528 or train['trajectories']!=130560:
        raise ValueError('Actual COMPLETE6528 train G20 required; partial inputs are not accepted')
    consumer=validation_consumer(val)
    train_ref=ref(train_path);val_ref=ref(validation_path)
    if train_ref['sha256']!='81e3e6b74666e3a0eb32d1b0a0fb9e6db13d54f30a351437ce57f13be46bc229':raise ValueError('Known actual authentic G20 complete SHA differs')
    if val_ref['sha256']!=VALIDATION_COMPLETE_SHA or any(consumer[k]['sha256']!=want for k,want in VALIDATION_CONSUMER_SHA.items()):
        raise ValueError('Known actual sealed validation complete/consumer SHA differs')
    pub=u.read(base/'source-publication.json');root=Path(pub['code_root'])
    if root.parent!=base or root.name!='code-'+pub['commit'][:12]:raise ValueError('Independent immutable code root required')
    for name,item in pub['files'].items():u.checked(dict(path=str(root/name),**item))
    reference_suite=Path(PARENT)/'algorithm-improvement-20261021/matched-loss128-v2'
    accepted=u.read(reference_suite/'preparation.json')
    if accepted['status']!='MATCHED_LOSS128_SUITE_PREPARED_ACTUAL_CPU_TESTS_PASSED':raise ValueError('Accepted matched128 source reference required')
    source_ref=accepted['variants']['event']['config'];reference=u.read(u.checked(source_ref))
    if reference['source_checkpoint']['sha256']!=u.SOURCE_SHA:raise ValueError('Sealed E1 SHA differs')
    for key in ('source_checkpoint','source_config','source_bank','model_config','bank','features','plan'):u.checked(reference[key])
    groups,_manifests,acceptance=accepted_g20(train_ref)
    old_suite=u.read(Path(PARENT)/'algorithm-improvement-20261021/g10-matched6528-v1/preparation.json')
    old_plan_ref=old_suite['source_plan'];old_plan=u.read(u.checked(old_plan_ref))
    old_complete=u.read(u.checked(old_suite['g10_complete']));old_groups=u.read(u.checked(old_complete['groups']))['train']
    assert_old10_prefix(groups,old_groups)
    plan=shared_plan(groups,u.read(reference['plan']['path']),old_plan,old_plan_ref)
    # Every immutable validation output and its pre-generation freeze is SHA bound.
    for item in val['outputs'].values():u.checked(item)
    for key in ('plan_freeze','feature_source_SHA_audit','fixed_G5_EM_evaluation_audit'):u.checked(val[key])
    for key in ('groups','features','fixed_G5_plan'):u.checked(consumer[key])
    validation_groups=u.read(consumer['groups']['path'])[consumer['group_split']]
    fixed_plan=u.read(consumer['fixed_G5_plan']['path']);frozen=frozen_validation(validation_groups,fixed_plan)
    if len(frozen)!=2048:raise ValueError('Require all2048 precommitted validation questions')
    paths=u.read(consumer['features']['path'])
    if not isinstance(paths,list) or len(set(paths))!=len(paths) or not all(isinstance(p,str) and Path(p).is_absolute() for p in paths):
        raise ValueError('Flat native manifest paths required, never event mapping descriptors')
    # Bind manifest metadata only. GPU FeatureReader still verifies every used
    # native row; this does not claim a fresh bulk feature-payload SHA scan here.
    manifests=[ref(p) for p in paths]
    for p in paths:
        value=u.read(p)
        if value['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or tuple(value[k] for k in ('feature_layer','dimension','question_max_tokens','prefix_tokens'))!=(26,3584,273,20):
            raise ValueError('Expected authentic nativeL26/Q273/P20 manifests')
    diagnostic=u.read(reference['bank']['path'])['diagnostic'];fixed_diag=fixed_existing_g5(diagnostic)
    if len(diagnostic)!=512:raise ValueError('Existing512 diagnostic cohort differs')
    external=validation_groups+diagnostic
    if (len({g['qid'] for g in external})!=2560 or len({g['question_hash'] for g in external})!=2560 or
            {g['qid'] for g in groups}&{g['qid'] for g in external} or
            {g['question_hash'] for g in groups}&{g['question_hash'] for g in external}):
        raise ValueError('Training/diagnostic/validation question identity overlap')
    (base/'inputs').mkdir(exist_ok=True)
    plan_path=base/'inputs/plan.json';diag_path=base/'inputs/diagnostic512-fixed-g5.json'
    immutable(plan_path,plan);immutable(diag_path,fixed_diag)
    evaluations=[dict(name='diagnostic512',role='validation',bank=reference['bank'],split='diagnostic',features=reference['features'],
        fixed_g5=ref(diag_path),screening_only=True,previously_observed=True),
        dict(name='selection2048FreshG5',role='validation',bank=consumer['groups'],split=consumer['group_split'],
        features=consumer['features'],fixed_g5=consumer['fixed_G5_plan'],screening_only=True,previously_observed_questions=True,
        two_fresh_G5_events=True,untouched_final=False,selection_used_EM=False)]
    hashes={n:r['sha256'] for n,r in pub['files'].items()};variants={}
    for name in VARIANTS:
        dest=base/name;dest.mkdir(exist_ok=True);(dest/'logs').mkdir(exist_ok=True)
        config=make_config(reference,name,hashes,pub['commit'],train_ref,val_ref,ref(plan_path),evaluations)
        validate_plan(plan,groups,config,forbidden=[g['qid'] for g in external])
        immutable(dest/'config.json',config);script=render_sbatch(name,str(dest),str(root));script_path=dest/'fit.sbatch'
        if script_path.exists():
            if script_path.read_text()!=script:raise ValueError('Existing sbatch differs')
        else:script_path.write_text(script)
        variants[name]=dict(base=str(dest),job_name='q7b-g20m6528-'+name,config=ref(dest/'config.json'),sbatch=ref(script_path))
    done=dict(status='MATCHED6528_AUTHENTIC_G20_B48_136_STRICT_EQUAL_VIEW_SUITE_PREPARED',commit=pub['commit'],code_root=str(root),
        g20_complete=train_ref,validation_g10_complete=val_ref,source_plan=ref(plan_path),fixed_g5=consumer['fixed_G5_plan'],
        cpu_qualification=ref(base/'actual-cpu-tests.log'),source_reference_config=source_ref,variants=variants,
        heterogeneous_questions=sum(0<sum(g['labels'])<20 for g in groups),questions=6528,real_train_trajectories=130560,
        global_original_questions=48,updates=136,views_per_question=4,uniform_view_weight='1/4',
        train_labels_used_only_for_shared_plan_and_training_grouping=True,validation_selection_uses_EM=False,
        actual_validation_native_manifest_refs=manifests,source_Adam_steps=10464,optimizer_reset=False,
        raw_EMA_block_readout_preserved=True,same_G10_B48_136_batches_and_order=True,inherited_G10_plan=old_plan_ref,source_world_size=4,current_world_size=1,four_rank_bitexact_resume_claim=False,
        smaller_question_budget_acknowledged=True,equal_budget_to_B51_128_baseline_claim=False,
        same6528_cohort_as_old_matched128=True,actor_updates=0,checkpoint_tensors_loaded_on_login=False,
        model_tensors_loaded_on_login=False,source_group_size=20,supports_G20=True,comparison_between_random_and_balanced_exact_same_plan=True,G10_total_trajectory_compute_equal_claim=False,fresh_full_native_payload_SHA_claim=False,final_pool_read=False,
        no_automatic_submission=True,no_dependency=True)
    immutable(base/'preparation.json',done)
    return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--training-complete',type=Path,required=True)
    p.add_argument('--validation-complete',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.base,a.training_complete,a.validation_complete)),flush=True)
