"""CPU JSON-only preparation, gated by actual E4 scoring and CPU selection."""
import argparse,copy,json
from pathlib import Path
from . import probe_ablation_train as u
from . import evaluate_frozen_validation as ev
from .hard_enrichment_train import ROLE,SOURCE_SHA,SOURCE_STEPS,LOSS,validate_config,validate_plan,render_sbatch

PARENT=ev.PARENT
TARGET=PARENT+'/algorithm-improvement-20261021/hard-vs-random-E4-477-v1'
SELECT=PARENT+'/algorithm-improvement-20261021/hard-enrichment-20261021/selection-complete.json'
VALIDATION=PARENT+'/algorithm-improvement-20261021/validation-g10-20261021/materialized/bank/complete.json'
VALIDATION_SHA='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'


def ref(p):return ev.proof(p)


def immutable(p,v):
    if p.exists():
        if u.read(p)!=v:raise ValueError('Existing immutable source/config differs: '+str(p))
    else:u.atomic(p,v)


def check_selection(selected):
    if (selected['status']!='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE' or
            set(selected['outputs'])!={'targeted','random_control'} or selected['checkpoint']['sha256']!=SOURCE_SHA or
            selected['checkpoint']['steps']!=SOURCE_STEPS or selected['checkpoint']['epoch']!=4 or
            selected['GPU_used'] is not False or selected['actor_updates']!=0 or selected['probe_updates']!=0):
        raise ValueError('Both actual selected22896 plans must exist from complete E4 scorer, never assumed')
    expected=dict(same_core=True,same_unique_count=True,same_macro_class_counts=True,same_seed=True,same_update_budget=True,
        Mixed_K1_K2_K3_K4_matched=False,initial_warmup_unseen_claim=False,previous_finetuning_unseen_claim=False)
    if selected['controls']!=expected:raise ValueError('Actual paired controls differ')
    for name,item in selected['outputs'].items():
        summary=item['summary']
        if (summary['branch']!=name or summary['total_unique_questions']!=22896 or summary['core_questions']!=6528 or
                summary['supplement_questions']!=16368 or summary['supplement_class_counts']!=dict(allwrong=4092,allcorrect=4092,Mixed=8184) or
                summary['update_budget']!=477 or summary['global_groups']!=48 or summary['epochs']!=1 or
                summary['validation_or_final_questions_used'] is not False or
                summary['question_selection_uses_scores']!=(name=='targeted')):
            raise ValueError('Selected branch macro-class/data/update budget binding differs')


def make_config(name,selected,selection_ref,source,source_receipt,source_config,model_config,manifest_refs,hashes,commit):
    check_selection(selected);item=selected['outputs'][name]
    c=dict(role=ROLE,variant=name,source_checkpoint=source,source_receipt=source_receipt,source_config=source_config,
        source_bank=selected['source_bank'],bank=item['groups'],features=selected['features'],plan=item['plan'],
        selection_complete=selection_ref,candidate_plan=selected['source_candidates'],scoring_complete=selected['source_scoring_receipt'],
        diagnostic_bank=selected['source_bank'],validation_complete=dict(path=VALIDATION,bytes=4492,sha256=VALIDATION_SHA),
        feature_manifest_refs=manifest_refs,model_dir=source['model_dir'],model_config=model_config,
        update_budget=477,fit_questions=22896,global_groups=48,micro_groups=2,save_every=32,
        prefix_tokens=20,question_max_tokens=273,lr=5e-6,ema_decay=.99,weight_decay=0.,
        recipe=dict(id='trajectory1-three-event1-inter-rank1',loss_kwargs=copy.deepcopy(LOSS)),
        seed=item['summary']['seed'],code_hashes=copy.deepcopy(hashes),source_commit=commit,
        source_world_size=4,current_world_size=1,canonical_source_rng_rank=0,four_rank_bitexact_resume_claim=False,
        source_raw_ema_adam_RNG_preserved=True,E1_reset=False,actor_updates=0,
        training_role='same-current-E4-probe targeted repair versus macro-class-matched random',untouched_final_read=False)
    validate_config(c);return c


def prepare(base):
    if str(base)!=TARGET or base.resolve()!=base:raise ValueError('Use the new isolated E4 hard/random suite only')
    selected=u.read(SELECT);check_selection(selected);selection_ref=ref(SELECT)
    for key in ('source_candidates','source_scoring_receipt','source_predictions','source_config','source_bank','features','source_matched_plan'):
        ev.checked(selected[key])
    for item in selected['exclusion_refs']:ev.checked(item)
    scoring=u.read(selected['source_scoring_receipt']['path'])
    if (scoring['status']!='E4_PREFIX20_TRAIN_ONLY_CANDIDATE_SCORING_COMPLETE' or
            scoring['questions']!=32768 or scoring['trajectories']!=163840 or
            scoring['binding']['candidate_plan']!=selected['source_candidates'] or
            scoring['predictions']!=selected['source_predictions'] or
            scoring['binding']['checkpoint']['sha256']!=SOURCE_SHA):raise ValueError('Actual selected scoring lineage differs')
    source=copy.deepcopy(scoring['binding']['checkpoint']);receipt=ref(ev.checked(source['source_receipt']))
    sr=u.read(receipt['path']);source_config=ref(ev.checked(source['source_config']));old=u.read(source_config['path'])
    if (sr['epoch']!=4 or sr['steps']!=SOURCE_STEPS or sr['cursor']!=0 or sr['completed_epoch'] is not True or
            sr['independent_cpu_reload'] is not True or sr['sha256']!=SOURCE_SHA or
            Path(sr['path']).stat().st_size!=sr['bytes'] or old['fit_lr']!=5e-6 or old['fit_ema']!=.99):
        raise ValueError('Only sealed actual E4 raw/EMA/Adam14418 source metadata accepted')
    # Source tensor bytes are full-SHA verified and loaded only in allocated GPU jobs.
    model_config=ref(Path(source['model_dir'])/'config.json');mc=u.read(model_config['path'])
    if (mc['num_hidden_layers'],mc['hidden_size'])!=(28,3584):raise ValueError('Official7B native block/head geometry differs')
    vr=dict(path=VALIDATION,bytes=4492,sha256=VALIDATION_SHA);validation=u.read(ev.checked(vr))
    if validation['status']!=ev.BANK_STATUS or not validation['fixed_G5_was_chosen_before_own_EM']:raise ValueError('Actual frozen fixed validation G5 required')
    validation_groups=u.read(ev.checked(validation['consumer_inputs']['groups']))['validation']
    full=u.read(selected['source_bank']['path']);lookup={g['qid']:g for g in full['train']};diag=full['diagnostic']
    if len(lookup)!=169028 or len(diag)!=512:raise ValueError('Canonical169028/fixeddiag512 required')
    forbidden={g['qid'] for g in diag+validation_groups};hashes={g['question_hash'] for g in diag+validation_groups}
    if len(forbidden)!=2560:raise ValueError('Evaluation question identities overlap')
    for name,item in selected['outputs'].items():
        train=u.read(ev.checked(item['groups']))['train'];plan=u.read(ev.checked(item['plan']));validate_plan(plan,train,forbidden,hashes)
        for g in train:
            if any(g[k]!=lookup[g['qid']][k] for k in ('qid','question_hash','trajectory_ids','labels')):raise ValueError('Selected train group not authentic canonical own EM/IDs')
    feature_paths=u.read(selected['features']['path']);refs=[]
    for path in feature_paths:
        descriptor=ref(path);v=u.read(path)
        if (v['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or
                (v['feature_layer'],v['prefix_tokens'],v['dimension'],v['question_max_tokens'])!=(26,20,3584,273)):
            raise ValueError('Only native L26/P20/Q273/D3584 accepted')
        refs.append(descriptor)
    (base/'inputs').mkdir(exist_ok=True);mp=base/'inputs/native-feature-manifest-refs.json';immutable(mp,refs)
    pub=u.read(base/'source-publication.json');code=Path(pub['code_root']);hashes={name:item['sha256'] for name,item in pub['files'].items()}
    for name,item in pub['files'].items():ev.checked(dict(path=str(code/name),**item))
    variants={}
    for name in ('targeted','random_control'):
        root=base/name;root.mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
        c=make_config(name,selected,selection_ref,source,receipt,source_config,model_config,ref(mp),hashes,pub['commit'])
        immutable(root/'config.json',c);script=render_sbatch(name,str(root),str(code));path=root/'fit.sbatch'
        if path.exists():
            if path.read_text()!=script:raise ValueError('Immutable paired sbatch differs')
        else:path.write_text(script)
        variants[name]=dict(base=str(root),job_name='q7b-hard477-'+name,config=ref(root/'config.json'),sbatch=ref(path))
    done=dict(status='E4_HARD_RANDOM_PAIRED477_INPUTS_AND_CPU_TESTS_PREPARED_NOT_GPU_QUALIFIED',commit=pub['commit'],code_root=str(code),
        selection_complete=selection_ref,validation_complete=vr,source_checkpoint=source,source_adam_steps=SOURCE_STEPS,
        variants=variants,fit_questions=22896,fit_updates_per_variant=477,global_groups=48,micro_groups=2,save_every=32,
        common_loss=LOSS,same_macro_class_ratios=True,Mixed_internal_K_distribution_matched=False,
        checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False,no_dependency=True,no_automatic_submission=True,
        source_raw_ema_adam_RNG_preserved=True,E1_reset=False,four_rank_bitexact_resume_claim=False,
        actual_gpu_next_update_replay_passed=False,cpu_tests=ref(base/'actual-cpu-tests.log'),actor_updates=0)
    immutable(base/'preparation.json',done);return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.base)))
