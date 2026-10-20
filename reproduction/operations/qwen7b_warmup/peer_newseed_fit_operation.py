"""CPU gates and GPU entry for the separate authentic NQ8192 probe branch.

Preparation reads sealed JSON/feature indexes, never checkpoint tensors. Bulk
feature SHA and full source-state loading remain inside the owned GPU job.
There is no scheduler, automatic rollback, or reader substitution in this file.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import time
import traceback

from . import peer_newseed_train as fit
from . import peer_newseed_ingest as raw

B = raw.B
TARGET = fit.TARGET
JOB_NAME = 'q7b-peer8192-g10-hf-fit'
PREPARED = 'PEER_NQ8192_G10_HF_FIT_PREPARED_ACTUAL_CPU_TESTS_PASSED'
VALIDATION_STATUS = 'NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'
VALIDATION_SOURCE_SHA = '5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'
FIXED_G5_SHA = 'c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
DIAGNOSTIC_REPLAY_SHA = 'c593363dfaca0edf3f142e1fa5f93bd7f461432ce7b0f1e186101039ac163dc4'
VALIDATION_READER_AUDIT = dict(path=str(raw.A/'validation-fixedG5-hf-replay-20261021/independent-reader-audit.json'),
    bytes=2595,sha256='7c37920375dc13eff72cf0fef1b99153a03bed3113bd9e5c22646d60c99a8b6f')
COMPLETE_STATUS = 'REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE'


def metadata_features(manifests, groups, *, allow_unused=False):
    """Bind actual replay index rows, without hashing all feature payloads twice."""
    from .replay_feature_reader import ReplayFeatureReader, FILES
    from .replay_probe_layers import CAPTURE_KIND, MODEL, REVISION
    wanted = {t: g['qid'] for g in groups for t in g['trajectory_ids']}
    if len(wanted) != sum(len(g['trajectory_ids']) for g in groups):
        raise ValueError('Repeated requested feature trajectory')
    seen = set(); paths = set(); selected = []; all_ids = set(); prompts = {}
    checker = object.__new__(ReplayFeatureReader); checker.layer = 26; checker.dimension = 3584
    for descriptor in manifests:
        path = raw.checked(descriptor)
        if str(path) in paths: raise ValueError('Repeated feature manifest')
        paths.add(str(path)); m = raw.read(path)
        if (m.get('status') != 'LOSSLESS_HF_REPLAY_FEATURES_COMPLETE'
                or m.get('source_runtime') != 'huggingface_frozen_causal_prefill'
                or m.get('capture_layer') != 26 or m.get('feature_layer') != 26
                or m.get('capture_kind') != CAPTURE_KIND
                or m.get('dimension') != 3584 or m.get('question_max_tokens') != 273
                or m.get('prefix_tokens') != 20 or m.get('model') != MODEL
                or m.get('model_revision') != REVISION
                or m.get('actor_optimizer_updates') != 0
                or m.get('bit_exact_native_vllm_claimed') is not False
                or m.get('layout') != 'lossless_question_base_xor_policy_v1'
                or m.get('feature_dtype', 'bfloat16_bits_uint16') != 'bfloat16_bits_uint16'
                or set(m['files']) != FILES):
            raise ValueError('Expected immutable frozen-HF L26/Q273/P20 BF16 features')
        for name, item in m['files'].items():
            p = Path(item['path'])
            if p != path.parent/name or not p.is_file() or p.stat().st_size != item['bytes']:
                raise ValueError('Feature payload locator/actual bytes differ')
        raw.checked(m['files']['index.jsonl'])
        if m['files']['prototype.bin']['bytes'] != 273*3584*2:
            raise ValueError('Feature prototype dimensions differ')
        count = 0; relevant = False
        with (path.parent/'index.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line); checker.validate_row(row, m); count += 1
                tid = row['trajectory_id']
                if tid in all_ids: raise ValueError('Repeated actual trajectory between feature manifests')
                all_ids.add(tid)
                if tid not in wanted:
                    if not allow_unused: raise ValueError('Unexpected fitting feature trajectory')
                    continue
                if row['qid'] != wanted[tid]: raise ValueError('Own feature qid/trajectory binding differs')
                key = (row['question_length'], row['prompt_sha256'])
                if prompts.setdefault(row['qid'], key) != key:
                    raise ValueError('Question context identity differs between its own features')
                seen.add(tid); relevant = True
        if count != m['count']: raise ValueError('Actual feature index count differs')
        if relevant: selected.append(copy.deepcopy(descriptor))
    if seen != set(wanted): raise ValueError('Actual HF feature coverage is incomplete')
    return selected, dict(trajectories=len(seen),questions=len(prompts),manifests=len(selected),
        index_SHA_fresh=True,all_bulk_feature_payload_SHA_verified=False,
        full_payload_SHA_required_inside_GPU_reader=True)


def make_config(reference, publication, training_input, plan, evaluations, domain_scope):
    from .probe_ablation_train import SOURCE_SHA
    if reference['source_checkpoint']['sha256'] != SOURCE_SHA:
        raise ValueError('Exact completed E1 source checkpoint is required')
    c = {k: copy.deepcopy(reference[k]) for k in
         ('source_checkpoint','source_config','source_bank','model_config','model_dir')}
    c.update(g10_complete=copy.deepcopy(training_input),plan=copy.deepcopy(plan),
        update_budget=256,global_questions=32,smaller_question_budget_acknowledged=True,
        micro_trajectories=5,save_every=16,max_virtual_views=128,max_inter_view_pairs=4096,
        grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),
        recipe=dict(id='peer-NQ8192-authentic-G10-HF-E1-onepass',
                    loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),
        evaluations=copy.deepcopy(evaluations),
        code_hashes={n:v['sha256'] for n,v in publication['files'].items()},
        source_commit=publication['commit'],fit_questions=8192,fit_real_trajectories=81920,
        source_adam_steps=10464,source_world_size=4,current_world_size=1,
        canonical_source_rng_rank=0,source_all_four_rank_RNG_preserved=True,
        feature_backend='huggingface_frozen_causal_prefill',domain_scope=copy.deepcopy(domain_scope),
        actor_updates=0,optimizer_reset=False,readout_reset=False,block_reset=False,
        original_question_total_weight=1.,view_weight_numerator=1,view_weight_denominator=4,
        first_actual_update_replay_required=True,four_rank_bitexact_resume_claim=False,
        screening_only=True,equal_budget_to_controlled6528_claim=False,
        final_pool_read=False,validation_features_only_in_evaluation=True)
    return c


def render_sbatch(base, code_root):
    base = str(base); code_root = str(code_root)
    if Path(base) != TARGET or Path(code_root).parent != TARGET:
        raise ValueError('Independent peer8192 root/code child required')
    for value in (base,code_root):
        if any(c.isspace() for c in value) or any(c in value for c in '\r\n;`$'):
            raise ValueError('Unsafe remote path')
    template = '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-peer8192-g10-hf-fit
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
#SBATCH --output=@@PEER_BASE@@/logs/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@PEER_CODE_QUOTED@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_newseed_fit_operation --action train --base @@PEER_BASE_QUOTED@@
'''
    return template.replace('@@PEER_BASE@@',base).replace('@@PEER_CODE_QUOTED@@',shlex.quote(code_root)).replace('@@PEER_BASE_QUOTED@@',shlex.quote(base))


def validation_inputs(validation_complete):
    """Actual new fixed-G5 capture only; shared by early E4 and later fitting."""
    from .probe_g10_train import frozen_validation
    validation=raw.read(validation_complete)
    if (validation.get('status')!=VALIDATION_STATUS or validation.get('questions')!=2048
            or validation.get('trajectories')!=10240 or validation.get('capture_layers')!=[26]
            or validation.get('source_runtime')!='huggingface_frozen_causal_prefill'
            or validation.get('model_revision')!=raw.REVISION
            or validation.get('source_complete',{}).get('sha256')!=VALIDATION_SOURCE_SHA
            or validation.get('fixed_G5_sha256')!=FIXED_G5_SHA
            or validation.get('new_G5_selected') is not False or validation.get('labels_read') is not False
            or validation.get('actor_updates')!=0 or validation.get('probe_updates')!=0
            or validation.get('new_responses_generated')!=0 or validation.get('final_pool_read') is not False):
        raise ValueError('Actual NEW precommitted fixed2048 G5 HF capture required, never old R2 selection')
    for key in ('source_complete','source_selection','source_groups','source_fixed_G5','features'):
        raw.checked(validation[key])
    audit=raw.read(raw.checked(VALIDATION_READER_AUDIT))
    if (audit.get('status')!='ALL_10240_NEW_FIXED_G5_HF_ROWS_RECONSTRUCTED_RETAINED_SHA_PASSED'
            or audit.get('complete')!=raw.ref(validation_complete) or audit.get('features')!=validation['features']
            or audit.get('all_manifest_and_payload_SHA256') is not True
            or audit.get('all_reconstructed_row_SHA256') is not True or audit.get('trajectories')!=10240
            or audit.get('fixed_G5_sha256')!=FIXED_G5_SHA or audit.get('labels_read') is not False
            or audit.get('final_pool_read') is not False or audit.get('actor_updates')!=0 or audit.get('probe_updates')!=0):
        raise ValueError('Actual independent reconstruction of all10240 new validation feature rows required')
    groups=raw.read(validation['source_groups']['path'])['validation']
    plan=raw.read(validation['source_fixed_G5']['path']);fixed=frozen_validation(groups,plan)
    if len(fixed)!=2048 or plan.get('fixed_G5',plan)['summary']['selected_g5_sha256']!=FIXED_G5_SHA:
        raise ValueError('Original label-blind new validation fixedG5 changed')
    manifests=raw.read(validation['features']['path'])
    if manifests!=validation['feature_manifests_by_capture_layer']['26']:
        raise ValueError('Validation manifest list binding differs')
    _,proof=metadata_features(manifests,fixed)
    e=dict(name='selection2048FreshG5HF',role='validation',bank=validation['source_groups'],split='validation',
        features=validation['features'],fixed_g5=validation['source_fixed_G5'],
        backend='huggingface_frozen_causal_prefill',new_G5_selected=False,untouched_final=False)
    return validation,groups,proof,e


def sealed_e4(base):
    e4base=B/'balanced-rank1-from-epoch1-prefix20-20261020'
    receipt=raw.read(e4base/'probe/epoch-0004.json');config=raw.read(e4base/'config.json')
    if (receipt.get('sha256')!='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
            or receipt.get('steps')!=14418 or receipt.get('epoch')!=4 or receipt.get('completed_epoch') is not True
            or receipt.get('cursor')!=0 or receipt.get('independent_cpu_reload') is not True):
        raise ValueError('Actual sealed E4 baseline lineage differs')
    raw.immutable(base/'inputs/E4-source-sealed-receipt.json',receipt)
    value={k:receipt[k] for k in ('path','bytes','sha256','steps')}
    value.update(name='E4',epoch=4,source_receipt=raw.ref(base/'inputs/E4-source-sealed-receipt.json'),
        source_config=raw.ref(e4base/'config.json'),model_dir=config['model_dir'])
    return value


def sealed_e1(base):
    prep=raw.read(raw.A/'matched-loss128-v2/preparation.json')
    reference=raw.read(raw.checked(prep['variants']['event']['config']))
    source=reference['source_checkpoint'];receipt=raw.read(Path(source['path']).with_suffix('.json'))
    if (receipt.get('sha256')!='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
            or receipt.get('steps')!=10464 or receipt.get('epoch')!=1 or receipt.get('completed_epoch') is not True
            or receipt.get('cursor')!=0 or receipt.get('independent_cpu_reload') is not True):
        raise ValueError('Actual sealed E1 baseline lineage differs')
    raw.immutable(base/'inputs/E1-source-sealed-receipt.json',receipt)
    value={k:receipt[k] for k in ('path','bytes','sha256','steps')}
    value.update(name='E1',epoch=1,source_receipt=raw.ref(base/'inputs/E1-source-sealed-receipt.json'),
        source_config=reference['source_config'],source_bank=reference['source_bank'],model_dir=reference['model_dir'])
    return value


def prepare_baseline(base,validation_complete):
    base=Path(base)
    if base!=TARGET:raise ValueError('Only isolated peer8192 parent is allowed')
    pub=raw.read(base/'source-publication.json');root=Path(pub['code_root'])
    for name,item in pub['files'].items():raw.checked(dict(path=str(root/name),**item))
    validation,groups,proof,e=validation_inputs(validation_complete)
    (base/'inputs').mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
    source=sealed_e4(base);e1=sealed_e1(base)
    if source['model_dir']!=e1['model_dir']:raise ValueError('E1/E4 official geometry differs')
    c=dict(model_dir=source['model_dir'],model_config=raw.ref(Path(source['model_dir'])/'config.json'),
        same_HF_frozen_validation_E4_baseline=source,same_HF_frozen_validation_E1_baseline=e1,evaluations=[e],
        code_hashes={n:v['sha256'] for n,v in pub['files'].items()},source_commit=pub['commit'],
        source_validation_complete=raw.ref(validation_complete),validation_reader_audit=copy.deepcopy(VALIDATION_READER_AUDIT),
        fitting_authorized=False,actor_updates=0,probe_updates=0)
    path=base/'inputs/E4-only-config.json';raw.immutable(path,c)
    script=render_sbatch(base,root).replace('#SBATCH --job-name='+JOB_NAME,'#SBATCH --job-name=q7b-peer-E4-same-HF-val').replace('#SBATCH --time=02:00:00','#SBATCH --time=00:30:00').replace('/logs/train-%j.log','/logs/E4-baseline-%j.log').replace('--action train --base','--action baseline --base')
    sbatch=base/'E4-baseline.sbatch'
    if sbatch.exists() and sbatch.read_text()!=script:raise ValueError('Immutable early E4 baseline script differs')
    if not sbatch.exists():sbatch.write_text(script)
    done=dict(status='SEALED_E4_NEW_FROZEN_G5_HF_BASELINE_PREPARED',commit=pub['commit'],code_root=str(root),
        config=raw.ref(path),sbatch=raw.ref(sbatch),validation_complete=raw.ref(validation_complete),
        questions=2048,trajectories=10240,feature_metadata=proof,optimizer_constructed=False,
        actor_updates=0,probe_updates=0,job_name='q7b-peer-E4-same-HF-val')
    raw.immutable(base/'E4-baseline-preparation.json',done);return done


def prepare(base, raw_complete, identity_plan, feature_completes, validation_complete, diagnostic_complete):
    """All actual complete gates precede immutable training config creation."""
    from .probe_g10_train import frozen_validation, validate_plan
    from .deploy_g10_pilot_suite import fixed_existing_g5
    base=Path(base)
    if base != TARGET or base.resolve() != base: raise ValueError('Only isolated declared peer8192 target')
    paths=[raw_complete,identity_plan]+list(feature_completes)+[validation_complete,diagnostic_complete]
    if len(feature_completes)!=4 or any(not Path(p).is_file() for p in paths):
        raise ValueError('Actual raw/identity/four feature/new validation/diagnostic complete files required')
    pub=raw.read(base/'source-publication.json');root=Path(pub['code_root'])
    if root.parent != base or root.name != 'code-'+pub['commit'][:12]:
        raise ValueError('Immutable published code root required')
    for name,item in pub['files'].items():raw.checked(dict(path=str(root/name),**item))
    if (base/'preparation.json').exists():
        old=raw.read(base/'preparation.json')
        expected=[raw.ref(p) for p in paths]
        if old['source_refs']!=expected or old['commit']!=pub['commit']:
            raise ValueError('Existing preparation belongs to different immutable sources')
        for key in ('config','sbatch','training_input','source_plan','cpu_tests'):raw.checked(old[key])
        return old
    validation,valgroups,valproof,valevaluation=validation_inputs(validation_complete)
    reference_preparation=raw.read(raw.A/'matched-loss128-v2/preparation.json')
    if reference_preparation['status']!='MATCHED_LOSS128_SUITE_PREPARED_ACTUAL_CPU_TESTS_PASSED':
        raise ValueError('Accepted E1 reference preparation required')
    reference=raw.read(raw.checked(reference_preparation['variants']['event']['config']))
    # Source state is not deserialized on the login node. The GPU runner hashes,
    # loads and exactly checks all raw/EMA/Adam/RNG state before its first update.
    for key in ('source_config','source_bank','model_config','bank'):raw.checked(reference[key])
    source_checkpoint=Path(reference['source_checkpoint']['path'])
    if not source_checkpoint.is_file() or source_checkpoint.stat().st_size<1024**3:
        raise ValueError('Actual full source checkpoint file is missing')
    diagnostic=raw.read(reference['bank']['path'])['diagnostic']
    if len(diagnostic)!=512:raise ValueError('Original diagnostic512 identity changed')
    dr=raw.ref(diagnostic_complete);d=raw.read(diagnostic_complete)
    if (dr['sha256']!=DIAGNOSTIC_REPLAY_SHA
            or d['status']!='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED'
            or d['source_runtime']!='huggingface_frozen_causal_prefill' or d['actor_updates']!=0
            or d['new_responses_generated']!=0 or d['full_replay_shared_file_SHA_verified'] is not True):
        raise ValueError('Accepted R2 HF replay required only for original diagnostic512')
    dmanifests=raw.read(raw.checked(d['features_by_capture_layer']['26']))
    dmanifests,diagproof=metadata_features(dmanifests,diagnostic,allow_unused=True)
    rc,ip=raw.ref(raw_complete),raw.ref(identity_plan);fc=[raw.ref(p) for p in feature_completes]
    if (base/'training-input.json').exists():
        own=raw.read(base/'training-input.json')
        if (own['raw_complete']!=rc or own['identity_plan']!=ip or own['feature_shard_completes']!=fc):
            raise ValueError('Existing peer training acceptance differs')
    else:fit.accept_inputs(base/'training-input.json',rc,ip,fc)
    groups,manifests,acceptance=fit.accepted_peer_g10(raw.ref(base/'training-input.json'))
    manifests,trainproof=metadata_features(manifests,groups)
    # Both training and new validation use the SAME actual qualified batching
    # primitive/golden; diagnostic uses the separately qualified scalar-HF bank.
    gold=validation['batching']['golden'];core=validation['batching']['core_sha256']
    for item in fc:
        capture=raw.read(item['path'])
        if capture.get('batching',{}).get('golden')!=gold or capture['batching'].get('core_sha256')!=core:
            raise ValueError('Training/new-validation qualified HF primitive/golden differs')
    external=diagnostic+valgroups
    if (len({g['qid'] for g in external})!=2560 or len({g['question_hash'] for g in external})!=2560
            or {g['qid'] for g in groups}&{g['qid'] for g in external}
            or {g['question_hash'] for g in groups}&{g['question_hash'] for g in external}):
        raise ValueError('Train/diagnostic/new validation qid or hash overlap')
    (base/'inputs').mkdir(exist_ok=True);(base/'logs').mkdir(exist_ok=True)
    for name,value in [('diagnostic512-features-HF.json',dmanifests),('diagnostic512-fixed-G5.json',fixed_existing_g5(diagnostic))]:
        raw.immutable(base/'inputs'/name,value)
    evaluations=[dict(name='diagnostic512',role='validation',bank=reference['bank'],split='diagnostic',
        features=raw.ref(base/'inputs/diagnostic512-features-HF.json'),fixed_g5=raw.ref(base/'inputs/diagnostic512-fixed-G5.json'),
        backend='huggingface_frozen_causal_prefill',previously_observed=True,old_R2_selection2048_evaluated=False),
        valevaluation]
    config=make_config(reference,pub,raw.ref(base/'training-input.json'),raw.ref(base/'plan.json'),evaluations,acceptance['domain_scope'])
    prepare_baseline(base,validation_complete);e4=sealed_e4(base)
    if e4['model_dir']!=reference['model_dir']:raise ValueError('E4/source E1 official model geometry differs')
    config['same_HF_frozen_validation_E4_baseline']=e4
    config['validation_reader_audit']=copy.deepcopy(VALIDATION_READER_AUDIT)
    config['historical_native_E4_result_used_as_matched_baseline']=False
    validate_plan(raw.read(base/'plan.json'),groups,config,forbidden=[g['qid'] for g in external])
    raw.immutable(base/'config.json',config)
    script=render_sbatch(base,root);path=base/'fit.sbatch'
    if path.exists() and path.read_text()!=script:raise ValueError('Existing immutable fit script differs')
    if not path.exists():path.write_text(script)
    tests=base/'actual-cpu-tests.log'
    if not tests.is_file() or tests.stat().st_size<10:raise ValueError('Actual published CPU tests required')
    done=dict(status=PREPARED,commit=pub['commit'],code_root=str(root),config=raw.ref(base/'config.json'),
        sbatch=raw.ref(path),training_input=raw.ref(base/'training-input.json'),source_plan=raw.ref(base/'plan.json'),
        cpu_tests=raw.ref(tests),source_refs=[raw.ref(p) for p in paths],job_name=JOB_NAME,
        questions=8192,real_trajectories=81920,updates=256,global_questions=32,
        source_adam_steps=10464,expected_final_adam_steps=10720,views_per_question=4,view_weight=.25,
        source_all_four_rank_RNG_preserved=True,canonical_source_rng_rank=0,optimizer_reset=False,
        domain_scope=acceptance['domain_scope'],feature_backend='huggingface_frozen_causal_prefill',
        actual_metadata_audits=dict(train=trainproof,diagnostic512=diagproof,new_validation2048=valproof),
        training_validation_golden=gold,batched_primitive_sha256=core,actor_updates=0,
        checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False,
        final_pool_read=False,no_dependency=True,no_priority_escalation=True)
    raw.immutable(base/'preparation.json',done);return done


def evaluate_e4_baseline(base):
    """Score sealed E1/E4 using one shared reader and the identical new HF pool."""
    import torch
    from . import probe_ablation_train as u
    from .evaluate_frozen_validation import validate_checkpoint_metadata, evaluate
    from .model import Probe
    config_path=base/'inputs/E4-only-config.json';c=raw.read(config_path);source=c['same_HF_frozen_validation_E4_baseline']
    e1=c['same_HF_frozen_validation_E1_baseline']
    e=next(x for x in c['evaluations'] if x['name']=='selection2048FreshG5HF')
    dest=base/'E4-same-HF-validation';dest.mkdir(exist_ok=True)
    binding=dict(config_sha256=u.sha(config_path),source_checkpoint=source,source_E1_checkpoint=e1,
                 evaluation=e,actor_updates=0,probe_updates=0,optimizer_constructed=False)
    if (dest/'complete.json').exists():
        done=raw.read(dest/'complete.json')
        if any(done[k]!=v for k,v in binding.items()):raise ValueError('Prior same-HF E4 baseline binding differs')
        for value in done['checkpoints'].values():raw.checked(value['predictions'])
        return done
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise ValueError('Same-HF E4 baseline requires the owned single GPU job')
    if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Fixed deterministic startup flags required')
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    props=torch.cuda.get_device_properties(0)
    if 'H100' not in props.name or props.total_memory<75*1024**3:raise ValueError('Dedicated H100 required')
    root=Path(__file__).resolve().parents[3]
    for name,want in c['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or u.sha(root/name)!=want:
            raise ValueError('Immutable early EMA comparison runtime changed')
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    raw.checked(c['model_config']);model=Probe(c['model_dir'],273,20).to(device).eval().requires_grad_(False)
    groups=__import__(__package__+'.probe_g10_train',fromlist=['frozen_validation']).frozen_validation(
        raw.read(raw.checked(e['bank']))[e['split']],raw.read(raw.checked(e['fixed_g5'])))
    reader=fit.replay_reader(raw.read(raw.checked(e['features'])))
    results={}
    try:
        for descriptor in (e1,source):
            name=descriptor['name'];output=dest/name;output.mkdir(exist_ok=True)
            own=dict(config_sha256=u.sha(config_path),checkpoint=descriptor,evaluation=e)
            if (output/'complete.json').exists():
                v=raw.read(output/'complete.json')
                if any(v[k]!=value for k,value in own.items()):raise ValueError('Prior EMA comparison binding differs')
                raw.checked(v['predictions']);results[name]=v;continue
            receipt=raw.read(raw.checked(descriptor['source_receipt']));sc=raw.read(raw.checked(descriptor['source_config']))
            state=torch.load(raw.checked(descriptor),map_location='cpu',mmap=True,weights_only=False)
            if name=='E4':validate_checkpoint_metadata(descriptor,receipt,sc,state)
            elif (state.get('steps')!=10464 or state.get('epoch')!=1 or state.get('completed_epoch') is not True
                    or state.get('cursor')!=0 or state.get('actor_updates')!=0
                    or state.get('config_sha256')!=descriptor['source_config']['sha256']
                    or state.get('bank_sha256')!=descriptor['source_bank']['sha256']
                    or len(state.get('rank_states',[]))!=4
                    or {s['rank'] for s in state['rank_states']}!={0,1,2,3}):
                raise ValueError('Actual E1 comparison source state differs')
            ema_hash=u.model_mapping_hash(state['ema'])
            if any(s['ema_sha256']!=ema_hash or s['steps']!=descriptor['steps'] for s in state['rank_states']):
                raise ValueError('Sealed source EMA/four-rank step hashes disagree')
            model.load_state_dict(state['ema'],strict=True);u.restore_buffers(model,state['ema_buffers'])
            native=copy.deepcopy(state['ema_buffers']);del state
            before=u.model_hash(model)
            metrics=evaluate(model,reader,groups,device,output/'predictions.jsonl.gz',dest/'progress.json',name+'-same-HF',micro_groups=8)
            if (before!=u.model_hash(model) or before!=ema_hash
                    or any(p.requires_grad or p.grad is not None for p in model.parameters())):
                raise ValueError('Evaluation changed EMA state or computed parameter gradients')
            u.restore_buffers(model,native)
            if u.tensor_hash(u.buffers(model))!=u.tensor_hash(native):raise ValueError('EMA native buffers failed exact restoration')
            v=dict(status='ONE_SEALED_EMA_SAME_NEW_HF_FIXED_G5_COMPLETE',metrics=metrics,
                predictions=raw.ref(output/'predictions.jsonl.gz'),ema_sha256=ema_hash,
                optimizer_constructed=False,actor_updates=0,probe_updates=0,**own)
            raw.immutable(output/'complete.json',v);results[name]=v
    finally:reader.close()
    ids={(v['metrics']['ordered_qids_sha256'],v['metrics']['ordered_own_trajectory_ids_sha256']) for v in results.values()}
    if len(ids)!=1:raise ValueError('E1 and E4 did not use identical fixed-G5 identities')
    done=dict(status='SEALED_E1_E4_SAME_NEW_FROZEN_G5_HF_BASELINE_COMPLETE',metrics=results['E4']['metrics'],
        predictions=results['E4']['predictions'],checkpoints=results,same_question_and_trajectory_ids=True,**binding)
    raw.immutable(dest/'complete.json',done);del model;torch.cuda.empty_cache();return done


def train_entry(base):
    import torch
    from .probe_ablation_train import atomic
    started=time.time();normal=False
    with (base/'peer8192.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            c=raw.read(base/'config.json');root=Path(__file__).resolve().parents[3]
            for name,want in c['code_hashes'].items():
                if Path(name).is_absolute() or '..' in Path(name).parts or u_sha(root/name)!=want:
                    raise ValueError('Immutable fitting/baseline runtime changed: '+name)
            if not (base/'E4-same-HF-validation/complete.json').exists():
                raise ValueError('Separate same-HF E4 baseline job must complete before fitting')
            evaluate_e4_baseline(base)  # verifies immutable prior receipt; performs no duplicate GPU inference
            fit.run(base);normal=True
        except BaseException as error:
            atomic(base/'failure.json',dict(type=type(error).__name__,message=str(error),
                traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')))
            raise
        finally:
            if torch.cuda.is_initialized():
                atomic(base/'gpu-memory.json',dict(status='ACTUAL_SAME_PROCESS_CUDA_ALLOCATOR_PEAK',
                    device=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),started=started,time=time.time(),
                    exit_normally=normal,job_id=os.environ.get('SLURM_JOB_ID')))


def u_sha(path):
    # Kept metadata-only and import-light for source binding validation.
    return raw.ref(path)['sha256']


def resume_decision(latest, progress=None, intent=None, complete=None):
    """Metadata-only conservative gate; GPU independently rehashes/restores state."""
    if complete:
        if complete.get('status')!=COMPLETE_STATUS or complete.get('cursor')!=256 or complete.get('steps')!=10720:
            raise ValueError('Malformed final completion')
        return 'COMPLETE_NO_SUBMISSION'
    if not latest:
        if progress or intent:raise ValueError('Unresolved initial update; cannot reset source or guess outcome')
        return 'INITIAL_SOURCE_FORK'
    cursor=latest.get('cursor');steps=latest.get('steps')
    if type(cursor) is not int or not 1<=cursor<=256 or steps!=10464+cursor:
        raise ValueError('Exact saved source-step/cursor required')
    if progress and (progress.get('cursor',0)>cursor or progress.get('steps',steps)>steps):
        raise ValueError('Unsaved later accepted progress; refuse automatic rollback')
    if intent and intent.get('next_cursor',0)>cursor:
        raise ValueError('Unresolved later update intent; refuse automatic rollback')
    if set(latest.get('accepted_hashes',{}))!={'raw','ema','adam','raw_buffers','ema_buffers','rng'}:
        raise ValueError('All six saved next-update state components required')
    return 'RESUME_EXACT_ACCEPTED_CURSOR' if cursor<256 else 'RESUME_FINAL_EVALUATION_ONLY'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','prepare-baseline','baseline','train'),required=True)
    p.add_argument('--base',type=Path,default=TARGET);p.add_argument('--raw-complete',type=Path);p.add_argument('--identity-plan',type=Path)
    p.add_argument('--feature-complete',type=Path,action='append');p.add_argument('--validation-complete',type=Path);p.add_argument('--diagnostic-complete',type=Path)
    a=p.parse_args();base=a.base.resolve()
    if base!=TARGET:p.error('Only declared remote peer8192 target is permitted')
    if a.action=='train':train_entry(base)
    elif a.action=='baseline':
        with (base/'E4-baseline.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:print(json.dumps(evaluate_e4_baseline(base),sort_keys=True),flush=True)
            except BaseException as error:
                raw.atomic(base/'E4-baseline-failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time()))
                raise
    elif a.action=='prepare-baseline':print(json.dumps(prepare_baseline(base,a.validation_complete),sort_keys=True),flush=True)
    else:
        value=prepare(base,a.raw_complete,a.identity_plan,a.feature_complete or [],a.validation_complete,a.diagnostic_complete)
        print(json.dumps({k:value[k] for k in ('status','questions','updates','global_questions','config','code_root')},sort_keys=True),flush=True)


if __name__=='__main__':main()
