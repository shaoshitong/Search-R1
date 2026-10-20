"""Typed CPU predeployment binding; deliberately no GPU train/submit action.

Only the already accepted randomG20 recipe's G5 view objective changes. This
preparation preserves its exact E1/fullAdam refs, 136 B48 batches, source G20
bank and native fixed-G5 evaluation refs. Source tensor SHA acceptance is
inherited explicitly; this metadata preparation does not reread model/bulk
features or claim fresh tensor validation or actual GPU qualification.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from . import probe_ablation_train as u
from .probe_exact_uniform_g5_dp import MODE

ROOT = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
REFERENCE = ROOT / 'g20-matched6528-v1/random/config.json'
TARGET = ROOT / 'g20-exact-uniform15504-dp-preparation-v1'
REFERENCE_SHA = '428161412354177e509fe0336e1b5cf9eb7beb28cd58c0ce7c78c88335338a1c'
PLAN_SHA = '9d2ec808aa4365d80c7c7031487e0e746ad2165ceb2bfd59214aa631ab587e7a'
TRAIN_SHA = '81e3e6b74666e3a0eb32d1b0a0fb9e6db13d54f30a351437ce57f13be46bc229'
VALIDATION_SHA = '5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'
REQUIRED = ('exact_uniform_g5_dp_loss.py', 'probe_exact_uniform_g5_dp.py',
    'g5_event_scores_dp.py', 'exact_uniform_g5_loss.py', 'g20_grouping.py')


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=u.sha(path))


def make_config(reference, plan, *, hashes, commit):
    if (reference['source_checkpoint']['sha256'] != u.SOURCE_SHA
            or reference['global_questions'] != 48 or reference['update_budget'] != 136
            or reference['fit_questions'] != 6528 or reference['fit_real_trajectories'] != 130560
            or reference['grouping'].get('mode') != 'random'
            or reference.get('view_weight_denominator') != 4
            or len(commit) != 40):
        raise ValueError('Actual sealed-E1 random4 authentic-G20/B48x136 reference required')
    used = [q for batch in plan['batches'] for q in batch]
    if (len(plan['batches']) != 136 or plan['batch_epochs'] != [0] * 136
            or any(len(b) != 48 for b in plan['batches'])
            or len(used) != 6528 or len(set(used)) != 6528
            or set(used) != set(plan['cohort_qids'])):
        raise ValueError('Exact same6528 question identities/order and136 batches required')
    required = {'reproduction/operations/qwen7b_warmup/' + name for name in REQUIRED}
    if not required <= set(hashes):
        raise ValueError('New DP/updater/source-validation immutable bindings missing')
    if reference['recipe']['loss_kwargs'] != dict(trajectory_weight=1., three_event_weight=1.):
        raise ValueError('Preserve the accepted trajectory1 plus three-event1 objective')
    for evaluation in reference['evaluations']:
        if evaluation['role'] != 'validation':
            raise ValueError('No final pool in this preparation')
    config = copy.deepcopy(reference)
    config.update(grouping=dict(mode=MODE, source_group_size=20, combination_chunk=256),
        recipe=dict(id='g20-matched6528-exact-uniform15504-DP',
            loss_kwargs=dict(trajectory_weight=1., three_event_weight=1.)),
        code_hashes=copy.deepcopy(hashes), source_commit=commit,
        inherited_reference_plan=copy.deepcopy(reference['plan']),
        original_question_total_weight=1., source_group_size=20,
        combinations_per_question=15504, view_weight_numerator=1, view_weight_denominator=15504,
        exact_uniform_all_G5=True, maximum_resident_G5_rows=48 * 256,
        maximum_resident_pattern_rows=0, maximum_resident_DP_event_states=3 * 48 * 256,
        retained_logit_VJP_elements=48 * 20, first_order_loss_VJP_only=True,
        no_extra_native_model_forwards_relative_to_random4=True,
        inherited_E1_raw_EMA_Adam10464_RNG=True, optimizer_reset=False,
        explicit_launch_authorized=False, GPU_requested=0,
        deployment_stage='TYPED_PREPARATION_ONLY_NOT_GPU_QUALIFIED',
        training_entrypoint=None, updater_module='reproduction.operations.qwen7b_warmup.probe_exact_uniform_g5_dp',
        future_GPU_requires_explicit_authorization_and_complete_runner_wiring=True,
        runtime_scope='native_original_G20_features_and_identical_native_fixed_G5_validation',
        source_heterogeneous20_is_not_used_as_G5_target=True)
    config.pop('max_virtual_views', None)
    config.pop('max_inter_view_pairs', None)
    return config


def prepare(base, benchmark):
    if base != TARGET or base.resolve() != base:
        raise ValueError('Independent pinned DP preparation directory required')
    if u.sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Pinned actual743370 randomG20 configuration changed')
    original = u.read(REFERENCE)
    for key, expected in (('plan', PLAN_SHA), ('g20_complete', TRAIN_SHA),
            ('validation_g10_complete', VALIDATION_SHA)):
        if original[key]['sha256'] != expected:
            raise ValueError('Inherited authentic source/plan/validation binding changed')
        u.checked(original[key])
    u.checked(original['model_config'])
    train = u.read(original['g20_complete']['path'])
    if (train['status'] != 'ACTUAL_ORIGINAL5_PLUS_THREE_INDEPENDENT_G5_G20_SHA_OWN_EM_SEEDS_VERIFIED'
            or train['questions'] != 6528 or train['trajectories'] != 130560):
        raise ValueError('Existing authentic G20 acceptance required')
    results = u.read(benchmark)
    if (results['status'] != 'ACTUAL_CPU2_Q48_G20_ALL15504_DP_BENCHMARK_COMPLETE'
            or not results['synthetic_only_not_learning_evidence'] or results['GPU_requested'] != 0
            or len(results['variants']) != 3
            or any(v['peak_RSS_bytes'] > 3 * 1024 ** 3 for v in results['variants'])):
        raise ValueError('Actual finite CPU equivalence/memory benchmark must pass first')
    pub = u.read(base / 'source-publication.json')
    code = Path(pub['code_root'])
    hashes = {name: entry['sha256'] for name, entry in pub['files'].items()}
    for name, sha in hashes.items():
        if u.sha(code / name) != sha:
            raise ValueError('Immutable published DP source differs')
    plan = u.read(original['plan']['path'])
    config = make_config(original, plan, hashes=hashes, commit=pub['commit'])
    config['reference_config'] = ref(REFERENCE)
    config['CPU_benchmark'] = ref(benchmark)
    config['source_tensor_acceptance'] = dict(mode='INHERITED_PREVIOUS_AUTHENTIC_G20_AND_SEALED_E1_ACCEPTANCE',
        source_checkpoint=original['source_checkpoint'], bulk_tensor_SHA_repeated=False)
    path = base / 'config.json'
    if path.exists():
        if u.read(path) != config:
            raise ValueError('Existing immutable typed preparation differs')
    else:
        u.atomic(path, config)
    result = dict(status='ACTUAL_G20_ALL15504_DP_TYPED_INPUTS_PREPARED_NO_GPU_LAUNCH',
        config=ref(path), reference_config=ref(REFERENCE), source_plan=original['plan'],
        g20_complete=original['g20_complete'], validation_g10_complete=original['validation_g10_complete'],
        source_checkpoint=original['source_checkpoint'], source_commit=pub['commit'],
        source_publication=ref(base / 'source-publication.json'), CPU_benchmark=ref(benchmark),
        questions=6528, source_group_size=20, real_train_trajectories=130560,
        global_questions=48, updates=136, combinations_per_question=15504,
        actor_updates=0, GPU_requested=0, model_or_checkpoint_tensor_loaded=False,
        feature_tensor_loaded=False, optimizer_reset=False, source_Adam_steps=10464,
        actual_GPU_qualification=False, epoch=time.time())
    u.atomic(base / 'preparation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.benchmark)), flush=True)
