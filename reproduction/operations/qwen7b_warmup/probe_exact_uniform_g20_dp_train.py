"""Explicit authentic-G20 all15504 native GPU experiment, separate from CPU gate.

Only this entry wires the DP updater into the native G20 fullstate transaction.
The original runner's default updater and plan validator retain their behavior.
Reference random4 is used solely to verify the unchanged question plan; actual
training grouping/objective is explicitly uniform over every authentic G5.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time
import traceback
import torch
from . import probe_ablation_train as u
from . import probe_g20_train as inherited
from .probe_exact_uniform_g5_dp import MODE, bounded_exact_dp_update
from .prepare_exact_uniform_g20_dp import (make_config as cpu_config, ROOT,
    TARGET as CPU_TARGET, REFERENCE, REFERENCE_SHA, PLAN_SHA, TRAIN_SHA, VALIDATION_SHA, ref)

TARGET = ROOT / 'g20-exact-uniform15504-dp-v1'
RECIPE = 'g20-matched6528-exact-uniform15504-DP'
ENTRY = 'reproduction.operations.qwen7b_warmup.probe_exact_uniform_g20_dp_train'
REQUIRED = ('probe_exact_uniform_g20_dp_train.py', 'probe_g20_train.py',
    'exact_uniform_g5_dp_loss.py', 'probe_exact_uniform_g5_dp.py', 'g5_event_scores_dp.py')
ACTUAL_GROUPING = dict(mode='exact_uniform_G20_all15504', source_group_size=20,
    combinations_per_question=15504, views_per_question=15504,
    view_weight_numerator=1, view_weight_denominator=15504,
    original_question_total_weight=1, combination_chunk=256,
    all_combinations_exactly_once=True, no_inter_view_pair_matrix=True)


def require_config(config):
    if (config['recipe'] != dict(id=RECIPE, loss_kwargs=dict(trajectory_weight=1., three_event_weight=1.))
            or config['grouping'] != dict(mode=MODE, source_group_size=20, combination_chunk=256)
            or config['actual_grouping'] != ACTUAL_GROUPING
            or config['reference_grouping'] != dict(mode='random', seed=1729)
            or config['plan'] != config['inherited_reference_plan']
            or config['plan']['sha256'] != PLAN_SHA
            or config['g20_complete']['sha256'] != TRAIN_SHA
            or config['validation_g10_complete']['sha256'] != VALIDATION_SHA
            or config['source_checkpoint']['sha256'] != u.SOURCE_SHA
            or config['global_questions'] != 48 or config['update_budget'] != 136
            or config['fit_questions'] != 6528 or config['fit_real_trajectories'] != 130560
            or config['training_entrypoint'] != ENTRY
            or not config.get('explicit_launch_authorized') or config.get('GPU_requested') != 1
            or config.get('max_virtual_views') is not None or config.get('max_inter_view_pairs') is not None):
        raise ValueError('Explicit authorized unchanged-source/order/budget real-G20 all15504 experiment required')
    if not {'reproduction/operations/qwen7b_warmup/' + name for name in REQUIRED} <= set(config['code_hashes']):
        raise ValueError('Immutable native callback plus DP runtime binding missing')


def plan_validator(plan, groups, config, forbidden=()):
    require_config(config)
    reference = dict(config, grouping=copy.deepcopy(config['reference_grouping']))
    # Only question-plan compatibility uses this explicitly identified historical
    # random4 metadata. No G5 views are sampled or passed to the actual updater.
    return inherited.validate_plan(plan, groups, reference, forbidden)


def update_callback(model, ema, optimizer, reader, lookup, qids, config, device, epoch):
    require_config(config)
    if epoch != 0 or len(qids) != 48 or len(set(qids)) != 48:
        raise ValueError('One unchanged complete B48 question batch at matched epoch0 required')
    return bounded_exact_dp_update(model, ema, optimizer, reader, [lookup[q] for q in qids], device,
        loss_kwargs=config['recipe']['loss_kwargs'], combination_chunk=256,
        micro_trajectories=config['micro_trajectories'])


def make_config(reference, plan, *, hashes, commit, cpu_preparation):
    config = cpu_config(reference, plan, hashes=hashes, commit=commit)
    config.update(reference_grouping=copy.deepcopy(reference['grouping']),
        actual_grouping=copy.deepcopy(ACTUAL_GROUPING), explicit_launch_authorized=True,
        GPU_requested=1, training_entrypoint=ENTRY, CPU_preparation=copy.deepcopy(cpu_preparation),
        deployment_stage='EXPLICITLY_AUTHORIZED_MATCHED_G20_ALL15504_NATIVE_TRAINING',
        future_GPU_requires_explicit_authorization_and_complete_runner_wiring=False,
        global_virtual_G5_views=48 * 15504, inter_view_pairs_allocated=0,
        maximum_working_G5_chunk_rows=48 * 256, full_all_view_graph_retained=False,
        execution_authorization='Parent/root explicit GO following actual finite CPU DP qualification; single normal one-H100 matched136 job only')
    require_config(config)
    return config


def prepare(base):
    if base != TARGET or base.resolve() != base:
        raise ValueError('Independent pinned actual-training root required')
    if u.sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Original743370 randomG20 config changed')
    reference = u.read(REFERENCE)
    for key, expected in (('plan', PLAN_SHA), ('g20_complete', TRAIN_SHA), ('validation_g10_complete', VALIDATION_SHA)):
        if reference[key]['sha256'] != expected:
            raise ValueError('Inherited accepted source/plan identity differs')
        u.checked(reference[key])
    cpu_path = CPU_TARGET / 'preparation.json'
    accepted = u.read(cpu_path)
    if (accepted['status'] != 'ACTUAL_G20_ALL15504_DP_TYPED_INPUTS_PREPARED_NO_GPU_LAUNCH'
            or accepted['config']['sha256'] != '815b6c14f1705aab5aecded0981ce5050c4a046bf4bb471c0f8087f229462da4'):
        raise ValueError('Actual separately completed CPU DP qualification required')
    u.checked(accepted['CPU_benchmark'])
    publication = u.read(base / 'source-publication.json')
    root = Path(publication['code_root'])
    hashes = {name: value['sha256'] for name, value in publication['files'].items()}
    for name, sha in hashes.items():
        if u.sha(root / name) != sha:
            raise ValueError('New immutable native GPU runtime differs')
    cpu = u.read(u.checked(accepted['config']))
    for name in ('exact_uniform_g5_dp_loss.py', 'g5_event_scores_dp.py', 'probe_exact_uniform_g5_dp.py'):
        path = 'reproduction/operations/qwen7b_warmup/' + name
        if hashes[path] != cpu['code_hashes'][path]:
            raise ValueError('Actual CPU-qualified DP loss/updater source changed')
    config = make_config(reference, u.read(reference['plan']['path']), hashes=hashes,
        commit=publication['commit'], cpu_preparation=ref(cpu_path))
    config['reference_config'] = ref(REFERENCE)
    path = base / 'config.json'
    if path.exists():
        if u.read(path) != config:
            raise ValueError('Existing training configuration preserved')
    else:
        u.atomic(path, config)
    result = dict(status='ACTUAL_NATIVE_G20_ALL15504_MATCHED136_PREPARED_NOT_SUBMITTED',
        config=ref(path), source_publication=ref(base / 'source-publication.json'),
        code_root=str(root), source_commit=publication['commit'], reference_config=ref(REFERENCE),
        plan=reference['plan'], source_checkpoint=reference['source_checkpoint'],
        g20_complete=reference['g20_complete'], validation_g10_complete=reference['validation_g10_complete'],
        CPU_preparation=ref(cpu_path), source_Adam_steps=10464,
        questions=6528, real_trajectories=130560, global_original_questions=48, updates=136,
        actual_grouping=ACTUAL_GROUPING, optimizer_reset=False, actor_updates=0,
        GPU_requested=0, GPU_training_authorized=True, actual_GPU_qualification=False, epoch=time.time())
    u.atomic(base / 'preparation.json', result)
    return result


def run(base):
    config = u.read(base / 'config.json')
    require_config(config)
    normal, started = False, time.time()
    try:
        inherited.run(base, update_callback=update_callback, plan_validator=plan_validator)
        normal = True
    finally:
        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            u.atomic(base / 'gpu-memory.json', dict(status='ACTUAL_G20_ALL15504_DP_SAME_PROCESS_CUDA_PEAK',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                device=torch.cuda.get_device_name(device), started=started,
                time=time.time(), exit_normally=normal))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--action', choices=('prepare', 'train'), required=True)
    args = parser.parse_args()
    if args.base != TARGET or args.base.resolve() != args.base:
        raise ValueError('Independent fixed G20 all15504 output required')
    args.base.mkdir(parents=True, exist_ok=True)
    with (args.base / 'g20-train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.action == 'prepare':
                print(json.dumps(prepare(args.base)), flush=True)
            else:
                run(args.base)
        except BaseException as error:
            u.atomic(args.base / 'failure.json', dict(type=type(error).__name__,
                message=str(error), traceback=traceback.format_exc(), time=time.time()))
            raise
