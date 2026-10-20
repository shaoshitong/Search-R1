"""Frozen E2/E4/event128 EMA comparison on one accepted label-blind G5 pool.

Evaluation only: no optimizer is constructed, no parameters are updated, and
neither labels nor checkpoint scores change the frozen question/trajectory IDs.
Raw predictions and all tensor reads stay in the allocated remote GPU job.
Completed per-checkpoint results can be freshly verified after interruption;
an interrupted checkpoint is evaluated again, never confused with completion.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

PARENT = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
BANK_STATUS = 'VALIDATION_ACTUAL_TWO_FRESH_G5_G10_SHA_OWN_EM_VERIFIED'
STATUS = 'FIXED_VALIDATION_G5_E2_E4_EVENT128_EMA_EVALUATION_COMPLETE'
EXPECTED = {'E2': (11782, 2), 'E4': (14418, 4), 'event128': (10592, None)}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def proof(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''): h.update(data)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def remote(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or
            not str(path).startswith(PARENT+'/')):
        raise ValueError('All source/model/prediction paths must remain remote: '+str(path))
    return path


def checked(item):
    path = remote(item['path']); actual = proof(path)
    if actual['sha256'] != item['sha256'] or ('bytes' in item and actual['bytes'] != item['bytes']):
        raise ValueError('SHA/size changed: '+str(path))
    return path


def fixed_groups(groups, plan):
    """Freeze IDs first, then join only those trajectories' own strict EM."""
    plan = plan.get('fixed_G5', plan)
    if plan['summary'].get('role') != 'validation' or plan['summary'].get('labels_consulted') is not False:
        raise ValueError('Explicit precommitted label-blind validation G5 required')
    selected = plan['groups']; by = {g['qid']: g for g in groups}
    if len(by) != len(groups) or len(selected) != len(by) or {g['qid'] for g in selected} != set(by):
        raise ValueError('Frozen validation question coverage differs')
    if digest(selected) != plan['summary']['selected_g5_sha256']:
        raise ValueError('Frozen selection IDs changed')
    out = []; seen = set()
    for row in selected:
        if set(row) != {'qid', 'trajectory_ids'}:
            raise ValueError('Labels or extra selection fields entered precommitted G5')
        own = by[row['qid']]; ids = row['trajectory_ids']
        if (len(ids) != 5 or len(set(ids)) != 5 or seen.intersection(ids) or
                len(own['trajectory_ids']) != 10 or len(set(own['trajectory_ids'])) != 10 or
                len(own['labels']) != 10 or not set(ids) <= set(own['trajectory_ids']) or
                any(type(v) is not int or v not in (0, 1) for v in own['labels'])):
            raise ValueError('Distinct own real G10 IDs/strict EM required')
        seen.update(ids); labels = dict(zip(own['trajectory_ids'], own['labels']))
        out.append(dict(qid=row['qid'], trajectory_ids=list(ids), labels=[labels[t] for t in ids]))
    return out


def validate_checkpoint_metadata(descriptor, receipt, config, state):
    name = descriptor['name']
    if name not in EXPECTED or descriptor['steps'] != EXPECTED[name][0] or state['steps'] != descriptor['steps']:
        raise ValueError('Wrong fixed checkpoint step identity')
    if state.get('actor_updates') != 0 or state['config_sha256'] != descriptor['source_config']['sha256']:
        raise ValueError('Checkpoint actor/config lineage differs')
    if config['model_dir'] != descriptor['model_dir']:
        raise ValueError('Checkpoint official model metadata differs')
    for key in ('path', 'sha256', 'bytes', 'steps'):
        if receipt.get(key) != descriptor.get(key):
            raise ValueError('Sealed checkpoint receipt differs: '+key)
    if name == 'event128':
        if (state.get('role') != 'NEW_SINGLE_GPU_MATCHED_PROBE_LOSS_ABLATION' or
                state['cursor'] != 128 or receipt['cursor'] != 128 or config['update_budget'] != 128 or
                config['recipe']['id'] != 'event' or config['recipe']['loss_kwargs'] !=
                {'trajectory_weight': 1., 'three_event_weight': 1.} or
                len(state['source_all_rank_states']) != 4 or
                state['matched_binding_sha256'] != receipt['matched_binding_sha256'] or
                state['accepted_hashes'] != receipt['accepted_hashes']):
            raise ValueError('Require actual accepted three-event128 completed update budget')
        if any('rng' not in row for row in state['source_all_rank_states']):
            raise ValueError('Missing original four-rank provenance')
    else:
        epoch = EXPECTED[name][1]
        if (state.get('role') != 'EPOCH1_BALANCED_RANK1_CONTINUATION' or state['epoch'] != epoch or
                receipt['epoch'] != epoch or state['completed_epoch'] is not True or
                receipt['completed_epoch'] is not True or state['cursor'] != 0 or receipt['cursor'] != 0 or
                receipt['independent_cpu_reload'] is not True or
                config['prefix_tokens'] != 20 or config['question_max_tokens'] != 273 or
                config['native_head_layer'] != 27 or config['fit_ema'] != .99 or
                len(state['rank_states']) != 4 or {row['rank'] for row in state['rank_states']} != set(range(4)) or
                any(row['steps'] != descriptor['steps'] or 'rng' not in row for row in state['rank_states'])):
            raise ValueError('Require sealed completed balanced epoch with four-rank provenance')


def summarize(rows, top_k=512):
    """Keep the established fixed-.5 group definitions and deterministic ties."""
    from .probe_ablation_train import prediction_metrics
    if type(top_k) is not int or not 1 <= top_k <= len(rows):
        raise ValueError('Explicit selection budget must fit the question pool')
    if (len({row['qid'] for row in rows}) != len(rows) or
            any(len(row['strict_em']) != 5 or len(row['p_j']) != 5 or
                any(type(y) is not int or y not in (0, 1) for y in row['strict_em']) or
                any(not math.isfinite(p) or not 0 <= p <= 1 for p in row['p_j']) or
                not math.isfinite(row['predicted_mixed']) or not 0 <= row['predicted_mixed'] <= 1 for row in rows)):
        raise ValueError('Unique own real G5 questions and finite probabilities required')
    selected = sorted(rows, key=lambda row: (-row['predicted_mixed'], str(row['qid'])))[:top_k]
    pool = prediction_metrics(rows); chosen = prediction_metrics(selected)
    # Expose both nonMixed subclasses' FPRs, rather than only their aggregate.
    for metrics in (pool, chosen):
        metrics['allwrong_false_positive_rate'] = metrics['classwise']['allwrong']['false_positive_rate']
        metrics['allcorrect_false_positive_rate'] = metrics['classwise']['allcorrect']['false_positive_rate']
        metrics['fixed_threshold'] = .5
    return dict(pool=pool, selected512=chosen, selection_budget=top_k,
        selected_qids_sha256=digest([row['qid'] for row in selected]),
        ordered_qids_sha256=digest([row['qid'] for row in rows]),
        ordered_own_trajectory_ids_sha256=digest([row['trajectory_ids'] for row in rows]),
        labels_used_to_select=False, threshold_fitted=False,
        tie_break='descending_q_then_lexicographic_string_qid')


def accepted_validation(reference):
    complete = read(checked(reference))
    if (complete['status'] != BANK_STATUS or complete['role'] != 'validation' or
            complete['untouched_final'] is not False or complete['tuning_allowed'] is not True or
            complete['fitting_on_validation_allowed'] is not False or complete['synthetic_EM'] is not False or
            complete['actor_optimizer_updates'] != 0 or complete['questions'] != 2048 or
            complete['trajectories'] != 20480 or complete['model_revision'] != REVISION or
            complete['fixed_G5_was_chosen_before_own_EM'] is not True):
        raise ValueError('Only accepted viewed validation, never untouched-final/fitting data')
    for descriptor in complete['outputs'].values(): checked(descriptor)
    checked(complete['plan_freeze'])
    consumer = complete['consumer_inputs']
    if (consumer['group_split'] != 'validation' or consumer['selection_labels_consulted'] is not False or
            consumer['features_format'] != 'flat_list_of_native_complete_paths'):
        raise ValueError('Accepted validation consumer contract differs')
    for key, filename in (('groups', 'groups.json'), ('features', 'features.json'),
                          ('fixed_G5_plan', 'fixed-G5-selection.json')):
        if consumer[key] != complete['outputs'][filename]:
            raise ValueError('Consumer file differs from accepted output')
    frozen = fixed_groups(read(checked(consumer['groups']))['validation'],
                          read(checked(consumer['fixed_G5_plan'])))
    audit = read(checked(complete['fixed_G5_EM_evaluation_audit']))['validation']
    if frozen != audit: raise ValueError('Fixed G5 own EM differs from independent merge audit')
    manifests = read(checked(consumer['features']))
    refs = read(checked(complete['feature_source_SHA_audit']))
    descriptors = [row for own in refs.values() for row in own]
    if manifests != list(dict.fromkeys(row['path'] for row in descriptors)):
        raise ValueError('Feature source audit differs from canonical flat manifest list')
    from .packed_features import FeatureReader
    for descriptor in descriptors:
        path = checked(descriptor); value = read(path)
        if (value['status'] != 'LOSSLESS_NATIVE_FEATURES_COMPLETE' or value['feature_layer'] != 26 or
                value['prefix_tokens'] != 20 or value['dimension'] != 3584 or value['question_max_tokens'] != 273):
            raise ValueError('Native L26 question+Prefix20 geometry required')
        for key in ('index.jsonl', 'prototype.bin'):
            if checked(value['files'][key]).resolve() != (path.parent/key).resolve():
                raise ValueError('Native index/prototype own path binding differs')
    reader = FeatureReader(manifests)
    for group in frozen:
        for tid in group['trajectory_ids']:
            if tid not in reader.records or reader.records[tid][1]['qid'] != group['qid']:
                reader.close(); raise ValueError('Frozen own trajectory absent/wrong question in native reader')
    return frozen, reader, complete


def evaluate(model, reader, groups, device, destination, progress, name, micro_groups=2):
    import torch
    from .probe_ablation_loss import probe_ablation_loss
    model.eval(); rows = []; total = {k: 0. for k in ('trajectory_bce', 'group_bce', 'group_event_nll')}
    started = time.monotonic(); io_seconds = compute_seconds = 0.
    with torch.inference_mode():
        for start in range(0, len(groups), micro_groups):
            part = groups[start:start+micro_groups]; before = time.monotonic()
            batch = reader.get([tid for g in part for tid in g['trajectory_ids']], device)
            torch.cuda.synchronize(device); io_seconds += time.monotonic()-before; before = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16): z = model(batch).reshape(len(part), 5)
            labels = torch.tensor([g['labels'] for g in part], device=device, dtype=torch.float32)
            # One forward supplies scores and all diagnostic losses. No loss.backward().
            loss = probe_ablation_loss(z, labels, trajectory_weight=1., three_event_weight=1.)
            probabilities = z.sigmoid().cpu().tolist(); q = loss['mixed_probability'].cpu().tolist()
            for key in total: total[key] += len(part)*float(loss[key])
            for group, p, score in zip(part, probabilities, q):
                rows.append(dict(qid=group['qid'], trajectory_ids=group['trajectory_ids'],
                    p_j=p, predicted_mixed=score, strict_em=group['labels']))
            torch.cuda.synchronize(device); compute_seconds += time.monotonic()-before
            if start % (32*micro_groups) == 0 or len(rows) == len(groups):
                atomic(progress, dict(checkpoint=name, completed_questions=len(rows), expected_questions=len(groups),
                    completed_trajectories=5*len(rows), elapsed_seconds=time.monotonic()-started,
                    feature_io_seconds=io_seconds, compute_seconds=compute_seconds,
                    probe_optimizer_updates=0, actor_optimizer_updates=0, observed_epoch=time.time()))
            del z, loss, batch, labels
    temporary = destination.with_suffix('.tmp')
    with gzip.open(temporary, 'wt') as stream:
        for row in rows: stream.write(json.dumps(row, allow_nan=False)+'\n')
    temporary.replace(destination)
    metrics = summarize(rows)
    metrics['diagnostic_losses'] = {key: value/len(groups) for key, value in total.items()}
    metrics['timing'] = dict(elapsed_seconds=time.monotonic()-started,
        feature_io_seconds=io_seconds, compute_seconds=compute_seconds)
    return metrics


def run(config_path, expected_config_sha):
    # This guard runs before importing torch or any module capable of tensor IO.
    if (platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID') or
            not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('Only a remote authorized Slurm GPU job may load checkpoints/features')
    config_path = remote(config_path)
    if proof(config_path)['sha256'] != expected_config_sha: raise ValueError('Fixed evaluator config SHA differs')
    config = read(config_path); base = remote(config['base'])
    if not base.is_dir(): raise ValueError('Prepared isolated output root required')
    with (base/'evaluation.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_run(config_path, config, expected_config_sha)


def locked_run(config_path, config, config_sha):
    import torch
    from . import probe_ablation_train as u
    from .model import Probe, native_buffers, restore_buffers
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1 or
            os.environ.get('FLASH_ATTENTION_DETERMINISTIC') != '1' or
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('One GPU with fixed deterministic startup flags required')
    descriptors = config['checkpoints']
    if ([row['name'] for row in descriptors] != ['E2', 'E4', 'event128'] or
            config['evaluation_role'] != 'viewed_recipe_selection_validation_not_untouched_final' or
            config['probe_optimizer_updates_authorized'] is not False or
            config['actor_optimizer_updates_authorized'] is not False):
        raise ValueError('Fixed three-checkpoint evaluation-only configuration required')
    if type(config['micro_groups']) is not int or not 1 <= config['micro_groups'] <= 8:
        raise ValueError('Bounded inference micro batch must contain1..8 complete G5 groups')
    root = Path(__file__).resolve().parents[3]
    required = {'reproduction/operations/qwen7b_warmup/'+name for name in
                ('evaluate_frozen_validation.py', 'model.py', 'packed_features.py',
                 'probe_ablation_train.py', 'probe_ablation_loss.py')}
    if not required <= set(config['code_hashes']):
        raise ValueError('Missing immutable evaluator/model/reader/scoring source bindings')
    for name, wanted in config['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or proof(root/name)['sha256'] != wanted:
            raise ValueError('Published immutable source changed: '+name)
    os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = '273'; os.environ['SEARCH_G1_B_PREFIX_TOKENS'] = '20'
    device = torch.device('cuda', 0); torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    base = remote(config['base']); output = base/'evaluation-output'; output.mkdir(exist_ok=True)
    binding = dict(config=proof(config_path), validation_complete=config['validation_complete'],
        fixed_threshold=.5, top_k=512, checkpoints=descriptors, probe_optimizer_updates=0,
        actor_optimizer_updates=0, untouched_final=False, fitting_performed=False)
    if (output/'intent.json').exists():
        if read(output/'intent.json') != binding: raise ValueError('Prior evaluation has different immutable inputs')
    else: atomic(output/'intent.json', binding)
    groups, reader, acceptance = accepted_validation(config['validation_complete'])
    if len(groups) != 2048: raise ValueError('Exactly2048 fixed questions required')
    model_config = checked(config['model_config']); model_dir = remote(config['model_dir'])
    if model_config != model_dir/'config.json': raise ValueError('Official model config path differs')
    model = Probe(str(model_dir), 273, 20).eval().requires_grad_(False).to(device)
    results = {}
    try:
        for descriptor in descriptors:
            name = descriptor['name']; result_dir = output/name; result_dir.mkdir(exist_ok=True)
            cp = checked(descriptor); receipt = read(checked(descriptor['source_receipt']))
            source_config = read(checked(descriptor['source_config']))
            if (result_dir/'complete.json').exists():
                prior = read(result_dir/'complete.json')
                if prior['config_sha256'] != config_sha or prior['checkpoint'] != descriptor:
                    raise ValueError('Prior completed evaluation binding differs')
                checked(prior['predictions']); results[name] = prior
                continue
            state = torch.load(cp, map_location='cpu', mmap=True, weights_only=False)
            validate_checkpoint_metadata(descriptor, receipt, source_config, state)
            model.load_state_dict(state['ema'], strict=True); restore_buffers(model, state['ema_buffers'])
            ema_sha = u.model_hash(model); buffer_sha = u.tensor_hash(native_buffers(model))
            expected_ema = (state['accepted_hashes']['ema'] if name == 'event128'
                            else state['rank_states'][0]['ema_sha256'])
            if ema_sha != expected_ema: raise ValueError('Loaded actual EMA parameter hash differs')
            if name != 'event128' and any(row['ema_sha256'] != ema_sha for row in state['rank_states']):
                raise ValueError('Original four-rank EMA hashes disagree')
            if name == 'event128' and buffer_sha != state['accepted_hashes']['ema_buffers']:
                raise ValueError('Accepted native EMA buffers differ')
            prediction_path = result_dir/'predictions.jsonl.gz'
            metrics = evaluate(model, reader, groups, device, prediction_path, output/'progress.json', name,
                               micro_groups=config['micro_groups'])
            if u.model_hash(model) != ema_sha or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError('Frozen evaluation changed parameters or generated gradients')
            restore_buffers(model, state['ema_buffers'])
            if u.tensor_hash(native_buffers(model)) != buffer_sha:
                raise ValueError('Original EMA native buffers did not restore exactly')
            if proof(cp) != dict(path=str(cp), bytes=descriptor['bytes'], sha256=descriptor['sha256']):
                raise ValueError('Source checkpoint changed during evaluation')
            result = dict(status='ONE_FROZEN_EMA_ON_SAME_VALIDATION_G5_COMPLETE', checkpoint=descriptor,
                config_sha256=config_sha, metrics=metrics, predictions=proof(prediction_path),
                ema_parameter_sha256=ema_sha, ema_native_buffers_sha256=buffer_sha,
                frozen_parameters_unchanged=True, optimizer_constructed=False, probe_optimizer_updates=0,
                actor_optimizer_updates=0, completed_epoch_time=time.time())
            atomic(result_dir/'complete.json', result); results[name] = result
            del state; torch.cuda.empty_cache()
    finally: reader.close()
    identities = {(r['metrics']['ordered_qids_sha256'], r['metrics']['ordered_own_trajectory_ids_sha256'])
                  for r in results.values()}
    if len(identities) != 1: raise ValueError('Checkpoint evaluations used different validation IDs')
    if (output/'complete.json').exists():
        prior = read(output/'complete.json')
        if prior['status'] != STATUS or prior['binding'] != binding or prior['checkpoints'] != results:
            raise ValueError('Existing aggregate acceptance differs from freshly verified child results')
        return prior
    report = dict(status=STATUS, binding=binding, checkpoints=results, questions=2048, trajectories=10240,
        feature_layer=26, native_block=27, prefix_tokens=20, question_max_tokens=273,
        score='1-prod(sigmoid(z))-prod(1-sigmoid(z))_stable_logspace', fixed_threshold=.5,
        checkpoint_selection_performed=False, same_question_and_trajectory_ids=True,
        comparison_scope='historical_checkpoints_same_validation_not_matched_update_budget',
        actor_optimizer_updates=0, probe_optimizer_updates=0, optimizer_constructed=False,
        untouched_final=False, fitting_performed=False, labels_used_to_select_trajectories=False,
        completed_epoch_time=time.time())
    atomic(output/'complete.json', report)
    print(json.dumps(dict(status=STATUS, output=str(output), checkpoints={name:row['metrics']
        for name,row in results.items()})), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True); args = parser.parse_args()
    try: run(args.config, args.config_sha256)
    except Exception as error:
        if platform.system() == 'Linux' and os.environ.get('SLURM_JOB_ID'):
            path = args.config.parent.parent/'failure.json'
            atomic(path, dict(status='FROZEN_VALIDATION_EVALUATION_FAILED', error=repr(error),
                traceback=traceback.format_exc(), observed_epoch=time.time()))
        raise


if __name__ == '__main__': main()
