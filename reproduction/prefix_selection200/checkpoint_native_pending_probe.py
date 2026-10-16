"""Once-only CPU qualification of actor10/probe0 after the first probe fault.

This is a recovery prerequisite for the four-rank question+5/period10 run.  It
does not qualify a completed joint outer10: actor10 is complete, while its first
50-write probe phase has not started.  Full model/Adam reads stay on the remote
CPU, use mmap and bounded finite checks, and never call CUDA or a model forward.
The sealed continuation inputs are reused; no generation or evaluation occurs.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .actor import REVISION, _state_binding, learning_rate
from .checkpoint_cpu_once import (canonical, digest, hash_file, need, qwen_shapes,
    read, signature, tensor_cpu, validate_adam, validate_rng)
from .contract import CANDIDATE_QUESTIONS, DIM, G, PREFIX_TOKENS, QMAX, WIDTH, WORLD
from .prefix5_model import NATIVE_BUFFER_LAYOUTS, PARAMETER_SHAPES

STATUS = 'QUALIFIED_NATIVE_PENDING_PROBE'
PENDING_OUTER = 10
JOINT_OUTER = 9


def _canonical_path(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'Require an absolute canonical recovery path')
    for part in (path, *path.parents):
        need(not part.is_symlink(), 'Recovery input must not contain a symlink component')
    need(path.resolve(strict=True) == path, 'Recovery input path is not canonical')
    return path


class Seals:
    """Full-file SHA once per path; retain exact inode/data timestamps throughout."""
    def __init__(self, base, allowed=()):
        self.base = _canonical_path(base)
        self.allowed = {_canonical_path(p) for p in allowed}
        self.files = {}
        self.directories = {}

    def path(self, path):
        path = _canonical_path(path)
        need(self.base in path.parents or path in self.allowed,
             'Recovery file is neither owned by the source run nor an explicitly pinned source asset')
        return path

    def file(self, path, expected=None):
        path = self.path(path)
        need(path.is_file(), 'Recovery input is not a regular file')
        key = str(path)
        if key not in self.files:
            self.files[key] = hash_file(path, expected)
        result = self.files[key]
        now = path.stat()
        need((now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) ==
             (result['dev'], result['inode'], result['bytes'], result['mtime_ns']),
             'Sealed recovery input changed')
        if expected is not None:
            need(result['sha256'] == expected['sha256'] and result['bytes'] == expected['bytes'],
                 'Recovery file differs from its producer full-file receipt')
        return result

    def json(self, path, expected=None):
        self.file(path, expected)
        value = read(path)
        self.file(path)
        return value

    def tree(self, directory):
        directory = self.path(directory)
        need(directory.is_dir(), 'Missing sealed recovery input directory')
        paths = []
        for parent, dirs, names in os.walk(directory, followlinks=False):
            for name in dirs:
                self.path(Path(parent) / name)
            for name in names:
                paths.append(self.path(Path(parent) / name))
        self.directories[str(directory)] = sorted(str(p) for p in paths)
        for path in sorted(paths):
            self.file(path)

    def mmap_torch(self, path, producer):
        import torch
        path = self.path(path)
        seal = self.file(path, producer)
        # All current native producer checkpoints use torch's zip serialization.
        # No legacy fallback may silently materialize multi-GB state in RAM.
        value = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
        self.file(path)
        return value, dict(seal, actual_cpu_deserialization=True, cpu_mmap=True)

    def verify_producer_references(self, value):
        if isinstance(value, dict):
            if all(key in value for key in ('path', 'bytes', 'sha256')):
                self.file(value['path'], value)
            for child in value.values():
                self.verify_producer_references(child)
        elif isinstance(value, list):
            for child in value:
                self.verify_producer_references(child)

    def finish(self):
        for path in list(self.files):
            self.file(path)
        for name, expected in self.directories.items():
            actual = []
            for parent, dirs, files in os.walk(name, followlinks=False):
                for child in dirs:
                    self.path(Path(parent) / child)
                actual.extend(str(self.path(Path(parent) / child)) for child in files)
            need(sorted(actual) == expected, 'A recovery input directory changed during qualification')
        return [self.files[key] for key in sorted(self.files)]


def _unwrap_terminal(value):
    if isinstance(value.get('stdout'), str):
        need(value.get('exit_code') == 0 and value.get('remote_mutations') is False,
             'Terminal observation wrapper was not a successful readonly observation')
        return json.loads(value['stdout'])
    return value


def _terminal(seals, path, expected_sha, config):
    proof = seals.file(path)
    need(proof['sha256'] == expected_sha, 'The explicitly selected terminal failure proof changed')
    ended = _unwrap_terminal(seals.json(path))
    need(ended.get('run_id') == config['run_id'] and ended.get('execution_id') == config['execution_id']
         and ended.get('source_commit') == config['source_commit'], 'Terminal proof belongs to another execution')
    cleanup = ended.get('cleanup', {})
    need(cleanup.get('run_id') == config['run_id'] and cleanup.get('cleaned') is True
         and cleanup.get('owned_births_all_gone') is True and cleanup.get('remaining_owned_births') == []
         and ended.get('remaining_bound_processes') == [] and ended.get('probe_files') == [],
         'Actual failure must have zero probe files and all recorded owned births gone')
    failed = ended.get('failed_job', {})
    args = failed.get('arguments', [])
    need(failed.get('exit_code') == 1 and failed.get('state') == 'STOPPED'
         and 'reproduction.prefix_selection200.probe' in args and 'update' in args
         and '--outer' in args and args[args.index('--outer') + 1] == '10',
         'Expected the exact first native period10 probe job failure')
    errors = ended.get('error_logs', [])
    need(len(errors) == 1 and errors[0].get('path') == failed.get('log')
         and any('Require the complete512 test or5120 train G5 identity index' in line
                 for line in errors[0].get('error_lines', [])),
         'Failure was not the recorded pre-optimizer G5-count metrics rejection')
    seals.file(errors[0]['path'], errors[0])
    actual_cleanup = seals.json(seals.base / 'logs/cleanup-status.json')
    need(actual_cleanup == cleanup, 'Latest actual cleanup differs from the terminal observation')
    return proof, dict(owned_births_all_gone=True, probe_writes_observed=0,
        failed_before_first_probe_optimizer_construction=True,
        failure_log_sha256=errors[0]['sha256'], failed_probe_exit_code=1)


def _hf(seals, actor, config):
    import torch
    from safetensors import safe_open
    directory = seals.path(actor['hf_model_dir'])
    need(actor['model_master_dtype'] == 'torch.float32' and actor['rollout_compute_dtype'] == 'bfloat16',
         'Recovery must retain the actual FP32 actor masters')
    need(set(p.name for p in directory.iterdir()) == set(actor['hf_files'])
         and all(Path(name).name == name for name in actor['hf_files']),
         'Actor HF directory differs from its complete producer file set')
    for name, producer in actor['hf_files'].items():
        seals.file(directory / name, producer)
    saved = seals.json(directory / 'config.json')
    original = seals.json(Path(config['official_model_dir']) / 'config.json')
    for key in ('model_type', 'hidden_size', 'num_hidden_layers', 'intermediate_size', 'vocab_size',
                'num_attention_heads', 'num_key_value_heads', 'tie_word_embeddings'):
        need(saved.get(key) == original.get(key), 'Native actor architecture differs from the official asset')
    need((saved['hidden_size'], saved['num_hidden_layers'], saved['num_attention_heads'],
          saved['num_key_value_heads'], saved['intermediate_size']) == (2048, 36, 16, 2, 11008),
         'Unsupported native actor architecture')
    shapes = qwen_shapes(saved)
    index = seals.json(directory / 'model.safetensors.index.json')['weight_map']
    need(len(shapes) == 435 and set(index) == set(shapes), 'Require all435 canonical native actor tensors')
    seen, elements = set(), 0
    for name in sorted(set(index.values())):
        need(Path(name).name == name and name in actor['hf_files'], 'HF index shard escaped its producer file set')
        path = directory / name
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for key in handle.keys():
                need(key in shapes and key not in seen and index.get(key) == name,
                     'Duplicate, omitted or misplaced canonical HF tensor')
                value = handle.get_tensor(key)
                tensor_cpu(value, dtype=torch.float32, shape=shapes[key], finite=True)
                seen.add(key); elements += value.numel(); del value
        seals.file(path)
    need(seen == set(shapes), 'Canonical HF finite read did not cover every saved tensor')
    return dict(canonical_tensor_count=len(seen), fp32_elements=elements,
                all_finite=True, cpu_safetensors_mmap=True, forward_performed=False)


def _rank_state(seals, actor, config, outer, rank, rollout_sha):
    need(len(actor['rank_states']) == WORLD and len(actor['rank_receipts']) == WORLD,
         'Native actor complete state requires exactly four rank files/receipts')
    path = seals.path(actor['rank_states'][rank]['path'])
    need(path == Path(actor['state_path']) / ('rank-%d.pt' % rank), 'Actor rank state is outside its saved attempt')
    saved, file_proof = seals.mmap_torch(path, actor['rank_states'][rank])
    need(saved['run_id'] == config['run_id'] and saved['completed_outer'] == outer and saved['rank'] == rank
         and saved['binding'] == actor['binding'] == _state_binding(config)
         and saved['actual_adam_steps'] == 20 * outer and saved['scheduler_completed_outer'] == outer
         and saved['model_master_dtype'] == 'torch.float32'
         and saved['optimizer_layout'] == 'FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4'
         and saved['original_rollout_sha256'] == rollout_sha,
         'Actor rank state differs from its actual4-rank Adam/scheduler/rollout boundary')
    layout, groups = saved['optimizer_parameter_layout'], saved['optimizer']['param_groups']
    need(len(layout) == len(groups), 'Actor optimizer group layout is incomplete')
    shapes = []
    for group, records in zip(groups, layout):
        need(len(group['params']) == len(records), 'Native FSDP flat-parameter order differs')
        for row in records:
            need(row['dtype'] == 'torch.float32' and isinstance(row['shape'], list)
                 and all(type(n) is int and n >= 0 for n in row['shape'])
                 and math.prod(row['shape']) == row['numel'], 'Native FP32 FSDP parameter shape is invalid')
            shapes.append(row['shape'])
    adam = validate_adam(saved['optimizer'], shapes, 20 * outer, learning_rate(outer), weight_decay=.01)
    scheduler = saved['scheduler']
    need(scheduler['last_epoch'] == outer and scheduler['_step_count'] == outer + 1
         and scheduler['base_lrs'] == [1e-6] and scheduler['_last_lr'] == [learning_rate(outer)],
         'Native actor scheduler does not retain completed outer10 and next warmup LR')
    rng = validate_rng(saved['rng'])
    receipt = actor['rank_receipts'][rank]
    ledger = seals.file(receipt['ledger_path'])
    need(receipt['rank'] == rank and receipt['outer'] == outer and receipt['new_adam_steps'] == 20
         and receipt['actual_adam_steps'] == 20 * outer and receipt['opportunities'] == 20
         and ledger['sha256'] == receipt['ledger_sha256'], 'Actor complete receipt differs from its actual20-write ledger')
    with Path(receipt['ledger_path']).open() as handle:
        writes = [row for row in map(json.loads, handle) if row['event'] == 'ACTUAL_ADAM_WRITE']
    need(len(writes) == 20 and all(row['rank'] == rank and row['outer'] == outer
         and row['opportunity'] == i and row['writes_this_attempt'] == i + 1
         and row['actual_adam_steps'] == 20 * (outer - 1) + i + 1 for i, row in enumerate(writes)),
         'Actor native ledger does not prove the actual20 returned writes per rank')
    del saved; gc.collect()
    return dict(rank=rank, actual_adam_steps=20 * outer, actual_fp32_adam_states=adam,
                scheduler_completed_outer=outer, rng=rng, file=file_proof, ledger=ledger)


def _probe0(seals, probe, config):
    import torch
    path = seals.base / 'probe/latest.pt'
    saved, file_proof = seals.mmap_torch(path, probe)
    need(saved['run_id'] == config['run_id'] and saved['completed_outer'] == 0
         and saved['actual_adam_steps_per_rank'] == [0] * WORLD and saved['optimizer'] is None
         and saved['rank_rng'] is None and saved['learning_rate'] == 5e-6,
         'Only the untouched first-update probe0 with empty optimizer/RNG is resumable here')
    binding = saved['binding']
    need(saved['binding_sha256'] == digest(binding) == probe['binding_sha256'], 'Native initial probe binding SHA differs')
    required = dict(role='prefix_selection200_actorhidden_period10_question5_correctness_probe',
        run_id=config['run_id'], representation_source='current_sampling_actor_native_layer27_postblock',
        independent_encoder_forward=False, probe_update_period=10, updates_per_period=50,
        encoder_revision=REVISION, question_max_tokens=273, head_max_tokens=278, prefix_tokens=5,
        feature_layer=27, feature_boundary='post_block', source_probe_sha256=config['probe_source_sha256'],
        source_epoch=config['probe_source_epoch'], source_model_kind='ema',
        source_warmup_run=config['probe_source_run'], warmup_model_weights_only=True,
        warmup_optimizer_imported=False, online_initial_adam_steps=0, learning_rate=5e-6,
        group_weight=1., loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE', updates_per_outer=0,
        global_batch_average=256, actual_global_batches=[255, 260], world_size=4, accumulation=1,
        rank_questions_per_period=640, rank_trajectories_per_period=3200,
        selected_questions=512, candidate_questions=2560, exploration=0,
        diagnostic_manifest_sha256=config['fixed_test_manifest_sha256'], final_outer_probe_update=False,
        probe_total_actual_adam_per_rank=950)
    need(all(binding.get(key) == value for key, value in required.items()), 'Native pending probe scientific recipe differs')
    model = saved['model']
    need(set(model) == set(PARAMETER_SHAPES) and len(model) == 14, 'Missing original native block35 or flattened binary head')
    for name, shape in PARAMETER_SHAPES.items():
        tensor_cpu(model[name], dtype=torch.float32, shape=shape, finite=True)
    buffers = saved['model_nonpersistent_buffers']
    layout = next((row for row in NATIVE_BUFFER_LAYOUTS if set(row) == set(buffers)), None)
    need(layout is not None and len(buffers) == 2, 'Retain both actual Transformers4.46.3 native RoPE buffers')
    for name, shape in layout.items():
        tensor_cpu(buffers[name], dtype=torch.float32, shape=shape, finite=True)
    imported = saved['warmup_model_import']
    need(imported['source_epoch'] == config['probe_source_epoch'] == 5
         and imported['source_model_kind'] == 'ema' and imported['source_warmup_run'] == config['probe_source_run']
         and imported['warmup_optimizer_imported'] is False and imported['warmup_rng_imported'] is False
         and imported['warmup_model_weights_only'] is True and imported['learned_tensor_count'] == 14,
         'Probe initialization is not the approved complete warmup5 EMA model-only import')
    alias = seals.file(seals.base / 'probe/head-outer-0000.pt', probe)
    need((alias['dev'], alias['inode']) == (file_proof['dev'], file_proof['inode']), 'Initial probe alias does not bind the same untouched inode')
    initialization = seals.json(seals.base / 'probe/initialization-complete.json')
    need(initialization['run_id'] == config['run_id'] and initialization['source_sha256'] == config['probe_source_sha256']
         and initialization['source_epoch'] == 5 and initialization['source_model_kind'] == 'ema'
         and initialization['receipt'] == probe and initialization['warmup_updates_reexecuted'] == 0,
         'Actual immutable probe initialization receipt changed')
    del saved; gc.collect()
    return dict(file=file_proof, fp32_learned_tensors=14, fp32_native_rope_buffers=2,
        all_finite=True, actual_adam_steps_per_rank=[0] * WORLD, optimizer_empty=True,
        rank_rng_absent_before_first_optimizer=True, source_probe_sha256=config['probe_source_sha256'])


def _pool_and_data(seals, config, joint, actor, probe):
    from .pool import reserve_indices
    from .periodic_data import select_actual_mixed_groups
    root = seals.base / 'outers/outer-000010'
    paths = dict(pool_binding=seals.base / 'question-pool/binding.json',
        pool_state=seals.base / 'question-pool/state.json', reservation=root / 'pool-reservation.json',
        source_pool=Path(config['source_question_pool']), candidates=root / 'selection.json',
        questions=root / 'questions.json', prefix=root / 'prefix/complete.json',
        score=root / 'probe-score/manifest.json', score_index=root / 'probe-score/index.json',
        predicted=root / 'probe-score/selection.json', pool_selection=root / 'probe-score/pool-selection.json',
        actual_selection=root / 'probe-score/actual-em-selection.json',
        full_pool=root / 'full-pool/complete.json', full=root / 'full/complete.json',
        diagnostic=root / 'diagnostic-complete.json', actor_binding=root / 'actor-binding.json')
    data = {key: seals.json(path) for key, path in paths.items()}
    for name in ('prefix', 'probe-score', 'full-pool', 'full', 'diagnostic'):
        seals.tree(root / name)
    # Actual actor attempt configuration is a required resume input; retain all
    # small top-level phase configurations without loading any prompt into proof.
    for path in root.iterdir():
        if path.is_file():
            seals.file(path)
    for key in ('prefix', 'score', 'full_pool', 'full', 'diagnostic'):
        seals.verify_producer_references(data[key])
    pool, binding, reservation, source = (data[k] for k in ('pool_state', 'pool_binding', 'reservation', 'source_pool'))
    size = binding['eligible_questions']
    need(size == 168510 == source['eligible_unique_questions'] == len(source['records'])
         and source['canonical_rows'] == 169028 and source['excluded_diagnostic_questions'] == 512
         and source['canonical_train_sha256'] == config['canonical_train_sha256']
         and binding['source'] == str(paths['source_pool'])
         and binding['sha256'] == seals.file(paths['source_pool'])['sha256'], 'Canonical candidate source pool identity differs')
    need(pool['candidate_questions'] == 2560 and pool['next_outer'] == 11 and pool['consumed_candidates'] == 25600
         and pool['cycle'] * size + pool['cursor'] == 25600 and pool['seed'] == int(config.get('seed', 1729))
         and len(pool['order']) == size and sorted(pool['order']) == list(range(size)),
         'The pending actor10 candidate reservation/order/cursor is not complete')
    previous = seals.json(joint['pool_snapshot'])
    need(seals.file(joint['pool_snapshot'])['sha256'] == joint['pool_sha256']
         and previous['next_outer'] == 10 and previous['consumed_candidates'] == 23040,
         'Last committed joint9 lost its original candidate cursor')
    ids, cycles, after = reserve_indices(previous, 10)
    need(reservation['outer'] == 10 and reservation['pool_sha256'] == binding['sha256']
         and reservation['before_sha256'] == digest(previous) and reservation['after_sha256'] == digest(pool)
         and reservation['indices'] == ids and reservation['cycles'] == cycles
         and reservation['after'] == pool == after, 'Pending reservation10 does not continue the exact saved joint9 order')
    candidate, questions = data['candidates'], data['questions']
    need(candidate['questions'] == questions['questions'] == 2560
         and questions['selection_sha256'] == seals.file(paths['candidates'])['sha256']
         and len(candidate['rows']) == len(questions['rows']) == 2560, 'Incomplete pending candidate question bundle')
    for i, row in enumerate(candidate['rows']):
        original = source['records'][ids[i]]
        need(row['candidate_ordinal'] == i and row['pool_index'] == ids[i] and row['pool_cycle'] == cycles[i]
             and all(row.get(key) == value for key, value in original.items())
             and all(questions['rows'][i].get(key) == value for key, value in row.items()),
             'Pending candidate identities differ from their exact canonical reservation')
    prefix, score, full_pool, full = (data[k] for k in ('prefix', 'score', 'full_pool', 'full'))
    actor_version = actor['sampling_actor_version']
    need(prefix['run_id'] == config['run_id'] and prefix['outer'] == 10
         and prefix['questions'] == 2560 and prefix['trajectories'] == 12800
         and prefix['actor_version'] == actor_version
         and score['run_id'] == config['run_id'] and score['outer'] == 10
         and score['actor_version'] == actor_version
         and score['prefix_scan_sha256'] == prefix['scan_sha256']
         and score['probe_before_checkpoint_sha256'] == probe['sha256']
         and score['index_sha256'] == seals.file(paths['score_index'])['sha256']
         and score['selection_sha256'] == seals.file(paths['predicted'])['sha256'],
         'Pending scored native hidden/prefix/sampling actor/probe provenance differs')
    scan_path = Path(score['native_hidden_scan'])
    need(seals.base in scan_path.parents, 'Pending train native hidden escaped the source run')
    scan = seals.json(scan_path / 'scan-complete.json')
    need(seals.file(scan_path / 'scan-complete.json')['sha256'] == prefix['scan_sha256']
         and scan['count'] == 12800 and scan['actor_version'] == actor_version
         and scan['hidden_feature_actor_version'] == actor_version and scan['head_max_tokens'] == 278
         and scan['actor_hidden_native'] is True and scan['independent_encoder_forward'] is False
         and len(scan['actor_hidden_shards']) == 4, 'Pending train native hidden dimensions/version differ')
    seals.verify_producer_references(scan)
    index = data['score_index']
    need(len(index) == 12800 and len({row['trajectory_id'] for row in index}) == 12800,
         'Native score identity index is not unique complete2560G5')
    own = seals.path(full_pool['own_em']['path']); labels = [json.loads(line) for line in own.open()]
    need(full_pool['questions'] == 2560 and full_pool['trajectories'] == 12800
         and full_pool['actor_version'] == actor_version and full_pool['prefix_scan_sha256'] == prefix['scan_sha256']
         and full_pool['selection_sha256'] == seals.file(paths['pool_selection'])['sha256'],
         'Pending full-pool G5 outcomes changed their actual same-prefix generation event')
    actual = data['actual_selection']
    selected, selection_stats = select_actual_mixed_groups(data['pool_selection']['selected'], labels,
        fallback_mode=config['actual_em_selection_fallback'],
        provenance=dict(actor_version=actor_version, prefix_scan_sha256=prefix['scan_sha256'],
            pool_selection_sha256=seals.file(paths['pool_selection'])['sha256']),
        expected_prefixes={row['trajectory_id']: row['prefix_sha256'] for row in index})
    need(actual['selected'] == selected and all(actual.get(key) == value for key, value in selection_stats.items())
         and actual['outcome_labels_used'] is True and actual['full_pool_own_em_sha256'] == full_pool['own_em']['sha256']
         and actual['full_pool_complete_sha256'] == seals.file(paths['full_pool'])['sha256'],
         'Actual-EM mixed-first actor selection is not its recorded deterministic512 result')
    need(full['questions'] == 512 and full['trajectories'] == 2560 and full['projected_from_full_pool'] is True
         and full['projected_generation_requests'] == 0
         and full['selection_sha256'] == seals.file(paths['actual_selection'])['sha256']
         and full['actor_version'] == actor_version
         and full['full_pool_complete_sha256'] == seals.file(paths['full_pool'])['sha256']
         and actor['original_rollout_path'] == str(paths['full'])
         and actor['original_rollout_sha256'] == seals.file(paths['full'])['sha256'],
         'Actor10 is not bound to the exact preserved real-EM Top512 rollout')
    diagnostic = data['diagnostic']
    test_path = seals.path(diagnostic['scan_path'])
    test = seals.json(test_path / 'scan-complete.json')
    need(diagnostic['run_id'] == config['run_id'] and diagnostic['outer'] == 10
         and diagnostic['actor_version'] == actor_version and diagnostic['questions'] == 512
         and diagnostic['trajectories'] == 2560 and diagnostic['training_questions_used'] is False
         and diagnostic['independent_encoder_forward'] is False
         and diagnostic['scan_sha256'] == seals.file(test_path / 'scan-complete.json')['sha256']
         and test['actor_version'] == actor_version and test['count'] == 2560,
         'Existing fixed512 diagnostic is incomplete or uses another sampling actor')
    seals.verify_producer_references(test)
    need(not (root / 'complete.json').exists(), 'Pending outer10 cannot already be a completed joint boundary')
    return dict(pool_next_outer=pool['next_outer'], pool_consumed_candidates=pool['consumed_candidates'],
        pool_cursor=pool['cursor'], pool_cycle=pool['cycle'], eligible_questions=size,
        actual_mixed_candidates=selection_stats['actual_mixed_candidate_count'],
        chosen_mixed_count=selection_stats['chosen_mixed_count'], fallback_count=selection_stats['fallback_count'],
        train_own_trajectories=len(labels), fixed_test_trajectories=test['count'],
        pending_actor_rollout_sha256=seals.file(paths['full'])['sha256']), pool


def _qualify(args):
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run the qualification with CUDA_VISIBLE_DEVICES=')
    torch.set_num_threads(1)
    base = _canonical_path(args.base)
    need(base.parent == Path('/data/shaoshitong') and base.name.startswith('search-g1-prefix-selection200-'),
         'Qualify only this canonical isolated A800 native experiment')
    config_path = base / 'preparation/launch-config.json'
    config = read(config_path)
    need(config['base'] == str(base) and config['run_id'] == args.expected_run
         and config['execution_id'] == args.expected_execution and config['source_commit'] == args.expected_source
         and hash_file(config_path)['sha256'] == args.expected_config_sha256
         and config['world_size'] == WORLD == 4 and config['model_revision'] == REVISION
         and config['actor_hidden_native'] is True and config['candidate_questions'] == CANDIDATE_QUESTIONS == 2560
         and config['selected_questions'] == 512 and config['responses_per_question'] == G == 5
         and config['prefix_policy_tokens'] == PREFIX_TOKENS == 5 and QMAX == 273 and WIDTH == 278 and DIM == 2048
         and config['probe_update_period'] == 10 and config['probe_updates_per_period'] == 50
         and config['probe_learning_rate'] == 5e-6 and config['probe_update_at_final'] is False
         and config['nonmixed_advantage_mode'] == 'zero_anchor_all_correct_only'
         and config['actual_em_selection_fallback'] == 'probe_score_fill',
         'Source run/config/source or period10/question5 science differs')
    allowed = [args.terminal_proof, str(Path(config['official_model_dir']) / 'config.json'), config['source_question_pool']]
    for key in ('fixed_test_identity_path', 'probe_source_cpu_proof'):
        if config.get(key): allowed.append(config[key])
    seals = Seals(base, allowed)
    seals.file(config_path, dict(bytes=config_path.stat().st_size, sha256=args.expected_config_sha256))
    terminal_file, terminal = _terminal(seals, args.terminal_proof, args.terminal_proof_sha256, config)
    failed_output = base / 'outers/outer-000010/probe-update'
    need(not failed_output.exists(),
         'Probe output exists: first-update zero-write recovery needs a different evidence audit')
    joint = seals.json(base / 'latest-joint.json')
    need(joint['run_id'] == config['run_id'] and joint['outer'] == 9
         and joint['source_commit'] == config['source_commit']
         and seals.json(base / 'outers/outer-000009/complete.json') == joint,
         'Only last committed joint9 with native pending actor10 is supported')
    for outer in range(1, 10):
        root = base / 'outers' / ('outer-%06d' % outer)
        previous = seals.json(root / 'complete.json')
        need(previous['run_id'] == config['run_id'] and previous['outer'] == outer
             and previous['source_commit'] == config['source_commit']
             and previous['actor']['completed_outer'] == outer
             and previous['actor']['actual_adam_steps_per_rank'] == [20 * outer] * 4
             and previous['actor']['scheduler_completed_outer'] == outer
             and previous['probe']['completed_outer'] == 0
             and previous['probe']['actual_adam_steps_per_rank'] == [0] * 4,
             'Committed prefix1..9 has a gap or mixed actor/probe counters')
        for name in ('probe-frozen.json',):
            if (root / name).exists(): seals.file(root / name)
    previous_qa = base / 'outers/outer-000005/validation/complete.json'
    if previous_qa.is_file():
        seals.file(previous_qa)
    actor, probe = seals.json(base / 'actor/latest.json'), seals.json(base / 'probe/latest.json')
    need(actor['run_id'] == probe['run_id'] == config['run_id'] and actor['source_commit'] == config['source_commit']
         and actor['completed_outer'] == 10 and actor['actual_adam_steps_per_rank'] == [200] * 4
         and actor['new_adam_steps_per_rank'] == [20] * 4 and actor['scheduler_completed_outer'] == 10
         and actor['next_learning_rate'] == learning_rate(10)
         and probe['completed_outer'] == 0 and probe['actual_adam_steps_per_rank'] == [0] * 4,
         'Pending actor/probe actual counters differ from actor10/probe0')
    need(joint['probe'] == probe and actor['sampling_actor_version'] == joint['actor']['actor_version'],
         'Pending actor10 changed the committed joint9 sampling actor/probe0 version')
    snapshot = seals.file(joint['probe_snapshot'], probe)
    latest_probe_file = seals.file(base / 'probe/latest.pt', probe)
    need((snapshot['dev'], snapshot['inode']) == (latest_probe_file['dev'], latest_probe_file['inode']),
         'Joint9 probe snapshot is not the same retained untouched initial model')
    need(seals.json(Path(actor['state_path']) / 'complete.json') == actor,
         'Actor latest metadata differs from its complete original attempt')
    pool_summary, pool = _pool_and_data(seals, config, joint, actor, probe)
    hf = _hf(seals, actor, config)
    ranks = [_rank_state(seals, actor, config, 10, rank, actor['original_rollout_sha256']) for rank in range(4)]
    driver, driver_file = seals.mmap_torch(actor['driver_state']['path'], actor['driver_state'])
    need(driver['run_id'] == config['run_id'] and driver['completed_outer'] == 10
         and driver['actor_version'] == actor['actor_version'] and driver['candidate_pool_state']['value'] == pool
         and driver['candidate_pool_state']['sha256'] == hashlib.sha256(canonical(pool) + b'\n').hexdigest()
         and driver['prefix_probe_receipt']['value'] == probe
         and driver['prefix_probe_receipt']['sha256'] == seals.file(base / 'probe/latest.json')['sha256'],
         'Actual driver did not save the reserved candidate10 cursor and selection-time probe0')
    driver_rng = validate_rng(driver['rng']); del driver; gc.collect()
    probe_summary = _probe0(seals, probe, config)
    fixed_identity = config.get('fixed_test_identity_path')
    if fixed_identity:
        need(seals.file(fixed_identity)['sha256'] == config['fixed_test_identity_sha256'],
             'Pinned fixed512 identity manifest changed')
    # The producer/terminal journals are small but part of the immutable recovery
    # evidence. Their absolute paths are not promoted into the new execution.
    for name in ('logs/process-ledger.json', 'logs/job-ledger.json', 'logs/jobs.json', 'logs/qa-pending.json',
                 'preparation/submit-intent.json', 'probe/initialization-intent.json'):
        path = base / name
        if path.is_file(): seals.file(path)
    need(not failed_output.exists(), 'Probe output appeared during qualification')
    files = seals.finish()
    return dict(status=STATUS, schema_version=1, run_id=config['run_id'], source_base=str(base),
        source_execution_id=config['execution_id'], source_commit=config['source_commit'],
        source_config_sha256=args.expected_config_sha256, terminal_proof=str(terminal_file['path']),
        terminal_proof_sha256=terminal_file['sha256'], completed_joint_outer=9, pending_outer=10,
        actor_completed_outer=10, actor_adam_per_rank=[200] * 4, scheduler_completed_outer=10,
        probe_completed_outer=0, probe_adam_per_rank=[0] * 4, probe_update_new_adam_per_rank=[0] * 4,
        owned_births_all_gone=True, probe_writes_observed=0,
        failed_probe_output_directory=str(failed_output), failed_probe_output_empty=True,
        pool_next_outer=11, pool_consumed_candidates=25600, pool_cursor=pool['cursor'], pool_cycle=pool['cycle'],
        candidate_mixed_count=pool_summary['actual_mixed_candidates'],
        selected_mixed_count=pool_summary['chosen_mixed_count'], fallback_count=pool_summary['fallback_count'],
        actor_reward_mean=actor['reward']['score_mean'],
        actor_advantage_group_counts=actor['reward']['advantage_summary']['actual_group_counts'],
        source_probe_sha256=config['probe_source_sha256'], files=files, verified_files=files,
        pending_inputs=pool_summary, actor_hf=hf, actor_rank_cpu_states=ranks,
        actor_driver_file=driver_file, actor_driver_rng=driver_rng, initial_probe=probe_summary,
        terminal=terminal, independent_full_file_sha_performed=True, actual_cpu_deserialization_performed=True,
        actual_cuda_reload_accepted=False, gpu_forward_performed=False, cuda_rng_applied=False,
        training_modified=False, sampler_advanced=False, optimizer_reset=False,
        rollout_regenerated=False, new_evaluation_performed=False, independent_backup_performed=False,
        large_payload_transferred_to_Mac=False)


def qualify(args):
    output = Path(args.output)
    need(output.is_absolute() and '..' not in output.parts and output.parent.is_dir(),
         'Create the new remote recovery preparation directory before CPU qualification')
    _canonical_path(output.parent)
    need(str(output).startswith('/data/shaoshitong/search-g1-prefix-selection200-')
         and output.parent.name == 'preparation' and not output.is_symlink(),
         'Qualification proof must stay in a canonical remote experiment preparation directory')
    intent = output.with_name(output.stem + '-intent.json')
    lock_path = output.with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            previous = read(output)
            need(previous.get('run_id') == args.expected_run
                 and previous.get('source_execution_id') == args.expected_execution
                 and previous.get('source_config_sha256') == args.expected_config_sha256
                 and previous.get('terminal_proof_sha256') == args.terminal_proof_sha256,
                 'Existing qualification belongs to a different recovery request')
            return dict(status='EXISTING_CPU_PENDING_PROBE_PROOF_NO_REPLAY', proof_status=previous['status'],
                        proof=hash_file(output), independent_large_file_read_this_call=False)
        need(not intent.exists(), 'Existing incomplete CPU qualification intent requires explicit reconciliation')
        with intent.open('x') as handle:
            json.dump(dict(run_id=args.expected_run, source_execution_id=args.expected_execution,
                source_base=args.base, source_config_sha256=args.expected_config_sha256,
                terminal_proof_sha256=args.terminal_proof_sha256, started_unix=time.time()), handle, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        started = time.monotonic()
        try:
            proof = _qualify(args)
        except BaseException as error:
            proof = dict(status='NATIVE_PENDING_PROBE_CPU_QUALIFICATION_FAILED', run_id=args.expected_run,
                source_execution_id=args.expected_execution, source_config_sha256=args.expected_config_sha256,
                terminal_proof_sha256=args.terminal_proof_sha256, error_type=type(error).__name__, error=str(error),
                actual_cuda_reload_accepted=False, gpu_forward_performed=False, training_modified=False)
        proof.update(reader_source_sha256=hash_file(__file__)['sha256'], elapsed_seconds=time.monotonic() - started,
                     checked_unix=time.time())
        with output.open('x') as handle:
            json.dump(proof, handle, sort_keys=True, allow_nan=False); handle.flush(); os.fsync(handle.fileno())
        return dict(status=proof['status'], proof=hash_file(output),
            actor_completed_outer=proof.get('actor_completed_outer'), actor_adam_per_rank=proof.get('actor_adam_per_rank'),
            probe_completed_outer=proof.get('probe_completed_outer'), probe_adam_per_rank=proof.get('probe_adam_per_rank'),
            error=proof.get('error'), actual_cuda_reload_accepted=False, gpu_forward_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'expected-run', 'expected-execution', 'expected-source', 'expected-config-sha256',
                 'terminal-proof', 'terminal-proof-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(qualify(parser.parse_args()), sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
