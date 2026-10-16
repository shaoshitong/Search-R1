"""Once-only remote CPU qualification of joint27 and its completed prefix28.

This reads complete producer files and saved state, never a model forward or
CUDA API.  Joint27 is the restoration boundary; reservation28 and its original
native prefix/score are retained, while failed uncommitted continuation bytes
are not promoted to accepted trajectories.  CPU qualification is not CUDA
reload acceptance, a completed outer28, or an independent backup.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import time

from .actor import REVISION, _state_binding, learning_rate
from .checkpoint_cpu_once import digest, hash_file, need, read, tensor_cpu, validate_adam, validate_rng
from .checkpoint_native_pending_probe import Seals, _canonical_path, _hf, _rank_state, _unwrap_terminal
from .contract import CANDIDATE_QUESTIONS, G, PREFIX_TOKENS, QMAX, WIDTH, WORLD
from .periodic import probe_boundary
from .pool import reserve_indices, state_sha
from .prefix5_model import NATIVE_BUFFER_LAYOUTS, PARAMETER_SHAPES

STATUS = 'NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'
ADAPTER = 'native_a800_joint_v1'
OUTER = 27
PENDING = 28


def validate_terminal(value, config):
    """Typed actual continuation fault and owned-terminal proof, no PID guesses."""
    ended = _unwrap_terminal(value)
    need(ended.get('run_id') == config['run_id'] and ended.get('execution_id') == config['execution_id']
         and ended.get('source_commit') == config['source_commit'] and ended.get('base') == config['base'],
         'Actual terminal proof belongs to another source execution')
    need(ended.get('pipeline_alive') is False and ended.get('actual_births_present') == []
         and ended.get('remaining_bound_processes') == []
         and type(ended.get('recorded_owned_births_checked')) is int
         and ended['recorded_owned_births_checked'] > 0,
         'Recorded pipeline and every owned birth must actually be gone')
    cleanup = ended.get('cleanup', {}).get('value', {})
    need(cleanup.get('run_id') == config['run_id'] and cleanup.get('cleaned') is True
         and cleanup.get('owned_births_all_gone') is True and cleanup.get('remaining_owned_births') == [],
         'Actual owned-resource cleanup was not accepted')
    jobs = ended.get('failed_or_live_jobs')
    need(isinstance(jobs, list) and len(jobs) == 1
         and jobs[0].get('name') == 'outer-000028-continue' and jobs[0].get('state') == 'STOPPED'
         and jobs[0].get('exit_code') == 1 and jobs[0].get('live_births') == [],
         'Only the actual stopped continuation28 fault may use this adapter')
    need(ended.get('full28_complete_exists') is False and ended.get('actor28_attempt_dirs') == [],
         'No complete continuation28 or actor28 attempt may be discarded')
    logs = ended.get('logs', [])
    need(any('/outers/outer-000028/full/rank-0.log' in row.get('path', '')
             and any('CUDA error: an illegal memory access was encountered' in line
                     for line in row.get('safe_error_lines', [])) for row in logs),
         'Terminal evidence does not contain the actual rank0 CUDA illegal-memory fault')
    return ended


def _terminal(seals, args, config):
    file_proof = seals.file(args.terminal_proof)
    need(file_proof['sha256'] == args.terminal_proof_sha256, 'Selected actual terminal proof changed')
    ended = validate_terminal(seals.json(args.terminal_proof), config)
    cleanup_file = seals.file(ended['cleanup']['path'])
    need(cleanup_file['sha256'] == ended['cleanup']['sha256']
         and seals.json(ended['cleanup']['path']) == ended['cleanup']['value'],
         'Actual cleanup file differs from the exact terminal proof')
    for row in ended['logs']:
        seals.file(row['path'], row)
    if ended.get('pipeline_status'):
        record = ended['pipeline_status']; actual = seals.file(record['path'])
        need(actual['sha256'] == record['sha256'] and seals.json(record['path']) == record['value'],
             'Actual terminal pipeline stage changed')
    return ended, file_proof


def _recipe(config, args):
    required = dict(base=args.base, run_id=args.expected_run, execution_id=args.expected_execution,
        source_commit=args.expected_source, world_size=4, model_revision=REVISION,
        actor_hidden_native=True, candidate_questions=2560, selected_questions=512,
        responses_per_question=5, prefix_policy_tokens=5, exploration_questions=0,
        probe_update_period=10, probe_updates_per_period=50, probe_learning_rate=5e-6,
        probe_epochs_per_update=1, probe_update_at_final=False, probe_global_batch_average=256,
        actor_global_mini_batch=128, actor_ppo_epochs=1,
        nonmixed_advantage_mode='zero_anchor_all_correct_only', actual_em_selection_fallback='probe_score_fill',
        independent_encoder_forward=False)
    need(all(config.get(key) == value for key, value in required.items()), 'A800 native question5/period10 recipe differs')
    need((WORLD, QMAX, PREFIX_TOKENS, WIDTH, G, CANDIDATE_QUESTIONS) == (4, 273, 5, 278, 5, 2560),
         'Qualified scientific dimensions differ from the actual A800 implementation')


def _history(seals, config):
    parent = config.get('resume_pending_probe')
    need(isinstance(parent, dict) and parent.get('completed_joint_outer') == 9
         and parent.get('pending_outer') == 10, 'This source must retain its explicit initial pending-probe recovery chain')
    imported = seals.json(seals.base / 'preparation/resume-pending-probe-receipt.json')
    need(imported['run_id'] == config['run_id'] and imported['execution_id'] == config['execution_id']
         and imported['source_commit'] == config['source_commit']
         and imported['parent_source_commit'] == parent['source_commit']
         and imported['qualification_proof_sha256'] == parent['qualification_proof_sha256'],
         'Original imported-prefix source/qualification binding changed')
    inherited = {row['target']: row for row in imported['inherited_metadata_files']}
    history = {}
    for n in range(1, OUTER + 1):
        root = seals.base / 'outers' / ('outer-%06d' % n)
        path = root / 'complete.json'; complete = seals.json(path); boundary = probe_boundary(n)
        expected_source = parent['source_commit'] if n <= 9 else config['source_commit']
        need(complete['run_id'] == config['run_id'] and complete['outer'] == n
             and complete['source_commit'] == expected_source
             and complete['actor']['completed_outer'] == n
             and complete['actor']['actual_adam_steps_per_rank'] == [20*n] * WORLD
             and complete['actor']['scheduler_completed_outer'] == n
             and complete['probe']['completed_outer'] == boundary['last_update_outer']
             and complete['probe']['actual_adam_steps_per_rank'] == [boundary['actual_adam_per_rank']] * WORLD,
             'Complete joint prefix1..27 changed its source or actual actor/probe clocks')
        if n <= 9:
            origin = inherited.get(str(path), {})
            need(origin.get('run_id') == config['run_id'] and origin.get('outer') == n
                 and origin.get('source_commit') == expected_source
                 and origin.get('sha256') == seals.file(path)['sha256'],
                 'Inherited joint metadata differs from its original sealed recovery chain')
        for name in ('probe-frozen.json', 'probe-update/metrics.json', 'validation/complete.json'):
            extra = root / name
            if extra.is_file():
                seals.file(extra)
                if n <= 9 and str(extra) in inherited:
                    need(seals.file(extra)['sha256'] == inherited[str(extra)]['sha256'],
                         'Inherited existing QA metadata changed')
        history[n] = complete
    return history


def _pool_metadata(seals, config, joint):
    from .pool import fixed_test_identities
    source_path = Path(config['source_question_pool'])
    source = seals.json(source_path)
    binding = seals.json(seals.base / 'question-pool/binding.json')
    pool = seals.json(joint['pool_snapshot']); current = seals.json(seals.base / 'question-pool/state.json')
    root = seals.base / 'outers/outer-000028'
    reservation = seals.json(root / 'pool-reservation.json')
    need(seals.file(joint['pool_snapshot'])['sha256'] == joint['pool_sha256'], 'Joint27 pool snapshot file SHA changed')
    size = binding['eligible_questions']
    need(size == source['eligible_unique_questions'] == len(source['records']) == 168510
         and source['canonical_rows'] == 169028 and source['excluded_diagnostic_questions'] == 512
         and source['canonical_train_sha256'] == config['canonical_train_sha256']
         and binding['source'] == str(source_path)
         and binding['sha256'] == seals.file(source_path)['sha256'] == config['source_question_pool_sha256']
         and binding['candidate_questions'] == 2560 and binding['responses_per_question'] == 5,
         'Canonical question pool/diagnostic exclusion/source SHA differs')
    for value, n in ((pool, 27), (current, 28)):
        need(value['seed'] == int(config.get('seed', 1729)) and value['candidate_questions'] == 2560
             and value['next_outer'] == n+1 and value['consumed_candidates'] == 2560*n
             and value['cycle'] * size + value['cursor'] == 2560*n
             and len(value['order']) == size and sorted(value['order']) == list(range(size)),
             'Saved candidate permutation/cycle/cursor does not match its actual reserved outer')
    ids, cycles, after = reserve_indices(pool, 28)
    need(reservation['outer'] == 28 and reservation['pool_sha256'] == binding['sha256']
         and reservation['before_sha256'] == state_sha(pool)
         and reservation['after_sha256'] == state_sha(current)
         and reservation['after'] == current == after and reservation['indices'] == ids
         and reservation['cycles'] == cycles, 'Reservation28 must continue the exact saved joint27 candidate RNG/order')
    candidates = seals.json(root / 'selection.json'); questions = seals.json(root / 'questions.json')
    need(candidates['run_id'] == questions['run_id'] == config['run_id'] and candidates['outer'] == 28
         and candidates['questions'] == questions['questions'] == 2560
         and questions['selection_sha256'] == seals.file(root / 'selection.json')['sha256']
         and len(candidates['rows']) == len(questions['rows']) == 2560,
         'Pending28 authentic candidate question bundle is incomplete')
    for i, record in enumerate(candidates['rows']):
        original = source['records'][ids[i]]
        need(record['candidate_ordinal'] == i and record['pool_index'] == ids[i] and record['pool_cycle'] == cycles[i]
             and all(record.get(key) == value for key, value in original.items())
             and all(questions['rows'][i].get(key) == value for key, value in record.items()),
             'Pending28 question identity changed after its actual pool reservation')
    if config.get('fixed_test_identity_path'):
        need(seals.file(config['fixed_test_identity_path'])['sha256'] == config['fixed_test_identity_sha256'],
             'Original fixed512 identity file changed')
        fixed = fixed_test_identities(config)
        fixed_ids = {row['qid'] for row in fixed}; fixed_text = {row['normalized_question_hash'] for row in fixed}
        need(not any(row['qid'] in fixed_ids or row['normalized_question_hash'] in fixed_text
                     for row in candidates['rows']), 'Pending28 candidate pool overlaps fixed512 diagnostic questions')
    return pool, current, reservation


def _prefix_metadata(seals, config, joint, actor, probe):
    root = seals.base / 'outers/outer-000028'
    prefix = seals.json(root / 'prefix/complete.json')
    score = seals.json(root / 'probe-score/manifest.json')
    selection = seals.json(root / 'probe-score/selection.json')
    binding = seals.json(root / 'actor-binding.json')
    need(binding['run_id'] == config['run_id'] and binding['outer'] == 28
         and binding['actor_version'] == actor['actor_version']
         and binding['actor_model_dir'] == actor['hf_model_dir']
         and binding['actor_model_files_sha256'] == {name: row['sha256'] for name, row in actor['hf_files'].items()},
         'Retained prefix28 sampling actor is not the qualified complete actor27')
    need(prefix['run_id'] == config['run_id'] and prefix['outer'] == 28 and prefix['phase'] == 'prefix'
         and prefix['actor_version'] == actor['actor_version'] and prefix['actor_model_dir'] == actor['hf_model_dir']
         and prefix['config_sha256'] == seals.file(seals.base / 'preparation/launch-config.json')['sha256']
         and prefix['questions'] == 2560 and prefix['trajectories'] == 12800
         and prefix['model_revision'] == REVISION and prefix['head_prefix_tokens'] == 5
         and prefix['candidate_questions_sha256'] == seals.file(root / 'questions.json')['sha256']
         and prefix['owned_native_worker_exit_codes'] == {str(rank): 0 for rank in range(4)}
         and len(prefix['ranks']) == 4, 'Prefix28 is not a complete actual four-worker same-actor native phase')
    for rank, receipt in enumerate(prefix['ranks']):
        actual = seals.json(root / 'prefix' / ('rank-%d-complete.json' % rank))
        need(actual == receipt and receipt['run_id'] == config['run_id'] and receipt['outer'] == 28
             and receipt['phase'] == 'prefix' and receipt['rank'] == rank
             and receipt['questions'] == 640 and receipt['trajectories'] == 3200
             and receipt['actor_version'] == actor['actor_version']
             and receipt['config_sha256'] == prefix['config_sha256'],
             'Original prefix rank receipt identity/count/actor/config differs')
    scan_path = seals.path(prefix['scan_path']); scan = seals.json(scan_path / 'scan-complete.json')
    need(prefix['scan_sha256'] == seals.file(scan_path / 'scan-complete.json')['sha256']
         and scan['run_id'] == config['run_id'] and scan['outer'] == 28
         and scan['actor_version'] == actor['actor_version'] == scan['hidden_feature_actor_version']
         and scan['count'] == 12800 and scan['actor_hidden_native'] is True
         and scan['independent_encoder_forward'] is False and scan['head_max_tokens'] == 278
         and scan['max_policy_prefix_tokens'] == 5 and len(scan['actor_hidden_shards']) == 4,
         'Native prefix28 scan changed its exact actor version or question5 dimensions')
    need(score['run_id'] == selection['run_id'] == config['run_id'] and score['outer'] == selection['outer'] == 28
         and score['actor_version'] == selection['actor_version'] == actor['actor_version']
         and score['count'] == 12800 and score['selected_count'] == 2560
         and score['question_max_tokens'] == 273 and score['head_max_tokens'] == 278
         and score['prefix_scan_sha256'] == selection['prefix_scan_sha256'] == prefix['scan_sha256']
         and score['probe_before_checkpoint_sha256'] == selection['probe_before_checkpoint_sha256'] == probe['sha256']
         and score['no_outcome_labels'] is True and score['independent_encoder_forward'] is False
         and score['selection_sha256'] == seals.file(root / 'probe-score/selection.json')['sha256']
         and score['index_sha256'] == seals.file(root / 'probe-score/index.json')['sha256']
         and score['candidate_scores_sha256'] == seals.file(root / 'probe-score/candidate-scores.jsonl')['sha256']
         and score['native_hidden_scan'] == str(scan_path),
         'Existing prefix28 scores must bind exactly actor27/probe20 and every original native trajectory')
    need(selection['selected_questions'] == 512 and selection['candidate_questions'] == 2560
         and selection['trajectories_per_question'] == 5 and selection['exploration'] == 0
         and selection['outcome_labels_used'] is False and selection['provisional_for_actor'] is False,
         'Ordinary outer28 selection must remain label-free predicted Top512')
    groups = selection['selected']; qids = set(); tids = set(); ordinals = set()
    need(len(groups) == 512, 'Original predicted selection does not contain512 groups')
    for group in groups:
        need(group['qid'] not in qids and group['candidate_ordinal'] not in ordinals
             and len(group['trajectory_ids']) == 5 and len(set(group['trajectory_ids'])) == 5
             and not (tids & set(group['trajectory_ids'])), 'Selected question/G5 identities repeat')
        qids.add(group['qid']); ordinals.add(group['candidate_ordinal']); tids.update(group['trajectory_ids'])
    need(not (root / 'complete.json').exists() and not (root / 'full/complete.json').exists(),
         'Pending28 cannot already be a complete outer or continuation')
    need(not list((seals.base / 'actor/outers').glob('outer-0028-attempt-*')),
         'An actual actor28 attempt exists and needs a different partial-state audit')
    return root, prefix, score, scan, selection


def _complete_continuation_ranks(seals, config, root, actor, prefix, selection):
    """Qualify committed rank receipts only, never a partial gzip or progress file."""
    from .rollout import iter_rows
    allowed = []
    for rank in range(4):
        path = root / 'full' / ('rank-%d-complete.json' % rank)
        if not path.is_file():
            continue
        receipt = seals.json(path)
        need(receipt['run_id'] == config['run_id'] and receipt['outer'] == 28
             and receipt['phase'] == 'continue' and receipt['rank'] == rank
             and receipt['actor_version'] == actor['actor_version'] and receipt['questions'] == 128
             and receipt['trajectories'] == 640
             and receipt['config_sha256'] == seals.file(seals.base / 'preparation/launch-config.json')['sha256'],
             'Existing continuation rank receipt is not a complete original selected128G5 shard')
        for key in ('full', 'own_em'):
            seals.file(receipt[key]['path'], receipt[key])
        expected = {tid: row['qid'] for row in selection['selected'][rank::4] for tid in row['trajectory_ids']}
        labels = list(iter_rows(receipt['own_em']['path']))
        need(len(labels) == 640 and len({row['trajectory_id'] for row in labels}) == 640
             and all(expected.get(row['trajectory_id']) == row['qid']
                     and type(row['strict_em']) is int and row['strict_em'] in (0, 1)
                     and row['actor_version'] == actor['actor_version']
                     and row['prefix_scan_sha256'] == prefix['scan_sha256']
                     and row['selection_sha256'] == seals.file(root / 'probe-score/selection.json')['sha256']
                     for row in labels), 'Complete continuation rank lost its authentic same-prefix labels/G5 identities')
        allowed.append(dict(rank=rank, receipt_path=str(path), receipt_sha256=seals.file(path)['sha256'],
            config_sha256=receipt['config_sha256'], artifacts={key: receipt[key] for key in ('full', 'own_em')}))
    return allowed


def _probe(seals, config, probe, joint20):
    import torch
    saved, file_proof = seals.mmap_torch(seals.base / 'probe/latest.pt', probe)
    need(saved['run_id'] == config['run_id'] and saved['completed_outer'] == 20
         and saved['actual_adam_steps_per_rank'] == [100] * 4 and saved['learning_rate'] == 5e-6
         and saved['binding_sha256'] == digest(saved['binding']) == probe['binding_sha256'],
         'Actual probe must retain the complete native period20/Adam100 state')
    binding = saved['binding']
    required = dict(role='prefix_selection200_actorhidden_period10_question5_correctness_probe',
        run_id=config['run_id'], representation_source='current_sampling_actor_native_layer27_postblock',
        independent_encoder_forward=False, probe_update_period=10, updates_per_period=50,
        encoder_revision=REVISION, question_max_tokens=273, head_max_tokens=278, prefix_tokens=5,
        feature_layer=27, feature_boundary='post_block', source_probe_sha256=config['probe_source_sha256'],
        source_epoch=5, source_model_kind='ema', source_warmup_run=config['probe_source_run'],
        warmup_model_weights_only=True, warmup_optimizer_imported=False, online_initial_adam_steps=0,
        learning_rate=5e-6, group_weight=1., loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE',
        updates_per_outer=0, global_batch_average=256, actual_global_batches=[255, 260], world_size=4,
        accumulation=1, rank_questions_per_period=640, rank_trajectories_per_period=3200,
        selected_questions=512, candidate_questions=2560, exploration=0,
        diagnostic_manifest_sha256=config['fixed_test_manifest_sha256'], final_outer_probe_update=False,
        probe_total_actual_adam_per_rank=950)
    need(all(binding.get(key) == value for key, value in required.items()), 'Saved probe scientific/input/loss binding differs')
    need(joint20['probe'] == probe and saved['actor_version'] == joint20['rollout']['actor_version']
         and saved['scored_features_manifest_sha256'] == joint20['probe_metrics']['scored_features_manifest_sha256']
         and saved['own_em_sha256'] == joint20['probe_metrics']['own_em_sha256'],
         'Probe20 weights/Adam do not bind the actual period20 sampling actor and own G5 outcomes')
    model = saved['model']; need(set(model) == set(PARAMETER_SHAPES) and len(model) == 14, 'Missing native block35/binary head parameter')
    for name, shape in PARAMETER_SHAPES.items():
        tensor_cpu(model[name], dtype=torch.float32, shape=shape, finite=True)
    buffers = saved['model_nonpersistent_buffers']
    layout = next((row for row in NATIVE_BUFFER_LAYOUTS if set(row) == set(buffers)), None)
    need(layout is not None and len(buffers) == 2, 'Both actual native RoPE buffers must be retained')
    for name, shape in layout.items():
        tensor_cpu(buffers[name], dtype=torch.float32, shape=shape, finite=True)
    adam = validate_adam(saved['optimizer'], [value.shape for value in model.values()], 100, 5e-6, weight_decay=0.)
    need(len(saved['rank_rng']) == 4 and [row['rank'] for row in saved['rank_rng']] == list(range(4)),
         'Probe checkpoint must contain all four saved rank RNG streams')
    rng = []
    for rank, state in enumerate(saved['rank_rng']):
        need(state['actual_adam_steps'] == 100, 'Probe RNG belongs to another true Adam boundary')
        rng.append(dict(rank=rank, actual_adam_steps=100, validation=validate_rng(state)))
    initialization = seals.json(seals.base / 'probe/initialization-complete.json')
    need(initialization['run_id'] == config['run_id'] and initialization['source_epoch'] == 5
         and initialization['source_model_kind'] == 'ema' and initialization['source_sha256'] == config['probe_source_sha256']
         and initialization['warmup_updates_reexecuted'] == 0,
         'Saved online probe changed its explicitly retained warmup5 EMA model-only lineage')
    del saved; gc.collect()
    return dict(file=file_proof, model_fp32_tensors=14, native_rope_fp32_buffers=2, all_finite=True,
        actual_fp32_adam_states=adam, actual_adam_steps_per_rank=[100] * 4, rank_rng=rng,
        learning_rate=5e-6, source_warmup_weights_reloaded=False)


def _qualify(args):
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use CUDA_VISIBLE_DEVICES= for remote CPU-only qualification')
    torch.set_num_threads(1)
    base = _canonical_path(args.base)
    need(base.parent == Path('/data/shaoshitong') and base.name.startswith('search-g1-prefix-selection200-'),
         'Qualify this canonical isolated A800 source only')
    config_path = base / 'preparation/launch-config.json'; config = read(config_path)
    _recipe(config, args)
    allowed = [args.terminal_proof, str(Path(config['official_model_dir']) / 'config.json'), config['source_question_pool']]
    if config.get('fixed_test_identity_path'): allowed.append(config['fixed_test_identity_path'])
    seals = Seals(base, allowed)
    need(seals.file(config_path)['sha256'] == args.expected_config_sha256, 'Source complete launch configuration changed')
    ended, terminal_file = _terminal(seals, args, config)
    history = _history(seals, config); joint = seals.json(base / 'latest-joint.json')
    need(joint == history[27] and joint['source_commit'] == config['source_commit']
         and seals.file(base / 'latest-joint.json')['sha256'] == ended['latest_joint']['sha256'],
         'Latest complete joint27 differs from the actual terminal identity proof')
    actor, probe = seals.json(base / 'actor/latest.json'), seals.json(base / 'probe/latest.json')
    need(joint['actor'] == actor and joint['probe'] == probe and actor['binding'] == _state_binding(config)
         and actor['source_commit'] == config['source_commit']
         and actor['completed_outer'] == 27 and actor['actual_adam_steps_per_rank'] == [540] * 4
         and actor['new_adam_steps_per_rank'] == [20] * 4 and actor['scheduler_completed_outer'] == 27
         and actor['next_learning_rate'] == learning_rate(27)
         and probe['completed_outer'] == 20 and probe['actual_adam_steps_per_rank'] == [100] * 4
         and seals.file(base / 'actor/latest.json')['sha256'] == ended['actor']['metadata_sha256']
         and seals.file(base / 'probe/latest.json')['sha256'] == ended['probe']['metadata_sha256'],
         'Complete actor27/probe20 metadata does not match the actual terminal saved state')
    need(seals.json(Path(actor['state_path']) / 'complete.json') == actor, 'Actor27 attempt producer receipt changed')
    pool, current, reservation = _pool_metadata(seals, config, joint)
    root, prefix, score, scan, selection = _prefix_metadata(seals, config, joint, actor, probe)
    # Small semantic/provenance checks above precede every multi-GB hidden/model
    # read.  Only producer-committed inputs are sealed; failed continuation data
    # without a complete rank receipt is intentionally left unqualified.
    for name in ('prefix', 'probe-score'):
        seals.tree(root / name)
    seals.verify_producer_references(prefix); seals.verify_producer_references(scan)
    seals.verify_producer_references(score)
    continuation = _complete_continuation_ranks(seals, config, root, actor, prefix, selection)
    for item in scan['files'].items():
        name, producer = item
        need(Path(name).name == name, 'Native scan file name escaped its accepted directory')
        seals.file(Path(prefix['scan_path']) / name, producer)
    rollout_path = seals.path(actor['original_rollout_path'])
    need(seals.file(rollout_path)['sha256'] == actor['original_rollout_sha256']
         and seals.json(rollout_path) == joint['rollout'], 'Actor27 lost its own actual retained rollout binding')
    hf = _hf(seals, actor, config)
    ranks = [_rank_state(seals, actor, config, 27, rank, actor['original_rollout_sha256']) for rank in range(4)]
    driver, driver_file = seals.mmap_torch(actor['driver_state']['path'], actor['driver_state'])
    need(driver['run_id'] == config['run_id'] and driver['completed_outer'] == 27
         and driver['actor_version'] == actor['actor_version'] and driver['candidate_pool_state']['value'] == pool
         and driver['candidate_pool_state']['sha256'] == seals.file(joint['pool_snapshot'])['sha256']
         and driver['prefix_probe_receipt']['value'] == probe
         and driver['prefix_probe_receipt']['sha256'] == seals.file(base / 'probe/latest.json')['sha256'],
         'Actor27 driver does not retain its true candidate27/probe20 boundary')
    driver_rng = validate_rng(driver['rng']); del driver; gc.collect()
    snapshot = seals.file(joint['probe_snapshot'], probe); latest = seals.file(base / 'probe/latest.pt', probe)
    need((snapshot['dev'], snapshot['inode']) == (latest['dev'], latest['inode']), 'Joint27 probe snapshot is not the same immutable probe20 inode')
    probe_proof = _probe(seals, config, probe, history[20])
    for name in ('logs/process-ledger.json', 'logs/job-ledger.json', 'logs/jobs.json', 'logs/qa-pending.json',
                 'preparation/submit-intent.json', 'probe/initialization-intent.json'):
        if (base / name).is_file(): seals.file(base / name)
    need(not (root / 'complete.json').exists() and not (root / 'full/complete.json').exists()
         and not list((base / 'actor/outers').glob('outer-0028-attempt-*')),
         'A new complete continuation/actor28 appeared during qualification')
    files = seals.finish()
    phase_hashes = {'prefix': seals.file(root / 'prefix/complete.json')['sha256'],
                    'probe-score': seals.file(root / 'probe-score/manifest.json')['sha256']}
    return dict(status=STATUS, adapter=ADAPTER, schema_version=1, run_id=config['run_id'],
        source_base=str(base), source_execution_id=config['execution_id'], source_commit=config['source_commit'],
        source_config_sha256=args.expected_config_sha256, outer=27, completed_outer=27, completed_joint_outer=27,
        source_joint_sha256=seals.file(base / 'latest-joint.json')['sha256'],
        joint_complete_sha256=seals.file(base / 'outers/outer-000027/complete.json')['sha256'],
        actor_completed_outer=27, actor_adam_per_rank=[540] * 4, actor_actual_adam_steps_per_rank=[540] * 4,
        scheduler_completed_outer=27, probe_completed_outer=20, probe_last_update_outer=20,
        probe_adam_per_rank=[100] * 4, probe_actual_adam_steps_per_rank=[100] * 4,
        pool_next_outer=28, pool_consumed_candidates=69120, pool_cursor=pool['cursor'], pool_cycle=pool['cycle'],
        pending_outer=28, qualified_prefix_outer=28, current_pool_next_outer=29,
        current_pool_consumed_candidates=71680, current_consumed_candidates=71680,
        current_pool_cursor=current['cursor'], current_pool_cycle=current['cycle'],
        reservation_before_object_sha256=reservation['before_sha256'], reservation_after_object_sha256=reservation['after_sha256'],
        phase_hashes=phase_hashes, continuation_rank_whitelist=continuation,
        owned_births_all_gone=True, actual_actor28_attempts=0, failed_continuation_complete=False,
        terminal_proof=str(terminal_file['path']), terminal_proof_sha256=terminal_file['sha256'],
        files=files, verified_files=files, actor_hf=hf, actor_rank_cpu_states=ranks,
        actor_driver_file=driver_file, actor_driver_rng=driver_rng, probe=probe_proof,
        native_recipe=dict(world_size=4, question_max_tokens=273, prefix_tokens=5, head_max_tokens=278,
            candidate_questions=2560, selected_questions=512, responses_per_question=5,
            probe_update_period=10, probe_updates_per_period=50, probe_learning_rate=5e-6),
        independent_full_file_sha_performed=True, actual_cpu_deserialization_performed=True,
        actual_cuda_reload_accepted=False, gpu_forward_performed=False, cuda_rng_applied=False,
        training_modified=False, sampler_advanced=False, optimizer_reset=False,
        rollout_regenerated=False, new_evaluation_performed=False, independent_backup_performed=False,
        source_warmup_checkpoint_read=False, large_payload_transferred_to_Mac=False)


def qualify(args):
    output = Path(args.output)
    need(output.is_absolute() and '..' not in output.parts and output.parent.is_dir(),
         'Prepare the new remote namespace before this one CPU qualification')
    _canonical_path(output.parent)
    need(str(output).startswith('/data/shaoshitong/search-g1-prefix-selection200-')
         and output.parent.name == 'preparation' and not output.is_symlink(),
         'Keep the CPU proof in the new canonical remote preparation directory')
    intent = output.with_name(output.stem + '-intent.json')
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            old = read(output)
            need(old.get('run_id') == args.expected_run and old.get('source_execution_id') == args.expected_execution
                 and old.get('source_config_sha256') == args.expected_config_sha256
                 and old.get('terminal_proof_sha256') == args.terminal_proof_sha256,
                 'Existing once CPU qualification belongs to a different source request')
            return dict(status='EXISTING_A800_JOINT_CPU_PROOF_NO_REPLAY', proof_status=old['status'],
                        proof=hash_file(output), independent_large_file_read_this_call=False)
        need(not intent.exists(), 'Incomplete prior qualification intent requires explicit reconciliation')
        with intent.open('x') as handle:
            json.dump(dict(run_id=args.expected_run, source_base=args.base,
                source_execution_id=args.expected_execution, source_commit=args.expected_source,
                source_config_sha256=args.expected_config_sha256, terminal_proof_sha256=args.terminal_proof_sha256,
                started_unix=time.time()), handle, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        started = time.monotonic()
        try:
            proof = _qualify(args)
        except BaseException as error:
            proof = dict(status='NATIVE_A800_JOINT_CPU_QUALIFICATION_FAILED', adapter=ADAPTER,
                run_id=args.expected_run, source_execution_id=args.expected_execution,
                source_config_sha256=args.expected_config_sha256, terminal_proof_sha256=args.terminal_proof_sha256,
                error_type=type(error).__name__, error=str(error), actual_cuda_reload_accepted=False,
                gpu_forward_performed=False, training_modified=False)
        proof.update(reader_source_sha256=hash_file(__file__)['sha256'], elapsed_seconds=time.monotonic() - started,
                     checked_unix=time.time())
        with output.open('x') as handle:
            json.dump(proof, handle, sort_keys=True, allow_nan=False); handle.flush(); os.fsync(handle.fileno())
        return dict(status=proof['status'], proof=hash_file(output), outer=proof.get('outer'),
            actor_adam_per_rank=proof.get('actor_adam_per_rank'), probe_adam_per_rank=proof.get('probe_adam_per_rank'),
            error=proof.get('error'), actual_cuda_reload_accepted=False, gpu_forward_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'expected-run', 'expected-execution', 'expected-source', 'expected-config-sha256',
                 'terminal-proof', 'terminal-proof-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(qualify(parser.parse_args()), sort_keys=True, allow_nan=False))


def inspect(base, outer=27, *, expected_run, expected_execution, expected_source,
            expected_config_sha, terminal_proof, terminal_sha, output):
    """Python entry matching the bounded controller's explicit source contract."""
    need(type(outer) is int and outer == OUTER, 'This concrete adapter qualifies the actual joint27 fault only')
    return qualify(argparse.Namespace(base=str(base), expected_run=expected_run,
        expected_execution=expected_execution, expected_source=expected_source,
        expected_config_sha256=expected_config_sha, terminal_proof=str(terminal_proof),
        terminal_proof_sha256=terminal_sha, output=str(output)))


if __name__ == '__main__': main()
