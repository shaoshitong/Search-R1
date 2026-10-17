"""Import a verified complete joint boundary into a new execution namespace.

Actor assets keep their immutable source paths. Probe tensors and pool JSON use
hardlinks; later atomic publication replaces the destination only. No old run
is modified, no rollout is regenerated, and neither optimizer is reset.
"""
import json
import os
from pathlib import Path
import time

from .actor import read, sha, restore_metadata, _state_binding
from reproduction.prefix20_online.launch import once


def pending_reservation(source, committed_pool, outer, run_id):
    """Keep a failed next outer's original questions without consuming twice."""
    from .pool import reserve_indices, state_sha
    source = Path(source)
    live = read(source / 'question-pool/state.json')
    if live == committed_pool:
        return None
    following = source / 'outers' / ('outer-%06d' % (outer + 1))
    if (following / 'complete.json').exists():
        raise ValueError('A newer complete outer must be recovered instead')
    saved = read(following / 'pool-reservation.json')
    indices, cycles, after = reserve_indices(committed_pool, outer + 1)
    binding = read(source / 'question-pool/binding.json')
    selection = read(following / 'selection.json')
    questions = read(following / 'questions.json')
    if (saved['outer'] != outer + 1 or saved['pool_sha256'] != binding['sha256']
            or saved['before_sha256'] != state_sha(committed_pool)
            or saved['after_sha256'] != state_sha(after) or saved['after'] != after
            or saved['indices'] != indices or saved['cycles'] != cycles or live != after
            or selection['run_id'] != run_id or selection['outer'] != outer + 1
            or selection['questions'] != 5120 or len(selection['rows']) != 5120
            or questions['run_id'] != run_id or questions['outer'] != outer + 1
            or questions['questions'] != 5120 or len(questions['rows']) != 5120
            or questions['selection_sha256'] != sha(following / 'selection.json')):
        raise ValueError('Uncommitted next reservation differs from its saved complete boundary')
    for ordinal, (selected, question, index, cycle) in enumerate(
            zip(selection['rows'], questions['rows'], indices, cycles)):
        if (selected['pool_index'] != index or selected['pool_cycle'] != cycle
                or selected['candidate_ordinal'] != ordinal
                or any(question[key] != value for key, value in selected.items())):
            raise ValueError('Reserved question order/content differs')
    return dict(outer=outer + 1, source_root=str(following), questions=5120,
                files={name: sha(following / name) for name in
                       ('pool-reservation.json', 'selection.json', 'questions.json')},
                restore_committed_pool_before_reservation=True,
                no_second_candidate_consumption=True)


def boundary(config, joint, actor, probe, pool):
    outer = config['resume_joint']['completed_outer']
    from .periodic import probe_boundary
    native=config.get('actor_hidden_native',False)
    expected=probe_boundary(outer) if native else dict(last_update_outer=outer,actual_adam_per_rank=10*outer)
    if (not 0 < outer <= 200 or joint['run_id'] != config['run_id'] or joint['outer'] != outer
            or joint['actor'] != actor or joint['probe'] != probe
            or probe['run_id'] != config['run_id'] or probe['completed_outer'] != expected['last_update_outer']
            or probe['actual_adam_steps_per_rank'] != [expected['actual_adam_per_rank']] * 8
            or pool['next_outer'] != outer + 1 or pool['consumed_candidates'] != 5120 * outer
            or pool['cycle'] * len(pool['order']) + pool['cursor'] != 5120 * outer):
        raise ValueError('Restore only the complete joint actor/probe/candidate boundary')
    if config.get('grounding_refresh_enabled'):
        from .grounding import validate_joint
        validate_joint(config, joint)
        # At a due milestone the actor state is valid before its refit. The
        # launcher restores/adopts the new evaluator before the NEXT update.
        if (actor['binding'] != _state_binding(config)
                or actor['actual_adam_steps_per_rank'] != [20 * outer] * 8
                or actor['scheduler_completed_outer'] != outer):
            raise ValueError('Restore the exact actor/Adam/scheduler scientific boundary')
    else:
        restore_metadata(config, actor, outer + 1)
    return outer


def validate_imported_boundary(base, config):
    base = Path(base)
    receipt = read(base / 'preparation/resume-joint-receipt.json')
    joint = read(base / 'latest-joint.json')
    outer = boundary(config, joint, read(base / 'actor/latest.json'),
                     read(base / 'probe/latest.json'), read(base / 'question-pool/state.json'))
    if (receipt['completed_outer'] != outer or receipt['run_id'] != config['run_id']
            or receipt['joint_sha256'] != sha(base / 'latest-joint.json')
            or receipt['probe_inode'] != (base / 'probe/latest.pt').stat().st_ino):
        raise ValueError('Imported complete state differs from its explicit recovery receipt')
    return outer


def import_complete(base):
    base = Path(base); config = read(base / 'preparation/launch-config.json')
    spec = config['resume_joint']; source = Path(spec['source_base'])
    receipt = base / 'preparation/resume-joint-receipt.json'
    intent = base / 'preparation/resume-joint-intent.json'
    if receipt.exists():
        return dict(status='EXISTING_JOINT_IMPORT_NO_REPLAY', receipt=read(receipt))
    if intent.exists():
        return dict(status='EXISTING_JOINT_IMPORT_INTENT_RECONCILE_REQUIRED')
    if (source == base or source.parent != base.parent or source.resolve(strict=True) != source
            or not source.name.startswith('search-g1-prefix-selection200-')):
        raise ValueError('Recovery source must be this canonical remote experiment')
    original = read(source / 'preparation/launch-config.json')
    if sha(source / 'preparation/launch-config.json') != spec['source_config_sha256']:
        raise ValueError('Recovery source configuration changed')
    terminal = Path(spec['terminal_proof']); proof = Path(spec['checkpoint_cpu_proof'])
    if sha(terminal) != spec['terminal_proof_sha256'] or sha(proof) != spec['checkpoint_cpu_proof_sha256']:
        raise ValueError('Actual terminal or independent checkpoint proof changed')
    ended, checked = read(terminal), read(proof)
    native=config.get('actor_hidden_native',False)
    actual_gpu=ended.get('owned_gpu',[]) if native else ended['gpu']['stdout'].strip()
    expected_status='NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS' if native else 'PREFIX_SELECTION200_JOINT_MILESTONE_FULL_SHA_CPU_PASS'
    if (ended['actual_births_present'] or ended['native_job_cgroup_paths'] or actual_gpu
            or checked['status'] != expected_status
            or checked['run_id'] != config['run_id'] or checked['outer'] != spec['completed_outer']):
        raise ValueError('Recovery requires real exit/resource release and actual complete checkpoint readback')
    science = ('model_revision', 'model_files_sha256', 'candidate_questions', 'selected_questions',
        'responses_per_question', 'prefix_policy_tokens', 'exploration_questions', 'probe_learning_rate',
        'probe_epochs_per_outer', 'probe_updates_per_outer', 'probe_global_batch_average',
        'actor_global_mini_batch', 'actor_ppo_epochs', 'seed', 'official_model_dir',
        'encoder_model_dir', 'canonical_train_sha256', 'fixed_test_manifest_sha256', 'grounding_probe_version')
    if native:science+=('actor_hidden_native','hidden_capture_contract','probe_update_period','probe_updates_per_period','probe_epochs_per_update')
    if config.get('grounding_refresh_enabled'):
        science += ('grounding_refresh_enabled', 'grounding_refit_rounds',
                    'grounding_manifest_sha256', 'grounding_calibration_sha256',
                    'grounding_calibration_question_ids')
    if any(config[k] != original[k] for k in science):
        raise ValueError('Recovery cannot change the scientific training contract')
    def checked_stats(value):
        if isinstance(value, dict):
            if all(k in value for k in ('path', 'bytes', 'inode', 'mtime_ns')):
                info = Path(value['path']).stat()
                if (info.st_ino, info.st_size, info.st_mtime_ns) != (value['inode'], value['bytes'], value['mtime_ns']):
                    raise ValueError('Independently read checkpoint file changed after CPU validation')
            for item in value.values(): checked_stats(item)
        elif isinstance(value, list):
            for item in value: checked_stats(item)
    checked_stats(checked['files'] if native else checked['details'])
    joint = read(source / 'latest-joint.json'); actor = read(source / 'actor/latest.json')
    probe = read(source / 'probe/latest.json'); pool = read(joint['pool_snapshot'])
    outer = boundary(config, joint, actor, probe, pool)
    reservation = pending_reservation(source, pool, outer, config['run_id'])
    if (sha(source / 'outers' / ('outer-%06d' % outer) / 'complete.json') != (ended['joint_complete_sha256'] if native else checked['joint_complete_sha256'])
            or (source / 'probe/latest.pt').stat().st_ino != Path(joint['probe_snapshot']).stat().st_ino):
        raise ValueError('Latest complete checkpoint/pool differs from its independent readback')
    once(intent, dict(run_id=config['run_id'], completed_outer=outer, source_base=str(source),
                      checkpoint_cpu_proof_sha256=spec['checkpoint_cpu_proof_sha256'], started_unix=time.time()))
    assets = []
    def link(src, dst):
        src = Path(src)
        if src.is_symlink() or not src.is_file(): raise ValueError('Recovery assets must be real source files')
        dst.parent.mkdir(parents=True, exist_ok=True); os.link(src, dst)
        assets.append(dict(source=str(src), target=str(dst), bytes=src.stat().st_size, inode=src.stat().st_ino))
    link(source / 'actor/latest.json', base / 'actor/latest.json')
    for name in ('initialization-complete.json', 'latest.json', 'latest.pt'):
        link(source / 'probe' / name, base / 'probe' / name)
    link(source / 'question-pool/binding.json', base / 'question-pool/binding.json')
    link(joint['pool_snapshot'], base / 'question-pool/state.json')
    link(source / 'latest-joint.json', base / 'latest-joint.json')
    if config.get('grounding_refresh_enabled'):
        from .grounding import recovery_state
        from reproduction.prefix20_online.launch import publish
        current_grounding = recovery_state(source, config, joint)
        (base / 'grounding').mkdir(parents=True, exist_ok=True)
        publish(base / 'grounding/latest.json', current_grounding)
    # Small completed metrics retain the full learning history; large original
    # trajectories/checkpoints remain at their recorded immutable source paths.
    for n in range(1, outer + 1):
        src = source / 'outers' / ('outer-%06d' % n)
        for name in ('complete.json', 'probe-update/metrics.json', 'probe-frozen.json',
                     'grounding-state-at-completion.json', 'grounding-refresh.json'):
            path = src / name
            if path.is_file(): link(path, base / 'outers' / src.name / name)
    if reservation:
        following = Path(reservation['source_root'])
        for name, digest in reservation['files'].items():
            if sha(following / name) != digest:
                raise ValueError('Reserved question bundle changed before import')
            link(following / name, base / 'outers' / following.name / name)
    result = dict(status='COMPLETE_JOINT_STATE_IMPORTED_NO_RESET', run_id=config['run_id'],
        source_base=str(source), completed_outer=outer, next_outer=outer + 1,
        joint_sha256=sha(base / 'latest-joint.json'), probe_inode=(base / 'probe/latest.pt').stat().st_ino,
        actor_adam_per_rank=actor['actual_adam_steps_per_rank'], probe_adam_per_rank=probe['actual_adam_steps_per_rank'],
        scheduler_completed_outer=actor['scheduler_completed_outer'], candidate_cursor=pool['cursor'],
        candidate_cycle=pool['cycle'], assets=assets, source_modified=False, optimizer_reset=False,
        large_assets_transferred_to_Mac=False, pending_reservation=reservation)
    if config.get('grounding_refresh_enabled'):
        result['grounding_state'] = current_grounding
        result['grounding_refit_pending_before_next_actor'] = current_grounding['round'] != (
            min(outer // 50 * 50, 150))
    once(receipt, result)
    validate_imported_boundary(base, config)
    return result
