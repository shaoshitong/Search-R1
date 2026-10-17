"""Authorized retained50 rollback: preserve trained state, change probe LR only.

This is an explicit human rollback, not fault recovery to an older helper point.
Source50 and its completed QA/grounding fit stay immutable. No warmup, rollout,
actor update, probe Adam write or model forward is performed by this importer.
"""
import argparse
import json
import os
from pathlib import Path

from .actor import read, sha
from .checkpoint_cpu_once import digest
from .resume import boundary, validate_imported_boundary
from reproduction.prefix20_online.launch import once, publish


def validate_authorization(config):
    spec = config.get('resume_joint', {})
    human = spec.get('human_authorization', {})
    if (config.get('probe_learning_rate') != 1e-5
            or spec.get('completed_outer') != 50
            or human.get('resume_from50') is not True
            or human.get('probe_lr_only') is not True
            or human.get('raw_human_message') != '需要变成1e-5跑，探针部分'):
        raise ValueError('Require the actual explicit retained50 probe-only LR authorization')


def validate_change(config, original):
    validate_authorization(config)
    allowed = {'base', 'source_commit', 'resume_joint', 'resume_probe_lr_transition',
               'probe_learning_rate', 'pool_state_path', 'prefix_probe_receipt'}
    if (original['probe_learning_rate'] != 1e-6
            or set(config) - set(original) - allowed
            or any(config.get(key) != value for key, value in original.items() if key not in allowed)):
        raise ValueError('Only probe LR and isolated execution/recovery paths may change')
    for key, suffix in (('pool_state_path', 'question-pool/state.json'),
                        ('prefix_probe_receipt', 'probe/latest.json')):
        if key in config and config[key] != str(Path(config['base']) / suffix):
            raise ValueError('Recovery must not write into source live state')


def rebind_trained_probe(saved):
    """Share immutable tensors/RNG; copy only groups and authorized LR metadata."""
    if (saved['completed_outer'] != 40 or saved['actual_adam_steps_per_rank'] != [200] * 8
            or saved['binding']['learning_rate'] != 1e-6 or saved['learning_rate'] != 1e-6
            or saved['binding_sha256'] != digest(saved['binding'])
            or saved['optimizer'] is None or len(saved['rank_rng']) != 8):
        raise ValueError('Retained50 must contain the genuine trained40 probe/Adam200')
    if any(group['lr'] != 1e-6 for group in saved['optimizer']['param_groups']):
        raise ValueError('Saved Adam parameter groups differ from actual source rate')
    for rank, rng in enumerate(saved['rank_rng']):
        if rng['rank'] != rank or rng['actual_adam_steps'] != 200:
            raise ValueError('Preserve all eight actual trained RNG streams')
    optimizer = dict(saved['optimizer'], param_groups=[dict(group, lr=1e-5)
                         for group in saved['optimizer']['param_groups']])
    changed = dict(saved, optimizer=optimizer, binding=dict(saved['binding'], learning_rate=1e-5),
                   learning_rate=1e-5)
    changed['binding_sha256'] = digest(changed['binding'])
    return changed


def assert_equal(a, b):
    """Actual CPU equality of tensor moments, weights, counters and saved RNG."""
    import torch
    import numpy as np
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError('A trained tensor changed during LR-only recovery')
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
            raise ValueError('Saved NumPy RNG changed')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('Saved state fields changed')
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('Saved sequence state changed')
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif a != b:
        raise ValueError('Saved scalar/counter changed')


def import_complete50(base):
    import torch
    from .probe import atomic_checkpoint
    from .grounding import recovery_state
    base = Path(base); config = read(base / 'preparation/launch-config.json')
    spec = config['resume_joint']; source = Path(spec['source_base'])
    prep = base / 'preparation'; receipt = prep / 'resume-joint-receipt.json'
    intent = prep / 'resume-joint-intent.json'
    if receipt.exists():
        validate_imported_boundary(base, config)
        return dict(status='EXISTING_TRAINED50_LR_IMPORT_NO_REPLAY', receipt=read(receipt))
    if intent.exists():
        raise RuntimeError('Existing LR import intent requires reconciliation, never replay')
    original = read(source / 'preparation/launch-config.json')
    validate_change(config, original)
    if (source == base or source.parent != base.parent or source.resolve() != source
            or sha(source / 'preparation/launch-config.json') != spec['source_config_sha256']):
        raise ValueError('Preserve exact source execution/configuration identity')
    terminal = Path(spec['terminal_proof']); cpu = Path(spec['checkpoint_cpu_proof'])
    if sha(terminal) != spec['terminal_proof_sha256'] or sha(cpu) != spec['checkpoint_cpu_proof_sha256']:
        raise ValueError('Terminal or independently verified50 proof changed')
    ended, checked = read(terminal), read(cpu)
    if (ended.get('status') != 'HUMAN_STOP_ORIGINAL_INODE_FULL_SHA_OWNED_RELEASED'
            or ended['actual_births_present'] or ended['native_job_cgroup_paths'] or ended['owned_gpu']
            or ended['run_id'] != config['run_id'] or ended['job_id'] != str(spec['source_job_id'])
            or ended['latest_complete_outer'] < 50
            or checked['status'] != 'NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'
            or checked['run_id'] != config['run_id'] or checked['outer'] != 50):
        raise ValueError('Require actual scoped source stop and verified retained50')
    for item in checked['files']:
        info = Path(item['path']).stat()
        if (info.st_ino, info.st_size, info.st_mtime_ns) != (item['inode'], item['bytes'], item['mtime_ns']):
            raise ValueError('Independently verified immutable50 file changed')
    root = source / 'outers/outer-000050'; path = root / 'complete.json'
    if sha(path) != spec['source_joint_sha256']:
        raise ValueError('Explicit retained50 differs; do not substitute latest or older state')
    joint = read(path); actor, probe = joint['actor'], joint['probe']
    pool = read(joint['pool_snapshot']); boundary(config, joint, actor, probe, pool)
    saved = torch.load(joint['probe_snapshot'], map_location='cpu', weights_only=False)
    changed = rebind_trained_probe(saved)
    grounding = recovery_state(source, config, joint)
    if grounding['round'] != 50 or not grounding.get('adoption_receipt'):
        raise ValueError('Reuse the actual fitted/adopted actor0050 reward evaluator')
    once(intent, dict(run_id=config['run_id'], completed_outer=50,
                     source_base=str(source), source_joint_sha256=spec['source_joint_sha256'],
                     human_authorization=spec['human_authorization'], probe_lr_change=[1e-6, 1e-5]))
    for name in ('actor', 'probe', 'question-pool', 'outers/outer-000050', 'grounding'):
        (base / name).mkdir(parents=True, exist_ok=True)
    assets = []
    def link(src, dst):
        src, dst = Path(src), Path(dst)
        if src.is_symlink() or not src.is_file():
            raise ValueError('Import exact immutable source files only')
        dst.parent.mkdir(parents=True, exist_ok=True); os.link(src, dst)
        assets.append(dict(source=str(src), target=str(dst), bytes=src.stat().st_size, inode=src.stat().st_ino))
    rebound = atomic_checkpoint(base / 'probe', changed, 40)
    restored = torch.load(base / 'probe/latest.pt', map_location='cpu', weights_only=False)
    assert_equal(changed, restored)
    assert_equal(saved['model'], restored['model'])
    assert_equal(saved['optimizer']['state'], restored['optimizer']['state'])
    assert_equal(saved['rank_rng'], restored['rank_rng'])
    for before, after in zip(saved['optimizer']['param_groups'], restored['optimizer']['param_groups']):
        assert_equal({k: v for k, v in before.items() if k != 'lr'},
                     {k: v for k, v in after.items() if k != 'lr'})
    if any(group['lr'] != 1e-5 for group in restored['optimizer']['param_groups']):
        raise ValueError('Actual serialized optimizer rate differs')
    once(base / 'probe/initialization-complete.json', dict(
        read(source / 'probe/initialization-complete.json'),
        imported_trained_state_from50=True, warmup_updates_reexecuted=0))
    change = dict(run_id=config['run_id'], resume_outer=50, source_probe_receipt=probe,
                  rebound_probe_receipt=rebound, probe_actual_adam_per_rank=[200] * 8,
                  source_learning_rate=1e-6, learning_rate=1e-5,
                  model_and_adam_moments_and_eight_rng_equal=True,
                  new_probe_adam_writes=0, human_authorization=spec['human_authorization'])
    once(base / 'probe/lr-change-receipt.json', change)
    once(base / 'actor/latest.json', actor)
    link(source / 'question-pool/binding.json', base / 'question-pool/binding.json')
    link(joint['pool_snapshot'], base / 'question-pool/state.json')
    for n in range(1, 50):
        previous = source / 'outers' / ('outer-%06d' % n)
        for name in ('complete.json', 'probe-update/metrics.json', 'probe-frozen.json',
                     'grounding-state-at-completion.json'):
            if (previous / name).is_file():
                link(previous / name, base / 'outers' / previous.name / name)
    destination = base / 'outers/outer-000050'
    link(joint['pool_snapshot'], destination / 'pool-state-at-completion.json')
    link(base / 'probe/latest.pt', destination / 'probe-state-at-completion.pt')
    imported = dict(joint, probe=rebound, probe_snapshot=str(destination / 'probe-state-at-completion.pt'),
                    pool_snapshot=str(destination / 'pool-state-at-completion.json'),
                    source_joint_sha256=spec['source_joint_sha256'], approved_probe_lr_change=[1e-6, 1e-5])
    once(destination / 'complete.json', imported); once(base / 'latest-joint.json', imported)
    publish(base / 'grounding/latest.json', grounding)
    # Startup adopts the already fitted50 heads on eight new replicas, then
    # writes a NEW50 adoption receipt; never import the source's old live PIDs.
    qa = root / 'validation/complete.json'
    if sha(qa) != spec['source_qa50_sha256'] or read(qa)['actor_version'] != actor['actor_version']:
        raise ValueError('Reuse only the exact complete actor0050 QA result')
    link(qa, base / 'recovered-validation/complete.json')
    result = dict(status='HUMAN_RETAINED50_TRAINED_PROBE_LR_ONLY_IMPORTED', run_id=config['run_id'],
        completed_outer=50, next_outer=51, source_base=str(source),
        source_latest_outer_discarded_after50=ended['latest_complete_outer'],
        joint_sha256=sha(base / 'latest-joint.json'), probe_inode=(base / 'probe/latest.pt').stat().st_ino,
        actor_adam_per_rank=[1000] * 8, probe_adam_per_rank=[200] * 8,
        scheduler_completed_outer=50, candidate_cursor=pool['cursor'], candidate_cycle=pool['cycle'],
        probe_lr_before=1e-6, probe_lr_after=1e-5, probe_model_tensors_equal=len(saved['model']),
        optimizer_reset=False, model_and_adam_moments_and_eight_rng_equal=True,
        actor_state_changed=False, new_probe_adam_writes=0, grounding_round=50,
        first_future_probe_update_outer=60, source_modified=False, assets=assets,
        large_assets_transferred_to_Mac=False)
    once(receipt, result); validate_imported_boundary(base, config)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--base', required=True)
    print(json.dumps(import_complete50(parser.parse_args().base)))
