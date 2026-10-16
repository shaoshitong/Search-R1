"""Human-authorized rollback to retained joint5; change only future probe LR.

The original actor/Adam/scheduler/RNG and question permutation remain immutable.
The probe has zero online Adam at5: only its future-LR metadata is rebound.
No warmup, rollout or model forward occurs in this CPU import.
"""
import argparse
import json
import os
from pathlib import Path

from .actor import read, sha
from .resume import boundary, validate_imported_boundary
from reproduction.prefix20_online.launch import once


SCIENCE = ('run_id', 'model_revision', 'model_files_sha256', 'candidate_questions',
           'selected_questions', 'responses_per_question', 'prefix_policy_tokens',
           'exploration_questions', 'probe_epochs_per_outer', 'probe_updates_per_outer',
           'probe_global_batch_average', 'actor_global_mini_batch', 'actor_ppo_epochs',
           'seed', 'official_model_dir', 'encoder_model_dir', 'canonical_train_sha256',
           'fixed_test_manifest_sha256', 'grounding_probe_version', 'actor_hidden_native',
           'hidden_capture_contract', 'probe_update_period', 'probe_updates_per_period',
           'probe_epochs_per_update', 'actor_global_micro_batch_size', 'world_size',
           'outers', 'validation_questions_sha256', 'final_questions_sha256')


def validate_change(config, original):
    if any(config[key] != original[key] for key in SCIENCE):
        raise ValueError('The authorized rollback changes only probe learning rate')
    if original['probe_learning_rate'] != 1e-6 or config['probe_learning_rate'] != 1e-5:
        raise ValueError('Require the actual authorized probe LR1e-6 to1e-5 change')
    spec = config['resume_joint']
    if spec['completed_outer'] != 5 or not spec['human_authorization'].get('resume_from5'):
        raise ValueError('The human explicitly selected retained complete5')


def rebind_untrained_probe(saved, rate):
    from .checkpoint_cpu_once import digest
    if (saved['completed_outer'] != 0 or saved['actual_adam_steps_per_rank'] != [0] * 8
            or saved['optimizer'] is not None or saved['rank_rng'] is not None
            or saved['binding']['learning_rate'] != 1e-6):
        raise ValueError('Only the genuine pre20 online-Adam0 probe may be rebound')
    changed = dict(saved, binding=dict(saved['binding'], learning_rate=rate), learning_rate=rate)
    changed['binding_sha256'] = digest(changed['binding'])
    return changed


def import_complete5(base):
    import torch
    from .probe import atomic_checkpoint
    base = Path(base); config = read(base / 'preparation/launch-config.json')
    spec = config['resume_joint']; source = Path(spec['source_base'])
    receipt = base / 'preparation/resume-joint-receipt.json'
    intent = base / 'preparation/resume-joint-intent.json'
    if receipt.exists():
        return dict(status='EXISTING_LR5_IMPORT_NO_REPLAY', receipt=read(receipt))
    if intent.exists():
        return dict(status='EXISTING_LR5_IMPORT_INTENT_RECONCILE_REQUIRED')
    original = read(source / 'preparation/launch-config.json')
    validate_change(config, original)
    if (base == source or base.parent != source.parent or source.resolve() != source
            or sha(source / 'preparation/launch-config.json') != spec['source_config_sha256']):
        raise ValueError('Source execution/configuration identity differs')
    terminal, proof = Path(spec['terminal_proof']), Path(spec['checkpoint_cpu_proof'])
    if sha(terminal) != spec['terminal_proof_sha256'] or sha(proof) != spec['checkpoint_cpu_proof_sha256']:
        raise ValueError('Actual terminal or retained checkpoint proof changed')
    ended, checked = read(terminal), read(proof)
    if (ended['actual_births_present'] or ended['native_job_cgroup_paths'] or ended['owned_gpu']
            or checked['status'] != 'NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'
            or checked['run_id'] != config['run_id'] or checked['outer'] != 5):
        raise ValueError('Require stopped old execution and independently verified joint5')
    for item in checked['files']:
        info = Path(item['path']).stat()
        if (info.st_ino, info.st_size, info.st_mtime_ns) != (item['inode'], item['bytes'], item['mtime_ns']):
            raise ValueError('Verified immutable checkpoint asset changed')
    path = source / 'outers/outer-000005/complete.json'
    if sha(path) != spec['source_joint_sha256']:
        raise ValueError('Explicit retained joint5 metadata changed')
    joint = read(path); actor, probe = joint['actor'], joint['probe']
    pool = read(joint['pool_snapshot'])
    boundary(config, joint, actor, probe, pool)
    probe_path = Path(joint['probe_snapshot'])
    if probe_path.stat().st_size != probe['bytes']:
        raise ValueError('Source joint5 probe size changed')
    saved = torch.load(probe_path, map_location='cpu', weights_only=False)
    changed = rebind_untrained_probe(saved, config['probe_learning_rate'])
    once(intent, dict(run_id=config['run_id'], completed_outer=5, source_base=str(source),
                      source_joint_sha256=spec['source_joint_sha256'],
                      human_authorization=spec['human_authorization'], probe_lr_change=[1e-6, 1e-5]))
    for name in ('actor', 'probe', 'question-pool', 'outers/outer-000005'):
        (base / name).mkdir(parents=True, exist_ok=True)
    assets = []
    def link(src, dst):
        src, dst = Path(src), Path(dst)
        if not src.is_file() or src.is_symlink():
            raise ValueError('Import immutable real source assets only')
        dst.parent.mkdir(parents=True, exist_ok=True); os.link(src, dst)
        assets.append(dict(source=str(src), target=str(dst), inode=src.stat().st_ino))
    for n in range(1, 6):
        src = source / 'outers' / ('outer-%06d' % n)
        for name in ('complete.json', 'probe-frozen.json'):
            if n == 5 and name == 'complete.json':
                continue
            if (src / name).is_file():
                link(src / name, base / 'outers' / src.name / name)
    new_probe = atomic_checkpoint(base / 'probe', changed, 0)
    restored = torch.load(base / 'probe/latest.pt', map_location='cpu', weights_only=False)
    if (set(restored['model']) != set(saved['model'])
            or not all(torch.equal(restored['model'][key], value) for key, value in saved['model'].items())):
        raise ValueError('Probe model values changed during LR-only metadata rebind')
    initial = read(source / 'probe/initialization-complete.json')
    once(base / 'probe/initialization-complete.json', dict(initial, receipt=new_probe,
         human_lr_metadata_change=[1e-6, 1e-5], imported_from_joint=5, model_values_equal=True))
    once(base / 'actor/latest.json', actor)
    link(source / 'question-pool/binding.json', base / 'question-pool/binding.json')
    link(joint['pool_snapshot'], base / 'question-pool/state.json')
    destination = base / 'outers/outer-000005'
    link(joint['pool_snapshot'], destination / 'pool-state-at-completion.json')
    link(base / 'probe/latest.pt', destination / 'probe-state-at-completion.pt')
    imported = dict(joint, probe=new_probe, probe_snapshot=str(destination / 'probe-state-at-completion.pt'),
                    pool_snapshot=str(destination / 'pool-state-at-completion.json'),
                    approved_probe_lr_change=[1e-6, 1e-5], source_joint_sha256=spec['source_joint_sha256'])
    once(destination / 'complete.json', imported); once(base / 'latest-joint.json', imported)
    qa = source / 'outers/outer-000005/validation/complete.json'
    if sha(qa) != spec['source_qa5_sha256']:
        raise ValueError('Original already-completed QA5 receipt changed')
    q = read(qa)
    if q['actor_version'] != actor['actor_version'] or q['qa_count'] != 512:
        raise ValueError('Reuse only the exact saved actor5 QA512 result')
    link(qa, base / 'recovered-validation/complete.json')
    result = dict(status='HUMAN_RETAINED5_IMPORTED_PROBE_LR_ONLY_CHANGED', run_id=config['run_id'],
        completed_outer=5, next_outer=6, joint_sha256=sha(base / 'latest-joint.json'),
        probe_inode=(base / 'probe/latest.pt').stat().st_ino,
        actor_adam_per_rank=actor['actual_adam_steps_per_rank'], probe_adam_per_rank=[0]*8,
        scheduler_completed_outer=5, candidate_cursor=pool['cursor'], candidate_cycle=pool['cycle'],
        source_base=str(source), source_joint_sha256=spec['source_joint_sha256'],
        probe_lr_before=1e-6, probe_lr_after=1e-5, probe_model_tensors_equal=len(saved['model']),
        actor_state_changed=False, optimizer_reset=False, source_modified=False,
        large_assets_transferred_to_Mac=False, assets=assets)
    once(receipt, result); validate_imported_boundary(base, config)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(import_complete5(args.base)))
