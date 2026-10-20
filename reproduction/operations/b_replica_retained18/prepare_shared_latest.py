"""Prepare an isolated, CPU-qualified same-rank continuation without dispatch."""
import argparse
import json
from pathlib import Path
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    source, base = Path(args.source), Path(args.base)
    from reproduction.prefix_selection200.diverse_resume import read, sha, once
    from reproduction.prefix_selection200.diverse_native_resume import prepare, PROTECTED, HARDWARE_FOUR
    joint = read(source / 'latest-joint.json')
    proof = source / 'preparation' / ('recovery%d-cpu' % args.outer) / 'full-state-proof.json'
    qualified = read(proof)
    if (joint['outer'] != args.outer or qualified['outer'] != args.outer
            or qualified['joint_sha256'] != sha(source / 'latest-joint.json')):
        raise ValueError('Prepare only the exact latest qualified complete joint')
    old = read(source / 'preparation' / ('outer-%06d-launch-config.json' % args.outer))

    def rebase(value):
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str) and (value == str(source) or value.startswith(str(source) + '/')):
            return str(base) + value[len(str(source)):]
        return value

    config = rebase(old)
    config.update(base=str(base), execution_id=uuid.uuid4().hex, source_commit=args.commit,
                  expected_execution_node='UNASSIGNED', preparation_only=True,
                  training_authorized=False, experiment_assignment_pending=False)
    world=old['world_size']
    if world not in (4,8):raise ValueError('Retain native rank count')
    config['resume_joint'] = dict(adapter='b_diverse_4to8_cp50_v1',
        recovery_mode='diverse_same_four_latest_joint_v1' if world==4 else 'diverse_same_eight_latest_joint_v1', source_base=str(source),
        completed_outer=args.outer)
    for key in ('resume_cached_sampling', 'runtime_compatibility', 'startup_sampling_resume'):
        config.pop(key, None)
    protected = {name: sha(source / 'repo/reproduction/prefix_selection200' / name)
                 for name in PROTECTED if world==8 or name not in HARDWARE_FOUR}
    for name, digest in protected.items():
        if sha(base / 'repo/reproduction/prefix_selection200' / name) != digest:
            raise ValueError('Scientific runtime changed: ' + name)
    compatibility = dict(status='DIVERSE_SAME_FOUR_NATIVE_RECOVERY' if world==4 else 'DIVERSE_SAME_EIGHT_NATIVE_RECOVERY',
        runtime_commit=args.commit, binding_commit=old['actor_checkpoint_binding_commit'],
        source_commit=old['source_commit'], source_base=str(source),
        completed_outer=args.outer, protected_sha256=protected)
    if world==4:
        compatibility['hardware_dispatch_sha256']={name:dict(source=sha(source/'repo/reproduction/prefix_selection200'/name),target=sha(base/'repo/reproduction/prefix_selection200'/name))for name in HARDWARE_FOUR}
    compatibility_path = base / 'preparation/runtime-compatibility.json'
    once(compatibility_path, compatibility)
    config['runtime_compatibility'] = dict(path=str(compatibility_path), sha256=sha(compatibility_path))
    once(base / 'preparation/launch-config.json', config)
    imported = prepare(source, base, config, proof)
    migrations = []
    for path in sorted((source / 'preparation').glob('prefix-migration-*.json')):
        target = base / 'preparation' / path.name
        with target.open('xb') as output:
            output.write(path.read_bytes())
        migrations.append(dict(path=str(target), sha256=sha(target)))
    result = dict(status='LATEST_NATIVE_JOINT_PREPARED_AWAITING_TARGET',
        outer=args.outer, next_outer=args.outer + 1, base=str(base),
        world_size=world,global_micro=world,local_micro=1,accumulation=128//world,
        prefix_tokens=config['prefix_policy_tokens'], actor=imported['actor_adam'],
        probe=imported['probe_adam'], prefix_migrations=migrations,
        source_commit=args.commit, optimizer_reset=False, training_started=False,
        ready_unix=time.time())
    once(base / 'preparation/native-recovery-complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
