"""Resume a committed actor whose periodic probe has not written any updates.

This is deliberately a different boundary from ``resume_joint``.  Actor10 and
the candidate reservation10 already exist, while the last joint commit is9.
The CPU qualifier seals the saved actor/optimizer/RNG, unchanged probe0, and
the exact already sampled inputs.  Importing does no tensor deserialization,
sampling, scoring, actor update, or reset.  The new execution completes probe10
before it can publish joint10 or reserve candidates11.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .actor import read, sha, restore_metadata
from .contract import CANDIDATE_QUESTIONS, WORLD
from .periodic import is_probe_update, probe_boundary
from reproduction.prefix20_online.launch import once


SCIENCE_KEYS = (
    'run_id', 'model_revision', 'model_files_sha256', 'candidate_questions',
    'selected_questions', 'responses_per_question', 'prefix_policy_tokens',
    'exploration_questions', 'probe_learning_rate', 'probe_epochs_per_update',
    'probe_update_period', 'probe_updates_per_period', 'probe_update_at_final',
    'probe_global_batch_average', 'actor_global_mini_batch', 'actor_ppo_epochs',
    'seed', 'world_size', 'official_model_dir', 'encoder_model_dir', 'canonical_train_sha256',
    'source_question_pool', 'probe_source_sha256', 'probe_source_epoch',
    'probe_source_model_kind', 'probe_source_run', 'fixed_test_identity_path',
    'fixed_test_identity_sha256',
    'fixed_test_manifest_sha256', 'grounding_probe_version', 'actor_hidden_native',
    'hidden_capture_contract', 'nonmixed_advantage_mode', 'actual_em_selection_fallback',
)


def validate_boundary(config, source_config, joint, actor, probe, pool, reservation):
    """Prove the asymmetric saved boundary without calling it complete joint10."""
    spec = config['resume_pending_probe']
    n, previous = spec['pending_outer'], spec['completed_joint_outer']
    if (type(n) is not int or n != previous + 1 or not is_probe_update(n)
            or config.get('resume_joint') or spec['source_commit'] != source_config['source_commit']
            or config['source_commit'] == source_config['source_commit']
            or spec['source_execution_id'] != source_config['execution_id']
            or config['execution_id'] == source_config['execution_id']
            or any(config.get(k) != source_config.get(k) for k in SCIENCE_KEYS)):
        raise ValueError('Pending-probe recovery must preserve science in a new source/execution')
    before = probe_boundary(previous)
    if (joint['run_id'] != config['run_id'] or joint['outer'] != previous
            or joint['source_commit'] != spec['source_commit']
            or joint['actor']['completed_outer'] != previous
            or joint['actor']['actual_adam_steps_per_rank'] != [20*previous]*WORLD
            or joint['actor']['scheduler_completed_outer'] != previous
            or joint['probe'] != probe
            or probe['run_id'] != config['run_id']
            or probe['completed_outer'] != before['last_update_outer']
            or probe['actual_adam_steps_per_rank'] != [before['actual_adam_per_rank']]*WORLD
            or pool['next_outer'] != n+1
            or pool['consumed_candidates'] != n*CANDIDATE_QUESTIONS
            or pool['cycle']*len(pool['order'])+pool['cursor'] != n*CANDIDATE_QUESTIONS
            or reservation['outer'] != n or reservation['after'] != pool):
        raise ValueError('Require joint9, completed actor10, unchanged probe0, and reserved pool10')
    # Actor restoration is the exact existing actor/Adam/scheduler contract.  It
    # is intentionally validated as the parent of actor11, not rerun as actor10.
    restore_metadata(config, actor, n+1)
    if actor['new_adam_steps_per_rank'] != [20]*WORLD:
        raise ValueError('Pending outer actor must have all20 real returned Adam writes')
    return n


def validate_qualification(config, qualified):
    """Consume a full-SHA CPU seal, retaining its explicit non-CUDA scope."""
    spec = config['resume_pending_probe']; n = spec['pending_outer']
    before = probe_boundary(n-1)
    expected = dict(status='QUALIFIED_NATIVE_PENDING_PROBE', run_id=config['run_id'],
        source_base=spec['source_base'], source_execution_id=spec['source_execution_id'],
        source_commit=spec['source_commit'], source_config_sha256=spec['source_config_sha256'],
        completed_joint_outer=spec['completed_joint_outer'], pending_outer=n,
        actor_completed_outer=n, actor_adam_per_rank=[20*n]*WORLD,
        scheduler_completed_outer=n, probe_completed_outer=before['last_update_outer'],
        probe_adam_per_rank=[before['actual_adam_per_rank']]*WORLD,
        pool_next_outer=n+1, pool_consumed_candidates=n*CANDIDATE_QUESTIONS,
        probe_update_new_adam_per_rank=[0]*WORLD, owned_births_all_gone=True,
        terminal_proof_sha256=spec['terminal_proof_sha256'])
    if any(qualified.get(k) != value for k, value in expected.items()):
        raise ValueError('CPU pending-probe qualification differs from this exact asymmetric boundary')
    rows = qualified.get('verified_files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Full-SHA CPU qualification must seal the actual saved state and cached inputs')
    seen = set()
    for row in rows:
        path = Path(row['path'])
        if (str(path) in seen or path.is_symlink() or not path.is_file()
                or path.resolve(strict=True) != path or not str(path).startswith('/data/')):
            raise ValueError('CPU-qualified assets must be unique canonical remote files')
        seen.add(str(path)); info = path.stat()
        if ((info.st_size, info.st_ino, info.st_mtime_ns) !=
                (row['bytes'], row['inode'], row['mtime_ns'])
                or ('device' in row and info.st_dev != row['device'])
                or ('dev' in row and info.st_dev != row['dev'])
                or len(row['sha256']) != 64):
            raise ValueError('CPU-sealed state or cached sampling input changed after qualification')
    source = Path(spec['source_base'])
    mandatory = ['preparation/launch-config.json', 'latest-joint.json', 'actor/latest.json',
                 'probe/latest.json', 'probe/latest.pt', 'probe/initialization-complete.json',
                 'question-pool/binding.json', 'question-pool/state.json']
    root = source/'outers'/('outer-%06d'%n)
    required = {str(source/name) for name in mandatory} | {
        str(root/name) for name in ('pool-reservation.json', 'actor-binding.json',
            'probe-score/manifest.json', 'probe-score/selection.json',
            'probe-score/actual-em-selection.json', 'full/complete.json',
            'full-pool/complete.json', 'full-pool/own-em.jsonl', 'diagnostic-complete.json')}
    required |= {str(source/'outers'/('outer-%06d'%outer)/'complete.json')
                 for outer in range(1, n)}
    for outer in (5, 50, 100, 150):
        old_qa = source/'outers'/('outer-%06d'%outer)/'validation/complete.json'
        if outer < n and old_qa.is_file():
            required.add(str(old_qa))
    if not required <= seen:
        raise ValueError('CPU seal omits required pending-boundary state or immutable input metadata')
    return {row['path']: row for row in rows}


def _source_checks(base, config):
    spec = config['resume_pending_probe']; source = Path(spec['source_base'])
    if (source == base or source.parent != base.parent or source.resolve(strict=True) != source
            or not source.name.startswith('search-g1-prefix-selection200-')):
        raise ValueError('Pending recovery requires a distinct canonical execution namespace')
    if (sha(source/'preparation/launch-config.json') != spec['source_config_sha256']
            or sha(spec['terminal_proof']) != spec['terminal_proof_sha256']
            or sha(spec['qualification_proof']) != spec['qualification_proof_sha256']):
        raise ValueError('Pending recovery configuration/actual terminal/CPU qualification seal changed')
    qualified = read(spec['qualification_proof'])
    sealed = validate_qualification(config, qualified)
    return spec, source, qualified, sealed


def validate_imported_pending(base, config):
    base = Path(base); spec, source, qualified, sealed = _source_checks(base, config)
    receipt = read(base/'preparation/resume-pending-probe-receipt.json')
    n = validate_boundary(config, read(source/'preparation/launch-config.json'),
        read(base/'latest-joint.json'), read(base/'actor/latest.json'),
        read(base/'probe/latest.json'), read(base/'question-pool/state.json'),
        read(source/'outers'/('outer-%06d'%spec['pending_outer'])/'pool-reservation.json'))
    if (receipt['run_id'] != config['run_id'] or receipt['execution_id'] != config['execution_id']
            or receipt['source_commit'] != config['source_commit']
            or receipt['qualification_proof_sha256'] != spec['qualification_proof_sha256']
            or receipt['pending_outer'] != n or receipt['joint_completed_outer'] != n-1
            or receipt['joint_sha256'] != sha(base/'latest-joint.json')
            or receipt['probe_inode'] != (base/'probe/latest.pt').stat().st_ino
            or (base/'outers'/('outer-%06d'%n)/'probe-update').exists()):
        raise ValueError('Imported pending boundary changed or already contains a probe-update attempt')
    return n-1


def import_pending(base):
    base = Path(base); config = read(base/'preparation/launch-config.json')
    receipt = base/'preparation/resume-pending-probe-receipt.json'
    intent = base/'preparation/resume-pending-probe-intent.json'
    if receipt.exists():
        validate_imported_pending(base, config)
        return dict(status='EXISTING_PENDING_PROBE_IMPORT_NO_REPLAY', receipt=read(receipt))
    if intent.exists():
        raise RuntimeError('Pending-probe import intent exists without receipt; reconcile, never replay')
    spec, source, qualified, sealed = _source_checks(base, config)
    n = validate_boundary(config, read(source/'preparation/launch-config.json'),
        read(source/'latest-joint.json'), read(source/'actor/latest.json'),
        read(source/'probe/latest.json'), read(source/'question-pool/state.json'),
        read(source/'outers'/('outer-%06d'%spec['pending_outer'])/'pool-reservation.json'))
    old_root = source/'outers'/('outer-%06d'%n)
    if (old_root/'complete.json').exists() or (old_root/'probe-update').exists():
        raise ValueError('Only the actually failed before-write probe boundary may use this recovery')
    once(intent, dict(run_id=config['run_id'], execution_id=config['execution_id'],
        pending_outer=n, source_base=str(source), started_unix=time.time(),
        qualification_proof_sha256=spec['qualification_proof_sha256']))
    assets, inherited_metadata = [], []
    def link(src, dst):
        src, dst = Path(src), Path(dst)
        if src.is_symlink() or not src.is_file():
            raise ValueError('Imported state must be real immutable files')
        dst.parent.mkdir(parents=True, exist_ok=True); os.link(src, dst)
        row = dict(source=str(src), target=str(dst), bytes=src.stat().st_size,
                   inode=src.stat().st_ino)
        if str(src) in sealed:
            row['sha256'] = sealed[str(src)]['sha256']
            if src.suffix == '.json' and sha(src) != row['sha256']:
                raise ValueError('Imported metadata differs from its CPU full-SHA qualification')
        assets.append(row)
        return row
    for name in ('actor/latest.json', 'probe/initialization-complete.json', 'probe/latest.json',
                 'probe/latest.pt', 'question-pool/binding.json', 'question-pool/state.json',
                 'latest-joint.json'):
        link(source/name, base/name)
    for outer in range(1, n):
        old = source/'outers'/('outer-%06d'%outer)
        completed = read(old/'complete.json')
        expected_probe = probe_boundary(outer)
        if (completed['run_id'] != config['run_id'] or completed['outer'] != outer
                or completed['source_commit'] != spec['source_commit']
                or completed['actor']['completed_outer'] != outer
                or completed['actor']['actual_adam_steps_per_rank'] != [20*outer]*WORLD
                or completed['actor']['scheduler_completed_outer'] != outer
                or completed['probe']['completed_outer'] != expected_probe['last_update_outer']
                or completed['probe']['actual_adam_steps_per_rank'] != [expected_probe['actual_adam_per_rank']]*WORLD):
            raise ValueError('Inherited historical joint source/run/counters differ from the explicit parent chain')
        for name in ('complete.json', 'probe-frozen.json', 'probe-update/metrics.json'):
            if (old/name).is_file():
                row = link(old/name, base/'outers'/old.name/name)
                if name == 'complete.json':
                    inherited_metadata.append(dict(row, run_id=config['run_id'], outer=outer,
                        source_commit=spec['source_commit'], role='inherited_complete_joint'))
        # Only the existing small QA receipt is inherited.  Its original output
        # shards and sampling/evaluation records remain at their immutable paths.
        if (old/'validation/complete.json').is_file():
            row = link(old/'validation/complete.json', base/'outers'/old.name/'validation/complete.json')
            inherited_metadata.append(dict(row, run_id=config['run_id'], outer=outer,
                source_commit=spec['source_commit'], role='inherited_existing_qa_metadata'))
    # Keep only tiny metadata aliases in this namespace; all original tensor,
    # rollout, hidden-feature and token payloads remain at their sealed paths.
    for name in ('pool-reservation.json', 'selection.json', 'actor-binding.json',
                 'prefix/complete.json', 'probe-score/complete.json', 'probe-score/manifest.json',
                 'probe-score/selection.json', 'probe-score/actual-em-selection.json',
                 'full/complete.json', 'full-pool/complete.json', 'diagnostic-complete.json'):
        if (old_root/name).is_file():
            link(old_root/name, base/'outers'/old_root.name/name)
    result = dict(status='COMMITTED_ACTOR_PENDING_PROBE_IMPORTED_NO_RESET',
        run_id=config['run_id'], execution_id=config['execution_id'], source_commit=config['source_commit'],
        source_base=str(source), source_execution_id=spec['source_execution_id'],
        parent_source_commit=spec['source_commit'], joint_completed_outer=n-1, pending_outer=n,
        source_inputs_root=str(old_root), joint_sha256=sha(base/'latest-joint.json'),
        probe_inode=(base/'probe/latest.pt').stat().st_ino,
        qualification_proof_sha256=spec['qualification_proof_sha256'],
        actor_adam_per_rank=[20*n]*WORLD, scheduler_completed_outer=n,
        probe_adam_per_rank=read(base/'probe/latest.json')['actual_adam_steps_per_rank'],
        actor_updates_to_reexecute=0, candidates_to_resample=0, outcomes_to_rescore=0,
        independent_encoder_forwards=0, probe_initialized_again=False,
        full_state_import=True, cuda_restore_verified=False,
        source_modified=False, optimizer_reset=False, assets=assets,
        inherited_metadata_files=inherited_metadata)
    once(receipt, result)
    validate_imported_pending(base, config)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(import_pending(args.base), allow_nan=False))


if __name__ == '__main__':
    main()
