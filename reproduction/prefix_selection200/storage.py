"""Delete only consumed, run-owned rotating tensors after an actual joint commit.

Producer hashes are retained as provenance, never reported as a new readback.
Tokens, outcomes, trajectories, questions, manifests, metrics and milestone
checkpoints remain. A pending deletion intent is reconciled, never replayed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from .contract import WORLD, PROBE_PERIOD, PROBE_UPDATES, G
from .candidate_schedule import candidate_count


REMOTE_PARENT = Path('/mnt/data/shaoshitong')
A800_REMOTE_PARENT = Path('/data/shaoshitong')
PREFIX = 'search-g1-prefix-selection200-'
MILESTONES = {0, 5, 50, 100, 150, 200}
HPC3_BASE = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')


def hpc3_namespace(base):
    """Admit only this experiment and numbered sibling recovery executions."""
    return (base.parent == HPC3_BASE.parent and
            re.fullmatch(re.escape(HPC3_BASE.name)+r'(?:-r[1-9][0-9]*)?',base.name) is not None)


def read(path):
    return json.loads(Path(path).read_text())


def small_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def owned(base, path):
    """Reject lexical traversal, symlink aliases and paths outside this run."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or base not in path.parents:
        raise ValueError('Deletion asset is outside the current run namespace: '+str(path))
    cursor = path
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError('Deletion asset or ancestor is a symlink: '+str(cursor))
        cursor = cursor.parent
    if path.resolve(strict=False) != path:
        raise ValueError('Deletion asset is not its canonical run-owned path')
    return path


def validate_base(base):
    base = Path(base)
    hpc3 = hpc3_namespace(base)
    if (not base.is_absolute() or '..' in base.parts or (not hpc3 and (base.parent not in (REMOTE_PARENT, A800_REMOTE_PARENT)
            or not base.name.startswith(PREFIX))) or base.resolve(strict=True) != base):
        raise ValueError('Pruning is restricted to a canonical new prefix-selection200 namespace')
    config = read(owned(base, base/'preparation/launch-config.json'))
    native=config.get('actor_hidden_native',False)
    clock_ok=(config.get('probe_update_period')==PROBE_PERIOD and config.get('probe_updates_per_period')==PROBE_UPDATES
              and config.get('probe_updates_per_outer')==0 and config.get('probe_update_at_final') is False) if native else config.get('probe_updates_per_outer')==10
    if (config['base'] != str(base) or not config['run_id'] or config['outers'] != 200
            or config['world_size'] != WORLD or not clock_ok):
        raise ValueError('Current namespace does not have the approved actor/probe run binding')
    return base, config


def paths_in(value, base):
    if isinstance(value, dict):
        return {p for item in value.values() for p in paths_in(item, base)}
    if isinstance(value, list):
        return {p for item in value for p in paths_in(item, base)}
    if isinstance(value, str) and value.startswith(str(base)+'/'):
        path = Path(value)
        # Generic run containers describe the namespace, not a retained asset.
        if path not in (base/'actor', base/'actor/outers', base/'probe', base/'outers'):
            return {owned(base, path)}
    return set()


def related(path, references):
    return any(path == ref or path in ref.parents or ref in path.parents for ref in references)



def actual_actor_boundary(actor, outer):
    """Validate a complete20-opportunity receipt using real retained Adam writes."""
    from .actor import valid_actual_steps
    try:
        actual = actor['actual_adam_steps_per_rank']
        new = actor['new_adam_steps_per_rank']
        skipped = actor['skipped_update_opportunities']
        receipts = actor['rank_receipts']
        if (not valid_actual_steps(actual, outer) or not isinstance(new, list) or len(new) != WORLD or
                any(type(count) is not int or not 1 <= count <= 20 for count in new) or len(set(new)) != 1 or
                not isinstance(skipped, list) or skipped != [20 - count for count in new] or
                any(type(count) is not int for count in skipped) or
                not valid_actual_steps([actual[rank] - new[rank] for rank in range(WORLD)], outer - 1) or
                actor['opportunities_processed'] != 20 or type(actor['opportunities_processed']) is not int or
                actor['complete_outer'] is not True or actor['status'] != 'complete' or
                actor['full_budget_complete'] is not (new[0] == 20) or
                actor['automatic_rollback_performed'] is not False or
                len(receipts) != WORLD or {row['rank'] for row in receipts} != set(range(WORLD))):
            return False
        for row in receipts:
            rank = row['rank']; statuses = row['opportunity_statuses']
            if (row['actual_adam_steps'] != actual[rank] or row['new_adam_steps'] != new[rank] or
                    row['optimizer_writes_before'] != actual[rank] - new[rank] or
                    row['optimizer_writes_after'] != actual[rank] or row['optimizer_writes_this_attempt'] != new[rank] or
                    row['opportunities_processed'] != 20 or row['opportunities'] != 20 or
                    row['skipped_update_opportunities'] != skipped[rank] or
                    row['complete_outer'] is not True or row['full_budget_complete'] is not (new[rank] == 20) or
                    row['automatic_rollback_performed'] is not False or not isinstance(statuses, list) or len(statuses) != 20 or
                    any(status not in ('written', 'written_with_post_guard', 'skipped_guard') for status in statuses) or
                    statuses.count('skipped_guard') != skipped[rank] or
                    statuses.count('written_with_post_guard') != row['written_with_post_guard']):
                return False
        return True
    except (KeyError, TypeError, IndexError):
        return False

def current_boundary(base, config, outer):
    root = base/'outers'/('outer-%06d'%outer)
    joint = read(owned(base, root/'complete.json'))
    latest_joint = read(owned(base, base/'latest-joint.json'))
    actor = read(owned(base, base/'actor/latest.json'))
    probe = read(owned(base, base/'probe/latest.json'))
    native=config.get('actor_hidden_native',False)
    if native:
        from .periodic import is_probe_update,probe_boundary
        periodic=is_probe_update(outer);boundary=probe_boundary(outer)
    else:
        periodic=True;boundary=dict(last_update_outer=outer,actual_adam_per_rank=10*outer)
    update=read(owned(base,root/('probe-update/complete.json' if periodic else 'probe-frozen.json')))
    if (joint != latest_joint or joint['run_id'] != config['run_id'] or joint['outer'] != outer
            or joint['source_commit'] != config['source_commit']
            or joint['actor'] != actor or joint['probe'] != probe
            or actor['run_id'] != config['run_id'] or probe['run_id'] != config['run_id']
            or actor['completed_outer'] != outer or probe['completed_outer'] != boundary['last_update_outer']
            or not actual_actor_boundary(actor, outer)
            or probe['actual_adam_steps_per_rank'] != [boundary['actual_adam_per_rank']]*WORLD
            or actor['scheduler_completed_outer'] != outer
            or update['status'] != ('UNSELECTED2560G5_PERIODIC_PROBE_50_ADAM_COMPLETE' if native and periodic
                else 'PROBE_FROZEN_NO_OPTIMIZER_UPDATE' if native else 'SELECTED512G5_ONLINE_PROBE_10_ADAM_COMPLETE')
            or (periodic and update['run_id'] != config['run_id']) or update['outer'] != outer
            or (periodic and update['checkpoint'] != probe)
            or update['new_adam_steps_per_rank'] != (PROBE_UPDATES if native and periodic else 0 if native else 10)
            or update['actual_adam_steps_per_rank'] != [boundary['actual_adam_per_rank']]*WORLD):
        raise ValueError('Actual actor20-opportunity/probe boundaries and this outer joint commit must agree')
    ledger = read(owned(base, base/'logs/process-ledger.json'))
    if ledger['run_id'] != config['run_id']:
        raise ValueError('Process ownership ledger belongs to another run')
    active = [name for name, row in ledger['jobs'].items()
              if re.match(r'outer-\d{6}-', name) and row.get('state') != 'STOPPED']
    if active:
        raise RuntimeError('A run-owned outer phase is still active: '+', '.join(active))
    snapshot = owned(base, joint['probe_snapshot'])
    latest_probe = owned(base, base/'probe/latest.pt')
    if (snapshot.stat().st_dev, snapshot.stat().st_ino) != (
            latest_probe.stat().st_dev, latest_probe.stat().st_ino):
        raise ValueError('Joint probe snapshot is not the actually committed latest state')
    references = paths_in([joint, actor, probe], base) | {latest_probe, snapshot}
    for milestone in sorted(MILESTONES-{0}):
        path = base/'outers'/('outer-%06d'%milestone)/'complete.json'
        if path.is_file():
            saved = read(owned(base, path))
            if saved['run_id'] == config['run_id']:
                # Preserve the milestone's resumable actor/probe state. Its
                # historical sampling-input actor is lineage metadata, not a
                # second milestone checkpoint requiring permanent retention.
                references |= paths_in([saved['actor'], saved['probe'], saved['probe_snapshot']], base)
    return root, joint, update, references


def planned_assets(base, config, outer, root, joint, update, references):
    plan = {}

    def add(path, producer, role):
        path = owned(base, path)
        if related(path, references):
            return
        if not path.exists():
            return
        info = path.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_size != producer['bytes']
                or not re.fullmatch(r'[0-9a-f]{64}', producer['sha256'])):
            raise ValueError('Declared consumed tensor size/type/provenance differs: '+str(path))
        plan[str(path)] = dict(path=str(path), role=role, producer_sha256=producer['sha256'],
            producer_bytes=producer['bytes'], sha256_source='producer_receipt_not_rehashed',
            dev=info.st_dev, inode=info.st_ino, bytes=info.st_size, links_before=info.st_nlink,
            contents_read_or_rehashed=False)

    score = root/'probe-score'
    manifest_path = owned(base, score/'manifest.json')
    manifest = read(manifest_path)
    if config.get('actor_hidden_native'):
        if (manifest['role']!='current_actor_native_hidden_features' or manifest['run_id']!=config['run_id']
                or manifest['outer']!=outer or manifest['count']!=candidate_count(config,outer)*G or manifest['independent_encoder_forward']):
            raise ValueError('Consumed native features do not belong to this completed outer')
        scans=[root/'prefix/scan/scan-complete.json']
        diag=root/'diagnostic-complete.json'
        if diag.exists():scans.append(Path(read(diag)['scan_path'])/'scan-complete.json')
        for scan in scans:
            proof=read(owned(base,scan))
            if not proof['actor_hidden_native'] or proof['run_id']!=config['run_id']:
                raise ValueError('Do not delete another run feature shard')
            for item in proof['actor_hidden_shards']:
                add(item['path'],item,'consumed_current_actor_native_hidden')
    else:
        if (manifest['role'] != 'selected_actual_prefix_official_encoder_features'
                or manifest['run_id'] != config['run_id'] or manifest['outer'] != outer
                or manifest['count'] != 2560
                or update['scored_features_manifest_sha256'] != small_sha(manifest_path)
                or update['selection_sha256'] != manifest['selection_sha256']
                or {row['rank'] for row in manifest['shards']} != set(range(WORLD))
                or sum(row['count'] for row in manifest['shards']) != 2560):
            raise ValueError('Current feature shards are not bound to the completed online update')
        for row in manifest['shards']:
            if row['file'] != 'selected-features-rank-%d.pt'%row['rank']:
                raise ValueError('Only declared selected-prefix feature tensors may be pruned')
            add(score/row['file'], row, 'consumed_selected_official_encoder_features')
    previous = outer-1
    # Imported history keeps authentic SOURCE absolute paths. It is outside
    # this execution namespace and must never be pruned as a local snapshot.
    if previous <= config.get('resume_joint',{}).get('completed_outer',0):
        previous=0
    if previous > 0 and previous not in MILESTONES:
        old_root = base/'outers'/('outer-%06d'%previous)
        old_complete = old_root/'complete.json'
        if old_complete.is_file():
            saved = read(owned(base, old_complete))
            if saved['run_id'] == config['run_id'] and saved['outer'] == previous:
                expected = old_root/'probe-state-at-completion.pt'
                if saved['probe_snapshot'] != str(expected):
                    raise ValueError('Previous probe snapshot path differs from its owned outer')
                from .periodic import probe_boundary
                expected_steps=(probe_boundary(previous)['actual_adam_per_rank'] if config.get('actor_hidden_native') else 10*previous)
                if saved['probe']['actual_adam_steps_per_rank'] != [expected_steps]*WORLD:
                    raise ValueError('Previous probe producer is not a complete actual boundary')
                add(expected, saved['probe'], 'previous_nonmilestone_probe_snapshot')

    actors = base/'actor/outers'
    if actors.is_dir():
        for attempt in sorted(actors.iterdir()):
            match = re.fullmatch(r'outer-(\d{4})-attempt-[0-9a-f]+', attempt.name)
            if not match:
                continue
            owned(base, attempt)
            complete = attempt/'complete.json'
            if not complete.is_file():
                continue  # Failed or incomplete attempts retain their evidence/state.
            proof = read(owned(base, complete))
            n = int(match[1])
            if (proof['run_id'] != config['run_id'] or n >= outer or n in MILESTONES
                    or proof.get('milestone') or related(attempt, references)):
                continue
            if (proof['role'] != 'prefix_selection200_actor_complete' or proof['completed_outer'] != n
                    or not actual_actor_boundary(proof, n)
                    or proof['scheduler_completed_outer'] != n
                    or proof['state_path'] != str(attempt) or proof['state_dir'] != str(attempt)):
                raise ValueError('Rotating actor checkpoint does not match its own complete producer')
            hfdir = attempt/'huggingface'
            if any(proof[key] != str(hfdir) for key in ('model_dir', 'hf_model_dir', 'actor_model_dir')):
                raise ValueError('Actor HF aliases no longer bind the run-owned state directory')
            for name, producer in proof['hf_files'].items():
                if re.fullmatch(r'(model(?:-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin)', name):
                    add(hfdir/name, producer, 'unreferenced_nonmilestone_actor_hf_weight')
            states = proof['rank_states']
            if len(states) != WORLD or {Path(row['path']).name for row in states} != {'rank-%d.pt'%r for r in range(WORLD)}:
                raise ValueError('Actor producer must describe all four rank optimizer/RNG states')
            for producer in states:
                path = owned(base, producer['path'])
                if path.parent != attempt:
                    raise ValueError('Actor rank state escapes its own completed attempt')
                add(path, producer, 'unreferenced_nonmilestone_actor_rank_state')
            driver = proof['driver_state']
            if driver['path'] != str(attempt/'driver.pt'):
                raise ValueError('Actor driver state escapes its own completed attempt')
            add(attempt/'driver.pt', driver, 'unreferenced_nonmilestone_actor_driver_state')
    return list(plan.values()), small_sha(manifest_path)


def prune_completed(base, outer):
    base, config = validate_base(base)
    if type(outer) is not int or not 1 <= outer <= 200:
        raise ValueError('Prune only an actual authorized outer1..200')
    root = base/'outers'/('outer-%06d'%outer)
    result_path = root/'storage-pruned.json'
    intent_path = root/'storage-prune-intent.json'
    with (base/'preparation/storage-prune.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if result_path.exists():
            existing = read(owned(base, result_path))
            if existing['run_id'] != config['run_id'] or existing['outer'] != outer:
                raise ValueError('Existing deletion receipt belongs to another run/outer')
            return dict(status='EXISTING_STORAGE_PRUNE_RECEIPT_NO_REPLAY', receipt=existing)
        if intent_path.exists():
            return dict(status='EXISTING_STORAGE_PRUNE_INTENT_RECONCILE_REQUIRED',
                        intent_path=str(intent_path), deleted_this_call=0)
        root, joint, update, references = current_boundary(base, config, outer)
        plan, feature_manifest_sha = planned_assets(base, config, outer, root, joint, update, references)
        once(intent_path, dict(run_id=config['run_id'], outer=outer, assets=plan,
            joint_complete_sha256=small_sha(root/'complete.json'),
            consumed_features_manifest_sha256=feature_manifest_sha, started_unix=time.time(),
            large_tensor_hash_repeated=False))
        journal = root/'storage-deletions.jsonl'
        deleted = []
        for row in plan:
            path = owned(base, row['path'])
            info = path.stat()
            if (info.st_dev, info.st_ino, info.st_size) != (row['dev'], row['inode'], row['bytes']):
                raise ValueError('Run-owned tensor inode changed after deletion planning')
            with journal.open('a') as handle:
                handle.write(json.dumps(dict(row, action='unlink_intent'), sort_keys=True)+'\n')
                handle.flush(); os.fsync(handle.fileno())
            path.unlink()
            entry = dict(row, action='unlinked', estimated_reclaimed_bytes=info.st_size if info.st_nlink == 1 else 0)
            deleted.append(entry)
            with journal.open('a') as handle:
                handle.write(json.dumps(entry, sort_keys=True)+'\n')
                handle.flush(); os.fsync(handle.fileno())
        result = dict(status='COMPLETED_RUN_OWNED_ROTATING_TENSORS_PRUNED', run_id=config['run_id'],
            outer=outer, deleted=deleted, deleted_paths=len(deleted),
            logical_bytes_unlinked=sum(row['bytes'] for row in deleted),
            estimated_reclaimed_bytes=sum(row['estimated_reclaimed_bytes'] for row in deleted),
            milestone_checkpoints_preserved=sorted(MILESTONES),
            current_latest_and_joint_references_preserved=True,
            prefix_tokens_own_em_full_trajectories_questions_metrics_manifests_preserved=True,
            old_runs_and_original_source_data_touched=False, tensor_contents_read_or_rehashed=False,
            completed_unix=time.time())
        once(result_path, result)
        return result
