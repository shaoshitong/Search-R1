"""Stage an isolated continuation from a completed head and reserved questions.

Runs on the target node before submission. Original runtime, head, pool,
partial rollouts and logs remain unchanged. New latest checkpoint publication
uses atomic replacement, so an initial hard link cannot mutate the source file.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def hardlink_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return 'hardlink_immutable_until_atomic_latest_replacement'
    except OSError as error:
        import errno
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise ValueError('Remote checkpoint copy differs')
        return 'remote_copy_full_sha_matched'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def binding_sha(binding):
    return hashlib.sha256(canonical(binding)).hexdigest()


def learning_rate_transition(source_config, target_config, source_claim, latest):
    """Authorize exactly the requested LR cut while preserving checkpoint bytes."""
    restore = target_config['restore']
    before, after = float(source_config['learning_rate']), float(target_config['learning_rate'])
    if before == after:
        return None
    if (restore.get('allow_learning_rate_change') is not True
            or restore.get('source_learning_rate') != before
            or restore.get('target_learning_rate') != after
            or before != 3e-4 or after != 3e-5):
        raise ValueError('Only the explicitly authorized3e-4to3e-5 LR change is permitted')
    source_binding = source_claim['binding']
    source_sha = binding_sha(source_binding)
    if source_claim['binding_sha256'] != source_sha or latest['binding_sha256'] != source_sha:
        raise ValueError('Source claim and completed head binding differ')
    if source_binding['learning_rate'] != before:
        raise ValueError('Source head binding LR differs from source launch contract')
    target_binding = dict(source_binding, learning_rate=after)
    return dict(schema_version=1, source_binding=source_binding, target_binding=target_binding,
                source_binding_sha256=source_sha, target_binding_sha256=binding_sha(target_binding),
                source_learning_rate=before, target_learning_rate=after,
                source_complete_epoch=int(latest['epoch']), start_epoch=int(latest['epoch'])+1,
                run_id=target_config['run_id'], authorized_change_only='learning_rate',
                source_checkpoint_sha256=latest['sha256'], optimizer_moments_preserved=True,
                actual_adam_steps_preserved=latest['actual_adam_steps_per_rank'], rank_rng_preserved=True)


def validate_pool_boundary(pool, head_state, live_state, pending_selection, completed_epoch, pool_sha256):
    """Reconstruct the one possible pending selection from the complete boundary."""
    from reproduction.prefix20_online.pool import next_selection
    start = completed_epoch+1
    if head_state['next_epoch'] != start:
        raise ValueError('Completed head pool snapshot points to a different next epoch')
    if head_state['consumed_questions'] != completed_epoch*head_state['epoch_questions']:
        raise ValueError('Completed head pool snapshot has a different consumed question count')
    if pending_selection is None:
        if live_state != head_state:
            raise ValueError('Question pool advanced without its immutable pending selection')
        return dict(pending_selection_reused=False, pool_phase='complete_head_boundary')
    expected, after = next_selection(pool, head_state, start)
    expected['pool_sha256'] = pool_sha256
    if pending_selection != expected:
        raise ValueError('Pending questions/order differ from the complete head pool boundary')
    if live_state == after:
        phase = 'reserved_selection_and_pool_state_published'
    elif live_state == head_state:
        phase = 'reserved_selection_published_before_pool_state'
    else:
        raise ValueError('Live pool permutation/cursor differs from the exact pending selection')
    return dict(pending_selection_reused=True, pool_phase=phase,
                pending_question_order_sha256=expected['question_order_sha256'])


def uncommitted_head_updates(train, start_epoch, completed_steps):
    """Report observed discarded writes; never add them to restored Adam counts."""
    attempt = Path(train)/'epochs'/('epoch-%04d'%start_epoch)
    if (attempt/'complete.json').exists():
        raise ValueError('A newer complete head exists; independently verify and restore it instead')
    observed = [0]*8
    for rank in range(8):
        path = attempt/('rank-%d-updates.jsonl'%rank)
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row['epoch'] != start_epoch or row['actual_adam_steps'] != completed_steps+observed[rank]+1:
                raise ValueError('Uncommitted rank update evidence has inconsistent epoch/Adam ordering')
            observed[rank] += 1
    artifacts = [dict(path=str(path), bytes=path.stat().st_size) for path in Path(train).iterdir()
                 if path.is_file() and ('.partial' in path.name or '.tmp-' in path.name)]
    return dict(interrupted_epoch=start_epoch, observed_updates_per_rank=observed,
                source_uncommitted_artifacts=artifacts,
                unlogged_last_write_not_inferred=True, added_to_restored_adam_count=False)


def stage(config, *, terminal=None):
    """Copy only this stopped source's exact completed/reserved boundaries.

    ``terminal`` is injectable for CPU fixtures. Native deployments query the
    actual source job and require both a terminal state and owned cleanup.
    """
    from reproduction.common8.native_allocation import fields
    base = Path(config['base']).resolve()
    restore = config['restore']
    source = Path(restore['source_base']).resolve()
    if source == base or (terminal is None and not str(source).startswith('/mnt/data/shaoshitong/search-g1-prefix20-online1000-')):
        raise ValueError('Recovery needs a distinct experiment namespace')
    prep = base/'preparation'
    intent = prep/'restore-intent.json'
    if intent.exists():
        if (prep/'restore-receipt.json').exists():
            receipt = read(prep/'restore-receipt.json')
            if receipt['source_base'] != str(source) or receipt['head_sha256'] != restore['checkpoint_sha256']:
                raise ValueError('Existing restore receipt differs; no replay')
            transition = receipt.get('learning_rate_transition')
            restored_lr = transition['target_learning_rate'] if transition else read(source/'preparation/launch-config.json')['learning_rate']
            if config['learning_rate'] != restored_lr:
                raise ValueError('Existing restore receipt has a different target learning rate; no replay')
            config['start_epoch'] = receipt['start_epoch']
            return receipt
        raise RuntimeError('Existing partial restore intent requires inspection, not replay')
    source_config = read(source/'preparation/launch-config.json')
    for key in ('run_id','model_revision','model_files_sha256','canonical_train_sha256',
                'cached_features_manifest_sha256','seed','global_batch','epochs','save_every'):
        if source_config[key] != config[key]:
            raise ValueError('Recovery changed fixed training contract: '+key)
    submit = read(source/'preparation/submit-receipt.json')
    job_id = str(submit['job_id'])
    if restore.get('source_job_id') is not None and str(restore['source_job_id']) != job_id:
        raise ValueError('Recovery source job differs from the authorized stopped job')
    if terminal is None:
        if restore.get('terminal_evidence_path'):
            evidence = Path(restore['terminal_evidence_path'])
            if evidence.resolve().parent != source/'preparation' or sha(evidence) != restore['terminal_evidence_sha256']:
                raise ValueError('Pinned source terminal evidence differs')
            saved = read(evidence)
            raw = saved.get('native') or saved.get('facts',{}).get('native')
            if not isinstance(raw,str):
                raise ValueError('Pinned terminal evidence lacks the original native record')
        else:
            raw = subprocess.check_output(['scontrol','show','job',job_id,'-o'],
                                          text=True, stderr=subprocess.PIPE, timeout=30)
        terminal = fields(raw)
    if (str(terminal.get('JobId')) != job_id or
            terminal.get('JobState') not in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','COMPLETED')):
        raise ValueError('Source must be an actually terminal native job')
    cleanup = read(source/'logs/cleanup-status.json')
    if terminal.get('Comment') is not None and terminal['Comment'] != 'g1-prefix20-online:'+config['run_id']:
        raise ValueError('Source native job belongs to a different logical run')
    if terminal.get('Command') is not None and terminal['Command'] != str(source/'preparation/native-launch.sh'):
        raise ValueError('Source native launcher path differs')
    if cleanup.get('cleaned') is not True:
        raise ValueError('Source owned cleanup has not completed')
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    ledger = read(source/'logs/process-ledger.json')
    births = [row for job in ledger['jobs'].values() for row in job['ledger'].values()]
    actual_present = birth_states(births)
    if actual_present:
        raise ValueError('Source original owned process births remain present')
    cpu_proof_path = Path(restore['checkpoint_cpu_proof_path'])
    if sha(cpu_proof_path) != restore['checkpoint_cpu_proof_sha256']:
        raise ValueError('Independent CPU recovery proof bytes differ')
    cpu_proof = read(cpu_proof_path)
    old_train = source/'train'
    latest = read(old_train/'latest.json')
    completed = int(latest['epoch'])
    start = completed+1
    transition = learning_rate_transition(source_config, config, read(old_train/'run-claim.json'), latest)
    if completed != int(restore['completed_epoch']) or latest['sha256'] != restore['checkpoint_sha256']:
        raise ValueError('Latest actual head differs from independently checked recovery point')
    expected = completed*((25600+int(config['global_batch'])-1)//int(config['global_batch']))
    if latest['actual_adam_steps_per_rank'] != [expected]*8:
        raise ValueError('Completed eight-rank actual Adam boundary differs')
    if (cpu_proof['full_sha256'] != restore['checkpoint_sha256'] or cpu_proof['epoch'] != completed
            or cpu_proof['actual_adam_steps_per_rank'] != [expected]*8
            or cpu_proof['rank_rng'] != list(range(8)) or cpu_proof['cpu_load_performed'] is not True):
        raise ValueError('Independent CPU proof does not verify this exact complete head')
    if sha(old_train/'latest.pt') != restore['checkpoint_sha256']:
        raise ValueError('Actual source head/Adam/RNG bytes differ')
    boundary = read(old_train/'epochs'/('epoch-%04d'%completed)/'complete.json')
    if boundary['sha256'] != latest['sha256']:
        raise ValueError('Source epoch completion does not bind latest head')
    pool = source/'question-pool'
    pool_sha256 = sha(pool/'pool.json')
    state = read(pool/'state.json')
    pending = source/'epochs'/('epoch-%04d'%start)
    selection_path = pending/'selection.json'
    pending_selection = read(selection_path) if selection_path.is_file() else None
    snapshot_path = source/'epochs'/('epoch-%04d'%completed)/'pool-state-at-head-completion.json'
    if snapshot_path.exists():
        head_state = read(snapshot_path)
    elif state['next_epoch'] == start and pending_selection is None:
        # A stop can occur after head completion but before its small pool snapshot.
        head_state = state
    else:
        raise ValueError('Advanced pool recovery requires the completed head pool snapshot')
    pool_boundary = validate_pool_boundary(read(pool/'pool.json'), head_state, state,
                                          pending_selection, completed, pool_sha256)
    discarded = uncommitted_head_updates(old_train, start, expected)
    if (any(discarded['observed_updates_per_rank']) or discarded['source_uncommitted_artifacts']) and restore.get('discard_uncommitted_head_updates') is not True:
        raise ValueError('Uncommitted head outputs need explicit latest-complete recovery authorization')
    once(intent, dict(source_base=str(source), source_job_id=job_id,
         completed_epoch=completed, start_epoch=start, head_sha256=latest['sha256'],
         source_latest_receipt_sha256=sha(old_train/'latest.json'),
         source_pool_sha256=sha(pool/'pool.json'), source_pool_state_sha256=sha(pool/'state.json'),
         source_terminal=terminal, source_original_cleanup=cleanup,
         actual_original_births_present=0, independent_cpu_proof_sha256=restore['checkpoint_cpu_proof_sha256'],
         learning_rate_transition=transition, pool_boundary=pool_boundary,
         discarded_uncommitted_head=discarded, created_unix=time.time()))
    new_train = base/'train'
    new_train.mkdir(exist_ok=False)
    copied = []
    for path in old_train.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(old_train)
        if any('.partial' in item or '.tmp-' in item for item in relative.parts):
            continue
        if relative.parts[0] == 'epochs' and int(relative.parts[1].split('-')[-1]) > completed:
            continue
        target = new_train/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.pt' and (path.name == 'latest.pt' or path.name.startswith('head-epoch-')):
            method = hardlink_or_copy(path, target)
        else:
            shutil.copyfile(path, target)
            method = 'remote_copy'
        copied.append(dict(relative_path=str(relative), bytes=path.stat().st_size, method=method))
    preserved = new_train/('head-epoch-%04d-recovery-source.pt'%completed)
    if preserved.exists():
        if sha(preserved) != latest['sha256']:
            raise ValueError('Existing copied recovery-source head differs')
    else:
        hardlink_or_copy(old_train/'latest.pt', preserved)
    if transition is not None:
        transition_path = new_train/'learning-rate-transition.json'
        if transition_path.exists():
            transition_path.unlink()  # Only the copied new-run metadata, never the source.
        once(transition_path, transition)
        claim = dict(read(old_train/'run-claim.json'), binding=transition['target_binding'],
                     binding_sha256=transition['target_binding_sha256'],
                     source_binding_sha256=transition['source_binding_sha256'],
                     learning_rate_transition_path=str(transition_path))
        partial = new_train/'run-claim.json.lr-transition.partial'
        once(partial, claim)
        os.replace(partial, new_train/'run-claim.json')
    new_pool = base/'question-pool'
    new_pool.mkdir(exist_ok=False)
    for name in ('pool.json','state.json'):
        shutil.copyfile(pool/name, new_pool/name)
    # Earlier selections/questions and pool snapshots remain reusable. No partial
    # generated shard or rank completion receipt is carried into the new runtime.
    retained = []
    for epoch in range(1, start+1):
        old_epoch = source/'epochs'/('epoch-%04d'%epoch)
        for name in ('selection.json','questions.json','pool-state-at-head-completion.json'):
            if not (old_epoch/name).is_file():
                continue
            target = base/'epochs'/old_epoch.name/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_epoch/name, target)
            retained.append(dict(epoch=epoch, name=name, sha256=sha(target)))
    if pending_selection is not None:
        selected = read(base/'epochs'/('epoch-%04d'%start)/'selection.json')
        if selected['epoch'] != start or selected['pool_sha256'] != sha(new_pool/'pool.json'):
            raise ValueError('Pending reserved questions changed during isolated recovery')
    config['start_epoch'] = start
    result = dict(status='ISOLATED_HEAD_POOL_RECOVERY_STAGED', source_base=str(source),
         source_job_id=job_id, run_id=config['run_id'], completed_epoch=completed, start_epoch=start,
         head_sha256=latest['sha256'], actual_adam_steps_per_rank=latest['actual_adam_steps_per_rank'],
         source_latest_preserved=True, original_partial_rollouts_preserved=True,
         reserved_pending_selection_reused=pending_selection is not None,
         pool_boundary=pool_boundary, discarded_uncommitted_head=discarded,
         learning_rate_transition=transition,
         pool_next_epoch=state['next_epoch'], pool_cursor=state['cursor'],
         pool_sha256=sha(new_pool/'pool.json'), pool_state_sha256=sha(new_pool/'state.json'),
         source_checkpoint_preserved_path=str(preserved), copied_training_files=copied,
         retained_question_metadata=retained, new_runtime_commit=config['source_commit'],
         source_runtime_commit=source_config['source_commit'], checked_unix=time.time())
    once(prep/'restore-receipt.json', result)
    return result
