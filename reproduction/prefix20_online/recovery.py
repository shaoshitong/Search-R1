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
            config['start_epoch'] = receipt['start_epoch']
            return receipt
        raise RuntimeError('Existing partial restore intent requires inspection, not replay')
    source_config = read(source/'preparation/launch-config.json')
    for key in ('run_id','model_revision','model_files_sha256','canonical_train_sha256',
                'cached_features_manifest_sha256','seed','global_batch','learning_rate','epochs','save_every'):
        if source_config[key] != config[key]:
            raise ValueError('Recovery changed fixed training contract: '+key)
    submit = read(source/'preparation/submit-receipt.json')
    job_id = str(submit['job_id'])
    if terminal is None:
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
    state = read(pool/'state.json')
    if state['next_epoch'] not in (start, start+1):
        raise ValueError('Question pool is not at the next or already-reserved epoch')
    pending = source/'epochs'/('epoch-%04d'%start)
    if state['next_epoch'] == start+1 and not (pending/'selection.json').is_file():
        raise ValueError('Advanced question pool requires its immutable pending selection')
    # Preserve an incomplete head phase as evidence. This helper is for a
    # failure before its first new Adam write, not for silently losing updates.
    attempt = old_train/'epochs'/('epoch-%04d'%start)
    for path in attempt.glob('rank-*-updates.jsonl') if attempt.exists() else []:
        if path.stat().st_size:
            raise ValueError('Pending epoch already wrote Adam; explicit recovery accounting required')
    once(intent, dict(source_base=str(source), source_job_id=job_id,
         completed_epoch=completed, start_epoch=start, head_sha256=latest['sha256'],
         source_latest_receipt_sha256=sha(old_train/'latest.json'),
         source_pool_sha256=sha(pool/'pool.json'), source_pool_state_sha256=sha(pool/'state.json'),
         source_terminal=terminal, source_original_cleanup=cleanup,
         actual_original_births_present=0, independent_cpu_proof_sha256=restore['checkpoint_cpu_proof_sha256'],
         created_unix=time.time()))
    new_train = base/'train'
    new_train.mkdir(exist_ok=False)
    copied = []
    for path in old_train.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(old_train)
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
    hardlink_or_copy(old_train/'latest.pt', preserved)
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
    if state['next_epoch'] == start+1:
        selected = read(base/'epochs'/('epoch-%04d'%start)/'selection.json')
        if selected['epoch'] != start or selected['pool_sha256'] != sha(new_pool/'pool.json'):
            raise ValueError('Pending reserved questions changed during isolated recovery')
    config['start_epoch'] = start
    result = dict(status='ISOLATED_HEAD_POOL_RECOVERY_STAGED', source_base=str(source),
         source_job_id=job_id, run_id=config['run_id'], completed_epoch=completed, start_epoch=start,
         head_sha256=latest['sha256'], actual_adam_steps_per_rank=latest['actual_adam_steps_per_rank'],
         source_latest_preserved=True, original_partial_rollouts_preserved=True,
         reserved_pending_selection_reused=state['next_epoch']==start+1,
         pool_next_epoch=state['next_epoch'], pool_cursor=state['cursor'],
         pool_sha256=sha(new_pool/'pool.json'), pool_state_sha256=sha(new_pool/'state.json'),
         source_checkpoint_preserved_path=str(preserved), copied_training_files=copied,
         retained_question_metadata=retained, new_runtime_commit=config['source_commit'],
         source_runtime_commit=source_config['source_commit'], checked_unix=time.time())
    once(prep/'restore-receipt.json', result)
    return result
