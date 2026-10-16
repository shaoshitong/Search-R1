"""Native4 prefix20 paired checkpoints, HPC3 sources -> HKUST3 archives.

Pure stdlib, compatible with Python 3.8. This module never transfers files,
loads tensors/models, requests GPUs, trains, or creates an HPC3 restore copy.
The completed-pair producer owns CPU tensor/state validation. This layer checks
the exact producer graph plus explicit recovery dependencies by full-file SHA.
Its only final status is SOURCE_ARCHIVE_FULL_SHA_PASS, never CUDA acceptance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import time

PERSONAL_ROOT = Path('/data/user/jxu957/experiments')
DEFAULT_BASE = PERSONAL_ROOT/'qprobe-teacher-warmup200-prefix20-joint4-20261016'
MODEL = PERSONAL_ROOT/'topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725'
DATA = PERSONAL_ROOT/'topd-20261008/runtime/assets/common-instruct-0b002b'
ARCHIVES = Path('/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/checkpoints')
STEPS = (5, 50, 100, 150, 200)
KIND = 'prefix20_cross_host_checkpoint_archive'
STATUS = 'SOURCE_ARCHIVE_FULL_SHA_PASS'
SNAPSHOT_STATUS = 'SOURCE_FULL_SHA_SNAPSHOT_PASS'
PAIR_STATUS = 'QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS'
ARCHITECTURE = 'qwen2_prefix20_trajectory_em_v1'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_NAMES = ('manifest.json', 'train.parquet', 'validation.parquet', 'stage.parquet',
              'final.parquet', 'exclusions-audit.json', 'calibration-300.parquet', 'audit.json')
LINKED_DATA_NAMES = frozenset(DATA_NAMES[:6])
ASSETS = {
    'head.safetensors': (308473196, '535c7bfee07374dde88b7d7f9e4f08ac874d46088ef961f15ea6ab7dd849a560'),
    'probe-config.json': (1940, 'be7456474ad62f3c874a4fb176d9858fa8cef385077e96247a3e8c795e3347d7'),
    'eligible-question-pool.json': (41242673, 'ae13710254e7ae77cf5531b62a755ec2000bc3fba20051bbafaff1f6714aa8fc'),
}
MODEL_SHA256 = {
    'config.json': 'eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b',
    'generation_config.json': 'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'model-00001-of-00002.safetensors': '67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2',
    'model-00002-of-00002.safetensors': 'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d',
    'model.safetensors.index.json': 'bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}


def need(condition, message):
    if not condition:
        raise ValueError('PREFIX20_ARCHIVE_' + message)


def under(path, parent):
    try:
        Path(path).relative_to(parent)
        return True
    except ValueError:
        return False


def absolute(value):
    need(isinstance(value, str) and value and '\x00' not in value, 'INVALID_PATH')
    path = PurePosixPath(value)
    need(path.is_absolute() and '..' not in path.parts and str(path) == value, 'NONCANONICAL_ABSOLUTE_PATH')
    return Path(value)


def relative(value):
    need(isinstance(value, str) and value and '\x00' not in value, 'INVALID_RELATIVE_PATH')
    path = PurePosixPath(value)
    need(not path.is_absolute() and '..' not in path.parts and str(path) == value and value != '.',
         'NONCANONICAL_RELATIVE_PATH')
    return path


def own_base(value):
    path = absolute(str(value))
    need(path.parent == PERSONAL_ROOT and re.fullmatch(
        r'qprobe-teacher-warmup200-prefix20-[A-Za-z0-9._-]+', path.name) is not None,
        'DEDICATED_PREFIX20_BASE_REQUIRED')
    return path


def binding(base, commit, run_id, step):
    base = own_base(base)
    need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit) is not None,
         'FULL_SOURCE_COMMIT_REQUIRED')
    need(isinstance(run_id, str) and re.fullmatch('[0-9a-f]{32}', run_id) is not None, 'RUN_ID_REQUIRED')
    need(type(step) is int and step in STEPS, 'FIXED_MILESTONE_REQUIRED')
    return base


def digest_text(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256_REQUIRED')
    return value


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def document_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def document_sha(value):
    return hashlib.sha256(document_bytes(value)).hexdigest()


def safe_components(path, *, require_exists=True):
    path = absolute(str(path))
    for member in reversed((path,) + tuple(path.parents)):
        try:
            mode = member.lstat().st_mode
        except FileNotFoundError:
            need(not require_exists, 'PATH_MISSING')
            continue
        need(not stat.S_ISLNK(mode), 'SYMLINK_COMPONENT')
        if member != path:
            need(stat.S_ISDIR(mode), 'NON_DIRECTORY_COMPONENT')
    return path


def regular(path):
    path = safe_components(path)
    value = path.lstat()
    need(stat.S_ISREG(value.st_mode), 'NONREGULAR_PAYLOAD_FILE')
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def validate_identity(value, *, device=True):
    keys = {'inode', 'bytes', 'mtime_ns'} | ({'device'} if device else set())
    need(isinstance(value, dict) and set(value) == keys, 'FILE_IDENTITY_SCHEMA_DIFFERS')
    need(all(type(value[k]) is int and value[k] >= 0 for k in keys), 'FILE_IDENTITY_VALUES_INVALID')
    need(value['inode'] > 0, 'FILE_INODE_INVALID')


def checked_file(path, expected_sha, *, expected_identity=None, expected_bytes=None, fsync=False):
    digest_text(expected_sha)
    before = regular(path)
    if expected_identity is not None:
        need(all(before[k] == expected_identity[k] for k in expected_identity), 'PRODUCER_FILE_IDENTITY_CHANGED')
    if expected_bytes is not None:
        need(before['bytes'] == expected_bytes, 'FILE_BYTES_DIFFER')
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        fd_identity = lambda: dict(zip(('device', 'inode', 'bytes', 'mtime_ns'),
            (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino,
             os.fstat(descriptor).st_size, os.fstat(descriptor).st_mtime_ns)))
        need(fd_identity() == before, 'OPENED_FILE_IDENTITY_DIFFERS')
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8*2**20)
            if not chunk:
                break
            digest.update(chunk)
        need(digest.hexdigest() == expected_sha, 'FULL_SHA_DIFFERS')
        if fsync:
            os.fsync(descriptor)
        need(fd_identity() == before, 'OPENED_FILE_CHANGED_DURING_FULL_READ')
    finally:
        os.close(descriptor)
    need(regular(path) == before, 'FILE_CHANGED_DURING_FULL_READ')
    return before


def publish(path, value):
    """Exclusive immutable publication, durable file and containing directory."""
    path = Path(path)
    safe_components(path.parent)
    need(path.parent.is_dir(), 'PUBLICATION_PARENT_NOT_DIRECTORY')
    temporary = path.with_name(path.name + '.partial-' + str(time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(document_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))  # Fails on an existing result: inspect, never overwrite.
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_directory(path.parent)


def fsync_directory(path):
    safe_components(path)
    need(Path(path).is_dir(), 'FSYNC_NOT_DIRECTORY')
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def producer_records(value):
    if isinstance(value, dict):
        if all(k in value for k in ('path', 'inode', 'bytes', 'mtime_ns', 'full_read_sha256')):
            yield value
        else:
            for member in value.values():
                yield from producer_records(member)
    elif isinstance(value, list):
        for member in value:
            yield from producer_records(member)


def producer_rows(receipt, base):
    result = {}
    for row in producer_records(receipt):
        path = absolute(row['path'])
        need(under(path, base) or under(path, MODEL), 'PRODUCER_PATH_OUTSIDE_SOURCE_OR_OFFICIAL_MODEL')
        identity = {k: row[k] for k in ('inode', 'bytes', 'mtime_ns')}
        validate_identity(identity, device=False)
        digest_text(row['full_read_sha256'])
        if str(path) in result:
            old = result[str(path)]
            need(all(old[k] == row[k] for k in ('inode', 'bytes', 'mtime_ns', 'full_read_sha256')),
                 'CONFLICTING_PRODUCER_RECORDS')
        result[str(path)] = row
    need(result, 'EMPTY_PRODUCER_RECORD_GRAPH')
    return result


def pair_path(base, step):
    return base/'artifacts/question-probe/completed-pairs'/('global_step_' + str(step))/'source-receipt.json'


def checkpoint_path(base, step):
    return base/'checkpoints/search-g1-common8/actor'/('global_step_' + str(step))


def validate_pair(receipt, run, reward, base, commit, run_id, step):
    need(isinstance(receipt, dict) and isinstance(run, dict), 'PAIR_DOCUMENT_REQUIRED')
    fields = {'schema_version', 'status', 'kind', 'step', 'run_id', 'checkpoint', 'source_base',
        'source_run_manifest', 'world_size', 'actor_snapshot_id', 'ranks', 'files', 'boundary_driver',
        'boundary_worker_files', 'historical_artifacts', 'reward_probe_manifest', 'reward_files',
        'reward_probe_version', 'reward_probe_round', 'question_probe_state_path', 'head_path',
        'feature_spec_sha', 'head_version', 'hf_saved_tensor_count', 'attempt_progress', 'driver_state',
        'actual_cuda_restore_accepted', 'rng_boundary_stage', 'scope', 'checked_unix'}
    need(set(receipt) == fields, 'PRODUCER_RECEIPT_SCHEMA_DIFFERS')
    need(receipt.get('schema_version') == 1 and receipt.get('kind') == 'question_probe_pair' and
         receipt.get('status') == PAIR_STATUS and receipt.get('world_size') == 4 and
         type(receipt.get('step')) is int and receipt['step'] == step and
         receipt.get('run_id') == run_id and receipt.get('source_base') == str(base) and
         receipt.get('actor_snapshot_id') == run_id + ':' + str(step) and
         receipt.get('actual_cuda_restore_accepted') is False, 'PAIR_BINDING_DIFFERS')
    cp = checkpoint_path(base, step)
    pair_dir = pair_path(base, step).parent
    need(receipt.get('checkpoint') == str(cp) and
         receipt.get('question_probe_state_path') == str(cp/'training_state/question-probe-state.pt') and
         receipt.get('head_path') == str(cp/'training_state/prefix20-rank-0.pt') and
         receipt.get('head_version') == 'prefix20-epoch68-outer-%04d' % step,
         'PAIR_CHECKPOINT_OR_HEAD_PATH_DIFFERS')
    need(run.get('run_id') == run_id and run.get('source_commit') == commit and
         run.get('training_world_size') == 4 and run.get('target_updates') == 200 and
         run.get('data_selection_mode') == 'prefix20_probe' and
         run.get('initial_model') == str(MODEL) and run.get('reference_model') == str(MODEL) and
         run.get('model_revision') == REVISION and run.get('initial_model_sha256') == MODEL_SHA256,
         'FROZEN_RUN_OR_OFFICIAL_MODEL_BINDING_DIFFERS')
    config = run.get('question_probe', {})
    expected = dict(probe_architecture=ARCHITECTURE, candidate_questions=5120, selected_questions=512,
        exploration_questions=0, responses_per_question=5, head_learning_rate=1e-6,
        head_batch_size=256, head_epochs_per_outer=1, replay_window_outer=0,
        model_dir=str(MODEL), head_safetensors=str(base/'assets/prefix20/head.safetensors'),
        probe_config=str(base/'assets/prefix20/probe-config.json'),
        eligible_pool_path=str(base/'assets/prefix20/eligible-question-pool.json'))
    need(isinstance(config, dict) and all(config.get(k) == v and type(config.get(k)) is type(v)
         for k, v in expected.items()), 'PREFIX20_SCIENTIFIC_CONTRACT_DIFFERS')
    inputs = run.get('input_sha256')
    need(isinstance(inputs, dict) and set(inputs) == set(DATA_NAMES), 'CANONICAL_INPUT_SET_DIFFERS')
    for value in inputs.values():
        digest_text(value)
    digest_text(receipt['feature_spec_sha'])
    need(type(receipt['hf_saved_tensor_count']) is int and receipt['hf_saved_tensor_count'] > 0,
         'HF_SAVED_TENSOR_COUNT_INVALID')
    need(receipt['rng_boundary_stage'] in ('post_development_and_probe_adoption', 'saved_checkpoint_before_development'),
         'RNG_BOUNDARY_STAGE_DIFFERS')
    history = receipt['historical_artifacts']
    need(isinstance(history, dict) and set(history) == {'checkpoints', 'development', 'reward_archives',
        'raw_training_prefix_verified', 'scope'} and history['raw_training_prefix_verified'] is False,
        'HISTORICAL_ARTIFACT_SCHEMA_DIFFERS')
    need(isinstance(history['checkpoints'], list) and isinstance(history['reward_archives'], list) and
        isinstance(history['development'], dict) and set(history['development']) <= {'before', 'after5'},
        'HISTORICAL_ARTIFACT_LIST_DIFFERS')
    history_steps = []
    for row in history['checkpoints']:
        previous = row.get('step')
        need(type(previous) is int and previous in STEPS and previous <= step and
             row.get('path') == str(checkpoint_path(base, previous)/'complete.json'), 'HISTORICAL_CHECKPOINT_PATH_DIFFERS')
        history_steps.append(previous)
    need(len(history_steps) == len(set(history_steps)), 'DUPLICATE_HISTORICAL_CHECKPOINT')
    for key, row in history['development'].items():
        name = 'formal-inline-development-before.json' if key == 'before' else 'formal-inline-development-after.json'
        need(row.get('path') == str(base/'logs'/name), 'HISTORICAL_DEVELOPMENT_PATH_DIFFERS')
    versions = []
    for row in history['reward_archives']:
        path = absolute(row['path'])
        need(under(path, base/'artifacts') and path.name == 'version-manifest.json' and
             path.parent.parent == base/'artifacts' and path.parent.name.startswith('round_') and
             isinstance(row.get('version'), str) and row['version'].startswith(run_id+':'),
             'HISTORICAL_REWARD_ARCHIVE_PATH_DIFFERS')
        versions.append(row['version'])
    need(len(versions) == len(set(versions)), 'DUPLICATE_HISTORICAL_REWARD_ARCHIVE')
    rows = producer_rows(receipt, base)
    cp_files = receipt.get('files')
    need(isinstance(cp_files, list) and cp_files, 'CHECKPOINT_FILE_LIST_REQUIRED')
    names = []
    for row in cp_files:
        name = str(relative(row['relative_path']))
        need(row['path'] == str(cp/name), 'CHECKPOINT_RELATIVE_PATH_DIFFERS')
        names.append(name)
    required = {'complete.json', 'config.json', 'model.safetensors.index.json',
                'training_state/manifest.json', 'training_state/driver.pt',
                'training_state/question-probe-state.pt'}
    required |= {'training_state/rank-%d.pt' % rank for rank in range(4)}
    required |= {'training_state/prefix20-rank-%d.pt' % rank for rank in range(4)}
    need(len(names) == len(set(names)) and required <= set(names), 'ACTOR_PROBE_DRIVER_FILE_SET_INCOMPLETE')
    need(receipt.get('source_run_manifest', {}).get('path') == str(pair_dir/'run-manifest.json') and
         receipt.get('boundary_driver', {}).get('path') == str(pair_dir/'boundary-driver.pt') and
         receipt.get('reward_probe_manifest', {}).get('path') == str(pair_dir/'reward-probe-manifest.json'),
         'BOUNDARY_DOCUMENT_PATH_DIFFERS')
    boundaries = receipt.get('boundary_worker_files', [])
    need(sorted(r.get('rank') for r in boundaries) == list(range(4)) and all(
        r['path'] == str(pair_dir/'boundary-worker-rng'/('rank-%d.pt' % r['rank'])) for r in boundaries),
        'FOUR_BOUNDARY_RNG_FILES_REQUIRED')
    ranks = receipt.get('ranks', [])
    need([r.get('rank') for r in ranks] == list(range(4)), 'FOUR_ACTOR_RANKS_REQUIRED')
    progress = receipt.get('attempt_progress', {})
    need(progress.get('completed_outer') == step and type(progress.get('attempt_index')) is int and
         progress['attempt_index'] >= step and type(progress.get('cumulative_writes')) is int and
         progress['cumulative_writes'] >= step and
         progress.get('aborted_attempts') == progress['attempt_index'] - step and
         type(progress.get('full_budget_outer')) is int and 0 <= progress['full_budget_outer'] <= step,
         'ACTOR_PROGRESS_COUNTERS_INVALID')
    ids = ranks[0].get('used_attempt_ids', [])
    need(isinstance(ids, list) and len(ids) == len(set(ids)) == progress['attempt_index'] and
         all(isinstance(a, str) and re.fullmatch('[0-9a-f]{32}', a) for a in ids), 'ATTEMPT_ID_SET_INVALID')
    need(all(r.get('completed_outer') == step and r.get('scheduler_last_epoch') == step and
        r.get('scheduler_step_count') == step + 1 and r.get('actual_optimizer_writes') == progress['cumulative_writes'] and
        r.get('last_attempt_index') == progress['attempt_index'] and r.get('used_attempt_ids') == ids and
        r.get('optimizer_tensors_finite') is True and r.get('policy_rejections') == ranks[0].get('policy_rejections')
        for r in ranks), 'ACTOR_RANK_COUNTERS_DIFFER')
    need(type(ranks[0].get('policy_rejections')) is int and ranks[0]['policy_rejections'] >= 0,
         'ACTOR_REJECTION_COUNT_INVALID')
    driver = receipt['driver_state']
    need(isinstance(driver, dict) and set(driver) == {'completed_outer', 'source_consumed'} and
         driver['completed_outer'] == step and type(driver['source_consumed']) is int and
         driver['source_consumed'] >= step*5120, 'DRIVER_DATA_COUNTERS_INVALID')
    need(isinstance(receipt.get('reward_files'), list) and receipt['reward_files'], 'REWARD_DEPENDENCIES_REQUIRED')
    validate_reward(receipt, reward, base, run_id, step)
    return rows


def validate_reward(receipt, reward, base, run_id, step):
    round_id = min(step//50, 3)*50
    need(isinstance(reward, dict) and reward.get('run_id') == run_id and type(reward.get('round')) is int and
        reward['round'] == round_id and reward.get('version') == receipt['reward_probe_version'] and
        receipt['reward_probe_round'] == round_id and isinstance(reward['version'], str) and
        reward['version'].startswith(run_id+':'+str(round_id)+':') and
        len(reward['version']) > len(run_id+':'+str(round_id)+':'), 'REWARD_VERSION_BINDING_DIFFERS')
    heads = reward.get('head_sha256')
    need(isinstance(heads, dict) and set(heads) == {'commit_probe.pkl', 'boundary_probe.pkl'}, 'REWARD_HEAD_SET_DIFFERS')
    expected = {}
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        path = absolute(reward[key])
        need(under(path, base/'artifacts') and path.name == name, 'REWARD_HEAD_PATH_DIFFERS')
        expected[str(path)] = digest_text(heads[name])
    encoder = absolute(reward['encoder'])
    if encoder != MODEL:
        need(encoder.parent == base/'checkpoints/search-g1-common8/actor' and
            re.fullmatch('global_step_(5|50|100|150|200)', encoder.name) is not None and
            int(encoder.name.split('_')[-1]) <= step, 'REWARD_ENCODER_PATH_DIFFERS')
    hashes = reward.get('encoder_sha256')
    essential = set(MODEL_SHA256) - {'vocab.json', 'merges.txt', 'generation_config.json'}
    need(isinstance(hashes, dict) and essential <= set(hashes) <= set(MODEL_SHA256), 'REWARD_ENCODER_FILE_SET_DIFFERS')
    for name, digest in hashes.items():
        digest_text(digest)
        if encoder == MODEL:
            need(digest == MODEL_SHA256[name], 'REWARD_OFFICIAL_ENCODER_SHA_DIFFERS')
        expected[str(encoder/name)] = digest
    actual = {}
    for row in receipt['reward_files']:
        path = row['path']
        need(path not in actual, 'DUPLICATE_REWARD_FILE')
        actual[path] = row['full_read_sha256']
    need(actual == expected, 'REWARD_HEAD_OR_ENCODER_CLOSURE_INCOMPLETE')


def dependency_spec(base, run):
    result = {}
    for name in DATA_NAMES:
        # The six official data links are intentionally resolved only at these
        # exact paths. Payload and restored targets must contain regular files.
        source = DATA/name if name in LINKED_DATA_NAMES else base/'data'/name
        result['canonical_data/' + name] = dict(source_path=str(source),
            bound_path=str(base/'data'/name), sha256=run['input_sha256'][name], bytes=None)
    for name, (size, digest) in ASSETS.items():
        result['initial_probe/' + name] = dict(source_path=str(base/'assets/prefix20'/name),
            bound_path=str(base/'assets/prefix20'/name), sha256=digest, bytes=size)
    for name, digest in MODEL_SHA256.items():
        result['frozen_model/' + name] = dict(source_path=str(MODEL/name),
            bound_path=str(MODEL/name), sha256=digest, bytes=None)
    return result


def validate_snapshot(value, *, base, commit, run_id, step):
    """Validate the complete graph without reading HPC3 paths on HKUST3."""
    base = binding(base, commit, run_id, step)
    required = {'schema_version', 'kind', 'status', 'run_id', 'step', 'world_size', 'source_base',
        'training_source_commit', 'source_receipt', 'source_receipt_sha256', 'pair_receipt', 'run_manifest', 'reward_manifest',
        'dependencies', 'files', 'bytes', 'actual_cuda_restore_accepted', 'checked_unix'}
    need(isinstance(value, dict) and set(value) == required, 'SNAPSHOT_SCHEMA_DIFFERS')
    need(value['schema_version'] == 2 and value['kind'] == KIND and value['status'] == SNAPSHOT_STATUS and
         value['run_id'] == run_id and value['source_base'] == str(base) and value['training_source_commit'] == commit and
         type(value['step']) is int and value['step'] == step and value['world_size'] == 4 and
         value['actual_cuda_restore_accepted'] is False, 'SNAPSHOT_BINDING_DIFFERS')
    need(type(value['checked_unix']) in (int, float) and math.isfinite(value['checked_unix']) and
         0 < value['checked_unix'] <= time.time() + 60, 'SNAPSHOT_TIME_INVALID')
    need(value['source_receipt'] == str(pair_path(base, step)), 'SOURCE_RECEIPT_PATH_DIFFERS')
    digest_text(value['source_receipt_sha256'])
    rows = validate_pair(value['pair_receipt'], value['run_manifest'], value['reward_manifest'], base, commit, run_id, step)
    spec = dependency_spec(base, value['run_manifest'])
    need(isinstance(value['dependencies'], dict) and set(value['dependencies']) == set(spec), 'DEPENDENCY_SET_DIFFERS')
    for role, expected in spec.items():
        actual = value['dependencies'][role]
        need(isinstance(actual, dict) and set(actual) == set(expected) and all(
             actual[k] == v for k, v in expected.items() if k != 'bytes' or v is not None), 'DEPENDENCY_BINDING_DIFFERS')
        need(type(actual['bytes']) is int and actual['bytes'] >= 0, 'DEPENDENCY_BYTES_INVALID')
    files = value['files']
    need(isinstance(files, list) and files, 'SNAPSHOT_FILE_LIST_REQUIRED')
    by_path = {}
    previous = None
    for row in files:
        need(isinstance(row, dict) and set(row) == {'source_path', 'payload_path', 'bytes', 'sha256', 'source_identity'},
             'SNAPSHOT_FILE_SCHEMA_DIFFERS')
        path = absolute(row['source_path'])
        need(previous is None or previous < str(path), 'DUPLICATE_OR_UNSORTED_SOURCE_PATH')
        previous = str(path)
        payload = relative(row['payload_path'])
        need(str(payload) == 'payload/' + str(path.relative_to('/')), 'PAYLOAD_PATH_MAPPING_DIFFERS')
        digest_text(row['sha256'])
        validate_identity(row['source_identity'])
        need(type(row['bytes']) is int and row['bytes'] >= 0 and row['bytes'] == row['source_identity']['bytes'],
             'SNAPSHOT_FILE_BYTES_INVALID')
        by_path[str(path)] = row
    expected_paths = set(rows) | {value['source_receipt']} | {r['source_path'] for r in value['dependencies'].values()}
    need(set(by_path) == expected_paths, 'SNAPSHOT_EXACT_FILE_SET_DIFFERS')
    for path, producer in rows.items():
        row = by_path[path]
        need(row['sha256'] == producer['full_read_sha256'] and all(
             row['source_identity'][k] == producer[k] for k in ('inode', 'bytes', 'mtime_ns')),
             'SNAPSHOT_PRODUCER_BINDING_DIFFERS')
    need(by_path[value['source_receipt']]['sha256'] == value['source_receipt_sha256'], 'RECEIPT_SHA_BINDING_DIFFERS')
    for dep in value['dependencies'].values():
        need(by_path[dep['source_path']]['sha256'] == dep['sha256'] and
             by_path[dep['source_path']]['bytes'] == dep['bytes'], 'SNAPSHOT_DEPENDENCY_FILE_BINDING_DIFFERS')
    need(type(value['bytes']) is int and value['bytes'] == sum(r['bytes'] for r in files), 'SNAPSHOT_TOTAL_BYTES_DIFFER')
    return value


def exact_tree(directory, expected_names):
    """Reject symlinks (including dangling), devices, extra files/empty dirs."""
    directory = safe_components(directory)
    need(directory.is_dir(), 'PAYLOAD_DIRECTORY_REQUIRED')
    expected_files = {str(relative(name)) for name in expected_names}
    expected_dirs = {'.'}
    for name in expected_files:
        expected_dirs.update(str(parent) for parent in PurePosixPath(name).parents)
    actual_files, actual_dirs = set(), {'.'}
    for root, dirs, files in os.walk(str(directory), followlinks=False):
        current = Path(root)
        for name in dirs:
            path = current/name
            mode = path.lstat().st_mode
            need(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), 'UNSAFE_TREE_DIRECTORY')
            actual_dirs.add(str(path.relative_to(directory)))
        for name in files:
            path = current/name
            need(stat.S_ISREG(path.lstat().st_mode), 'UNSAFE_TREE_FILE')
            actual_files.add(str(path.relative_to(directory)))
    need(actual_files == expected_files and actual_dirs == expected_dirs, 'EXACT_PAYLOAD_TREE_DIFFERS')
    return actual_dirs


def _read_json(path):
    before = regular(path)
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'r') as stream:
        info = os.fstat(stream.fileno())
        need((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == tuple(
             before[k] for k in ('device', 'inode', 'bytes', 'mtime_ns')), 'JSON_OPENED_FILE_IDENTITY_DIFFERS')
        value = json.load(stream, object_pairs_hook=no_duplicates,
                          parse_constant=lambda _: need(False, 'NONFINITE_JSON_VALUE'))
    need(regular(path) == before, 'JSON_FILE_CHANGED_DURING_READ')
    return value


def _check_source_documents(snapshot):
    need(_read_json(snapshot['source_receipt']) == snapshot['pair_receipt'], 'ORIGINAL_RECEIPT_DOCUMENT_CHANGED')
    run_path = snapshot['pair_receipt']['source_run_manifest']['path']
    need(_read_json(run_path) == snapshot['run_manifest'], 'FROZEN_RUN_DOCUMENT_CHANGED')
    reward_path = snapshot['pair_receipt']['reward_probe_manifest']['path']
    need(_read_json(reward_path) == snapshot['reward_manifest'], 'FROZEN_REWARD_DOCUMENT_CHANGED')
    for role, dep in snapshot['dependencies'].items():
        bound = Path(dep['bound_path'])
        if role.startswith('canonical_data/') and bound.name in LINKED_DATA_NAMES:
            safe_components(bound.parent)
            need(bound.is_symlink() and os.readlink(str(bound)) == dep['source_path'], 'OFFICIAL_DATA_LINK_DIFFERS')


def snapshot(base, step, *, commit, run_id):
    base = binding(base, commit, run_id, step)
    need(os.getuid() == 14090, 'JXU_SOURCE_OBSERVER_REQUIRED')
    safe_components(base)
    source = pair_path(base, step)
    receipt = _read_json(source)
    run_path = source.parent/'run-manifest.json'
    need(isinstance(receipt, dict) and receipt.get('source_run_manifest', {}).get('path') == str(run_path),
         'BOUNDARY_RUN_DOCUMENT_PATH_DIFFERS')
    run = _read_json(run_path)
    reward_path = source.parent/'reward-probe-manifest.json'
    need(receipt.get('reward_probe_manifest', {}).get('path') == str(reward_path), 'BOUNDARY_REWARD_DOCUMENT_PATH_DIFFERS')
    reward = _read_json(reward_path)
    rows = producer_rows(receipt, base)
    validate_pair(receipt, run, reward, base, commit, run_id, step)
    cp = checkpoint_path(base, step)
    exact_tree(cp, [r['relative_path'] for r in receipt['files']])
    index = _read_json(cp/'model.safetensors.index.json')
    weight_map = index.get('weight_map')
    need(isinstance(weight_map, dict) and weight_map and all(
         isinstance(name, str) and str(relative(name)) in {r['relative_path'] for r in receipt['files']}
         for name in weight_map.values()), 'HF_SHARD_INDEX_INCOMPLETE')
    need(len(weight_map) == receipt['hf_saved_tensor_count'], 'HF_SAVED_TENSOR_COUNT_DIFFERS')
    dependencies = dependency_spec(base, run)
    for role, dep in dependencies.items():
        bound = Path(dep['bound_path'])
        if role.startswith('canonical_data/') and bound.name in LINKED_DATA_NAMES:
            safe_components(bound.parent)
            need(bound.is_symlink() and os.readlink(str(bound)) == dep['source_path'], 'OFFICIAL_DATA_LINK_DIFFERS')
        identity = regular(dep['source_path'])
        if dep['bytes'] is not None:
            need(identity['bytes'] == dep['bytes'], 'DEPENDENCY_SIZE_DIFFERS')
        dep['bytes'] = identity['bytes']
    directory = base/'preparation/checkpoint-archives'/('global_step_' + str(step))
    safe_components(directory.parent, require_exists=False)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    publish(directory/'claim.json', dict(role='fresh_source_archive_snapshot', run_id=run_id, step=step,
        training_source_commit=commit, source_receipt=str(source), pid=os.getpid(), started_unix=time.time()))
    source_identity = regular(source)
    source_sha = sha(source)
    need(regular(source) == source_identity, 'RECEIPT_CHANGED_DURING_FULL_READ')
    expected = {p: dict(sha256=r['full_read_sha256'], bytes=r['bytes'],
        identity={k: r[k] for k in ('inode', 'bytes', 'mtime_ns')}) for p, r in rows.items()}
    expected[str(source)] = dict(sha256=source_sha, bytes=source_identity['bytes'], identity=source_identity)
    for dep in dependencies.values():
        if dep['source_path'] in expected:
            need(expected[dep['source_path']]['sha256'] == dep['sha256'] and
                 expected[dep['source_path']]['bytes'] == dep['bytes'], 'DEPENDENCY_PRODUCER_CONFLICT')
        else:
            expected[dep['source_path']] = dict(sha256=dep['sha256'], bytes=dep['bytes'], identity=None)
    files = []
    with (directory/'source-progress.jsonl').open('x') as progress:
        for path, record in sorted(expected.items()):
            identity = checked_file(path, record['sha256'], expected_identity=record['identity'], expected_bytes=record['bytes'])
            row = dict(source_path=path, payload_path='payload/' + str(Path(path).relative_to('/')),
                bytes=record['bytes'], sha256=record['sha256'], source_identity=identity)
            files.append(row)
            progress.write(json.dumps(dict(source_path=path, verified_files=len(files),
                verified_bytes=sum(r['bytes'] for r in files), checked_unix=time.time())) + '\n')
            progress.flush()
            os.fsync(progress.fileno())
    value = dict(schema_version=2, kind=KIND, status=SNAPSHOT_STATUS, run_id=run_id, step=step,
        world_size=4, source_base=str(base), training_source_commit=commit, source_receipt=str(source),
        source_receipt_sha256=source_sha, pair_receipt=receipt, run_manifest=run, reward_manifest=reward, dependencies=dependencies,
        files=files, bytes=sum(r['bytes'] for r in files), actual_cuda_restore_accepted=False, checked_unix=time.time())
    validate_snapshot(value, base=base, commit=commit, run_id=run_id, step=step)
    _check_source_documents(value)
    path = directory/'source-snapshot.json'
    publish(path, value)
    return dict(snapshot_file=str(path), snapshot_sha256=sha(path), snapshot=value)


def load_snapshot(path, expected_sha, *, base, commit, run_id, step, archive=False):
    base = binding(base, commit, run_id, step)
    expected_path = (ARCHIVES/run_id/('global_step_' + str(step))/'source-snapshot.json' if archive else
                     base/'preparation/checkpoint-archives'/('global_step_' + str(step))/'source-snapshot.json')
    need(absolute(str(path)) == expected_path, 'SNAPSHOT_FILE_PATH_DIFFERS')
    checked_file(path, expected_sha)
    value = _read_json(path)
    return validate_snapshot(value, base=base, commit=commit, run_id=run_id, step=step)


def source_readback(snapshot_path, expected_sha, *, base, commit, run_id, step):
    need(os.getuid() == 14090, 'JXU_SOURCE_OBSERVER_REQUIRED')
    value = load_snapshot(snapshot_path, expected_sha, base=base, commit=commit, run_id=run_id, step=step)
    for row in value['files']:
        checked_file(row['source_path'], row['sha256'], expected_identity=row['source_identity'], expected_bytes=row['bytes'])
    _check_source_documents(value)
    exact_tree(checkpoint_path(Path(base), step), [r['relative_path'] for r in value['pair_receipt']['files']])
    return dict(snapshot_file=str(snapshot_path), snapshot_sha256=expected_sha, all_source_files_fresh_sha_pass=True,
        files=len(value['files']), bytes=value['bytes'], run_id=run_id, step=step,
        training_source_commit=commit, actual_cuda_restore_accepted=False, checked_unix=time.time())


def archive_readback(directory, snapshot_path, expected_sha, *, base, commit, run_id, step):
    """HKUST3 only, after the controller proves actual copy exit status zero."""
    need(socket.gethostname() == 'xielab3', 'ACTUAL_HKUST3_OBSERVER_REQUIRED')
    directory = absolute(str(directory))
    need(directory == ARCHIVES/run_id/('global_step_' + str(step)), 'ACTUAL_HKUST3_DESTINATION_REQUIRED')
    safe_components(directory)
    value = load_snapshot(snapshot_path, expected_sha, base=base, commit=commit, run_id=run_id, step=step, archive=True)
    names = [str(PurePosixPath(r['payload_path']).relative_to('payload')) for r in value['files']]
    dirs = exact_tree(directory/'payload', names)
    files = []
    with (directory/'archive-progress.jsonl').open('x') as progress:
        for row in value['files']:
            path = directory/row['payload_path']
            identity = checked_file(path, row['sha256'], expected_bytes=row['bytes'], fsync=True)
            files.append(dict(row, archive_path=str(path), archive_identity=identity))
            progress.write(json.dumps(dict(payload_path=row['payload_path'], verified_files=len(files),
                verified_bytes=sum(r['bytes'] for r in files), checked_unix=time.time())) + '\n')
            progress.flush()
            os.fsync(progress.fileno())
    archived = lambda source: directory/'payload'/str(Path(source).relative_to('/'))
    need(_read_json(archived(value['source_receipt'])) == value['pair_receipt'], 'ARCHIVED_PAIR_DOCUMENT_DIFFERS')
    need(_read_json(archived(value['pair_receipt']['source_run_manifest']['path'])) == value['run_manifest'],
         'ARCHIVED_FROZEN_RUN_DOCUMENT_DIFFERS')
    need(_read_json(archived(value['pair_receipt']['reward_probe_manifest']['path'])) == value['reward_manifest'],
         'ARCHIVED_FROZEN_REWARD_DOCUMENT_DIFFERS')
    exact_tree(directory/'payload', names)
    for name in sorted(dirs, key=lambda p: (len(PurePosixPath(p).parts), p), reverse=True):
        fsync_directory(directory/'payload'/name)
    fsync_directory(directory)
    proof = dict(schema_version=2, kind=KIND, status=STATUS, run_id=run_id, step=step, world_size=4,
        source_base=str(base), training_source_commit=commit, source_receipt=value['source_receipt'],
        source_receipt_sha256=value['source_receipt_sha256'], source_snapshot_sha256=expected_sha,
        archive_root=str(directory), archive_host='xielab3', observer_pid=os.getpid(), files=files,
        bytes=value['bytes'], all_archive_files_fresh_sha_pass=True, all_archive_files_fsync_completed=True,
        source_retention_recheck_pending=True, hpc3_archive_copies=0, hpc3_restore_targets_created=0,
        actual_cuda_restore_accepted=False, checked_unix=time.time())
    path = directory/'archive-receipt.json'
    publish(path, proof)
    return dict(archive_receipt_file=str(path), archive_receipt_sha256=sha(path), archive_receipt=proof)


def validate_archive_proof(wrapper, snapshot, expected_sha):
    need(isinstance(wrapper, dict) and set(wrapper) == {
        'archive_receipt_file', 'archive_receipt_sha256', 'archive_receipt'}, 'ARCHIVE_WRAPPER_SCHEMA_DIFFERS')
    proof = wrapper['archive_receipt']
    need(isinstance(proof, dict) and set(proof) == {'schema_version', 'kind', 'status', 'run_id', 'step',
        'world_size', 'source_base', 'training_source_commit', 'source_receipt', 'source_receipt_sha256',
        'source_snapshot_sha256', 'archive_root', 'archive_host', 'observer_pid', 'files', 'bytes',
        'all_archive_files_fresh_sha_pass', 'all_archive_files_fsync_completed', 'source_retention_recheck_pending',
        'hpc3_archive_copies', 'hpc3_restore_targets_created', 'actual_cuda_restore_accepted', 'checked_unix'},
        'ARCHIVE_PROOF_SCHEMA_DIFFERS')
    need(document_sha(proof) == digest_text(wrapper['archive_receipt_sha256']), 'ARCHIVE_RECEIPT_SHA_DIFFERS')
    need(proof['schema_version'] == 2 and proof['kind'] == KIND and proof['status'] == STATUS and
        proof['archive_host'] == 'xielab3' and proof['actual_cuda_restore_accepted'] is False and
        proof['all_archive_files_fresh_sha_pass'] is True and proof['all_archive_files_fsync_completed'] is True and
        proof['source_retention_recheck_pending'] is True and type(proof['hpc3_archive_copies']) is int and
        proof['hpc3_archive_copies'] == 0 and type(proof['hpc3_restore_targets_created']) is int and
        proof['hpc3_restore_targets_created'] == 0 and type(proof['observer_pid']) is int and proof['observer_pid'] > 0,
        'ARCHIVE_COMPLETE_PROOF_INVALID')
    for key in ('run_id', 'step', 'world_size', 'source_base', 'training_source_commit',
                'source_receipt', 'source_receipt_sha256', 'bytes'):
        need(proof[key] == snapshot[key] and type(proof[key]) is type(snapshot[key]), 'ARCHIVE_SOURCE_BINDING_DIFFERS')
    root = ARCHIVES/snapshot['run_id']/('global_step_' + str(snapshot['step']))
    need(proof['source_snapshot_sha256'] == expected_sha and proof['archive_root'] == str(root) and
         wrapper['archive_receipt_file'] == str(root/'archive-receipt.json'), 'ARCHIVE_PATH_OR_SNAPSHOT_BINDING_DIFFERS')
    need(type(proof['checked_unix']) in (int, float) and math.isfinite(proof['checked_unix']) and
        snapshot['checked_unix'] <= proof['checked_unix'] <= time.time() + 60, 'ARCHIVE_TIME_INVALID')
    need(isinstance(proof['files'], list) and len(proof['files']) == len(snapshot['files']), 'ARCHIVE_FILE_COUNT_DIFFERS')
    fields = {'source_path', 'payload_path', 'bytes', 'sha256', 'source_identity'}
    for original, row in zip(snapshot['files'], proof['files']):
        need(isinstance(row, dict) and set(row) == fields | {'archive_path', 'archive_identity'} and
             {key: row[key] for key in fields} == original, 'ARCHIVE_FILE_BINDINGS_DIFFER')
        need(row['archive_path'] == str(root/row['payload_path']), 'ARCHIVE_FILE_PATH_DIFFERS')
        validate_identity(row['archive_identity'])
        need(row['archive_identity']['bytes'] == row['bytes'], 'ARCHIVE_FILE_IDENTITY_BYTES_DIFFER')
    return proof


def accept_archive(base, snapshot_path, expected_sha, archive_wrapper, *, commit, run_id, step):
    base = binding(base, commit, run_id, step)
    snapshot_value = load_snapshot(snapshot_path, expected_sha, base=base, commit=commit, run_id=run_id, step=step)
    proof = validate_archive_proof(archive_wrapper, snapshot_value, expected_sha)
    source_proof = source_readback(snapshot_path, expected_sha, base=base, commit=commit, run_id=run_id, step=step)
    value = dict(proof, source_readback=source_proof, original_sources_retained=True,
        source_retention_recheck_pending=False,
        source_snapshot=dict(path=str(snapshot_path), sha256=expected_sha),
        archive_receipt=dict(path=archive_wrapper['archive_receipt_file'], sha256=archive_wrapper['archive_receipt_sha256']),
        checked_unix=time.time())
    path = Path(snapshot_path).parent/'complete.json'
    publish(path, value)
    return dict(complete_file=str(path), complete_sha256=sha(path), status=STATUS, step=step,
        files=len(proof['files']), bytes=proof['bytes'], actual_cuda_restore_accepted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('snapshot', 'source-readback', 'archive-readback', 'accept-archive'))
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--snapshot')
    parser.add_argument('--snapshot-sha')
    parser.add_argument('--archive')
    args = parser.parse_args()
    kw = dict(commit=args.commit, run_id=args.run_id, step=args.step)
    if args.mode == 'snapshot':
        value = snapshot(args.base, args.step, commit=args.commit, run_id=args.run_id)
    elif args.mode == 'source-readback':
        value = source_readback(args.snapshot, args.snapshot_sha, base=args.base, **kw)
    elif args.mode == 'archive-readback':
        value = archive_readback(args.archive, args.snapshot, args.snapshot_sha, base=args.base, **kw)
    else:
        value = accept_archive(args.base, args.snapshot, args.snapshot_sha, json.load(sys.stdin), **kw)
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
