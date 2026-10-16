"""CPU-only, once-scoped source sealing and canonical identity preparation.

The old warmup may continue running. Its current complete latest inode is
hardlinked into this isolated namespace before hashing/loading, so a later
atomic replacement cannot change the imported weights. Any observed race
rejects the attempt and preserves its candidate; this entry never stops a job.
No token arrays, feature tensors, encoder or GPU are read or executed.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .contract import DIM, PREFIX_TOKENS, QMAX, WIDTH, WORLD, HIDDEN_CAPTURE_CONTRACT
from .pool import identity

COMMON_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
COMPOSITE_SHA = '8ac391ba26466e9fdbe9220b157bec81294ead69a09a0b7c2c4994cc7873a2c7'
PREFIX = '/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def remote(value):
    path = Path(value)
    if not path.is_absolute() or not str(path).startswith('/data/'):
        raise ValueError('CPU preparation requires explicit remote /data paths')
    return path


def stable_bytes(path, limit=16 * 1024 ** 2):
    path = Path(path)
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('CPU metadata input is not a bounded regular file: ' + str(path))
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
    if (len(raw) != before.st_size or signature(before) != signature(after)
            or signature(after) != signature(path.stat())):
        raise ValueError('CPU metadata changed while reading: ' + str(path))
    return raw


def read(path):
    return json.loads(stable_bytes(path))


def full_sha(path):
    path = Path(path)
    result = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('SHA input is not a regular remote file')
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            result.update(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError('Remote input changed during full SHA: ' + str(path))
    return result.hexdigest()


def checked_json(path, expected_sha):
    raw = stable_bytes(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Pinned JSON SHA differs: ' + str(path))
    return json.loads(raw)


def once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush(); os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_identity():
    fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=os.getpid(), birth=int(fields[19]), sid=os.getsid(0), uid=os.getuid(),
                observed_unix=time.time(), cuda_model_or_forward_used=False)


def validate_epoch_closure(latest, checkpoint, metrics, complete, handle):
    """The mutable latest pointer must refer to one fully closed native epoch."""
    epoch = latest.get('epoch')
    steps = latest.get('ema_updates')
    if (type(epoch) is not int or not 1 <= epoch <= 200 or latest.get('completed_epoch') is not True
            or type(steps) is not int or steps < 1 or latest.get('run_id') != handle['run_id']
            or latest.get('execution_id') != handle['execution_id']
            or latest.get('source_commit') != handle['source_commit']
            or latest.get('next_update_in_epoch') != 0
            or latest.get('actual_adam_steps_per_rank') != [steps] * WORLD
            or latest.get('counters', {}).get('complete_epochs') != epoch
            or latest.get('counters', {}).get('optimizer_steps') != steps
            or latest.get('counters', {}).get('ema_updates') != steps
            or latest['counters'].get('global_groups_seen') != 445440 * epoch
            or latest['counters'].get('global_trajectories_seen') != 2227200 * epoch
            or checkpoint != latest):
        raise ValueError('Warmup latest is not its current fully saved four-rank complete epoch')
    if (complete.get('status') != 'FULL_MERGED_EPOCH_COMPLETE'
            or complete.get('checkpoint') != checkpoint or complete.get('metrics') != metrics
            or metrics.get('epoch') != epoch or metrics.get('run_id') != handle['run_id']
            or metrics.get('execution_id') != handle['execution_id']
            or metrics.get('source_commit') != handle['source_commit']
            or metrics.get('ema_updates') != steps or metrics.get('counters') != latest['counters']
            or metrics.get('same_train_test_weight_kind') != 'ema0.99'
            or metrics.get('train_ema', {}).get('groups') != 445440
            or metrics.get('train_ema', {}).get('trajectories') != 2227200
            or metrics.get('test_ema', {}).get('groups') != 512
            or metrics.get('test_ema', {}).get('trajectories') != 2560
            or latest.get('epoch_metrics_sha256') != digest(metrics)):
        raise ValueError('Warmup metrics/checkpoint/complete are not one closed same-EMA epoch')
    return epoch


def seal_warmup_source(base, handle, handle_sha):
    import torch
    from .prefix5_model import warmup_model_weights
    old = remote(handle['base']).resolve(strict=True)
    train = old / 'train'
    latest_path = train / 'latest.json'
    latest_raw = stable_bytes(latest_path)
    latest = json.loads(latest_raw)
    epoch = latest.get('epoch')
    if type(epoch) is not int or not 1 <= epoch <= 200:
        raise ValueError('Latest warmup has no actual complete epoch pointer')
    directory = train / 'epochs' / ('epoch-%04d' % epoch)
    meta_paths = {name: directory / (name + '.json') for name in ('checkpoint', 'metrics', 'complete')}
    raw_metadata = {name: stable_bytes(path) for name, path in meta_paths.items()}
    metadata = {name: json.loads(raw) for name, raw in raw_metadata.items()}
    validate_epoch_closure(latest, metadata['checkpoint'], metadata['metrics'], metadata['complete'], handle)
    source = remote(latest.get('immutable_path', latest['path']))
    if source.is_symlink() or not source.is_file() or train not in source.resolve(strict=True).parents:
        raise ValueError('Warmup source must be a regular file inside the actual handle training directory')
    before = source.stat()
    if before.st_size != latest['bytes']:
        raise ValueError('Actual complete warmup checkpoint bytes differ from producer metadata')
    frozen = base / 'preparation' / ('warmup-source-epoch-%04d.pt' % epoch)
    os.link(source, frozen, follow_symlinks=False)
    descriptor = os.open(frozen.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if signature(before) != signature(frozen.stat()) or stable_bytes(latest_path) != latest_raw:
        raise ValueError('Warmup latest changed while its candidate inode was being sealed; candidate preserved')
    actual_sha = full_sha(frozen)
    if actual_sha != latest['sha256']:
        raise ValueError('Frozen warmup candidate failed full producer SHA; candidate preserved')
    with frozen.open('rb') as stream:
        loaded_before = os.fstat(stream.fileno())
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        loaded_after = os.fstat(stream.fileno())
    if signature(loaded_before) != signature(loaded_after) or signature(loaded_after) != signature(frozen.stat()):
        raise ValueError('Frozen candidate changed during CPU checkpoint load')
    model, buffers, extraction = warmup_model_weights(saved, epoch=epoch, model_kind='ema',
                                                     source_run=handle['run_id'])
    if (saved.get('binding') is None or digest(saved['binding']) != latest['binding_sha256']
            or saved.get('epoch_metrics') != metadata['metrics']
            or saved.get('epoch_metrics_sha256') != latest['epoch_metrics_sha256']):
        raise ValueError('Loaded complete state does not bind its actual producer/metrics metadata')
    feature_sha = saved['binding']['feature_manifest_sha256']
    if stable_bytes(latest_path) != latest_raw or any(stable_bytes(meta_paths[name]) != raw
            for name, raw in raw_metadata.items()):
        raise ValueError('Warmup closure metadata changed during source qualification; candidate preserved')
    result = dict(schema_version=1, status='COMPLETE_WARMUP_EMA_MODEL_ONLY_CPU_QUALIFIED',
        source_checkpoint=str(frozen), path=str(frozen), source_epoch=epoch, source_sha256=actual_sha,
        source_bytes=frozen.stat().st_size, source_inode=frozen.stat().st_ino,
        source_device=frozen.stat().st_dev, model_kind='ema', model_only_import=True, finite=True,
        source_warmup_base=str(old), source_warmup_run=handle['run_id'],
        source_warmup_execution=handle['execution_id'], source_runtime_commit=handle['source_commit'],
        source_handle_sha256=handle_sha, original_checkpoint_path=str(source),
        original_checkpoint_inode=before.st_ino, original_checkpoint_device=before.st_dev,
        original_latest_metadata_sha256=hashlib.sha256(latest_raw).hexdigest(),
        original_epoch_metadata={name:dict(path=str(meta_paths[name]), sha256=hashlib.sha256(raw).hexdigest())
                                 for name, raw in raw_metadata.items()},
        source_feature_manifest_sha256=feature_sha, source_epoch_metrics_sha256=digest(metadata['metrics']),
        native_model_extraction=extraction, full_sha_verified=True, cpu_load_verified=True,
        warmup_optimizer_imported=False, warmup_rng_imported=False, cuda_reload_verified=False,
        old_training_stopped=False, original_latest_may_be_replaced_after_this_seal=True,
        question_max_tokens=QMAX, prefix_tokens=PREFIX_TOKENS, head_max_tokens=WIDTH,
        hidden_size=DIM, observed_unix=time.time())
    del model, buffers, saved
    gc.collect()
    once(base / 'preparation' / 'source-qualification.json', result)
    return result


def _jsonl(path, expected_sha, expected_bytes=None):
    path = remote(path)
    if (expected_bytes is not None and path.stat().st_size != expected_bytes) or full_sha(path) != expected_sha:
        raise ValueError('Pinned identity/index JSONL bytes or full SHA differ: ' + str(path))
    before = path.stat()
    with path.open(encoding='utf8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
    if signature(path.stat()) != signature(before):
        raise ValueError('Immutable identity/index JSONL changed during CPU scan')


def read_fixed_test_source(template):
    composite_path = remote(template['composite_manifest_path'])
    if template['composite_manifest_sha256'] != COMPOSITE_SHA:
        raise ValueError('Only the accepted complete parent composite dataset is authorized')
    composite = checked_json(composite_path, COMPOSITE_SHA)
    if (composite.get('schema_version') != 1 or composite.get('prefix_tokens') != PREFIX_TOKENS
            or composite.get('expected_train_groups') != 445440 or composite.get('expected_test_groups') != 512
            or composite.get('question_max_tokens') != QMAX
            or composite.get('encoder_revision') != 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
            or composite.get('model_files_sha256') != template['model_files_sha256']):
        raise ValueError('Accepted parent composite identity or train/test event counts differ')
    identity_path = remote(composite['question_identity_path'])
    identities = {}
    for row in _jsonl(identity_path, composite['question_identity_sha256']):
        qid = str(row['qid'])
        value = dict(question_hash=row['question_hash'], normalized_question_hash=row['normalized_question_hash'])
        if qid in identities and identities[qid] != value:
            raise ValueError('Accepted canonical identity JSONL contains a conflicting qid')
        identities[qid] = value
    if len(identities) != 169028:
        raise ValueError('Accepted question identity JSONL does not cover the canonical169028 corpus')
    descriptors = [item for item in composite['sources'] if item['split'] == 'test']
    if len(descriptors) != 1:
        raise ValueError('Require the one original fixed random512 G5 source')
    descriptor = descriptors[0]
    if descriptor['expected_groups'] != 512 or descriptor['expected_trajectories'] != 2560:
        raise ValueError('Fixed original G5 descriptor count differs')
    inputs = remote(descriptor.get('inputs_dir', descriptor.get('inputs')))
    scan = checked_json(inputs / 'scan-complete.json', descriptor['scan_sha256'])
    proof = dict(scan.get('source_files', {}))
    proof.update(scan.get('files', {}))
    index_proof = proof['index.jsonl']
    fixed, counts, tids = {}, Counter(), set()
    for row in _jsonl(inputs / 'index.jsonl', index_proof['sha256'], index_proof['bytes']):
        qid, tid = str(row['qid']), str(row['trajectory_id'])
        if tid in tids or qid not in identities or row['question_hash'] != identities[qid]['question_hash']:
            raise ValueError('Fixed512 G5 trajectory/accepted question identity differs')
        if row.get('normalized_question_hash', identities[qid]['normalized_question_hash']) != identities[qid]['normalized_question_hash']:
            raise ValueError('Fixed512 normalized question identity differs')
        tids.add(tid); counts[qid] += 1
        fixed[qid] = dict(qid=qid, **identities[qid])
    if len(fixed) != 512 or len(tids) != 2560 or set(counts.values()) != {5}:
        raise ValueError('Require exactly the original512 identities and five records per question')
    return fixed, dict(composite_manifest_path=str(composite_path), composite_manifest_sha256=COMPOSITE_SHA,
        question_identity_path=str(identity_path), question_identity_sha256=composite['question_identity_sha256'],
        test_source_id=descriptor['source_id'], test_source_event_id=descriptor['source_event_id'],
        test_source_scan_path=str(inputs / 'scan-complete.json'), test_source_scan_sha256=descriptor['scan_sha256'],
        test_source_index_path=str(inputs / 'index.jsonl'), test_source_index_sha256=index_proof['sha256'],
        original_g5_trajectories=2560, feature_tensor_files_read=0, token_array_files_read=0,
        new_generation=0, new_em_scoring=0)


def canonical_pool(template, fixed):
    import pyarrow.parquet as pq
    train = remote(template['canonical_train'])
    if full_sha(train) != template['canonical_train_sha256']:
        raise ValueError('Canonical training parquet full SHA differs')
    common_path = remote(template['common_manifest_path'])
    if template['common_manifest_sha256'] != COMMON_SHA:
        raise ValueError('Common original split manifest SHA differs')
    checked_json(common_path, COMMON_SHA)
    train_before = train.stat()
    records, by_qid, by_semantic = [], {}, {}
    for batch in pq.ParquetFile(train).iter_batches(batch_size=4096,
            columns=['prompt', 'data_source', 'extra_info']):
        for original in batch.to_pylist():
            record = dict(source_index=len(records), **identity(original))
            if record['qid'] in by_qid:
                raise ValueError('Canonical stable question IDs repeat')
            by_qid[record['qid']] = record
            semantic = json.loads(record['qid'])
            if not isinstance(semantic, list) or len(semantic) != 2:
                raise ValueError('Canonical stable qid format differs')
            by_semantic[tuple(semantic)] = record
            records.append(record)
    if signature(train_before) != signature(train.stat()):
        raise ValueError('Canonical training parquet changed during its identity scan')
    if len(records) != 169028:
        raise ValueError('Canonical training corpus must contain exactly169028 original rows')
    mapped, mapping = {}, []
    for qid, source in fixed.items():
        actual = by_qid.get(qid)
        method = 'exact_qid'
        if actual is None:
            try:
                semantic = json.loads(qid)
            except (ValueError, TypeError):
                semantic = None
            if isinstance(semantic, list) and len(semantic) == 2 and all(isinstance(item, str) for item in semantic):
                actual = by_semantic.get(tuple(semantic))
                method = 'exact_decoded_source_and_index_json_format_only'
        if actual is None or any(actual[key] != source[key] for key in ('question_hash', 'normalized_question_hash')):
            raise ValueError('Fixed source qid/hash has no exact canonical identity; no heuristic remapping permitted')
        if actual['qid'] in mapped:
            raise ValueError('Two original fixed test identities map to the same canonical question')
        mapped[actual['qid']] = {key:actual[key] for key in ('qid', 'question_hash', 'normalized_question_hash')}
        if qid != actual['qid']:
            mapping.append(dict(source_qid=qid, canonical_qid=actual['qid'], method=method,
                question_hash=actual['question_hash'], normalized_question_hash=actual['normalized_question_hash']))
    test_text = {row['normalized_question_hash'] for row in mapped.values()}
    if len(mapped) != 512 or len(test_text) != 512:
        raise ValueError('The fixed512 canonical question identities repeat normalized content')
    seen, eligible = set(), []
    for record in records:
        norm = record['normalized_question_hash']
        if record['qid'] in mapped or norm in test_text or norm in seen:
            continue
        eligible.append(record); seen.add(norm)
    if len(eligible) < 5120:
        raise ValueError('Eligible canonical unique candidate pool is too small')
    pool = dict(schema_version=1, role='canonical_nonoverlap_question_pool',
        canonical_train_sha256=template['canonical_train_sha256'], canonical_rows=169028,
        eligible_unique_questions=len(eligible), excluded_diagnostic_questions=512,
        normalized_identity='NFKC_full_original_prompt_whitespace_collapse_case_sensitive',
        fixed_test_qids_sha256=digest(sorted(mapped)), fixed_test_normalized_text_sha256=digest(sorted(test_text)),
        records=eligible, epoch1_cached_qids=[], initial_selection='fresh_permutation_no_teacher_cursor',
        exact_semantic_qid_format_mappings=mapping)
    fixed_manifest = dict(schema_version=1, role='prefix5_fixed_test_question_identities', questions=512,
        trajectories=2560, records=[mapped[qid] for qid in sorted(mapped)],
        exact_semantic_qid_format_mappings=mapping, original_sampling_event_preserved=True)
    return pool, fixed_manifest


def prepare(template_config_path, old_warmup_handle_path):
    """Run once on the execution host; does not claim, stop or launch a trainer."""
    template_config_path, old_warmup_handle_path = remote(template_config_path), remote(old_warmup_handle_path)
    template_raw, handle_raw = stable_bytes(template_config_path), stable_bytes(old_warmup_handle_path)
    template, handle = json.loads(template_raw), json.loads(handle_raw)
    base = remote(template['base'])
    if not str(base).startswith(PREFIX) or not base.is_dir() or base.resolve(strict=True) != base:
        raise ValueError('Use the staged isolated canonical A800 prefix5 namespace')
    directory = base / 'preparation'
    directory.mkdir(exist_ok=True)
    intent = dict(schema_version=1, role='a800_prefix5_cpu_prepare_once', base=str(base),
        run_id=template['run_id'], source_commit=template['source_commit'], process=process_identity(),
        template_config_path=str(template_config_path), template_config_sha256=hashlib.sha256(template_raw).hexdigest(),
        old_warmup_handle_path=str(old_warmup_handle_path), old_warmup_handle_sha256=hashlib.sha256(handle_raw).hexdigest(),
        old_training_stopped=False, cuda_model_or_forward_used=False)
    once(directory / 'source-cpu-prepare-intent.json', intent)
    try:
        source = seal_warmup_source(base, handle, hashlib.sha256(handle_raw).hexdigest())
        fixed, data_source = read_fixed_test_source(template)
        pool, fixed_manifest = canonical_pool(template, fixed)
        fixed_manifest.update(source_identity=data_source)
        fixed_path = directory / 'fixed512-identities.json'
        pool_path = base / 'question-pool' / 'source-pool.json'
        once(fixed_path, fixed_manifest); once(pool_path, pool)
        config = dict(template, base=str(base), world_size=WORLD, prefix_policy_tokens=PREFIX_TOKENS,
            hidden_capture_contract=HIDDEN_CAPTURE_CONTRACT, independent_encoder_forward=False,
            probe_source_checkpoint=source['source_checkpoint'], probe_source_sha256=source['source_sha256'],
            probe_source_bytes=source['source_bytes'], probe_source_inode=source['source_inode'],
            probe_source_epoch=source['source_epoch'], probe_source_model_kind='ema',
            probe_source_run=handle['run_id'], probe_source_cpu_proof=str(directory / 'source-qualification.json'),
            probe_source_cpu_proof_sha256=full_sha(directory / 'source-qualification.json'),
            fixed_test_identity_path=str(fixed_path), fixed_test_identity_sha256=full_sha(fixed_path),
            fixed_test_manifest_sha256=source['source_feature_manifest_sha256'],
            source_question_pool=str(pool_path), source_question_pool_sha256=full_sha(pool_path),
            pool_state_path=str(base / 'question-pool' / 'state.json'),
            prefix_probe_receipt=str(base / 'probe' / 'latest.json'))
        # NVML metadata only. The warmup may still own training cards; launch
        # requires them to be idle after the explicitly authorized switch.
        from .a800_runtime import gpu_inventory, compute_processes
        mapping = gpu_inventory()
        training, auxiliary = config['physical_gpu_indices'], config['auxiliary_gpu_indices']
        if (len(training) != WORLD or len(set(training)) != WORLD or auxiliary != [5, 6]
                or set(training) & set(auxiliary)
                or [mapping[i]['uuid'] for i in training] != config['physical_gpu_uuids']):
            raise ValueError('Physical training UUID mapping differs from the source execution')
        aux_uuids = [mapping[i]['uuid'] for i in auxiliary]
        occupied = {row['uuid'] for row in compute_processes()}
        if occupied & set(aux_uuids) or any(mapping[i]['memory_MiB'] > 64 for i in auxiliary):
            raise RuntimeError('Auxiliary GPU admission rejected; preserve all jobs and source')
        if config.get('auxiliary_gpu_uuids') not in (None, aux_uuids):
            raise ValueError('Prepared auxiliary UUID identity changed')
        config['auxiliary_gpu_uuids'] = aux_uuids
        from .launch import validate_config
        validate_config(config, base)
        once(directory / 'launch-config.json', config)
        complete = dict(schema_version=1, status='A800_PREFIX5_CPU_PREPARE_COMPLETE', base=str(base),
            run_id=template['run_id'], source_commit=template['source_commit'], source_epoch=source['source_epoch'],
            source_checkpoint=source['source_checkpoint'], source_sha256=source['source_sha256'], model_kind='ema',
            model_only_import=True, finite=True, source_qualification=str(directory / 'source-qualification.json'),
            source_qualification_sha256=full_sha(directory / 'source-qualification.json'),
            launch_config=str(directory / 'launch-config.json'), launch_config_sha256=full_sha(directory / 'launch-config.json'),
            fixed_test_identity_path=str(fixed_path), fixed_test_identity_sha256=full_sha(fixed_path),
            source_question_pool=str(pool_path), source_question_pool_sha256=full_sha(pool_path),
            canonical_questions=169028, eligible_unique_questions=pool['eligible_unique_questions'], fixed_test_questions=512,
            data_source=data_source, old_training_stopped=False, optimizer_or_rng_imported=False,
            feature_tensor_files_read=0, token_array_files_read=0, cuda_model_or_forward_used=False,
            new_generation=0, new_em_scoring=0, process=process_identity(), observed_unix=time.time())
        once(directory / 'source-cpu-prepare-complete.json', complete)
        once(directory / 'source-cpu-prepare-exit.json', dict(exit_code=0, process=process_identity(),
            complete_sha256=full_sha(directory / 'source-cpu-prepare-complete.json')))
        return complete
    except BaseException as error:
        once(directory / 'source-cpu-prepare-exit.json', dict(exit_code=1, process=process_identity(),
            error_type=type(error).__name__, error=str(error), failed_candidate_preserved=True,
            same_once_claim_must_not_be_replayed=True, old_training_stopped=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template-config', required=True)
    parser.add_argument('--old-warmup-handle', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.template_config, args.old_warmup_handle), sort_keys=True))


if __name__ == '__main__':
    main()
