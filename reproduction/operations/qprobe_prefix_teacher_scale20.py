"""Remote frozen official G5 dataset: 20 shards x 5120 unique questions.

The first shard reuses the qualified 5120-question teacher cache. The remaining
190 chunks use one long-lived four-rank teacher and owned two-GPU retriever.
No Head, actor optimization, reward fitting or feature encoder is invoked.
Only this attempt's scored, SHA-verified full rollouts may be compacted.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
RUNTIME_SHA = 'fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
OLD_TEACHER = '/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
OLD_IDENTITY_SHA = '88998de7a30455b9135b02cc3c82251fbc1ba1765708244538b6a66c1371e871'
OLD_COMPLETE_SHA = '07f4e2e3937c814637b783468ff1062af7ba701e1ac1e136da2dd56f838b45b1'
TEST_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-16034e59031e4f21a323ebb2556bef7f'
FORMAL_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-e9c4c584107b4624bcc9787aaf24b7a2'
REUSE_CACHE = '/data/shaoshitong/Search-G1-qprobe-prefix-correctness-warmup20-c3aefa891a304aae8f8f40c340fed2f8/features/cache'
REUSE_MANIFEST_SHA = '0d1a004c5677888dad4646c17525577bfe57acf1a413efd97ba8d36a59170298'
QUESTIONS, REUSED, CHUNK, GROUP = 102400, 5120, 512, 5
EXECUTION_KEYS = ('agent_workers', 'max_inflight_trajectories', 'generation_max_num_seqs',
                  'generation_max_batched_tokens', 'retrieval_concurrency')
FAST = dict(zip(EXECUTION_KEYS, (1024, 1024, 256, 32768, 32)))
PACKED_FILES = ('packed_input_ids.npy', 'packed_position_ids.npy', 'packed_policy_mask.npy',
                'offsets.npy', 'lengths.npy', 'index.jsonl')
HELPERS = ('a800_qprobe_native_new512_diagnostic.py', 'qprobe_prefix_correctness_features.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def remote(path):
    p = Path(path)
    assert p.is_absolute() and p.resolve() == p and str(p).startswith(('/data/', '/mnt/', '/backup/'))
    return p


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b'\n'
    if path.exists():
        assert path.read_bytes() == data, 'Immutable dataset file differs: ' + str(path)
        return
    with path.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    sync_dir(path.parent)


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temp.open('xb') as stream: stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path); sync_dir(path.parent)


def progress(root, **value):
    value.update(observed_unix=time.time())
    with (root / 'progress.jsonl').open('ab') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    publish(root / 'progress.json', value)
    print(json.dumps(value, allow_nan=False), flush=True)


def load_helper(filename, expected=None):
    path = Path(__file__).resolve().parent / filename
    assert path.is_file() and not path.is_symlink()
    if expected is not None: assert sha(path) == expected, 'Published support source changed: ' + filename
    name = '_scale20_' + filename[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execution_plan(out):
    plan = read(out / 'scale20-plan.json')
    assert plan['operator_sha256'] == sha(__file__)
    for name, expected in plan['helper_sha256'].items(): assert sha(Path(__file__).resolve().parent / name) == expected
    root = remote(plan['dataset_dir'])
    assert sha(root / 'plan.json') == plan['dataset_plan_sha256']
    assert sha(root / 'selection.json') == plan['selection_sha256'] == sha(out / 'selection.json')
    return plan, root


def old_sources():
    root = remote(OLD_TEACHER)
    assert sha(root / 'identity.json') == OLD_IDENTITY_SHA and sha(root / 'complete.json') == OLD_COMPLETE_SHA
    identity = read(root / 'identity.json'); complete = read(root / 'complete.json')
    assert complete['questions'] == REUSED and complete['trajectories'] == REUSED * GROUP
    assert complete['identity_sha256'] == OLD_IDENTITY_SHA and len(complete['chunks']) == 10
    descriptors = []; groups = []
    for index, entry in enumerate(complete['chunks']):
        path = root / entry['relative_path']; assert path.parent == root and sha(path) == entry['sha256']
        labels = read(path); assert labels['index'] == index and len(labels['groups']) == CHUNK
        raw = labels['raw_rollout']; raw_path = root / raw['relative_path']
        assert raw_path.parent == root and raw_path.stat().st_size == raw['bytes']
        descriptors.append(dict(label_path=str(path), label_sha256=entry['sha256'],
            raw_path=str(raw_path), raw_sha256=raw['sha256'], raw_bytes=raw['bytes']))
        groups.extend(labels['groups'])
    assert [(g['qid'], g['question_hash']) for g in groups] == [(g['qid'], g['question_hash']) for g in identity['questions']]
    assert len({g['qid'] for g in groups}) == REUSED
    return identity, complete, descriptors, groups


def science_equal(actual, original):
    for name in ('model_revision', 'model_files_sha256', 'train_parquet_sha256', 'data_manifest_sha256',
                 'retriever_version', 'feature_spec_sha', 'agent_config', 'max_turns', 'scorer_sources', 'rollout_sources'):
        assert actual[name] == original[name], 'Frozen scientific identity differs: ' + name
    a, b = dict(actual['sampling_config']), dict(original['sampling_config'])
    for key in EXECUTION_KEYS: a.pop(key); b.pop(key)
    assert a == b, 'A sampling/scientific field outside the five declared execution limits changed'
    assert a['temperature'] == a['top_p'] == 1.0 and a['dtype'] == 'bfloat16' and a['n_agent'] == 5


def stage(args, out):
    import numpy as np
    import pyarrow.parquet as pq
    from search_g1.qprobe_features import original_question_and_id, question_text_sha
    from search_g1.qprobe_selection import QuestionPool
    from search_g1.mixed_selection import question_identity
    from search_g1.qprobe_warmup import moments, teacher_identity
    from search_g1.rollout_phase_contract import validate
    source = remote(args.formal_base); runtime = remote(args.runtime_repo)
    assert runtime == out / 'repo', 'Native behavior proof requires a physical, pristine BASE/repo clone'
    original = read(source / 'logs/run-manifest.json')
    old, old_complete, descriptors, old_groups = old_sources()
    test = remote(args.test_base); assert str(test) == TEST_BASE
    heldout = read(test / 'teacher-cache/labels.json'); test_identity = read(test / 'teacher-cache/identity.json')
    assert heldout['questions'] == CHUNK and heldout['trajectories'] == CHUNK * GROUP
    assert heldout['selection_sha256'] == sha(test / 'selection.json')
    assert test_identity['sampling_config'] == old['sampling_config']
    science_equal(test_identity, old)
    assert [(g['qid'], g['question_hash']) for g in heldout['groups']] == [(g['qid'], g['question_hash']) for g in read(test / 'selection.json')['questions']]
    for g in old_groups + heldout['groups']: assert moments(g['strict_em']) == (g['correct_count'], g['mean'], g['std'])
    reuse = remote(REUSE_CACHE); assert sha(reuse/'manifest.json') == REUSE_MANIFEST_SHA
    reuse_manifest = read(reuse/'manifest.json'); reuse_spec = reuse_manifest['spec']
    assert reuse_manifest['count'] == 28160 and reuse_manifest['split_counts'] == dict(train=25600, test=2560)
    assert reuse_spec['encoder_files_sha256'] == old['model_files_sha256']
    assert reuse_spec['encoder_revision'] == old['model_revision'] and reuse_spec['feature_layer'] == 27
    assert reuse_spec['representation'] == 'actual_prompt_policy_prefix5_alltokens_layer27_v1'
    assert reuse_spec['feature_dtype'] == reuse_spec['encoder_compute_dtype'] == 'bfloat16'
    for name, proof in reuse_manifest['files'].items():
        item = reuse/name; assert item.resolve() == item and item.stat().st_size == proof['bytes']
    assert sha(reuse/'index.jsonl') == reuse_manifest['files']['index.jsonl']['sha256']
    expected_rows = [(g['trajectory_ids'][j], split, g['qid'], g['question_hash'], g['strict_em'][j])
        for split, values in (('train',old_groups),('test',heldout['groups'])) for g in values for j in range(GROUP)]
    actual_rows = []
    with (reuse/'index.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line); actual_rows.append((r['trajectory_id'],r['split'],r['qid'],r['question_hash'],r['strict_em']))
    assert actual_rows == expected_rows, 'Existing train/test feature identities or actual EM labels differ'
    train = source / 'data/train.parquet'
    assert sha(train) == old['train_parquet_sha256'] == original['input_sha256']['train.parquet']
    frame = pq.read_table(train, columns=['prompt', 'data_source', 'extra_info']).to_pylist()
    assert len(frame) == 169028
    by_id = {}
    for index, row in enumerate(frame):
        question, qid = original_question_and_id(row)
        assert qid not in by_id
        by_id[qid] = dict(source_index=index, qid=qid, question=question, question_hash=question_text_sha(question))
    test_ids = {g['qid'] for g in heldout['groups']}; test_hashes = {g['question_hash'] for g in heldout['groups']}
    assert len(test_ids) == CHUNK
    test_contents = {question_identity(by_id[qid]['question']) for qid in test_ids}
    selected = []; pool = QuestionPool(QUESTIONS)
    for g in old_groups:
        row = by_id[g['qid']]; assert row['question_hash'] == g['question_hash']
        assert row['qid'] not in test_ids and row['question_hash'] not in test_hashes and question_identity(row['question']) not in test_contents
        assert pool.reserve(row['qid'], row['question']) is not None
        selected.append(row)
    for index in np.random.default_rng(args.seed).permutation(len(frame)):
        question, qid = original_question_and_id(frame[int(index)])
        row = by_id[qid]
        if qid in test_ids or row['question_hash'] in test_hashes or question_identity(question) in test_contents: continue
        if pool.reserve(qid, question) is None: continue
        selected.append(row)
        if len(selected) == QUESTIONS: break
    assert len(selected) == len({r['qid'] for r in selected}) == QUESTIONS
    root = remote(args.dataset_dir); root.mkdir(parents=True, exist_ok=True)
    with (root / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = dict(schema_version=1, seed=args.seed, canonical_questions=169028, reused_questions=REUSED,
            added_questions=QUESTIONS-REUSED, question_count=QUESTIONS,
            sampling='original5120 in original order, then seeded canonical permutation with unique identity/content',
            questions=selected, feature_spec_sha256=old['feature_spec_sha'],
            original_train_sha256=sha(train), old_identity_sha256=OLD_IDENTITY_SHA,
            old_complete_sha256=OLD_COMPLETE_SHA, heldout_selection_sha256=sha(test / 'selection.json'),
            split_proof=dict(disjoint_qid=True, disjoint_question_hash=True, disjoint_normalized_question_content=True,
                fixed_heldout_questions=CHUNK, fixed_heldout_teacher_reused=True, heldout_labels_used_for_training=False))
        once(root / 'selection.json', selection)
        document = copy.deepcopy(read(source / 'logs/resolved-training-config.json')['resolved_config'])
        rollout = document['actor_rollout_ref']['rollout']; assert rollout == old['sampling_config']
        limits = dict(original['infra_rollout'])
        if args.execution_profile == 'fast': limits.update(FAST)
        validate(limits, 4)
        for key in EXECUTION_KEYS: rollout[key] = limits[key]
        assert rollout['gpu_memory_utilization'] == original['vllm_gpu_memory_utilization'] == .25
        protocol = teacher_identity(original, feature_spec_sha=old['feature_spec_sha'], sampling_config=rollout,
            agent_config={k: document['data'][k] for k in old['agent_config']},
            max_turns=document['max_turns'], seed=args.seed, questions=[])
        science_equal(protocol, old)
        protocol['execution_profile'] = dict(name=args.execution_profile, limits=limits,
            allowed_changed_keys=list(EXECUTION_KEYS), original_limits={k: old['sampling_config'][k] for k in EXECUTION_KEYS})
        plan = dict(schema_version=1, role='frozen_official_G5_scale20_dataset', question_count=QUESTIONS,
            trajectories=QUESTIONS*GROUP, reused_questions=REUSED, new_questions=QUESTIONS-REUSED,
            shards=20, questions_per_shard=REUSED, chunks_per_shard=10, questions_per_chunk=CHUNK,
            responses_per_question=GROUP, seed=args.seed, raw_policy=args.raw_policy,
            storage_schema='actual_prompt_policy_prefix5_packed_int32_v1',
            selection_sha256=sha(root / 'selection.json'), original_teacher_cache=OLD_TEACHER,
            old_identity_sha256=OLD_IDENTITY_SHA, old_complete_sha256=OLD_COMPLETE_SHA,
            heldout_base=str(test), heldout_selection_sha256=sha(test / 'selection.json'),
            heldout_labels_sha256=sha(test / 'teacher-cache/labels.json'),
            original_chunk_sources=descriptors, protocol=protocol, strict_protocol=old_complete['protocol'],
            runtime_source_commit=RUNTIME_SHA, split_proof=selection['split_proof'],
            encoder_spec=reuse_spec, reuse=dict(cache_dir=str(reuse),manifest_sha256=REUSE_MANIFEST_SHA,
                train_trajectories=25600,test_trajectories=2560,count=28160,source_ids_and_labels_exact_equal=True),
            resume_policy='immutable completed chunks are fully SHA-verified and skipped; a new execution owns only missing chunks',
            resume_rng_policy='fresh official workers in a new execution; never regenerate accepted G5 events',
            training_forbidden=True)
        once(root / 'plan.json', plan)
    for name, expected in original['source_sha256'].items():
        path = runtime / name; assert path.resolve() == path and sha(path) == expected
    for name, expected in original['initial_model_sha256'].items(): assert sha(Path(original['initial_model']) / name) == expected
    for name in ('logs', 'data', 'teacher-cache', 'teacher-boundary', 'artifacts', 'checkpoints'): (out / name).mkdir(exist_ok=True)
    for name, expected in original['input_sha256'].items():
        target = out / 'data' / name; assert not target.exists(); shutil.copyfile(source / 'data' / name, target); assert sha(target) == expected
    run = copy.deepcopy(original)
    run.update(run_id=uuid.uuid4().hex, execution_id=uuid.uuid4().hex, execution_host=socket.gethostname(),
        started_unix=time.time(), completed_updates=0, diagnostic_only=True,
        diagnostic_kind='scale20_frozen_official_G5_prefix_dataset', actor_training_budget=0, head_training_budget=0,
        source_formal_base=str(source), source_formal_run_id=original['run_id'], source_formal_execution_id=original['execution_id'],
        checkpoint_dir=str(out / 'checkpoints/search-g1-common8'), train_log=str(out / 'logs/diagnostic-teacher.log'),
        gpu_placement_bound=False, training_physical_indices=[], training_gpu_uuids=[], excluded_physical_indices=[],
        infra_rollout=limits)
    for name in ('pipeline_identity', 'full_state_restore', 'resume_path', 'actor_checkpoint', 'reference_checkpoint',
                 'gpu_placement_binding_receipt', 'gpu_placement_binding_sha256', 'gpu_placement_bound_unix',
                 'assigned_gpu_uuids', 'physical_gpu_inventory', 'gpu_placement_semantics', 'ray_tmp_dir_identity'): run.pop(name, None)
    run['question_probe'] = copy.deepcopy(original['question_probe'])
    run['question_probe'].update(teacher_warmup_cache=str(out / 'teacher-cache'), output_dir=str(out / 'artifacts/question-probe-unused'))
    run['ray_tmp_dir'] = '/tmp/g-' + run['execution_id']
    shutil.copyfile(root / 'selection.json', out / 'selection.json')
    run['diagnostic_selection_sha256'] = sha(out / 'selection.json')
    once(out / 'logs/run-manifest.json', run)
    document['data']['train_files'] = str(out / 'data/train.parquet'); document['data']['val_files'] = str(out / 'data/stage.parquet')
    document['trainer']['default_local_dir'] = run['checkpoint_dir']
    if 'question_probe' in document: document['question_probe'] = copy.deepcopy(run['question_probe'])
    once(out / 'teacher-config.json', document)
    execution = dict(dataset_dir=str(root), dataset_plan_sha256=sha(root / 'plan.json'), selection_sha256=sha(root / 'selection.json'),
        operator_sha256=sha(__file__), helper_sha256={name: sha(Path(__file__).resolve().parent / name) for name in HELPERS},
        requested_training_indices=args.train_indices, eligible_training_indices=[0, 1, 2, 3, 4],
        excluded_protected_indices=[7], auxiliary_indices=[5, 6], runtime_source_commit=RUNTIME_SHA)
    once(out / 'scale20-plan.json', execution)
    result = dict(role='scale20_dataset_inputs_staged', run_id=run['run_id'], execution_id=run['execution_id'],
        question_count=QUESTIONS, reused_questions=REUSED, new_questions=QUESTIONS-REUSED, trajectories=QUESTIONS*GROUP,
        selection_sha256=run['diagnostic_selection_sha256'], config_sha256=sha(out / 'teacher-config.json'),
        dataset_dir=str(root), dataset_plan_sha256=execution['dataset_plan_sha256'], execution_profile=protocol['execution_profile'],
        requested_training_indices=args.train_indices, auxiliary_indices=[5, 6], actor_training_budget=0, head_training_budget=0,
        raw_policy=args.raw_policy, checked_unix=time.time())
    once(out / 'stage-complete.json', result); print(json.dumps(result), flush=True)


def chunk_path(root, index): return root / 'shards' / ('shard-%02d' % (index // 10)) / ('chunk-%02d' % (index % 10))


def verify_chunk(path, expected_rows, plan_sha, seen):
    from search_g1.qprobe_warmup import moments, verify_frozen
    value = read(path / 'commit.json'); assert value['dataset_plan_sha256'] == plan_sha
    assert value['questions'] == CHUNK and value['trajectories'] == CHUNK * GROUP
    for name, proof in value['files'].items():
        item = path / name; assert item.is_file() and not item.is_symlink() and item.stat().st_size == proof['bytes'] and sha(item) == proof['sha256']
    labels = read(path / 'labels.json'); groups = labels['groups']; assert len(groups) == CHUNK
    assert [(g['qid'], g['question_hash']) for g in groups] == [(r['qid'], r['question_hash']) for r in expected_rows]
    verify_frozen(labels['frozen_workers']['before'], labels['frozen_workers']['after'], 4)
    ids = []
    for g in groups:
        assert moments(g['strict_em']) == (g['correct_count'], g['mean'], g['std'])
        assert len(g['trajectory_ids']) == len(set(g['trajectory_ids'])) == GROUP
        ids.extend(g['trajectory_ids'])
    assert len(set(ids)) == CHUNK * GROUP and not seen.intersection(ids)
    seen.update(ids)
    return value, labels


def compact_chunk(raw, groups, target, raw_proof, prefix_helper, index):
    import numpy as np
    from search_g1.qprobe_runtime import _complete_generation
    raw['output'].to('cpu')
    _complete_generation(raw['output'], raw['identity'])
    tids = raw['identity']['trajectory_ids']
    assert tids == raw['output'].non_tensor_batch['behavior_trajectory_id'].tolist() == [t for g in groups for t in g['trajectory_ids']]
    assert raw['question_bindings'] == [dict(qid=g['qid'], question_hash=g['question_hash']) for g in groups]
    partial = target.with_name(target.name + '.partial-' + uuid.uuid4().hex); partial.mkdir()
    inputs, positions, masks, rows = [], [], [], []; pad_id = None
    for local, tid in enumerate(tids):
        ids, pos, mask, meta, pad = prefix_helper.actual_prefix(raw, local)
        assert bool(((ids >= 0) & (ids <= np.iinfo(np.int32).max)).all()) and bool((pos <= np.iinfo(np.int32).max).all())
        if pad_id is None: pad_id = pad
        assert pad == pad_id
        g = groups[local // GROUP]; label = g['strict_em'][local % GROUP]
        assert type(label) is int and label in (0, 1)
        rows.append(dict(row=local, trajectory_id=tid, split='train', qid=g['qid'], question_hash=g['question_hash'], strict_em=label,
            source_raw_sha256=raw_proof['sha256'], source_raw_row=local,
            actor_run_id=raw['identity']['run_id'], policy_snapshot_id=raw['identity']['policy_snapshot_id'], **meta))
        inputs.append(ids.astype('<i4')); positions.append(pos.astype('<i4')); masks.append(mask)
    lengths = np.array([r['length'] for r in rows], dtype='<i4')
    offsets = np.concatenate((np.zeros(1, dtype='<i8'), np.cumsum(lengths, dtype='<i8')))
    for name, array in (('packed_input_ids.npy', np.concatenate(inputs)), ('packed_position_ids.npy', np.concatenate(positions)),
                        ('packed_policy_mask.npy', np.concatenate(masks)), ('offsets.npy', offsets), ('lengths.npy', lengths)):
        with (partial / name).open('xb') as stream: np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    with (partial / 'index.jsonl').open('xb') as stream:
        for row, offset in zip(rows, offsets): row['offset'] = int(offset); stream.write(canonical(row) + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    a = np.load(partial / 'packed_input_ids.npy', mmap_mode='r', allow_pickle=False)
    p = np.load(partial / 'packed_position_ids.npy', mmap_mode='r', allow_pickle=False)
    m = np.load(partial / 'packed_policy_mask.npy', mmap_mode='r', allow_pickle=False)
    for row in rows:
        lo = row['offset']; hi = lo + row['length']
        for arr, dtype, key in ((a, '<i8', 'input_ids_sha256'), (p, '<i8', 'position_ids_sha256'), (m, np.bool_, 'policy_mask_sha256')):
            assert hashlib.sha256(arr[lo:hi].astype(dtype).tobytes()).hexdigest() == row[key]
    summary = dict(schema_version=1, ordinal=index, count=len(rows), trajectories=len(rows), total_tokens=int(offsets[-1]), minimum_tokens=int(lengths.min()),
        global_max_tokens=int(lengths.max()), short_prefix_count=sum(r['short_policy_prefix'] for r in rows),
        unknown_zero_policy_count=0, actual_pad_token_id=pad_id, exact_int64_roundtrip_verified=True,
        input_ids_dtype='int32', position_ids_dtype='int32', policy_mask_dtype='bool', attention_mask='all packed entries valid',
        prefix_rule='actual raw prompt and valid context through fifth policy sampling_mask token, no future/decode/tokenize',
        files={name: dict(bytes=(partial / name).stat().st_size, sha256=sha(partial / name)) for name in PACKED_FILES})
    once(partial / 'manifest.json', summary); sync_dir(partial)
    if target.exists():
        assert read(target / 'manifest.json') == summary
        for name, proof in summary['files'].items(): assert sha(target / name) == proof['sha256']
        shutil.rmtree(partial)
    else: partial.rename(target); sync_dir(target.parent)
    return summary


def commit_raw(root, index, raw, groups, raw_path, origin, plan, plan_sha, prefix_helper, expected_rows, *, source_label=None, metrics=None):
    from search_g1.qprobe_warmup import verify_frozen
    target = chunk_path(root, index); target.mkdir(parents=True, exist_ok=True)
    frozen = raw['frozen_workers']; verify_frozen(frozen['before'], frozen['after'], 4)
    expected_protocol = plan['strict_protocol']
    assert {k: raw['contracts'][k] for k in expected_protocol} == expected_protocol
    proof = dict(path=str(raw_path), sha256=sha(raw_path), bytes=raw_path.stat().st_size, origin=origin)
    prefix = compact_chunk(raw, groups, target / 'prefix', proof, prefix_helper, index)
    labels = dict(schema_version=1, groups=groups, questions=CHUNK, trajectories=CHUNK*GROUP,
        identity=raw['identity'], contracts=raw['contracts'], frozen_workers=frozen, raw_rollout=proof,
        source_label=source_label, actual_async_rollout=metrics or {},
        actual_actor_optimizer_updates=0, actual_head_optimizer_updates=0)
    once(target / 'labels.json', labels)
    files = {str(Path('prefix') / name): item for name, item in prefix['files'].items()}
    for name in ('labels.json', 'prefix/manifest.json'): files[name] = dict(bytes=(target / name).stat().st_size, sha256=sha(target / name))
    result = dict(schema_version=1, role='actual_frozen_official_G5_scored_prefix_chunk', index=index,
        shard=index//10, shard_chunk=index%10, dataset_plan_sha256=plan_sha, questions=CHUNK, trajectories=CHUNK*GROUP,
        raw_rollout=proof, raw_policy=plan['raw_policy'], files=files,
        total_prefix_tokens=prefix['total_tokens'], minimum_tokens=prefix['minimum_tokens'], global_max_tokens=prefix['global_max_tokens'],
        actual_zero_optimizer_frozen_readback=True, actual_actor_optimizer_updates=0, actual_head_optimizer_updates=0,
        strict_em_histogram={str(k): sum(g['correct_count']==k for g in groups) for k in range(6)},
        original_qualified_cache_unchanged=origin=='reused_qualified_teacher')
    once(target / 'commit.json', result)
    # The accepted commit seals labels, exact prefix round trips and all file
    # hashes before removing only a new raw file under this chunk's own path.
    verify_chunk(target, expected_rows, plan_sha, set())
    if origin == 'new_official_teacher' and plan['raw_policy'] == 'compact-after-verified':
        assert raw_path == target / 'rollout.pt' and raw_path.resolve().parent == target.resolve()
        raw_path.unlink(); sync_dir(target)
        once(target / 'raw-compacted.json', dict(raw_sha256=proof['sha256'], raw_bytes=proof['bytes'],
            accepted_commit_sha256=sha(target / 'commit.json'), removed_after_labels_and_prefix_full_sha=True))
    return result


def scale_teacher_fit(trainer, settings):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
    from search_g1.rollout_phase_contract import configure_generation, scalar_metrics
    from search_g1.qprobe_runtime import QProbeCandidateStream, _complete_generation, _contracts
    from search_g1.mixed_runtime import strict_vectors
    from search_g1.qprobe_warmup import moments, verify_frozen, teacher_identity
    from search_g1.training_state import atomic_save
    from reproduction.common8.behavior_runtime import prepare_identity, attach_generation_identity
    out = remote(settings['output_dir']); execution, root = execution_plan(out)
    plan = read(root / 'plan.json'); rows = read(root / 'selection.json')['questions']; plan_sha = sha(root / 'plan.json')
    run = read(out / 'logs/run-manifest.json'); assert run['diagnostic_only'] and run['actor_training_budget'] == run['head_training_budget'] == 0
    assert len(rows) == QUESTIONS and len(trainer.train_dataloader.dataset.dataframe) == 169028
    prefix_helper = load_helper(HELPERS[1], execution['helper_sha256'][HELPERS[1]])
    trainer.global_steps = 1
    config = GenerationConfig(max_turns=trainer.config.max_turns, max_start_length=trainer.config.data.max_start_length,
        max_prompt_length=trainer.config.data.max_prompt_length, max_response_length=trainer.config.data.max_response_length,
        max_obs_length=trainer.config.data.max_obs_length, num_gpus=4, no_think_rl=trainer.config.algorithm.no_think_rl,
        search_url=trainer.config.retriever.url, topk=trainer.config.retriever.topk)
    configure_generation(config, trainer.config.actor_rollout_ref.rollout); assert config.async_phase_enabled is True
    bound = teacher_identity(run, feature_spec_sha=plan['protocol']['feature_spec_sha'],
        sampling_config=OmegaConf.to_container(trainer.config.actor_rollout_ref.rollout, resolve=True),
        agent_config={k: getattr(trainer.config.data, k) for k in plan['protocol']['agent_config']},
        max_turns=trainer.config.max_turns, seed=plan['seed'], questions=[])
    science_equal(bound, plan['protocol']); assert bound['sampling_config'] == plan['protocol']['sampling_config']
    once(out / 'teacher-cache/identity.json', bound)
    manager = LLMGenerationManager(tokenizer=trainer.tokenizer, actor_rollout_wg=trainer.actor_rollout_wg, config=config)
    stream = QProbeCandidateStream(trainer.train_dataloader)
    lock = (root / 'writer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        before = trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(out / 'teacher-boundary/before'))
        verify_frozen(before, before, 4)
        completed = []; seen = set(); new_count = 0; generation_seconds = 0.; initial_time = time.monotonic(); new_wall_start = None
        for index in range(QUESTIONS // CHUNK):
            target = chunk_path(root, index); part = rows[index*CHUNK:(index+1)*CHUNK]
            if (target / 'commit.json').exists():
                receipt, _ = verify_chunk(target, part, plan_sha, seen); completed.append(receipt)
                if receipt['raw_rollout']['origin'] == 'new_official_teacher' and plan['raw_policy'] == 'compact-after-verified' and (target / 'rollout.pt').exists():
                    assert sha(target / 'rollout.pt') == receipt['raw_rollout']['sha256']
                    (target / 'rollout.pt').unlink(); sync_dir(target)
                progress(root, stage='VERIFIED_COMPLETED_CHUNK_REUSED', completed_questions=len(completed)*CHUNK,
                    total_questions=QUESTIONS, completed_chunks=len(completed), resumed_without_generation=True)
                continue
            target.mkdir(parents=True, exist_ok=True)
            if index < REUSED // CHUNK:
                source = plan['original_chunk_sources'][index]
                labels = read(source['label_path']); assert sha(source['label_path']) == source['label_sha256']
                assert sha(source['raw_path']) == source['raw_sha256']
                raw = torch.load(source['raw_path'], map_location='cpu', weights_only=False)
                receipt = commit_raw(root, index, raw, labels['groups'], Path(source['raw_path']), 'reused_qualified_teacher',
                    plan, plan_sha, prefix_helper, part, source_label=dict(path=source['label_path'], sha256=source['label_sha256']))
            else:
                raw_path = target / 'rollout.pt'
                if raw_path.exists():
                    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
                    assert raw['question_bindings'] == [dict(qid=r['qid'], question_hash=r['question_hash']) for r in part]
                    groups = raw['scored_groups']; metrics = raw['actual_async_rollout']
                else:
                    candidates = DataProto.from_dict(tensors={'qprobe_source_index': torch.tensor([r['source_index'] for r in part])},
                        non_tensors={'candidate_original_question': np.array([r['question'] for r in part], dtype=object),
                                     'candidate_stable_id': np.array([r['qid'] for r in part], dtype=object)})
                    parent = stream.materialize(candidates, list(range(CHUNK)))
                    identity = prepare_identity(parent, 0, repeats=GROUP, teacher_warmup=True, candidate_offset=index*CHUNK)
                    repeated = parent.repeat(repeat_times=GROUP, interleave=True)
                    gen = repeated.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids']); attach_generation_identity(gen, identity, do_sample=True)
                    manager.timing_raw = {}; start = time.monotonic()
                    if new_wall_start is None: new_wall_start = start
                    output = manager.run_llm_loop(gen_batch=gen, initial_input_ids=gen.batch['input_ids'][:, -config.max_start_length:].clone().long())
                    elapsed = time.monotonic() - start; generation_seconds += elapsed; new_count += CHUNK
                    _complete_generation(output, identity)
                    contracts = _contracts(trainer, output, run); vectors = strict_vectors(parent, output, trainer.tokenizer)
                    assert len(vectors) == CHUNK
                    after = trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(out / 'teacher-boundary' / ('chunk-%03d' % index)))
                    frozen = verify_frozen(before, after, 4)
                    groups = []
                    for local, (row, vector) in enumerate(zip(part, vectors)):
                        count, mean, std = moments(vector)
                        groups.append(dict(qid=row['qid'], question_hash=row['question_hash'], strict_em=vector,
                            correct_count=count, mean=mean, std=std, trajectory_ids=identity['trajectory_ids'][local*GROUP:(local+1)*GROUP]))
                    metrics = dict(scalar_metrics(output), measured_generation_seconds=elapsed, timing_raw=manager.timing_raw)
                    raw = dict(output=output, identity=identity, contracts=contracts, frozen_workers=frozen,
                        question_bindings=[dict(qid=r['qid'], question_hash=r['question_hash']) for r in part],
                        scored_groups=groups, actual_async_rollout=metrics)
                    if raw_path.with_name(raw_path.name+'.partial').exists():
                        old_partial=raw_path.with_name(raw_path.name+'.partial')
                        old_partial.rename(raw_path.with_name(raw_path.name+'.interrupted-'+uuid.uuid4().hex))
                        sync_dir(target)
                    atomic_save(raw_path, raw)
                receipt = commit_raw(root, index, raw, groups, raw_path, 'new_official_teacher', plan, plan_sha, prefix_helper, part, metrics=metrics)
            verify_chunk(target, part, plan_sha, seen); completed.append(receipt)
            remaining = QUESTIONS - len(completed)*CHUNK
            rate = new_count/generation_seconds if generation_seconds else None
            wall_rate = new_count/(time.monotonic()-new_wall_start) if new_wall_start is not None else None
            progress(root, stage='ACTUAL_G5_PREFIX_CHUNK_DURABLE', completed_questions=len(completed)*CHUNK,
                total_questions=QUESTIONS, completed_trajectories=len(completed)*CHUNK*GROUP, completed_chunks=len(completed),
                completed_shards=len(completed)//10, current_shard=index//10, current_chunk=index%10,
                new_questions_generated_this_execution=new_count, measured_generation_seconds=generation_seconds,
                observed_generation_questions_per_second=rate, generation_only_remaining_seconds=remaining/rate if rate else None,
                observed_end_to_end_new_questions_per_second=wall_rate, estimated_remaining_seconds=remaining/wall_rate if wall_rate else None,
                execution_elapsed_seconds=time.monotonic()-initial_time, last_chunk_raw_bytes=receipt['raw_rollout']['bytes'],
                last_chunk_compact_bytes=sum(p['bytes'] for p in receipt['files'].values()),
                actual_actor_optimizer_updates=0, actual_head_optimizer_updates=0)
            del raw
        after = trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(out / 'teacher-boundary/after'))
        frozen = verify_frozen(before, after, 4)
        restored = trainer.actor_rollout_wg.restore_frozen_teacher_rng(str(out / 'teacher-boundary/before'))
        assert len(restored) == 4 and all(item['rng_equal'] for item in restored)
        assert len(seen) == QUESTIONS*GROUP and len(completed) == 200
        chunks_manifest=[dict(index=i, relative_path=str((chunk_path(root, i)/'commit.json').relative_to(root)),
            sha256=sha(chunk_path(root, i)/'commit.json')) for i in range(200)]
        once(root/'chunk-manifest.json',dict(schema_version=1,dataset_plan_sha256=plan_sha,chunks=chunks_manifest))
        spec=copy.deepcopy(plan['encoder_spec'])
        spec['global_max_tokens']=max(spec['global_max_tokens'],max(c['global_max_tokens'] for c in completed))
        compact_manifest=dict(schema_version=1,stage='PREFIX_TEACHER_SCALE20_COMPACT_COMPLETE',runtime_commit=RUNTIME_SHA,
            spec=spec,train_questions=QUESTIONS,train_trajectories=QUESTIONS*GROUP,new_questions=QUESTIONS-REUSED,
            new_trajectories=(QUESTIONS-REUSED)*GROUP,reuse=plan['reuse'],split_proof=plan['split_proof'],
            dataset_plan_sha256=plan_sha,selection_sha256=plan['selection_sha256'],
            shards=[dict(ordinal=i,path=str(chunk_path(root,i)/'prefix'),
                manifest_sha256=sha(chunk_path(root,i)/'prefix/manifest.json'),count=CHUNK*GROUP) for i in range(10,200)])
        for i in range(200): assert read(chunk_path(root,i)/'prefix/manifest.json')['actual_pad_token_id']==spec['pad_token_id']
        once(root/'compact-manifest.json',compact_manifest)
        summary = dict(schema_version=1, role='FROZEN_OFFICIAL_SCALE20_G5_PREFIX_DATASET_COMPLETE',
            question_count=QUESTIONS, questions=QUESTIONS, trajectories=len(seen), reused_questions=REUSED, new_questions=QUESTIONS-REUSED,
            chunks=200, shards=20, dataset_plan_sha256=plan_sha, selection_sha256=plan['selection_sha256'],
            split_proof=plan['split_proof'], fixed_heldout_questions=CHUNK, heldout_regenerated=False,
            raw_policy=plan['raw_policy'], total_prefix_tokens=sum(c['total_prefix_tokens'] for c in completed),
            global_max_tokens=spec['global_max_tokens'], minimum_tokens=min(c['minimum_tokens'] for c in completed),
            actual_actor_optimizer_updates=0, actual_head_optimizer_updates=0,
            chunk_manifest_path=str(root/'chunk-manifest.json'),chunk_manifest_sha256=sha(root/'chunk-manifest.json'),
            compact_manifest_path=str(root/'compact-manifest.json'),compact_manifest_sha256=sha(root/'compact-manifest.json'),
            old_identity_sha256=OLD_IDENTITY_SHA, old_complete_sha256=OLD_COMPLETE_SHA,
            protocol_sha256=hashlib.sha256(canonical(plan['protocol'])).hexdigest())
        once(root / 'complete.json', summary)
        once(out / 'teacher-cache/frozen-final.json', dict(frozen_workers=frozen, worker_rng_restored=restored))
        once(out / 'teacher-complete.json', dict(summary, run_id=run['run_id'], execution_id=run['execution_id'],
            dataset_dir=str(root), measured_new_questions_this_execution=new_count, measured_generation_seconds=generation_seconds,
            elapsed_seconds=time.monotonic()-initial_time, checked_unix=time.time()))
        progress(root, stage=summary['role'], completed_questions=QUESTIONS, completed_trajectories=QUESTIONS*GROUP, completed_chunks=200)
    finally: fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def teacher(args, out, diagnostic):
    """The published diagnostic ownership/bootstrap, with actual four-card routing."""
    import ray
    from reproduction.a800.owned_processes import identity, publish as owned_publish
    from reproduction.common8 import native_allocation as native
    execution, root = execution_plan(out)
    assert not ray.is_initialized() and os.environ.get('SLURM_JOB_ID') is None
    run = read(out / 'logs/run-manifest.json'); staged = read(out / 'stage-complete.json')
    assert run['diagnostic_only'] and run['diagnostic_selection_sha256'] == sha(out/'selection.json') == staged['selection_sha256']
    assert staged['config_sha256'] == sha(out/'teacher-config.json')
    # Read NVML and bind the requested idle subset through the original receipt
    # writer/admission. Only resource selection is adapted in this process.
    original_selector = native.select_idle_local_placement
    def authorized_selector(rows, applications):
        physical = native._local_inventory(rows); occupied = native._local_application_indices(physical, applications)
        idle = {i for i, r in physical.items() if i not in occupied and r['used_mib'] <= native.LOCAL_IDLE_MAX_USED_MIB and r['util_percent'] == 0}
        assert {5, 6} <= idle
        requested = execution['requested_training_indices']
        chosen = requested if requested is not None else sorted(idle.intersection(execution['eligible_training_indices']))[:4]
        assert len(chosen) == len(set(chosen)) == 4 and set(chosen) <= idle.intersection(execution['eligible_training_indices'])
        return list(chosen), sorted(set(physical)-set(chosen)-{5, 6})
    os.environ.update(SEARCH_G1_BASE=str(out), SEARCH_G1_PROFILE='common8', SEARCH_G1_TRAINING_WORLD_SIZE='4',
        PYTHONPATH=str(args.runtime_repo), NCCL_NVLS_ENABLE='0', VLLM_ATTENTION_BACKEND='XFORMERS',
        TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='8', PYTHONUNBUFFERED='1', NO_PROXY='127.0.0.1,localhost',
        C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include')
    for key in ('RAY_ADDRESS', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'): os.environ.pop(key, None)
    native.select_idle_local_placement = authorized_selector
    try: run = native.bind_idle_local_placement(out)
    finally: native.select_idle_local_placement = original_selector
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, run['training_physical_indices']))
    once(out/'teacher-generation-once.json', dict(run_id=run['run_id'], execution_id=run['execution_id'],
        supervisor=identity(os.getpid()), training_indices=run['training_physical_indices'], actor_training_forbidden=True, head_training_forbidden=True, claimed_unix=time.time()))
    once(out/'teacher-launch.json', dict(run_id=run['run_id'], execution_id=run['execution_id'], supervisor=identity(os.getpid()),
        argv=sys.argv, operator_sha256=sha(__file__), runtime_repo=str(args.runtime_repo), dataset_dir=str(root),
        stdout_stderr_policy='caller streams and real owned child log retained', launched_unix=time.time()))
    locks = []; lock_root = Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks'); lock_root.mkdir(parents=True, exist_ok=True)
    try:
        for gpu_uuid in run['assigned_gpu_uuids'].values():
            lock = (lock_root/(gpu_uuid+'.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(lock)
        admission = native.admit(out)
        assert admission['pipeline_identity']['pid'] == os.getpid() and admission['node'] == socket.gethostname()
        assert admission['actual_allocated_indices'] == sorted(run['training_physical_indices']+[5, 6])
        run['pipeline_identity'] = admission['pipeline_identity']; owned_publish(out/'logs/run-manifest.json', run)
        once(out/'logs/checkpoint-disk-admission.json', dict(accepted=True, run_id=run['run_id'],
            role='diagnostic_zero_training_checkpoint_budget', save_steps=[], required_free_bytes=0,
            actual_free_bytes=shutil.disk_usage(out).free, diagnostic_only=True, checked_unix=time.time()))
        required = 10*1024**3 if read(root/'plan.json')['raw_policy']=='compact-after-verified' else 160*1024**3
        assert shutil.disk_usage(root).free >= required, 'Insufficient actual remote dataset storage'
        from reproduction.common8.run_pipeline import Pipeline
        pipeline = Pipeline(out, run)
        try:
            pipeline.stage('SCALE20_STAGING_ORIGINAL_LOCAL_RETRIEVAL')
            stager = pipeline.start('diagnostic-retrieval-stage', [sys.executable, '-m', 'reproduction.common8.stage_retrieval', '--base', str(out)], [])
            pipeline.wait('diagnostic-retrieval-stage', stager)
            nvidia = importlib.util.find_spec('nvidia'); torch_spec = importlib.util.find_spec('torch')
            assert nvidia and nvidia.submodule_search_locations and torch_spec and torch_spec.origin
            libs = list(dict.fromkeys(p for r in nvidia.submodule_search_locations for p in sorted(Path(r).glob('*/lib')) if p.is_dir())) + [Path(torch_spec.origin).parent/'lib']
            retriever = pipeline.start('diagnostic-retriever', [sys.executable, '-m', 'reproduction.common8.retriever'], [5, 6],
                {'PYTHONPATH': run['retriever_gpu_lib']+':'+str(args.runtime_repo), 'LD_LIBRARY_PATH': ':'.join(map(str, libs))})
            health = pipeline.endpoint('diagnostic-retriever', retriever, '18010/health')
            assert health['ok'] is True and health['run_id']==run['run_id'] and health['execution_id']==run['execution_id']
            assert health['shard_count']==2 and health['slurm_job_id'] is None and health['admission_sha256']==sha(out/'logs/retrieval-admission.json')
            once(out/'diagnostic-retriever-ready.json', dict(health=health, owned_leader=pipeline.owned.jobs['diagnostic-retriever']['leader'], checked_unix=time.time()))
            ray_root=Path(run['ray_tmp_dir']); ray_root.mkdir(mode=0o700, exist_ok=False)
            try:
                ray.init(address='local', num_cpus=16, num_gpus=4, object_store_memory=8*1024**3, _temp_dir=str(ray_root),
                    runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM':'true', 'NCCL_DEBUG':'WARN'}})
                import pickle
                from search_g1.qprobe_driver_gpu import capture_cpu_rng, cpu_rng_sha256, child_environment, _cluster_identity
                context=ray.get_runtime_context(); cluster=_cluster_identity(context.gcs_address,context.namespace)
                state=capture_cpu_rng(); route=dict(gpu_uuid=run['training_gpu_uuids'][0], physical_index=run['training_physical_indices'][0])
                parent={k:v for k,v in identity(os.getpid()).items() if k in ('pid','start_ticks','sid','uid')}
                payload=dict(route=route, cluster=cluster, cpu_rng=state, cpu_rng_sha256=cpu_rng_sha256(state), parent_identity=parent)
                raw=pickle.dumps(payload,protocol=4); assert len(raw)<=4*1024**2
                with (out/'diagnostic-driver-payload.pkl').open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                once(out/'diagnostic-driver-once.json', dict(run_id=run['run_id'], execution_id=run['execution_id'], parent_identity=parent,
                    route=route, payload_sha256=sha(out/'diagnostic-driver-payload.pkl'), allocation_sha256=sha(out/'logs/native-allocation-admission.json'),
                    placement_sha256=run['gpu_placement_binding_sha256'], formal_training_claim_created=False, claimed_unix=time.time()))
                child=pipeline.owned.start('diagnostic-teacher-driver', [sys.executable,str(Path(__file__).resolve()), '--action','teacher-child',
                    '--runtime-repo',str(args.runtime_repo),'--formal-base',str(args.formal_base),'--output-dir',str(out)],
                    child_environment(route,pipeline.env),out/'logs/diagnostic-teacher-driver.log',Path(args.runtime_repo))
                print(json.dumps(dict(stage='SCALE20_FRESH_BIRTH_UUID_DRIVER_STARTED',pid=child.pid,gpu_uuid=route['gpu_uuid'],
                    log=str(out/'logs/diagnostic-teacher-driver.log'))),flush=True)
                pipeline.wait('diagnostic-teacher-driver',child)
            finally: ray.shutdown()
        finally:
            pipeline.owned.stop_all()
            if (out/'logs/corpus-cache-claim.json').exists():
                from reproduction.a800_common.cache import cleanup
                cleanup(out)
        assert (out/'teacher-complete.json').exists()
        once(out/'teacher-exit.json',dict(exit_code=0,run_id=run['run_id'],execution_id=run['execution_id'],
            actual_actor_optimizer_updates=0,actual_head_optimizer_updates=0,checked_unix=time.time()))
    finally:
        for lock in reversed(locks): fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',required=True,choices=('stage','teacher','teacher-child','observe'))
    p.add_argument('--runtime-repo'); p.add_argument('--formal-base',default=FORMAL_BASE); p.add_argument('--output-dir',required=True)
    p.add_argument('--dataset-dir'); p.add_argument('--test-base',default=TEST_BASE)
    p.add_argument('--seed',type=int,default=2026101603)
    p.add_argument('--execution-profile',choices=('original','fast'),default='fast')
    p.add_argument('--raw-policy',choices=('compact-after-verified','retain-full'))
    p.add_argument('--train-indices',type=lambda x:[int(i) for i in x.split(',')])
    args=p.parse_args(); out=remote(args.output_dir)
    if args.action=='observe':
        execution,root=execution_plan(out)
        result={name:read(path) for name,path in (('progress',root/'progress.json'),('complete',root/'complete.json'),
            ('teacher_exit',out/'teacher-exit.json'),('teacher_failure',out/'teacher-failure.json'),
            ('driver_failure',out/'diagnostic-driver-failure.json')) if path.exists()}
        if 'complete' in result: result['complete'].pop('chunks_manifest',None)
        result.update(dataset_dir=str(root),output_dir=str(out)); print(json.dumps(result,allow_nan=False)); return
    assert args.runtime_repo
    diagnostic=load_helper(HELPERS[0]); _,out=diagnostic.bind(args)
    if args.action=='stage':
        assert args.dataset_dir and args.raw_policy
        if args.train_indices is not None:
            assert len(args.train_indices)==len(set(args.train_indices))==4 and set(args.train_indices)<=set(range(5))
        os.environ['CUDA_VISIBLE_DEVICES']=''; stage(args,out); return
    execution_plan(out)
    diagnostic.teacher_fit=scale_teacher_fit
    try:
        if args.action=='teacher-child': diagnostic.teacher_child(args,out)
        else: teacher(args,out,diagnostic)
    except BaseException as error:
        failure=dict(role='scale20_frozen_official_teacher_failure',error=repr(error),traceback=traceback.format_exc(),
            pid=os.getpid(),runtime_source_commit=RUNTIME_SHA,automatic_retry=False,failed_unix=time.time())
        target=out/('diagnostic-driver-failure.json' if args.action=='teacher-child' else 'teacher-failure.json')
        once(target,failure)
        if args.action=='teacher' and not (out/'teacher-exit.json').exists(): once(out/'teacher-exit.json',dict(exit_code=1,**failure))
        raise


if __name__=='__main__': main()
