"""Append a typed, self-contained prefix5 dataset to an existing HF archive repo.

Build is CPU only and reads sealed IDs, positions, masks and strict-EM labels.
Upload uses bounded Mac memory buffers; credentials are hidden stdin only.
No rollout, scoring, feature/model readback, fitting or GPU work is performed.
"""
import argparse
import collections
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
sys.dont_write_bytecode = True

REPOSITORY = 'Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d'
CONFIG = 'a800_frozen_g5_prefix5_scale20_20261016'
PREFIX = 'additions/' + CONFIG
RUNTIME = 'fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
COMMON = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
RUN = '62407ad0ed404586998439ca1f55b902'
EXECUTION = 'dd3032e313e14ce2b9c8fea996aaf721'
BASE = '/data/shaoshitong/Search-G1-a800-fresh200-0850db2977bd4801b9ced300b0ab387f'
COMPLETE_SHA = 'ea1aecb682d8faf184f7b4c32d52bb48b88aeb1bc483242993cf349c51a85fcf'
PLAN_SHA = 'ea8640c5bba11e1b2689614efa45e5eb53d43276417abb1bbadd112a1ab9b2c3'
SELECTION_SHA = '491bd9de6d4d15d8a388b8fe32e17b3f85121e0f4ca5ae712a0ea1427e71e938'
REUSE_SHA = '0d1a004c5677888dad4646c17525577bfe57acf1a413efd97ba8d36a59170298'
OLD_IDENTITY_SHA = '88998de7a30455b9135b02cc3c82251fbc1ba1765708244538b6a66c1371e871'
OLD_COMPLETE_SHA = '07f4e2e3937c814637b783468ff1062af7ba701e1ac1e136da2dd56f838b45b1'
TEST_LABEL_SHA = 'd7d056477666558a2dd0fbaeb0f7ee8aff679e80c5730fbb0a3a8c684395740b'
INPUT_FILES = ('packed_input_ids.npy', 'packed_position_ids.npy', 'packed_policy_mask.npy',
               'offsets.npy', 'lengths.npy', 'index.jsonl')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def progress(out, **value):
    value['checked_unix'] = time.time()
    with (out / 'progress.jsonl').open('ab') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(value, allow_nan=False), flush=True)


def remote_path(path):
    p = Path(path)
    assert p.is_absolute() and p.resolve() == p and str(p).startswith('/data/shaoshitong/')
    return p


def packed(directory, proofs):
    import numpy as np
    directory = remote_path(directory)
    for name in INPUT_FILES:
        item = directory / name
        proof = proofs[name]
        assert item.is_file() and not item.is_symlink()
        assert item.stat().st_size == proof['bytes'] and sha(item) == proof['sha256']
    arrays = {name: np.load(directory / name, mmap_mode='r', allow_pickle=False)
              for name in INPUT_FILES if name.endswith('.npy')}
    rows = [json.loads(line) for line in (directory / 'index.jsonl').open()]
    lengths, offsets = arrays['lengths.npy'], arrays['offsets.npy']
    assert len(offsets) == len(rows) + 1 and len(lengths) == len(rows) and int(offsets[0]) == 0
    assert np.array_equal(np.diff(offsets), lengths) and bool((lengths > 0).all())
    assert all(len(arrays[n]) == int(offsets[-1]) for n in INPUT_FILES[:3])
    return rows, arrays


def build(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    runtime = remote_path(args.runtime_repo)
    assert str(runtime) == BASE + '/repo'
    assert subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip() == RUNTIME
    assert not subprocess.check_output(['git', '-C', str(runtime), 'status', '--porcelain']).strip()
    sys.path.insert(0, str(runtime))
    from search_g1.qprobe_features import original_question_and_id, question_text_sha
    from search_g1.mixed_selection import question_identity
    from search_g1.qprobe_warmup import verify_frozen, moments

    root = remote_path(args.dataset_dir)
    assert str(root) == BASE + '/dataset'
    assert sha(root / 'complete.json') == COMPLETE_SHA and sha(root / 'plan.json') == PLAN_SHA
    assert sha(root / 'selection.json') == SELECTION_SHA
    done, plan, selection = read(root / 'complete.json'), read(root / 'plan.json'), read(root / 'selection.json')
    assert done['questions'] == 102400 and done['trajectories'] == 512000 and done['chunks'] == 200
    assert done['actual_actor_optimizer_updates'] == done['actual_head_optimizer_updates'] == 0
    assert done['old_identity_sha256'] == OLD_IDENTITY_SHA and done['old_complete_sha256'] == OLD_COMPLETE_SHA
    assert sha(root / 'chunk-manifest.json') == done['chunk_manifest_sha256']
    assert sha(root / 'compact-manifest.json') == done['compact_manifest_sha256']
    chunks = read(root / 'chunk-manifest.json')['chunks']
    assert [c['index'] for c in chunks] == list(range(200))
    run = read(Path(BASE) / 'logs/run-manifest.json')
    assert run['run_id'] == RUN and run['execution_id'] == EXECUTION and run['model_revision'] == REVISION
    assert plan['protocol']['data_manifest_sha256'] == COMMON
    canonical_train = Path(BASE) / 'data/train.parquet'
    assert sha(canonical_train) == plan['protocol']['train_parquet_sha256']
    frame = pq.read_table(canonical_train, columns=['prompt', 'data_source', 'extra_info']).to_pylist()
    assert len(frame) == 169028
    questions = {}
    for item in frame:
        question, qid = original_question_and_id(item)
        assert qid not in questions
        questions[qid] = (question, item['data_source'], question_text_sha(question))
    del frame
    test_base = remote_path(plan['heldout_base'])
    assert sha(test_base / 'teacher-cache/labels.json') == TEST_LABEL_SHA == plan['heldout_labels_sha256']
    test_labels = read(test_base / 'teacher-cache/labels.json')
    test_groups = test_labels['groups']
    assert len(test_groups) == 512 and test_labels['trajectories'] == 2560
    train_qids = [r['qid'] for r in selection['questions']]
    test_qids = [g['qid'] for g in test_groups]
    assert len(train_qids) == len(set(train_qids)) == 102400 and len(set(test_qids)) == 512
    assert set(train_qids).isdisjoint(test_qids)
    assert len({questions[q][2] for q in train_qids}) == 102400
    assert len({question_identity(questions[q][0]) for q in train_qids}) == 102400
    assert {questions[q][2] for q in train_qids}.isdisjoint(questions[q][2] for q in test_qids)
    assert {question_identity(questions[q][0]) for q in train_qids}.isdisjoint(question_identity(questions[q][0]) for q in test_qids)
    model = Path(run['initial_model'])
    tokenizer_hashes = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json'):
        path = model / name
        if path.exists():
            tokenizer_hashes[name] = sha(path)
            if name in run['initial_model_sha256']:
                assert tokenizer_hashes[name] == run['initial_model_sha256'][name]
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    out = remote_path(args.output_dir)
    assert str(out).startswith('/data/shaoshitong/Search-G1-hf-scale20-prefix5-')
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    data = out / PREFIX / 'data'
    data.mkdir(parents=True)
    schema = pa.schema([
        ('question_id', pa.string()), ('question_hash', pa.string()), ('question_text', pa.string()),
        ('data_source', pa.string()), ('trajectory_id', pa.string()), ('response_index', pa.int8()),
        ('split', pa.string()), ('source_origin', pa.string()), ('actual_prompt_text', pa.string()),
        ('context_through_prefix_text', pa.string()), ('generated_policy_prefix_text', pa.string()),
        ('input_ids', pa.list_(pa.int32())), ('position_ids', pa.list_(pa.int32())),
        ('policy_mask', pa.list_(pa.bool_())), ('attention_mask', pa.list_(pa.bool_())),
        ('final_binary_em', pa.int8()), ('group_correct_count', pa.int8()), ('group_is_mixed', pa.bool_()),
        ('actual_prompt_tokens', pa.int16()), ('policy_prefix_tokens', pa.int8()),
        ('total_actual_policy_tokens', pa.int32()), ('short_policy_prefix', pa.bool_()),
        ('has_complete_response', pa.bool_()), ('source_chunk', pa.int16()),
        ('source_run_id', pa.string()), ('policy_snapshot_id', pa.string()),
        ('source_raw_sha256', pa.string()), ('source_label_sha256', pa.string()),
        ('source_prefix_manifest_sha256', pa.string()), ('prefix_sha256', pa.string()),
        ('input_ids_sha256', pa.string()), ('position_ids_sha256', pa.string()),
        ('policy_mask_sha256', pa.string()), ('model_revision', pa.string()),
    ])
    seen = set()
    counters = {s: collections.Counter() for s in ('train', 'test')}
    files, sources = {}, []
    train_writer = None

    def convert(rows, arrays, groups, split, origin, source_chunk, label_sha, prefix_sha):
        expected = [t for g in groups for t in g['trajectory_ids']]
        assert [r['trajectory_id'] for r in rows] == expected and len(expected) == len(groups) * 5
        records = []
        for i, r in enumerate(rows):
            assert r['trajectory_id'] not in seen
            seen.add(r['trajectory_id'])
            g = groups[i // 5]
            assert moments(g['strict_em']) == (g['correct_count'], g['mean'], g['std'])
            assert r['qid'] == g['qid'] and r['question_hash'] == g['question_hash'] == questions[r['qid']][2]
            assert r['strict_em'] == g['strict_em'][i % 5] and type(r['strict_em']) is int
            start, length = int(r['offset']), int(r['length'])
            ids = arrays['packed_input_ids.npy'][start:start+length].astype('<i8')
            positions = arrays['packed_position_ids.npy'][start:start+length].astype('<i8')
            mask = arrays['packed_policy_mask.npy'][start:start+length].astype(np.bool_)
            assert len(ids) == len(positions) == len(mask) == length
            assert bool((ids >= 0).all()) and bool((positions >= 0).all())
            assert int(ids.max()) <= np.iinfo(np.int32).max and int(positions.max()) <= np.iinfo(np.int32).max
            k, prompt = int(r['policy_prefix_tokens']), int(r['actual_prompt_tokens'])
            assert 1 <= k <= 5 and int(mask.sum()) == k and not bool(mask[:prompt].any())
            assert bool(mask[-1]) and r['short_policy_prefix'] == (r['total_actual_policy_tokens'] < 5)
            for arr, key in ((ids, 'input_ids_sha256'), (positions, 'position_ids_sha256'), (mask, 'policy_mask_sha256')):
                assert hashlib.sha256(arr.tobytes()).hexdigest() == r[key]
            digest = hashlib.sha256(canonical(dict(version=1, length=length, policy_tokens=k)) + ids.tobytes() + positions.tobytes() + mask.tobytes()).hexdigest()
            assert digest == r['prefix_sha256']
            decode = lambda values: tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            records.append(dict(question_id=r['qid'], question_hash=r['question_hash'],
                question_text=questions[r['qid']][0], data_source=questions[r['qid']][1],
                trajectory_id=r['trajectory_id'], response_index=i % 5, split=split, source_origin=origin,
                actual_prompt_text=decode(ids[:prompt].tolist()), context_through_prefix_text=decode(ids.tolist()),
                generated_policy_prefix_text=decode(ids[mask].tolist()), input_ids=ids.tolist(),
                position_ids=positions.tolist(), policy_mask=mask.tolist(), attention_mask=[True]*length,
                final_binary_em=r['strict_em'], group_correct_count=g['correct_count'], group_is_mixed=0 < g['correct_count'] < 5,
                actual_prompt_tokens=prompt, policy_prefix_tokens=k, total_actual_policy_tokens=r['total_actual_policy_tokens'],
                short_policy_prefix=r['short_policy_prefix'], has_complete_response=False, source_chunk=source_chunk,
                source_run_id=r['actor_run_id'], policy_snapshot_id=r['policy_snapshot_id'], source_raw_sha256=r['source_raw_sha256'],
                source_label_sha256=label_sha, source_prefix_manifest_sha256=prefix_sha, prefix_sha256=digest,
                input_ids_sha256=r['input_ids_sha256'], position_ids_sha256=r['position_ids_sha256'],
                policy_mask_sha256=r['policy_mask_sha256'], model_revision=REVISION))
            counters[split]['rows'] += 1
            counters[split]['correct'] += r['strict_em']
            counters[split]['short_prefix'] += int(r['short_policy_prefix'])
            counters[split][origin] += 1
        return pa.Table.from_pylist(records, schema=schema)

    try:
        for ordinal, entry in enumerate(chunks):
            chunk = root / Path(entry['relative_path']).parent
            assert chunk.resolve().is_relative_to(root) and sha(chunk / 'commit.json') == entry['sha256']
            commit = read(chunk / 'commit.json')
            assert commit['index'] == ordinal and commit['dataset_plan_sha256'] == PLAN_SHA
            assert commit['questions'] == 512 and commit['trajectories'] == 2560
            assert commit['actual_actor_optimizer_updates'] == commit['actual_head_optimizer_updates'] == 0
            for name in ('labels.json', 'prefix/manifest.json'):
                p = chunk / name
                assert p.stat().st_size == commit['files'][name]['bytes'] and sha(p) == commit['files'][name]['sha256']
            labels, prefix = read(chunk / 'labels.json'), read(chunk / 'prefix/manifest.json')
            groups = labels['groups']
            assert [(g['qid'],g['question_hash']) for g in groups] == [(r['qid'],r['question_hash']) for r in selection['questions'][ordinal*512:(ordinal+1)*512]]
            verify_frozen(labels['frozen_workers']['before'], labels['frozen_workers']['after'], 4)
            assert {k:labels['contracts'][k] for k in plan['strict_protocol']} == plan['strict_protocol']
            if ordinal >= 10:
                assert commit['raw_rollout']['origin'] == 'new_official_teacher' and not (chunk / 'rollout.pt').exists()
                compacted = read(chunk / 'raw-compacted.json')
                assert compacted['accepted_commit_sha256'] == entry['sha256'] and compacted['removed_after_labels_and_prefix_full_sha']
            else:
                assert commit['raw_rollout']['origin'] == 'reused_qualified_teacher'
            rows, arrays = packed(chunk / 'prefix', prefix['files'])
            assert len(rows) == 2560 and prefix['exact_int64_roundtrip_verified'] and prefix['unknown_zero_policy_count'] == 0
            origin = 'reused_qualified_teacher' if ordinal < 10 else 'new_official_teacher'
            table = convert(rows, arrays, groups, 'train', origin, ordinal,
                            commit['files']['labels.json']['sha256'], commit['files']['prefix/manifest.json']['sha256'])
            if ordinal % 10 == 0:
                if train_writer is not None:
                    train_writer.close()
                path = data / ('train-%05d-of-00020.parquet' % (ordinal // 10))
                train_writer = pq.ParquetWriter(path, schema, compression='zstd', use_dictionary=True)
            train_writer.write_table(table, row_group_size=2560)
            sources.append(dict(chunk=ordinal, commit_sha256=entry['sha256'],
                label_sha256=commit['files']['labels.json']['sha256'], prefix_manifest_sha256=commit['files']['prefix/manifest.json']['sha256'],
                raw_sha256=commit['raw_rollout']['sha256'], raw_origin=origin))
            if ordinal % 10 == 9:
                progress(out, stage='HF_PREFIX5_TRAIN_SHARD_WRITTEN', completed_train_chunks=ordinal+1,
                         completed_train_rows=counters['train']['rows'], correct=counters['train']['correct'])
            del table, rows, arrays
    finally:
        if train_writer is not None:
            train_writer.close()
    reuse = remote_path(plan['reuse']['cache_dir'])
    assert sha(reuse / 'manifest.json') == REUSE_SHA
    reuse_manifest = read(reuse / 'manifest.json')
    rows, arrays = packed(reuse, reuse_manifest['files'])
    test_rows = [r for r in rows if r['split'] == 'test']
    assert len(test_rows) == 2560
    table = convert(test_rows, arrays, test_groups, 'test', 'reused_fixed_test', -1, TEST_LABEL_SHA, REUSE_SHA)
    pq.write_table(table, data / 'test-00000-of-00001.parquet', compression='zstd', row_group_size=2560)
    assert len(seen) == 514560 and counters['train']['rows'] == 512000 and counters['test']['rows'] == 2560
    assert counters['train']['reused_qualified_teacher'] == 25600 and counters['train']['new_official_teacher'] == 486400
    assert counters['test']['correct'] == 559
    # Independent Parquet readback proves every row's exact typed token payload survived serialization.
    readback_ids, readback_questions = set(), {'train':set(), 'test':set()}
    for path in sorted(data.glob('*.parquet')):
        split = 'train' if path.name.startswith('train') else 'test'
        count, correct = 0, 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2560, columns=[
                'trajectory_id','question_id','response_index','final_binary_em','input_ids','position_ids','policy_mask',
                'input_ids_sha256','position_ids_sha256','policy_mask_sha256']):
            groups = collections.defaultdict(list)
            for r in batch.to_pylist():
                assert r['trajectory_id'] not in readback_ids and r['final_binary_em'] in (0,1)
                readback_ids.add(r['trajectory_id']); readback_questions[split].add(r['question_id'])
                groups[r['question_id']].append(r['response_index'])
                for key, dtype, digest in (('input_ids','<i8','input_ids_sha256'),('position_ids','<i8','position_ids_sha256'),('policy_mask',np.bool_,'policy_mask_sha256')):
                    assert hashlib.sha256(np.asarray(r[key],dtype=dtype).tobytes()).hexdigest() == r[digest]
                count += 1; correct += r['final_binary_em']
            assert all(sorted(values) == [0,1,2,3,4] for values in groups.values())
        assert count == (25600 if split == 'train' else 2560)
        files[str(path.relative_to(out))] = dict(bytes=path.stat().st_size,sha256=sha(path),rows=count,correct=correct,split=split)
        progress(out,stage='HF_PREFIX5_PARQUET_READBACK_VERIFIED',files=len(files),rows=count,bytes=path.stat().st_size)
    assert readback_ids == seen and readback_questions['train'] == set(train_qids) and readback_questions['test'] == set(test_qids)
    assert sum(x['correct'] for x in files.values() if x['split']=='train') == counters['train']['correct']
    summary = dict(schema_version=1,stage='HF_PREFIX5_SCALE20_EXPORT_COMPLETE',repository=REPOSITORY,config=CONFIG,
        source_run_id=RUN,source_execution_id=EXECUTION,runtime_source_commit=RUNTIME,exporter_sha256=sha(__file__),
        source_complete_sha256=COMPLETE_SHA,source_plan_sha256=PLAN_SHA,source_selection_sha256=SELECTION_SHA,
        source_chunk_manifest_sha256=done['chunk_manifest_sha256'],source_compact_manifest_sha256=done['compact_manifest_sha256'],
        split_counts={s:dict(c,questions=len(readback_questions[s])) for s,c in counters.items()},
        total_rows=514560,total_questions=102912,train_question_count=102400,fixed_test_question_count=512,
        original_train_reused_questions=5120,new_train_questions=97280,original_train_reused_rows=25600,new_train_rows=486400,
        all_five_responses_same_split=True,disjoint_question_ids_hashes_normalized_text=True,
        prefix_policy_tokens=5,has_complete_responses=False,strict_em_from_original_complete_response=True,
        special_tokens_preserved=True,attention_mask='All stored entries valid; padding already removed by original producer.',
        prefix_rule='Exact original prompt and valid context through fifth policy token inclusive; short1..4 retained; no future tokens, no re-tokenization.',
        old_teacher_identity_sha256=OLD_IDENTITY_SHA,old_teacher_complete_sha256=OLD_COMPLETE_SHA,fixed_test_labels_sha256=TEST_LABEL_SHA,
        reused_original_events_match_existing_teacher_archive=True,new_generation=False,new_scoring=False,gpu_used=False,
        model_revision=REVISION,data_manifest_sha256=COMMON,tokenizer_file_sha256=tokenizer_hashes,
        files=files,sources=sources,checked_unix=time.time())
    once(out / PREFIX / 'provenance.json',summary)
    card = f'''# A800 frozen G5 prefix5 dataset — 2026-10-16

This additional, self-contained Parquet configuration contains **102,400 unique training questions × 5 = 512,000 response events**, plus the fixed, isolated **512-question test × 5 = 2,560 events**. Train includes 25,600 original qualified events (5,120 questions), clearly marked `reused_qualified_teacher`, and 486,400 newly generated events (97,280 questions), marked `new_official_teacher`. Test events are reused, never regenerated. These original 5,120 and test 512 events also exist in the repository's historical teacher archive; their inclusion here provides a directly loadable complete configuration and does not represent new sampling.

**This configuration is prefix5, not the historical prefix20 epoch68 data.** Every row stores the original question, actual prompt, exact valid input IDs/position IDs/policy mask through the fifth generated policy token, decoded prefix text preserving special tokens, and that trajectory's original final binary strict-EM label. The final label was scored from the complete response before compaction. The new full responses were deliberately removed only after prefix/label SHA acceptance; they cannot be reconstructed from these rows. `has_complete_response` is false. Short1–4 prefixes are retained accurately. Padding is absent, so the stored attention mask is all true. No scoring, generation or fitting was performed for this export.

The frozen teacher is official Qwen/Qwen2.5-3B-Instruct revision `{REVISION}`. The common canonical manifest SHA256 remains `{COMMON}`. All five responses to each question stay in the same split. The fixed test is a random heldout subset of the training pool, **not official dev512**. Group-mixed describes one actual G5 draw, not an intrinsic question attribute. Labels retain natural frequencies, without balancing/resampling. Train question IDs, hashes and normalized text are unique and disjoint from test.

```python
from datasets import load_dataset
data = load_dataset("{REPOSITORY}", "{CONFIG}", token=True)
```

The original epoch68 archives, probe/optimizer weights and existing metadata are preserved. No feature bank, model, optimizer, credentials or environment are included in this addition. Per-file content SHA256, exact source commit/label/prefix bindings and typed-Parquet readback proofs are in `provenance.json`. Upstream licenses continue to apply; no new license over upstream QA/search text or Qwen is asserted.
'''
    (out / PREFIX / 'README.md').write_text(card)
    once(out / 'build-complete.json',summary)
    progress(out,stage=summary['stage'],split_counts=summary['split_counts'],files=len(files),bytes=sum(p['bytes'] for p in files.values()))


def upload_memory(args):
    import requests
    import yaml
    from huggingface_hub import HfApi, CommitOperationAdd
    operations = Path(__file__).resolve().parent
    root = operations.parents[2]
    spec = importlib.util.spec_from_file_location('prefix5_existing_route', operations / 'a800_qprobe_prefix_scale20.py')
    route = importlib.util.module_from_spec(spec); spec.loader.exec_module(route)
    out = remote_path(args.output_dir)
    assert str(out).startswith('/data/shaoshitong/Search-G1-hf-scale20-prefix5-')

    def remote(code,binary=False):
        command = route.route().ssh_command(code)
        assert command[-1].startswith('python3 -c ')
        command[-1] = '/data/search-g1/env/bin/python -c ' + command[-1][len('python3 -c '):]
        result = subprocess.run(command,capture_output=True,timeout=180)
        if result.returncode:
            raise RuntimeError('Existing remote route failed')
        return result.stdout if binary else json.loads(result.stdout)

    manifest = remote('import json,pathlib; print(pathlib.Path('+repr(str(out/'build-complete.json'))+').read_text())')
    assert manifest['config']==CONFIG and manifest['repository']==REPOSITORY and manifest['total_rows']==514560
    expected = {k:dict(bytes=v['bytes'],sha256=v['sha256']) for k,v in manifest['files'].items()}
    for name in (PREFIX+'/provenance.json',PREFIX+'/README.md'):
        p = out/name
        proof = remote('import hashlib,pathlib,json; p=pathlib.Path('+repr(str(p))+'); print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))')
        expected[name]=proof
    assert len(manifest['files'])==21 and all(v['bytes']<=64*1024**2 for v in expected.values())
    token=getpass.getpass('HF credential (hidden): ')
    assert isinstance(token,str) and token.startswith('hf_')
    api=HfApi(token=token)
    headers={'Authorization':'Bearer '+token}
    initial=api.repo_info(REPOSITORY,repo_type='dataset',files_metadata=True)
    assert initial.private is True
    baseline={i.rfilename:i for i in initial.siblings}
    required=('EXPORT_MANIFEST.json','original-teacher-and-fixed-test.tar.gz.part000','probe-epoch-0068.tar.gz.part000')
    assert all(n in baseline for n in required)
    response=requests.get('https://huggingface.co/datasets/'+REPOSITORY+'/resolve/'+initial.sha+'/manifests/original-teacher-and-fixed-test.tar.gz.manifest.json',headers=headers,timeout=(15,60))
    response.raise_for_status()
    archive_manifest=response.json()
    archive_members={v['path']:v for v in archive_manifest['files']}
    for name,value in [('teacher5120-g5/identity.json',OLD_IDENTITY_SHA),('teacher5120-g5/complete.json',OLD_COMPLETE_SHA),('fixed-test512-g5/labels.json',TEST_LABEL_SHA)]:
        assert archive_members[name]['sha256']==value
    original_archive_manifest_sha256=hashlib.sha256(response.content).hexdigest()
    def fingerprint(item):
        return dict(bytes=item.size,blob_id=item.blob_id,lfs_sha256=item.lfs.sha256 if item.lfs else None)
    old_fingerprints={n:fingerprint(v) for n,v in baseline.items() if n!='README.md' and not n.startswith(PREFIX+'/')}
    progress_events=[]
    def emit(**value):
        value['checked_unix']=time.time();progress_events.append(value);print(json.dumps(value),flush=True)
    commits=[];parent=initial.sha
    names=sorted(expected)
    for start in range(0,len(names),4):
        info=api.repo_info(REPOSITORY,repo_type='dataset',files_metadata=True)
        actual_files={i.rfilename:i for i in info.siblings}
        assert info.private==initial.private and all(n in actual_files and fingerprint(actual_files[n])==p for n,p in old_fingerprints.items())
        parent=info.sha;memory={}
        for name in names[start:start+4]:
            proof=expected[name]
            if name in actual_files:
                item=actual_files[name]
                assert item.size==proof['bytes'], 'Existing addition differs; never overwrite it.'
                if item.lfs is not None:
                    assert item.lfs.sha256==proof['sha256'], 'Existing addition differs; never overwrite it.'
                else:
                    response=requests.get('https://huggingface.co/datasets/'+REPOSITORY+'/resolve/'+info.sha+'/'+name,headers=headers,timeout=(15,60))
                    response.raise_for_status()
                    assert hashlib.sha256(response.content).hexdigest()==proof['sha256'], 'Existing addition differs; never overwrite it.'
                continue
            blob=remote('import pathlib,sys; p=pathlib.Path('+repr(str(out/name))+'); assert p.stat().st_size=='+str(proof['bytes'])+'; sys.stdout.buffer.write(p.read_bytes())',binary=True)
            assert len(blob)==proof['bytes'] and hashlib.sha256(blob).hexdigest()==proof['sha256']
            memory[name]=blob
            emit(stage='HF_PREFIX5_MEMORY_RELAY_VERIFIED',file=name,bytes=len(blob))
        assert sum(map(len,memory.values()))<=256*1024**2
        if memory:
            commit=api.create_commit(REPOSITORY,repo_type='dataset',parent_commit=parent,
                operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=b) for n,b in memory.items()],
                commit_message='Append sealed A800 prefix5 dataset shards (2026-10-16)')
            commits.append(commit.oid);parent=commit.oid
            emit(stage='HF_PREFIX5_SHARD_BATCH_COMMITTED',commit_sha=parent,files=len(memory))
        memory.clear()
    info=api.repo_info(REPOSITORY,repo_type='dataset',files_metadata=True)
    current={i.rfilename:i for i in info.siblings}
    assert all(n in current for n in expected)
    response=requests.get('https://huggingface.co/datasets/'+REPOSITORY+'/resolve/'+info.sha+'/README.md',headers=headers,timeout=(15,60))
    response.raise_for_status();old_readme=response.text;body=old_readme;metadata={}
    if old_readme.startswith('---\n'):
        end=old_readme.find('\n---',4);assert end!=-1
        metadata=yaml.safe_load(old_readme[4:end]) or {};body=old_readme[end+4:].lstrip('\n')
    configs=metadata.get('configs',[]);assert isinstance(configs,list)
    configured=[c for c in configs if c.get('config_name')==CONFIG]
    config=dict(config_name=CONFIG,data_files=[dict(split='train',path=PREFIX+'/data/train-*.parquet'),dict(split='test',path=PREFIX+'/data/test-*.parquet')])
    if configured:
        assert configured==[config]
        final_commit=info.sha
    else:
        configs.append(config);metadata['configs']=configs
        body+='\n\n## Additional A800 prefix5 data (2026-10-16)\n\nConfiguration `'+CONFIG+'` adds 102,400 unique training questions / 512,000 response events and a fixed 512-question / 2,560-event test split. This is prefix5 data with original final strict EM, distinct from the historical prefix20 epoch68 archives. Original archives and probe state remain unchanged. [Addition details]('+PREFIX+'/README.md).\n'
        new_readme=('---\n'+yaml.safe_dump(metadata,sort_keys=False,allow_unicode=True).strip()+'\n---\n\n'+body).encode()
        commit=api.create_commit(REPOSITORY,repo_type='dataset',parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=new_readme)],
            commit_message='Register self-contained A800 prefix5 dataset configuration (2026-10-16)')
        final_commit=commit.oid;commits.append(final_commit)
    emit(stage='HF_PREFIX5_CONFIG_COMMITTED',commit_sha=final_commit)
    final=api.repo_info(REPOSITORY,repo_type='dataset',revision=final_commit,files_metadata=True)
    actual={i.rfilename:i for i in final.siblings}
    assert final.private==initial.private and all(n in actual and fingerprint(actual[n])==p for n,p in old_fingerprints.items())
    verified={}
    for name,proof in expected.items():
        item=actual[name];assert item.size==proof['bytes']
        if item.lfs is not None:assert item.lfs.sha256==proof['sha256']
        response=requests.get('https://huggingface.co/datasets/'+REPOSITORY+'/resolve/'+final_commit+'/'+name,headers=headers,timeout=(15,120),stream=True)
        response.raise_for_status();h=hashlib.sha256();size=0
        for blob in response.iter_content(1024**2):h.update(blob);size+=len(blob)
        assert size==proof['bytes'] and h.hexdigest()==proof['sha256']
        verified[name]=dict(proof,remote_readback_sha256=h.hexdigest())
        emit(stage='HF_PREFIX5_REMOTE_FILE_VERIFIED',file=name,verified_files=len(verified),bytes=size)
    response=requests.get('https://huggingface.co/datasets/'+REPOSITORY+'/resolve/'+final_commit+'/README.md',headers=headers,timeout=(15,60));response.raise_for_status()
    assert body.split('\n\n## Additional A800 prefix5 data (2026-10-16)')[0] in response.text
    boundary=response.text.find('\n---',4)
    assert response.text.startswith('---\n') and boundary!=-1
    assert config in yaml.safe_load(response.text[4:boundary])['configs']
    receipt=dict(stage='HF_PREFIX5_SCALE20_DATASET_VERIFIED',repository=REPOSITORY,config=CONFIG,commit_sha=final_commit,
        previous_commit=initial.sha,commits=commits,private=final.private,files=verified,existing_archive_fingerprints=old_fingerprints,
        root_readme_sha256=hashlib.sha256(response.content).hexdigest(),original_archive_manifest_sha256=original_archive_manifest_sha256,
        existing_files_preserved=True,split_counts=manifest['split_counts'],total_rows=514560,
        source_complete_sha256=COMPLETE_SHA,exporter_sha256=sha(__file__),local_relay='bounded_memory_only',
        local_dataset_files_written=False,credentials_persisted=False,feature_bank_or_weights_uploaded=False,
        new_generation=False,new_scoring=False,checked_unix=time.time())
    once(root/'preparation/qprobe-hf-scale20-prefix5-verified-20261016.json',receipt)
    remote('import json,pathlib,os; p=pathlib.Path('+repr(str(out/'hf-verified.json'))+'); v='+repr(receipt)+'; f=p.open("x"); json.dump(v,f,sort_keys=True,allow_nan=False); f.flush(); os.fsync(f.fileno()); f.close(); print(json.dumps(dict(saved=True)))')
    token=None
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=('build','upload-memory'),required=True)
    parser.add_argument('--runtime-repo',required=True)
    parser.add_argument('--dataset-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    try:
        if args.action=='build':build(args)
        else:upload_memory(args)
    except BaseException as error:
        # Error class only: library exception text can contain signed URLs or source text.
        print(json.dumps(dict(stage='HF_PREFIX5_ACTION_FAILED',error_class=type(error).__name__,checked_unix=time.time())),flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
