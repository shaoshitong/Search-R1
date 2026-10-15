"""CPU-only export of the immutable 5120 + new512 frozen-teacher G5 events.

Credentials are read only from stdin by the upload action, never logged or
saved. No login, rollout, scoring rerun, GPU access, or training is performed.
"""
import argparse
import collections
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

RUNTIME = 'fcb77204f48d5cb8ec21c68df7d4a2c937b3557a'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
MANIFEST = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
REPOSITORY = 'Sutoonq/Trajectory_in_Search'
CONFIG = 'official_qwen25_frozen_g5_20261016'
PROVIDER_SHA = '4a8adfb2adeef5de6bd80a4abe28172a6a989b56f76ab0b329b04af50567d95f'
TEST_LABEL_SHA = 'd7d056477666558a2dd0fbaeb0f7ee8aff679e80c5730fbb0a3a8c684395740b'
TEST_SELECTION_SHA = '6d2cc37534ccd10df2be5b4cf0deaaf0e8cfa248174a636b5028c1dbc9258133'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def once(path, value):
    with Path(path).open('xb') as stream:
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


def setup(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    repo = Path(args.runtime_repo)
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                   text=True).strip() == RUNTIME
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
    sys.path.insert(0, str(repo))
    out = Path(args.output_dir)
    assert out.is_absolute() and out.resolve() == out and str(out).startswith('/data/shaoshitong/')
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    return out


def build(args, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from search_g1.qprobe_features import original_question_and_id
    from search_g1.qprobe_warmup import verify_frozen

    torch.set_num_threads(4)
    provider_path = Path(args.provider_path)
    assert sha(provider_path) == PROVIDER_SHA
    spec = importlib.util.spec_from_file_location('immutable_prefix_source_provider', provider_path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    provider_args = argparse.Namespace(teacher_cache=provider.OLD_TEACHER,
                                      test_base=provider.TEST_BASE)
    assert sha(Path(provider.TEST_BASE) / 'teacher-cache/labels.json') == TEST_LABEL_SHA
    assert sha(Path(provider.TEST_BASE) / 'selection.json') == TEST_SELECTION_SHA
    descriptors, isolation, run = provider.source_descriptors(provider_args)
    assert read(Path(provider.OLD_TEACHER) / 'identity.json')['data_manifest_sha256'] == MANIFEST
    needed = {g['qid'] for d in descriptors for g in d['groups']}
    questions = {}
    for row in pq.read_table(Path(provider.TEST_BASE) / 'data/train.parquet',
                             columns=['prompt', 'data_source', 'extra_info']).to_pylist():
        text, qid = original_question_and_id(row)
        if qid in needed:
            assert qid not in questions
            questions[qid] = (text, row['data_source'])
    assert set(questions) == needed and len(questions) == 5632
    model = Path(provider.MODEL)
    tokenizer_names = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                       'vocab.json', 'merges.txt', 'added_tokens.json')
    tokenizer_proof = {}
    for name in tokenizer_names:
        path = model / name
        if path.exists():
            value = sha(path)
            if name in run['initial_model_sha256']:
                assert value == run['initial_model_sha256'][name]
            tokenizer_proof[name] = value
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                               trust_remote_code=False)
    data_dir = out / 'data' / CONFIG
    data_dir.mkdir(parents=True, exist_ok=False)
    writers = {}
    counters = {split: collections.Counter() for split in ('train', 'test')}
    seen_trajectories = set()
    seen_questions = {'train': set(), 'test': set()}
    sources = []
    schema = pa.schema([
        ('question_id', pa.string()), ('question_hash', pa.string()),
        ('response_index', pa.int8()), ('trajectory_id', pa.string()),
        ('question_text', pa.string()), ('data_source', pa.string()),
        ('actual_prompt_text', pa.string()), ('response_trajectory_text', pa.string()),
        ('policy_only_text', pa.string()), ('full_context_text', pa.string()),
        ('final_binary_em', pa.int8()), ('group_correct_count', pa.int8()),
        ('group_is_mixed', pa.bool_()), ('source_split', pa.string()),
        ('model_revision', pa.string()), ('policy_snapshot_id', pa.string()),
        ('source_run_id', pa.string()), ('source_raw_sha256', pa.string()),
        ('source_label_sha256', pa.string()), ('source_raw_row', pa.int32()),
        ('actual_prompt_token_count', pa.int32()), ('response_token_count', pa.int32()),
        ('policy_token_count', pa.int32()),
    ])
    try:
        for descriptor in descriptors:
            raw = provider.load_verified_raw(descriptor)
            output = raw['output']
            batch = output.batch
            identity = raw['identity']
            groups = descriptor['groups']
            verify_frozen(raw['frozen_workers']['before'], raw['frozen_workers']['after'], 4)
            tids = identity['trajectory_ids']
            assert tids == output.non_tensor_batch['behavior_trajectory_id'].tolist()
            assert tids == [tid for group in groups for tid in group['trajectory_ids']]
            assert raw['question_bindings'] == [dict(qid=g['qid'], question_hash=g['question_hash']) for g in groups]
            assert len(tids) == len(set(tids)) == 2560
            assert output.meta_info['policy_snapshot_id'] == identity['policy_snapshot_id']
            p = batch['prompts'].shape[1]
            assert batch['input_ids'].shape[1] == p + batch['sampling_mask'].shape[1]
            assert torch.equal(batch['input_ids'][:, :p], batch['prompts'])
            assert torch.equal(batch['input_ids'][:, p:], batch['responses'])
            assert torch.equal(batch['info_mask'][:, p:].bool(), batch['sampling_mask'])
            assert bool(((batch['attention_mask'] == 0) | (batch['attention_mask'] == 1)).all())
            valid = batch['attention_mask'].bool()
            assert not bool((batch['sampling_mask'] & ~valid[:, p:]).any())
            rows = []
            split = descriptor['split']
            for index, tid in enumerate(tids):
                assert tid not in seen_trajectories
                seen_trajectories.add(tid)
                group = groups[index // 5]
                question_text, data_source = questions[group['qid']]
                em = group['strict_em'][index % 5]
                assert type(em) is int and em in (0, 1)
                prompt_ids = batch['prompts'][index][valid[index, :p]].tolist()
                response_ids = batch['responses'][index][valid[index, p:]].tolist()
                policy_ids = batch['responses'][index][batch['sampling_mask'][index]].tolist()
                full_ids = batch['input_ids'][index][valid[index]].tolist()
                assert full_ids == prompt_ids + response_ids and policy_ids
                decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)
                rows.append(dict(question_id=group['qid'], question_hash=group['question_hash'],
                    response_index=index % 5, trajectory_id=tid, question_text=question_text,
                    data_source=data_source, actual_prompt_text=decode(prompt_ids),
                    response_trajectory_text=decode(response_ids), policy_only_text=decode(policy_ids),
                    full_context_text=decode(full_ids), final_binary_em=em,
                    group_correct_count=group['correct_count'], group_is_mixed=0 < group['correct_count'] < 5,
                    source_split=split, model_revision=REVISION,
                    policy_snapshot_id=identity['policy_snapshot_id'], source_run_id=identity['run_id'],
                    source_raw_sha256=descriptor['raw_sha256'], source_label_sha256=descriptor['label_sha256'],
                    source_raw_row=index, actual_prompt_token_count=len(prompt_ids),
                    response_token_count=len(response_ids), policy_token_count=len(policy_ids)))
                counters[split]['rows'] += 1
                counters[split]['correct'] += em
                seen_questions[split].add(group['qid'])
            table = pa.Table.from_pylist(rows, schema=schema)
            if split not in writers:
                writers[split] = pq.ParquetWriter(data_dir / (split + '-00000-of-00001.parquet'),
                                                  schema, compression='zstd', use_dictionary=True)
            writers[split].write_table(table, row_group_size=2560)
            sources.append({key: value for key, value in descriptor.items()
                            if key not in ('groups', 'raw_path', 'label_path')})
            progress(out, stage='HF_DATASET_EXPORT_CHUNK', verified_sources=len(sources),
                     trajectories=len(seen_trajectories), split_counts={s: dict(c) for s, c in counters.items()})
            del table, rows, raw, output, batch
    finally:
        for writer in writers.values():
            writer.close()
    assert len(seen_trajectories) == 28160
    assert seen_questions['train'].isdisjoint(seen_questions['test'])
    assert {s: len(v) for s, v in seen_questions.items()} == dict(train=5120, test=512)
    assert dict(counters['train']) == dict(rows=25600, correct=5627)
    assert dict(counters['test']) == dict(rows=2560, correct=559)
    files = {}
    for split in ('train', 'test'):
        path = data_dir / (split + '-00000-of-00001.parquet')
        table = pq.read_table(path, columns=['question_id', 'trajectory_id', 'final_binary_em', 'response_index'])
        groups = collections.defaultdict(list)
        for row in table.to_pylist():
            assert row['final_binary_em'] in (0, 1)
            groups[row['question_id']].append(row['response_index'])
        assert all(sorted(indices) == [0, 1, 2, 3, 4] for indices in groups.values())
        assert len(set(table.column('trajectory_id').to_pylist())) == counters[split]['rows']
        files[str(path.relative_to(out))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    manifest = dict(schema_version=1, config_name=CONFIG, hf_repository=REPOSITORY,
        split_counts={s: dict(c, questions=len(seen_questions[s])) for s, c in counters.items()},
        total_unique_questions=5632, total_unique_trajectories=28160, g=5,
        existing_strict_em_reused=True, scorer_rerun=False, new_generation=False,
        gpu_used=False, training_started=False, special_tokens_preserved=True,
        text_rule='Decode original input_ids after attention_mask removes padding; policy_only_text additionally uses original sampling_mask.',
        test_definition='New random 512 training-pool questions selected on 2026-10-16; NOT the official dev512.',
        mixed_definition='0 < sum(strict_em for this one actual five-response sampling event) < 5; not an immutable question attribute.',
        data_manifest_sha256=MANIFEST, model_revision=REVISION,
        tokenizer_files_sha256=tokenizer_proof, source_isolation=isolation, sources=sources,
        files=files, runtime_source_commit=RUNTIME, exporter_sha256=sha(__file__),
        source_provider_sha256=PROVIDER_SHA, checked_unix=time.time())
    once(data_dir / 'provenance.json', manifest)
    card = f'''# Frozen Qwen2.5-3B G5 trajectories — 2026-10-16

This configuration contains 28,160 actual frozen-teacher trajectories for 5,632 unique questions. `train` is the original 5,120-question teacher warmup (25,600 responses; 5,627 correct); `test` is a newly sampled, isolated 512-question diagnostic set (2,560 responses; 559 correct). This test split is **not** the official NQ/Hotpot dev512.

Each row contains one of five actual responses, original question text, decoded actual prompt, complete response trajectory (including search/observation/continuation and special tokens), policy-only text, full context, and the existing strict binary EM label. Padding is removed using the original attention mask; only `policy_only_text` applies the sampling mask. No text was regenerated and no EM was rescored.

The teacher is Qwen/Qwen2.5-3B-Instruct at revision `{REVISION}`. The unchanged common manifest SHA256 is `{MANIFEST}`. Source, tokenizer and complete raw-rollout SHA256 proofs are in `data/{CONFIG}/provenance.json`.

Group correctness and mixed status describe **this one actual G5 sampling event**, not a permanent property of a question. All five responses to a question remain in the same split. The original and new question sets have disjoint IDs, hashes, and normalized question content. Labels have their original natural class frequencies; they are not balanced by resampling in this uploaded dataset.

These are existing trajectories only. The separately running 102,400-question dataset build and all model weights, credentials and environments are excluded. Upstream question/source licenses remain applicable; this dataset does not assert a new license over upstream question or retrieved text.
'''
    (out / 'CONFIG_README.md').write_text(card)
    once(out / 'build-complete.json', manifest)
    progress(out, stage='HF_DATASET_BUILD_COMPLETE', files=files,
             split_counts=manifest['split_counts'], manifest_sha256=sha(out / 'build-complete.json'))


def upload(args, out):
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    import requests
    import yaml
    import pyarrow.parquet as pq

    token = json.load(sys.stdin)['token']
    assert isinstance(token, str) and token.startswith('hf_')
    api = HfApi(token=token)
    manifest = read(out / 'build-complete.json')
    assert manifest['config_name'] == CONFIG and manifest['total_unique_trajectories'] == 28160
    try:
        info = api.repo_info(REPOSITORY, repo_type='dataset', files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(REPOSITORY, repo_type='dataset', private=True, exist_ok=False)
        info = api.repo_info(REPOSITORY, repo_type='dataset', files_metadata=True)
    previous = info.sha
    siblings = {item.rfilename: item for item in info.siblings}
    paths = set(manifest['files']) | {'data/' + CONFIG + '/provenance.json', 'configs/' + CONFIG + '/README.md'}
    assert not paths.intersection(siblings), 'Dated configuration already exists; preserve it and inspect before retrying.'
    headers = {'Authorization': 'Bearer ' + token}
    old_readme = ''
    if 'README.md' in siblings:
        response = requests.get('https://huggingface.co/datasets/' + REPOSITORY + '/resolve/' + previous + '/README.md',
                                headers=headers, timeout=(15, 60))
        response.raise_for_status()
        old_readme = response.text
    metadata = {}
    body = old_readme
    if old_readme.startswith('---\n'):
        boundary = old_readme.find('\n---', 4)
        assert boundary != -1
        metadata = yaml.safe_load(old_readme[4:boundary]) or {}
        body = old_readme[boundary + 4:].lstrip('\n')
    assert isinstance(metadata, dict)
    configs = metadata.get('configs', [])
    assert isinstance(configs, list) and all(item.get('config_name') != CONFIG for item in configs)
    configs.append(dict(config_name=CONFIG, data_files=[
        dict(split='train', path='data/' + CONFIG + '/train-00000-of-00001.parquet'),
        dict(split='test', path='data/' + CONFIG + '/test-00000-of-00001.parquet')]))
    metadata['configs'] = configs
    if not body:
        body = '# Trajectory in Search\n'
    body += '\n\n## Frozen teacher G5 export (2026-10-16)\n\nConfiguration `' + CONFIG + '` contains the original 5120-question warmup and new random 512-question diagnostic trajectories. [Configuration details](configs/' + CONFIG + '/README.md). Existing configurations and files are preserved.\n'
    new_readme = '---\n' + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip() + '\n---\n\n' + body
    operations = []
    expected = {}
    for relative in sorted(manifest['files']):
        path = out / relative
        assert sha(path) == manifest['files'][relative]['sha256']
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(path)))
        expected[relative] = manifest['files'][relative]
    for relative, local in [('data/' + CONFIG + '/provenance.json', out / 'data' / CONFIG / 'provenance.json'),
                            ('configs/' + CONFIG + '/README.md', out / 'CONFIG_README.md')]:
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(local)))
        expected[relative] = dict(bytes=local.stat().st_size, sha256=sha(local))
    readme_bytes = new_readme.encode()
    operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=readme_bytes))
    expected['README.md'] = dict(bytes=len(readme_bytes), sha256=hashlib.sha256(readme_bytes).hexdigest())
    progress(out, stage='HF_UPLOAD_STARTING', previous_commit=previous, private=info.private,
             files=len(operations), bytes=sum(item['bytes'] for item in expected.values()))
    commit = api.create_commit(REPOSITORY, repo_type='dataset', operations=operations,
                              commit_message='Add frozen Qwen G5 5120 + new512 trajectories (2026-10-16)',
                              parent_commit=previous)
    progress(out, stage='HF_UPLOAD_COMMITTED', commit_sha=commit.oid)
    actual = api.repo_info(REPOSITORY, repo_type='dataset', revision=commit.oid, files_metadata=True)
    actual_files = {item.rfilename: item for item in actual.siblings}
    assert actual.private == info.private
    assert set(siblings).issubset(actual_files), 'Existing files disappeared'
    verified = {}
    verify_dir = out / 'verification'
    verify_dir.mkdir(exist_ok=False)
    for relative, expected_file in expected.items():
        item = actual_files[relative]
        assert item.size == expected_file['bytes']
        lfs = item.lfs
        if lfs is not None:
            assert lfs.sha256 == expected_file['sha256']
        response = requests.get('https://huggingface.co/datasets/' + REPOSITORY + '/resolve/' + commit.oid + '/' + relative,
                                headers=headers, timeout=(15, 120), stream=True)
        response.raise_for_status()
        path = verify_dir / Path(relative).name
        h = hashlib.sha256()
        with path.open('xb') as stream:
            for chunk in response.iter_content(1024**2):
                stream.write(chunk); h.update(chunk)
        assert path.stat().st_size == expected_file['bytes'] and h.hexdigest() == expected_file['sha256']
        checked = dict(expected_file, remote_download_sha256=h.hexdigest())
        if path.suffix == '.parquet':
            table = pq.read_table(path, columns=['trajectory_id', 'question_id', 'final_binary_em', 'response_index'])
            split = 'train' if path.name.startswith('train') else 'test'
            em = table.column('final_binary_em').to_pylist()
            qids = table.column('question_id').to_pylist()
            ids = table.column('trajectory_id').to_pylist()
            groups = collections.defaultdict(list)
            for row in table.to_pylist(): groups[row['question_id']].append(row['response_index'])
            assert len(em) == manifest['split_counts'][split]['rows'] and set(em) <= {0, 1}
            assert sum(em) == manifest['split_counts'][split]['correct']
            assert len(set(qids)) == manifest['split_counts'][split]['questions'] and len(set(ids)) == len(ids)
            assert all(sorted(v) == [0, 1, 2, 3, 4] for v in groups.values())
            checked.update(rows=len(em), correct=sum(em), unique_questions=len(set(qids)))
        verified[relative] = checked
        path.unlink()
        progress(out, stage='HF_REMOTE_FILE_VERIFIED', commit_sha=commit.oid,
                 verified_files=len(verified), file=relative, bytes=checked['bytes'])
    verify_dir.rmdir()
    receipt = dict(stage='HF_TRAJECTORY_DATASET_VERIFIED', repository=REPOSITORY, config=CONFIG,
                   commit_sha=commit.oid, previous_commit=previous, private=actual.private,
                   files=verified, split_counts=manifest['split_counts'], existing_files_preserved=True,
                   credentials_persisted=False, local_mac_staging_used=False, remote_verification_downloads_removed=True,
                   actual_generation_updates=0, actor_optimizer_updates=0, checked_unix=time.time())
    once(out / 'hf-verified.json', receipt)
    print(json.dumps(receipt), flush=True)


def upload_memory(args):
    """Use the Mac as an in-memory relay after actual A800 HF network failure."""
    import requests
    import yaml
    from huggingface_hub import HfApi, CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError

    operations_root = Path(__file__).resolve().parent
    root = operations_root.parents[2]
    route_spec = importlib.util.spec_from_file_location('hf_existing_physical_route',
        operations_root / 'a800_qprobe_prefix_scale20.py')
    route_module = importlib.util.module_from_spec(route_spec)
    route_spec.loader.exec_module(route_module)
    remote_out = Path(args.output_dir)
    assert str(remote_out).startswith('/data/shaoshitong/Search-G1-hf-trajectories-')

    def remote(code, binary=False):
        command = route_module.route().ssh_command(code)
        assert command[-1].startswith('python3 -c ')
        command[-1] = '/data/search-g1/env/bin/python -c ' + command[-1][len('python3 -c '):]
        result = subprocess.run(command, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError('Existing remote route failed')
        return result.stdout if binary else json.loads(result.stdout)

    metadata = remote('import json,pathlib; p=pathlib.Path(' + repr(str(remote_out)) + '); '
        'print(json.dumps(dict(manifest=json.loads((p/"build-complete.json").read_text()), '
        'card=(p/"CONFIG_README.md").read_text())))')
    manifest = metadata['manifest']
    assert manifest['total_unique_trajectories'] == 28160
    assert manifest['split_counts'] == dict(train=dict(rows=25600, correct=5627, questions=5120),
                                          test=dict(rows=2560, correct=559, questions=512))
    files = {}
    expected = dict(manifest['files'])
    assert sum(item['bytes'] for item in expected.values()) <= 256 * 1024**2
    for relative, value in expected.items():
        assert '..' not in Path(relative).parts and relative.startswith('data/' + CONFIG + '/')
        data = remote('import pathlib,sys; p=pathlib.Path(' + repr(str(remote_out / relative)) + '); '
            'assert p.stat().st_size == ' + str(value['bytes']) + '; sys.stdout.buffer.write(p.read_bytes())', binary=True)
        assert len(data) == value['bytes'] and hashlib.sha256(data).hexdigest() == value['sha256']
        files[relative] = data
        print(json.dumps(dict(stage='HF_MAC_MEMORY_RELAY_VERIFIED', file=relative,
                              bytes=len(data), checked_unix=time.time())), flush=True)
    provenance_path = 'data/' + CONFIG + '/provenance.json'
    provenance = remote('import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(' +
        repr(str(remote_out / provenance_path)) + ').read_bytes())', binary=True)
    assert json.loads(provenance) == manifest
    files[provenance_path] = provenance
    files['configs/' + CONFIG + '/README.md'] = metadata['card'].encode()
    token = json.load(sys.stdin)['token']
    assert isinstance(token, str) and token.startswith('hf_')
    api = HfApi(token=token)
    try:
        info = api.repo_info(REPOSITORY, repo_type='dataset', files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(REPOSITORY, repo_type='dataset', private=True, exist_ok=False)
        info = api.repo_info(REPOSITORY, repo_type='dataset', files_metadata=True)
    previous = info.sha
    siblings = {item.rfilename: item for item in info.siblings}
    assert not set(files).intersection(siblings), 'Dated configuration exists; inspect before retrying.'
    headers = {'Authorization': 'Bearer ' + token}
    old_readme = ''
    if 'README.md' in siblings:
        response = requests.get('https://huggingface.co/datasets/' + REPOSITORY + '/resolve/' + previous + '/README.md',
                                headers=headers, timeout=(15, 60))
        response.raise_for_status()
        old_readme = response.text
    card_metadata = {}
    body = old_readme
    if old_readme.startswith('---\n'):
        boundary = old_readme.find('\n---', 4)
        assert boundary != -1
        card_metadata = yaml.safe_load(old_readme[4:boundary]) or {}
        body = old_readme[boundary + 4:].lstrip('\n')
    configs = card_metadata.get('configs', [])
    assert isinstance(configs, list) and all(item.get('config_name') != CONFIG for item in configs)
    configs.append(dict(config_name=CONFIG, data_files=[
        dict(split='train', path='data/' + CONFIG + '/train-00000-of-00001.parquet'),
        dict(split='test', path='data/' + CONFIG + '/test-00000-of-00001.parquet')]))
    card_metadata['configs'] = configs
    if not body: body = '# Trajectory in Search\n'
    body += '\n\n## Frozen teacher G5 export (2026-10-16)\n\nConfiguration `' + CONFIG + '` contains the original 5120-question warmup and new random 512-question diagnostic trajectories. [Configuration details](configs/' + CONFIG + '/README.md). Existing configurations and files are preserved.\n'
    files['README.md'] = ('---\n' + yaml.safe_dump(card_metadata, sort_keys=False,
        allow_unicode=True).strip() + '\n---\n\n' + body).encode()
    expected = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                for name, data in files.items()}
    print(json.dumps(dict(stage='HF_MEMORY_UPLOAD_STARTING', previous_commit=previous,
        private=info.private, files=len(files), bytes=sum(v['bytes'] for v in expected.values()),
        checked_unix=time.time())), flush=True)
    result = api.create_commit(REPOSITORY, repo_type='dataset', parent_commit=previous,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in files.items()],
        commit_message='Add frozen Qwen G5 5120 + new512 trajectories (2026-10-16)')
    print(json.dumps(dict(stage='HF_MEMORY_UPLOAD_COMMITTED', commit_sha=result.oid,
                          checked_unix=time.time())), flush=True)
    actual = api.repo_info(REPOSITORY, repo_type='dataset', revision=result.oid, files_metadata=True)
    actual_files = {item.rfilename: item for item in actual.siblings}
    assert actual.private == info.private and set(siblings).issubset(actual_files)
    verified = {}
    for relative, value in expected.items():
        item = actual_files[relative]
        assert item.size == value['bytes']
        if item.lfs is not None:
            assert item.lfs.sha256 == value['sha256']
        response = requests.get('https://huggingface.co/datasets/' + REPOSITORY + '/resolve/' + result.oid + '/' + relative,
                                headers=headers, timeout=(15, 120), stream=True)
        response.raise_for_status()
        h = hashlib.sha256(); size = 0
        for chunk in response.iter_content(1024**2):
            h.update(chunk); size += len(chunk)
        assert size == value['bytes'] and h.hexdigest() == value['sha256']
        verified[relative] = dict(value, remote_download_sha256=h.hexdigest())
        print(json.dumps(dict(stage='HF_MEMORY_REMOTE_FILE_VERIFIED', file=relative,
            verified_files=len(verified), commit_sha=result.oid, checked_unix=time.time())), flush=True)
    receipt = dict(stage='HF_TRAJECTORY_DATASET_VERIFIED', repository=REPOSITORY,
        config=CONFIG, commit_sha=result.oid, previous_commit=previous, private=actual.private,
        files=verified, split_counts=manifest['split_counts'], existing_files_preserved=True,
        credentials_persisted=False, local_mac_staging_used=False, local_relay='memory_only',
        raw_tensor_transferred_to_mac=False, actor_optimizer_updates=0, checked_unix=time.time())
    local_receipt = root / 'preparation/qprobe-hf-trajectory-dataset-verified-20261016.json'
    once(local_receipt, receipt)
    remote('import pathlib,json,os; p=pathlib.Path(' + repr(str(remote_out / 'hf-verified.json')) + '); '
           'v=' + repr(receipt) + '; '
           'f=p.open("x"); json.dump(v,f,sort_keys=True,allow_nan=False); f.flush(); os.fsync(f.fileno()); f.close(); '
           'print(json.dumps(dict(saved=True)))')
    files.clear()
    token = None
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('build', 'upload', 'upload-memory'), required=True)
    parser.add_argument('--runtime-repo', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--provider-path')
    args = parser.parse_args()
    if args.action == 'upload-memory':
        try:
            upload_memory(args)
        except BaseException as error:
            print(json.dumps(dict(stage='HF_MEMORY_ACTION_FAILED', error_class=type(error).__name__,
                                  checked_unix=time.time())), flush=True)
            raise SystemExit(1)
        return
    out = setup(args)
    try:
        if args.action == 'build':
            assert not (out / 'build-complete.json').exists()
            build(args, out)
        else:
            assert not (out / 'hf-verified.json').exists()
            upload(args, out)
    except BaseException as error:
        # Never print request URLs, headers, exception messages or input secrets.
        receipt = dict(stage='HF_DATASET_ACTION_FAILED', action=args.action,
                       error_class=type(error).__name__, checked_unix=time.time())
        progress(out, **receipt)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
