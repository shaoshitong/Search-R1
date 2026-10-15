"""Remote canonical question pool: no repeats until exhaustion, then reshuffle.

Epoch one is the already cached 5,120-question teacher. Those questions consume
its first cycle. Every later epoch gets fresh trajectories, even after the pool
cycles; a boundary epoch never contains the same question twice. The fixed
512-question diagnostic is excluded by both stable ID and normalized text.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import unicodedata


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def immutable(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b'\n'
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing immutable pool artifact differs: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def question_hash(question):
    return hashlib.sha256(question.encode()).hexdigest()


def normalized_hash(question):
    # Same identity as search_g1.mixed_selection.question_identity.
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', question).split()).encode()).hexdigest()


def identity(row):
    chat = row['prompt']
    question = '\n'.join(message['content'] for message in chat)
    extra = row.get('extra_info', {})
    source_id = extra.get('index') if isinstance(extra, dict) else None
    if source_id is None:
        source_id = normalized_hash(question)
    return dict(qid=json.dumps([str(row['data_source']), str(source_id)]),
                question_hash=question_hash(question), normalized_question_hash=normalized_hash(question))


def initialize_records(canonical_rows, cached_rows, *, train_sha256, seed=1729, epoch_questions=5120, diagnostic_questions=512):
    """Pure CPU function shared by the remote initializer and focused fixtures."""
    cached_train = {row['qid']: row['question_hash'] for row in cached_rows if row['split'] == 'train'}
    cached_test = {row['qid']: row['question_hash'] for row in cached_rows if row['split'] == 'test'}
    if len(cached_train) != epoch_questions or len(cached_test) != diagnostic_questions or set(cached_train) & set(cached_test):
        raise ValueError('Cached train/test question membership differs')
    all_rows = [dict(source_index=i, **identity(row)) for i, row in enumerate(canonical_rows)]
    by_id = {row['qid']: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError('Canonical stable question IDs repeat')
    for qid, expected_hash in {**cached_train, **cached_test}.items():
        if qid not in by_id or by_id[qid]['question_hash'] != expected_hash:
            raise ValueError('Cached question is absent or changed in canonical training pool')
    test_text = {by_id[qid]['normalized_question_hash'] for qid in cached_test}
    teacher_text = {by_id[qid]['normalized_question_hash'] for qid in cached_train}
    if teacher_text & test_text or len(teacher_text) != epoch_questions:
        raise ValueError('Cached teacher overlaps diagnostic text or repeats question content')
    # Preserve cached teacher's canonical representative if duplicate corpus rows
    # contain that text, so epoch1 really counts toward first-cycle consumption.
    preferred = {by_id[qid]['normalized_question_hash']: qid for qid in cached_train}
    seen = set(); eligible = []
    for row in all_rows:
        normalized = row['normalized_question_hash']
        if row['qid'] in cached_test or normalized in test_text or normalized in seen:
            continue
        if normalized in preferred and row['qid'] != preferred[normalized]:
            continue
        eligible.append(row); seen.add(normalized)
    ids = {row['qid']: i for i, row in enumerate(eligible)}
    initial = [ids[qid] for qid in cached_train]
    initial_set = set(initial)
    remaining = [i for i in range(len(eligible)) if i not in initial_set]
    random.Random(seed).shuffle(remaining)
    if len(eligible) < epoch_questions or len(initial) != epoch_questions:
        raise ValueError('Eligible unique training pool is too small')
    pool = dict(schema_version=1, role='canonical_nonoverlap_question_pool',
                canonical_train_sha256=train_sha256, canonical_rows=len(all_rows),
                eligible_unique_questions=len(eligible), excluded_diagnostic_questions=diagnostic_questions,
                normalized_identity='NFKC_full_original_prompt_whitespace_collapse_case_sensitive',
                fixed_test_qids_sha256=hashlib.sha256(canonical(sorted(cached_test))).hexdigest(),
                fixed_test_normalized_text_sha256=hashlib.sha256(canonical(sorted(test_text))).hexdigest(),
                records=eligible, epoch1_cached_qids=list(cached_train))
    state = dict(schema_version=1, seed=seed, cycle=0, order=initial + remaining,
                 cursor=epoch_questions, next_epoch=2, consumed_questions=epoch_questions,
                 epoch_questions=epoch_questions,
                 epoch1_source='immutable_original_cached_5120_G5', question_cycle_policy='reshuffle_after_exhaustion')
    return pool, state


def next_selection(pool, state, epoch):
    """Consume a permutation; defer within-epoch repeats at cycle boundaries."""
    if epoch != state['next_epoch'] or epoch < 2:
        raise ValueError('Epoch pool selection must advance in order')
    out = dict(state); out['order'] = list(state['order'])
    selected = []; cycles = []; count = state['epoch_questions']; size = len(pool['records'])
    while len(selected) < count:
        if out['cursor'] == len(out['order']):
            out['cycle'] += 1
            shuffled = list(range(size)); random.Random(out['seed'] + out['cycle']).shuffle(shuffled)
            # Carry the old cycle's remainder first. A repeated question cannot
            # enter the same epoch when the new cycle starts mid-epoch.
            already = set(selected)
            out['order'] = [i for i in shuffled if i not in already] + [i for i in shuffled if i in already]
            out['cursor'] = 0
        take = min(count - len(selected), len(out['order']) - out['cursor'])
        part = out['order'][out['cursor']:out['cursor'] + take]
        if set(part) & set(selected):
            raise ValueError('A question repeated inside one epoch')
        selected.extend(part); cycles.extend([out['cycle']] * len(part)); out['cursor'] += take
    out['next_epoch'] = epoch + 1; out['consumed_questions'] += len(selected)
    rows = [dict(pool['records'][i], pool_index=i, pool_cycle=cycle)
            for i, cycle in zip(selected, cycles)]
    selection = dict(schema_version=1, role='fresh_rollout_epoch_selection', epoch=epoch,
                     questions=len(rows), trajectories=len(rows) * 5,
                     trajectory_group_size=5, fresh_rollout_required=True,
                     canonical_train_sha256=pool['canonical_train_sha256'],
                     fixed_test_qids_sha256=pool['fixed_test_qids_sha256'],
                     fixed_test_normalized_text_sha256=pool['fixed_test_normalized_text_sha256'],
                     question_order_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
                     pool_cycle_start=cycles[0], pool_cycle_end=cycles[-1], rows=rows)
    return selection, out


def prepare(config, epoch):
    import fcntl
    root = Path(config['base']) / 'question-pool'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pool_path = root / 'pool.json'; state_path = root / 'state.json'
        if not pool_path.exists():
            import pyarrow.parquet as pq
            path = Path(config['canonical_train'])
            if digest(path) != config['canonical_train_sha256']:
                raise ValueError('Canonical training parquet changed')
            cached = [json.loads(line) for line in (Path(config['cached_inputs']) / 'index.jsonl').open()]
            # Passing only identities keeps original feature/trajectory caches immutable.
            pool, state = initialize_records(pq.read_table(path).to_pylist(), cached,
                train_sha256=config['canonical_train_sha256'], seed=config.get('seed', 1729))
            immutable(pool_path, pool); atomic(state_path, state)
        pool, state = read(pool_path), read(state_path)
        if epoch == 1:
            return dict(epoch=1, source='cached', cached_inputs=config['cached_inputs'],
                        questions=5120, trajectories=25600, pool_sha256=digest(pool_path))
        target = Path(config['base']) / 'epochs' / ('epoch-%04d' % epoch) / 'selection.json'
        if target.exists():
            saved = read(target)
            if state['next_epoch'] == epoch:
                # Crash after immutable receipt and before state publication.
                expected, after = next_selection(pool, state, epoch)
                if saved != dict(expected, pool_sha256=digest(pool_path)):
                    raise ValueError('Pending selection receipt differs')
                atomic(state_path, after)
            elif state['next_epoch'] != epoch + 1:
                # Historical selection may be reused for observation only; never
                # roll back the cursor to produce a different trajectory pool.
                if state['next_epoch'] < epoch:
                    raise ValueError('Pool cursor precedes an existing future selection')
            return enrich_selection(config, target, saved)
        selection, after = next_selection(pool, state, epoch)
        selection['pool_sha256'] = digest(pool_path)
        immutable(target, selection); atomic(state_path, after)
        return enrich_selection(config, target, selection)


def enrich_selection(config, selection_path, selection):
    """Save the selected original questions and gold answers for later reuse."""
    import pyarrow.parquet as pq
    path = selection_path.parent / 'questions.json'
    if not path.exists():
        columns = ['prompt', 'reward_model', 'data_source', 'extra_info']
        table = pq.read_table(config['canonical_train'], columns=columns)
        selected = table.take([row['source_index'] for row in selection['rows']]).to_pylist()
        rows = []
        for binding, original in zip(selection['rows'], selected):
            actual = identity(original)
            if any(actual[k] != binding[k] for k in actual):
                raise ValueError('Selected canonical prompt/gold source changed')
            rows.append(dict(binding, **{k: original[k] for k in columns}))
        immutable(path, dict(schema_version=1, role='immutable_fresh_epoch_question_gold_bundle',
            epoch=selection['epoch'], selection_sha256=digest(selection_path),
            canonical_train_sha256=selection['canonical_train_sha256'], questions=5120,
            seed=config.get('seed', 1729), rows=rows))
    saved = read(path)
    if saved['selection_sha256'] != digest(selection_path) or saved['epoch'] != selection['epoch']:
        raise ValueError('Saved question bundle differs from its epoch selection')
    return dict(selection_path=str(selection_path), selection_sha256=digest(selection_path),
                questions_path=str(path), questions_sha256=digest(path), **selection)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--epoch', required=True, type=int)
    args = parser.parse_args(); value = prepare(read(args.config), args.epoch)
    print(json.dumps({k: v for k, v in value.items() if k != 'rows'}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
