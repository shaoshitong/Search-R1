"""Fresh candidate permutations over the existing diagnostic-excluded pool."""
import fcntl
import hashlib
import json
import random
from pathlib import Path

from reproduction.prefix20_online.pool import atomic, canonical, digest, identity, immutable, read
from .contract import PERIODIC_CANDIDATES, SELECTED_QUESTIONS, G
from .candidate_schedule import (candidate_count, consumed_candidates_through,
                                 validate_schedule, schedule_metadata)


def fixed_test_identities(config):
    """Read only the explicit512-G5 identity manifest, never cached features."""
    path, expected_sha = config.get('fixed_test_identity_path'), config.get('fixed_test_identity_sha256')
    if path is None and expected_sha is None:
        return None
    if (not isinstance(path, str) or not path.startswith(('/data/', '/mnt/'))
            or not isinstance(expected_sha, str) or len(expected_sha) != 64
            or digest(path) != expected_sha):
        raise ValueError('Fixed diagnostic identity path or exact file SHA differs')
    manifest = read(path)
    rows = manifest.get('records')
    if (manifest.get('schema_version') != 1 or manifest.get('questions') != 512
            or manifest.get('trajectories') != 2560 or not isinstance(rows, list) or len(rows) != 512):
        raise ValueError('Fixed diagnostic identity manifest must describe exactly512 original G5s')
    qids, texts = set(), set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('qid'), str)
                or not row['qid'] or row['qid'] in qids
                or any(not isinstance(row.get(key), str) or len(row[key]) != 64
                       for key in ('question_hash', 'normalized_question_hash'))
                or row['normalized_question_hash'] in texts):
            raise ValueError('Fixed diagnostic question identity is missing or duplicated')
        qids.add(row['qid']); texts.add(row['normalized_question_hash'])
    return rows


def initial_state(size, seed=1729):
    order = list(range(size))
    random.Random(seed).shuffle(order)
    # Keep the historical declaration byte-compatible on full-state resume.
    # The next reservation budget comes exclusively from candidate_schedule.
    return dict(seed=seed, cycle=0, order=order, cursor=0, next_outer=1,
                consumed_candidates=0, candidate_questions=PERIODIC_CANDIDATES)


def reserve_indices(state, outer, count=PERIODIC_CANDIDATES):
    """Cross-cycle reordering prevents a duplicate in the boundary batch."""
    if state['next_outer'] != outer or len(state['order']) < count:
        raise ValueError('Candidate pool order or size differs')
    out = dict(state, order=list(state['order']))
    selected, cycles = [], []
    size = len(out['order'])
    while len(selected) < count:
        if out['cursor'] == size:
            out['cycle'] += 1
            order = list(range(size))
            random.Random(out['seed'] + out['cycle']).shuffle(order)
            already = set(selected)
            out['order'] = [i for i in order if i not in already] + [i for i in order if i in already]
            out['cursor'] = 0
        n = min(count - len(selected), size - out['cursor'])
        part = out['order'][out['cursor']:out['cursor'] + n]
        if set(part) & set(selected):
            raise ValueError('Candidate repeats within one outer')
        selected.extend(part)
        cycles.extend([out['cycle']] * n)
        out['cursor'] += n
    out.update(next_outer=outer + 1, consumed_candidates=state['consumed_candidates'] + count)
    return selected, cycles, out



def reserve_for_outer(state, config, outer):
    """Continue the exact retained permutation/cursor/RNG with the next budget."""
    consumed = consumed_candidates_through(config, outer - 1)
    if (state['next_outer'] != outer or state['consumed_candidates'] != consumed
            or state['cycle'] * len(state['order']) + state['cursor'] != consumed):
        raise ValueError('Retained candidate pool clock differs from the segmented schedule')
    return reserve_indices(state, outer, candidate_count(config, outer))


def state_sha(state):
    return hashlib.sha256(canonical(state)).hexdigest()


def prepare(config, outer):
    """Reserve once, then retain original question/gold and real candidate order."""
    validate_schedule(config)
    count = candidate_count(config, outer)
    metadata = schedule_metadata(config, outer)
    if config.get('selected_questions') != SELECTED_QUESTIONS:
        raise ValueError('Candidate scheduling retains exactly512 selected questions')
    base = Path(config['base'])
    if not str(base).startswith(('/mnt/data/shaoshitong/search-g1-prefix-selection200-',
                                  '/data/shaoshitong/search-g1-prefix-selection200-',
                                  '/data/shaoshitong/Search-G1-prefix-selection200-',
                                  '/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')):
        raise ValueError('Candidate artifacts must stay in the remote experiment')
    root = base / 'question-pool'
    target = base / 'outers' / ('outer-%06d' % outer)
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    with (root / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = Path(config['source_question_pool'])
        fixed = fixed_test_identities(config)
        fixed_qids = {row['qid'] for row in fixed} if fixed is not None else None
        fixed_texts = {row['normalized_question_hash'] for row in fixed} if fixed is not None else None
        binding_path = root / 'binding.json'
        if not binding_path.exists():
            pool_sha = digest(source)
            pool = read(source)
            if (pool['canonical_train_sha256'] != config['canonical_train_sha256']
                    or pool['canonical_rows'] != 169028
                    or pool['excluded_diagnostic_questions'] != 512
                    or pool['eligible_unique_questions'] != len(pool['records'])):
                raise ValueError('Original diagnostic-excluded question pool differs')
            if fixed is not None and any(row['qid'] in fixed_qids or
                    row['normalized_question_hash'] in fixed_texts for row in pool['records']):
                raise ValueError('Candidate source overlaps the explicit fixed512 question identities')
            immutable(binding_path, dict(source=str(source), sha256=pool_sha,
                candidate_questions=PERIODIC_CANDIDATES, responses_per_question=G,
                eligible_questions=len(pool['records']),
                fixed_test_qids_sha256=pool['fixed_test_qids_sha256'],
                fixed_test_normalized_text_sha256=pool['fixed_test_normalized_text_sha256'],
                fixed_test_features_manifest=config.get('fixed_test_manifest_sha256'),
                fixed_test_identity_path=config.get('fixed_test_identity_path'),
                fixed_test_identity_sha256=config.get('fixed_test_identity_sha256')))
            atomic(root / 'state.json', initial_state(len(pool['records']), config.get('seed', 1729)))
        binding = read(binding_path)
        if (binding['source'] != str(source)
                or binding.get('candidate_questions') != PERIODIC_CANDIDATES
                or binding.get('responses_per_question') != G
                or binding.get('fixed_test_identity_path') != config.get('fixed_test_identity_path')
                or binding.get('fixed_test_identity_sha256') != config.get('fixed_test_identity_sha256')):
            raise ValueError('Candidate source changed')
        pool, state = read(source), read(root / 'state.json')
        if state.get('candidate_questions') != PERIODIC_CANDIDATES:
            raise ValueError('Pool state belongs to another candidate budget')
        reservation = target / 'pool-reservation.json'
        if reservation.exists():
            saved = read(reservation)
            if (saved['outer'] != outer or saved['pool_sha256'] != binding['sha256']
                    or any(saved.get(key) != value for key, value in metadata.items())):
                raise ValueError('Existing candidate reservation differs')
            if state_sha(state) == saved['before_sha256']:
                atomic(root / 'state.json', saved['after'])
            elif state_sha(state) != saved['after_sha256']:
                raise ValueError('Pool moved beyond this uncompleted outer; inspect before replay')
            indices, cycles, after = saved['indices'], saved['cycles'], saved['after']
        else:
            indices, cycles, after = reserve_for_outer(state, config, outer)
            immutable(reservation, dict(outer=outer, pool_sha256=binding['sha256'],
                before_sha256=state_sha(state), after_sha256=state_sha(after), after=after,
                indices=indices, cycles=cycles, **metadata))
            atomic(root / 'state.json', after)
        if (after['next_outer'] != outer + 1
                or after['consumed_candidates'] != consumed_candidates_through(config, outer)
                or after['cycle'] * len(after['order']) + after['cursor'] != after['consumed_candidates']
                or len(indices) != count or len(cycles) != count
                or len(set(indices)) != count):
            raise ValueError('Reservation must contain every scheduled distinct original question index')
        rows = [dict(pool['records'][index], pool_index=index, pool_cycle=cycle,
                     candidate_ordinal=ordinal)
                for ordinal, (index, cycle) in enumerate(zip(indices, cycles))]
        if fixed is not None and any(row['qid'] in fixed_qids or
                row['normalized_question_hash'] in fixed_texts for row in rows):
            raise ValueError('Reserved candidates overlap the explicit fixed512 diagnostic')
        selection = dict(run_id=config['run_id'], outer=outer, questions=count,
            trajectory_group_size=G, trajectories=count*G, source='fresh_prefix_sampling',
            pool_sha256=binding['sha256'], canonical_train_sha256=config['canonical_train_sha256'],
            fixed_test_qids_sha256=binding['fixed_test_qids_sha256'],
            fixed_test_normalized_text_sha256=binding['fixed_test_normalized_text_sha256'], rows=rows, **metadata)
        immutable(target / 'selection.json', selection)
        question_path = target / 'questions.json'
        if not question_path.exists():
            import pyarrow.parquet as pq
            table = pq.read_table(config['canonical_train'], columns=['prompt', 'reward_model', 'data_source', 'extra_info'])
            originals = table.take([row['source_index'] for row in rows]).to_pylist()
            questions = []
            for record, original in zip(rows, originals):
                actual = identity(original)
                if any(actual[key] != record[key] for key in actual):
                    raise ValueError('Actual candidate question/gold identity changed')
                questions.append(dict(record, **original))
            immutable(question_path, dict(run_id=config['run_id'], outer=outer,
                selection_sha256=digest(target / 'selection.json'), questions=count, rows=questions, **metadata))
        if read(question_path)['selection_sha256'] != digest(target / 'selection.json'):
            raise ValueError('Question bundle is not bound to this reservation')
        return question_path
