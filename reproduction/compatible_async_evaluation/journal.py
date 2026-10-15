"""Scalar-only legacy evaluation journal, ordered by canonical occurrence ordinal.

Never recover unjournalled results from console averages. A batch-size change
resumes at the durable prefix, retaining repeated questions as separate rows.
"""
import hashlib
import fcntl
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from reproduction.compatible_async_evaluation.contract import PROTOCOL_PATH, digest, protocol, verify_result


def question_count(binding):
    return protocol()['datasets'][binding['split']]['qa_count']
_ROW_FIELDS = frozenset(('question_id', 'data_source', 'em', 'f1', 'actual_calls',
    'valid_action', 'turns', 'answer_present', 'answer_sha256',
    'response_token_count', 'model_token_count'))
_BATCH = re.compile(r'batch-(\d{8})-(\d{8})\.json\Z')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def make_binding(run_id, checkpoint_manifest_sha256, validation_sha256,
                 question_ids, data_sources, checkpoint_step=200, split='final'):
    spec = protocol()
    ids, sources = list(question_ids), list(data_sources)
    _require(isinstance(run_id, str) and bool(run_id), 'Missing logical run identity')
    _require(type(checkpoint_step) is int and checkpoint_step in spec['checkpoints']['admitted_steps'], 'Unsupported legacy checkpoint step')
    _require(split in ('validation','final') and (split!='final' or checkpoint_step==200), 'Legacy final requires CP200')
    _require(_hash(checkpoint_manifest_sha256) and _hash(validation_sha256), 'Invalid checkpoint/data SHA')
    _require(len(ids) == len(sources) == spec['datasets'][split]['qa_count'], 'Need complete canonical split occurrences')
    _require(validation_sha256 == spec['datasets'][split]['sha256'], 'Legacy dataset SHA differs')
    _require(all(isinstance(v, str) and v for v in ids + sources), 'Invalid canonical identity')
    return dict(schema_version=1, role='compatible_async_evaluation_resume_binding', split=split, recipe_id=spec['recipe_id'], recipe_sha256=digest(PROTOCOL_PATH), run_id=run_id,
                checkpoint_step=checkpoint_step, checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                validation_sha256=validation_sha256, question_count=len(ids),
                source_counts=dict(Counter(sources)),
                occurrence_sha256=_sha([[i, q, s] for i, (q, s) in enumerate(zip(ids, sources))]))


def _check_binding(binding, ids, sources):
    expected = make_binding(binding['run_id'], binding['checkpoint_manifest_sha256'],
                            binding['validation_sha256'], ids, sources,
                            binding['checkpoint_step'], binding['split'])
    _require(binding == expected, 'Canonical occurrence/checkpoint binding changed')


def _row(row, ordinal, ids, sources):
    _require(isinstance(row, dict) and set(row) in (_ROW_FIELDS, _ROW_FIELDS | {'ordinal'}),
             'Unexpected measured row fields')
    if 'ordinal' in row:
        _require(type(row['ordinal']) is int and row['ordinal'] == ordinal, 'Occurrence ordinal changed')
    _require(0 <= ordinal < len(ids) and row['question_id'] == ids[ordinal] and
             row['data_source'] == sources[ordinal], 'Ordered question/source changed')
    for key in ('em', 'actual_calls', 'valid_action', 'turns', 'response_token_count', 'model_token_count'):
        _require(type(row[key]) is int and row[key] >= 0, 'Invalid integer: ' + key)
    _require(row['em'] in (0, 1) and type(row['f1']) in (int, float) and
             math.isfinite(row['f1']) and row['em'] <= row['f1'] <= 1, 'Invalid measured EM/F1')
    _require(1 <= row['turns'] <= 5 and row['actual_calls'] <= 4 and
             row['actual_calls'] <= row['valid_action'] <= row['turns'],
             'Invalid actual action counts')
    _require(row['model_token_count'] <= row['response_token_count'], 'Invalid token counts')
    _require(type(row['answer_present']) is bool, 'Invalid answer presence')
    _require((_hash(row['answer_sha256']) if row['answer_present'] else
             row['answer_sha256'] is None and row['em'] == row['f1'] == 0), 'Invalid answer/scoring identity')
    return dict(row, ordinal=ordinal)


def _atomic_once(path, value):
    """Publish a complete, fsynced document without overwriting an existing one."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.journal-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        # Same-filesystem link is an atomic exclusive publication. A crash before
        # publication leaves an ignored temporary file, never a partial batch.
        os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(name)


def _initialize(directory, binding):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _require(not directory.is_symlink(), 'Journal directory is a symlink')
    path = directory / 'binding.json'
    if not path.exists():
        try:
            _atomic_once(path, binding)
        except FileExistsError:
            pass
    _require(not path.is_symlink() and json.loads(path.read_text()) == binding,
             'Journal belongs to different evaluation')
    return directory


def load_prefix(directory, binding, question_ids, data_sources):
    """Authenticate every durable batch; refuse gaps, overlaps or changed rows."""
    ids, sources = list(question_ids), list(data_sources)
    _check_binding(binding, ids, sources)
    directory = _initialize(directory, binding)
    rows, previous = [], None
    batches = []
    for path in directory.iterdir():
        match = _BATCH.fullmatch(path.name)
        if match:
            batches.append((int(match[1]), int(match[2]), path))
        elif path.name.startswith('batch-'):
            raise ValueError('Malformed batch filename')
    for start, stop, path in sorted(batches):
        _require(start == len(rows) and start < stop <= question_count(binding) and not path.is_symlink(),
                 'Gap/overlap or invalid journal range')
        value = json.loads(path.read_text())
        _require(set(value) == {'schema_version', 'binding_sha256', 'start', 'stop',
                 'previous_batch_sha256', 'rows', 'rows_sha256'} and
                 value['schema_version'] == 1 and value['binding_sha256'] == _sha(binding) and
                 value['start'] == start and value['stop'] == stop and
                 value['previous_batch_sha256'] == previous, 'Batch identity/chain changed')
        measured = value['rows']
        _require(isinstance(measured, list) and len(measured) == stop - start and
                 value['rows_sha256'] == _sha(measured), 'Batch scores changed')
        rows.extend(_row(row, start + index, ids, sources) for index, row in enumerate(measured))
        previous = _sha(value)
    return rows


def append_batch(directory, binding, question_ids, data_sources, measured_rows):
    """Persist actual scalar rows after successful generation/scoring only."""
    directory = _initialize(directory, binding)
    with (directory / '.append.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _append_locked(directory, binding, question_ids, data_sources, measured_rows)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _append_locked(directory, binding, question_ids, data_sources, measured_rows):
    ids, sources = list(question_ids), list(data_sources)
    prefix = load_prefix(directory, binding, ids, sources)
    start = len(prefix)
    _require(isinstance(measured_rows, list) and bool(measured_rows), 'Empty measured batch')
    rows = [_row(row, start + index, ids, sources) for index, row in enumerate(measured_rows)]
    stop = start + len(rows)
    directory = Path(directory)
    prior = sorted(directory.glob('batch-????????-????????.json'))
    previous = _sha(json.loads(prior[-1].read_text())) if prior else None
    value = dict(schema_version=1, binding_sha256=_sha(binding), start=start, stop=stop,
                 previous_batch_sha256=previous, rows=rows, rows_sha256=_sha(rows))
    _atomic_once(directory / ('batch-%08d-%08d.json' % (start, stop)), value)
    return dict(cursor=stop, question_count=question_count(binding), batch_sha256=_sha(value))


def aggregate_final(binding, question_ids, data_sources, rows):
    """Produce final-identity-compatible metrics only for all 51713 occurrences."""
    ids, sources = list(question_ids), list(data_sources)
    _check_binding(binding, ids, sources)
    _require(len(rows) == question_count(binding), 'Legacy aggregation requires every canonical occurrence')
    measured = [_row(row, index, ids, sources) for index, row in enumerate(rows)]
    em, f1 = [r['em'] for r in measured], [r['f1'] for r in measured]
    calls = sum(r['actual_calls'] for r in measured)
    searched = sum(r['actual_calls'] > 0 for r in measured)
    valid = sum(r['valid_action'] for r in measured)
    turns = sum(r['turns'] for r in measured)
    counts = dict(Counter(sources))
    result = dict(run_id=binding['run_id'], split=binding['split'], step=binding['checkpoint_step'], final=binding['split']=='final',
        recipe_id=binding['recipe_id'], recipe_sha256=binding['recipe_sha256'], canonical_occurrence_order_verified=True, old_content_fp32_async=True,
        question_count=question_count(binding), qa_count=question_count(binding), question_ids=ids, data_sources=sources,
        em=em, f1=f1, rows=measured, counts=counts, source_counts=counts,
        correct=sum(em), weighted_em=math.fsum(em)/question_count(binding), weighted_f1=math.fsum(f1)/question_count(binding),
        validation_sha256=binding['validation_sha256'],
        checkpoint_manifest_sha256=binding['checkpoint_manifest_sha256'],
        occurrence_sha256=binding['occurrence_sha256'],
        source_metrics={s: dict(qa_count=counts[s], correct=sum(r['em'] for r in measured if r['data_source']==s),
            weighted_em=math.fsum(r['em'] for r in measured if r['data_source']==s)/counts[s],
            weighted_f1=math.fsum(r['f1'] for r in measured if r['data_source']==s)/counts[s]) for s in counts},
        executed_search_question_count=searched, executed_search_call_count=calls,
        executed_search_question_rate=searched/question_count(binding), executed_search_calls_per_question=calls/question_count(binding),
        valid_action_count=valid, action_opportunity_count=turns, valid_action_rate=valid/turns,
        journal_rows_sha256=_sha(measured))

    verify_result(result,binding['split'],binding['checkpoint_step'],ids,sources)
    return result
