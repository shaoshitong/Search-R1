"""Read actual request-group stop metadata without changing sampling.

These private records bind a real caller, sequence and completion. They do not
authenticate GPU probabilities, a checkpoint, or policy health.
"""
import copy
import hashlib
import inspect
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

ROLE = 'common8_actual_request_group_stop_provenance'
GENERATION_ROLE = 'common8_actual_loaded_generation_config'
LIMIT = 4 * 1024 * 1024
SAMPLING_KEYS = frozenset(('n', 'best_of', 'temperature', 'top_p', 'top_k', 'min_p',
    'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'logprobs', 'prompt_logprobs',
    'max_tokens', 'min_tokens', 'stop', 'stop_token_ids', 'ignore_eos', 'include_stop_str_in_output',
    'skip_special_tokens', 'spaces_between_special_tokens', 'truncate_prompt_tokens', 'seed',
    'detokenize', 'logits_processors', 'logit_bias', 'allowed_token_ids', 'guided_decoding',
    'do_sample', 'logprob_semantics'))
GENERATION_FIELDS = frozenset(('schema_version', 'role', 'run_id', 'slurm_job_id',
    'source_commit', 'source_map_sha256', 'model_path', 'generation_config_path',
    'generation_config_sha256', 'generation_config_eos_token_ids', 'loader_source',
    'actual_loaded_fields_sha256'))
PROVENANCE_FIELDS = frozenset(('schema_version', 'role', 'rollout_call_id', 'row_id',
    'request_id', 'seq_id', 'actual_request_tokens_sha256', 'caller_sampling_sha256',
    'effective_sampling_sha256', 'caller_is_actual_group_input', 'seq_eos_token_id',
    'tokenizer_eos_token_id', 'tokenizer_pad_token_id', 'effective_stop_token_ids',
    'effective_all_stop_token_ids', 'effective_sampling', 'generation_config', 'update_source'))


def _require(condition, reason):
    if not condition:
        raise ValueError('Invalid actual sampling capture: ' + reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _read(path):
    path = Path(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_size <= LIMIT, 'nonregular or oversized metadata')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        _require(identity(opened) == identity(before), 'metadata changed before open')
        pieces, count = [], 0
        while True:
            piece = os.read(fd, min(65536, LIMIT + 1 - count))
            if not piece:
                break
            pieces.append(piece); count += len(piece)
            _require(count <= LIMIT, 'oversized metadata')
        raw = b''.join(pieces)
        _require(identity(opened) == identity(os.fstat(fd)) == identity(path.lstat()) and
                 len(raw) == opened.st_size, 'metadata changed during read')
        return raw
    finally:
        os.close(fd)


def _ids(values, name, optional=False):
    if optional and values is None:
        return []
    if type(values) is int:
        values = [values]
    _require(type(values) in (list, tuple, set) and
             all(type(value) is int and value >= 0 for value in values), 'invalid IDs: ' + name)
    result = list(values)
    _require(len(result) == len(set(result)), 'duplicate IDs: ' + name)
    return sorted(result)


def callable_source(function):
    path = inspect.getsourcefile(function)
    _require(path is not None, 'missing actual installed callable source')
    path = Path(path).resolve()
    raw = _read(path)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                callable=function.__module__ + '.' + function.__qualname__)


def generation_source(model_config, loader, *, run=None):
    """Bind the original local file immediately before the real engine loader."""
    if run is None:
        run = _json(_read(Path(os.environ['SEARCH_G1_BASE']) / 'logs/run-manifest.json'))
    _require(run['profile'] == 'common8' and run['source_sha256'], 'missing actual common8 run')
    model = Path(model_config.model).resolve()
    expected_model = Path(run['initial_model']).resolve()
    restore = run.get('full_state_restore')
    if restore:
        # Admit only the independently validated checkpoint of this experiment.
        # Sampling metadata must still match the original official file below.
        from search_g1.full_state_restore import source, digest as file_digest
        receipt, verified, checkpoint, _, parent = source(restore['source_receipt'])
        _require(file_digest(receipt) == restore['source_receipt_sha256'] and
                 checkpoint == Path(restore['source_checkpoint']).resolve() and
                 parent['run_id'] == run['run_id'] and
                 parent['initial_model_sha256'] == run['initial_model_sha256'],
                 'restored sampling source binding changed')
        expected_model = checkpoint
    _require(model == expected_model and model.is_absolute(), 'loader uses a different model')
    path = model / 'generation_config.json'
    raw = _read(path); sha = hashlib.sha256(raw).hexdigest()
    if restore:
        original_raw = _read(Path(run['initial_model']) / 'generation_config.json')
        _require(hashlib.sha256(original_raw).hexdigest() ==
                 run['initial_model_sha256']['generation_config.json'], 'original generation file changed')
        if verified.get('kind') == 'dynrpo7b_native4':
            # Native producer receipts hash model/state files, not necessarily
            # save_pretrained's generation metadata. Authenticate every sampling
            # field against the pinned official file below; do not invent a
            # historical producer hash for an older native checkpoint.
            saved_sha = verified['actor_ref']['files_sha256'].get('generation_config.json')
            if saved_sha is not None:
                _require(sha == saved_sha, 'saved generation file changed')
        else:
            saved = next(value for value in verified['files']
                         if value['relative_path'] == 'generation_config.json')
            _require(sha == saved['full_read_sha256'], 'saved generation file changed')
        # save_pretrained updates this serialization-version field. Every
        # actual sampling field, including EOS IDs, must remain exactly equal.
        original_fields = _json(original_raw)
        saved_fields = _json(raw)
        for fields in (original_fields, saved_fields):
            _require(type(fields) is dict, 'generation file is not a document')
            fields.pop('transformers_version', None)
        _require(canonical(saved_fields) == canonical(original_fields),
                 'saved generation sampling fields changed')
    else:
        _require(sha == run['initial_model_sha256']['generation_config.json'], 'original generation file changed')
    document = _json(raw)
    _require(type(document) is dict, 'generation file is not a document')
    eos = _ids(document.get('eos_token_id'), 'generation EOS', optional=True)
    from reproduction.common8.topology import execution_job_id
    record = dict(schema_version=1, role=GENERATION_ROLE, run_id=run['run_id'],
        slurm_job_id=execution_job_id(run), source_commit=run['source_commit'],
        source_map_sha256=digest(run['source_sha256']), model_path=str(model),
        generation_config_path=str(path), generation_config_sha256=sha,
        generation_config_eos_token_ids=eos, loader_source=callable_source(loader))
    return document, record


def loaded_generation(source, actual_fields, expected_fields):
    """Keep actual loaded fields; independently compare to the bound original file."""
    _require(type(actual_fields) is dict and type(expected_fields) is dict and
             canonical(actual_fields) == canonical(expected_fields), 'actual loader fields differ from original file')
    _require(hashlib.sha256(_read(source['generation_config_path'])).hexdigest() ==
             source['generation_config_sha256'], 'generation file changed across actual loader')
    _require(_ids(actual_fields.get('eos_token_id'), 'loaded EOS', optional=True) ==
             source['generation_config_eos_token_ids'], 'actual loaded EOS differs')
    result = copy.deepcopy(source)
    result['actual_loaded_fields_sha256'] = digest(actual_fields)
    return result


def _sha(value, length=64):
    return type(value) is str and len(value) == length and all(c in '0123456789abcdef' for c in value)


def _source(value):
    _require(type(value) is dict and set(value) == {'path', 'sha256', 'callable'} and
             type(value['path']) is str and Path(value['path']).is_absolute() and
             _sha(value['sha256']) and type(value['callable']) is str and value['callable'],
             'invalid actual callable source binding')


def validate_effective_stop_provenance(proof, *, request_id, actual_request_token_ids,
                                      sampling, eos_token_id, pad_token_id):
    """Validate serialized metadata only; return (effective stop set, seq EOS).

    Actual caller identity and group/file readings are enforced at capture time.
    The caller must additionally bind run/source/job to its outer identity. No
    serialized Boolean or SHA proves fresh GPU execution by itself.
    """
    _require(type(proof) is dict and set(proof) == PROVENANCE_FIELDS and
             type(proof['schema_version']) is int and proof['schema_version'] == 1 and proof['role'] == ROLE,
             'stop provenance schema changed')
    for key in ('rollout_call_id', 'row_id', 'request_id'):
        _require(type(proof[key]) is str and proof[key], 'missing actual identity: ' + key)
    _require(type(proof['seq_id']) is int and proof['seq_id'] >= 0 and
             proof['request_id'] == request_id and proof['caller_is_actual_group_input'] is True,
             'actual request/caller binding changed')
    tokens = list(actual_request_token_ids)
    _require(tokens and all(type(token) is int and token >= 0 for token in tokens) and
             proof['actual_request_tokens_sha256'] == digest(tokens), 'actual request token binding changed')
    for key in ('actual_request_tokens_sha256', 'caller_sampling_sha256', 'effective_sampling_sha256'):
        _require(_sha(proof[key]), 'invalid metadata digest')
    _require(type(eos_token_id) is int and eos_token_id >= 0 and
             type(pad_token_id) is int and pad_token_id >= 0 and
             type(proof['tokenizer_eos_token_id']) is int and
             type(proof['tokenizer_pad_token_id']) is int and
             proof['tokenizer_eos_token_id'] == eos_token_id and proof['tokenizer_pad_token_id'] == pad_token_id,
             'actual tokenizer EOS/PAD binding changed')
    eos = proof['seq_eos_token_id']
    _require(eos is None or type(eos) is int and eos >= 0, 'invalid actual sequence EOS')
    generation = proof['generation_config']
    _require(type(generation) is dict and set(generation) == GENERATION_FIELDS and
             type(generation['schema_version']) is int and generation['schema_version'] == 1 and
             generation['role'] == GENERATION_ROLE and type(generation['run_id']) is str and
             0 < len(generation['run_id']) <= 512 and
             (generation['slurm_job_id'] is None or
              type(generation['slurm_job_id']) is str and generation['slurm_job_id'].isdigit()) and
             _sha(generation['source_commit'], 40), 'loaded generation provenance schema changed')
    for key in ('source_map_sha256', 'generation_config_sha256', 'actual_loaded_fields_sha256'):
        _require(_sha(generation[key]), 'invalid loaded generation digest')
    model = generation['model_path']
    _require(type(model) is str and Path(model).is_absolute() and '..' not in Path(model).parts and
             generation['generation_config_path'] == str(Path(model) / 'generation_config.json'),
             'generation source path changed')
    _source(generation['loader_source']); _source(proof['update_source'])
    generation_eos = generation['generation_config_eos_token_ids']
    for values, name in ((generation_eos, 'generation EOS'),
                         (proof['effective_stop_token_ids'], 'effective stop'),
                         (proof['effective_all_stop_token_ids'], 'effective all stop')):
        _require(type(values) is list and _ids(values, name) == values, 'noncanonical actual stop IDs')
    effective = proof['effective_sampling']
    _require(type(sampling) is dict and set(sampling) == SAMPLING_KEYS and
             type(effective) is dict and set(effective) == SAMPLING_KEYS and
             proof['caller_sampling_sha256'] == digest(sampling) and
             proof['effective_sampling_sha256'] == digest(effective) and
             canonical({key: value for key, value in effective.items() if key != 'stop_token_ids'}) ==
             canonical({key: value for key, value in sampling.items() if key != 'stop_token_ids'}),
             'caller/effective sampling fields changed')
    _require(type(sampling['do_sample']) is bool and sampling['ignore_eos'] is False and
             type(sampling['min_tokens']) is int and sampling['min_tokens'] == 0,
             'actual EOS suppression is not neutral')
    caller_stop = set(_ids(sampling['stop_token_ids'], 'caller stop', optional=True))
    expected_stop = caller_stop | {value for value in generation_eos if value != eos}
    expected_all = caller_stop | set(generation_eos) | ({eos} if eos is not None else set())
    _require(set(proof['effective_stop_token_ids']) == expected_stop and
             _ids(effective['stop_token_ids'], 'effective parameter stop') == proof['effective_stop_token_ids'] and
             set(proof['effective_all_stop_token_ids']) == expected_all,
             'actual stop IDs differ from bound caller/generation/sequence')
    return frozenset(proof['effective_all_stop_token_ids']), eos


class GroupCapture:
    def __init__(self, engine, caller, caller_metadata, call_id, row_ids, requests):
        self.caller = caller
        self.metadata = copy.deepcopy(caller_metadata)
        _require(type(caller_metadata) is dict and set(caller_metadata) == SAMPLING_KEYS,
                 'caller sampling schema changed')
        self.call_id = call_id
        self.row_ids = list(row_ids)
        self.requests = [list(row) for row in requests]
        _require(type(call_id) is str and call_id and len(self.row_ids) == len(self.requests) > 0 and
                 len(set(self.row_ids)) == len(self.row_ids) and
                 all(type(row) is str and row for row in self.row_ids), 'missing call/row identities')
        for row in self.requests:
            _require(row and all(type(token) is int and token >= 0 for token in row), 'invalid request tokens')
        self.generation = copy.deepcopy(engine._common8_generation_provenance)
        self.tokenizer_eos = engine.tokenizer.eos_token_id
        self.tokenizer_pad = engine.tokenizer.pad_token_id
        _require(self.generation['role'] == GENERATION_ROLE and
                 digest(engine.generation_config_fields) == self.generation['actual_loaded_fields_sha256'],
                 'actual in-memory generation config drifted')
        self.update_source = callable_source(type(caller).update_from_generation_config)
        self.records = {}
        self.pending = set()
        self.sequence_ids = set()
        _require(canonical(self.values(caller)) == canonical(self.metadata), 'caller metadata does not describe actual caller')

    def values(self, params):
        result = {}
        for key, value in self.metadata.items():
            result[key] = copy.deepcopy(value if key in ('do_sample', 'logprob_semantics')
                                       else getattr(params, key, None))
        canonical(result)
        return result

    def before(self, request_id, seq, caller):
        _require(caller is self.caller and canonical(self.values(caller)) == canonical(self.metadata),
                 'different or mutated actual caller')
        _require(type(request_id) is str and request_id and request_id not in self.pending and
                 request_id not in self.records and len(self.records) < len(self.requests), 'duplicate/extra actual request')
        _require(type(seq.seq_id) is int and seq.seq_id >= 0 and seq.seq_id not in self.sequence_ids,
                 'invalid/duplicate actual sequence identity')
        self.pending.add(request_id)
        self.sequence_ids.add(seq.seq_id)

    def after(self, request_id, seq, caller, group):
        _require(request_id in self.pending and group.request_id == request_id and
                 len(group.get_seqs()) == 1 and group.get_seqs()[0] is seq,
                 'actual group/sequence/request identity changed')
        _require(caller is self.caller and canonical(self.values(caller)) == canonical(self.metadata),
                 'engine changed caller policy')
        index = len(self.records)
        tokens = list(group.prompt_token_ids)
        _require(all(type(token) is int and token >= 0 for token in tokens) and
                 tokens == self.requests[index], 'actual group request tokens differ')
        eos = seq.eos_token_id
        _require(eos is None or type(eos) is int and eos >= 0, 'invalid actual sequence EOS')
        effective = self.values(group.sampling_params)
        caller_stop = _ids(self.metadata.get('stop_token_ids'), 'caller stop', optional=True)
        generation_eos = self.generation['generation_config_eos_token_ids']
        expected_all = set(caller_stop) | set(generation_eos) | ({eos} if eos is not None else set())
        expected_stop = set(caller_stop)
        if not effective['ignore_eos']:
            expected_stop.update(value for value in generation_eos if value != eos)
        stop = _ids(group.sampling_params.stop_token_ids, 'effective stop')
        all_stop = _ids(group.sampling_params.all_stop_token_ids, 'effective all stop')
        _require(set(stop) == expected_stop and set(all_stop) == expected_all, 'actual effective stop set differs from bound sources')
        _require(canonical({key: value for key, value in effective.items() if key != 'stop_token_ids'}) ==
                 canonical({key: value for key, value in self.metadata.items() if key != 'stop_token_ids'}),
                 'engine transformed actual sampling policy')
        record = dict(schema_version=1, role=ROLE, rollout_call_id=self.call_id,
            row_id=self.row_ids[index], request_id=request_id, seq_id=seq.seq_id,
            actual_request_tokens_sha256=digest(tokens), caller_sampling_sha256=digest(self.metadata),
            effective_sampling_sha256=digest(effective), caller_is_actual_group_input=True,
            seq_eos_token_id=eos, tokenizer_eos_token_id=self.tokenizer_eos,
            tokenizer_pad_token_id=self.tokenizer_pad, effective_stop_token_ids=stop,
            effective_all_stop_token_ids=all_stop, effective_sampling=effective,
            generation_config=copy.deepcopy(self.generation), update_source=copy.deepcopy(self.update_source))
        self.records[request_id] = record
        validate_effective_stop_provenance(record, request_id=request_id,
            actual_request_token_ids=tokens, sampling=self.metadata,
            eos_token_id=self.tokenizer_eos, pad_token_id=self.tokenizer_pad)
        self.pending.remove(request_id)

    def attach(self, behavior):
        _require(type(behavior) is list and len(behavior) == len(self.records) == len(self.requests) and
                 not self.pending and len({row['request_id'] for row in behavior}) == len(behavior),
                 'missing actual group/completion capture')
        for index, row in enumerate(behavior):
            record = self.records.get(row['request_id'])
            _require(record is not None and record['row_id'] == self.row_ids[index] and
                     row['actual_request_token_ids'] == self.requests[index] and
                     digest(row['actual_request_token_ids']) == record['actual_request_tokens_sha256'],
                     'completion differs from actual request group')
        for row in behavior:
            _require('effective_stop_provenance' not in row, 'duplicate stop provenance')
            row['effective_stop_provenance'] = copy.deepcopy(self.records[row['request_id']])


class RequestCapture:
    """Bind concurrent engine groups to their own caller and submitted tokens."""
    def __init__(self, engine):
        self.engine = engine
        self.requests = {}
        self.seen_requests = set()
        self.seen_sequences = set()

    def register(self, request_id, caller, metadata, call_id, row_id, tokens):
        _require(type(request_id) is str and request_id and request_id not in self.seen_requests,
                 'request identity reused inside sampling phase')
        self.requests[request_id] = GroupCapture(self.engine, caller, metadata, call_id, [row_id], [tokens])
        self.seen_requests.add(request_id)

    def before(self, request_id, seq, caller):
        _require(request_id in self.requests and seq.seq_id not in self.seen_sequences,
                 'unregistered request or reused engine sequence')
        self.requests[request_id].before(request_id, seq, caller)
        self.seen_sequences.add(seq.seq_id)

    def after(self, request_id, seq, caller, group):
        _require(request_id in self.requests, 'unregistered actual request group')
        self.requests[request_id].after(request_id, seq, caller, group)

    def attach(self, behavior):
        request_id = behavior['request_id']
        _require(request_id in self.requests, 'unknown or repeated engine completion')
        self.requests[request_id].attach([behavior])
        del self.requests[request_id]

    def abort(self, request_id):
        _require(request_id in self.requests, 'unknown request cancellation')
        del self.requests[request_id]


class ContinuousRequestService:
    """Single-owner SyncEngine scheduler; independent requests share decode steps.

    Ray serializes calls on each rank. No background thread, batch generate,
    distributed operation or completion-order row inference is used here.
    """
    def __init__(self, engine, capture, behavior_reader):
        self.engine = engine
        self.capture = capture
        self.behavior_reader = behavior_reader
        self.pending = {}
        self.completed_count = 0
        self.engine_steps = 0

    def submit(self, request_id, tokens, caller, metadata, context):
        _require(request_id not in self.pending, 'duplicate pending request')
        self.capture.register(request_id, caller, metadata, context['rollout_call_id'], context['row_id'], tokens)
        self.pending[request_id] = copy.deepcopy(context)
        # vLLM 0.6.3 accepts TokensPrompt here, preserving actual caller identity.
        self.engine.add_request(request_id, {'prompt_token_ids': list(tokens)}, caller)

    def advance(self, max_steps=8):
        _require(type(max_steps) is int and 1 <= max_steps <= 16, 'unbounded engine advance')
        completed = []
        for _ in range(max_steps):
            if not self.pending:
                _require(not self.engine.has_unfinished_requests(), 'engine has unbound requests')
                break
            _require(self.engine.has_unfinished_requests(), 'pending requests disappeared from engine')
            outputs = self.engine.step()
            self.engine_steps += 1
            for output in outputs:
                _require(output.request_id in self.pending, 'engine returned unbound request')
                if not output.finished:
                    continue
                rows = self.behavior_reader([output])
                _require(type(rows) is list and len(rows) == 1 and rows[0]['request_id'] == output.request_id,
                         'completion behavior identity differs')
                self.capture.attach(rows[0])
                context = self.pending.pop(output.request_id)
                completed.append(dict(request_id=output.request_id, behavior=rows[0], context=context))
                self.completed_count += 1
            if completed:
                break
        return completed

    def close(self, abort=False):
        _require(type(abort) is bool, 'ambiguous phase cancellation')
        if abort:
            for request_id in list(self.pending):
                self.engine.abort_request(request_id)
                self.capture.abort(request_id)
                del self.pending[request_id]
        _require(not self.pending and not self.capture.requests and not self.engine.has_unfinished_requests(),
                 'sampling phase closed before requests finished')
        return dict(completed_requests=self.completed_count, engine_steps=self.engine_steps)


@contextmanager
def capture_scope(engine, caller, metadata, call_id, row_ids, requests):
    previous = getattr(engine, '_common8_stop_capture', None)
    _require(previous is None, 'nested actual capture')
    capture = GroupCapture(engine, caller, metadata, call_id, row_ids, requests)
    engine._common8_stop_capture = capture
    try:
        yield capture
    finally:
        engine._common8_stop_capture = previous
