"""Pinned, event-preserving readers for authentic question + policy5 token scans.

Only small index records and group identities are kept in memory. Token payloads
are immutable NumPy mmaps; this module never decodes, generates, or re-scores a
trajectory. A source scan may retain more than five policy tokens, and authentic prefix5 sources are accepted directly.

Composite manifest schema (all paths refer to execution-host storage)::

  {"schema_version": 1, "prefix_tokens": 5,
   "encoder_revision": REVISION, "question_max_tokens": 273,
   "question_identity_path": "/remote/question-identities.jsonl",
   "question_identity_sha256": "...",
   "sources": [{"source_id": "h100-e0001", "source_event_id": "run:e0001",
     "inputs_dir": "/remote/inputs", "scan_sha256": "...",
     "split": "train", "expected_trajectories": 25600,
     "expected_groups": 5120}]}

Question identities can instead be present on every index row, or supplied per
source. Their normalized hash uses NFKC + whitespace collapse, case sensitive.
The manifest creator must derive those hashes from canonical original prompts.
Repeated qids in different producer events are retained. Only exact producer
event duplicates, with matching labels and retained token payloads, are removed.
"""

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path


REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
POLICY_TOKENS = 5
FEATURE_DIM = 2048
FEATURE_ROLE = 'actual_initial_request_plus_same_trajectory_retained_policy5'
NORMALIZED_IDENTITY = 'NFKC_full_original_prompt_whitespace_collapse_case_sensitive'
ARRAY_NAMES = ('packed_context_ids.npy', 'packed_policy_ids.npy',
               'packed_policy_positions.npy')
SUPPORTED_REPRESENTATIONS = {
    'multi_turn_retained_policy_prefix_layer27_v1',
    'initial_request_and_retained_policy5_layer27_postblock_v1',
    'initial_request_and_retained_policy20_layer27_postblock_v1',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    """Hash a stable regular file without materializing its payload."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError('Immutable input changed during SHA verification: ' + str(path))
    return digest.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def _path(value, parent=None):
    path = Path(value)
    if not path.is_absolute():
        if parent is None:
            raise ValueError('An explicit absolute input path is required')
        path = Path(parent) / path
    return path.resolve(strict=True)


def _digest(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Require an exact lowercase SHA256 for ' + field)
    return value


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer ' + field)
    return value


def expected_policy_positions(segments, max_positions=32768):
    """Preserve native request positions, including resets across tool turns."""
    expected, previous_turn = [], -1
    for segment in segments:
        offset, length, start, turn = (segment[key] for key in
                                      ('policy_start', 'policy_length', 'position_start', 'turn_index'))
        for value, field in ((offset, 'policy_start'), (length, 'policy_length'),
                             (start, 'position_start'), (turn, 'turn_index')):
            _integer(value, field)
        if offset != len(expected) or length < 1 or start < 1 or turn <= previous_turn:
            raise ValueError('Invalid retained policy segment order or coverage')
        if start + length > max_positions:
            raise ValueError('Policy position exceeds the official RoPE range')
        expected.extend(range(start, start + length))
        previous_turn = turn
    if not expected:
        raise ValueError('Zero-token trajectories cannot provide policy5 observations')
    return expected


def initial_question_segment(row):
    segments = row.get('segments')
    if not segments:
        raise ValueError('Missing original initial-request context')
    first = segments[0]
    if first['turn_index'] != 0 or first['policy_start'] != 0:
        raise ValueError('Actual turn0 request is required; a later tool context is not a question')
    length = _integer(first['request_length'], 'initial request length', 1)
    if first['context_length'] < length or first['position_start'] != length:
        raise ValueError('Initial request and native policy positions disagree')
    return dict(context_offset=first['context_offset'], context_length=length,
                question_length=length, turn_index=0, position_start=0)


def retained_five(row):
    """Take the first five *kept policy* tokens, never a discarded raw tail."""
    length = _integer(row.get('policy_length', row.get('length')), 'policy length', 1)
    retained = min(POLICY_TOKENS, length)
    chosen = []
    for original in row['segments']:
        if original['policy_start'] >= retained:
            break
        take = min(original['policy_length'], retained - original['policy_start'])
        if take < 1:
            raise ValueError('Invalid retained segment length')
        chosen.append(dict(original, policy_length=take,
                           context_length=original['request_length'] + take))
    bounds = [{key: segment[key] for key in
               ('turn_index', 'policy_start', 'policy_length', 'position_start')}
              for segment in chosen]
    if len(expected_policy_positions(bounds)) != retained:
        raise ValueError('The retained five-token policy prefix is incomplete')
    return chosen, bounds, retained


def _read_identities(path, expected_sha):
    _digest(expected_sha, 'question identity SHA')
    if sha(path) != expected_sha:
        raise ValueError('Canonical question identity mapping SHA differs')
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            rows = (json.loads(line) for line in handle if line.strip())
            records = list(rows)
    else:
        value = read(path)
        records = value.get('rows', value.get('records', value)) if isinstance(value, dict) else value
        if isinstance(records, dict):
            records = [dict(row, qid=qid) for qid, row in records.items()]
    result = {}
    for row in records:
        qid = str(row.get('qid', row.get('question_id')))
        if qid == 'None':
            raise ValueError('Question identity mapping lacks a stable qid')
        identity = dict(question_hash=_digest(row['question_hash'], 'question hash'),
                        normalized_question_hash=_digest(row['normalized_question_hash'], 'normalized question hash'))
        if qid in result and result[qid] != identity:
            raise ValueError('Conflicting canonical identity for one qid')
        result[qid] = identity
    return result


class Prefix5Source:
    """One immutable producer event scan with lazily opened token arrays."""
    def __init__(self, descriptor, manifest_dir, common_identities=None, verify_files=True):
        self.descriptor = dict(descriptor)
        self.source_id = str(descriptor['source_id'])
        self.event_id = str(descriptor['source_event_id'])
        if not self.source_id or not self.event_id:
            raise ValueError('Explicit source and actual producer-event identities are required')
        self.split = descriptor['split']
        if self.split not in ('train', 'test'):
            raise ValueError('Source requires an explicit train/test split')
        self.path = _path(descriptor.get('inputs_dir', descriptor.get('inputs')), manifest_dir)
        self.scan_path = self.path / 'scan-complete.json'
        self.scan_sha256 = sha(self.scan_path)
        if self.scan_sha256 != _digest(descriptor['scan_sha256'], 'source scan SHA'):
            raise ValueError('Pinned source scan SHA differs: ' + self.source_id)
        self.proof = read(self.scan_path)
        if self.proof.get('encoder_revision') != REVISION:
            raise ValueError('Wrong official encoder revision')
        if self.proof.get('representation') not in SUPPORTED_REPRESENTATIONS:
            raise ValueError('Require an authentic retained-policy5 token scan')
        # Both the typed producer capacity and all retained row lengths are
        # checked. A shorter real trajectory is allowed only under a >=5 source.
        capacities = [self.proof[key] for key in ('policy_max_tokens', 'max_policy_prefix_tokens',
                                                'max_raw_prefix_tokens') if key in self.proof]
        if not capacities:
            coverage = self.proof.get('prefix_coverage', {})
            if '5' in coverage:
                capacities = [5]
        if not capacities or max(_integer(value, 'source policy capacity', 1) for value in capacities) < 5:
            raise ValueError('Source has no producer proof of a policy5 observation boundary')
        self.arrays_path = _path(self.proof.get('source_inputs_dir', str(self.path)), self.path)
        self.index_path = self.path / 'index.jsonl'
        self.file_proofs = {}
        expected_files = dict(self.proof.get('source_files', {}))
        expected_files.update(self.proof.get('files', {}))
        for name in ('index.jsonl',) + ARRAY_NAMES:
            location = self.index_path if name == 'index.jsonl' else self.arrays_path / name
            expected = expected_files.get(name)
            if not expected:
                raise ValueError('Source scan lacks a pinned file SHA: ' + name)
            expected_sha = _digest(expected['sha256'], name)
            actual_size = location.stat().st_size
            if actual_size != expected.get('bytes'):
                raise ValueError('Source file size differs: ' + str(location))
            actual_sha = sha(location) if verify_files else None
            if verify_files and actual_sha != expected_sha:
                raise ValueError('Source file SHA differs: ' + str(location))
            self.file_proofs[name] = dict(path=str(location), sha256=expected_sha,
                                          bytes=actual_size, full_sha_verified=bool(verify_files))
        self.identities = common_identities or {}
        if descriptor.get('question_identity_path'):
            self.identities = _read_identities(
                _path(descriptor['question_identity_path'], manifest_dir),
                descriptor['question_identity_sha256'])
        self._arrays = None

    def arrays(self):
        if self._arrays is None:
            import numpy as np
            self._arrays = {name: np.load(self.arrays_path / name, mmap_mode='r', allow_pickle=False)
                            for name in ARRAY_NAMES}
            if any(array.ndim != 1 or array.dtype.kind not in 'iu'
                   for array in self._arrays.values()):
                raise ValueError('Packed token arrays must be one-dimensional integer arrays')
            if len(self._arrays[ARRAY_NAMES[1]]) != len(self._arrays[ARRAY_NAMES[2]]):
                raise ValueError('Policy token and position payload lengths differ')
        return self._arrays

    def close(self):
        if self._arrays:
            for array in self._arrays.values():
                mmap = getattr(array, '_mmap', None)
                if mmap is not None:
                    mmap.close()
        self._arrays = None

    def identity(self, row):
        qid = str(row['qid'])
        expected = self.identities.get(qid)
        qhash = _digest(row['question_hash'], 'row question hash')
        normalized = row.get('normalized_question_hash')
        if expected:
            if qhash != expected['question_hash']:
                raise ValueError('Canonical question hash differs from its source row')
            if normalized is not None and normalized != expected['normalized_question_hash']:
                raise ValueError('Row normalized identity differs from canonical mapping')
            normalized = expected['normalized_question_hash']
        if normalized is None:
            raise ValueError('Normalized question identity is required for test isolation; provide the canonical identity mapping')
        return qid, qhash, _digest(normalized, 'row normalized question hash')

    def normalize(self, original):
        if type(original.get('strict_em')) is not int or original['strict_em'] not in (0, 1):
            raise ValueError('Each trajectory requires its original binary strict EM')
        question = initial_question_segment(original)
        segments, bounds, length = retained_five(original)
        qid, qhash, normalized = self.identity(original)
        return dict(original, qid=qid, question_hash=qhash, normalized_question_hash=normalized,
                    question_length=question['question_length'], question_context_offset=question['context_offset'],
                    policy_length=length, prefix_length=length, length=length,
                    segments=segments, segment_bounds=bounds, source_id=self.source_id,
                    source_event_id=self.event_id)

    def payload_fingerprint(self, row):
        """Prove equality of retained inputs/positions/labels when an event repeats."""
        import numpy as np
        arrays = self.arrays()
        contexts, tokens, positions = (arrays[name] for name in ARRAY_NAMES)
        offset, length = _integer(row['policy_offset'], 'policy offset'), row['policy_length']
        if offset + length > len(tokens):
            raise ValueError('Retained policy range exceeds the source payload')
        actual_positions = np.asarray(positions[offset:offset + length], dtype=np.int64)
        if actual_positions.tolist() != expected_policy_positions(row['segment_bounds']):
            raise ValueError('Original policy positions differ from their turn boundaries')
        digest = hashlib.sha256(canonical(dict(qid=row['qid'], question_hash=row['question_hash'],
                                               normalized_question_hash=row['normalized_question_hash'],
                                               strict_em=row['strict_em'], segments=row['segment_bounds'])))
        digest.update(np.asarray(tokens[offset:offset + length], dtype='<i8').tobytes())
        digest.update(actual_positions.astype('<i8', copy=False).tobytes())
        for segment in row['segments']:
            lo = _integer(segment['context_offset'], 'context offset')
            req = _integer(segment['request_length'], 'request length', 1)
            take = segment['policy_length']
            size = req + take
            if segment['context_length'] != size or lo + size > len(contexts):
                raise ValueError('Actual causal context range is incomplete')
            start = offset + segment['policy_start']
            if not np.array_equal(contexts[lo + req:lo + size], tokens[start:start + take]):
                raise ValueError('Own policy token IDs differ from the actual causal context')
            digest.update(np.asarray(contexts[lo:lo + size], dtype='<i8').tobytes())
        return digest.hexdigest()

    def producer_trajectory_identity(self, row):
        # Raw bytes + original row + original trajectory identity prove the same
        # sampled event across differently packaged scans. No qid/text heuristic.
        raw_sha = row.get('source_raw_sha256')
        raw_row = row.get('source_raw_row')
        if raw_sha is not None and type(raw_row) is int and raw_row >= 0:
            return canonical(('raw', _digest(raw_sha, 'raw source SHA'), raw_row,
                              str(row['trajectory_id']))).decode()
        # Without an original-raw event identity, equality is proved only for
        # the very same pinned scan artifact. An arbitrary reused source name
        # cannot merge separately sampled observations with identical prefixes.
        return canonical(('producer_scan', self.scan_sha256, self.event_id,
                          str(row['trajectory_id']))).decode()

    def iter_rows(self):
        count = 0
        with self.index_path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                original = json.loads(line)
                count += 1
                if original['split'] == self.split:
                    yield offset, self.normalize(original)
        if count != self.proof['count']:
            raise ValueError('Source scan index count differs from its producer receipt')


class _RowIndex(Mapping):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, key):
        return self.dataset.get_rows([key])[0]

    def __iter__(self):
        return iter(self.dataset._refs)

    def __len__(self):
        return len(self.dataset._refs)


class CompositePrefix5Dataset:
    """Composite exact-event dataset; retain same-question new G5 observations."""
    def __init__(self, manifest_path, split='train', verify_files=True, max_open_sources=8):
        self.path = _path(manifest_path)
        self.manifest = read(self.path)
        self.manifest_sha = sha(self.path)
        if (self.manifest.get('schema_version') != 1 or self.manifest.get('prefix_tokens') != 5
                or self.manifest.get('encoder_revision') != REVISION):
            raise ValueError('Require the pinned official question+policy5 composite manifest')
        if split not in ('train', 'test'):
            raise ValueError('Explicit train/test dataset split required')
        self.split = split
        self.question_max_tokens = _integer(self.manifest['question_max_tokens'], 'global question maximum', 1)
        self.max_tokens = self.question_max_tokens + POLICY_TOKENS
        self.feature_dim = FEATURE_DIM
        self._refs, self.groups, self.sources = {}, [], {}
        self.ids_by_split = {'train': [], 'test': []}
        self._open_sources = OrderedDict()
        self.max_open_sources = _integer(max_open_sources, 'maximum open mmap sources', 1)
        common = {}
        if self.manifest.get('question_identity_path'):
            common = _read_identities(_path(self.manifest['question_identity_path'], self.path.parent),
                                      self.manifest['question_identity_sha256'])
        descriptors = self.manifest['sources']
        if not descriptors or not any(row['split'] == 'test' for row in descriptors):
            raise ValueError('The composite must bind its fixed test source, even for train-only reads')
        if len({str(row['source_id']) for row in descriptors}) != len(descriptors):
            raise ValueError('Composite source IDs must be unique')
        # Read fixed-test identities first. Exclusion is by both qid and original
        # normalized question text, independent of arbitrary source ordering.
        test_ids, test_hashes = set(), set()
        source_summaries, all_seen, test_seen = [], {}, set()
        accepted_counts = {'train': 0, 'test': 0}
        duplicate_counts = {'train': 0, 'test': 0}
        model_files = None
        ordered = sorted(descriptors, key=lambda item: item['split'] != 'test')
        for descriptor in ordered:
            source = Prefix5Source(descriptor, self.path.parent, common, verify_files)
            self.sources[source.source_id] = source
            current_model = source.proof['model_files_sha256']
            if model_files is None:
                model_files = current_model
            elif current_model != model_files:
                raise ValueError('Composite sources use different official encoder bytes')
            pending, count, observed_max, source_qids = defaultdict(list), 0, 0, defaultdict(int)
            for offset, row in source.iter_rows():
                count += 1
                qid, qhash, normalized = row['qid'], row['question_hash'], row['normalized_question_hash']
                source_qids[qid] += 1
                if row['question_length'] > self.question_max_tokens:
                    raise ValueError('A valid question exceeds the scanned global question maximum; truncation is forbidden')
                observed_max = max(observed_max, row['question_length'])
                if source.split == 'train' and (qid in test_ids or normalized in test_hashes):
                    raise ValueError('Training event overlaps the fixed test qid or normalized prompt')
                if source.split == 'test':
                    test_ids.add(qid)
                    test_hashes.add(normalized)
                # Validate every retained row against its own mmap, not just row
                # count or declared producer hashes. This is bounded CPU work.
                fingerprint = source.payload_fingerprint(row)
                actual_identity = source.producer_trajectory_identity(row)
                old = all_seen.get(actual_identity)
                if old is not None:
                    if old != (source.split, fingerprint):
                        raise ValueError('Repeated actual sampling event has conflicting split, input, or label')
                    duplicate_counts[source.split] += 1
                    continue
                all_seen[actual_identity] = (source.split, fingerprint)
                unique_tid = hashlib.sha256(actual_identity.encode()).hexdigest()
                if unique_tid in self._refs:
                    raise ValueError('Unexpected canonical event-trajectory identity collision')
                pending[qid].append((unique_tid, offset, row['strict_em'], qhash, normalized,
                                     actual_identity))
            expected_count = _integer(descriptor['expected_trajectories'], 'source trajectory count', 1)
            if count != expected_count:
                raise ValueError('Source selected split row count differs from the composite manifest')
            expected_groups = _integer(descriptor['expected_groups'], 'source group count', 1)
            if len(source_qids) != expected_groups or set(source_qids.values()) != {5}:
                raise ValueError('Source split is not its declared complete unique-qid G5 batch')
            # An alias source must duplicate all five siblings of a group. A
            # partially duplicated group is never silently completed elsewhere.
            for qid, siblings in sorted(pending.items()):
                if len(siblings) != 5 or len({item[3] for item in siblings}) != 1:
                    raise ValueError('Each accepted producer event must contain its own complete G5')
                siblings.sort(key=lambda item: item[5])
                event_id = hashlib.sha256(canonical([item[5] for item in siblings])).hexdigest()
                group = dict(event_id=event_id, event_group_id=event_id, source_event_id=source.event_id,
                             source_id=source.source_id, qid=qid, question_hash=siblings[0][3],
                             normalized_question_hash=siblings[0][4],
                             trajectory_ids=[item[0] for item in siblings], labels=[item[2] for item in siblings])
                if source.split == 'test':
                    if event_id in test_seen:
                        raise ValueError('Fixed test group is duplicated')
                    test_seen.add(event_id)
                accepted_counts[source.split] += 1
                if source.split == split:
                    self.groups.append(group)
                    for tid, offset, _label, _qhash, _normalized, _identity in siblings:
                        self._refs[tid] = (source.source_id, offset, event_id)
                        self.ids_by_split[split].append(tid)
            if expected_count != expected_groups * 5:
                raise ValueError('Source expected group and trajectory counts disagree')
            source_summaries.append(dict(source_id=source.source_id, source_event_id=source.event_id,
                split=source.split, inputs_dir=str(source.path), arrays_dir=str(source.arrays_path),
                scan_sha256=source.scan_sha256, input_trajectories=count, accepted_groups=len(pending),
                max_question_tokens=observed_max, files=source.file_proofs))
            source.close()
        if accepted_counts['test'] != self.manifest.get('expected_test_groups', 512):
            raise ValueError('Fixed test does not contain its complete expected G5 groups')
        for key, actual in (('expected_train_groups', accepted_counts['train']),
                            ('expected_test_groups', accepted_counts['test'])):
            if key in self.manifest and self.manifest[key] != actual:
                raise ValueError('Composite accepted unique event count differs: ' + key)
        self.ids = self.ids_by_split[split]
        self.index = _RowIndex(self)
        self.spec = dict(encoder_revision=REVISION, encoder_files_sha256=model_files,
                         question_max_tokens=self.question_max_tokens, head_max_tokens=self.max_tokens,
                         feature_layer=27, boundary='post_block', prefix_length=5,
                         question_hidden_stored=True, encoder_frozen=True, feature_role=FEATURE_ROLE)
        self.verification = dict(manifest_sha256=self.manifest_sha, split=split,
            accepted_event_groups=accepted_counts, exact_duplicate_trajectories_removed=duplicate_counts,
            fixed_test_qid_and_normalized_text_disjoint=True, source_summaries=source_summaries,
            prefix_tokens=5, prefix_window_unified_to_five=True, full_file_sha_verified=bool(verify_files),
            retained_row_payloads_verified=True, normalized_identity=NORMALIZED_IDENTITY)

    def __len__(self):
        return len(self.ids)

    def source_arrays(self, source_id):
        source = self.sources[source_id]
        self._open_sources.pop(source_id, None)
        self._open_sources[source_id] = source
        while len(self._open_sources) > self.max_open_sources:
            _key, previous = self._open_sources.popitem(last=False)
            previous.close()
        return source.arrays()

    def get_rows(self, trajectory_ids):
        result = []
        handles = {}
        try:
            for tid in trajectory_ids:
                source_id, offset, event_id = self._refs[tid]
                source = self.sources[source_id]
                if source_id not in handles:
                    handles[source_id] = source.index_path.open('rb')
                handle = handles[source_id]
                handle.seek(offset)
                row = source.normalize(json.loads(handle.readline()))
                row['original_trajectory_id'] = str(row['trajectory_id'])
                row['trajectory_id'] = tid
                row['event_id'] = row['event_group_id'] = event_id
                result.append(row)
        finally:
            for handle in handles.values():
                handle.close()
        return result

    def close(self):
        for source in self.sources.values():
            source.close()
        self._open_sources.clear()

