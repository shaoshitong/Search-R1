"""CPU-only conversion of pinned H100/A800 sampled events to question+policy5.

No rollout, model forward, tokenizer round-trip, or EM scoring occurs here. The
reference gzip archive is verified while streaming across its ordered parts;
only canonical questions, original token scans and compact prefix rows are
extracted. Full responses, model/optimizer weights and feature tensors are not
extracted. All outputs belong on the execution host's remote storage.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import unicodedata

from .data import (ARRAY_NAMES, CompositePrefix5Dataset, NORMALIZED_IDENTITY,
                   POLICY_TOKENS, REVISION, canonical, expected_policy_positions,
                   initial_question_segment, read, retained_five, sha)


CANONICAL_SHA = '5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1'
REPRESENTATION = 'initial_request_and_retained_policy5_layer27_postblock_v1'
COMPACT_NAMES = ('packed_input_ids.npy', 'packed_position_ids.npy', 'packed_policy_mask.npy',
                 'offsets.npy', 'lengths.npy', 'index.jsonl')


def once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def _verified(path, proof):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size != proof['bytes'] or sha(path) != proof['sha256']:
        raise ValueError('Pinned source file size/SHA differs: ' + str(path))
    return path


class _PartStream(io.RawIOBase):
    """Sequential parts with per-part and concatenated SHA during consumption."""
    def __init__(self, parts, expected_sha):
        self.parts, self.expected_sha = parts, expected_sha
        self.index, self.handle, self.part_hash = 0, None, None
        self.whole_hash, self.part_bytes, self.total = hashlib.sha256(), 0, 0

    def readable(self):
        return True

    def read(self, size=-1):
        # tarfile/gzip use bounded reads. Never implement an unbounded payload
        # materialization even if a caller passes read(-1).
        size = 1024 * 1024 if size < 0 else size
        pieces, left = [], size
        while left and self.index < len(self.parts):
            proof = self.parts[self.index]
            if self.handle is None:
                path = Path(proof['path']).resolve(strict=True)
                if path.stat().st_size != proof['bytes']:
                    raise ValueError('Archive part size differs')
                self.handle, self.part_hash, self.part_bytes = path.open('rb'), hashlib.sha256(), 0
            block = self.handle.read(left)
            if block:
                self.part_hash.update(block)
                self.whole_hash.update(block)
                self.part_bytes += len(block)
                self.total += len(block)
                pieces.append(block)
                left -= len(block)
            else:
                self.handle.close()
                self.handle = None
                if self.part_bytes != proof['bytes'] or self.part_hash.hexdigest() != proof['sha256']:
                    raise ValueError('Archive part content SHA differs')
                self.index += 1
        return b''.join(pieces)

    def finish(self):
        while self.read(1024 * 1024):
            pass
        if self.index != len(self.parts) or self.whole_hash.hexdigest() != self.expected_sha:
            raise ValueError('Concatenated reference archive SHA differs')


def extract_reference(descriptor, output):
    """Safely extract needed members only, checking every archived file hash."""
    output = Path(output).resolve()
    if output.exists():
        receipt = read(output / 'EXTRACTION-COMPLETE.json')
        if receipt['archive_sha256'] != descriptor['concatenated_sha256']:
            raise ValueError('Existing extraction belongs to another reference archive')
        for item in receipt['selected_files']:
            _verified(output / item['path'], item)
        return output
    output.mkdir(parents=True)
    stream = _PartStream(descriptor['parts'], descriptor['concatenated_sha256'])
    actual, selected, manifest = {}, [], None
    def wanted(name):
        return (name == 'canonical/train.parquet' or
                name.startswith('epoch1-and-fixed-test/inputs/') or
                (name.startswith('epochs/') and ('/prefix20/' in name or name.endswith('/rollout-complete.json'))))
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                raise ValueError('Only safe regular reference archive members are allowed')
            source = archive.extractfile(member)
            if source is None:
                raise ValueError('Unreadable regular archive member')
            digest, count = hashlib.sha256(), 0
            destination = output / relative
            target = None
            if wanted(member.name) or member.name == 'MANIFEST.json':
                destination.parent.mkdir(parents=True, exist_ok=True)
                target = destination.open('xb')
            try:
                for block in iter(lambda: source.read(8 * 1024**2), b''):
                    digest.update(block)
                    count += len(block)
                    if target:
                        target.write(block)
                if target:
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                if target:
                    target.close()
                source.close()
            if count != member.size or member.name in actual:
                raise ValueError('Archive member is truncated or duplicated')
            actual[member.name] = dict(path=member.name, bytes=count, sha256=digest.hexdigest())
            if wanted(member.name):
                selected.append(actual[member.name])
            if member.name == 'MANIFEST.json':
                manifest = read(destination)
    stream.finish()
    if manifest is None or manifest.get('snapshot_epoch') != 68 or manifest.get('encoder_revision') != REVISION:
        raise ValueError('Require the actual official reference epoch68 archive manifest')
    expected = {item['path']: item for item in manifest['files']}
    if set(actual) != set(expected) | {'MANIFEST.json'}:
        raise ValueError('Archive member inventory differs from its internal manifest')
    for name, proof in expected.items():
        if any(actual[name][key] != proof[key] for key in ('bytes', 'sha256')):
            raise ValueError('Archive member content differs from its internal manifest')
    once(output / 'EXTRACTION-COMPLETE.json', dict(archive_sha256=descriptor['concatenated_sha256'],
         parts=descriptor['parts'], selected_files=selected, internal_manifest_sha256=sha(output / 'MANIFEST.json'),
         all_members_verified=len(expected), full_responses_extracted=False, weights_extracted=False))
    return output


def extract_prepared_reference(descriptor, output):
    """Verify a derived prefix5 package without fetching the original8.4GB."""
    if 'manifest_path' in descriptor:
        manifest_path = Path(descriptor['manifest_path']).resolve(strict=True)
        if sha(manifest_path) != descriptor['manifest_sha256']:
            raise ValueError('Pinned slim transfer manifest SHA differs')
        transfer = read(manifest_path)
    else:
        transfer = dict(descriptor)
    if (transfer.get('role') != 'PREPARED_REFERENCE_PREFIX5_ARCHIVE' or transfer.get('prefix_tokens') != 5
            or transfer.get('encoder_revision') != REVISION or transfer.get('train_trajectories') != 1740800
            or transfer.get('test_trajectories') != 2560 or transfer.get('sources') != 68
            or transfer.get('canonical_train_sha256') != CANONICAL_SHA
            or transfer.get('canonical_questions', 169028) != 169028):
        raise ValueError('Wrong prepared reference dataset identity or count')
    parts = [dict(item) for item in transfer['parts']]
    if descriptor.get('parts_dir'):
        for item in parts:
            item['path'] = str(Path(descriptor['parts_dir']) / Path(item['path']).name)
    output = Path(output).resolve()
    if output.exists():
        complete = read(output / 'EXTRACTION-COMPLETE.json')
        if complete['archive_sha256'] != transfer['concatenated_sha256']:
            raise ValueError('Existing prepared extraction belongs to another archive')
        for item in transfer['files']:
            _verified(output / item['path'], item)
        return output
    output.mkdir(parents=True)
    stream = _PartStream(parts, transfer['concatenated_sha256'])
    expected = {item['path']: item for item in transfer['files']}
    actual = {}
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile() or member.name in actual:
                raise ValueError('Unsafe or duplicate slim archive member')
            if member.name != 'MANIFEST.json' and member.name not in expected:
                raise ValueError('Slim archive contains an unbound member')
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            digest, count = hashlib.sha256(), 0
            with destination.open('xb') as target:
                for block in iter(lambda: source.read(8 * 1024**2), b''):
                    digest.update(block)
                    count += len(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            source.close()
            actual[member.name] = dict(path=member.name, bytes=count, sha256=digest.hexdigest())
            if count != member.size:
                raise ValueError('Truncated slim reference member')
    stream.finish()
    if set(actual) != set(expected) | {'MANIFEST.json'}:
        raise ValueError('Slim reference member inventory differs')
    for name, proof in expected.items():
        if any(actual[name][key] != proof[key] for key in ('bytes', 'sha256')):
            raise ValueError('Slim reference member actual content SHA differs')
    if actual['MANIFEST.json']['sha256'] != transfer['internal_manifest_sha256']:
        raise ValueError('Slim internal manifest differs from the pinned transfer receipt')
    reference = read(output / 'reference-manifest.json')
    if (reference.get('source_archive_sha256') != transfer['source_archive_sha256']
            or sha(output / 'reference-manifest.json') != transfer['reference_manifest_sha256']
            or reference.get('train_trajectories') != transfer['train_trajectories']
            or reference.get('test_trajectories') != transfer['test_trajectories']
            or reference.get('expected_train_groups') != transfer['train_groups']
            or reference.get('expected_test_groups') != transfer['test_groups']
            or sum(item['split'] == 'train' for item in reference['sources']) != 68):
        raise ValueError('Derived reference lost its original archive identity')
    once(output / 'EXTRACTION-COMPLETE.json', dict(archive_sha256=transfer['concatenated_sha256'],
         source_archive_sha256=transfer['source_archive_sha256'], parts=parts, files_verified=len(expected),
         train_trajectories=1740800, test_trajectories=2560, no_original_large_archive_transfer=True))
    return output


def adopt_prepared_reference(reference, output, identities_path, cache_lookup=None, acceptance=None):
    """Relocate metadata into new derived scans; token array bytes stay exact."""
    reference, output = Path(reference), Path(output)
    manifest = read(reference / 'reference-manifest.json')
    identity_source = reference / manifest['question_identity_relative_path']
    if sha(identity_source) != manifest['question_identity_sha256']:
        raise ValueError('Prepared canonical question identity mapping changed')
    os.link(identity_source, identities_path)
    from .data import _read_identities
    identities = _read_identities(Path(identities_path), manifest['question_identity_sha256'])
    if len(identities) != manifest.get('canonical_questions', 169028):
        raise ValueError('Prepared canonical identity coverage differs')
    sources = []
    for descriptor in manifest['sources']:
        relative = PurePosixPath(descriptor['relative_inputs_dir'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe prepared source-relative path')
        original = reference / relative
        if sha(original / 'scan-complete.json') != descriptor['scan_sha256']:
            raise ValueError('Prepared original source scan SHA differs')
        proof = read(original / 'scan-complete.json')
        target = output / 'sources' / descriptor['source_id']
        target.mkdir(parents=True, exist_ok=False)
        for name in ARRAY_NAMES:
            _verified(original / name, proof['files'][name])
            os.link(original / name, target / name)
        count = 0
        with (original / 'index.jsonl').open() as source, (target / 'index.jsonl').open('xb') as destination:
            for line in source:
                row = json.loads(line)
                if cache_lookup and descriptor['source_id'] in ('h100-epoch-0001', 'fixed-test'):
                    link = cache_lookup.get(str(row['trajectory_id']))
                    if link:
                        expected = tuple(link['_original_identity'])
                        actual = (row.get('source_raw_sha256'), row.get('source_raw_row'),
                                  row['qid'], row['question_hash'], row['strict_em'])
                        if actual != expected:
                            raise ValueError('Prepared reference and old A800 cache event identity differ')
                        row.update({key: value for key, value in link.items() if key != '_original_identity'})
                destination.write(canonical(row) + b'\n')
                count += 1
            destination.flush()
            os.fsync(destination.fileno())
        if count != descriptor['expected_trajectories']:
            raise ValueError('Relocated source event count differs')
        proof = dict(proof, files={name: dict(path=str(target / name), bytes=(target / name).stat().st_size,
                                             sha256=sha(target / name)) for name in ARRAY_NAMES + ('index.jsonl',)},
                     source_inputs_dir=str(target), prepared_reference_source_scan_sha256=descriptor['scan_sha256'],
                     prepared_reference_source_archive_sha256=manifest['source_archive_sha256'],
                     relocated_from_original_directory=descriptor['inputs_dir'])
        if acceptance and descriptor['source_id'] in ('h100-epoch-0001', 'fixed-test'):
            proof.update(source_cache_acceptance_path=acceptance['path'], source_cache_acceptance_sha256=acceptance['sha256'])
        once(target / 'scan-complete.json', proof)
        sources.append(dict(descriptor, inputs_dir=str(target), scan_sha256=sha(target / 'scan-complete.json')))
    if (sum(item['expected_trajectories'] for item in sources if item['split'] == 'train') != manifest['train_trajectories']
            or sum(item['expected_trajectories'] for item in sources if item['split'] == 'test') != manifest['test_trajectories']):
        raise ValueError('Prepared reference does not preserve its bound train/test events')
    return identities, sources, manifest['model_files_sha256']


def canonical_identities(path, expected_sha, output):
    import pyarrow.parquet as pq
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError('Canonical training parquet SHA differs')
    identities = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=['prompt', 'data_source', 'extra_info']):
        for original in batch.to_pylist():
            question = '\n'.join(message['content'] for message in original['prompt'])
            normalized = ' '.join(unicodedata.normalize('NFKC', question).split())
            qhash = hashlib.sha256(question.encode()).hexdigest()
            nhash = hashlib.sha256(normalized.encode()).hexdigest()
            extra = original.get('extra_info') or {}
            upstream = extra.get('index')
            if upstream is None:
                upstream = nhash
            qid = json.dumps([str(original['data_source']), str(upstream)])
            if qid in identities:
                raise ValueError('Canonical stable question identity is duplicated')
            identities[qid] = dict(qid=qid, question_hash=qhash, normalized_question_hash=nhash)
    if len(identities) != 169028:
        raise ValueError('Canonical training corpus row count differs')
    with Path(output).open('xb') as handle:
        for row in identities.values():
            handle.write(canonical(row) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    return identities


class _Writer:
    """Stream integer arrays to disk; no epoch-sized Python token list."""
    def __init__(self, output, source_id, event_id, split, model_files, identities, provenance,
                 cache_acceptance=None):
        self.output, self.source_id, self.event_id, self.split = Path(output), source_id, event_id, split
        self.output.mkdir(parents=True, exist_ok=False)
        self.streams = {name: (self.output / (name + '.bin')).open('xb') for name in ARRAY_NAMES}
        self.index = (self.output / 'index.jsonl').open('xb')
        self.model_files, self.identities, self.provenance = model_files, identities, provenance
        self.cache_acceptance = cache_acceptance
        self.count = self.context_offset = self.policy_offset = self.qmax = 0
        self.groups, self.tids = Counter(), set()

    def add(self, original, segments, tokens, positions, cache_link=None):
        import numpy as np
        qid, tid = str(original['qid']), str(original['trajectory_id'])
        if tid in self.tids or type(original['strict_em']) is not int or original['strict_em'] not in (0, 1):
            raise ValueError('Duplicate event trajectory or invalid original EM')
        identity = self.identities[qid]
        if original['question_hash'] != identity['question_hash']:
            raise ValueError('Original question hash differs from canonical corpus')
        if not 1 <= len(tokens) <= 5 or len(positions) != len(tokens):
            raise ValueError('Require the actual same-trajectory first five kept tokens')
        bounds = [{key: segment[key] for key in ('turn_index', 'policy_start', 'policy_length', 'position_start')}
                  for segment in segments]
        if expected_policy_positions(bounds) != list(positions):
            raise ValueError('Retained original policy positions disagree with native turn metadata')
        records = []
        for segment in segments:
            segment = dict(segment)
            context = segment.pop('context_tokens')
            request, length, start = segment['request_length'], segment['policy_length'], segment['policy_start']
            if (len(context) != request + length or segment['position_start'] != request
                    or list(context[request:]) != list(tokens[start:start + length])):
                raise ValueError('Retained causal context and own policy IDs differ')
            segment.update(context_offset=self.context_offset, context_length=len(context))
            self.streams[ARRAY_NAMES[0]].write(np.asarray(context, dtype='<i8').tobytes())
            self.context_offset += len(context)
            records.append(segment)
        row = {key: value for key, value in original.items()
               if key not in ('segments', 'prefix_segments', 'policy_token_ids', 'policy_position_ids')}
        original_hashes = {key: original[key] for key in
                           ('policy_prefix_sha256', 'position_ids_sha256', 'input_ids_sha256',
                            'policy_mask_sha256', 'prefix_sha256') if key in original}
        for key in original_hashes:
            row.pop(key, None)
        row.update(row=self.count, split=self.split, qid=qid, trajectory_id=tid,
                   normalized_question_hash=identity['normalized_question_hash'],
                   length=len(tokens), policy_length=len(tokens), prefix_length=len(tokens),
                   policy_offset=self.policy_offset, segments=records, segment_bounds=bounds,
                   source_event_id=self.event_id, original_prefix_metadata_hashes=original_hashes,
                   policy_prefix_sha256=hashlib.sha256(canonical(list(tokens))).hexdigest(),
                   policy_position_ids_sha256=hashlib.sha256(canonical(list(positions))).hexdigest())
        question = initial_question_segment(row)
        row.update(question_length=question['question_length'], question_context_offset=question['context_offset'])
        if cache_link:
            row.update(cache_link)
        self.index.write(canonical(row) + b'\n')
        self.streams[ARRAY_NAMES[1]].write(np.asarray(tokens, dtype='<i8').tobytes())
        self.streams[ARRAY_NAMES[2]].write(np.asarray(positions, dtype='<i8').tobytes())
        self.policy_offset += len(tokens)
        self.count += 1
        self.groups[qid] += 1
        self.tids.add(tid)
        self.qmax = max(self.qmax, question['question_length'])

    def finish(self, expected_count):
        import numpy as np
        for handle in list(self.streams.values()) + [self.index]:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if self.count != expected_count or set(self.groups.values()) != {5}:
            raise ValueError('Converted source is not its complete own-G5 event batch')
        for name in ARRAY_NAMES:
            binary = self.output / (name + '.bin')
            values = np.memmap(binary, mode='r', dtype='<i8')
            with (self.output / name).open('xb') as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            del values
            binary.unlink()
        files = {name: dict(path=str(self.output / name), bytes=(self.output / name).stat().st_size,
                            sha256=sha(self.output / name)) for name in ARRAY_NAMES + ('index.jsonl',)}
        proof = dict(schema_version=1, role='unified_authentic_question_policy5_scan', representation=REPRESENTATION,
                     encoder_revision=REVISION, model_files_sha256=self.model_files, count=self.count,
                     split_counts={kind: self.count if kind == self.split else 0 for kind in ('train', 'test')},
                     policy_max_tokens=5, question_max_observed=self.qmax, files=files,
                     source_event_id=self.event_id, provenance=self.provenance,
                     normalized_identity=NORMALIZED_IDENTITY, new_rollouts=0, new_em_scoring=0)
        if self.cache_acceptance:
            proof.update(source_cache_acceptance_path=self.cache_acceptance['path'],
                         source_cache_acceptance_sha256=self.cache_acceptance['sha256'])
        once(self.output / 'scan-complete.json', proof)
        return dict(source_id=self.source_id, source_event_id=self.event_id, split=self.split,
                    inputs_dir=str(self.output), scan_sha256=sha(self.output / 'scan-complete.json'),
                    expected_groups=len(self.groups), expected_trajectories=self.count,
                    question_max_observed=self.qmax)


def _array_files(directory, proof, names):
    import numpy as np
    arrays = {}
    for name in names:
        path = _verified(Path(directory) / name, proof['files'][name])
        if name.endswith('.npy'):
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    return arrays


def _scan_source(directory, output, split, source_id, event_id, identities, cache_lookup=None, acceptance=None):
    directory = Path(directory)
    proof = read(directory / 'scan-complete.json')
    if proof['encoder_revision'] != REVISION:
        raise ValueError('Reference scan encoder revision differs')
    arrays = _array_files(directory, proof, ARRAY_NAMES + ('index.jsonl',))
    writer = _Writer(output, source_id, event_id, split, proof['model_files_sha256'], identities,
                     dict(source_scan_path=str(directory), source_scan_sha256=sha(directory / 'scan-complete.json')),
                     cache_acceptance=acceptance)
    count = 0
    with (directory / 'index.jsonl').open() as handle:
        for line in handle:
            original = json.loads(line)
            if original['split'] != split:
                continue
            segments, _bounds, length = retained_five(original)
            tokens = arrays[ARRAY_NAMES[1]][original['policy_offset']:original['policy_offset'] + length].tolist()
            positions = arrays[ARRAY_NAMES[2]][original['policy_offset']:original['policy_offset'] + length].tolist()
            for segment in segments:
                lo, size = segment['context_offset'], segment['context_length']
                segment['context_tokens'] = arrays[ARRAY_NAMES[0]][lo:lo + size].tolist()
            link = cache_lookup.get(str(original['trajectory_id'])) if cache_lookup else None
            if link:
                expected = link['_original_identity']
                actual = (original.get('source_raw_sha256'), original.get('source_raw_row'),
                          original['qid'], original['question_hash'], original['strict_em'])
                if actual != tuple(expected):
                    raise ValueError('Original teacher/test event differs from the proposed feature-cache event')
                link = {key: value for key, value in link.items() if key != '_original_identity'}
            writer.add(original, segments, tokens, positions, link)
            count += 1
    if count != (25600 if split == 'train' else 2560):
        raise ValueError('Original teacher/fixed test reference scan coverage differs')
    return writer.finish(count)


def _reuse_cache_links(descriptor, acceptance=None):
    """Reference accepted original28160 features without reading feature tensors."""
    import numpy as np
    root = Path(descriptor['path'])
    if sha(root / 'manifest.json') != descriptor['manifest_sha256']:
        raise ValueError('Original reused prefix feature manifest SHA differs')
    proof = read(root / 'manifest.json')
    arrays = _array_files(root, proof, COMPACT_NAMES)
    result = {}
    with (root / 'index.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            offset, length, prompt = int(row['offset']), int(row['length']), int(row['actual_prompt_tokens'])
            mask = arrays['packed_policy_mask.npy'][offset:offset + length]
            selected = np.flatnonzero(mask)
            if not 1 <= len(selected) <= 5 or selected[0] != prompt or selected[-1] != length - 1:
                raise ValueError('Original cached policy5 feature boundary differs')
            link = dict(source_cache=str(root), source_cache_manifest_sha256=descriptor['manifest_sha256'],
                        source_cache_row=row['row'], source_cache_offset=offset, source_cache_length=length,
                        selected_feature_indices=list(range(prompt)) + selected.tolist(),
                        _original_identity=(row['source_raw_sha256'], row['source_raw_row'], row['qid'],
                                            row['question_hash'], row['strict_em']))
            if acceptance:
                link.update(source_cache_accepted=True, source_cache_acceptance_sha256=acceptance['sha256'])
            tid = str(row['trajectory_id'])
            if tid in result:
                raise ValueError('Original cache repeats trajectory event IDs')
            result[tid] = link
    if len(result) != 28160 or proof['split_counts'] != dict(train=25600, test=2560):
        raise ValueError('Original feature-cache teacher/test coverage differs')
    return result


def _compact_source(directory, output, split, source_id, event_id, identities, model_files,
                    expected_count, source_cache=None, acceptance=None):
    import numpy as np
    directory = Path(directory)
    proof = read(directory / 'manifest.json')
    arrays = _array_files(directory, proof, COMPACT_NAMES)
    writer = _Writer(output, source_id, event_id, split, model_files, identities,
                     dict(source_compact_path=str(directory), source_manifest_sha256=sha(directory / 'manifest.json')),
                     cache_acceptance=acceptance)
    with (directory / 'index.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['split'] != split:
                continue
            offset, length = int(row['offset']), int(row['length'])
            if row['row'] >= len(arrays['lengths.npy']) or int(arrays['offsets.npy'][row['row']]) != offset:
                raise ValueError('Compact row offset differs from its pinned array')
            ids = arrays['packed_input_ids.npy'][offset:offset + length].astype('<i8')
            pos = arrays['packed_position_ids.npy'][offset:offset + length].astype('<i8')
            mask = arrays['packed_policy_mask.npy'][offset:offset + length].astype(bool)
            for values, key in ((ids, 'input_ids_sha256'), (pos, 'position_ids_sha256'), (mask, 'policy_mask_sha256')):
                if hashlib.sha256(values.tobytes()).hexdigest() != row[key]:
                    raise ValueError('Compact row original IDs/positions/mask SHA differs')
            qlen, actual = int(row['actual_prompt_tokens']), np.flatnonzero(mask)
            if (not 1 <= len(actual) <= 5 or len(actual) != row['policy_prefix_tokens']
                    or mask[:qlen].any() or actual[0] != qlen or actual[-1] != length - 1
                    or pos.tolist() != list(range(length))):
                raise ValueError('Compact prefix cannot be represented as the authentic native request positions')
            # A separate causal segment for each contiguous policy island keeps
            # all preceding observations in context while selecting no observed
            # hidden positions for the head. Native position=real request index.
            segments, start = [], 0
            for ordinal in range(1, len(actual) + 1):
                if ordinal < len(actual) and actual[ordinal] == actual[ordinal - 1] + 1:
                    continue
                request, end = int(actual[start]), int(actual[ordinal - 1]) + 1
                segments.append(dict(turn_index=len(segments), policy_start=start, policy_length=ordinal - start,
                    request_length=request, position_start=request, context_tokens=ids[:end].tolist()))
                start = ordinal
            link = None
            if source_cache:
                link = dict(source_cache=str(source_cache['path']), source_cache_manifest_sha256=source_cache['manifest_sha256'],
                            source_cache_row=row['row'], source_cache_offset=offset, source_cache_length=length,
                            selected_feature_indices=list(range(qlen)) + actual.tolist())
                if acceptance:
                    link.update(source_cache_accepted=True, source_cache_acceptance_sha256=acceptance['sha256'])
            writer.add(row, segments, ids[mask].tolist(), pos[mask].tolist(), link)
    return writer.finish(expected_count)


def _rollout_source(archive_root, epoch, output, identities):
    root = Path(archive_root) / 'epochs' / ('epoch-%04d' % epoch)
    complete = read(root / 'rollout-complete.json')
    if complete['model_revision'] != REVISION or complete['questions'] != 5120 or complete['trajectories'] != 25600:
        raise ValueError('Reference compact rollout identity/count differs')
    writer = _Writer(output, 'h100-epoch-%04d' % epoch, 'h100-original-epoch-%04d' % epoch,
                     'train', complete['model_files_sha256'], identities,
                     dict(archive_root=str(archive_root), original_complete_sha256=sha(root / 'rollout-complete.json')))
    for rank, pinned in enumerate(complete['shards']):
        # The archived complete receipt intentionally retains its historical
        # H100 paths. Resolve only by archive rank/member, preserving old SHA.
        path = root / 'prefix20' / ('rank-%d' % rank + Path(pinned['path']).suffix)
        _verified(path, pinned)
        with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as handle:
            count = 0
            for line in handle:
                original = json.loads(line)
                if original.get('epoch', epoch) != epoch:
                    raise ValueError('Archived row belongs to another original source epoch')
                full_segments = original['prefix_segments']
                total = min(5, len(original['policy_token_ids']))
                segments = []
                for item in full_segments:
                    if item['policy_start'] >= total:
                        break
                    take = min(item['policy_length'], total - item['policy_start'])
                    segments.append(dict(item, policy_length=take,
                                         context_tokens=item['context_tokens'][:item['request_length'] + take]))
                original = dict(original, source_rollout_sha256=pinned['sha256'], source_rollout_row=count)
                writer.add(original, segments, original['policy_token_ids'][:total],
                           original['policy_position_ids'][:total])
                count += 1
            if count != pinned['count']:
                raise ValueError('Archived original prefix shard count differs')
    return writer.finish(25600)


def build(config_path, output):
    config_path, output = Path(config_path).resolve(strict=True), Path(output).resolve()
    config = read(config_path)
    if config.get('schema_version') != 1 or config.get('prefix_tokens') != 5:
        raise ValueError('Require the explicit newly authorized prefix5 source plan')
    output.mkdir(parents=True, exist_ok=False)
    identities_path = output / 'question-identities.jsonl'
    acceptance = config.get('cache_acceptance')
    if acceptance and sha(acceptance['path']) != acceptance['sha256']:
        raise ValueError('Actual source-cache acceptance receipt SHA differs')
    reuse_lookup = _reuse_cache_links(config['reuse_cache'], acceptance) if config.get('reuse_cache') else None
    if config.get('prepared_reference_archive'):
        if config.get('reference_archive'):
            raise ValueError('Choose one prepared slim reference or the original archive, never both')
        reference = extract_prepared_reference(config['prepared_reference_archive'], output.parent / 'reference5')
        identities, sources, model_files = adopt_prepared_reference(reference, output, identities_path,
                                                                    reuse_lookup, acceptance)
    else:
        reference = extract_reference(config['reference_archive'], output.parent / 'reference')
        canonical_path = Path(config.get('canonical_train', {}).get('path', reference / 'canonical/train.parquet'))
        identities = canonical_identities(canonical_path, config.get('canonical_train', {}).get('sha256', CANONICAL_SHA), identities_path)
        initial = reference / 'epoch1-and-fixed-test/inputs'
        model_files = read(initial / 'scan-complete.json')['model_files_sha256']
        sources = [_scan_source(initial, output / 'sources/h100-epoch-0001', 'train', 'h100-epoch-0001',
                                'official-original-teacher5120-G5', identities, reuse_lookup, acceptance)]
        sources.append(_scan_source(initial, output / 'sources/fixed-test', 'test', 'fixed-test',
                                    'official-fixed-random512-G5', identities, reuse_lookup, acceptance))
        for epoch in range(2, 69):
            sources.append(_rollout_source(reference, epoch, output / ('sources/h100-epoch-%04d' % epoch), identities))
            print(json.dumps(dict(stage='REFERENCE_SOURCE_CONVERTED', source_epoch=epoch, new_rollouts=0)), flush=True)
    a800 = config['a800_dataset']
    a800_root = Path(a800['dataset_dir']).resolve(strict=True)
    if sha(a800_root / 'compact-manifest.json') != a800['compact_manifest_sha256']:
        raise ValueError('Actual A800 compact dataset manifest SHA differs')
    compact = read(a800_root / 'compact-manifest.json')
    if (compact['train_questions'] != 102400 or compact['train_trajectories'] != 512000
            or compact['spec']['encoder_revision'] != REVISION
            or compact['spec']['encoder_files_sha256'] != model_files):
        raise ValueError('A800 dataset count or official frozen model identity differs')
    feature_caches = dict(config.get('feature_caches', {}))
    composite_features = a800.get('feature_composite_manifest', config.get('feature_composite_manifest'))
    if composite_features:
        path = Path(composite_features['path'])
        if sha(path) != composite_features['sha256']:
            raise ValueError('Accepted A800 composite feature manifest SHA differs')
        features = read(path)
        if (features['stage'] != 'PREFIX_SCALE20_COMPOSITE_FEATURES_COMPLETE'
                or features['compact_manifest_sha256'] != a800['compact_manifest_sha256']
                or features['train_trajectories'] != 512000):
            raise ValueError('Composite feature producer does not bind this compact dataset')
        for item in features['shards']:
            feature_caches[str(item['ordinal'])] = dict(path=item['path'], manifest_sha256=item['manifest_sha256'])
    shards = compact['shards']
    if len(shards) != 190 or {item['ordinal'] for item in shards} != set(range(10, 200)):
        raise ValueError('A800 must contribute exactly its190 newly sampled chunks')
    for item in sorted(shards, key=lambda value: value['ordinal']):
        directory = Path(item['path'])
        if not directory.is_absolute():
            directory = a800_root / directory
        if sha(directory / 'manifest.json') != item['manifest_sha256']:
            raise ValueError('A800 compact shard manifest SHA differs')
        cache = feature_caches.get(str(item['ordinal']))
        if cache and sha(Path(cache['path']) / 'manifest.json') != cache['manifest_sha256']:
            raise ValueError('Feature-cache producer manifest differs')
        sources.append(_compact_source(directory, output / ('sources/a800-chunk-%04d' % item['ordinal']),
                        'train', 'a800-chunk-%04d' % item['ordinal'],
                        'a800-scale20-chunk-%04d' % item['ordinal'], identities, model_files, 2560,
                        source_cache=cache, acceptance=acceptance))
    manifest = dict(schema_version=1, role='unified_question_policy5_complete_event_dataset', prefix_tokens=5,
        encoder_revision=REVISION, model_files_sha256=model_files, normalized_identity=NORMALIZED_IDENTITY,
        question_identity_path=str(identities_path), question_identity_sha256=sha(identities_path),
        question_max_tokens=max(item['question_max_observed'] for item in sources),
        expected_train_groups=68 * 5120 + 97280, expected_test_groups=512, sources=sources,
        source_plan_path=str(config_path), source_plan_sha256=sha(config_path),
        identical_original_teacher_in_both_archives_counted_once=True,
        original_g5_sampling_events_preserved=True, new_rollouts=0, new_em_scoring=0,
        total_trajectory_events=(68 * 5120 + 97280 + 512) * 5)
    once(output / 'composite-manifest.json', manifest)
    # No tensors/models: this final reader proves complete own-G5 events, source
    # payloads and test isolation before a source manifest becomes executable.
    dataset = CompositePrefix5Dataset(output / 'composite-manifest.json', split='train')
    proof = dataset.verification
    dataset.close()
    once(output / 'PREPARE-COMPLETE.json', dict(stage='PINNED_PREFIX5_COMPOSITE_PREPARED',
        composite_manifest_sha256=sha(output / 'composite-manifest.json'), verification=proof,
        train_groups=manifest['expected_train_groups'], train_trajectories=manifest['expected_train_groups'] * 5,
        test_groups=512, test_trajectories=2560, new_rollouts=0, new_em_scoring=0, gpu_used=False,
        finished_unix=time.time()))
    return dict(output=str(output), manifest=str(output / 'composite-manifest.json'),
                manifest_sha256=sha(output / 'composite-manifest.json'), train_groups=manifest['expected_train_groups'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
