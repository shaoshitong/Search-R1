"""Derive a small prefix5 reference archive locally on its H100 source host.

Reads only pinned original CPU token/label assets. No model, GPU, rollout,
retrieval, scoring, or training is performed. The resulting archive contains
authentic question+policy5 token scans, not full trajectories or model weights.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

from .data import REVISION, canonical, read, sha
from .prepare import (ARRAY_NAMES, CANONICAL_SHA, _rollout_source, _scan_source,
                      canonical_identities, extract_reference, once)


PART_BYTES = 128 * 1024**2


def progress(output, stage, **values):
    result = dict(stage=stage, checked_unix=time.time(), **values)
    temporary = Path(output) / 'PROGRESS.json.partial'
    with temporary.open('wb') as handle:
        handle.write(canonical(result) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, Path(output) / 'PROGRESS.json')
    print(json.dumps(result, allow_nan=False), flush=True)


class _Parts:
    """Bounded part writer; every sealed part has an actual content hash."""
    def __init__(self, output, part_bytes=PART_BYTES):
        self.output, self.limit = Path(output), int(part_bytes)
        self.parts, self.handle, self.count, self.total = [], None, 0, 0
        self.part_hash, self.whole_hash = None, hashlib.sha256()

    def writable(self):
        return True

    def flush(self):
        if self.handle:
            self.handle.flush()

    def write(self, value):
        view = memoryview(value)
        while view:
            if self.handle is None:
                self.path = self.output / ('reference5.tar.gz.part%03d' % len(self.parts))
                self.handle = self.path.with_name(self.path.name + '.partial').open('xb')
                self.count, self.part_hash = 0, hashlib.sha256()
            size = min(len(view), self.limit - self.count)
            chunk = view[:size]
            self.handle.write(chunk)
            self.part_hash.update(chunk)
            self.whole_hash.update(chunk)
            self.count += size
            self.total += size
            view = view[size:]
            if self.count == self.limit:
                self.finish_part()
        return len(value)

    def finish_part(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        os.replace(self.path.with_name(self.path.name + '.partial'), self.path)
        digest = self.part_hash.hexdigest()
        if sha(self.path) != digest:
            raise ValueError('New slim archive part actual readback SHA differs')
        self.parts.append(dict(path=str(self.path), name=self.path.name, bytes=self.count,
                               sha256=digest, actual_readback_sha256=digest))
        progress(self.output, 'SLIM_ARCHIVE_PART_SEALED', completed_parts=len(self.parts), compressed_bytes=self.total)

    def close(self):
        self.finish_part()


def package_tokens(tokens_root, output, source_archive_sha256, *, part_bytes=PART_BYTES):
    tokens_root, output = Path(tokens_root), Path(output)
    reference = read(tokens_root / 'reference-manifest.json')
    if (sum(item['expected_trajectories'] for item in reference['sources'] if item['split'] == 'train') != reference['train_trajectories']
            or sum(item['expected_trajectories'] for item in reference['sources'] if item['split'] == 'test') != reference['test_trajectories']):
        raise ValueError('Reference package source counts differ from its bound event inventory')
    files = []
    for path in sorted(tokens_root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(tokens_root).as_posix()
        allowed = (relative in ('question-identities.jsonl', 'reference-manifest.json') or
                   (relative.startswith('sources/') and path.name in
                    ARRAY_NAMES + ('index.jsonl', 'scan-complete.json')))
        if path.is_symlink() or not allowed:
            raise ValueError('Only original unified token scans and metadata may enter the slim package')
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path)))
    metadata = dict(schema_version=1, role='PREPARED_REFERENCE_PREFIX5_ARCHIVE', encoder_revision=REVISION,
        prefix_tokens=5, source_archive_sha256=source_archive_sha256,
        train_groups=reference['expected_train_groups'], train_trajectories=reference['train_trajectories'],
        test_groups=reference['expected_test_groups'], test_trajectories=reference['test_trajectories'],
        sources=sum(item['split'] == 'train' for item in reference['sources']), canonical_train_sha256=CANONICAL_SHA,
        canonical_questions=reference.get('canonical_questions', 169028),
        reference_manifest_sha256=sha(tokens_root / 'reference-manifest.json'), files=files,
        new_rollouts=0, new_em_scoring=0, frozen_model_or_features_included=False)
    manifest_bytes = canonical(metadata) + b'\n'
    parts = _Parts(output, part_bytes=part_bytes)
    with gzip.GzipFile(filename='', mode='wb', fileobj=parts, compresslevel=6, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for record in files:
                path = tokens_root / record['path']
                info = tarfile.TarInfo(record['path'])
                info.size, info.mode, info.mtime = record['bytes'], 0o600, 0
                with path.open('rb') as handle:
                    archive.addfile(info, handle)
            info = tarfile.TarInfo('MANIFEST.json')
            info.size, info.mode, info.mtime = len(manifest_bytes), 0o600, 0
            archive.addfile(info, io.BytesIO(manifest_bytes))
    parts.close()
    result = dict(metadata, parts=parts.parts, concatenated_sha256=parts.whole_hash.hexdigest(), gzip_compresslevel=6,
                  compressed_bytes=parts.total, bytes=parts.total, part_bytes_limit=part_bytes,
                  internal_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  question_max_tokens=reference['question_max_tokens'])
    once(output / 'transfer-manifest.json', result)
    return result


def build_reference(config_path, output):
    config_path, output = Path(config_path).resolve(strict=True), Path(output).resolve()
    config = read(config_path)
    if config.get('schema_version') != 1 or config.get('prefix_tokens') != 5:
        raise ValueError('An explicit official reference-to-prefix5 source plan is required')
    output.mkdir(parents=True, exist_ok=False)
    progress(output, 'VERIFYING_LOCAL_ORIGINAL_ARCHIVE', new_rollouts=0, new_em_scoring=0)
    reference = extract_reference(config['reference_archive'], output / 'original-reference')
    tokens_root = output / 'tokens'
    tokens_root.mkdir()
    identities_path = tokens_root / 'question-identities.jsonl'
    canonical_descriptor = config.get('canonical_train', {})
    canonical_path = canonical_descriptor.get('path', reference / 'canonical/train.parquet')
    identities = canonical_identities(canonical_path, canonical_descriptor.get('sha256', CANONICAL_SHA), identities_path)
    initial = reference / 'epoch1-and-fixed-test/inputs'
    model_files = read(initial / 'scan-complete.json')['model_files_sha256']
    sources = [_scan_source(initial, tokens_root / 'sources/h100-epoch-0001', 'train', 'h100-epoch-0001',
                            'official-original-teacher5120-G5', identities)]
    sources.append(_scan_source(initial, tokens_root / 'sources/fixed-test', 'test', 'fixed-test',
                                'official-fixed-random512-G5', identities))
    progress(output, 'REFERENCE_SOURCE_CONVERTED', source_epoch=1, train_trajectories=25600,
             test_trajectories=2560, new_rollouts=0, new_em_scoring=0)
    for epoch in range(2, 69):
        sources.append(_rollout_source(reference, epoch, tokens_root / ('sources/h100-epoch-%04d' % epoch), identities))
        progress(output, 'REFERENCE_SOURCE_CONVERTED', source_epoch=epoch,
                 train_trajectories=epoch * 25600, test_trajectories=2560, new_rollouts=0, new_em_scoring=0)
    for descriptor in sources:
        descriptor['relative_inputs_dir'] = Path(descriptor['inputs_dir']).relative_to(tokens_root).as_posix()
    manifest = dict(schema_version=1, role='PREPARED_REFERENCE_PREFIX5_SCANS', prefix_tokens=5,
        encoder_revision=REVISION, model_files_sha256=model_files, question_identity_path=str(identities_path),
        question_identity_relative_path='question-identities.jsonl', question_identity_sha256=sha(identities_path),
        question_max_tokens=max(item['question_max_observed'] for item in sources), sources=sources,
        expected_train_groups=348160, expected_test_groups=512, train_trajectories=1740800, test_trajectories=2560,
        canonical_questions=len(identities),
        source_archive_sha256=config['reference_archive']['concatenated_sha256'],
        source_plan_sha256=sha(config_path), canonical_train_sha256=CANONICAL_SHA,
        original_sampling_events_preserved=True, new_rollouts=0, new_em_scoring=0)
    once(tokens_root / 'reference-manifest.json', manifest)
    progress(output, 'PACKAGING_SLIM_REFERENCE', train_trajectories=1740800, test_trajectories=2560)
    transfer = package_tokens(tokens_root, output, config['reference_archive']['concatenated_sha256'])
    complete = dict(stage='REFERENCE_PREFIX5_SLIM_COMPLETE', train_groups=348160, train_trajectories=1740800,
        test_groups=512, test_trajectories=2560, source_archive_sha256=transfer['source_archive_sha256'],
        transfer_manifest_path=str(output / 'transfer-manifest.json'),
        transfer_manifest_sha256=sha(output / 'transfer-manifest.json'), compressed_bytes=transfer['compressed_bytes'],
        parts=len(transfer['parts']), new_rollouts=0, new_em_scoring=0, gpu_used=False, checked_unix=time.time())
    once(output / 'REFERENCE-PREPARE-COMPLETE.json', complete)
    progress(output, 'REFERENCE_PREFIX5_SLIM_COMPLETE', **{k: v for k, v in complete.items() if k != 'stage'})
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build_reference(args.config, args.output), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
