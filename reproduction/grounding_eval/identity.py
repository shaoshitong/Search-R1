"""Fail-closed normal-source/model/canonical-NQ acceptance across three adapters.

This validates actual HF file bytes, not just an adapter's self-consistent label.
Canonical occurrence IDs come from the SHA-bound original final parquet.
"""
import json
from pathlib import Path
from .core import canonical, digest, read, sha

FINAL_DATA_SHA256 = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
HF_METADATA = ('config.json', 'tokenizer_config.json', 'tokenizer.json',
               'generation_config.json', 'special_tokens_map.json', 'added_tokens.json',
               'merges.txt', 'vocab.json')


def canonical_nq(data_path):
    path = Path(data_path)
    if sha(path) != FINAL_DATA_SHA256:
        raise ValueError('Normal source does not use the fixed common final parquet')
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    nq = {}
    for ordinal, row in enumerate(rows):
        if row['data_source'] != 'nq':
            continue
        if not isinstance(row.get('extra_info'), dict) or 'index' not in row['extra_info']:
            raise ValueError('Canonical NQ extra_info.index missing')
        nq[ordinal] = str(row['extra_info']['index'])
    if len(rows) != 51713 or len(nq) != 3610 or list(nq) != list(range(3610)):
        raise ValueError('Canonical final dataset NQ occurrence map differs')
    return nq


def _safe_files(checkpoint, files):
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing normal checkpoint file inventory')
    index_path = checkpoint / 'model.safetensors.index.json'
    index = read(index_path)
    required = set(index['weight_map'].values()) | {'model.safetensors.index.json'}
    required.update(name for name in HF_METADATA if (checkpoint / name).exists())
    required.update(('config.json', 'tokenizer_config.json', 'tokenizer.json'))
    if not required.issubset(files):
        raise ValueError('Normal checkpoint inventory omits HF model/tokenizer files: ' + str(sorted(required - set(files))))
    normalized = {}
    for name, entry in sorted(files.items()):
        if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe normal checkpoint inventory filename')
        expected = entry.get('sha256') if isinstance(entry, dict) else entry
        size = entry.get('size') if isinstance(entry, dict) else None
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Invalid normal checkpoint SHA256')
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError('Missing requested checkpoint file: ' + str(path))
        if size is not None and path.stat().st_size != size:
            raise ValueError('Requested checkpoint file size differs from normal source')
        actual = sha(path)
        if actual != expected:
            raise ValueError('Normal/CF checkpoint bytes differ: ' + name)
        normalized[name] = actual
    return normalized


def _validate_occurrences(rows, canonical_ids):
    if not rows:
        raise ValueError('Empty normal source')
    ordinals = []
    for row in rows:
        ordinal = row.get('evaluation_ordinal')
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError('Non-integer NQ evaluation ordinal')
        if row.get('data_source') != 'nq' or canonical_ids.get(ordinal) != str(row.get('qid')):
            raise ValueError('Normal occurrence does not match original final parquet')
        ordinals.append(ordinal)
    # A pilot is the canonical first N rows, enabling a prefix-preserving full resume.
    # Repeated question text/qid at different ordinals remains a distinct occurrence.
    expected = list(canonical_ids)[:len(rows)]
    if len(rows) > 3610 or ordinals != expected:
        raise ValueError('Missing, duplicated, reordered, or foreign NQ occurrences')
    return ordinals


def validate_normal_identity(rows, checkpoint_path, normal_config_path):
    root = Path(normal_config_path).resolve().parent
    normal_config = read(normal_config_path)
    checkpoint = Path(checkpoint_path).resolve()
    a800 = root / 'binding.json'
    standard = root / 'normal-binding.json'
    if a800.is_file() and standard.is_file():
        raise ValueError('Ambiguous normal adapter binding')
    if a800.is_file():
        schema = 'a800-common8-v2'
        binding_path = a800; binding = read(binding_path)
        if normal_config != binding.get('normal_config') or digest(normal_config) != binding.get('normal_config_sha256'):
            raise ValueError('Normal configuration differs from A800 binding')
        source_checkpoint = Path(binding['checkpoint'])
        complete = read(source_checkpoint / 'complete.json')
        files = complete.get('files_sha256')
        if not isinstance(files, dict):
            raise ValueError('Missing original normal checkpoint published manifest')
        # A800's audited identity includes the publication receipt itself.
        files = dict(files, **{'complete.json': sha(source_checkpoint / 'complete.json')})
        # Independent inventories may include additional small HF metadata.
        for name in HF_METADATA + ('model.safetensors.index.json',):
            if name not in files and (source_checkpoint / name).is_file():
                files[name] = sha(source_checkpoint / name)
        if digest(files) != binding.get('checkpoint_identity_sha256'):
            raise ValueError('Normal A800 checkpoint identity differs from its published files')
        data_sha = binding.get('dataset_sha256')
        data_path = binding.get('dataset_path') or str(Path(str(source_checkpoint).split('/checkpoints/', 1)[0]) / 'data/final.parquet')
        row_identity = binding['checkpoint_identity_sha256']
        for row in rows:
            if row.get('checkpoint_id') != row_identity or row.get('normal_config_sha256') != binding['normal_config_sha256']:
                raise ValueError('Foreign A800 normal row model/config')
            original = dict(row); checksum = original.pop('source_record_hash', None)
            if checksum != digest(original):
                raise ValueError('A800 normal row checksum differs')
    elif standard.is_file():
        binding_path = standard; binding = read(binding_path)
        files = binding.get('checkpoint_files')
        if not isinstance(files, dict) or not files:
            raise ValueError('Missing normal checkpoint files')
        if all(isinstance(value, dict) for value in files.values()):
            schema = 'hpc3-legacy-manager-v1'
            if normal_config not in (binding, binding.get('normal_protocol'), binding.get('job')):
                raise ValueError('Normal configuration differs from HPC3 binding')
            if digest(files) != binding.get('checkpoint_hash'):
                raise ValueError('HPC3 normal checkpoint hash differs')
            data_sha = binding.get('data_sha256'); data_path = binding['job']['data_path']
            for row in rows:
                if row.get('checkpoint_hash') != binding['checkpoint_hash'] or row.get('normal_config_hash') != digest(binding):
                    raise ValueError('Foreign HPC3 normal row model/config')
                original = dict(row); original.pop('source_shard_sha256', None)
                checksum = original.pop('source_record_hash', None)
                if checksum != digest(original):
                    raise ValueError('HPC3 normal row checksum differs')
        elif all(isinstance(value, str) for value in files.values()):
            schema = 'h100-legacy-manager-v1'
            if normal_config != binding:
                raise ValueError('Normal configuration differs from H100 binding')
            data_sha = binding.get('data_sha256'); data_path = binding.get('data_path')
            if not data_path:
                raise ValueError('Explicit H100 canonical final parquet path missing')
            for row in rows:
                if row.get('normal_binding_sha256') != digest(binding) or row.get('checkpoint_files_sha256') != digest(files):
                    raise ValueError('Foreign H100 normal row model/config')
        else:
            raise ValueError('Unknown normal checkpoint inventory schema')
    else:
        raise ValueError('No supported normal adapter binding beside normal-config')
    if data_sha != FINAL_DATA_SHA256:
        raise ValueError('Normal adapter binding has a different final dataset SHA')
    verified_files = _safe_files(checkpoint, files)
    ids = canonical_nq(data_path)
    ordinals = _validate_occurrences(rows, ids)
    return dict(schema=schema, normal_binding_path=str(binding_path),
        normal_binding_sha256=sha(binding_path), normal_configuration_sha256=digest(normal_config),
        checkpoint_path=str(checkpoint), checkpoint_files_sha256=digest(verified_files),
        verified_checkpoint_files=verified_files, data_path=str(data_path),
        data_sha256=FINAL_DATA_SHA256, canonical_nq_occurrences_sha256=digest(list(ids.items())),
        source_occurrences_sha256=digest([[i, ids[i]] for i in ordinals]),
        count=len(rows), pilot_only=len(rows) != 3610)
