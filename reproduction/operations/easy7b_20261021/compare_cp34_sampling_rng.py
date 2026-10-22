"""Optional CPU-only comparison of twelve saved CP34/outer35 RNG sidecars.

Only the fixed completed-boundary and first-chunk worker RNG files are read.
No model, Adam, raw rollout, CUDA operation or training RPC is used. These are
trusted, task-owned torch.save files containing NumPy/Python RNG tuples; loading
them requires weights_only=False, always with map_location='cpu'. A missing
file/field is UNKNOWN. Differences never request training restart or rollback.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import zipfile

import numpy as np
import torch


OLD_BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022')
NEW_BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022')
RUN = '232f20e112ac4a978c5aba52a0367195'
OLD_COMMIT = 'ebb6844db6a1ae739a06d5cc31d3510e7b4779b1'
NEW_COMMIT = 'b76d7a05245a5eb7adc75385e2db803c67acce43'
OLD_ATTEMPT = '18ace454753b415287eeb23f4e9e1077'
NEW_ATTEMPT = 'ac512297cc994a62a6f77a76ebeb2994'
STEP, WRITES, WORLD = 34, 680, 4
MAX_SMALL = 512 * 1024
FIELDS = ('torch_cpu', 'torch_cuda', 'numpy', 'python', 'cuda_device',
          'actor_cuda_saved', 'rollout_cuda_saved', 'request_generators',
          'vllm_version', 'boundary')
ROLES = ('checkpoint34_boundary', 'old35_before_generation', 'new35_before_generation')
PAIRS = ((ROLES[0], ROLES[1]), (ROLES[0], ROLES[2]), (ROLES[1], ROLES[2]))


def need(condition, message):
    if not condition:
        raise ValueError('CP34_SAMPLING_RNG_' + message)


def attempt(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{32}', value) is not None,
         'FULL_ATTEMPT_ID_REQUIRED')
    return value


def _describe(value):
    if isinstance(value, torch.Tensor):
        return dict(type='torch.Tensor', dtype=str(value.dtype), shape=list(value.shape),
                    device=str(value.device), bytes=value.numel()*value.element_size())
    if isinstance(value, np.ndarray):
        return dict(type='numpy.ndarray', dtype=value.dtype.str, shape=list(value.shape), bytes=value.nbytes)
    if isinstance(value, dict):
        return dict(type='dict', length=len(value), entries={str(k): _describe(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        result = dict(type=type(value).__name__, length=len(value))
        if len(value) <= 8:
            result['items'] = [_describe(v) for v in value]
        else:
            result['element_types'] = dict(sorted(Counter(type(v).__name__ for v in value).items()))
        return result
    return dict(type=type(value).__name__)


def fingerprint(value):
    """Hash typed content, including all RNG bytes; never expose those bytes."""
    sha = hashlib.sha256()

    def header(record):
        raw = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        sha.update(str(len(raw)).encode() + b':' + raw)

    def visit(item):
        if isinstance(item, torch.Tensor):
            need(item.device.type == 'cpu', 'CPU_RNG_TENSOR_REQUIRED')
            tensor = item.detach().contiguous()
            header(_describe(tensor)); sha.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            need(not item.dtype.hasobject, 'NUMPY_OBJECT_ARRAY_FORBIDDEN')
            header(_describe(item)); sha.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            header(dict(type='numpy.scalar', dtype=item.dtype.str)); sha.update(item.tobytes())
        elif isinstance(item, dict):
            need(all(type(k) is str for k in item), 'STRING_RNG_MAPPING_KEYS_REQUIRED')
            header(dict(type='dict', keys=sorted(item)))
            for key in sorted(item): visit(item[key])
        elif isinstance(item, (tuple, list)):
            header(dict(type=type(item).__name__, length=len(item)))
            for child in item: visit(child)
        elif item is None or type(item) in (bool, int, str):
            header(dict(type=type(item).__name__, value=item))
        elif type(item) is float:
            header(dict(type='float', value=item.hex()))
        elif type(item) is bytes:
            header(dict(type='bytes', bytes=len(item))); sha.update(item)
        else:
            raise ValueError('CP34_SAMPLING_RNG_UNSUPPORTED_TYPE_' + type(item).__name__)
    visit(value)
    return dict(sha256=sha.hexdigest(), structure=_describe(value), is_none=value is None)


def _read(path, rank, commit):
    """Bounded file read, then CPU deserialization; no other path is discovered."""
    need(path.resolve() == path, 'CANONICAL_NONSYMLINK_PATH_REQUIRED')
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_SMALL,
             'SMALL_REGULAR_RNG_SIDECAR_REQUIRED')
        raw = stream.read(MAX_SMALL + 1); after = os.fstat(stream.fileno())
        need(len(raw) == before.st_size and
             (before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'RNG_FILE_CHANGED_DURING_READ')
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            need(sum(row.file_size for row in archive.infolist()) <= MAX_SMALL,
                 'EXPANDED_RNG_SIDECAR_TOO_LARGE')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    need(type(value) is dict and 'optimizer' not in value and 'scheduler' not in value,
         'RNG_ONLY_SIDECAR_REQUIRED')
    expected = dict(run_id=RUN, source_commit=commit, rank=rank, world_size=WORLD,
                    completed_outer=STEP, actual_optimizer_writes=WRITES)
    need(all(value.get(key) == item for key, item in expected.items()), 'SIDECAR_IDENTITY_DIFFERS')
    rng = value.get('rng'); need(type(rng) is dict, 'RNG_MAPPING_REQUIRED')
    for key in ('torch_cpu', 'torch_cuda', 'actor_cuda_saved', 'rollout_cuda_saved'):
        if key in rng and rng[key] is not None:
            tensor = rng[key]
            need(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and
                 tensor.dtype == torch.uint8 and tensor.ndim == 1, 'BYTE_RNG_STATE_REQUIRED_' + key)
    return dict(status='LOADED', path=str(path), bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(), identity=expected,
        fields={key: fingerprint(item) for key, item in sorted(rng.items())})


def _observe(path, rank, commit):
    try:
        return _read(path, rank, commit)
    except FileNotFoundError:
        return dict(status='UNKNOWN', path=str(path), reason='SMALL_RNG_SIDECAR_MISSING', fields={})
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as error:
        return dict(status='UNKNOWN', path=str(path), reason='SIDECAR_OBSERVATION_ERROR',
                    error=type(error).__name__ + ': ' + str(error)[:512], fields={})


def _equal(left, right):
    if left is None or right is None: return 'UNKNOWN'
    return 'EQUAL' if left['sha256'] == right['sha256'] else 'DIFFERENT'


def _comparisons(fields):
    pairs = {a + '__vs__' + b: _equal(fields.get(a), fields.get(b)) for a, b in PAIRS}
    values = list(pairs.values())
    return dict(pairs=pairs, all_three=('UNKNOWN' if 'UNKNOWN' in values else
                'EQUAL' if all(v == 'EQUAL' for v in values) else 'DIFFERENT'))


def compare(old_attempt=OLD_ATTEMPT, new_attempt=NEW_ATTEMPT):
    attempt(old_attempt); attempt(new_attempt)
    roots = {
        ROLES[0]: OLD_BASE/'preparation/rolling-checkpoints/global_step_34/boundary',
        ROLES[1]: OLD_BASE/'artifacts/mixed-candidate-pools'/(old_attempt+'-pool-0')/'0-worker-rng',
        ROLES[2]: NEW_BASE/'artifacts/mixed-candidate-pools'/(new_attempt+'-pool-0')/'0-worker-rng',
    }
    rows = []
    for rank in range(WORLD):
        files = {role: _observe(root/('rank-'+str(rank)+'.pt'), rank,
                               NEW_COMMIT if role == ROLES[2] else OLD_COMMIT)
                 for role, root in roots.items()}
        keys = sorted(set(FIELDS).union(*(set(value['fields']) for value in files.values())))
        comparisons = {key: _comparisons({role: value['fields'].get(key)
                                         for role, value in files.items()}) for key in keys}
        # __enter__ swaps rollout_cuda_saved into global CUDA. The idle global
        # torch_cuda and historical actor_cuda_saved are different namespaces.
        sampling = {}
        for role, value in files.items():
            field = value['fields'].get('rollout_cuda_saved')
            sampling[role] = field if field is not None and not field['is_none'] else None
        rows.append(dict(rank=rank, files=files, same_field_comparisons=comparisons,
            next_rollout_private_rng=_comparisons(sampling)))
    unknown = any(c['all_three'] == 'UNKNOWN' for row in rows for c in row['same_field_comparisons'].values())
    return dict(status='RNG_BOUNDARY_COMPARISON_UNKNOWN' if unknown else 'RNG_BOUNDARY_COMPARISON_COMPLETE',
        run_id=RUN, checkpoint_outer=STEP, compared_next_outer=35, world_size=WORLD,
        old_attempt=old_attempt, new_attempt=new_attempt, pool_retry=0, chunk_offset=0, ranks=rows,
        namespace_notes=dict(capture_scope='idle engine outside generation phase',
            next_sampling_state='rollout_cuda_saved; sharding-manager __enter__ loads it into CUDA',
            torch_cuda='currently active idle/actor CUDA state; not interchangeable with rollout_cuda_saved',
            actor_cuda_saved='previous actor state, overwritten from active CUDA on next __enter__',
            request_generators='saved engine request-local generators; equality is a separate comparison',
            replay_limit='Equal RNG state does not prove equal asynchronous request/batch order or equal sampled output'),
        cpu_only=True, cuda_operations=False, model_or_adam_files_read=False, raw_rollout_files_read=False,
        training_gate=False, training_failure_inferred=False, training_modified=False,
        restart_or_rollback_requested=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-attempt', default=OLD_ATTEMPT)
    parser.add_argument('--new-attempt', default=NEW_ATTEMPT)
    args = parser.parse_args(argv)
    try:
        result = compare(args.old_attempt, args.new_attempt)
    except ValueError as error:
        print(json.dumps(dict(status='RNG_COMPARISON_ARGUMENT_ERROR', error=str(error),
            training_modified=False, training_gate=False, training_failure_inferred=False)))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0  # UNKNOWN and DIFFERENT are observations, not training failures.


if __name__ == '__main__': raise SystemExit(main())
