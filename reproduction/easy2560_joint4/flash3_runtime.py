"""Private copy of the already compiled HPC FA3 runtime, without pip changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


SOURCES = {
    'interface': ('/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/lib/python3.10/site-packages/flash_attn_interface.py',
                  'a50f564fa1fc4cdd85ffa61f79e913a463c2066e519f9c5e8bc2e5c74457fa38', 26902),
    'extension': ('/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/lib/python3.10/site-packages/flash_attn_3_cuda.cpython-310-x86_64-linux-gnu.so',
                  'be3e87f7b0401f71be424916d4ca031f90aa21be70b479eec0c9898763dcb345', 742209528),
    'cudart': ('/share/anaconda3/envs/flash_attn3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12',
               '095296727cfb5f51e5b3ae69a3cb88cbd7b6592f314ace0fdfc4e1d758c2aa85', 716128),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def binding(base):
    root = Path(base) / 'preparation/flash3-runtime'
    record = json.loads((root / 'ready.json').read_text())
    if record['status'] != 'PINNED_PRIVATE_FA3_RUNTIME_SHA_PASS':
        raise ValueError('Incomplete private FA3 runtime')
    result = {}
    for role, (_, expected, size) in SOURCES.items():
        path = root / record['files'][role]['name']
        if (path.is_symlink() or not path.is_file() or path.resolve() != path or
                path.stat().st_size != size or record['files'][role]['sha256'] != expected):
            raise ValueError('Private FA3 file identity differs: ' + role)
        result[role + '_path'] = str(path)
        result[role + '_sha256'] = expected
    return result


def prepare(base):
    from .control import own_base
    base = own_base(base)
    root = base / 'preparation/flash3-runtime'
    if root.exists():
        result = binding(base)
        for role in SOURCES:
            if digest(result[role + '_path']) != result[role + '_sha256']:
                raise ValueError('Existing private FA3 payload changed')
        return result
    if (base / 'job.id').exists() or (base / 'logs/run-manifest.json').exists():
        raise ValueError('Prepare FA3 only in a new, unsubmitted run base')
    stage = root.with_name('.flash3-runtime.partial-' + uuid.uuid4().hex)
    stage.mkdir()
    files = {}
    # A failed copy is retained for inspection. Never alter the source runtime.
    for role, (source, expected, size) in SOURCES.items():
        source = Path(source)
        if source.is_symlink() or source.stat().st_size != size or digest(source) != expected:
            raise ValueError('Pinned compiled FA3 source differs: ' + role)
        target = stage / source.name
        with source.open('rb') as stream, target.open('xb') as output:
            shutil.copyfileobj(stream, output, 8 * 1024**2)
            output.flush(); os.fsync(output.fileno())
        if digest(target) != expected:
            raise ValueError('Private compiled FA3 copy failed: ' + role)
        files[role] = dict(name=target.name, sha256=expected, bytes=size)
    with (stage / 'ready.json').open('x') as stream:
        json.dump(dict(status='PINNED_PRIVATE_FA3_RUNTIME_SHA_PASS', files=files,
                       shared_environment_modified=False), stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    stage.rename(root)
    fd = os.open(str(root.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return binding(base)
