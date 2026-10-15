"""Constants and durable metadata for isolated CoSearch inference only."""
import hashlib
import json
import os
import pathlib

BASE = pathlib.Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
TRAIN = BASE/'runtime/cosearch-instruct200-common-em-20261011'
OUT = BASE/'runtime/cosearch-final200-four-gpu-20261015'
TOOLS = OUT/'tools'
SOURCE = OUT/'source'
PYTHON = BASE/'source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'
N = 51713


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def env():
    values = dict(os.environ)
    built = json.loads((BASE/'runtime/pytrec-eval-05-hpc-build/build-complete.json').read_text())
    extension = pathlib.Path(built['extension'])
    assert built['status'] == 'PYTREC_05_NATIVE_IMPORT_METRICS_PASS'
    assert sha(extension) == built['sha256']
    values['PYTHONPATH'] = ':'.join(map(str, [TOOLS, SOURCE, SOURCE/'verl', extension.parent,
        BASE/'runtime/flash-attn-281-hpc-build/lib',
        BASE/'source/author-native/env/lib/python3.11/site-packages']))
    values.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
                  GLOO_SOCKET_IFNAME='lo',NCCL_SOCKET_IFNAME='lo',VLLM_HOST_IP='127.0.0.1',
                  PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                  HF_HOME=str(OUT/'cache/hf'), HF_DATASETS_CACHE=str(OUT/'cache/datasets'),
                  TMPDIR=str(OUT/'cache/tmp'), OMP_NUM_THREADS='8',
                  MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                  PYTHONUNBUFFERED='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    for key in ['RAY_ADDRESS', 'PYTHONHOME', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy']:
        values.pop(key, None)
    return values
