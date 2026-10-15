"""Own output paths and durable metadata; shared inputs are never written."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import re

DEFAULT_BASE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016')
BASE = Path(os.environ.get('SEARCH_G1_BASE',str(DEFAULT_BASE)))
MODEL = Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725')
DATA = Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b')
TEACHER = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015/teacher-warmup-cache')
REUSED = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
PIN = '1af0d14937a3da6666eec63e8caa50c192d481e6'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def own(base):
    base = Path(base)
    if (base != BASE or base.resolve() != BASE or base.is_symlink() or
            re.fullmatch(re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?',str(base)) is None):
        raise ValueError('This deployment can write only its independent HPC3 namespace')
    return base


def publish(path, value, *, once=False):
    path = Path(path)
    if not path.is_relative_to(BASE) or path.is_symlink():
        raise ValueError('Metadata output is outside this experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b'\n'
    if once:
        with path.open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
    else:
        temporary = path.with_name(path.name + '.partial')
        with temporary.open('wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def progress(phase, **values):
    event = dict(phase=phase, checked_unix=time.time(), job=os.environ.get('SLURM_JOB_ID'), **values)
    publish(BASE/'preparation/progress.json', event)
    with (BASE/'logs/preparation-progress.jsonl').open('ab') as f:
        f.write(canonical(event) + b'\n'); f.flush()
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def verify_source(commit):
    record = read(BASE/'preparation/source-publication.json')
    source = BASE/'source'/commit
    if record['commit'] != commit or record['pinned_base'] != PIN or not record['pushed_and_fetched']:
        raise ValueError('The source is not the published independent checkout')
    for name, digest in record['files'].items():
        if sha(source/name) != digest:
            raise ValueError('Deployed source differs: '+name)
    if (BASE/'repo').resolve() != source:
        raise ValueError('The run repo is not this immutable source')
    return record


def teacher_labels():
    """Read the existing immutable teacher without acquiring a writer handle."""
    from search_g1.qprobe_warmup import TeacherCache
    identity = read(TEACHER/'identity.json')
    if identity['model_revision'] != 'aa8e72537993ba99e69dfaafa59ed015b17504d1':
        raise ValueError('Wrong official teacher revision')
    cache = TeacherCache.__new__(TeacherCache)
    cache.root, cache.identity = TEACHER, identity
    with (TEACHER/'writer.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        receipt, groups, _ = cache.verify_complete()
    if receipt != read(TEACHER/'complete.json') or len(groups) != 5120:
        raise ValueError('Incomplete teacher cache')
    labels = [dict(g, complete=True, selection_source='bootstrap_all_candidate', **receipt['protocol']) for g in groups]
    return labels, dict(teacher_identity_sha256=sha(TEACHER/'identity.json'),
                       teacher_complete_sha256=sha(TEACHER/'complete.json'),
                       original_feature_spec_sha256=identity['feature_spec_sha'],
                       protocol=receipt['protocol'], questions=5120, trajectories_reused=25600)


def normalize_preparation_gpu():
    """Map the actual Slurm global minor to UUID before CUDA initialization."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Preparation requires a real Slurm GPU allocation')
    raw = os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES', '')
    tokens = raw.split(',')
    if len(tokens) != 1 or not tokens[0]:
        raise ValueError('Preparation requires exactly one allocated GPU')
    from reproduction.common8.native_allocation import nvml_minor_number
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    inventory = [list(map(str.strip, row.split(','))) for row in rows.splitlines()]
    inventory = [row+[str(nvml_minor_number(row[1]))] for row in inventory]
    if tokens[0].startswith('GPU-'):
        choices = [row[1] for row in inventory if row[1] == tokens[0]]
    else:
        choices = [row[1] for row in inventory if row[2] == tokens[0]]
    if len(choices) != 1:
        raise ValueError('Allocated Slurm GPU could not be bound to one UUID')
    os.environ['CUDA_VISIBLE_DEVICES'] = choices[0]
    return choices[0]
