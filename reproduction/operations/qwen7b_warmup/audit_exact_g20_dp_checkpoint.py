"""Independent CPU fullstate reload for the sole accepted exact-G20 DP job.

Run as a CPU-only overlapping step inside that owned64GiB allocation. No
model instance/forward, GPU use, optimizer update or producer file mutation.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from . import probe_ablation_train as u

BASE = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g20-exact-uniform15504-dp-v1')
CONFIG_SHA = 'af679a2a8ff93642a4573bb297f23ff954cca38b6749f0c7ff10c111e2cce4f4'
JOB = '743485'


def proof(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=u.sha(path))


def audit(cursor):
    if cursor not in (1, 136) or os.environ.get('SLURM_JOB_ID') != JOB:
        raise ValueError('Only the owned G20 allocation and accepted CP1/final counters allowed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.is_initialized():
        raise ValueError('Independent audit must use CPU only')
    torch.set_num_threads(1)
    if u.sha(BASE / 'config.json') != CONFIG_SHA:
        raise ValueError('Exact actual experiment config changed')
    path = BASE / 'checkpoints' / ('cursor-%06d.pt' % cursor)
    before = path.stat()
    replay = u.read(BASE / 'next-update-replay.json')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if (state['cursor'] != cursor or state['steps'] != 10464 + cursor
            or state['config_sha256'] != CONFIG_SHA or state['actor_updates'] != 0
            or state['optimizer_reset'] or state['readout_reset'] or state['block_reset']
            or state['role'] != 'ISOLATED_REAL_G20_QUESTION_WEIGHTED_PROBE'
            or replay['status'] != 'PASSED' or replay['actual'] != replay['expected']):
        raise ValueError('Accepted actual G20 fullstate/replay lineage differs')
    if any(int(s['step']) != 10464 + cursor for s in state['optimizer']['state'].values()):
        raise ValueError('Every populated Adam state must match the complete cursor')
    actual = dict(raw=u.model_mapping_hash(state['raw']), ema=u.model_mapping_hash(state['ema']),
        adam=u.optimizer_blob_hash(state['optimizer']), raw_buffers=u.tensor_hash(state['raw_buffers']),
        ema_buffers=u.tensor_hash(state['ema_buffers']), rng=u.rng_hash(state['rng']))
    if actual != state['accepted_hashes'] or (cursor == 1 and actual != replay['actual']):
        raise ValueError('Independent CPU six-component fullstate hash mismatch')
    source = state['source_all_rank_states']
    if len(source) != 4 or {s['rank'] for s in source} != {0, 1, 2, 3}:
        raise ValueError('Original four-rank RNG archive missing')
    config = u.read(BASE / 'config.json')
    for key in ('source_checkpoint', 'g20_complete', 'recipe'):
        if state[key] != config[key]:
            raise ValueError('Checkpoint source/real20/actual objective binding differs')
    checkpoint = proof(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Immutable accepted checkpoint changed during independent read')
    latest = u.read(BASE / 'latest.json')
    if latest['cursor'] == cursor and any(latest[k] != checkpoint[k] for k in ('path', 'bytes', 'sha256')):
        raise ValueError('Full file SHA differs from accepted producer receipt')
    result = dict(status='EXACT_G20_DP_INDEPENDENT_FULL_SHA_AND_CPU_SIX_COMPONENT_RELOAD_PASSED',
        checkpoint=checkpoint, cursor=cursor, steps=10464 + cursor,
        config_sha256=CONFIG_SHA, component_hashes=actual,
        source_all_four_RNG_preserved=True, original_actor_updates=0,
        optimizer_reset=False, GPU_used=False, probe_extra_updates=0,
        job_id=os.environ['SLURM_JOB_ID'], step_id=os.environ.get('SLURM_STEP_ID'),
        hostname=os.uname().nodename, audit_source=proof(__file__), epoch=time.time())
    u.atomic(BASE / ('independent-cp-%06d.json' % cursor), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cursor', type=int, choices=(1, 136), required=True)
    args = parser.parse_args()
    with (BASE / ('independent-cp-%06d.lock' % args.cursor)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            print(json.dumps(audit(args.cursor)), flush=True)
        except BaseException as error:
            u.atomic(BASE / ('independent-cp-%06d-failure.json' % args.cursor),
                dict(error_type=type(error).__name__, error=str(error), epoch=time.time()))
            raise
